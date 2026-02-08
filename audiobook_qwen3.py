#!/usr/bin/env python3
"""
Create audiobook files from text using Qwen3-TTS voice cloning.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib.util
import json
import re
import shutil
import shlex
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

APP_VERSION = "0.1.0"
DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
DEFAULT_OUTPUT_NAME = "audiobook.mp3"
DEFAULT_MAX_CHARS = 1800
DEFAULT_PAUSE_MS = 300
DEFAULT_MAX_INFERENCE_CHARS = 2600
DEFAULT_ATTN_IMPLEMENTATION = "sdpa"
DEFAULT_DTYPE = "bfloat16"


@dataclass
class TextBatch:
    index: int
    start_paragraph: int
    end_paragraph: int
    text: str
    char_count: int


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def read_text_file(path: Path) -> str:
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Unable to decode text file: {path}")


def write_text_file(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def split_into_paragraphs(raw_text: str) -> list[str]:
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    chunks = re.split(r"\n\s*\n+", normalized)
    paragraphs: list[str] = []
    for chunk in chunks:
        cleaned = re.sub(r"[ \t]+", " ", chunk.strip())
        cleaned = re.sub(r"\n+", " ", cleaned).strip()
        if cleaned:
            paragraphs.append(cleaned)
    return paragraphs


def build_batches(paragraphs: list[str], max_chars_per_batch: int) -> list[TextBatch]:
    if max_chars_per_batch < 100:
        raise ValueError("--max-chars-per-batch must be at least 100.")

    batches: list[TextBatch] = []
    current: list[str] = []
    start_idx = 0
    current_len = 0

    def flush(end_idx: int) -> None:
        nonlocal current, start_idx, current_len
        if not current:
            return
        text = "\n\n".join(current)
        batches.append(
            TextBatch(
                index=len(batches) + 1,
                start_paragraph=start_idx,
                end_paragraph=end_idx,
                text=text,
                char_count=len(text),
            )
        )
        current = []
        current_len = 0

    for idx, paragraph in enumerate(paragraphs):
        add_len = len(paragraph) if not current else len(paragraph) + 2
        if current and current_len + add_len > max_chars_per_batch:
            flush(idx - 1)
        if not current:
            start_idx = idx
            current.append(paragraph)
            current_len = len(paragraph)
            continue
        current.append(paragraph)
        current_len += add_len

    flush(len(paragraphs) - 1)
    return batches


class StopController:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._stop_requested = False
        self._abort_current_batch = False
        self._force_stop = False

    @property
    def stop_requested(self) -> bool:
        with self._lock:
            return self._stop_requested

    @property
    def force_stop(self) -> bool:
        with self._lock:
            return self._force_stop

    @property
    def abort_current_batch(self) -> bool:
        with self._lock:
            return self._abort_current_batch

    def request_stop(self) -> None:
        with self._lock:
            self._stop_requested = True

    def request_abort_current_batch(self) -> None:
        with self._lock:
            self._stop_requested = True
            self._abort_current_batch = True

    def install(self) -> None:
        def handler(_signum: int, _frame: Any) -> None:
            with self._lock:
                if not self._stop_requested:
                    self._stop_requested = True
                elif not self._abort_current_batch:
                    self._abort_current_batch = True
                else:
                    self._force_stop = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)


class DefragProgressView:
    CHARS_PER_BLOCK = 200
    BLOCKS_PER_ROW = 120
    DONE_FLASH_SECONDS = 1.4
    SPINNER_FRAMES = ["|", "/", "-", "\\"]
    COLOR_RESET = "\x1b[0m"
    COLOR_RED = "\x1b[91m"
    COLOR_BLUE = "\x1b[94m"
    COLOR_GREEN = "\x1b[92m"
    COLOR_WHITE = "\x1b[97m"
    COLOR_CYAN = "\x1b[96m"
    COLOR_YELLOW = "\x1b[93m"
    COLOR_MAGENTA = "\x1b[95m"
    COLOR_DIM = "\x1b[90m"

    def __init__(
        self,
        total_batches: int,
        total_chars: int,
        completed_batches: int,
        completed_chars: int,
        enabled: bool,
        batch_char_counts: list[int] | None = None,
    ) -> None:
        self.total_batches = max(1, total_batches)
        self.total_chars = max(1, total_chars)
        self.completed_batches = max(0, completed_batches)
        self.completed_chars = max(0, completed_chars)
        self.enabled = enabled and sys.stdout.isatty()
        self.current_batch = 0
        self.current_batch_chars = 0
        self.current_batch_label = "idle"
        self.active_batch_start = 0
        self.active_batch_end = 0
        self.stop_requested = False
        self.status = "Loading model..."
        self._batch_durations: list[float] = []
        self._start_time = time.time()
        self._tick = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._done_flash_until: dict[int, float] = {}

        if batch_char_counts and len(batch_char_counts) == self.total_batches:
            normalized = [max(1, int(v)) for v in batch_char_counts]
        else:
            avg_chars = max(1, self.total_chars // self.total_batches)
            normalized = [avg_chars for _ in range(self.total_batches)]
        self._batch_char_counts = normalized
        self._batch_block_counts = [
            max(1, (chars + self.CHARS_PER_BLOCK - 1) // self.CHARS_PER_BLOCK)
            for chars in self._batch_char_counts
        ]

    def start(self) -> None:
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    def set_status(self, message: str) -> None:
        with self._lock:
            self.status = message

    def set_active_batch(
        self,
        batch_number: int,
        batch_chars: int,
        message: str,
        batch_label: str | None = None,
        active_batch_count: int = 1,
    ) -> None:
        with self._lock:
            self.current_batch = batch_number
            self.current_batch_chars = batch_chars
            self.current_batch_label = (
                batch_label
                if batch_label
                else f"batch {batch_number}/{self.total_batches} ({batch_chars} chars)"
            )
            self.active_batch_start = max(1, batch_number)
            self.active_batch_end = min(
                self.total_batches, batch_number + max(1, active_batch_count) - 1
            )
            self.status = message

    def mark_batch_complete(self, batch_chars: int, duration_seconds: float) -> None:
        with self._lock:
            self.completed_batches += 1
            self.completed_chars += batch_chars
            just_done = self.completed_batches
            self._done_flash_until[just_done] = time.time() + self.DONE_FLASH_SECONDS
            self.current_batch = 0
            self.current_batch_chars = 0
            self.current_batch_label = "idle"
            self._batch_durations.append(duration_seconds)
            if len(self._batch_durations) > 12:
                self._batch_durations.pop(0)
            self.status = "Batch complete."

    def mark_stop_requested(self) -> None:
        with self._lock:
            self.stop_requested = True
            self.status = "Stop requested; finishing current batch."

    def mark_abort_requested(self) -> None:
        with self._lock:
            self.stop_requested = True
            self.status = "Abort requested; trying to cancel current batch."

    def mark_batch_aborted(self, canceled: bool) -> None:
        with self._lock:
            self.current_batch = 0
            self.current_batch_chars = 0
            self.current_batch_label = "idle"
            self.status = (
                "Batch canceled before execution; leaving it for resume."
                if canceled
                else "Batch could not be interrupted in-flight; keeping it for resume."
            )

    def stop(self, final_status: str | None = None) -> None:
        if final_status:
            self.set_status(final_status)
        if not self.enabled:
            if final_status:
                print(final_status)
            return
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.5)
        sys.stdout.write("\x1b[?25h")
        sys.stdout.flush()

    def _render_loop(self) -> None:
        while not self._stop_event.is_set():
            self._tick += 1
            self._draw_frame(self._build_frame())
            time.sleep(0.12)
        self._draw_frame(self._build_frame())

    def _draw_frame(self, frame: str) -> None:
        sys.stdout.write("\x1b[?25l")
        sys.stdout.write("\x1b[H\x1b[2J")
        sys.stdout.write(frame)
        if not frame.endswith("\n"):
            sys.stdout.write("\n")
        sys.stdout.flush()

    def _batch_color(
        self,
        batch_number: int,
        completed_batches: int,
        active_start: int,
        active_end: int,
        now_ts: float,
        done_flash_until: dict[int, float],
    ) -> str:
        if active_start <= batch_number <= active_end:
            return self.COLOR_BLUE
        if batch_number <= completed_batches:
            if done_flash_until.get(batch_number, 0.0) > now_ts:
                return self.COLOR_GREEN
            return self.COLOR_WHITE
        return self.COLOR_RED

    def _build_frame(self) -> str:
        with self._lock:
            completed_batches = self.completed_batches
            completed_chars = self.completed_chars
            current_batch_label = self.current_batch_label
            active_start = self.active_batch_start
            active_end = self.active_batch_end
            status = self.status
            stop_requested = self.stop_requested
            done_flash_until = dict(self._done_flash_until)
            avg_batch_seconds = (
                sum(self._batch_durations) / len(self._batch_durations)
                if self._batch_durations
                else 0.0
            )

        ratio = min(1.0, completed_batches / self.total_batches)
        bar_width = 44
        fill = int(ratio * bar_width)
        bar = "#" * fill + "." * (bar_width - fill)
        percent = ratio * 100.0

        elapsed = time.time() - self._start_time
        eta = avg_batch_seconds * max(0, self.total_batches - completed_batches)
        now_ts = time.time()
        spinner = self.SPINNER_FRAMES[self._tick % len(self.SPINNER_FRAMES)]

        total_blocks = sum(self._batch_block_counts)
        completed_blocks = sum(self._batch_block_counts[:completed_batches])
        throughput = completed_chars / elapsed if elapsed > 0 else 0.0
        beam_index = (self._tick * 3) % max(1, total_blocks)
        beat = self._tick % 4

        lines: list[str] = []
        line_parts: list[str] = []
        visible_len = 0
        separator = f"{self.COLOR_DIM}|{self.COLOR_RESET}"
        block_cursor = 0
        for batch_number, block_count in enumerate(self._batch_block_counts, start=1):
            color = self._batch_color(
                batch_number=batch_number,
                completed_batches=completed_batches,
                active_start=active_start,
                active_end=active_end,
                now_ts=now_ts,
                done_flash_until=done_flash_until,
            )
            if active_start <= batch_number <= active_end:
                active_char = [">", "=", "~", ">"][beat]
                segment = [active_char for _ in range(block_count)]
            else:
                segment = ["#" for _ in range(block_count)]

            if block_cursor <= beam_index < block_cursor + block_count:
                beam_offset = beam_index - block_cursor
                segment[beam_offset] = "*"

            segment_text = "".join(segment)
            if "*" in segment_text:
                beam_offset = segment_text.index("*")
                token = (
                    f"{color}{segment_text[:beam_offset]}"
                    f"{self.COLOR_CYAN}*{color}{segment_text[beam_offset + 1:]}"
                    f"{self.COLOR_RESET}{separator}"
                )
            else:
                token = f"{color}{segment_text}{self.COLOR_RESET}{separator}"

            token_visible = block_count + 1
            if visible_len + token_visible > self.BLOCKS_PER_ROW and line_parts:
                lines.append("".join(line_parts))
                line_parts = []
                visible_len = 0
            line_parts.append(token)
            visible_len += token_visible
            block_cursor += block_count
        if line_parts:
            lines.append("".join(line_parts))
        if not lines:
            lines.append("(no blocks)")

        scan_bar_len = 38
        scan_pos = self._tick % scan_bar_len
        scan_chars = ["."] * scan_bar_len
        scan_chars[scan_pos] = ">"
        if scan_pos + 1 < scan_bar_len:
            scan_chars[scan_pos + 1] = ">"
        scanline = (
            f"{self.COLOR_DIM}{''.join(scan_chars[:max(0, scan_pos - 2)])}"
            f"{self.COLOR_CYAN}{''.join(scan_chars[max(0, scan_pos - 2):scan_pos + 2])}"
            f"{self.COLOR_DIM}{''.join(scan_chars[scan_pos + 2:])}{self.COLOR_RESET}"
        )

        header = [
            f"{self.COLOR_CYAN}=============================================================={self.COLOR_RESET}",
            f"{self.COLOR_MAGENTA}{spinner}{self.COLOR_RESET} Qwen3-TTS Defrag Animator",
            f"[{bar}] {percent:6.2f}%  batches {completed_batches}/{self.total_batches}  chars {completed_chars}/{self.total_chars}",
            f"elapsed {format_duration(elapsed)}  eta {format_duration(eta)}  throughput {throughput:7.1f} chars/s",
            f"current {current_batch_label}",
            f"blocks: {completed_blocks}/{total_blocks}  (1 block = {self.CHARS_PER_BLOCK} chars)  beam=*",
            f"scanline {scanline}",
            f"status: {self.COLOR_YELLOW}{status}{self.COLOR_RESET}",
            "legend: red=pending  blue=working  green=done-now  white=completed  cyan=* scan",
            "stop: requested (will exit after current batch)"
            if stop_requested
            else "stop: running (Ctrl+C once stop-after-batch, twice try abort-current)",
            f"{self.COLOR_CYAN}=============================================================={self.COLOR_RESET}",
            "",
        ]
        return "\n".join(header + lines)


def load_state(state_path: Path) -> dict[str, Any]:
    return json.loads(state_path.read_text(encoding="utf-8"))


def save_state(state_path: Path, state: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(state, indent=2, ensure_ascii=True), encoding="utf-8")
    tmp_path.replace(state_path)


def resolve_paths(base_dir: Path, parts: Iterable[str]) -> list[Path]:
    resolved: list[Path] = []
    for item in parts:
        candidate = Path(item)
        if not candidate.is_absolute():
            candidate = (base_dir / candidate).resolve()
        resolved.append(candidate)
    return resolved


def ps_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def create_continue_assets(
    run_dir: Path,
    state_path: Path,
    script_path: Path,
    remaining_batches: list[TextBatch],
    next_batch_number: int,
    runtime_options: dict[str, Any],
) -> dict[str, str]:
    remaining_text_path = run_dir / f"continue_from_batch_{next_batch_number:05d}.txt"
    remaining_text = "\n\n".join(batch.text for batch in remaining_batches).strip() + "\n"
    write_text_file(remaining_text_path, remaining_text)

    command_args = [
        "--text-file",
        str(remaining_text_path),
        "--resume-state",
        str(state_path),
        "--reference-audio",
        str(runtime_options["reference_audio"]),
        "--output",
        str(runtime_options["output_wav"]),
        "--max-chars-per-batch",
        str(runtime_options["max_chars_per_batch"]),
        "--pause-ms",
        str(runtime_options["pause_ms"]),
        "--mp3-quality",
        str(runtime_options["mp3_quality"]),
        "--inference-batch-size",
        str(runtime_options["inference_batch_size"]),
        "--max-inference-chars",
        str(runtime_options["max_inference_chars"]),
        "--language",
        str(runtime_options["language"]),
        "--model-id",
        str(runtime_options["model_id"]),
        "--device",
        str(runtime_options["device"]),
        "--dtype",
        str(runtime_options["dtype"]),
        "--attn-implementation",
        str(runtime_options["attn_implementation"]),
    ]
    if runtime_options.get("reference_text_file"):
        command_args.extend(
            ["--reference-text-file", str(runtime_options["reference_text_file"])]
        )
    if runtime_options.get("x_vector_only_mode"):
        command_args.append("--x-vector-only-mode")
    if runtime_options.get("no_defrag_ui"):
        command_args.append("--no-defrag-ui")

    sh_script_path = run_dir / f"continue_from_batch_{next_batch_number:05d}.sh"
    sh_cmd = " ".join(
        [shlex.quote("python3"), shlex.quote(str(script_path.resolve()))]
        + [shlex.quote(arg) for arg in command_args]
    )
    write_text_file(
        sh_script_path,
        "#!/usr/bin/env bash\nset -euo pipefail\n\n" + sh_cmd + "\n",
    )

    ps1_script_path = run_dir / f"continue_from_batch_{next_batch_number:05d}.ps1"
    ps_cmd = " ".join(
        ["python", ps_quote(str(script_path.resolve()))]
        + [ps_quote(arg) for arg in command_args]
    )
    write_text_file(
        ps1_script_path,
        "$ErrorActionPreference = 'Stop'\n\n" + ps_cmd + "\n",
    )

    return {
        "remaining_text_file": str(remaining_text_path),
        "continue_script_sh": str(sh_script_path),
        "continue_script_ps1": str(ps1_script_path),
    }

def require_runtime_dependencies() -> tuple[Any, Any, Any, Any]:
    missing: list[str] = []
    try:
        import numpy as np  # type: ignore
    except ImportError:
        np = None
        missing.append("numpy")
    try:
        import soundfile as sf  # type: ignore
    except ImportError:
        sf = None
        missing.append("soundfile")
    try:
        import torch  # type: ignore
    except ImportError:
        torch = None
        missing.append("torch")
    try:
        from qwen_tts import Qwen3TTSModel  # type: ignore
    except ImportError:
        Qwen3TTSModel = None
        missing.append("qwen-tts")

    if missing:
        raise RuntimeError(
            "Missing runtime dependencies: "
            + ", ".join(missing)
            + ". Run scripts/install_linux_nvidia.sh or pip install -r requirements.txt."
        )
    return np, sf, torch, Qwen3TTSModel


def choose_attention_implementation(requested: str) -> tuple[str, str | None]:
    if requested != "flash_attention_2":
        return requested, None
    has_flash_attn = importlib.util.find_spec("flash_attn") is not None
    if has_flash_attn:
        return requested, None
    return (
        "sdpa",
        "flash-attn is not installed; falling back from flash_attention_2 to sdpa.",
    )


def choose_dtype(torch: Any, requested: str, device: str) -> tuple[str, str | None]:
    device_lower = device.lower()
    if device_lower.startswith("cpu"):
        if requested != "float32":
            return "float32", "CPU mode is active; falling back dtype to float32."
        return requested, None

    if not device_lower.startswith("cuda"):
        return requested, None

    if requested != "bfloat16":
        return requested, None

    bf16_supported = False
    try:
        if hasattr(torch.cuda, "is_bf16_supported"):
            bf16_supported = bool(torch.cuda.is_bf16_supported())
    except Exception:
        bf16_supported = False

    if bf16_supported:
        return requested, None
    return (
        "float16",
        "bfloat16 requested but this GPU does not report bf16 support; falling back to float16.",
    )


def check_cuda_arch_compatibility(torch: Any, device: str) -> tuple[bool, str | None]:
    def extract_arch_code(tag: str, prefix: str) -> int | None:
        if not tag.startswith(prefix):
            return None
        suffix = tag[len(prefix) :]
        if not suffix.isdigit():
            return None
        return int(suffix)

    device_lower = device.lower()
    if not device_lower.startswith("cuda"):
        return False, None
    if not torch.cuda.is_available():
        return True, (
            "CUDA device was requested but torch.cuda.is_available() is False. "
            "Install CUDA-enabled PyTorch or use --device cpu."
        )

    try:
        index = 0
        if ":" in device:
            index = int(device.split(":", 1)[1])
        cap_major, cap_minor = torch.cuda.get_device_capability(index)
        device_name = torch.cuda.get_device_name(index)
        target_arch = f"sm_{cap_major}{cap_minor}"
        supported_arches = list(torch.cuda.get_arch_list())
    except Exception:
        return False, None

    if target_arch in supported_arches:
        return False, None

    target_code = cap_major * 10 + cap_minor
    sm_codes = [
        code
        for code in (
            extract_arch_code(arch, "sm_") for arch in supported_arches
        )
        if code is not None
    ]
    compute_codes = [
        code
        for code in (
            extract_arch_code(arch, "compute_") for arch in supported_arches
        )
        if code is not None
    ]

    # CUDA cubins are forward-compatible within the same major architecture.
    # Example: sm_86 kernels can execute on sm_89 (RTX 4090).
    same_major_compatible = any(
        (code // 10) == cap_major and (code % 10) <= cap_minor for code in sm_codes
    )
    if same_major_compatible:
        return False, (
            f"No exact {target_arch} kernel in this torch build; using same-major compatibility. "
            f"GPU: {device_name}. Installed arches: {supported_arches}."
        )

    # PTX may still JIT on newer GPUs when compute targets are present.
    ptx_candidate = any(code <= target_code for code in compute_codes)
    if ptx_candidate:
        return False, (
            f"No exact {target_arch} kernel in this torch build; PTX JIT may be used. "
            f"GPU: {device_name}. Installed arches: {supported_arches}."
        )

    return False, (
        "PyTorch CUDA build may not include optimal kernels for your GPU "
        f"({device_name}, {target_arch}). Installed arches: {supported_arches}. "
        "If runtime errors appear, reinstall torch with a different CUDA index URL "
        "(for example cu128) or use a matching pod image."
    )


def combine_parts_with_pause(
    sf: Any,
    np: Any,
    part_paths: list[Path],
    output_wav_path: Path,
    pause_ms: int,
) -> tuple[int, float]:
    if not part_paths:
        raise RuntimeError("No audio parts exist to combine.")

    first_audio, sample_rate = sf.read(str(part_paths[0]), always_2d=True)
    channels = first_audio.shape[1]
    pause_samples = int(sample_rate * (max(0, pause_ms) / 1000.0))
    silence = np.zeros((pause_samples, channels), dtype=np.float32)

    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    total_samples = 0
    with sf.SoundFile(
        str(output_wav_path),
        mode="w",
        samplerate=sample_rate,
        channels=channels,
        subtype="PCM_16",
    ) as writer:
        writer.write(first_audio.astype(np.float32))
        total_samples += first_audio.shape[0]
        for part_path in part_paths[1:]:
            audio, sr = sf.read(str(part_path), always_2d=True)
            if sr != sample_rate:
                raise RuntimeError(
                    f"Sample rate mismatch while combining parts: {part_path} ({sr}) expected {sample_rate}."
                )
            if audio.shape[1] != channels:
                if channels == 1 and audio.shape[1] > 1:
                    audio = audio.mean(axis=1, keepdims=True)
                elif channels > 1 and audio.shape[1] == 1:
                    audio = np.repeat(audio, channels, axis=1)
                else:
                    raise RuntimeError(
                        f"Channel mismatch while combining parts: {part_path} ({audio.shape[1]} channels)."
                    )
            if pause_samples > 0:
                writer.write(silence)
                total_samples += pause_samples
            writer.write(audio.astype(np.float32))
            total_samples += audio.shape[0]
    duration_seconds = total_samples / float(sample_rate)
    return int(sample_rate), float(duration_seconds)


def convert_wav_to_mp3(input_wav: Path, output_mp3: Path, quality_level: int) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError(
            "ffmpeg is required for MP3 output but was not found in PATH."
        )
    output_mp3.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_wav),
        "-vn",
        "-codec:a",
        "libmp3lame",
        "-q:a",
        str(quality_level),
        str(output_mp3),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            "High-quality MP3 conversion failed via ffmpeg."
            + (f" Detail: {detail}" if detail else "")
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create audiobooks from text using Qwen3-TTS voice cloning."
    )
    parser.add_argument("--text-file", type=Path, help="Input text file.")
    parser.add_argument(
        "--reference-audio",
        type=str,
        help="Reference audio for voice clone (local path, URL, base64, etc.).",
    )
    ref_group = parser.add_mutually_exclusive_group(required=False)
    ref_group.add_argument(
        "--reference-text", type=str, help="Transcript for the reference audio."
    )
    ref_group.add_argument(
        "--reference-text-file", type=Path, help="Path to transcript text file."
    )
    parser.add_argument(
        "--x-vector-only-mode",
        action="store_true",
        help="Use only speaker embedding; no reference transcript needed.",
    )
    parser.add_argument(
        "--resume-state",
        type=Path,
        help="Resume from an existing session_state.json file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Final output path (.mp3 recommended, .wav supported). Default: <run_dir>/audiobook.mp3",
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=Path("runs"),
        help="Root folder for run artifacts on new sessions.",
    )
    parser.add_argument(
        "--max-chars-per-batch",
        type=int,
        default=DEFAULT_MAX_CHARS,
        help=f"Maximum characters per batch (default: {DEFAULT_MAX_CHARS}).",
    )
    parser.add_argument(
        "--pause-ms",
        type=int,
        default=DEFAULT_PAUSE_MS,
        help=f"Pause between batch chunks in milliseconds (default: {DEFAULT_PAUSE_MS}).",
    )
    parser.add_argument(
        "--mp3-quality",
        type=int,
        default=0,
        help="MP3 quality for libmp3lame (0=best, 9=smallest). Used when output is .mp3.",
    )
    parser.add_argument(
        "--inference-batch-size",
        type=int,
        default=1,
        help=(
            "Compatibility option. Batched inference is disabled for stability; "
            "values greater than 1 are ignored."
        ),
    )
    parser.add_argument(
        "--max-inference-chars",
        type=int,
        default=DEFAULT_MAX_INFERENCE_CHARS,
        help=(
            "Compatibility option retained for older continue scripts; ignored."
        ),
    )
    parser.add_argument(
        "--language",
        type=str,
        default="Auto",
        help="Language for generation. Use Auto for automatic detection.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=DEFAULT_MODEL_ID,
        help=f"Qwen model id/path (default: {DEFAULT_MODEL_ID}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='Device map for qwen-tts model loading (default: "cuda:0").',
    )
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default=DEFAULT_DTYPE,
        help="Model dtype.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=DEFAULT_ATTN_IMPLEMENTATION,
        help=(
            "Attention implementation for qwen-tts model loading "
            f"(default: {DEFAULT_ATTN_IMPLEMENTATION})."
        ),
    )
    parser.add_argument(
        "--no-defrag-ui",
        action="store_true",
        help="Disable defrag-style progress UI.",
    )
    parser.add_argument(
        "--stop-after-batch",
        type=int,
        default=0,
        help="Testing helper: stop after N batches in this invocation.",
    )
    return parser.parse_args()

def main() -> int:
    args = parse_args()
    stop_controller = StopController()
    stop_controller.install()

    if args.max_chars_per_batch < 100:
        print("ERROR: --max-chars-per-batch must be at least 100.", file=sys.stderr)
        return 2
    if args.pause_ms < 0:
        print("ERROR: --pause-ms cannot be negative.", file=sys.stderr)
        return 2
    if args.mp3_quality < 0 or args.mp3_quality > 9:
        print("ERROR: --mp3-quality must be between 0 and 9.", file=sys.stderr)
        return 2
    if args.inference_batch_size < 1:
        print("ERROR: --inference-batch-size must be at least 1.", file=sys.stderr)
        return 2

    is_resume = args.resume_state is not None
    if is_resume:
        state_path = args.resume_state.resolve()
        if not state_path.exists():
            print(f"ERROR: Resume state file not found: {state_path}", file=sys.stderr)
            return 2
        state = load_state(state_path)
        run_dir = Path(state["run_dir"]).resolve()
    else:
        if not args.text_file:
            print("ERROR: --text-file is required for new runs.", file=sys.stderr)
            return 2
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = (args.run_root / f"{args.text_file.stem}_{run_id}").resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        state_path = run_dir / "session_state.json"
        state = {}

    run_dir.mkdir(parents=True, exist_ok=True)
    parts_dir = run_dir / "parts"
    parts_dir.mkdir(parents=True, exist_ok=True)

    text_file = args.text_file or (Path(state["remaining_text_file"]) if state.get("remaining_text_file") else None)
    if not text_file:
        print("ERROR: --text-file is required.", file=sys.stderr)
        return 2
    text_file = text_file.resolve()
    if not text_file.exists():
        print(f"ERROR: text file not found: {text_file}", file=sys.stderr)
        return 2

    reference_audio = args.reference_audio or state.get("reference_audio")
    if not reference_audio:
        print("ERROR: --reference-audio is required.", file=sys.stderr)
        return 2

    if args.reference_text_file:
        reference_text = read_text_file(args.reference_text_file.resolve()).strip()
    elif args.reference_text:
        reference_text = args.reference_text.strip()
    elif state.get("reference_text"):
        reference_text = str(state["reference_text"]).strip()
    elif state.get("reference_text_file"):
        reference_text = read_text_file(Path(state["reference_text_file"]).resolve()).strip()
    else:
        reference_text = ""

    x_vector_only_mode = bool(args.x_vector_only_mode or state.get("x_vector_only_mode"))
    if not x_vector_only_mode and not reference_text:
        print(
            "ERROR: reference transcript is required (or set --x-vector-only-mode).",
            file=sys.stderr,
        )
        return 2

    model_id = (
        state.get("model_id")
        if is_resume and args.model_id == DEFAULT_MODEL_ID and state.get("model_id")
        else args.model_id
    )
    max_chars = (
        int(state["max_chars_per_batch"])
        if is_resume
        and args.max_chars_per_batch == DEFAULT_MAX_CHARS
        and state.get("max_chars_per_batch")
        else args.max_chars_per_batch
    )
    pause_ms = (
        int(state["pause_ms"])
        if is_resume and args.pause_ms == DEFAULT_PAUSE_MS and state.get("pause_ms")
        else args.pause_ms
    )
    mp3_quality = (
        int(state["mp3_quality"])
        if is_resume and args.mp3_quality == 0 and state.get("mp3_quality") is not None
        else args.mp3_quality
    )
    inference_batch_size = (
        int(state["inference_batch_size"])
        if is_resume
        and args.inference_batch_size == 1
        and state.get("inference_batch_size")
        else args.inference_batch_size
    )
    max_inference_chars = (
        int(state["max_inference_chars"])
        if is_resume
        and args.max_inference_chars == DEFAULT_MAX_INFERENCE_CHARS
        and state.get("max_inference_chars")
        else args.max_inference_chars
    )
    if inference_batch_size != 1:
        warning = (
            f"--inference-batch-size={inference_batch_size} requested, "
            "but batched inference is disabled for stability; forcing to 1."
        )
        print(f"WARNING: {warning}")
        inference_batch_size = 1
    language = (
        str(state["language"])
        if is_resume and args.language == "Auto" and state.get("language")
        else args.language
    )
    device = (
        str(state["device"])
        if is_resume and args.device == "cuda:0" and state.get("device")
        else args.device
    )
    dtype_name = (
        str(state["dtype"])
        if is_resume and args.dtype == DEFAULT_DTYPE and state.get("dtype")
        else args.dtype
    )
    attn = (
        str(state["attn_implementation"])
        if is_resume
        and args.attn_implementation == DEFAULT_ATTN_IMPLEMENTATION
        and state.get("attn_implementation")
        else args.attn_implementation
    )
    output_target = (
        args.output.resolve()
        if args.output
        else Path(state.get("output_audio") or state.get("output_wav")).resolve()
        if is_resume and (state.get("output_audio") or state.get("output_wav"))
        else run_dir / DEFAULT_OUTPUT_NAME
    )
    if output_target.suffix == "":
        output_target = output_target.with_suffix(".mp3")
    output_suffix = output_target.suffix.lower()
    if output_suffix not in (".mp3", ".wav"):
        print(
            "ERROR: --output must end with .mp3 or .wav (or omit extension to default to .mp3).",
            file=sys.stderr,
        )
        return 2
    if output_suffix == ".mp3" and not shutil.which("ffmpeg"):
        print(
            "ERROR: ffmpeg is required for MP3 output but was not found in PATH.",
            file=sys.stderr,
        )
        return 2

    reference_text_file: Path | None = None
    if reference_text:
        reference_text_file = run_dir / "reference_text.txt"
        write_text_file(reference_text_file, reference_text + "\n")
    elif state.get("reference_text_file"):
        candidate = Path(state["reference_text_file"]).resolve()
        if candidate.exists():
            reference_text_file = candidate

    paragraphs = split_into_paragraphs(read_text_file(text_file))
    if not paragraphs:
        print(f"ERROR: no usable paragraphs found in {text_file}.", file=sys.stderr)
        return 2
    batches = build_batches(paragraphs, max_chars)

    part_files: list[str] = list(state.get("part_files", []))
    for existing_part in resolve_paths(run_dir, part_files):
        if not existing_part.exists():
            print(f"ERROR: missing part file from state: {existing_part}", file=sys.stderr)
            return 2

    existing_batches = len(part_files)
    existing_chars = int(state.get("completed_characters", 0))
    total_batches = existing_batches + len(batches)
    total_chars = existing_chars + sum(batch.char_count for batch in batches)
    fallback_batch_chars = max(1, existing_chars // max(1, existing_batches)) if existing_batches else max_chars
    raw_saved_batch_chars = state.get("batch_char_counts", [])
    saved_batch_chars: list[int] = []
    if isinstance(raw_saved_batch_chars, list):
        for value in raw_saved_batch_chars[:existing_batches]:
            try:
                saved_batch_chars.append(max(1, int(value)))
            except (TypeError, ValueError):
                saved_batch_chars.append(fallback_batch_chars)
    if len(saved_batch_chars) < existing_batches:
        saved_batch_chars.extend([fallback_batch_chars] * (existing_batches - len(saved_batch_chars)))
    if len(saved_batch_chars) > existing_batches:
        saved_batch_chars = saved_batch_chars[:existing_batches]

    new_batch_chars = [batch.char_count for batch in batches]
    all_batch_chars_for_view = saved_batch_chars + new_batch_chars
    state_batch_chars = list(saved_batch_chars)

    progress = DefragProgressView(
        total_batches=total_batches,
        total_chars=total_chars,
        completed_batches=existing_batches,
        completed_chars=existing_chars,
        enabled=not args.no_defrag_ui,
        batch_char_counts=all_batch_chars_for_view,
    )
    progress.start()

    try:
        np, sf, torch, Qwen3TTSModel = require_runtime_dependencies()
        arch_fatal, arch_message = check_cuda_arch_compatibility(torch, device)
        if arch_message:
            progress.set_status(arch_message)
            if args.no_defrag_ui:
                print(f"WARNING: {arch_message}")
        if arch_fatal and arch_message:
            raise RuntimeError(arch_message)

        dtype_name, dtype_warning = choose_dtype(torch, dtype_name, device)
        if dtype_warning:
            progress.set_status(dtype_warning)
            if args.no_defrag_ui:
                print(f"WARNING: {dtype_warning}")

        attn, attn_warning = choose_attention_implementation(attn)
        if attn_warning:
            progress.set_status(attn_warning)
            if args.no_defrag_ui:
                print(f"WARNING: {attn_warning}")
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        try:
            model = Qwen3TTSModel.from_pretrained(
                model_id,
                device_map=device,
                dtype=dtype_map[dtype_name],
                attn_implementation=attn,
            )
        except Exception as model_exc:
            if attn != "sdpa":
                fallback_msg = (
                    f"Model load failed with attn '{attn}', retrying with 'sdpa'. "
                    f"Reason: {model_exc}"
                )
                progress.set_status(fallback_msg)
                if args.no_defrag_ui:
                    print(f"WARNING: {fallback_msg}")
                attn = "sdpa"
                model = Qwen3TTSModel.from_pretrained(
                    model_id,
                    device_map=device,
                    dtype=dtype_map[dtype_name],
                    attn_implementation=attn,
                )
            else:
                raise
        progress.set_status("Building clone prompt...")
        clone_prompt = model.create_voice_clone_prompt(
            ref_audio=reference_audio,
            ref_text=reference_text if reference_text else None,
            x_vector_only_mode=x_vector_only_mode,
        )

        state.update(
            {
                "app_version": APP_VERSION,
                "created_at": state.get("created_at", now_iso()),
                "updated_at": now_iso(),
                "run_dir": str(run_dir),
                "source_text_file": str(text_file),
                "reference_audio": reference_audio,
                "reference_text": reference_text,
                "reference_text_file": str(reference_text_file)
                if reference_text_file
                else None,
                "x_vector_only_mode": x_vector_only_mode,
                "output_audio": str(output_target),
                "output_wav": str(output_target),
                "model_id": model_id,
                "device": device,
                "dtype": dtype_name,
                "attn_implementation": attn,
                "language": language,
                "max_chars_per_batch": max_chars,
                "pause_ms": pause_ms,
                "mp3_quality": mp3_quality,
                "inference_batch_size": inference_batch_size,
                "max_inference_chars": max_inference_chars,
                "part_files": part_files,
                "batch_char_counts": state_batch_chars,
                "completed_batches": existing_batches,
                "completed_characters": existing_chars,
                "stopped_early": False,
                "remaining_text_file": None,
                "continue_script_sh": None,
                "continue_script_ps1": None,
            }
        )
        save_state(state_path, state)

        completed_this_run = 0
        completed_chars_total = existing_chars
        runtime_options = {
            "reference_audio": reference_audio,
            "reference_text_file": reference_text_file,
            "output_wav": output_target,
            "max_chars_per_batch": max_chars,
            "pause_ms": pause_ms,
            "mp3_quality": mp3_quality,
            "inference_batch_size": inference_batch_size,
            "max_inference_chars": max_inference_chars,
            "language": language,
            "model_id": model_id,
            "device": device,
            "dtype": dtype_name,
            "attn_implementation": attn,
            "x_vector_only_mode": x_vector_only_mode,
            "no_defrag_ui": args.no_defrag_ui,
        }
        script_path = Path(__file__).resolve()

        for local_index, batch in enumerate(batches, start=1):
            if stop_controller.force_stop:
                break

            global_batch = existing_batches + completed_this_run + 1
            batch_label = f"batch {global_batch}/{total_batches} ({batch.char_count} chars)"

            progress.set_active_batch(
                global_batch,
                batch.char_count,
                f"Generating {batch_label}...",
                batch_label=batch_label,
                active_batch_count=1,
            )
            if args.no_defrag_ui:
                print(f"{batch_label}: generating...")

            started = time.time()
            kwargs: dict[str, Any] = {
                "text": batch.text,
                "voice_clone_prompt": clone_prompt,
            }
            if language.lower() != "auto":
                kwargs["language"] = language

            wavs: Any
            sample_rate: Any
            cancel_succeeded = False
            discard_batch_after_completion = False
            stop_message_shown = False
            abort_message_shown = False
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(model.generate_voice_clone, **kwargs)
                while True:
                    try:
                        wavs, sample_rate = future.result(timeout=0.15)
                        break
                    except concurrent.futures.TimeoutError:
                        if stop_controller.stop_requested and not stop_message_shown:
                            stop_message_shown = True
                            progress.mark_stop_requested()
                            if args.no_defrag_ui:
                                print(
                                    "Stop requested; finishing current batch (Ctrl+C again to try abort)."
                                )
                        if stop_controller.abort_current_batch and not abort_message_shown:
                            abort_message_shown = True
                            progress.mark_abort_requested()
                            if args.no_defrag_ui:
                                print(
                                    "Abort requested; attempting to cancel current batch..."
                                )
                            cancel_succeeded = future.cancel()
                            if cancel_succeeded:
                                break
                            discard_batch_after_completion = True
                        continue

            if cancel_succeeded:
                progress.mark_batch_aborted(canceled=True)
                stop_controller.request_stop()
                if args.no_defrag_ui:
                    print(
                        f"Batch {global_batch}: canceled before execution; will remain in continue file."
                    )
                break

            if not wavs:
                raise RuntimeError(f"Empty audio output for {batch_label}.")

            if discard_batch_after_completion:
                progress.mark_batch_aborted(canceled=False)
                stop_controller.request_stop()
                if args.no_defrag_ui:
                    print(
                        f"Batch {global_batch}: could not be interrupted; output discarded so it remains in continue file."
                    )
                break

            duration = time.time() - started
            part_path = parts_dir / f"batch_{global_batch:05d}.wav"
            sf.write(str(part_path), wavs[0], sample_rate)
            part_files.append(str(part_path.relative_to(run_dir).as_posix()))

            completed_this_run += 1
            completed_chars_total += batch.char_count
            state_batch_chars.append(batch.char_count)
            progress.mark_batch_complete(batch.char_count, duration)
            if args.no_defrag_ui:
                print(f"Batch {global_batch}: complete in {duration:.1f}s")

            state["part_files"] = part_files
            state["batch_char_counts"] = state_batch_chars
            state["completed_batches"] = len(part_files)
            state["completed_characters"] = completed_chars_total
            state["sample_rate"] = int(sample_rate)
            state["updated_at"] = now_iso()
            save_state(state_path, state)

            if args.stop_after_batch > 0 and local_index >= args.stop_after_batch:
                stop_controller.request_stop()
            if stop_controller.stop_requested:
                progress.mark_stop_requested()
                break

        combined_wav_path = (
            output_target
            if output_suffix == ".wav"
            else run_dir / "audiobook_combined_intermediate.wav"
        )
        progress.set_status("Combining audio parts with pauses...")
        sample_rate_out, duration_out = combine_parts_with_pause(
            sf=sf,
            np=np,
            part_paths=resolve_paths(run_dir, part_files),
            output_wav_path=combined_wav_path,
            pause_ms=pause_ms,
        )
        if output_suffix == ".mp3":
            progress.set_status("Encoding high-quality MP3...")
            convert_wav_to_mp3(
                input_wav=combined_wav_path,
                output_mp3=output_target,
                quality_level=mp3_quality,
            )
            if combined_wav_path.exists():
                try:
                    combined_wav_path.unlink()
                except OSError:
                    pass

        remaining = batches[completed_this_run:]
        if remaining:
            next_batch = existing_batches + completed_this_run + 1
            assets = create_continue_assets(
                run_dir=run_dir,
                state_path=state_path,
                script_path=script_path,
                remaining_batches=remaining,
                next_batch_number=next_batch,
                runtime_options=runtime_options,
            )
            state.update(
                {
                    "stopped_early": True,
                    "remaining_text_file": assets["remaining_text_file"],
                    "continue_script_sh": assets["continue_script_sh"],
                    "continue_script_ps1": assets["continue_script_ps1"],
                    "updated_at": now_iso(),
                }
            )
            save_state(state_path, state)
            progress.stop("Stopped early. Continue assets were generated.")
            print("Stopped early. Continue assets were generated.")
            print(f"Audio: {output_target}")
            print(f"State: {state_path}")
            print(f"Remaining text: {assets['remaining_text_file']}")
            print(f"Continue script (bash): {assets['continue_script_sh']}")
            print(f"Continue script (powershell): {assets['continue_script_ps1']}")
            print(f"Current duration: {duration_out:.1f}s at {sample_rate_out} Hz")
            return 0

        state.update(
            {
                "stopped_early": False,
                "remaining_text_file": None,
                "continue_script_sh": None,
                "continue_script_ps1": None,
                "updated_at": now_iso(),
            }
        )
        save_state(state_path, state)
        progress.stop("Audiobook generation complete.")
        print(f"Done: {output_target}")
        print(f"Batches rendered this run: {completed_this_run}")
        print(f"Total combined duration: {duration_out:.1f}s at {sample_rate_out} Hz")
        print(f"State: {state_path}")
        return 0
    except Exception as exc:
        error_text = str(exc)
        if "device-side assert triggered" in error_text.lower():
            error_text += (
                " | CUDA context is now invalid for this process. Restart and retry with "
                "single-batch generation (this app now forces one batch per inference call)."
            )
        if "no kernel image is available for execution on the device" in error_text:
            error_text += (
                " | likely GPU/torch CUDA architecture mismatch. "
                "Try: (1) reinstall newer torch CUDA wheels (e.g. cu128 on RunPod), "
                "(2) use --dtype float16, (3) use --attn-implementation sdpa."
            )
        progress.stop(f"Failed: {exc}")
        print(f"ERROR: {error_text}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
