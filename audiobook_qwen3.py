#!/usr/bin/env python3
"""
Create audiobook WAV files from text using Qwen3-TTS voice cloning.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

APP_VERSION = "0.1.0"
DEFAULT_MODEL_ID = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
DEFAULT_OUTPUT_NAME = "audiobook.wav"
DEFAULT_MAX_CHARS = 1800
DEFAULT_PAUSE_MS = 300


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
        self._force_stop = False

    @property
    def stop_requested(self) -> bool:
        with self._lock:
            return self._stop_requested

    @property
    def force_stop(self) -> bool:
        with self._lock:
            return self._force_stop

    def request_stop(self) -> None:
        with self._lock:
            self._stop_requested = True

    def install(self) -> None:
        def handler(_signum: int, _frame: Any) -> None:
            with self._lock:
                if not self._stop_requested:
                    self._stop_requested = True
                else:
                    self._force_stop = True

        signal.signal(signal.SIGINT, handler)
        signal.signal(signal.SIGTERM, handler)


class DefragProgressView:
    def __init__(
        self,
        total_batches: int,
        total_chars: int,
        completed_batches: int,
        completed_chars: int,
        enabled: bool,
    ) -> None:
        self.total_batches = max(1, total_batches)
        self.total_chars = max(1, total_chars)
        self.completed_batches = max(0, completed_batches)
        self.completed_chars = max(0, completed_chars)
        self.enabled = enabled and sys.stdout.isatty()
        self.current_batch = 0
        self.current_batch_chars = 0
        self.stop_requested = False
        self.status = "Loading model..."
        self._batch_durations: list[float] = []
        self._start_time = time.time()
        self._tick = 0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if not self.enabled:
            return
        self._thread = threading.Thread(target=self._render_loop, daemon=True)
        self._thread.start()

    def set_status(self, message: str) -> None:
        with self._lock:
            self.status = message

    def set_active_batch(self, batch_number: int, batch_chars: int, message: str) -> None:
        with self._lock:
            self.current_batch = batch_number
            self.current_batch_chars = batch_chars
            self.status = message

    def mark_batch_complete(self, batch_chars: int, duration_seconds: float) -> None:
        with self._lock:
            self.completed_batches += 1
            self.completed_chars += batch_chars
            self.current_batch = 0
            self.current_batch_chars = 0
            self._batch_durations.append(duration_seconds)
            if len(self._batch_durations) > 12:
                self._batch_durations.pop(0)
            self.status = "Batch complete."

    def mark_stop_requested(self) -> None:
        with self._lock:
            self.stop_requested = True
            self.status = "Stop requested; finishing current batch."

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

    def _build_frame(self) -> str:
        with self._lock:
            completed_batches = self.completed_batches
            completed_chars = self.completed_chars
            current_batch = self.current_batch
            current_batch_chars = self.current_batch_chars
            status = self.status
            stop_requested = self.stop_requested
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

        cols = 60
        rows = 8
        total_cells = cols * rows
        done_cells = int(total_cells * ratio)
        cells = ["." for _ in range(total_cells)]
        for idx in range(done_cells):
            cells[idx] = "#"

        if current_batch > 0 and done_cells < total_cells:
            remain = total_cells - done_cells
            span = min(max(4, remain // 10), remain)
            start = done_cells + (self._tick * 3) % max(1, remain)
            for offset in range(span):
                pos = done_cells + ((start - done_cells + offset) % remain)
                cells[pos] = "+"

        head = (self._tick * 5) % total_cells
        cells[head] = "@"
        grid = ["".join(cells[row * cols : (row + 1) * cols]) for row in range(rows)]

        current_label = (
            f"batch {current_batch}/{self.total_batches} ({current_batch_chars} chars)"
            if current_batch
            else "idle"
        )
        header = [
            "Qwen3-TTS Audiobook Defrag Progress",
            f"[{bar}] {percent:6.2f}%  batches {completed_batches}/{self.total_batches}  chars {completed_chars}/{self.total_chars}",
            f"elapsed {format_duration(elapsed)}  eta {format_duration(eta)}  current {current_label}",
            f"status: {status}",
            "legend: @ head  + active  # done  . queued",
            "stop: requested (will exit after current batch)"
            if stop_requested
            else "stop: running (press Ctrl+C once for graceful stop)",
            "",
        ]
        return "\n".join(header + grid)


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
        help="Output audiobook WAV path. Default: <run_dir>/audiobook.wav",
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
        default="bfloat16",
        help="Model dtype.",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default="flash_attention_2",
        help="Attention implementation for qwen-tts model loading.",
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
        if is_resume and args.dtype == "bfloat16" and state.get("dtype")
        else args.dtype
    )
    attn = (
        str(state["attn_implementation"])
        if is_resume
        and args.attn_implementation == "flash_attention_2"
        and state.get("attn_implementation")
        else args.attn_implementation
    )
    output_wav = (
        args.output.resolve()
        if args.output
        else Path(state["output_wav"]).resolve()
        if is_resume and state.get("output_wav")
        else run_dir / DEFAULT_OUTPUT_NAME
    )

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

    progress = DefragProgressView(
        total_batches=total_batches,
        total_chars=total_chars,
        completed_batches=existing_batches,
        completed_chars=existing_chars,
        enabled=not args.no_defrag_ui,
    )
    progress.start()

    try:
        np, sf, torch, Qwen3TTSModel = require_runtime_dependencies()
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=device,
            dtype=dtype_map[dtype_name],
            attn_implementation=attn,
        )
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
                "output_wav": str(output_wav),
                "model_id": model_id,
                "device": device,
                "dtype": dtype_name,
                "attn_implementation": attn,
                "language": language,
                "max_chars_per_batch": max_chars,
                "pause_ms": pause_ms,
                "part_files": part_files,
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
            "output_wav": output_wav,
            "max_chars_per_batch": max_chars,
            "pause_ms": pause_ms,
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
            progress.set_active_batch(
                global_batch,
                batch.char_count,
                f"Generating batch {global_batch}/{total_batches}...",
            )
            if args.no_defrag_ui:
                print(
                    f"Batch {global_batch}/{total_batches} ({batch.char_count} chars): generating..."
                )

            started = time.time()
            kwargs: dict[str, Any] = {
                "text": batch.text,
                "voice_clone_prompt": clone_prompt,
            }
            if language.lower() != "auto":
                kwargs["language"] = language
            wavs, sample_rate = model.generate_voice_clone(**kwargs)
            if not wavs:
                raise RuntimeError(f"Empty audio output for batch {global_batch}.")

            part_path = parts_dir / f"batch_{global_batch:05d}.wav"
            sf.write(str(part_path), wavs[0], sample_rate)
            part_files.append(str(part_path.relative_to(run_dir).as_posix()))

            duration = time.time() - started
            completed_this_run += 1
            completed_chars_total += batch.char_count
            progress.mark_batch_complete(batch.char_count, duration)
            if args.no_defrag_ui:
                print(f"Batch {global_batch}: complete in {duration:.1f}s")

            state["part_files"] = part_files
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

        progress.set_status("Combining audio parts with pauses...")
        sample_rate_out, duration_out = combine_parts_with_pause(
            sf=sf,
            np=np,
            part_paths=resolve_paths(run_dir, part_files),
            output_wav_path=output_wav,
            pause_ms=pause_ms,
        )

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
            print("Stopped early after current batch.")
            print(f"Audio: {output_wav}")
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
        print(f"Done: {output_wav}")
        print(f"Batches rendered this run: {completed_this_run}")
        print(f"Total combined duration: {duration_out:.1f}s at {sample_rate_out} Hz")
        print(f"State: {state_path}")
        return 0
    except Exception as exc:
        progress.stop(f"Failed: {exc}")
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
