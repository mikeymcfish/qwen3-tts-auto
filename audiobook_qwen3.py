#!/usr/bin/env python3
"""
Create audiobook files from text using MOSS-TTS or Qwen3-TTS voice cloning.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import importlib
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
from io import TextIOBase
from pathlib import Path
from typing import Any, Callable, Iterable

APP_VERSION = "0.1.0"
DEFAULT_TTS_BACKEND = "moss-delay"
DEFAULT_MODEL_ID_BY_BACKEND = {
    "moss-delay": "OpenMOSS-Team/MOSS-TTS",
    "moss-local": "OpenMOSS-Team/MOSS-TTS-Local-Transformer",
    "qwen": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}
DEFAULT_MODEL_ID = DEFAULT_MODEL_ID_BY_BACKEND[DEFAULT_TTS_BACKEND]
DEFAULT_OUTPUT_NAME = "audiobook.mp3"
DEFAULT_MAX_CHARS = 1800
DEFAULT_PAUSE_MS = 300
DEFAULT_CHAPTER_PAUSE_MS = 0
DEFAULT_MAX_INFERENCE_CHARS = 2600
DEFAULT_MAX_NEW_TOKENS = 4096
DEFAULT_MOSS_AUDIO_TEMPERATURE = 1.7
DEFAULT_MOSS_AUDIO_TOP_P = 0.8
DEFAULT_MOSS_AUDIO_TOP_K = 25
DEFAULT_MOSS_AUDIO_REPETITION_PENALTY = 1.0
DEFAULT_ATTN_IMPLEMENTATION = "sdpa"
DEFAULT_DTYPE = "bfloat16"
REFERENCE_AUDIO_EXTENSIONS = {
    ".wav",
    ".mp3",
    ".flac",
    ".m4a",
    ".aac",
    ".ogg",
    ".opus",
    ".webm",
}
BREAK_TAG = "[BREAK]"
CHAPTER_TAG = "[CHAPTER]"


@dataclass
class TextBatch:
    index: int
    start_paragraph: int
    end_paragraph: int
    text: str
    char_count: int
    starts_chapter: bool = False
    forced_break_before: bool = False
    chapter_title: str | None = None


def normalize_tts_backend(value: str) -> str:
    raw = (value or "").strip().lower()
    aliases = {
        "moss": "moss-delay",
        "moss-tts": "moss-delay",
        "delay": "moss-delay",
        "moss-delay": "moss-delay",
        "local": "moss-local",
        "moss-local": "moss-local",
        "moss-local-transformer": "moss-local",
        "qwen3": "qwen",
        "qwen": "qwen",
        "auto": "auto",
    }
    return aliases.get(raw, raw)


def infer_backend_from_model_id(model_id: str) -> str:
    lowered = (model_id or "").strip().lower()
    if "moss" in lowered:
        if "local" in lowered:
            return "moss-local"
        return "moss-delay"
    return "qwen"


def resolve_tts_backend(requested_backend: str, model_id: str) -> str:
    normalized = normalize_tts_backend(requested_backend)
    if normalized == "auto":
        return infer_backend_from_model_id(model_id)
    if normalized not in {"qwen", "moss-delay", "moss-local"}:
        raise ValueError(
            f"Unsupported --tts-backend '{requested_backend}'. "
            "Use one of: qwen, moss-delay, moss-local, auto."
        )
    return normalized


def is_moss_backend(backend: str) -> bool:
    return backend in {"moss-delay", "moss-local"}


def recommended_default_model_id(backend: str) -> str:
    return DEFAULT_MODEL_ID_BY_BACKEND.get(backend, DEFAULT_MODEL_ID)


def parse_cuda_device_index(device: str) -> int:
    if ":" not in device:
        return 0
    suffix = device.split(":", 1)[1]
    if suffix.isdigit():
        return int(suffix)
    return 0


def detect_cuda_total_memory_gb(torch: Any, device: str) -> float | None:
    device_lower = (device or "").lower()
    if not device_lower.startswith("cuda"):
        return None
    if not getattr(torch.cuda, "is_available", lambda: False)():
        return None
    try:
        props = torch.cuda.get_device_properties(parse_cuda_device_index(device))
        total_bytes = float(getattr(props, "total_memory", 0))
    except Exception:
        return None
    if total_bytes <= 0:
        return None
    return total_bytes / float(1024**3)


def choose_inference_batch_size(
    requested: int,
    backend: str,
    torch: Any,
    device: str,
    max_chars_per_batch: int,
) -> tuple[int, str | None]:
    if backend == "qwen":
        if requested > 1:
            return (
                1,
                f"--inference-batch-size={requested} requested, but qwen backend is single-item only; forcing to 1.",
            )
        return (1, None)

    if requested >= 1:
        return (requested, None)

    if not (device or "").lower().startswith("cuda"):
        return (1, "Auto inference batch size selected 1 (non-CUDA device).")
    if not torch.cuda.is_available():
        return (1, "Auto inference batch size selected 1 (CUDA unavailable).")

    memory_gb = detect_cuda_total_memory_gb(torch, device)
    if memory_gb is None:
        return (1, "Auto inference batch size selected 1 (unable to detect GPU memory).")

    if backend == "moss-delay":
        if memory_gb >= 48:
            base = 4
        elif memory_gb >= 32:
            base = 3
        elif memory_gb >= 24:
            base = 2
        else:
            base = 1
    else:
        if memory_gb >= 48:
            base = 8
        elif memory_gb >= 24:
            base = 6
        elif memory_gb >= 16:
            base = 4
        elif memory_gb >= 12:
            base = 2
        else:
            base = 1

    if max_chars_per_batch >= 2600:
        base = max(1, base - 1)
    elif max_chars_per_batch <= 1200:
        base = min(8, base + 1)

    return (
        base,
        (
            f"Auto inference batch size selected {base} for {backend} on "
            f"GPU memory {memory_gb:.1f} GiB with max chars {max_chars_per_batch}."
        ),
    )


def apply_continuation_chain_constraints(
    backend: str,
    continuation_chain: bool,
    inference_batch_size: int,
) -> tuple[int, str | None]:
    if not continuation_chain:
        return inference_batch_size, None
    if not is_moss_backend(backend):
        raise ValueError(
            "--continuation-chain is supported only for moss-delay or moss-local backends."
        )
    if inference_batch_size != 1:
        return (
            1,
            "--continuation-chain requires sequential inference; forcing --inference-batch-size=1.",
        )
    return 1, None


def is_cuda_oom_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    indicators = (
        "cuda out of memory",
        "out of memory",
        "cublas_status_alloc_failed",
        "cuda error: out of memory",
    )
    return any(marker in text for marker in indicators)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def format_plain_progress(
    completed_batches: int,
    total_batches: int,
    completed_chars: int,
    total_chars: int,
    elapsed_seconds: float,
    average_batch_seconds: float,
) -> str:
    safe_total_batches = max(1, total_batches)
    safe_total_chars = max(1, total_chars)
    batch_pct = (max(0, completed_batches) / safe_total_batches) * 100.0
    char_pct = (max(0, completed_chars) / safe_total_chars) * 100.0
    remaining_batches = max(0, safe_total_batches - max(0, completed_batches))
    eta_seconds = max(0.0, average_batch_seconds * remaining_batches)
    return (
        "progress: "
        f"batches {completed_batches}/{safe_total_batches} ({batch_pct:5.1f}%), "
        f"chars {completed_chars}/{safe_total_chars} ({char_pct:5.1f}%), "
        f"elapsed {format_duration(elapsed_seconds)}, "
        f"eta {format_duration(eta_seconds)}"
    )


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


def find_matching_reference_text_file(reference_audio: str) -> Path | None:
    if not reference_audio:
        return None
    try:
        audio_path = Path(reference_audio).expanduser()
    except Exception:
        return None
    if not audio_path.exists() or not audio_path.is_file():
        return None
    transcript_path = audio_path.with_suffix(".txt")
    if transcript_path.exists() and transcript_path.is_file():
        return transcript_path.resolve()
    return None


def scan_reference_audio_candidates(scan_dir: Path) -> list[tuple[Path, Path]]:
    if not scan_dir.exists() or not scan_dir.is_dir():
        return []
    candidates: list[tuple[Path, Path]] = []
    for path in sorted(scan_dir.iterdir(), key=lambda item: item.name.lower()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in REFERENCE_AUDIO_EXTENSIONS:
            continue
        transcript_path = path.with_suffix(".txt")
        if not transcript_path.exists() or not transcript_path.is_file():
            continue
        candidates.append((path.resolve(), transcript_path.resolve()))
    return candidates


def prompt_for_reference_audio_selection(
    candidates: list[tuple[Path, Path]],
    scan_dir: Path,
    input_fn: Callable[[str], str] = input,
    output_stream: TextIOBase | Any = sys.stdout,
) -> Path | None:
    if not candidates:
        return None
    print("", file=output_stream)
    print("No --reference-audio was provided.", file=output_stream)
    print(
        f"Found {len(candidates)} audio files with matching .txt transcripts in:",
        file=output_stream,
    )
    print(f"  {scan_dir}", file=output_stream)
    for idx, (audio_path, transcript_path) in enumerate(candidates, start=1):
        print(
            f"  {idx}. {audio_path.name} (transcript: {transcript_path.name})",
            file=output_stream,
        )

    while True:
        try:
            raw_choice = input_fn(
                "Select a reference audio number (or 'q' to cancel): "
            ).strip()
        except (EOFError, KeyboardInterrupt):
            return None
        if raw_choice.lower() in {"q", "quit", "exit"}:
            return None
        if raw_choice.isdigit():
            idx = int(raw_choice)
            if 1 <= idx <= len(candidates):
                return candidates[idx - 1][0]
        print(
            f"Invalid selection. Enter a number between 1 and {len(candidates)}, or 'q'.",
            file=output_stream,
        )


def split_into_paragraphs(raw_text: str) -> list[str]:
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not normalized:
        return []
    normalized = re.sub(
        r"(?i)\[(BREAK|CHAPTER)\]",
        lambda match: f"\n\n[{match.group(1).upper()}]\n\n",
        normalized,
    )
    chunks = re.split(r"\n\s*\n+", normalized)
    paragraphs: list[str] = []
    for chunk in chunks:
        cleaned = re.sub(r"[ \t]+", " ", chunk.strip())
        cleaned = re.sub(r"\n+", " ", cleaned).strip()
        if cleaned:
            paragraphs.append(cleaned)
    return paragraphs


def extract_chapter_titles_from_raw_text(raw_text: str) -> list[str]:
    normalized = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    chapter_pattern = re.compile(r"(?i)\[CHAPTER\]")
    titles: list[str] = []
    for line in lines:
        matches = list(chapter_pattern.finditer(line))
        if not matches:
            continue
        for idx, match in enumerate(matches):
            title_start = match.end()
            title_end = matches[idx + 1].start() if idx + 1 < len(matches) else len(line)
            title_text = re.sub(r"[ \t]+", " ", line[title_start:title_end].strip())
            titles.append(title_text)
    return titles


def split_into_sentences(paragraph: str) -> list[str]:
    stripped = paragraph.strip()
    if not stripped:
        return []
    parts = re.split(r"(?<=[.!?])\s+", stripped)
    sentences = [part.strip() for part in parts if part.strip()]
    return sentences if sentences else [stripped]


def split_text_to_fit(text: str, max_chars: int) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    words = text.split(" ")
    current_words: list[str] = []
    current_len = 0
    for word in words:
        if not word:
            continue
        add_len = len(word) if not current_words else len(word) + 1
        if current_words and current_len + add_len > max_chars:
            chunks.append(" ".join(current_words))
            current_words = [word]
            current_len = len(word)
            continue
        current_words.append(word)
        current_len += add_len
    if current_words:
        chunks.append(" ".join(current_words))
    if not chunks:
        chunks = [text]

    bounded: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            bounded.append(chunk)
            continue
        start = 0
        while start < len(chunk):
            bounded.append(chunk[start : start + max_chars])
            start += max_chars
    return bounded


def split_paragraph_for_batches(paragraph: str, max_chars: int) -> list[str]:
    sentences = split_into_sentences(paragraph)
    if not sentences:
        return []

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        sentence_parts = split_text_to_fit(sentence, max_chars)
        for part in sentence_parts:
            if not current:
                current = part
                continue
            if len(current) + 1 + len(part) <= max_chars:
                current = f"{current} {part}"
                continue
            chunks.append(current)
            current = part
    if current:
        chunks.append(current)
    return chunks


def parse_control_tag(paragraph: str) -> str | None:
    token = paragraph.strip().upper()
    if token == BREAK_TAG:
        return "break"
    if token == CHAPTER_TAG:
        return "chapter"
    return None


def build_batch_boundary_types(
    total_batches: int, chapter_batch_numbers: list[int] | None = None
) -> list[str]:
    count = max(1, int(total_batches))
    chapter_numbers: set[int] = set()
    for value in chapter_batch_numbers or []:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number >= 1:
            chapter_numbers.add(number)
    boundary_types: list[str] = ["none"]
    for batch_number in range(2, count + 1):
        boundary_types.append("chapter" if batch_number in chapter_numbers else "natural")
    return boundary_types


def build_batches(
    paragraphs: list[str],
    max_chars_per_batch: int,
    chapter_titles: list[str] | None = None,
) -> list[TextBatch]:
    if max_chars_per_batch < 100:
        raise ValueError("--max-chars-per-batch must be at least 100.")

    batches: list[TextBatch] = []
    current_text = ""
    start_idx = 0
    end_idx = 0
    current_starts_chapter = False
    current_forced_break_before = False
    current_chapter_title: str | None = None
    pending_chapter = False
    pending_forced_break = False
    pending_chapter_title: str | None = None
    chapter_title_index = 0

    def flush() -> None:
        nonlocal current_text, start_idx, end_idx, current_starts_chapter, current_forced_break_before, current_chapter_title
        if not current_text:
            return
        batches.append(
            TextBatch(
                index=len(batches) + 1,
                start_paragraph=start_idx,
                end_paragraph=end_idx,
                text=current_text,
                char_count=len(current_text),
                starts_chapter=current_starts_chapter,
                forced_break_before=current_forced_break_before,
                chapter_title=current_chapter_title,
            )
        )
        current_text = ""
        start_idx = 0
        end_idx = 0
        current_starts_chapter = False
        current_forced_break_before = False
        current_chapter_title = None

    for idx, paragraph in enumerate(paragraphs):
        control = parse_control_tag(paragraph)
        if control == "break":
            flush()
            pending_forced_break = True
            continue
        if control == "chapter":
            flush()
            pending_chapter = True
            pending_forced_break = True
            if chapter_titles and chapter_title_index < len(chapter_titles):
                candidate_title = str(chapter_titles[chapter_title_index]).strip()
                pending_chapter_title = candidate_title if candidate_title else None
            else:
                pending_chapter_title = None
            chapter_title_index += 1
            continue

        paragraph_chunks = split_paragraph_for_batches(paragraph, max_chars_per_batch)
        for chunk_index, chunk in enumerate(paragraph_chunks):
            separator = ""
            if current_text:
                separator = "\n\n" if chunk_index == 0 else " "
            add_len = len(separator) + len(chunk)
            if current_text and len(current_text) + add_len > max_chars_per_batch:
                flush()
                separator = ""

            if not current_text:
                start_idx = idx
                end_idx = idx
                current_text = chunk
                current_starts_chapter = pending_chapter
                current_forced_break_before = pending_forced_break
                current_chapter_title = pending_chapter_title
                pending_chapter = False
                pending_forced_break = False
                pending_chapter_title = None
                continue

            current_text += separator + chunk
            end_idx = idx

    flush()
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
    MAX_BLOCKS_PER_ROW = 120
    MIN_BLOCKS_PER_ROW = 32
    DONE_FLASH_SECONDS = 1.4
    RENDER_INTERVAL_SECONDS = 0.14
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
    COLOR_LIGHT_GRAY = "\x1b[37m"
    COLOR_BLACK = "\x1b[30m"
    GLYPH_PENDING = "\u25ae"
    GLYPH_WORKING_A = "\u25ae"
    GLYPH_WORKING_B = "\u25af"
    GLYPH_DONE = "\u25ae"
    GLYPH_COMPLETE = "\u25ae"
    GLYPH_BREAK = "\u25ae"
    GLYPH_CHAPTER = "\u25ae"
    GLYPH_EMPTY = "\u25af"

    def __init__(
        self,
        total_batches: int,
        total_chars: int,
        completed_batches: int,
        completed_chars: int,
        enabled: bool,
        batch_char_counts: list[int] | None = None,
        batch_boundary_types: list[str] | None = None,
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
        self._alt_buffer_active = False
        self._last_lines: list[str] = []

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
        if batch_boundary_types and len(batch_boundary_types) == self.total_batches:
            normalized_boundary_types = [
                value if value in {"none", "natural", "chapter"} else "natural"
                for value in batch_boundary_types
            ]
            normalized_boundary_types[0] = "none"
            self._batch_boundary_types = normalized_boundary_types
        else:
            self._batch_boundary_types = build_batch_boundary_types(self.total_batches)

    def start(self) -> None:
        if not self.enabled:
            return
        # Use alternate screen buffer to avoid visible flicker in normal terminal history.
        sys.stdout.write("\x1b[?1049h\x1b[H\x1b[?25l")
        sys.stdout.flush()
        self._alt_buffer_active = True
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
        if self._alt_buffer_active:
            sys.stdout.write("\x1b[?1049l")
            self._alt_buffer_active = False
        sys.stdout.flush()

    def _render_loop(self) -> None:
        while not self._stop_event.is_set():
            self._tick += 1
            self._draw_frame(self._build_frame())
            if self._stop_event.wait(self.RENDER_INTERVAL_SECONDS):
                break
        self._draw_frame(self._build_frame())

    def _draw_frame(self, frame: str) -> None:
        lines = frame.splitlines()
        if lines == self._last_lines:
            return
        max_lines = max(len(lines), len(self._last_lines))
        for row_index in range(max_lines):
            new_line = lines[row_index] if row_index < len(lines) else ""
            old_line = self._last_lines[row_index] if row_index < len(self._last_lines) else None
            if new_line == old_line:
                continue
            sys.stdout.write(f"\x1b[{row_index + 1};1H")
            sys.stdout.write(new_line)
            sys.stdout.write("\x1b[K")
        self._last_lines = lines
        sys.stdout.flush()

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
        bar = (
            f"{self.COLOR_GREEN}{self.GLYPH_DONE * fill}{self.COLOR_RESET}"
            f"{self.COLOR_DIM}{self.GLYPH_EMPTY * (bar_width - fill)}{self.COLOR_RESET}"
        )
        percent = ratio * 100.0

        elapsed = time.time() - self._start_time
        eta = avg_batch_seconds * max(0, self.total_batches - completed_batches)
        now_ts = time.time()
        spinner = self.SPINNER_FRAMES[self._tick % len(self.SPINNER_FRAMES)]
        terminal_columns = shutil.get_terminal_size(fallback=(120, 40)).columns
        blocks_per_row = max(
            self.MIN_BLOCKS_PER_ROW,
            min(self.MAX_BLOCKS_PER_ROW, max(8, terminal_columns - 8)),
        )

        total_blocks = sum(self._batch_block_counts)
        completed_blocks = sum(self._batch_block_counts[:completed_batches])
        throughput = completed_chars / elapsed if elapsed > 0 else 0.0
        beat = self._tick % 4

        lines: list[str] = []
        line_parts: list[str] = []
        visible_len = 0
        for batch_number, block_count in enumerate(self._batch_block_counts, start=1):
            active_batch = active_start <= batch_number <= active_end
            if active_batch:
                color = self.COLOR_BLUE if beat % 2 == 0 else self.COLOR_MAGENTA
                glyph = (
                    self.GLYPH_WORKING_A if beat % 2 == 0 else self.GLYPH_WORKING_B
                )
            elif batch_number <= completed_batches:
                if done_flash_until.get(batch_number, 0.0) > now_ts:
                    color = self.COLOR_GREEN
                    glyph = self.GLYPH_DONE
                else:
                    color = self.COLOR_WHITE
                    glyph = self.GLYPH_COMPLETE
            else:
                color = self.COLOR_RED
                glyph = self.GLYPH_PENDING

            token = f"{color}{glyph * block_count}{self.COLOR_RESET}"
            token_visible = block_count
            if visible_len + token_visible > blocks_per_row and line_parts:
                lines.append("".join(line_parts))
                line_parts = []
                visible_len = 0
            line_parts.append(token)
            visible_len += token_visible

            if batch_number < self.total_batches:
                boundary_kind = self._batch_boundary_types[batch_number]
                if boundary_kind == "chapter":
                    marker_token = (
                        f"{self.COLOR_LIGHT_GRAY}{self.GLYPH_CHAPTER}{self.COLOR_RESET}"
                    )
                else:
                    marker_token = f"{self.COLOR_BLACK}{self.GLYPH_BREAK}{self.COLOR_RESET}"
                if visible_len + 1 > blocks_per_row and line_parts:
                    lines.append("".join(line_parts))
                    line_parts = []
                    visible_len = 0
                line_parts.append(marker_token)
                visible_len += 1
        if line_parts:
            lines.append("".join(line_parts))
        if not lines:
            lines.append("(no blocks)")

        border = "=" * min(64, max(40, terminal_columns - 2))

        header = [
            f"{self.COLOR_CYAN}{border}{self.COLOR_RESET}",
            f"{self.COLOR_MAGENTA}{spinner}{self.COLOR_RESET} TTS Defrag Animator",
            f"[{bar}] {percent:6.2f}%  batches {completed_batches}/{self.total_batches}  chars {completed_chars}/{self.total_chars}",
            f"elapsed {format_duration(elapsed)}  eta {format_duration(eta)}  throughput {throughput:7.1f} chars/s",
            f"current {current_batch_label}",
            f"blocks: {completed_blocks}/{total_blocks}  (1 block = {self.CHARS_PER_BLOCK} chars)  grid={blocks_per_row}/row",
            f"status: {self.COLOR_YELLOW}{status}{self.COLOR_RESET}",
            f"legend: {self.COLOR_RED}{self.GLYPH_PENDING}{self.COLOR_RESET}=pending  "
            f"{self.COLOR_BLUE}{self.GLYPH_WORKING_A}{self.COLOR_RESET}=working  "
            f"{self.COLOR_GREEN}{self.GLYPH_DONE}{self.COLOR_RESET}=done-now  "
            f"{self.COLOR_WHITE}{self.GLYPH_COMPLETE}{self.COLOR_RESET}=completed  "
            f"{self.COLOR_LIGHT_GRAY}{self.GLYPH_CHAPTER}{self.COLOR_RESET}=chapter  "
            f"{self.COLOR_BLACK}{self.GLYPH_BREAK}{self.COLOR_RESET}=natural-break",
            "stop: requested (will exit after current batch)"
            if stop_requested
            else "stop: running (Ctrl+C once stop-after-batch, twice try abort-current)",
            f"{self.COLOR_CYAN}{border}{self.COLOR_RESET}",
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
    remaining_segments: list[str] = []
    for batch in remaining_batches:
        if batch.starts_chapter:
            chapter_title = sanitize_chapter_title(batch.chapter_title)
            if chapter_title:
                remaining_segments.append(f"{CHAPTER_TAG} {chapter_title}")
            else:
                remaining_segments.append(CHAPTER_TAG)
        elif batch.forced_break_before:
            remaining_segments.append(BREAK_TAG)
        remaining_segments.append(batch.text)
    remaining_text = "\n\n".join(remaining_segments).strip() + "\n"
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
        "--chapter-pause-ms",
        str(runtime_options["chapter_pause_ms"]),
        "--mp3-quality",
        str(runtime_options["mp3_quality"]),
        "--inference-batch-size",
        str(runtime_options["inference_batch_size"]),
        "--max-inference-chars",
        str(runtime_options["max_inference_chars"]),
        "--max-new-tokens",
        str(runtime_options["max_new_tokens"]),
        "--language",
        str(runtime_options["language"]),
        "--tts-backend",
        str(runtime_options["tts_backend"]),
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
    if runtime_options.get("use_chapters"):
        command_args.append("--use-chapters")
    if runtime_options.get("continuation_chain"):
        command_args.append("--continuation-chain")
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

def require_runtime_dependencies(backend: str) -> tuple[Any, Any, Any, Any, Any, Any]:
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

    Qwen3TTSModel: Any = None
    AutoModel: Any = None
    AutoProcessor: Any = None
    if backend == "qwen":
        try:
            from qwen_tts import Qwen3TTSModel  # type: ignore
        except ImportError:
            missing.append("qwen-tts")
    elif is_moss_backend(backend):
        try:
            from transformers import AutoModel, AutoProcessor  # type: ignore
        except ImportError:
            missing.append("transformers")

    if missing:
        install_hint = (
            "Run scripts/install_linux_nvidia.sh or pip install -r requirements.txt."
        )
        if backend == "qwen":
            install_hint += (
                " For qwen backend, also install qwen dependencies with "
                "`pip install -r requirements-qwen.txt` "
                "(prefer a separate virtual environment from MOSS)."
            )
        raise RuntimeError(
            "Missing runtime dependencies: "
            + ", ".join(missing)
            + ". "
            + install_hint
        )
    return np, sf, torch, Qwen3TTSModel, AutoModel, AutoProcessor


def check_sox_installation() -> str | None:
    if shutil.which("sox"):
        return None
    return (
        "SoX is not installed (`sox` not found in PATH). "
        "Qwen-TTS audio preprocessing may fail. "
        "Install with: apt-get update && apt-get install -y sox"
    )


def probe_flash_attn_runtime() -> tuple[bool, str | None]:
    has_pkg = importlib.util.find_spec("flash_attn") is not None
    has_ext = importlib.util.find_spec("flash_attn_2_cuda") is not None
    if not has_pkg and not has_ext:
        return False, (
            "flash-attn is not installed; falling back from flash_attention_2 to sdpa."
        )

    probe_modules = (
        "flash_attn_2_cuda",
        "flash_attn.flash_attn_interface",
        "flash_attn",
    )
    import_errors: list[str] = []
    for module_name in probe_modules:
        if importlib.util.find_spec(module_name) is None:
            continue
        try:
            importlib.import_module(module_name)
            return True, None
        except Exception as exc:
            detail = f"{exc.__class__.__name__}: {exc}"
            import_errors.append(f"{module_name} ({detail})")
            if "undefined symbol" in str(exc):
                return False, (
                    "flash-attn is installed but incompatible with this torch build "
                    f"({detail}). Falling back to sdpa. "
                    "Fix by reinstalling a matching flash-attn build, or uninstall it with "
                    "`python -m pip uninstall -y flash-attn flash_attn`."
                )

    if import_errors:
        return False, (
            "flash-attn is installed but failed runtime import "
            f"({import_errors[0]}). Falling back to sdpa."
        )
    return False, (
        "flash-attn package was detected but runtime modules were not importable; "
        "falling back to sdpa."
    )


def choose_attention_implementation(requested: str) -> tuple[str, str | None]:
    if requested != "flash_attention_2":
        return requested, None
    flash_ok, warning = probe_flash_attn_runtime()
    if flash_ok:
        return requested, None
    return ("sdpa", warning)


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
    chapter_batch_numbers: list[int] | None = None,
    chapter_pause_ms: int = 0,
) -> tuple[int, float, list[int]]:
    if not part_paths:
        raise RuntimeError("No audio parts exist to combine.")

    first_audio, sample_rate = sf.read(str(part_paths[0]), always_2d=True)
    channels = first_audio.shape[1]
    pause_samples = int(sample_rate * (max(0, pause_ms) / 1000.0))
    chapter_pause_samples = int(sample_rate * (max(0, chapter_pause_ms) / 1000.0))
    chapter_starts: set[int] = set()
    for value in chapter_batch_numbers or []:
        try:
            batch_number = int(value)
        except (TypeError, ValueError):
            continue
        if batch_number >= 1:
            chapter_starts.add(batch_number)

    output_wav_path.parent.mkdir(parents=True, exist_ok=True)
    total_samples = 0
    part_start_samples: list[int] = []
    with sf.SoundFile(
        str(output_wav_path),
        mode="w",
        samplerate=sample_rate,
        channels=channels,
        subtype="PCM_16",
    ) as writer:
        part_start_samples.append(total_samples)
        writer.write(first_audio.astype(np.float32))
        total_samples += first_audio.shape[0]
        for batch_number, part_path in enumerate(part_paths[1:], start=2):
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
            gap_samples = compute_inter_batch_pause_samples(
                base_pause_samples=pause_samples,
                chapter_pause_samples=chapter_pause_samples,
                next_batch_number=batch_number,
                chapter_batch_numbers=chapter_starts,
            )
            if gap_samples > 0:
                silence = np.zeros((gap_samples, channels), dtype=np.float32)
                writer.write(silence)
                total_samples += gap_samples
            part_start_samples.append(total_samples)
            writer.write(audio.astype(np.float32))
            total_samples += audio.shape[0]
    duration_seconds = total_samples / float(sample_rate)
    return int(sample_rate), float(duration_seconds), part_start_samples


def compute_inter_batch_pause_samples(
    base_pause_samples: int,
    chapter_pause_samples: int,
    next_batch_number: int,
    chapter_batch_numbers: set[int],
) -> int:
    gap = max(0, int(base_pause_samples))
    if int(next_batch_number) in chapter_batch_numbers:
        gap += max(0, int(chapter_pause_samples))
    return gap


def chapter_time_entries_from_batches(
    chapter_batch_numbers: list[int],
    part_start_samples: list[int],
    sample_rate: int,
) -> list[tuple[float, int]]:
    if sample_rate <= 0:
        return []
    chapter_entries: list[tuple[float, int]] = []
    seen_starts: set[int] = set()
    for batch_number in chapter_batch_numbers:
        index = batch_number - 1
        if index < 0 or index >= len(part_start_samples):
            continue
        start_sample = part_start_samples[index]
        if start_sample in seen_starts:
            continue
        seen_starts.add(start_sample)
        chapter_entries.append((start_sample / float(sample_rate), batch_number))
    return sorted(chapter_entries, key=lambda item: item[0])


def chapter_start_times_from_batches(
    chapter_batch_numbers: list[int],
    part_start_samples: list[int],
    sample_rate: int,
) -> list[float]:
    return [
        start_seconds
        for start_seconds, _batch_number in chapter_time_entries_from_batches(
            chapter_batch_numbers=chapter_batch_numbers,
            part_start_samples=part_start_samples,
            sample_rate=sample_rate,
        )
    ]


def sanitize_chapter_title(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = re.sub(r"[ \t]+", " ", str(value).strip())
    return normalized if normalized else None


def escape_ffmetadata_value(value: str) -> str:
    escaped = value.replace("\\", "\\\\")
    escaped = escaped.replace("\n", " ").replace("\r", " ")
    escaped = escaped.replace(";", r"\;").replace("#", r"\#").replace("=", r"\=")
    return escaped


def build_ffmetadata_with_chapters(
    chapter_entries: list[tuple[float, str | None]],
    total_duration_seconds: float,
) -> str:
    total_ms = max(0, int(round(total_duration_seconds * 1000.0)))
    normalized_entries: list[tuple[int, str | None]] = []
    for start_seconds, title in chapter_entries:
        if start_seconds < 0.0:
            continue
        start_ms = max(0, int(round(start_seconds * 1000.0)))
        if start_ms >= total_ms:
            continue
        normalized_entries.append((start_ms, sanitize_chapter_title(title)))

    normalized_entries.sort(key=lambda item: item[0])
    deduped_entries: list[tuple[int, str | None]] = []
    seen_starts_ms: set[int] = set()
    for start_ms, title in normalized_entries:
        if start_ms in seen_starts_ms:
            continue
        seen_starts_ms.add(start_ms)
        deduped_entries.append((start_ms, title))

    lines = [";FFMETADATA1"]
    if not deduped_entries:
        return "\n".join(lines) + "\n"

    for index, (start_ms, custom_title) in enumerate(deduped_entries, start=1):
        next_start_ms = (
            deduped_entries[index][0]
            if index < len(deduped_entries)
            else max(start_ms + 1, total_ms)
        )
        end_ms = max(start_ms + 1, next_start_ms)
        title = custom_title if custom_title else f"Chapter {index}"
        lines.extend(
            [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={start_ms}",
                f"END={end_ms}",
                f"title={escape_ffmetadata_value(title)}",
            ]
        )
    return "\n".join(lines) + "\n"


def convert_wav_to_mp3(
    input_wav: Path,
    output_mp3: Path,
    quality_level: int,
    chapter_metadata_path: Path | None = None,
) -> None:
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
    ]
    if chapter_metadata_path:
        cmd.extend(
            [
                "-f",
                "ffmetadata",
                "-i",
                str(chapter_metadata_path),
                "-map_metadata",
                "1",
                "-map_chapters",
                "1",
                "-id3v2_version",
                "3",
            ]
        )
    cmd.extend(
        [
            "-map",
            "0:a",
            "-vn",
            "-codec:a",
            "libmp3lame",
            "-q:a",
            str(quality_level),
            str(output_mp3),
        ]
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            "High-quality MP3 conversion failed via ffmpeg."
            + (f" Detail: {detail}" if detail else "")
        )


def convert_audio_to_wav(
    input_audio: Path,
    output_wav: Path,
) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError(
            "ffmpeg is required to convert compressed reference audio to WAV for MOSS backend. "
            "Install ffmpeg or provide a WAV reference audio file."
        )
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_audio),
        "-map",
        "0:a",
        "-vn",
        "-ac",
        "1",
        "-c:a",
        "pcm_s16le",
        str(output_wav),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(
            "Reference audio conversion to WAV failed via ffmpeg."
            + (f" Detail: {detail}" if detail else "")
        )


def prepare_reference_audio_for_moss(
    reference_audio: str,
    run_dir: Path,
) -> tuple[str, str | None]:
    try:
        source = Path(reference_audio).expanduser()
    except Exception:
        return reference_audio, None

    if not source.exists() or not source.is_file():
        return reference_audio, None

    source_resolved = source.resolve()
    if source_resolved.suffix.lower() == ".wav":
        return str(source_resolved), None

    prepared_dir = run_dir / "prepared_audio"
    target_wav = prepared_dir / f"{source_resolved.stem}_moss_ref.wav"
    if not target_wav.exists() or target_wav.stat().st_mtime < source_resolved.stat().st_mtime:
        convert_audio_to_wav(source_resolved, target_wav)

    return (
        str(target_wav.resolve()),
        (
            f"Converted reference audio to WAV for MOSS compatibility: "
            f"{source_resolved.name} -> {target_wav.name}"
        ),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create audiobooks from text using MOSS-TTS or Qwen3-TTS voice cloning."
    )
    parser.add_argument("--text-file", type=Path, help="Input text file.")
    parser.add_argument(
        "--reference-audio",
        type=str,
        help=(
            "Reference audio for voice clone (local path, URL, base64, etc.). "
            "If omitted in interactive mode, the app can prompt from files in the text-file folder."
        ),
    )
    ref_group = parser.add_mutually_exclusive_group(required=False)
    ref_group.add_argument(
        "--reference-text", type=str, help="Transcript for the reference audio."
    )
    ref_group.add_argument(
        "--reference-text-file",
        type=Path,
        help=(
            "Path to transcript text file. If omitted, the app will try "
            "to use <reference-audio-basename>.txt from the same folder."
        ),
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
        "--chapter-pause-ms",
        type=int,
        default=DEFAULT_CHAPTER_PAUSE_MS,
        help=(
            "Additional pause inserted before chapter-start batches in milliseconds "
            f"(default: {DEFAULT_CHAPTER_PAUSE_MS})."
        ),
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
        default=0,
        help=(
            "Number of text batches to synthesize per model.generate call. "
            "Use 0 for auto-tuned size (default). "
            "Qwen backend always uses 1."
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
        help=(
            "Model id/path. "
            f"Default ({DEFAULT_TTS_BACKEND}): {DEFAULT_MODEL_ID}."
        ),
    )
    parser.add_argument(
        "--tts-backend",
        type=str,
        default=DEFAULT_TTS_BACKEND,
        help=(
            "Backend to use: moss-delay, moss-local, qwen, or auto "
            f"(default: {DEFAULT_TTS_BACKEND})."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=DEFAULT_MAX_NEW_TOKENS,
        help=(
            "Maximum generated audio tokens per inference call for MOSS backends "
            f"(default: {DEFAULT_MAX_NEW_TOKENS})."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help='Device target (default: "cuda:0").',
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
            "Attention implementation for model loading "
            f"(default: {DEFAULT_ATTN_IMPLEMENTATION})."
        ),
    )
    parser.add_argument(
        "--no-defrag-ui",
        action="store_true",
        help="Disable defrag-style UI and print detailed text status/progress logs.",
    )
    parser.add_argument(
        "--use-chapters",
        action="store_true",
        help=(
            f"Use {CHAPTER_TAG} markers as MP3 chapters and embed chapter metadata "
            "into the final .mp3 output."
        ),
    )
    parser.add_argument(
        "--stop-after-batch",
        type=int,
        default=0,
        help="Testing helper: stop after N batches in this invocation.",
    )
    parser.add_argument(
        "--continuation-chain",
        action="store_true",
        help=(
            "MOSS-only high-continuity mode: generate batches sequentially via continuation "
            "using previous generated audio+text as prefix context."
        ),
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
    if args.chapter_pause_ms < 0:
        print("ERROR: --chapter-pause-ms cannot be negative.", file=sys.stderr)
        return 2
    if args.mp3_quality < 0 or args.mp3_quality > 9:
        print("ERROR: --mp3-quality must be between 0 and 9.", file=sys.stderr)
        return 2
    if args.inference_batch_size < 0:
        print(
            "ERROR: --inference-batch-size must be >= 0 (0 means auto).",
            file=sys.stderr,
        )
        return 2
    if args.max_new_tokens < 1:
        print("ERROR: --max-new-tokens must be at least 1.", file=sys.stderr)
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
        scan_dir = text_file.parent.resolve()
        candidates = scan_reference_audio_candidates(scan_dir)
        if not candidates:
            print(
                "ERROR: --reference-audio is required. "
                "No audio files with matching .txt transcripts were found in "
                f"{scan_dir}.",
                file=sys.stderr,
            )
            return 2
        if not sys.stdin.isatty():
            print(
                "ERROR: --reference-audio is required in non-interactive mode. "
                f"Found {len(candidates)} candidate(s) in {scan_dir}; "
                "pass --reference-audio explicitly.",
                file=sys.stderr,
            )
            return 2
        selected_audio = prompt_for_reference_audio_selection(
            candidates=candidates,
            scan_dir=scan_dir,
        )
        if not selected_audio:
            print("ERROR: No reference audio selected.", file=sys.stderr)
            return 2
        reference_audio = str(selected_audio)
        print(f"Selected reference audio: {reference_audio}")

    auto_reference_text_file = find_matching_reference_text_file(reference_audio)
    auto_reference_text_path_hint = ""
    if not auto_reference_text_file:
        try:
            auto_reference_text_path_hint = str(
                Path(reference_audio).expanduser().with_suffix(".txt")
            )
        except Exception:
            auto_reference_text_path_hint = ""

    if args.reference_text_file:
        reference_text = read_text_file(args.reference_text_file.resolve()).strip()
    elif args.reference_text:
        reference_text = args.reference_text.strip()
    elif state.get("reference_text"):
        reference_text = str(state["reference_text"]).strip()
    elif state.get("reference_text_file"):
        reference_text = read_text_file(Path(state["reference_text_file"]).resolve()).strip()
    elif auto_reference_text_file:
        reference_text = read_text_file(auto_reference_text_file).strip()
    else:
        reference_text = ""

    requested_backend = (
        str(state.get("tts_backend"))
        if is_resume
        and args.tts_backend == DEFAULT_TTS_BACKEND
        and state.get("tts_backend")
        else args.tts_backend
    )
    if is_resume and args.tts_backend == DEFAULT_TTS_BACKEND and not state.get("tts_backend"):
        # Backward compatibility with sessions created before backend tracking existed.
        requested_backend = "auto"
    model_id = (
        str(state.get("model_id"))
        if is_resume and args.model_id == DEFAULT_MODEL_ID and state.get("model_id")
        else args.model_id
    )
    try:
        tts_backend = resolve_tts_backend(requested_backend, model_id)
    except ValueError as backend_exc:
        print(f"ERROR: {backend_exc}", file=sys.stderr)
        return 2

    if (
        args.model_id == DEFAULT_MODEL_ID
        and not (is_resume and state.get("model_id"))
        and normalize_tts_backend(args.tts_backend) in {"qwen", "moss-delay", "moss-local"}
    ):
        model_id = recommended_default_model_id(tts_backend)

    x_vector_only_mode = bool(args.x_vector_only_mode or state.get("x_vector_only_mode"))
    if tts_backend == "qwen" and not x_vector_only_mode and not reference_text:
        auto_hint = (
            f" (also looked for auto transcript: {auto_reference_text_path_hint})"
            if auto_reference_text_path_hint
            else ""
        )
        print(
            "ERROR: reference transcript is required for qwen backend "
            "(or set --x-vector-only-mode)."
            + auto_hint,
            file=sys.stderr,
        )
        return 2

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
    chapter_pause_ms = (
        int(state["chapter_pause_ms"])
        if is_resume
        and args.chapter_pause_ms == DEFAULT_CHAPTER_PAUSE_MS
        and state.get("chapter_pause_ms") is not None
        else args.chapter_pause_ms
    )
    mp3_quality = (
        int(state["mp3_quality"])
        if is_resume and args.mp3_quality == 0 and state.get("mp3_quality") is not None
        else args.mp3_quality
    )
    use_chapters = bool(args.use_chapters or state.get("use_chapters"))
    continuation_chain = bool(
        args.continuation_chain or state.get("continuation_chain")
    )
    if continuation_chain and not is_moss_backend(tts_backend):
        print(
            "ERROR: --continuation-chain is supported only with moss-delay or moss-local backends.",
            file=sys.stderr,
        )
        return 2
    if is_moss_backend(tts_backend):
        try:
            prepared_reference_audio, conversion_note = prepare_reference_audio_for_moss(
                reference_audio=reference_audio,
                run_dir=run_dir,
            )
        except RuntimeError as prep_exc:
            print(f"ERROR: {prep_exc}", file=sys.stderr)
            return 2
        if conversion_note:
            print(f"WARNING: {conversion_note}", file=sys.stderr)
        reference_audio = prepared_reference_audio
    inference_batch_size = (
        int(state["inference_batch_size"])
        if is_resume
        and args.inference_batch_size == 0
        and state.get("inference_batch_size") is not None
        else args.inference_batch_size
    )
    max_new_tokens = (
        int(state["max_new_tokens"])
        if is_resume
        and args.max_new_tokens == DEFAULT_MAX_NEW_TOKENS
        and state.get("max_new_tokens")
        else args.max_new_tokens
    )
    max_inference_chars = (
        int(state["max_inference_chars"])
        if is_resume
        and args.max_inference_chars == DEFAULT_MAX_INFERENCE_CHARS
        and state.get("max_inference_chars")
        else args.max_inference_chars
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
    if use_chapters and output_suffix != ".mp3":
        print(
            "WARNING: --use-chapters was requested, but chapter metadata is only embedded for .mp3 output.",
            file=sys.stderr,
        )

    reference_text_file: Path | None = None
    if reference_text:
        reference_text_file = run_dir / "reference_text.txt"
        write_text_file(reference_text_file, reference_text + "\n")
    elif state.get("reference_text_file"):
        candidate = Path(state["reference_text_file"]).resolve()
        if candidate.exists():
            reference_text_file = candidate

    source_text = read_text_file(text_file)
    chapter_titles_from_text = extract_chapter_titles_from_raw_text(source_text)
    paragraphs = split_into_paragraphs(source_text)
    if not paragraphs:
        print(f"ERROR: no usable paragraphs found in {text_file}.", file=sys.stderr)
        return 2
    batches = build_batches(
        paragraphs,
        max_chars,
        chapter_titles=chapter_titles_from_text,
    )

    part_files: list[str] = list(state.get("part_files", []))
    for existing_part in resolve_paths(run_dir, part_files):
        if not existing_part.exists():
            print(f"ERROR: missing part file from state: {existing_part}", file=sys.stderr)
            return 2

    existing_batches = len(part_files)
    raw_saved_chapter_batches = state.get("chapter_batch_numbers", [])
    state_chapter_batches: list[int] = []
    seen_saved_chapters: set[int] = set()
    if isinstance(raw_saved_chapter_batches, list):
        for value in raw_saved_chapter_batches:
            try:
                batch_number = int(value)
            except (TypeError, ValueError):
                continue
            if batch_number < 1 or batch_number > existing_batches:
                continue
            if batch_number in seen_saved_chapters:
                continue
            state_chapter_batches.append(batch_number)
            seen_saved_chapters.add(batch_number)
    raw_saved_chapter_titles = state.get("chapter_titles_by_batch", {})
    state_chapter_titles: dict[int, str] = {}
    if isinstance(raw_saved_chapter_titles, dict):
        for key, value in raw_saved_chapter_titles.items():
            try:
                batch_number = int(key)
            except (TypeError, ValueError):
                continue
            if batch_number < 1 or batch_number > existing_batches:
                continue
            if batch_number not in seen_saved_chapters:
                continue
            title = sanitize_chapter_title(str(value))
            if title:
                state_chapter_titles[batch_number] = title

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
    new_chapter_batches_for_view = [
        existing_batches + index
        for index, batch in enumerate(batches, start=1)
        if batch.starts_chapter
    ]
    all_chapter_batches_for_view = sorted(
        set(state_chapter_batches + new_chapter_batches_for_view)
    )
    batch_boundary_types_for_view = build_batch_boundary_types(
        total_batches=total_batches,
        chapter_batch_numbers=all_chapter_batches_for_view,
    )

    continuation_audio_source = str(reference_audio)
    continuation_prefix_text = reference_text.strip()
    if continuation_chain:
        if existing_batches > 0 and part_files:
            last_generated_text = str(state.get("last_generated_batch_text") or "").strip()
            last_part_path = resolve_paths(run_dir, [part_files[-1]])[0]
            if last_part_path.exists() and last_generated_text:
                continuation_audio_source = str(last_part_path)
                continuation_prefix_text = last_generated_text
            elif not continuation_prefix_text:
                print(
                    "ERROR: --continuation-chain resume requires either "
                    "`last_generated_batch_text` in session state or a reference transcript.",
                    file=sys.stderr,
                )
                return 2
            else:
                print(
                    "WARNING: Could not recover last generated continuation anchor from state; "
                    "falling back to original reference audio/text anchor.",
                    file=sys.stderr,
                )
        elif not continuation_prefix_text:
            print(
                "ERROR: --continuation-chain requires a reference transcript "
                "(`--reference-text` or `--reference-text-file`) for the initial anchor.",
                file=sys.stderr,
            )
            return 2

    progress = DefragProgressView(
        total_batches=total_batches,
        total_chars=total_chars,
        completed_batches=existing_batches,
        completed_chars=existing_chars,
        enabled=not args.no_defrag_ui,
        batch_char_counts=all_batch_chars_for_view,
        batch_boundary_types=batch_boundary_types_for_view,
    )
    progress.start()
    run_started_at = time.time()
    plain_last_status = ""
    plain_batch_durations: list[float] = []

    def emit_status(message: str, warning: bool = False) -> None:
        nonlocal plain_last_status
        status_text = f"WARNING: {message}" if warning else message
        progress.set_status(status_text)
        if args.no_defrag_ui and status_text != plain_last_status:
            print(f"status: {status_text}")
            plain_last_status = status_text

    def emit_progress(context: str | None = None) -> None:
        if not args.no_defrag_ui:
            return
        completed_batches_total = existing_batches + completed_this_run
        average_batch_seconds = (
            sum(plain_batch_durations) / len(plain_batch_durations)
            if plain_batch_durations
            else 0.0
        )
        line = format_plain_progress(
            completed_batches=completed_batches_total,
            total_batches=total_batches,
            completed_chars=completed_chars_total,
            total_chars=total_chars,
            elapsed_seconds=time.time() - run_started_at,
            average_batch_seconds=average_batch_seconds,
        )
        if context:
            line += f" | {context}"
        print(line)

    try:
        np, sf, torch, Qwen3TTSModel, AutoModel, AutoProcessor = require_runtime_dependencies(
            tts_backend
        )
        completed_this_run = 0
        completed_chars_total = existing_chars
        if args.no_defrag_ui:
            print("status: Detailed text progress mode is active (--no-defrag-ui).")
            emit_progress("starting run")

        if tts_backend == "qwen":
            sox_warning = check_sox_installation()
            if sox_warning:
                emit_status(sox_warning, warning=True)

        arch_fatal, arch_message = check_cuda_arch_compatibility(torch, device)
        if arch_message:
            emit_status(arch_message, warning=True)
        if arch_fatal and arch_message:
            raise RuntimeError(arch_message)

        dtype_name, dtype_warning = choose_dtype(torch, dtype_name, device)
        if dtype_warning:
            emit_status(dtype_warning, warning=True)

        attn, attn_warning = choose_attention_implementation(attn)
        if attn_warning:
            emit_status(attn_warning, warning=True)
        if is_moss_backend(tts_backend) and device.lower().startswith("cpu") and attn == "sdpa":
            attn = "eager"
            emit_status(
                "CPU mode with MOSS backend prefers eager attention; falling back to eager.",
                warning=True,
            )

        inference_batch_size, batch_size_warning = choose_inference_batch_size(
            requested=inference_batch_size,
            backend=tts_backend,
            torch=torch,
            device=device,
            max_chars_per_batch=max_chars,
        )
        if batch_size_warning:
            emit_status(batch_size_warning, warning=True)
        try:
            (
                inference_batch_size,
                continuation_batch_warning,
            ) = apply_continuation_chain_constraints(
                backend=tts_backend,
                continuation_chain=continuation_chain,
                inference_batch_size=inference_batch_size,
            )
        except ValueError as continuation_exc:
            raise RuntimeError(str(continuation_exc)) from continuation_exc
        if continuation_batch_warning:
            emit_status(continuation_batch_warning, warning=True)

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        model: Any
        processor: Any | None = None
        clone_prompt: Any | None = None
        torch_device = torch.device("cpu")
        sample_rate_hint: int | None = None
        if is_moss_backend(tts_backend):
            if device.lower().startswith("cuda") and torch.cuda.is_available():
                torch_device = torch.device(device)
            elif device.lower().startswith("cuda"):
                emit_status(
                    "CUDA device was requested but is unavailable; running MOSS on CPU.",
                    warning=True,
                )
                torch_device = torch.device("cpu")
            else:
                torch_device = torch.device(device)

            assert AutoModel is not None and AutoProcessor is not None
            processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
            )
            if hasattr(processor, "audio_tokenizer"):
                try:
                    processor.audio_tokenizer = processor.audio_tokenizer.to(torch_device)
                except Exception as tok_exc:
                    emit_status(
                        f"Audio tokenizer device placement failed ({tok_exc}); continuing with default placement.",
                        warning=True,
                    )

            model_kwargs: dict[str, Any] = {
                "trust_remote_code": True,
                "torch_dtype": dtype_map[dtype_name],
            }
            if attn:
                model_kwargs["attn_implementation"] = attn
            try:
                model = AutoModel.from_pretrained(model_id, **model_kwargs).to(torch_device)
            except Exception as model_exc:
                if attn not in {"sdpa", "eager"}:
                    fallback_attn = "sdpa" if torch_device.type == "cuda" else "eager"
                    fallback_msg = (
                        f"Model load failed with attn '{attn}', retrying with '{fallback_attn}'. "
                        f"Reason: {model_exc}"
                    )
                    emit_status(fallback_msg, warning=True)
                    attn = fallback_attn
                    model_kwargs["attn_implementation"] = attn
                    model = AutoModel.from_pretrained(model_id, **model_kwargs).to(torch_device)
                else:
                    raise
            model.eval()
            if x_vector_only_mode:
                emit_status(
                    "--x-vector-only-mode is qwen-only and has no effect for MOSS backends.",
                    warning=True,
                )
            sample_rate_hint = int(
                getattr(getattr(processor, "model_config", None), "sampling_rate", 24000)
            )
        else:
            assert Qwen3TTSModel is not None
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
                    emit_status(fallback_msg, warning=True)
                    attn = "sdpa"
                    model = Qwen3TTSModel.from_pretrained(
                        model_id,
                        device_map=device,
                        dtype=dtype_map[dtype_name],
                        attn_implementation=attn,
                    )
                else:
                    raise
            emit_status("Building clone prompt...")
            clone_prompt = model.create_voice_clone_prompt(
                ref_audio=reference_audio,
                ref_text=reference_text if reference_text else None,
                x_vector_only_mode=x_vector_only_mode,
            )
        emit_status(
            f"Loaded backend={tts_backend}, model={model_id}, inference_batch_size={inference_batch_size}."
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
                "tts_backend": tts_backend,
                "output_audio": str(output_target),
                "output_wav": str(output_target),
                "model_id": model_id,
                "device": device,
                "dtype": dtype_name,
                "attn_implementation": attn,
                "language": language,
                "max_chars_per_batch": max_chars,
                "pause_ms": pause_ms,
                "chapter_pause_ms": chapter_pause_ms,
                "mp3_quality": mp3_quality,
                "use_chapters": use_chapters,
                "continuation_chain": continuation_chain,
                "inference_batch_size": inference_batch_size,
                "max_inference_chars": max_inference_chars,
                "max_new_tokens": max_new_tokens,
                "part_files": part_files,
                "batch_char_counts": state_batch_chars,
                "chapter_batch_numbers": state_chapter_batches,
                "chapter_titles_by_batch": {
                    str(batch_number): title
                    for batch_number, title in sorted(state_chapter_titles.items())
                },
                "completed_batches": existing_batches,
                "completed_characters": existing_chars,
                "stopped_early": False,
                "remaining_text_file": None,
                "continue_script_sh": None,
                "continue_script_ps1": None,
                "last_generated_batch_text": state.get("last_generated_batch_text"),
                "last_generated_batch_file": state.get("last_generated_batch_file"),
            }
        )
        save_state(state_path, state)

        runtime_options = {
            "reference_audio": reference_audio,
            "reference_text_file": reference_text_file,
            "tts_backend": tts_backend,
            "output_wav": output_target,
            "max_chars_per_batch": max_chars,
            "pause_ms": pause_ms,
            "chapter_pause_ms": chapter_pause_ms,
            "mp3_quality": mp3_quality,
            "use_chapters": use_chapters,
            "continuation_chain": continuation_chain,
            "inference_batch_size": inference_batch_size,
            "max_inference_chars": max_inference_chars,
            "max_new_tokens": max_new_tokens,
            "language": language,
            "model_id": model_id,
            "device": device,
            "dtype": dtype_name,
            "attn_implementation": attn,
            "x_vector_only_mode": x_vector_only_mode,
            "no_defrag_ui": args.no_defrag_ui,
        }
        script_path = Path(__file__).resolve()

        next_local_index = 0
        active_inference_batch_size = max(1, inference_batch_size)
        while next_local_index < len(batches):
            if stop_controller.force_stop:
                break

            remaining_count = len(batches) - next_local_index
            group_size = min(active_inference_batch_size, remaining_count)
            active_batches = batches[next_local_index : next_local_index + group_size]

            first_global_batch = existing_batches + completed_this_run + 1
            last_global_batch = first_global_batch + len(active_batches) - 1
            group_char_count = sum(batch.char_count for batch in active_batches)
            if first_global_batch == last_global_batch:
                batch_label = (
                    f"batch {first_global_batch}/{total_batches} ({group_char_count} chars)"
                )
            else:
                batch_label = (
                    f"batches {first_global_batch}-{last_global_batch}/{total_batches} "
                    f"({group_char_count} chars total)"
                )

            progress.set_active_batch(
                first_global_batch,
                group_char_count,
                f"Generating {batch_label}...",
                batch_label=batch_label,
                active_batch_count=len(active_batches),
            )
            if args.no_defrag_ui:
                emit_status(f"Generating {batch_label}...")
                emit_progress(f"started {batch_label}")

            def run_generation_group() -> tuple[list[Any], int]:
                if is_moss_backend(tts_backend):
                    assert processor is not None
                    if continuation_chain:
                        if len(active_batches) != 1:
                            raise RuntimeError(
                                "--continuation-chain requires one batch per inference call."
                            )
                        current_batch = active_batches[0]
                        continuation_text = (
                            f"{continuation_prefix_text}\n{current_batch.text}".strip()
                            if continuation_prefix_text
                            else current_batch.text
                        )
                        message_kwargs: dict[str, Any] = {
                            "text": continuation_text,
                            "reference": [reference_audio],
                        }
                        if language.lower() != "auto":
                            message_kwargs["language"] = language
                        conversations = [
                            [
                                processor.build_user_message(**message_kwargs),
                                processor.build_assistant_message(
                                    audio_codes_list=[continuation_audio_source]
                                ),
                            ]
                        ]
                        mode = "continuation"
                    else:
                        conversations = []
                        for current_batch in active_batches:
                            message_kwargs = {
                                "text": current_batch.text,
                                "reference": [reference_audio],
                            }
                            if language.lower() != "auto":
                                message_kwargs["language"] = language
                            conversations.append(
                                [processor.build_user_message(**message_kwargs)]
                            )
                        mode = "generation"

                    packed = processor(conversations, mode=mode)
                    input_ids = packed["input_ids"].to(torch_device)
                    attention_mask = packed["attention_mask"].to(torch_device)
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        audio_temperature=float(DEFAULT_MOSS_AUDIO_TEMPERATURE),
                        audio_top_p=float(DEFAULT_MOSS_AUDIO_TOP_P),
                        audio_top_k=int(DEFAULT_MOSS_AUDIO_TOP_K),
                        audio_repetition_penalty=float(
                            DEFAULT_MOSS_AUDIO_REPETITION_PENALTY
                        ),
                    )
                    messages = processor.decode(outputs)
                    expected_messages = 1 if continuation_chain else len(active_batches)
                    if not messages or len(messages) != expected_messages:
                        raise RuntimeError(
                            "MOSS decode output size mismatch with requested batch size."
                        )

                    generated_audio: list[Any] = []
                    for message in messages:
                        if message is None or not getattr(message, "audio_codes_list", None):
                            raise RuntimeError("MOSS decode returned an empty audio result.")
                        audio_obj = message.audio_codes_list[0]
                        if hasattr(audio_obj, "detach"):
                            audio_obj = audio_obj.detach().float().cpu().numpy()
                        else:
                            audio_obj = np.asarray(audio_obj, dtype=np.float32)
                        if getattr(audio_obj, "ndim", 1) > 1:
                            audio_obj = audio_obj.reshape(-1)
                        generated_audio.append(audio_obj)

                    output_sample_rate = int(
                        sample_rate_hint
                        if sample_rate_hint is not None
                        else getattr(getattr(processor, "model_config", None), "sampling_rate", 24000)
                    )
                    return generated_audio, output_sample_rate

                assert clone_prompt is not None
                if len(active_batches) != 1:
                    raise RuntimeError("qwen backend only supports one batch per inference call.")
                kwargs: dict[str, Any] = {
                    "text": active_batches[0].text,
                    "voice_clone_prompt": clone_prompt,
                }
                if language.lower() != "auto":
                    kwargs["language"] = language
                wavs, sample_rate = model.generate_voice_clone(**kwargs)
                if not wavs:
                    raise RuntimeError("Qwen backend returned empty audio output.")
                return [wavs[0]], int(sample_rate)

            started = time.time()
            group_wavs: list[Any] | None = None
            sample_rate: int | None = None
            cancel_succeeded = False
            discard_group_after_completion = False
            retry_group_with_smaller_batch = False
            stop_message_shown = False
            abort_message_shown = False

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_generation_group)
                while True:
                    try:
                        group_wavs, sample_rate = future.result(timeout=0.15)
                        break
                    except concurrent.futures.TimeoutError:
                        if stop_controller.stop_requested and not stop_message_shown:
                            stop_message_shown = True
                            progress.mark_stop_requested()
                            if args.no_defrag_ui:
                                emit_status(
                                    "Stop requested; finishing current inference group (Ctrl+C again to try abort)."
                                )
                        if stop_controller.abort_current_batch and not abort_message_shown:
                            abort_message_shown = True
                            progress.mark_abort_requested()
                            if args.no_defrag_ui:
                                emit_status(
                                    "Abort requested; attempting to cancel current inference group..."
                                )
                            cancel_succeeded = future.cancel()
                            if cancel_succeeded:
                                break
                            discard_group_after_completion = True
                        continue
                    except Exception as gen_exc:
                        if (
                            is_moss_backend(tts_backend)
                            and len(active_batches) > 1
                            and is_cuda_oom_error(gen_exc)
                        ):
                            reduced = max(1, len(active_batches) // 2)
                            active_inference_batch_size = reduced
                            retry_group_with_smaller_batch = True
                            emit_status(
                                "CUDA OOM on grouped MOSS inference; reducing inference "
                                f"batch size to {reduced} and retrying.",
                                warning=True,
                            )
                            if torch.cuda.is_available():
                                try:
                                    torch.cuda.empty_cache()
                                except Exception:
                                    pass
                            break
                        raise

            if retry_group_with_smaller_batch:
                continue

            if cancel_succeeded:
                progress.mark_batch_aborted(canceled=True)
                stop_controller.request_stop()
                if args.no_defrag_ui:
                    emit_status(
                        f"Batch group starting at {first_global_batch}: canceled before execution; will remain in continue file."
                    )
                    emit_progress(f"batch group {first_global_batch}-{last_global_batch} canceled")
                break

            if group_wavs is None or sample_rate is None or len(group_wavs) == 0:
                raise RuntimeError(f"Empty audio output for {batch_label}.")

            if discard_group_after_completion:
                progress.mark_batch_aborted(canceled=False)
                stop_controller.request_stop()
                if args.no_defrag_ui:
                    emit_status(
                        "Current inference group could not be interrupted; output discarded so it remains in continue file."
                    )
                    emit_progress(
                        f"batch group {first_global_batch}-{last_global_batch} discarded for resume"
                    )
                break

            if len(group_wavs) != len(active_batches):
                raise RuntimeError(
                    "Generated audio count does not match planned batch count."
                )

            duration = time.time() - started
            per_batch_duration = duration / float(max(1, len(active_batches)))
            for batch_offset, batch in enumerate(active_batches):
                global_batch = first_global_batch + batch_offset
                audio_data = group_wavs[batch_offset]
                part_path = parts_dir / f"batch_{global_batch:05d}.wav"
                sf.write(str(part_path), audio_data, sample_rate)
                part_files.append(str(part_path.relative_to(run_dir).as_posix()))
                if continuation_chain and is_moss_backend(tts_backend):
                    continuation_audio_source = str(part_path.resolve())
                    continuation_prefix_text = batch.text.strip()

                completed_this_run += 1
                completed_chars_total += batch.char_count
                state_batch_chars.append(batch.char_count)
                if batch.starts_chapter:
                    state_chapter_batches.append(global_batch)
                    chapter_title = sanitize_chapter_title(batch.chapter_title)
                    if chapter_title:
                        state_chapter_titles[global_batch] = chapter_title
                plain_batch_durations.append(per_batch_duration)
                progress.mark_batch_complete(batch.char_count, per_batch_duration)
                if args.no_defrag_ui:
                    emit_status(f"Batch {global_batch}: complete in {per_batch_duration:.1f}s")
                    emit_progress(f"completed batch {global_batch}")

                state["part_files"] = part_files
                state["batch_char_counts"] = state_batch_chars
                state["chapter_batch_numbers"] = state_chapter_batches
                state["chapter_titles_by_batch"] = {
                    str(batch_number): title
                    for batch_number, title in sorted(state_chapter_titles.items())
                }
                state["completed_batches"] = len(part_files)
                state["completed_characters"] = completed_chars_total
                state["sample_rate"] = int(sample_rate)
                state["last_generated_batch_text"] = batch.text
                state["last_generated_batch_file"] = str(
                    part_path.relative_to(run_dir).as_posix()
                )
                state["updated_at"] = now_iso()
                save_state(state_path, state)

                if args.stop_after_batch > 0 and completed_this_run >= args.stop_after_batch:
                    stop_controller.request_stop()

            next_local_index += len(active_batches)
            if stop_controller.stop_requested:
                progress.mark_stop_requested()
                if args.no_defrag_ui:
                    emit_status("Stop requested; exiting after current inference group.")
                break

        combined_wav_path = (
            output_target
            if output_suffix == ".wav"
            else run_dir / "audiobook_combined_intermediate.wav"
        )
        emit_status("Combining audio parts with pauses...")
        if args.no_defrag_ui:
            emit_progress("combining parts")
        sample_rate_out, duration_out, part_start_samples = combine_parts_with_pause(
            sf=sf,
            np=np,
            part_paths=resolve_paths(run_dir, part_files),
            output_wav_path=combined_wav_path,
            pause_ms=pause_ms,
            chapter_batch_numbers=state_chapter_batches,
            chapter_pause_ms=chapter_pause_ms,
        )
        if output_suffix == ".mp3":
            chapter_metadata_path: Path | None = None
            if use_chapters:
                chapter_time_entries = chapter_time_entries_from_batches(
                    chapter_batch_numbers=state_chapter_batches,
                    part_start_samples=part_start_samples,
                    sample_rate=sample_rate_out,
                )
                chapter_entries_for_metadata: list[tuple[float, str | None]] = [
                    (
                        start_seconds,
                        state_chapter_titles.get(batch_number),
                    )
                    for start_seconds, batch_number in chapter_time_entries
                ]
                metadata_text = build_ffmetadata_with_chapters(
                    chapter_entries=chapter_entries_for_metadata,
                    total_duration_seconds=duration_out,
                )
                if "[CHAPTER]" in metadata_text:
                    chapter_metadata_path = run_dir / "chapters.ffmeta"
                    write_text_file(chapter_metadata_path, metadata_text)
            emit_status("Encoding high-quality MP3...")
            convert_wav_to_mp3(
                input_wav=combined_wav_path,
                output_mp3=output_target,
                quality_level=mp3_quality,
                chapter_metadata_path=chapter_metadata_path,
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
                "--inference-batch-size 1 and/or a smaller --max-chars-per-batch."
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
