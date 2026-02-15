#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Generator

import gradio as gr

from audiobook_qwen3 import (
    DEFAULT_ATTN_IMPLEMENTATION,
    DEFAULT_CHAPTER_PAUSE_MS,
    DEFAULT_DTYPE,
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_ID,
    DEFAULT_PAUSE_MS,
    DEFAULT_TTS_BACKEND,
    REFERENCE_AUDIO_EXTENSIONS,
)

APP_ROOT = Path(__file__).resolve().parent
CLI_SCRIPT = APP_ROOT / "audiobook_qwen3.py"
DEFAULT_RUN_ROOT = APP_ROOT / "runs"
MAX_LOG_LINES = 500


def _sanitize_output_name(name: str) -> str:
    cleaned = re.sub(r"[\\/:*?\"<>|]+", "_", name.strip())
    if not cleaned:
        cleaned = "audiobook.mp3"
    path = Path(cleaned)
    if path.suffix.lower() not in {".mp3", ".wav"}:
        cleaned = f"{cleaned}.mp3"
    return cleaned


def _stage_file(source_path: str | None, stage_dir: Path, stem_name: str) -> Path | None:
    if not source_path:
        return None
    src = Path(source_path).expanduser().resolve()
    if not src.exists() or not src.is_file():
        raise RuntimeError(f"Input file not found: {src}")
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem_name).strip("_") or "input"
    suffix = src.suffix if src.suffix else ".bin"
    target = stage_dir / f"{safe_stem}{suffix}"
    counter = 2
    while target.exists():
        target = stage_dir / f"{safe_stem}_{counter}{suffix}"
        counter += 1
    shutil.copy2(src, target)
    return target


def _write_text_input(text: str, target_path: Path) -> None:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(normalized + "\n", encoding="utf-8", newline="\n")


def _find_output_path(
    expected_output: Path | None,
    log_lines: list[str],
) -> Path | None:
    if expected_output and expected_output.exists():
        return expected_output.resolve()

    state_path: Path | None = None
    for line in reversed(log_lines):
        if line.startswith("Done: ") or line.startswith("Audio: "):
            candidate = Path(line.split(":", 1)[1].strip()).expanduser()
            if candidate.exists():
                return candidate.resolve()
        if line.startswith("State: "):
            candidate_state = Path(line.split(":", 1)[1].strip()).expanduser()
            if candidate_state.exists():
                state_path = candidate_state.resolve()
                break

    if state_path:
        try:
            state = json.loads(state_path.read_text(encoding="utf-8"))
            for key in ("output_audio", "output_wav"):
                value = state.get(key)
                if not value:
                    continue
                candidate = Path(str(value)).expanduser()
                if candidate.exists():
                    return candidate.resolve()
        except Exception:
            pass

    return None


def _build_status_message(title: str, detail: str | None = None) -> str:
    if detail:
        return f"### {title}\n\n{detail}"
    return f"### {title}"


def run_generation(
    text_input: str,
    text_file: str | None,
    reference_audio_file: str | None,
    reference_audio_path: str,
    reference_text: str,
    reference_text_file: str | None,
    x_vector_only_mode: bool,
    resume_state_file: str | None,
    output_name: str,
    run_root: str,
    max_chars_per_batch: int,
    pause_ms: int,
    chapter_pause_ms: int,
    mp3_quality: int,
    use_chapters: bool,
    language: str,
    tts_backend: str,
    model_id: str,
    device: str,
    dtype: str,
    attn_implementation: str,
    inference_batch_size: int,
    max_new_tokens: int,
    stop_after_batch: int,
) -> Generator[tuple[str, str | None, str | None, str], None, None]:
    log_lines: list[str] = []

    def push_log(line: str) -> str:
        log_lines.append(line.rstrip("\n"))
        if len(log_lines) > MAX_LOG_LINES:
            del log_lines[: len(log_lines) - MAX_LOG_LINES]
        return "\n".join(log_lines)

    yield _build_status_message("Preparing job..."), None, None, ""

    if not CLI_SCRIPT.exists():
        message = f"CLI script not found: {CLI_SCRIPT}"
        yield _build_status_message("Unable to start", message), None, None, message
        return

    run_root_path = Path(run_root or str(DEFAULT_RUN_ROOT)).expanduser().resolve()
    run_root_path.mkdir(parents=True, exist_ok=True)
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    job_root = run_root_path / "_gradio_jobs" / job_id
    staging_dir = job_root / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    expected_output: Path | None = None
    command: list[str] = [sys.executable, str(CLI_SCRIPT)]

    try:
        resume_state_path = _stage_file(resume_state_file, staging_dir, "resume_state")

        text_path: Path | None = None
        if text_input.strip():
            text_path = staging_dir / "book_text.txt"
            _write_text_input(text_input, text_path)
        elif text_file:
            text_path = _stage_file(text_file, staging_dir, "book_text")

        reference_audio_value: str | None = None
        if reference_audio_file:
            staged_audio = _stage_file(reference_audio_file, staging_dir, "reference_audio")
            reference_audio_value = str(staged_audio) if staged_audio else None
        elif reference_audio_path.strip():
            reference_audio_value = reference_audio_path.strip()

        reference_text_file_path = _stage_file(
            reference_text_file, staging_dir, "reference_text"
        )

        if not resume_state_path and not text_path:
            message = "Provide book text (paste text or upload a .txt file), or upload a resume state."
            yield _build_status_message("Missing input", message), None, None, message
            return
        if not resume_state_path and not reference_audio_value:
            message = "Provide reference audio (upload file or path/URL) for a new run."
            yield _build_status_message("Missing reference audio", message), None, None, message
            return
        if (
            reference_audio_file
            and not resume_state_path
            and not x_vector_only_mode
            and not reference_text.strip()
            and not reference_text_file_path
        ):
            message = (
                "Reference transcript is required for uploaded reference audio unless "
                "X-Vector only mode is enabled."
            )
            yield _build_status_message("Missing reference transcript", message), None, None, message
            return

        if text_path:
            command.extend(["--text-file", str(text_path)])
        if resume_state_path:
            command.extend(["--resume-state", str(resume_state_path)])
        if reference_audio_value:
            command.extend(["--reference-audio", reference_audio_value])
        if reference_text.strip():
            command.extend(["--reference-text", reference_text.strip()])
        elif reference_text_file_path:
            command.extend(["--reference-text-file", str(reference_text_file_path)])

        if x_vector_only_mode:
            command.append("--x-vector-only-mode")

        output_name = _sanitize_output_name(output_name)
        if not resume_state_path or output_name.strip():
            expected_output = job_root / output_name
            command.extend(["--output", str(expected_output)])

        command.extend(
            [
                "--run-root",
                str(run_root_path),
                "--max-chars-per-batch",
                str(int(max_chars_per_batch)),
                "--pause-ms",
                str(int(pause_ms)),
                "--chapter-pause-ms",
                str(int(chapter_pause_ms)),
                "--mp3-quality",
                str(int(mp3_quality)),
                "--language",
                language,
                "--tts-backend",
                tts_backend.strip() or DEFAULT_TTS_BACKEND,
                "--model-id",
                model_id.strip() or DEFAULT_MODEL_ID,
                "--device",
                device.strip() or "cuda:0",
                "--dtype",
                dtype,
                "--attn-implementation",
                attn_implementation,
                "--inference-batch-size",
                str(int(inference_batch_size)),
                "--max-new-tokens",
                str(int(max_new_tokens)),
                "--no-defrag-ui",
            ]
        )
        if use_chapters:
            command.append("--use-chapters")
        if stop_after_batch > 0:
            command.extend(["--stop-after-batch", str(int(stop_after_batch))])
    except Exception as exc:
        message = f"Input preparation failed: {exc}"
        yield _build_status_message("Unable to prepare job", message), None, None, message
        return

    push_log(f"$ {' '.join(command)}")
    yield _build_status_message("Running generation..."), None, None, "\n".join(log_lines)

    started = time.time()
    process = subprocess.Popen(
        command,
        cwd=str(APP_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    if process.stdout:
        for line in process.stdout:
            logs_text = push_log(line.rstrip("\n"))
            yield _build_status_message("Running generation..."), None, None, logs_text

    exit_code = process.wait()
    elapsed = time.time() - started
    logs_text = "\n".join(log_lines)
    output_path = _find_output_path(expected_output, log_lines)

    if exit_code != 0:
        detail = f"CLI exited with code {exit_code}. Check logs below."
        yield _build_status_message("Generation failed", detail), None, None, logs_text
        return

    if output_path and output_path.exists():
        detail = f"Completed in {elapsed:.1f}s\n\nOutput: `{output_path}`"
        yield (
            _build_status_message("Generation complete", detail),
            str(output_path),
            str(output_path),
            logs_text,
        )
        return

    detail = (
        f"Completed in {elapsed:.1f}s, but output file was not auto-detected. "
        "Review logs for the final path."
    )
    yield _build_status_message("Generation complete", detail), None, None, logs_text


def build_demo() -> gr.Blocks:
    css = """
    .app-shell {max-width: 1200px; margin: 0 auto;}
    .mono-log textarea {font-family: Consolas, "Cascadia Mono", Menlo, monospace !important;}
    """
    audio_types = sorted(REFERENCE_AUDIO_EXTENSIONS)

    with gr.Blocks(
        title="MOSS/Qwen Audiobook Studio",
        theme=gr.themes.Soft(primary_hue="sky", neutral_hue="slate"),
        css=css,
    ) as demo:
        gr.Markdown(
            "## MOSS/Qwen Audiobook Studio\n"
            "Generate audiobooks with chapter tags, pause controls, adaptive inference batching, and MP3 chapter metadata."
        )
        with gr.Row(elem_classes=["app-shell"]):
            with gr.Column(scale=2):
                gr.Markdown("### Input")
                text_input = gr.Textbox(
                    label="Book Text (optional)",
                    placeholder="Paste text here, or upload a .txt file below.",
                    lines=12,
                )
                text_file = gr.File(
                    label="Book Text File (.txt)",
                    file_types=[".txt"],
                    type="filepath",
                )
                reference_audio_file = gr.File(
                    label="Reference Audio File",
                    file_types=audio_types,
                    type="filepath",
                )
                reference_audio_path = gr.Textbox(
                    label="Reference Audio Path or URL (optional)",
                    placeholder="Use this if you do not upload audio.",
                )
                reference_text = gr.Textbox(
                    label="Reference Transcript (optional)",
                    placeholder="Required unless X-Vector only mode is enabled.",
                    lines=4,
                )
                reference_text_file = gr.File(
                    label="Reference Transcript File (.txt)",
                    file_types=[".txt"],
                    type="filepath",
                )
                x_vector_only_mode = gr.Checkbox(
                    label="X-Vector Only Mode (no transcript required)",
                    value=False,
                )
                resume_state_file = gr.File(
                    label="Resume State (session_state.json, optional)",
                    file_types=[".json"],
                    type="filepath",
                )

            with gr.Column(scale=1):
                gr.Markdown("### Output and Runtime")
                output_name = gr.Textbox(label="Output Filename", value="audiobook.mp3")
                run_root = gr.Textbox(label="Run Root Folder", value=str(DEFAULT_RUN_ROOT.resolve()))

                max_chars_per_batch = gr.Slider(
                    label="Max chars per batch",
                    minimum=100,
                    maximum=5000,
                    step=50,
                    value=DEFAULT_MAX_CHARS,
                )
                pause_ms = gr.Slider(
                    label="Pause between batches (ms)",
                    minimum=0,
                    maximum=5000,
                    step=25,
                    value=DEFAULT_PAUSE_MS,
                )
                chapter_pause_ms = gr.Slider(
                    label="Additional chapter pause (ms)",
                    minimum=0,
                    maximum=8000,
                    step=25,
                    value=DEFAULT_CHAPTER_PAUSE_MS,
                )
                mp3_quality = gr.Slider(
                    label="MP3 quality (0 best, 9 smallest)",
                    minimum=0,
                    maximum=9,
                    step=1,
                    value=0,
                )
                use_chapters = gr.Checkbox(label="Embed chapter metadata", value=True)
                language = gr.Dropdown(
                    label="Language",
                    choices=["Auto", "English", "Chinese", "Japanese", "Korean"],
                    value="Auto",
                    allow_custom_value=True,
                )

                with gr.Accordion("Advanced", open=False):
                    tts_backend = gr.Dropdown(
                        label="TTS backend",
                        choices=["moss-delay", "moss-local", "qwen", "auto"],
                        value=DEFAULT_TTS_BACKEND,
                    )
                    model_id = gr.Textbox(label="Model ID", value=DEFAULT_MODEL_ID)
                    device = gr.Textbox(label="Device", value="cuda:0")
                    dtype = gr.Dropdown(
                        label="DType",
                        choices=["float16", "bfloat16", "float32"],
                        value=DEFAULT_DTYPE,
                    )
                    attn_implementation = gr.Dropdown(
                        label="Attention implementation",
                        choices=["sdpa", "flash_attention_2"],
                        value=DEFAULT_ATTN_IMPLEMENTATION,
                    )
                    inference_batch_size = gr.Slider(
                        label="Inference batch size (0 = auto)",
                        minimum=0,
                        maximum=16,
                        step=1,
                        value=0,
                    )
                    max_new_tokens = gr.Slider(
                        label="Max new tokens (MOSS)",
                        minimum=256,
                        maximum=16384,
                        step=128,
                        value=DEFAULT_MAX_NEW_TOKENS,
                    )
                    stop_after_batch = gr.Number(
                        label="Stop after batch (testing)",
                        value=0,
                        precision=0,
                    )

        with gr.Row():
            run_button = gr.Button("Generate Audiobook", variant="primary", size="lg")
            clear_button = gr.Button("Clear Output", variant="secondary")

        status = gr.Markdown("### Ready")
        audio_output = gr.Audio(label="Output Preview", type="filepath", interactive=False)
        file_output = gr.File(label="Download Output", interactive=False)
        logs = gr.Textbox(label="Run Log", lines=20, elem_classes=["mono-log"])

        run_button.click(
            fn=run_generation,
            inputs=[
                text_input,
                text_file,
                reference_audio_file,
                reference_audio_path,
                reference_text,
                reference_text_file,
                x_vector_only_mode,
                resume_state_file,
                output_name,
                run_root,
                max_chars_per_batch,
                pause_ms,
                chapter_pause_ms,
                mp3_quality,
                use_chapters,
                language,
                tts_backend,
                model_id,
                device,
                dtype,
                attn_implementation,
                inference_batch_size,
                max_new_tokens,
                stop_after_batch,
            ],
            outputs=[status, audio_output, file_output, logs],
        )

        clear_button.click(
            fn=lambda: ("### Ready", None, None, ""),
            outputs=[status, audio_output, file_output, logs],
        )

    return demo


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    app = build_demo()
    app.queue(max_size=8).launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        share=_env_flag("GRADIO_SHARE", default=False),
    )
