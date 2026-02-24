#!/usr/bin/env python3
from __future__ import annotations

import html
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Generator

import gradio as gr
try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover - optional runtime dependency
    psutil = None

from audiobook_qwen3 import (
    DEFAULT_ATTN_IMPLEMENTATION,
    DEFAULT_CHAPTER_PAUSE_MS,
    DEFAULT_CONTINUATION_ANCHOR_SECONDS,
    DEFAULT_DTYPE,
    DEFAULT_MAX_CHARS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL_ID,
    DEFAULT_MOSS_AUDIO_REPETITION_PENALTY,
    DEFAULT_MOSS_AUDIO_TEMPERATURE,
    DEFAULT_MOSS_AUDIO_TOP_K,
    DEFAULT_MOSS_AUDIO_TOP_P,
    DEFAULT_PAUSE_MS,
    DEFAULT_TTS_BACKEND,
    REFERENCE_AUDIO_EXTENSIONS,
    build_batches,
    choose_attention_implementation,
    choose_dtype,
    extract_chapter_titles_from_raw_text,
    require_runtime_dependencies,
    split_into_paragraphs,
)

APP_ROOT = Path(__file__).resolve().parent
CLI_SCRIPT = APP_ROOT / "audiobook_qwen3.py"
DEFAULT_RUN_ROOT = APP_ROOT / "runs"
VOICES_DIR = APP_ROOT / "voices"
DEFAULT_VOICEGEN_MODEL_ID = "OpenMOSS-Team/MOSS-VoiceGenerator"
MAX_LOG_LINES = 500
PROGRESS_LINE_RE = re.compile(
    r"progress:\s+batches\s+(\d+)/(\d+)\s+\([^)]+\),\s+chars\s+(\d+)/(\d+)\s+\([^)]+\),\s+elapsed\s+([^,]+),\s+eta\s+(.+)$",
    re.IGNORECASE,
)
STATUS_LINE_RE = re.compile(r"^status:\s*(.+)$", re.IGNORECASE)
VOICE_METADATA_SUFFIX = ".voice.json"


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


def _format_bytes(size_bytes: int | float | None) -> str:
    if size_bytes is None:
        return "n/a"
    size = float(max(0, size_bytes))
    units = ["B", "KB", "MB", "GB", "TB"]
    index = 0
    while size >= 1024.0 and index < len(units) - 1:
        size /= 1024.0
        index += 1
    if index == 0:
        return f"{int(size)} {units[index]}"
    return f"{size:.1f} {units[index]}"


def _safe_file_size(path: Path | None) -> int:
    if not path or not path.exists() or not path.is_file():
        return 0
    try:
        return int(path.stat().st_size)
    except OSError:
        return 0


def _summarize_directory(path: Path) -> tuple[int, int]:
    total_size = 0
    file_count = 0
    if not path.exists() or not path.is_dir():
        return 0, 0
    for root, _dirs, files in os.walk(path):
        for filename in files:
            file_count += 1
            file_path = Path(root) / filename
            try:
                total_size += int(file_path.stat().st_size)
            except OSError:
                continue
    return total_size, file_count


def _extract_latest_progress(log_lines: list[str]) -> dict[str, int | str] | None:
    for line in reversed(log_lines):
        match = PROGRESS_LINE_RE.search(line.strip())
        if not match:
            continue
        try:
            return {
                "completed_batches": int(match.group(1)),
                "total_batches": int(match.group(2)),
                "completed_chars": int(match.group(3)),
                "total_chars": int(match.group(4)),
                "elapsed_text": match.group(5).strip(),
                "eta_text": match.group(6).strip(),
            }
        except (TypeError, ValueError):
            continue
    return None


def _extract_latest_status(log_lines: list[str]) -> str:
    for line in reversed(log_lines):
        match = STATUS_LINE_RE.match(line.strip())
        if match:
            return match.group(1).strip()
    for line in reversed(log_lines):
        stripped = line.strip()
        if stripped:
            return stripped
    return "(idle)"


def _find_state_path(log_lines: list[str]) -> Path | None:
    for line in reversed(log_lines):
        if not line.startswith("State: "):
            continue
        try:
            candidate = Path(line.split(":", 1)[1].strip()).expanduser()
        except Exception:
            continue
        if candidate.exists():
            return candidate.resolve()
    return None


def _read_state_summary(state_path: Path | None) -> dict[str, object]:
    if not state_path or not state_path.exists():
        return {}
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(state, dict):
        return {}
    return state


def _device_gpu_index(device_hint: str) -> int | None:
    match = re.search(r"cuda:(\d+)", str(device_hint).strip().lower())
    if not match:
        return 0 if "cuda" in str(device_hint).strip().lower() else None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def _query_gpu_metrics(
    device_hint: str,
    cache: dict[str, object],
    now_ts: float,
    ttl_seconds: float = 0.8,
) -> dict[str, object]:
    cached_ts = float(cache.get("gpu_ts", 0.0) or 0.0)
    if now_ts - cached_ts < ttl_seconds and isinstance(cache.get("gpu_data"), dict):
        return dict(cache["gpu_data"])  # type: ignore[index]

    result: dict[str, object] = {
        "available": False,
        "name": "n/a",
        "util_pct": None,
        "mem_used_mb": None,
        "mem_total_mb": None,
        "mem_pct": None,
        "temp_c": None,
        "power_w": None,
        "note": "nvidia-smi unavailable",
    }
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=1.2,
        )
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip() or "nvidia-smi failed")
        rows = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
        if not rows:
            raise RuntimeError("No GPU rows")
        gpu_index = _device_gpu_index(device_hint)
        if gpu_index is None or gpu_index < 0 or gpu_index >= len(rows):
            gpu_index = 0
        parts = [part.strip() for part in rows[gpu_index].split(",")]
        if len(parts) < 6:
            raise RuntimeError(f"Unexpected nvidia-smi output: {rows[gpu_index]}")
        name = parts[0]

        def _maybe_float(value: str) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None

        util_pct = _maybe_float(parts[1])
        mem_used_mb = _maybe_float(parts[2])
        mem_total_mb = _maybe_float(parts[3])
        temp_c = _maybe_float(parts[4])
        power_w = _maybe_float(parts[5])
        mem_pct = None
        if mem_used_mb is not None and mem_total_mb and mem_total_mb > 0:
            mem_pct = (mem_used_mb / mem_total_mb) * 100.0
        result = {
            "available": True,
            "name": name,
            "util_pct": util_pct,
            "mem_used_mb": mem_used_mb,
            "mem_total_mb": mem_total_mb,
            "mem_pct": mem_pct,
            "temp_c": temp_c,
            "power_w": power_w,
            "note": None,
        }
    except Exception as exc:
        result["note"] = str(exc)

    cache["gpu_ts"] = now_ts
    cache["gpu_data"] = dict(result)
    return result


def _metric_bar(
    label: str,
    value: str,
    percent: float | None,
    color: str = "#38bdf8",
    subtle: str | None = None,
) -> str:
    if percent is None:
        fill_pct = 0.0
        muted = " opacity-70"
    else:
        fill_pct = max(0.0, min(100.0, float(percent)))
        muted = ""
    subtle_html = (
        f"<div class='telemetry-subtle'>{html.escape(subtle)}</div>" if subtle else ""
    )
    return (
        f"<div class='telemetry-metric{muted}'>"
        f"<div class='telemetry-row'><span>{html.escape(label)}</span><span>{html.escape(value)}</span></div>"
        f"<div class='telemetry-bar'><div class='telemetry-fill' style='width:{fill_pct:.1f}%; background:{color};'></div></div>"
        f"{subtle_html}"
        f"</div>"
    )


def _render_telemetry_panel(
    *,
    log_lines: list[str],
    command: list[str] | None,
    job_root: Path | None,
    run_root: Path | None,
    expected_output: Path | None,
    device_hint: str,
    started_at: float | None,
    process_pid: int | None,
    cache: dict[str, object],
) -> str:
    now_ts = time.time()
    elapsed = max(0.0, now_ts - started_at) if started_at is not None else 0.0

    progress = _extract_latest_progress(log_lines)
    phase = _extract_latest_status(log_lines)
    state_path = _find_state_path(log_lines)
    state = _read_state_summary(state_path)

    run_dir: Path | None = None
    state_run_dir = state.get("run_dir")
    if state_run_dir:
        try:
            candidate = Path(str(state_run_dir)).expanduser()
            if candidate.exists():
                run_dir = candidate.resolve()
        except Exception:
            run_dir = None
    if run_dir is None and state_path:
        run_dir = state_path.parent.resolve()

    output_path = _find_output_path(expected_output, log_lines)
    output_size = _safe_file_size(output_path)
    state_size = _safe_file_size(state_path)

    job_size = 0
    job_files = 0
    if job_root:
        job_size, job_files = _summarize_directory(job_root)

    run_dir_size = 0
    run_dir_files = 0
    parts_count = 0
    parts_size = 0
    if run_dir and run_dir.exists():
        run_dir_size, run_dir_files = _summarize_directory(run_dir)
        parts_dir = run_dir / "parts"
        if parts_dir.exists():
            for part_path in parts_dir.glob("*.wav"):
                parts_count += 1
                parts_size += _safe_file_size(part_path)

    disk_pct = None
    disk_value = "n/a"
    if run_root:
        try:
            usage = shutil.disk_usage(run_root)
            if usage.total > 0:
                disk_pct = (usage.used / usage.total) * 100.0
            disk_value = (
                f"{_format_bytes(usage.used)} / {_format_bytes(usage.total)}"
            )
        except Exception:
            pass

    cpu_pct = None
    ram_pct = None
    ram_value = "n/a"
    proc_rss = None
    if psutil is not None:
        try:
            cpu_pct = float(psutil.cpu_percent(interval=None))
        except Exception:
            cpu_pct = None
        try:
            vm = psutil.virtual_memory()
            ram_pct = float(getattr(vm, "percent", 0.0))
            ram_value = f"{_format_bytes(getattr(vm, 'used', 0))} / {_format_bytes(getattr(vm, 'total', 0))}"
        except Exception:
            pass
        if process_pid:
            try:
                proc_rss = int(psutil.Process(int(process_pid)).memory_info().rss)
            except Exception:
                proc_rss = None

    gpu = _query_gpu_metrics(device_hint=device_hint, cache=cache, now_ts=now_ts)

    recent_blob = "\n".join(log_lines[-80:])
    unique_chars = len(set(recent_blob)) if recent_blob else 0
    entropy_pct = (
        min(100.0, (unique_chars / max(1, len(recent_blob))) * 800.0) if recent_blob else 0.0
    )
    log_rate = (len(log_lines) / elapsed * 60.0) if elapsed > 0 and log_lines else 0.0

    current_total = max(job_size, run_dir_size, output_size, parts_size)
    previous_total = float(cache.get("growth_prev_total", current_total) or current_total)
    previous_ts = float(cache.get("growth_prev_ts", now_ts) or now_ts)
    dt = max(1e-6, now_ts - previous_ts)
    growth_bps = max(0.0, (float(current_total) - previous_total) / dt)
    cache["growth_prev_total"] = float(current_total)
    cache["growth_prev_ts"] = now_ts

    command_flags = sum(1 for part in (command or []) if str(part).startswith("--"))
    command_chars = sum(len(str(part)) for part in (command or []))

    completed_batches = int(progress["completed_batches"]) if progress else 0
    total_batches = int(progress["total_batches"]) if progress else 0
    completed_chars = int(progress["completed_chars"]) if progress else 0
    total_chars = int(progress["total_chars"]) if progress else 0
    batch_pct = (completed_batches / total_batches * 100.0) if progress and total_batches > 0 else None
    char_pct = (completed_chars / total_chars * 100.0) if progress and total_chars > 0 else None

    output_share_pct = None
    if current_total > 0 and output_size > 0:
        output_share_pct = (output_size / float(current_total)) * 100.0
    parts_share_pct = None
    if current_total > 0 and parts_size > 0:
        parts_share_pct = (parts_size / float(current_total)) * 100.0

    pulse_pct = min(100.0, ((elapsed * 9.73) % 100.0)) if started_at is not None else 0.0
    fun_temp = gpu.get("temp_c")
    thermal_vibe = (
        min(100.0, max(0.0, (float(fun_temp) / 85.0) * 100.0))
        if isinstance(fun_temp, (int, float))
        else (ram_pct if ram_pct is not None else None)
    )
    queue_vibe = min(100.0, command_flags * 3.0 + (5.0 if total_batches else 0.0))

    top_metrics = [
        _metric_bar(
            "Batch Progress",
            f"{completed_batches}/{total_batches}" if progress else "n/a",
            batch_pct,
            color="#22c55e",
            subtle=(f"eta {progress['eta_text']}" if progress else None),
        ),
        _metric_bar(
            "Char Progress",
            f"{completed_chars}/{total_chars}" if progress else "n/a",
            char_pct,
            color="#10b981",
            subtle=(f"elapsed {progress['elapsed_text']}" if progress else None),
        ),
        _metric_bar(
            "GPU Util",
            (
                f"{gpu['util_pct']:.0f}%"
                if isinstance(gpu.get("util_pct"), (int, float))
                else "n/a"
            ),
            float(gpu["util_pct"]) if isinstance(gpu.get("util_pct"), (int, float)) else None,
            color="#60a5fa",
            subtle=str(gpu.get("name") or "n/a"),
        ),
        _metric_bar(
            "GPU VRAM",
            (
                f"{float(gpu['mem_used_mb']):.0f}/{float(gpu['mem_total_mb']):.0f} MB"
                if isinstance(gpu.get("mem_used_mb"), (int, float))
                and isinstance(gpu.get("mem_total_mb"), (int, float))
                else "n/a"
            ),
            float(gpu["mem_pct"]) if isinstance(gpu.get("mem_pct"), (int, float)) else None,
            color="#3b82f6",
            subtle=(
                f"{float(gpu['temp_c']):.0f}C, {float(gpu['power_w']):.0f}W"
                if isinstance(gpu.get("temp_c"), (int, float))
                and isinstance(gpu.get("power_w"), (int, float))
                else None
            ),
        ),
        _metric_bar("CPU", f"{cpu_pct:.0f}%" if cpu_pct is not None else "n/a", cpu_pct, color="#f59e0b"),
        _metric_bar(
            "System RAM",
            ram_value,
            ram_pct,
            color="#f97316",
            subtle=(f"worker rss {_format_bytes(proc_rss)}" if proc_rss is not None else None),
        ),
    ]

    file_metrics = [
        _metric_bar(
            "Run Disk",
            disk_value,
            disk_pct,
            color="#a78bfa",
            subtle=(f"run root {run_root}" if run_root else None),
        ),
        _metric_bar(
            "Job Staging+Output",
            _format_bytes(job_size),
            (job_size / max(1.0, float(current_total)) * 100.0) if current_total > 0 else None,
            color="#06b6d4",
            subtle=(f"{job_files} files" if job_files else None),
        ),
        _metric_bar(
            "Run Artifacts",
            _format_bytes(run_dir_size),
            (run_dir_size / max(1.0, float(current_total)) * 100.0) if current_total > 0 else None,
            color="#14b8a6",
            subtle=(f"{run_dir_files} files" if run_dir_files else ("waiting for run dir" if run_dir is None else None)),
        ),
        _metric_bar(
            "Parts WAVs",
            f"{parts_count} files / {_format_bytes(parts_size)}",
            parts_share_pct,
            color="#34d399",
            subtle=("parts dir detected" if parts_count > 0 else "no parts yet"),
        ),
        _metric_bar(
            "Output File",
            _format_bytes(output_size),
            output_share_pct,
            color="#93c5fd",
            subtle=(str(output_path) if output_path else "not detected yet"),
        ),
        _metric_bar(
            "State JSON",
            _format_bytes(state_size),
            (state_size / max(1.0, float(current_total)) * 100.0) if state_size > 0 and current_total > 0 else None,
            color="#c084fc",
            subtle=(str(state_path) if state_path else "not detected yet"),
        ),
    ]

    fun_metrics = [
        _metric_bar(
            "Log Velocity",
            f"{log_rate:.1f} lines/min",
            min(100.0, log_rate / 2.0),
            color="#fb7185",
            subtle=f"{len(log_lines)} buffered lines",
        ),
        _metric_bar(
            "Artifact Growth",
            f"{_format_bytes(growth_bps)}/s",
            min(100.0, growth_bps / (1024.0 * 1024.0) * 18.0),
            color="#f43f5e",
            subtle="size delta of current job/run artifacts",
        ),
        _metric_bar(
            "Log Entropy",
            f"{unique_chars} unique chars",
            entropy_pct,
            color="#e879f9",
            subtle="recent log character diversity",
        ),
        _metric_bar(
            "Scheduler Pulse",
            f"{pulse_pct:.0f}%",
            pulse_pct,
            color="#22d3ee",
            subtle="playful UI heartbeat (derived from elapsed time)",
        ),
        _metric_bar(
            "Thermal Vibe",
            (
                f"{float(fun_temp):.0f}C GPU"
                if isinstance(fun_temp, (int, float))
                else ("RAM proxy" if ram_pct is not None else "n/a")
            ),
            thermal_vibe if isinstance(thermal_vibe, (int, float)) else None,
            color="#ef4444",
            subtle="real GPU temp when available; otherwise system proxy",
        ),
        _metric_bar(
            "CLI Complexity",
            f"{command_flags} flags / {command_chars} chars",
            min(100.0, queue_vibe),
            color="#84cc16",
            subtle="wrapper command size",
        ),
    ]

    state_bits: list[str] = []
    if state:
        state_bits.append(
            f"completed={state.get('completed_batches', 0)}"
        )
        state_bits.append(
            f"sample_rate={state.get('sample_rate', 'n/a')}"
        )
        chapter_batches = state.get("chapter_batch_numbers")
        if isinstance(chapter_batches, list):
            state_bits.append(f"chapters={len(chapter_batches)}")
    state_line = " | ".join(state_bits) if state_bits else "state not loaded yet"

    phase_html = html.escape(phase)
    gpu_note = gpu.get("note")
    gpu_note_html = (
        f"<div class='telemetry-note'>GPU note: {html.escape(str(gpu_note))}</div>"
        if gpu_note and not bool(gpu.get("available"))
        else ""
    )
    run_dir_html = html.escape(str(run_dir)) if run_dir else "(waiting for session state)"
    job_root_html = html.escape(str(job_root)) if job_root else "(not started)"

    return (
        "<div class='telemetry-shell'>"
        "<div class='telemetry-header'>"
        "<div><h3>Tech Telemetry</h3><div class='telemetry-phase'>phase: "
        f"{phase_html}</div></div>"
        f"<div class='telemetry-uptime'>uptime: {elapsed:.1f}s</div>"
        "</div>"
        "<div class='telemetry-meta'>"
        f"<span>job_root: {job_root_html}</span>"
        f"<span>run_dir: {run_dir_html}</span>"
        f"<span>{html.escape(state_line)}</span>"
        "</div>"
        f"{gpu_note_html}"
        "<div class='telemetry-grid'>"
        "<div class='telemetry-card'><div class='telemetry-card-title'>System + Progress</div>"
        + "".join(top_metrics)
        + "</div>"
        "<div class='telemetry-card'><div class='telemetry-card-title'>Artifacts + Sizes</div>"
        + "".join(file_metrics)
        + "</div>"
        "<div class='telemetry-card'><div class='telemetry-card-title'>Lab Signals</div>"
        + "".join(fun_metrics)
        + "</div>"
        "</div>"
        "</div>"
    )


def _ensure_voices_dir() -> Path:
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    return VOICES_DIR


def _safe_voice_stem(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name or "").strip()).strip("._")
    return cleaned or datetime.now().strftime("voice_%Y%m%d_%H%M%S")


def _read_text_file_best_effort(path: Path) -> str | None:
    encodings = ("utf-8", "utf-8-sig", "cp1252", "latin-1")
    for encoding in encodings:
        try:
            return path.read_text(encoding=encoding).strip()
        except UnicodeDecodeError:
            continue
        except OSError:
            return None
    return None


def _is_voice_metadata_file(path: Path) -> bool:
    return path.name.lower().endswith(VOICE_METADATA_SUFFIX)


def _scan_voice_library_entries() -> list[dict[str, str]]:
    voices_dir = _ensure_voices_dir()
    entries: list[dict[str, str]] = []
    for path in sorted(voices_dir.iterdir(), key=lambda p: p.name.lower()):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in REFERENCE_AUDIO_EXTENSIONS:
            transcript_path = path.with_suffix(".txt")
            transcript_flag = " +txt" if transcript_path.exists() else ""
            entries.append(
                {
                    "label": f"[audio{transcript_flag}] {path.name}",
                    "value": str(path.resolve()),
                    "kind": "audio",
                }
            )
            continue
        if _is_voice_metadata_file(path):
            entries.append(
                {
                    "label": f"[voice-preset] {path.name}",
                    "value": str(path.resolve()),
                    "kind": "metadata",
                }
            )
    return entries


def _voice_dropdown_choices() -> list[tuple[str, str]]:
    return [(entry["label"], entry["value"]) for entry in _scan_voice_library_entries()]


def refresh_voice_library_dropdown() -> tuple[dict[str, Any], str]:
    choices = _voice_dropdown_choices()
    voices_dir = _ensure_voices_dir()
    summary = (
        f"Voice library refreshed: {len(choices)} file(s) in `{voices_dir}`."
        if choices
        else f"No voice files found yet in `{voices_dir}`."
    )
    return gr.update(choices=choices, value=None), summary


def _load_voice_metadata(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def apply_voice_library_selection(
    selected_value: str | None,
    current_reference_audio_path: str,
    current_reference_text: str,
    current_voice_lab_name: str,
    current_voice_lab_description: str,
) -> tuple[str, str, str, str, str]:
    ref_audio = current_reference_audio_path
    ref_text = current_reference_text
    voice_name = current_voice_lab_name
    voice_description = current_voice_lab_description

    if not selected_value:
        return ref_audio, ref_text, voice_name, voice_description, "Select a file from /voices first."

    selected_path = Path(selected_value).expanduser()
    if not selected_path.exists():
        return ref_audio, ref_text, voice_name, voice_description, f"Selected file does not exist: {selected_path}"

    try:
        selected_path = selected_path.resolve()
    except Exception:
        pass

    if selected_path.suffix.lower() in REFERENCE_AUDIO_EXTENSIONS:
        ref_audio = str(selected_path)
        transcript_path = selected_path.with_suffix(".txt")
        transcript_text = _read_text_file_best_effort(transcript_path) if transcript_path.exists() else None
        if transcript_text is not None:
            ref_text = transcript_text
        message = f"Loaded audio voice reference: `{selected_path.name}`"
        if transcript_text is not None:
            message += f" (+ transcript `{transcript_path.name}`)"
        return ref_audio, ref_text, voice_name, voice_description, message

    if _is_voice_metadata_file(selected_path):
        metadata = _load_voice_metadata(selected_path)
        stem = selected_path.name[: -len(VOICE_METADATA_SUFFIX)] if selected_path.name.lower().endswith(VOICE_METADATA_SUFFIX) else selected_path.stem
        voice_name = str(metadata.get("voice_name") or stem)
        voice_description = str(
            metadata.get("voice_description")
            or metadata.get("instruction")
            or metadata.get("description")
            or current_voice_lab_description
        )

        preview_audio_value = metadata.get("preview_audio")
        preview_audio_path: Path | None = None
        if isinstance(preview_audio_value, str) and preview_audio_value.strip():
            candidate = Path(preview_audio_value).expanduser()
            if not candidate.is_absolute():
                candidate = (selected_path.parent / candidate).resolve()
            if candidate.exists():
                preview_audio_path = candidate
        fallback_audio = selected_path.with_suffix(".wav")
        if preview_audio_path is None and fallback_audio.exists():
            preview_audio_path = fallback_audio
        if preview_audio_path is not None:
            ref_audio = str(preview_audio_path)
            transcript_path = preview_audio_path.with_suffix(".txt")
            transcript_text = _read_text_file_best_effort(transcript_path)
            if transcript_text:
                ref_text = transcript_text

        return (
            ref_audio,
            ref_text,
            voice_name,
            voice_description,
            f"Loaded voice preset: `{selected_path.name}`",
        )

    return ref_audio, ref_text, voice_name, voice_description, f"Unsupported voice file type: `{selected_path.name}`"


def _extract_audio_waveforms_from_decoded(
    decoded_items: Any,
    *,
    np: Any,
    sf: Any,
) -> list[Any]:
    items = list(decoded_items) if isinstance(decoded_items, (list, tuple)) else [decoded_items]
    audio_list: list[Any] = []
    for item in items:
        audio_obj: Any | None = None
        if hasattr(item, "content"):
            try:
                content_items = list(getattr(item, "content") or [])
            except Exception:
                content_items = []
            for content in reversed(content_items):
                if hasattr(content, "audio_url"):
                    audio_obj = getattr(content, "audio_url")
                    if audio_obj is not None:
                        break
        if audio_obj is None and hasattr(item, "audio_url"):
            audio_obj = getattr(item, "audio_url")
        if audio_obj is None and getattr(item, "audio_codes_list", None):
            try:
                audio_obj = item.audio_codes_list[0]
            except Exception:
                audio_obj = None
        if audio_obj is None:
            raise RuntimeError("Decoded result did not include audio content.")
        if isinstance(audio_obj, (str, Path)):
            audio_arr, _decoded_sr = sf.read(str(audio_obj), always_2d=True)
            audio_obj = audio_arr.reshape(-1)
        if hasattr(audio_obj, "detach"):
            audio_obj = audio_obj.detach().float().cpu().numpy()
        else:
            audio_obj = np.asarray(audio_obj, dtype=np.float32)
        if getattr(audio_obj, "ndim", 1) > 1:
            audio_obj = audio_obj.reshape(-1)
        audio_list.append(audio_obj.astype(np.float32, copy=False))
    return audio_list


def _build_moss_voice_description_user_message(
    processor: Any,
    preview_text: str,
    voice_description: str,
) -> Any:
    if not hasattr(processor, "build_user_message"):
        raise RuntimeError("Processor does not expose build_user_message().")
    build_user_message = processor.build_user_message

    attempts = [
        {"text": preview_text, "instruction": voice_description},
        {"text": preview_text, "voice_description": voice_description},
        {"text": preview_text, "speaker_description": voice_description},
        {"text": preview_text, "style_description": voice_description},
    ]

    try:
        signature = inspect.signature(build_user_message)
        params = signature.parameters
        if "instruction" in params:
            return build_user_message(text=preview_text, instruction=voice_description)
        for key in ("voice_description", "speaker_description", "style_description"):
            if key in params:
                return build_user_message(text=preview_text, **{key: voice_description})
    except Exception:
        pass

    for kwargs in attempts:
        try:
            return build_user_message(**kwargs)
        except TypeError:
            continue
    return build_user_message(text=preview_text)


def generate_voice_from_description(
    voice_name: str,
    voice_description: str,
    preview_text: str,
    model_id: str,
    device: str,
    dtype_name: str,
    attn_implementation: str,
    max_new_tokens: int,
    overwrite_existing: bool,
) -> tuple[str, str | None, str | None, str, dict[str, Any], str, str, str]:
    logs: list[str] = []

    def log(message: str) -> None:
        logs.append(message)

    name = _safe_voice_stem(voice_name)
    description = str(voice_description or "").strip()
    text = str(preview_text or "").strip()
    model_id_value = str(model_id or DEFAULT_VOICEGEN_MODEL_ID).strip() or DEFAULT_VOICEGEN_MODEL_ID
    device_value = str(device or "cuda:0").strip() or "cuda:0"
    dtype_value = str(dtype_name or DEFAULT_DTYPE).strip() or DEFAULT_DTYPE
    attn_value = str(attn_implementation or DEFAULT_ATTN_IMPLEMENTATION).strip() or DEFAULT_ATTN_IMPLEMENTATION

    if not description:
        message = "Voice description is required."
        return (
            _build_status_message("Voice Lab error", message),
            None,
            None,
            message,
            gr.update(),
            message,
            "",
            "",
        )
    if not text:
        message = "Preview text is required."
        return (
            _build_status_message("Voice Lab error", message),
            None,
            None,
            message,
            gr.update(),
            message,
            "",
            "",
        )

    _ensure_voices_dir()
    wav_path = (VOICES_DIR / f"{name}.wav").resolve()
    txt_path = wav_path.with_suffix(".txt")
    meta_path = (VOICES_DIR / f"{name}{VOICE_METADATA_SUFFIX}").resolve()
    if not overwrite_existing:
        for path in (wav_path, txt_path, meta_path):
            if path.exists():
                message = f"Refusing to overwrite existing file: {path.name}"
                return (
                    _build_status_message("Voice Lab error", message),
                    None,
                    None,
                    message,
                    gr.update(),
                    message,
                    "",
                    "",
                )

    torch_device_desc = "cpu"
    sample_rate = 24000
    output_audio_path: str | None = None
    output_file_path: str | None = None
    selected_ref_audio = ""
    selected_ref_text = ""
    attn_warning: str | None = None
    dtype_warning: str | None = None
    final_status = "Voice generation finished."
    try:
        np, sf, torch, _Qwen3TTSModel, AutoModel, AutoProcessor = require_runtime_dependencies("moss-delay")
        if AutoModel is None or AutoProcessor is None:
            raise RuntimeError("transformers AutoModel/AutoProcessor are unavailable.")

        attn_resolved, attn_warning = choose_attention_implementation(attn_value)
        dtype_resolved, dtype_warning = choose_dtype(torch, dtype_value, device_value)
        if attn_warning:
            log(f"warning: {attn_warning}")
        if dtype_warning:
            log(f"warning: {dtype_warning}")

        torch_device = torch.device("cpu")
        if device_value.lower().startswith("cuda") and torch.cuda.is_available():
            torch_device = torch.device(device_value)
        elif device_value.lower().startswith("cuda"):
            log("warning: CUDA requested but unavailable; falling back to CPU.")
            torch_device = torch.device("cpu")
        else:
            torch_device = torch.device(device_value)
        torch_device_desc = str(torch_device)
        log(f"device={torch_device_desc} dtype={dtype_resolved} attn={attn_resolved}")

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }

        log(f"loading processor: {model_id_value}")
        processor = AutoProcessor.from_pretrained(model_id_value, trust_remote_code=True)
        if hasattr(processor, "audio_tokenizer"):
            try:
                processor.audio_tokenizer = processor.audio_tokenizer.to(torch_device)
            except Exception as exc:
                log(f"warning: audio_tokenizer.to(...) failed: {exc}")

        model_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": dtype_map[dtype_resolved],
        }
        if attn_resolved:
            model_kwargs["attn_implementation"] = attn_resolved

        log("loading model...")
        model = AutoModel.from_pretrained(model_id_value, **model_kwargs).to(torch_device)
        model.eval()
        sample_rate = int(
            getattr(getattr(processor, "model_config", None), "sampling_rate", 0)
            or getattr(getattr(processor, "audio_processor", None), "sampling_rate", 24000)
        )

        log("building prompt from text description...")
        user_message = _build_moss_voice_description_user_message(
            processor=processor,
            preview_text=text,
            voice_description=description,
        )
        conversations = [[user_message]]

        log("encoding conversation...")
        try:
            packed = processor(conversations, mode="generation")
        except TypeError:
            packed = processor(conversations)
        model_inputs: dict[str, Any] = {}
        for key, value in dict(packed).items():
            if hasattr(torch, "Tensor") and isinstance(value, torch.Tensor):
                model_inputs[key] = value.to(torch_device)
            else:
                model_inputs[key] = value

        log("generating audio...")
        with torch.inference_mode():
            try:
                outputs = model.generate(
                    **model_inputs,
                    max_new_tokens=int(max_new_tokens),
                    audio_temperature=float(DEFAULT_MOSS_AUDIO_TEMPERATURE),
                    audio_top_p=float(DEFAULT_MOSS_AUDIO_TOP_P),
                    audio_top_k=int(DEFAULT_MOSS_AUDIO_TOP_K),
                    audio_repetition_penalty=float(DEFAULT_MOSS_AUDIO_REPETITION_PENALTY),
                )
            except TypeError:
                outputs = model.generate(
                    **model_inputs,
                    max_new_tokens=int(max_new_tokens),
                )

        log("decoding audio...")
        decode_fn = getattr(processor, "decode", None)
        if callable(decode_fn):
            try:
                decoded = decode_fn(outputs, sampling_rate=int(sample_rate))
            except TypeError:
                decoded = decode_fn(outputs)
        elif hasattr(processor, "batch_decode") and callable(processor.batch_decode):
            decoded = processor.batch_decode(outputs)
        else:
            raise RuntimeError("Processor does not expose decode()/batch_decode().")

        waveforms = _extract_audio_waveforms_from_decoded(decoded, np=np, sf=sf)
        if not waveforms:
            raise RuntimeError("No waveform returned.")

        waveform = waveforms[0]
        if waveform is None or len(waveform) == 0:
            raise RuntimeError("Generated preview waveform is empty.")

        VOICES_DIR.mkdir(parents=True, exist_ok=True)
        sf.write(str(wav_path), waveform, sample_rate)
        txt_path.write_text(text + "\n", encoding="utf-8", newline="\n")
        metadata = {
            "voice_name": name,
            "voice_description": description,
            "preview_text": text,
            "preview_audio": str(wav_path),
            "preview_text_file": str(txt_path),
            "model_id": model_id_value,
            "device": device_value,
            "dtype": dtype_resolved,
            "attn_implementation": attn_resolved,
            "max_new_tokens": int(max_new_tokens),
            "sample_rate": int(sample_rate),
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "generator": "scripts/gradio_voice_lab",
        }
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=True), encoding="utf-8")
        output_audio_path = str(wav_path)
        output_file_path = str(meta_path)
        selected_ref_audio = str(wav_path)
        selected_ref_text = text
        final_status = (
            f"Saved voice preview and preset in `{VOICES_DIR}` "
            f"({wav_path.name}, {txt_path.name}, {meta_path.name})."
        )
        log(final_status)
    except Exception as exc:
        log(f"error: {exc.__class__.__name__}: {exc}")
        final_status = f"Voice generation failed: {exc}"
        output_audio_path = None
        output_file_path = None
        selected_ref_audio = ""
        selected_ref_text = ""
    finally:
        try:
            if "torch" in locals() and torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    dropdown_choices = _voice_dropdown_choices()
    dropdown_update = gr.update(choices=dropdown_choices, value=output_audio_path if output_audio_path else None)
    status_title = "Voice Lab complete" if output_audio_path else "Voice Lab error"
    detail_lines = [final_status]
    if attn_warning:
        detail_lines.append(f"attn note: {attn_warning}")
    if dtype_warning:
        detail_lines.append(f"dtype note: {dtype_warning}")
    detail_lines.append(f"device: {torch_device_desc}")
    detail_lines.append(f"sample rate: {sample_rate} Hz")
    detail_lines.append(f"voice file: {wav_path.name}")
    detail = "\n\n".join(detail_lines[:2]) + ("\n\n" + "\n".join(detail_lines[2:]) if len(detail_lines) > 2 else "")
    return (
        _build_status_message(status_title, detail),
        output_audio_path,
        output_file_path,
        "\n".join(logs),
        dropdown_update,
        final_status,
        selected_ref_audio,
        selected_ref_text,
    )


def _chapter_regex_template(preset: str, custom_heading_word: str) -> str:
    preset_value = (preset or "").strip().lower()
    months = (
        "January|February|March|April|May|June|July|August|September|October|November|December|"
        "Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec"
    )
    if preset_value == "chapter number":
        return r"^\s*chapter\s+(?:\d+|[ivxlcdm]+)\b.*$"
    if preset_value == "part number":
        return r"^\s*part\s+(?:\d+|[ivxlcdm]+)\b.*$"
    if preset_value == "month day heading":
        return rf"^\s*(?:{months})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s+\d{{4}})?\b.*$"
    if preset_value == "roman numerals":
        return r"^\s*(?:chapter\s+)?[ivxlcdm]+\b.*$"
    if preset_value == "common headings":
        chapter_pat = r"chapter\s+(?:\d+|[ivxlcdm]+)\b"
        part_pat = r"part\s+(?:\d+|[ivxlcdm]+)\b"
        date_pat = rf"(?:{months})\s+\d{{1,2}}(?:st|nd|rd|th)?(?:,\s+\d{{4}})?\b"
        return rf"^\s*(?:{chapter_pat}|{part_pat}|{date_pat}).*$"
    if preset_value == "custom word + number":
        token = re.escape((custom_heading_word or "Section").strip())
        return rf"^\s*{token}\s+(?:\d+|[ivxlcdm]+)\b.*$"
    return ""


def build_chapter_regex_pattern(
    preset: str,
    custom_heading_word: str,
    custom_regex: str,
) -> tuple[str, str]:
    if (preset or "").strip().lower() == "custom regex":
        pattern = str(custom_regex or "").strip()
        if not pattern:
            return "", "Enter a custom regex pattern."
        return pattern, "Using custom regex."
    pattern = _chapter_regex_template(preset, custom_heading_word)
    if not pattern:
        return "", "No pattern generated yet."
    return pattern, f"Generated pattern for preset: {preset}"


def _compile_user_regex(pattern: str, case_sensitive: bool) -> re.Pattern[str]:
    flags = re.MULTILINE
    if not case_sensitive:
        flags |= re.IGNORECASE
    return re.compile(pattern, flags)


def _find_chapter_line_matches(text: str, regex: re.Pattern[str]) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    for line_no, line in enumerate((text or "").splitlines(), start=1):
        if regex.search(line):
            matches.append((line_no, line))
    return matches


def _render_match_preview(matches: list[tuple[int, str]], limit: int = 40) -> str:
    if not matches:
        return "### Regex Matches\n\nNo matches found."
    lines = [f"### Regex Matches\n\nFound **{len(matches)}** matching line(s)."]
    lines.append("")
    lines.append("| Line | Text |")
    lines.append("|---:|---|")
    for line_no, text in matches[:limit]:
        preview = str(text).strip().replace("|", r"\|")
        if len(preview) > 160:
            preview = preview[:157] + "..."
        lines.append(f"| {line_no} | `{preview}` |")
    if len(matches) > limit:
        lines.append("")
        lines.append(f"...and {len(matches) - limit} more.")
    return "\n".join(lines)


def _build_chunk_preview_markdown(text: str, max_chars_per_batch: int) -> str:
    source_text = str(text or "")
    if not source_text.strip():
        return "### Chunk Preview\n\nNo text to analyze."
    try:
        paragraphs = split_into_paragraphs(source_text)
        chapter_titles = extract_chapter_titles_from_raw_text(source_text)
        batches = build_batches(
            paragraphs,
            max_chars_per_batch=max(100, int(max_chars_per_batch)),
            chapter_titles=chapter_titles,
        )
    except Exception as exc:
        return f"### Chunk Preview\n\nError while analyzing chunking: `{exc}`"

    if not batches:
        return "### Chunk Preview\n\nNo batches were produced."

    total_chars = sum(int(getattr(batch, "char_count", 0) or 0) for batch in batches)
    near_limit = sum(
        1
        for batch in batches
        if int(getattr(batch, "char_count", 0) or 0) >= int(max_chars_per_batch * 0.9)
    )
    chapter_starts = sum(1 for batch in batches if bool(getattr(batch, "starts_chapter", False)))
    header = [
        "### Chunk Preview",
        "",
        f"- Batches: **{len(batches)}**",
        f"- Total chars: **{total_chars}**",
        f"- Avg chars/batch: **{(total_chars / len(batches)):.1f}**",
        f"- Near max (>= 90% of {int(max_chars_per_batch)}): **{near_limit}**",
        f"- Chapter starts in batches: **{chapter_starts}**",
        "",
        "| # | chars | cut | chapter | p-range | preview |",
        "|---:|---:|---|---|---|---|",
    ]
    rows: list[str] = []
    for batch in batches[:50]:
        batch_text = str(getattr(batch, "text", "") or "").strip().replace("\n", " / ")
        if len(batch_text) > 110:
            batch_text = batch_text[:107] + "..."
        chars = int(getattr(batch, "char_count", len(batch_text)))
        cut = "tight" if chars >= int(max_chars_per_batch * 0.9) else ("short" if chars <= max(120, int(max_chars_per_batch * 0.35)) else "normal")
        chapter_label = "yes" if bool(getattr(batch, "starts_chapter", False)) else ""
        p_range = f"{int(getattr(batch, 'start_paragraph', 0)) + 1}-{int(getattr(batch, 'end_paragraph', 0)) + 1}"
        rows.append(
            f"| {int(getattr(batch, 'index', 0))} | {chars} | {cut} | {chapter_label} | {p_range} | `{batch_text.replace('|', r'\\|')}` |"
        )
    if len(batches) > 50:
        rows.append(f"| ... | ... | ... | ... | ... | ... ({len(batches)-50} more) |")
    return "\n".join(header + rows)


def chapter_assist_preview(
    source_text: str,
    pattern: str,
    case_sensitive: bool,
    max_chars_per_batch: int,
) -> tuple[str, str]:
    text = str(source_text or "")
    regex_text = str(pattern or "").strip()
    if not regex_text:
        return "### Regex Matches\n\nNo pattern set.", _build_chunk_preview_markdown(text, max_chars_per_batch)
    try:
        regex = _compile_user_regex(regex_text, case_sensitive=case_sensitive)
    except re.error as exc:
        return f"### Regex Matches\n\nInvalid regex: `{exc}`", _build_chunk_preview_markdown(text, max_chars_per_batch)
    matches = _find_chapter_line_matches(text, regex)
    return _render_match_preview(matches), _build_chunk_preview_markdown(text, max_chars_per_batch)


def chapter_assist_insert_markers(
    source_text: str,
    pattern: str,
    case_sensitive: bool,
    max_chars_per_batch: int,
) -> tuple[str, str, str, str]:
    text = str(source_text or "")
    regex_text = str(pattern or "").strip()
    if not regex_text:
        chunk_preview = _build_chunk_preview_markdown(text, max_chars_per_batch)
        return text, "### Regex Matches\n\nNo pattern set.", chunk_preview, "No changes made."
    try:
        regex = _compile_user_regex(regex_text, case_sensitive=case_sensitive)
    except re.error as exc:
        chunk_preview = _build_chunk_preview_markdown(text, max_chars_per_batch)
        return text, f"### Regex Matches\n\nInvalid regex: `{exc}`", chunk_preview, "No changes made."

    matches = _find_chapter_line_matches(text, regex)
    if not matches:
        chunk_preview = _build_chunk_preview_markdown(text, max_chars_per_batch)
        return text, _render_match_preview(matches), chunk_preview, "No matches found. No chapter markers inserted."

    inserted = 0
    updated_lines: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        if regex.search(line):
            stripped = line.lstrip()
            indent = line[: len(line) - len(stripped)]
            if re.match(r"(?i)^\s*\[CHAPTER\]\b", line):
                updated_lines.append(line)
                continue
            updated_lines.append(f"{indent}[CHAPTER] {stripped}".rstrip())
            inserted += 1
        else:
            updated_lines.append(line)
    updated_text = "\n".join(updated_lines)

    match_preview = _render_match_preview(matches)
    chunk_preview = _build_chunk_preview_markdown(updated_text, max_chars_per_batch)
    status = f"Inserted {inserted} chapter marker(s)."
    return updated_text, match_preview, chunk_preview, status


def _copy_text_or_blank(value: str) -> str:
    return str(value or "")


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
    continuation_chain: bool,
    continuation_anchor_seconds: float,
    stop_after_batch: int,
) -> Generator[tuple[str, str | None, str | None, str, str], None, None]:
    log_lines: list[str] = []
    telemetry_cache: dict[str, object] = {}
    run_root_path: Path | None = None
    job_root: Path | None = None
    expected_output: Path | None = None
    command: list[str] | None = None
    started: float | None = None
    process_pid: int | None = None

    def push_log(line: str) -> str:
        log_lines.append(line.rstrip("\n"))
        if len(log_lines) > MAX_LOG_LINES:
            del log_lines[: len(log_lines) - MAX_LOG_LINES]
        return "\n".join(log_lines)

    def telemetry_html() -> str:
        return _render_telemetry_panel(
            log_lines=log_lines,
            command=command,
            job_root=job_root,
            run_root=run_root_path,
            expected_output=expected_output,
            device_hint=device,
            started_at=started,
            process_pid=process_pid,
            cache=telemetry_cache,
        )

    yield _build_status_message("Preparing job..."), None, None, "", telemetry_html()

    if not CLI_SCRIPT.exists():
        message = f"CLI script not found: {CLI_SCRIPT}"
        yield _build_status_message("Unable to start", message), None, None, message, telemetry_html()
        return

    run_root_path = Path(run_root or str(DEFAULT_RUN_ROOT)).expanduser().resolve()
    run_root_path.mkdir(parents=True, exist_ok=True)
    job_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    job_root = run_root_path / "_gradio_jobs" / job_id
    staging_dir = job_root / "staging"
    staging_dir.mkdir(parents=True, exist_ok=True)

    command = [sys.executable, str(CLI_SCRIPT)]

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
            yield _build_status_message("Missing input", message), None, None, message, telemetry_html()
            return
        if not resume_state_path and not reference_audio_value:
            message = "Provide reference audio (upload file or path/URL) for a new run."
            yield _build_status_message("Missing reference audio", message), None, None, message, telemetry_html()
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
            yield _build_status_message("Missing reference transcript", message), None, None, message, telemetry_html()
            return
        if (
            continuation_chain
            and not resume_state_path
            and not reference_text.strip()
            and not reference_text_file_path
        ):
            message = (
                "Continuation chain mode requires a reference transcript for the initial anchor "
                "(reference text or transcript file)."
            )
            yield _build_status_message("Missing continuation transcript", message), None, None, message, telemetry_html()
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
        if continuation_chain:
            command.extend(
                [
                    "--continuation-chain",
                    "--continuation-anchor-seconds",
                    str(float(continuation_anchor_seconds)),
                ]
            )
        if stop_after_batch > 0:
            command.extend(["--stop-after-batch", str(int(stop_after_batch))])
    except Exception as exc:
        message = f"Input preparation failed: {exc}"
        yield _build_status_message("Unable to prepare job", message), None, None, message, telemetry_html()
        return

    push_log(f"$ {' '.join(command)}")
    yield _build_status_message("Running generation..."), None, None, "\n".join(log_lines), telemetry_html()

    started = time.time()
    process = subprocess.Popen(
        command,
        cwd=str(APP_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    process_pid = int(process.pid)

    if process.stdout:
        for line in process.stdout:
            logs_text = push_log(line.rstrip("\n"))
            yield _build_status_message("Running generation..."), None, None, logs_text, telemetry_html()

    exit_code = process.wait()
    elapsed = time.time() - started
    logs_text = "\n".join(log_lines)
    output_path = _find_output_path(expected_output, log_lines)

    if exit_code != 0:
        detail = f"CLI exited with code {exit_code}. Check logs below."
        yield _build_status_message("Generation failed", detail), None, None, logs_text, telemetry_html()
        return

    if output_path and output_path.exists():
        detail = f"Completed in {elapsed:.1f}s\n\nOutput: `{output_path}`"
        yield (
            _build_status_message("Generation complete", detail),
            str(output_path),
            str(output_path),
            logs_text,
            telemetry_html(),
        )
        return

    detail = (
        f"Completed in {elapsed:.1f}s, but output file was not auto-detected. "
        "Review logs for the final path."
    )
    yield _build_status_message("Generation complete", detail), None, None, logs_text, telemetry_html()


def build_demo() -> gr.Blocks:
    css = """
    .app-shell {max-width: 1200px; margin: 0 auto;}
    .mono-log textarea {font-family: Consolas, "Cascadia Mono", Menlo, monospace !important;}
    .telemetry-shell {
        border: 1px solid rgba(148, 163, 184, 0.35);
        background: linear-gradient(180deg, rgba(2, 6, 23, 0.92), rgba(15, 23, 42, 0.96));
        color: #e2e8f0;
        border-radius: 14px;
        padding: 12px 14px;
        font-family: Consolas, "Cascadia Mono", Menlo, monospace;
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.03);
    }
    .telemetry-header {
        display: flex;
        justify-content: space-between;
        align-items: baseline;
        gap: 12px;
        margin-bottom: 8px;
    }
    .telemetry-header h3 {
        margin: 0;
        font-size: 15px;
        letter-spacing: 0.03em;
        color: #bae6fd;
    }
    .telemetry-phase {
        font-size: 12px;
        color: #93c5fd;
        opacity: 0.95;
        margin-top: 2px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        max-width: 70ch;
    }
    .telemetry-uptime {
        font-size: 12px;
        color: #cbd5e1;
        background: rgba(30, 41, 59, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.22);
        border-radius: 999px;
        padding: 4px 8px;
    }
    .telemetry-meta {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
        gap: 6px 10px;
        margin-bottom: 8px;
        font-size: 11px;
        color: #cbd5e1;
        opacity: 0.9;
    }
    .telemetry-meta span {
        display: block;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .telemetry-note {
        margin: 0 0 8px 0;
        font-size: 11px;
        color: #fcd34d;
        background: rgba(120, 53, 15, 0.20);
        border: 1px solid rgba(251, 191, 36, 0.25);
        border-radius: 8px;
        padding: 6px 8px;
    }
    .telemetry-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
        gap: 10px;
    }
    .telemetry-card {
        border-radius: 12px;
        border: 1px solid rgba(148, 163, 184, 0.2);
        background: rgba(15, 23, 42, 0.7);
        padding: 10px;
    }
    .telemetry-card-title {
        color: #bfdbfe;
        font-size: 12px;
        margin-bottom: 6px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .telemetry-metric { margin-bottom: 8px; }
    .telemetry-metric:last-child { margin-bottom: 0; }
    .telemetry-row {
        display: flex;
        justify-content: space-between;
        gap: 8px;
        font-size: 11px;
        color: #e5e7eb;
    }
    .telemetry-row span:first-child { color: #cbd5e1; }
    .telemetry-row span:last-child {
        color: #f8fafc;
        text-align: right;
        max-width: 55%;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .telemetry-bar {
        margin-top: 4px;
        height: 8px;
        border-radius: 999px;
        background: rgba(51, 65, 85, 0.9);
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.12);
    }
    .telemetry-fill {
        height: 100%;
        border-radius: 999px;
        box-shadow: 0 0 10px rgba(56, 189, 248, 0.25);
        transition: width 0.16s ease-out;
    }
    .telemetry-subtle {
        margin-top: 2px;
        color: #94a3b8;
        font-size: 10px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .telemetry-metric.opacity-70 { opacity: 0.7; }
    """
    audio_types = sorted(REFERENCE_AUDIO_EXTENSIONS)
    voice_library_choices = _voice_dropdown_choices()
    chapter_regex_presets = [
        "Common Headings",
        "Chapter Number",
        "Part Number",
        "Month Day Heading",
        "Roman Numerals",
        "Custom Word + Number",
        "Custom Regex",
    ]

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
                with gr.Row():
                    voice_library_dropdown = gr.Dropdown(
                        label="Voice Library (/voices)",
                        choices=voice_library_choices,
                        value=None,
                        allow_custom_value=False,
                        scale=5,
                    )
                    voice_library_refresh_button = gr.Button("Refresh", scale=1, variant="secondary")
                with gr.Row():
                    voice_library_apply_button = gr.Button(
                        "Use Selected Voice",
                        variant="secondary",
                        scale=2,
                    )
                    voice_library_status = gr.Markdown(
                        f"Using `{_ensure_voices_dir()}` for voice files.",
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
                        choices=["moss-delay", "moss-local", "moss-ttsd", "qwen", "auto"],
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
                    continuation_chain = gr.Checkbox(
                        label="Continuation chain (best continuity, slower)",
                        value=False,
                    )
                    continuation_anchor_seconds = gr.Slider(
                        label="Continuation anchor seconds",
                        minimum=2.0,
                        maximum=30.0,
                        step=0.5,
                        value=DEFAULT_CONTINUATION_ANCHOR_SECONDS,
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
        telemetry = gr.HTML(
            value=_render_telemetry_panel(
                log_lines=[],
                command=None,
                job_root=None,
                run_root=DEFAULT_RUN_ROOT.resolve(),
                expected_output=None,
                device_hint="cuda:0",
                started_at=None,
                process_pid=None,
                cache={},
            )
        )
        audio_output = gr.Audio(label="Output Preview", type="filepath", interactive=False)
        file_output = gr.File(label="Download Output", interactive=False)
        logs = gr.Textbox(label="Run Log", lines=20, elem_classes=["mono-log"])

        with gr.Tabs():
            with gr.Tab("Voice Lab"):
                gr.Markdown(
                    "Preview and save a synthetic voice generated from a text description using "
                    "MOSS VoiceGenerator. Saved voices go to `/voices` as `.wav + .txt + .voice.json`."
                )
                voice_lab_status = gr.Markdown("### Voice Lab Ready")
                with gr.Row():
                    voice_lab_name = gr.Textbox(
                        label="Voice Name",
                        placeholder="e.g. warm_narrator_v1",
                        value="voice_lab_preview",
                    )
                    voice_lab_overwrite = gr.Checkbox(
                        label="Overwrite existing files",
                        value=True,
                    )
                voice_lab_description = gr.Textbox(
                    label="Voice Description",
                    lines=4,
                    placeholder=(
                        "Describe the voice style (tone, age, accent, pacing, texture, emotion). "
                        "Example: Calm middle-aged male narrator, warm tone, measured pacing, slight breathiness."
                    ),
                )
                voice_lab_preview_text = gr.Textbox(
                    label="Preview Text",
                    lines=4,
                    value=(
                        "This is a voice preview generated from a text description. "
                        "If the tone sounds right, save it and reuse it from the voices folder."
                    ),
                )
                with gr.Accordion("Voice Lab Advanced", open=False):
                    voice_lab_model_id = gr.Textbox(
                        label="VoiceGenerator Model ID",
                        value=DEFAULT_VOICEGEN_MODEL_ID,
                    )
                    with gr.Row():
                        voice_lab_device = gr.Textbox(label="Device", value="cuda:0")
                        voice_lab_dtype = gr.Dropdown(
                            label="DType",
                            choices=["float16", "bfloat16", "float32"],
                            value=DEFAULT_DTYPE,
                        )
                        voice_lab_attn = gr.Dropdown(
                            label="Attention",
                            choices=["sdpa", "flash_attention_2", "eager"],
                            value=DEFAULT_ATTN_IMPLEMENTATION,
                        )
                    voice_lab_max_new_tokens = gr.Slider(
                        label="Max new tokens",
                        minimum=256,
                        maximum=8192,
                        step=64,
                        value=2048,
                    )
                with gr.Row():
                    voice_lab_generate_button = gr.Button(
                        "Generate Preview + Save Voice",
                        variant="primary",
                    )
                    voice_lab_refresh_button = gr.Button(
                        "Refresh /voices",
                        variant="secondary",
                    )
                voice_lab_preview_audio = gr.Audio(
                    label="Voice Preview Audio",
                    type="filepath",
                    interactive=False,
                )
                voice_lab_saved_file = gr.File(
                    label="Saved Voice Preset (.voice.json)",
                    interactive=False,
                )
                voice_lab_log = gr.Textbox(
                    label="Voice Lab Log",
                    lines=10,
                    elem_classes=["mono-log"],
                )

            with gr.Tab("Chapter Assist"):
                gr.Markdown(
                    "Regex-assisted chapter tagging and chunk preview. Build a pattern, preview matches, "
                    "insert `[CHAPTER]` markers, then push the text back into the main Book Text box."
                )
                chapter_assist_status = gr.Markdown("### Chapter Assist Ready")
                chapter_assist_text = gr.Textbox(
                    label="Chapter Assist Working Text",
                    lines=14,
                    placeholder="Paste text here or load from the main Book Text field.",
                )
                with gr.Row():
                    chapter_assist_load_main_button = gr.Button("Load From Book Text", variant="secondary")
                    chapter_assist_apply_main_button = gr.Button("Apply Back To Book Text", variant="secondary")
                with gr.Row():
                    chapter_regex_preset = gr.Dropdown(
                        label="Regex Preset",
                        choices=chapter_regex_presets,
                        value="Common Headings",
                    )
                    chapter_custom_word = gr.Textbox(
                        label="Custom Heading Word",
                        placeholder="e.g. Episode, Section",
                        value="Section",
                    )
                    chapter_case_sensitive = gr.Checkbox(
                        label="Case-sensitive regex",
                        value=False,
                    )
                chapter_regex_pattern = gr.Textbox(
                    label="Regex Pattern",
                    lines=2,
                    value=_chapter_regex_template("Common Headings", "Section"),
                )
                chapter_custom_regex = gr.Textbox(
                    label="Custom Regex (used when preset = Custom Regex)",
                    lines=2,
                    placeholder=r"^\s*Book\s+\d+\b.*$",
                )
                with gr.Row():
                    chapter_build_pattern_button = gr.Button("Build Pattern", variant="secondary")
                    chapter_preview_button = gr.Button("Preview Matches + Chunking", variant="secondary")
                    chapter_insert_button = gr.Button("Insert [CHAPTER] Markers", variant="primary")
                chapter_match_preview = gr.Markdown("### Regex Matches\n\nNo preview yet.")
                chapter_chunk_preview = gr.Markdown("### Chunk Preview\n\nNo preview yet.")

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
                continuation_chain,
                continuation_anchor_seconds,
                stop_after_batch,
            ],
            outputs=[status, audio_output, file_output, logs, telemetry],
        )

        clear_button.click(
            fn=lambda: (
                "### Ready",
                None,
                None,
                "",
                _render_telemetry_panel(
                    log_lines=[],
                    command=None,
                    job_root=None,
                    run_root=DEFAULT_RUN_ROOT.resolve(),
                    expected_output=None,
                    device_hint="cuda:0",
                    started_at=None,
                    process_pid=None,
                    cache={},
                ),
            ),
            outputs=[status, audio_output, file_output, logs, telemetry],
        )

        voice_library_refresh_button.click(
            fn=refresh_voice_library_dropdown,
            outputs=[voice_library_dropdown, voice_library_status],
        )
        voice_lab_refresh_button.click(
            fn=refresh_voice_library_dropdown,
            outputs=[voice_library_dropdown, voice_library_status],
        )
        voice_library_apply_button.click(
            fn=apply_voice_library_selection,
            inputs=[
                voice_library_dropdown,
                reference_audio_path,
                reference_text,
                voice_lab_name,
                voice_lab_description,
            ],
            outputs=[
                reference_audio_path,
                reference_text,
                voice_lab_name,
                voice_lab_description,
                voice_library_status,
            ],
        )
        voice_library_dropdown.change(
            fn=apply_voice_library_selection,
            inputs=[
                voice_library_dropdown,
                reference_audio_path,
                reference_text,
                voice_lab_name,
                voice_lab_description,
            ],
            outputs=[
                reference_audio_path,
                reference_text,
                voice_lab_name,
                voice_lab_description,
                voice_library_status,
            ],
        )
        voice_lab_generate_button.click(
            fn=generate_voice_from_description,
            inputs=[
                voice_lab_name,
                voice_lab_description,
                voice_lab_preview_text,
                voice_lab_model_id,
                voice_lab_device,
                voice_lab_dtype,
                voice_lab_attn,
                voice_lab_max_new_tokens,
                voice_lab_overwrite,
            ],
            outputs=[
                voice_lab_status,
                voice_lab_preview_audio,
                voice_lab_saved_file,
                voice_lab_log,
                voice_library_dropdown,
                voice_library_status,
                reference_audio_path,
                reference_text,
            ],
        )

        chapter_assist_load_main_button.click(
            fn=_copy_text_or_blank,
            inputs=[text_input],
            outputs=[chapter_assist_text],
        )
        chapter_assist_apply_main_button.click(
            fn=_copy_text_or_blank,
            inputs=[chapter_assist_text],
            outputs=[text_input],
        )
        chapter_build_pattern_button.click(
            fn=lambda preset, custom_word, custom_regex: (
                lambda pattern_msg: (
                    pattern_msg[0],
                    _build_status_message("Chapter Assist", pattern_msg[1]),
                )
            )(build_chapter_regex_pattern(preset, custom_word, custom_regex)),
            inputs=[chapter_regex_preset, chapter_custom_word, chapter_custom_regex],
            outputs=[chapter_regex_pattern, chapter_assist_status],
        )
        chapter_regex_preset.change(
            fn=lambda preset, custom_word, custom_regex: (
                lambda pattern_msg: (
                    pattern_msg[0],
                    _build_status_message("Chapter Assist", pattern_msg[1]),
                )
            )(build_chapter_regex_pattern(preset, custom_word, custom_regex)),
            inputs=[chapter_regex_preset, chapter_custom_word, chapter_custom_regex],
            outputs=[chapter_regex_pattern, chapter_assist_status],
        )
        chapter_preview_button.click(
            fn=lambda text, pattern, case_sensitive, max_chars: (
                lambda previews: (
                    previews[0],
                    previews[1],
                    _build_status_message("Chapter Assist", "Preview refreshed."),
                )
            )(chapter_assist_preview(text, pattern, case_sensitive, int(max_chars))),
            inputs=[chapter_assist_text, chapter_regex_pattern, chapter_case_sensitive, max_chars_per_batch],
            outputs=[chapter_match_preview, chapter_chunk_preview, chapter_assist_status],
        )
        chapter_insert_button.click(
            fn=lambda text, pattern, case_sensitive, max_chars: (
                lambda result: (
                    result[0],
                    result[1],
                    result[2],
                    _build_status_message("Chapter Assist", result[3]),
                )
            )(chapter_assist_insert_markers(text, pattern, case_sensitive, int(max_chars))),
            inputs=[chapter_assist_text, chapter_regex_pattern, chapter_case_sensitive, max_chars_per_batch],
            outputs=[
                chapter_assist_text,
                chapter_match_preview,
                chapter_chunk_preview,
                chapter_assist_status,
            ],
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
