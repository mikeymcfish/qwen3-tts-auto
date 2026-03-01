from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import soundfile as sf


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from audiobook_qwen3 import (  # noqa: E402
    build_ffmetadata_with_chapters,
    chapter_time_entries_from_batches,
    compute_inter_batch_pause_samples,
    order_part_paths_by_batch_number,
    resolve_paths,
    sanitize_chapter_title,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair MP3 chapter metadata from run artifacts (session_state.json + parts/*.wav). "
            "Useful when existing chapter timecodes are off."
        )
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--run-dir", type=Path, help="Run directory containing session_state.json and parts/.")
    source_group.add_argument("--state", type=Path, help="Path to session_state.json.")

    parser.add_argument(
        "--input-mp3",
        type=Path,
        help="Existing MP3 to patch. Default: output_audio/output_wav from state if .mp3.",
    )
    parser.add_argument(
        "--output-mp3",
        type=Path,
        help="Patched MP3 path. Default: <input>_chapters_fixed.mp3 (or in-place when --overwrite).",
    )
    parser.add_argument(
        "--ffmeta-out",
        type=Path,
        help="Where to write repaired ffmetadata. Default: <run_dir>/chapters.repaired.ffmeta",
    )
    parser.add_argument(
        "--offset-ms",
        type=int,
        default=0,
        help=(
            "Shift every chapter start by this many milliseconds after recompute. "
            "Positive shifts later, negative shifts earlier."
        ),
    )
    parser.add_argument(
        "--pause-ms",
        type=int,
        help="Override pause-ms instead of state value.",
    )
    parser.add_argument(
        "--chapter-pause-ms",
        type=int,
        help="Override chapter-pause-ms instead of state value.",
    )
    parser.add_argument(
        "--cbr-kbps",
        type=int,
        help=(
            "Re-encode output MP3 at constant bitrate (e.g. 128, 192, 256, 320) "
            "instead of stream-copy remux. Useful for players with VBR chapter seek issues."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite input MP3 in-place.",
    )
    parser.add_argument(
        "--write-ffmeta-only",
        action="store_true",
        help="Only write ffmetadata, do not remux MP3.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without writing/remuxing.",
    )
    return parser.parse_args()


def _load_state_from_args(args: argparse.Namespace) -> tuple[Path, Path, dict[str, Any]]:
    if args.state:
        state_path = args.state.expanduser().resolve()
        if not state_path.exists():
            raise RuntimeError(f"State file not found: {state_path}")
        run_dir = state_path.parent
    else:
        run_dir = args.run_dir.expanduser().resolve()
        state_path = run_dir / "session_state.json"
        if not state_path.exists():
            raise RuntimeError(f"session_state.json not found in run dir: {run_dir}")

    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"Could not parse state file {state_path}: {exc}") from exc

    state_run_dir_raw = state.get("run_dir")
    if state_run_dir_raw:
        try:
            state_run_dir = Path(str(state_run_dir_raw)).expanduser().resolve()
            if state_run_dir.exists():
                run_dir = state_run_dir
        except Exception:
            pass

    return run_dir, state_path, state


def _coerce_positive_int_list(raw: Any) -> list[int]:
    values: list[int] = []
    if not isinstance(raw, list):
        return values
    seen: set[int] = set()
    for item in raw:
        try:
            num = int(item)
        except (TypeError, ValueError):
            continue
        if num < 1 or num in seen:
            continue
        seen.add(num)
        values.append(num)
    return values


def _coerce_title_map(raw: Any) -> dict[int, str]:
    title_map: dict[int, str] = {}
    if not isinstance(raw, dict):
        return title_map
    for key, value in raw.items():
        try:
            batch_num = int(key)
        except (TypeError, ValueError):
            continue
        title = sanitize_chapter_title(str(value))
        if batch_num >= 1 and title:
            title_map[batch_num] = title
    return title_map


def _resolve_input_mp3(args: argparse.Namespace, state: dict[str, Any]) -> Path:
    if args.input_mp3:
        candidate = args.input_mp3.expanduser().resolve()
    else:
        output_audio = state.get("output_audio") or state.get("output_wav")
        if not output_audio:
            raise RuntimeError("No --input-mp3 provided and state has no output_audio/output_wav.")
        candidate = Path(str(output_audio)).expanduser().resolve()
    if not candidate.exists() or not candidate.is_file():
        raise RuntimeError(f"Input MP3 not found: {candidate}")
    if candidate.suffix.lower() != ".mp3":
        raise RuntimeError(f"Input file is not .mp3: {candidate}")
    return candidate


def _resolve_output_mp3(args: argparse.Namespace, input_mp3: Path) -> Path:
    if args.overwrite:
        return input_mp3
    if args.output_mp3:
        return args.output_mp3.expanduser().resolve()
    return input_mp3.with_name(f"{input_mp3.stem}_chapters_fixed.mp3").resolve()


def _compute_part_timing(
    part_paths: list[Path],
    chapter_batch_numbers: list[int],
    pause_ms: int,
    chapter_pause_ms: int,
    part_batch_numbers: list[int] | None,
) -> tuple[int, float, list[int]]:
    if not part_paths:
        raise RuntimeError("No part files found.")

    infos = [sf.info(str(path)) for path in part_paths]
    sample_rates = {int(info.samplerate) for info in infos}
    if len(sample_rates) != 1:
        raise RuntimeError(f"Part files have mixed sample rates: {sorted(sample_rates)}")
    sample_rate = sample_rates.pop()
    if sample_rate <= 0:
        raise RuntimeError("Invalid sample rate from part files.")

    part_lengths = [int(max(0, info.frames)) for info in infos]
    batch_numbers_in_order = (
        part_batch_numbers if part_batch_numbers and len(part_batch_numbers) == len(part_paths) else list(range(1, len(part_paths) + 1))
    )
    chapter_set = set(chapter_batch_numbers)
    base_pause_samples = int(sample_rate * (max(0, int(pause_ms)) / 1000.0))
    chapter_pause_samples = int(sample_rate * (max(0, int(chapter_pause_ms)) / 1000.0))

    part_start_samples: list[int] = []
    cursor = 0
    for idx, length in enumerate(part_lengths):
        part_start_samples.append(cursor)
        cursor += int(length)
        if idx < len(part_lengths) - 1:
            next_batch_number = int(batch_numbers_in_order[idx + 1])
            cursor += compute_inter_batch_pause_samples(
                base_pause_samples=base_pause_samples,
                chapter_pause_samples=chapter_pause_samples,
                next_batch_number=next_batch_number,
                chapter_batch_numbers=chapter_set,
            )

    total_duration_seconds = float(cursor) / float(sample_rate)
    return sample_rate, total_duration_seconds, part_start_samples


def _apply_offset_to_entries(
    entries: list[tuple[float, str | None]],
    *,
    offset_ms: int,
    total_duration_seconds: float,
) -> list[tuple[float, str | None]]:
    if offset_ms == 0:
        return entries
    total_ms = max(1, int(round(float(total_duration_seconds) * 1000.0)))
    shifted: list[tuple[float, str | None]] = []
    for start_seconds, title in entries:
        start_ms = int(round(start_seconds * 1000.0)) + int(offset_ms)
        start_ms = max(0, min(total_ms - 1, start_ms))
        shifted.append((start_ms / 1000.0, title))
    return shifted


def _remux_mp3_with_ffmeta(
    input_mp3: Path,
    output_mp3: Path,
    ffmeta_path: Path,
    overwrite: bool,
    cbr_kbps: int | None = None,
) -> None:
    ffmpeg_path = shutil.which("ffmpeg")
    if not ffmpeg_path:
        raise RuntimeError("ffmpeg not found in PATH.")

    target = output_mp3
    temp_target: Path | None = None
    if overwrite and input_mp3.resolve() == output_mp3.resolve():
        temp_target = output_mp3.with_name(output_mp3.stem + ".chapters_tmp.mp3")
        target = temp_target

    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        ffmpeg_path,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_mp3),
        "-f",
        "ffmetadata",
        "-i",
        str(ffmeta_path),
        "-map_metadata",
        "1",
        "-map_chapters",
        "1",
        "-id3v2_version",
        "3",
        "-map",
        "0:a",
    ]
    if cbr_kbps is not None:
        cmd.extend(
            [
                "-codec:a",
                "libmp3lame",
                "-b:a",
                f"{int(cbr_kbps)}k",
                # disable Xing VBR metadata header for stricter CBR behavior
                "-write_xing",
                "0",
            ]
        )
    else:
        cmd.extend(["-c", "copy"])
    cmd.append(str(target))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise RuntimeError(f"ffmpeg remux failed (exit {proc.returncode}). {detail}")

    if temp_target is not None:
        temp_target.replace(output_mp3)


def main() -> int:
    args = parse_args()
    if args.cbr_kbps is not None and int(args.cbr_kbps) <= 0:
        print("ERROR: --cbr-kbps must be a positive integer.", file=sys.stderr)
        return 2
    try:
        run_dir, state_path, state = _load_state_from_args(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    part_files_raw = state.get("part_files", [])
    if not isinstance(part_files_raw, list) or not part_files_raw:
        print("ERROR: state has no part_files.", file=sys.stderr)
        return 2

    try:
        resolved_parts = resolve_paths(run_dir, [str(item) for item in part_files_raw])
    except Exception as exc:
        print(f"ERROR: could not resolve part files: {exc}", file=sys.stderr)
        return 2
    for path in resolved_parts:
        if not path.exists() or not path.is_file():
            print(f"ERROR: missing part file: {path}", file=sys.stderr)
            return 2

    ordered_parts, ordered_batch_numbers = order_part_paths_by_batch_number(resolved_parts)
    chapter_batch_numbers = _coerce_positive_int_list(state.get("chapter_batch_numbers", []))
    chapter_titles_by_batch = _coerce_title_map(state.get("chapter_titles_by_batch", {}))
    if not chapter_batch_numbers:
        print("ERROR: state has no chapter_batch_numbers to repair.", file=sys.stderr)
        return 2

    pause_ms = int(args.pause_ms if args.pause_ms is not None else state.get("pause_ms", 0))
    chapter_pause_ms = int(
        args.chapter_pause_ms if args.chapter_pause_ms is not None else state.get("chapter_pause_ms", 0)
    )

    try:
        sample_rate, total_duration_seconds, part_start_samples = _compute_part_timing(
            part_paths=ordered_parts,
            chapter_batch_numbers=chapter_batch_numbers,
            pause_ms=pause_ms,
            chapter_pause_ms=chapter_pause_ms,
            part_batch_numbers=ordered_batch_numbers,
        )
    except Exception as exc:
        print(f"ERROR: could not compute part timing: {exc}", file=sys.stderr)
        return 2

    chapter_time_entries = chapter_time_entries_from_batches(
        chapter_batch_numbers=chapter_batch_numbers,
        part_start_samples=part_start_samples,
        sample_rate=sample_rate,
        part_batch_numbers=ordered_batch_numbers,
    )
    chapter_entries_with_titles: list[tuple[float, str | None]] = [
        (start_seconds, chapter_titles_by_batch.get(batch_number))
        for start_seconds, batch_number in chapter_time_entries
    ]
    chapter_entries_with_titles = _apply_offset_to_entries(
        chapter_entries_with_titles,
        offset_ms=int(args.offset_ms),
        total_duration_seconds=total_duration_seconds,
    )
    ffmeta_text = build_ffmetadata_with_chapters(
        chapter_entries=chapter_entries_with_titles,
        total_duration_seconds=total_duration_seconds,
    )
    if "[CHAPTER]" not in ffmeta_text:
        print("ERROR: repaired ffmetadata contains no chapter blocks.", file=sys.stderr)
        return 2

    ffmeta_out = (
        args.ffmeta_out.expanduser().resolve()
        if args.ffmeta_out
        else (run_dir / "chapters.repaired.ffmeta").resolve()
    )
    if args.dry_run:
        print(f"[dry-run] state={state_path}")
        print(f"[dry-run] run_dir={run_dir}")
        print(f"[dry-run] parts={len(ordered_parts)} sample_rate={sample_rate}")
        print(f"[dry-run] chapters={len(chapter_entries_with_titles)} pause_ms={pause_ms} chapter_pause_ms={chapter_pause_ms}")
        print(f"[dry-run] ffmeta_out={ffmeta_out}")
        if args.cbr_kbps is not None:
            print(f"[dry-run] cbr_kbps={int(args.cbr_kbps)}")
        if not args.write_ffmeta_only:
            try:
                input_mp3 = _resolve_input_mp3(args, state)
                output_mp3 = _resolve_output_mp3(args, input_mp3)
                print(f"[dry-run] input_mp3={input_mp3}")
                print(f"[dry-run] output_mp3={output_mp3}")
            except Exception as exc:
                print(f"[dry-run] input/output mp3 resolution failed: {exc}")
        return 0

    ffmeta_out.parent.mkdir(parents=True, exist_ok=True)
    ffmeta_out.write_text(ffmeta_text, encoding="utf-8", newline="\n")
    print(f"Wrote repaired ffmetadata: {ffmeta_out}")
    print(
        f"Computed {len(chapter_entries_with_titles)} chapter entries from {len(ordered_parts)} parts "
        f"(sample_rate={sample_rate}, pause_ms={pause_ms}, chapter_pause_ms={chapter_pause_ms}, offset_ms={args.offset_ms})."
    )

    if args.write_ffmeta_only:
        return 0

    try:
        input_mp3 = _resolve_input_mp3(args, state)
        output_mp3 = _resolve_output_mp3(args, input_mp3)
        _remux_mp3_with_ffmeta(
            input_mp3=input_mp3,
            output_mp3=output_mp3,
            ffmeta_path=ffmeta_out,
            overwrite=bool(args.overwrite),
            cbr_kbps=int(args.cbr_kbps) if args.cbr_kbps is not None else None,
        )
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Patched MP3 with repaired chapters: {output_mp3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
