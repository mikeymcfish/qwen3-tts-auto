from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Conservative reference-audio cleanup for voice cloning. "
            "Uses ffmpeg filters (denoise, trim silence, normalize loudness, resample)."
        )
    )
    parser.add_argument("inputs", nargs="+", help="Input audio file(s).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output folder for processed WAV files (used when --output-file is not set).",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        help="Output WAV path for a single input file.",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_prep",
        help="Suffix added to output filename when auto-generating outputs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files.",
    )
    parser.add_argument(
        "--ffmpeg-bin",
        type=str,
        default="ffmpeg",
        help="ffmpeg executable path/name.",
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=24000,
        help="Output sample rate for clone-ready WAVs (default: 24000).",
    )
    parser.add_argument(
        "--channels",
        type=int,
        choices=(1, 2),
        default=1,
        help="Output channel count (default: 1/mono).",
    )
    parser.add_argument(
        "--trim-silence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trim leading/trailing silence (default: on).",
    )
    parser.add_argument(
        "--trim-start-threshold-db",
        type=float,
        default=-40.0,
        help="Leading silence threshold in dB (default: -40).",
    )
    parser.add_argument(
        "--trim-stop-threshold-db",
        type=float,
        default=-42.0,
        help="Trailing silence threshold in dB (default: -42).",
    )
    parser.add_argument(
        "--trim-start-duration",
        type=float,
        default=0.08,
        help="Seconds of non-silence required to stop start trim (default: 0.08).",
    )
    parser.add_argument(
        "--trim-stop-duration",
        type=float,
        default=0.15,
        help="Seconds of silence required to trim tail (default: 0.15).",
    )
    parser.add_argument(
        "--highpass-hz",
        type=float,
        default=70.0,
        help="High-pass filter cutoff in Hz (default: 70). Set 0 to disable.",
    )
    parser.add_argument(
        "--lowpass-hz",
        type=float,
        default=0.0,
        help="Low-pass filter cutoff in Hz. Set 0 to disable (default).",
    )
    parser.add_argument(
        "--denoise",
        choices=("none", "afftdn", "arnndn"),
        default="afftdn",
        help="Denoise method (default: afftdn). arnndn requires a model path.",
    )
    parser.add_argument(
        "--afftdn-noise-reduction",
        type=float,
        default=8.0,
        help="afftdn noise reduction amount (default: 8).",
    )
    parser.add_argument(
        "--afftdn-noise-floor",
        type=float,
        default=-35.0,
        help="afftdn noise floor dB (default: -35).",
    )
    parser.add_argument(
        "--arnndn-model",
        type=Path,
        help="Path to an RNNoise model file for ffmpeg arnndn.",
    )
    parser.add_argument(
        "--arnndn-mix",
        type=float,
        default=1.0,
        help="arnndn wet mix (0..1, default: 1.0).",
    )
    parser.add_argument(
        "--compressor",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply light speech compression before loudnorm (default: off).",
    )
    parser.add_argument(
        "--compressor-threshold",
        type=float,
        default=0.12,
        help="acompressor threshold in linear amplitude (default: 0.12).",
    )
    parser.add_argument(
        "--compressor-ratio",
        type=float,
        default=2.0,
        help="acompressor ratio (default: 2.0).",
    )
    parser.add_argument(
        "--loudnorm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply EBU R128 loudness normalization (default: on).",
    )
    parser.add_argument(
        "--target-lufs",
        type=float,
        default=-20.0,
        help="Target integrated loudness in LUFS (default: -20).",
    )
    parser.add_argument(
        "--target-lra",
        type=float,
        default=7.0,
        help="Target loudness range for loudnorm (default: 7).",
    )
    parser.add_argument(
        "--true-peak-db",
        type=float,
        default=-1.5,
        help="True peak target for loudnorm in dBTP (default: -1.5).",
    )
    parser.add_argument(
        "--limiter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Apply a final limiter after loudnorm (default: on).",
    )
    parser.add_argument(
        "--limiter-limit",
        type=float,
        default=0.95,
        help="alimiter ceiling (0.0625..1.0, default: 0.95).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print ffmpeg command(s) without executing.",
    )
    parser.add_argument(
        "--print-filter",
        action="store_true",
        help="Print the generated ffmpeg filter chain.",
    )
    return parser.parse_args()


def _format_db(value: float) -> str:
    return f"{float(value):g}dB"


def _ffmpeg_filter_escape(value: str) -> str:
    # ffmpeg filter args treat ':' and '\' specially.
    normalized = str(value).replace("\\", "/")
    normalized = normalized.replace(":", r"\:")
    normalized = normalized.replace("'", r"\'")
    return f"'{normalized}'"


def build_filter_chain(args: argparse.Namespace) -> str:
    filters: list[str] = []

    if args.highpass_hz and float(args.highpass_hz) > 0:
        filters.append(f"highpass=f={float(args.highpass_hz):g}")
    if args.lowpass_hz and float(args.lowpass_hz) > 0:
        filters.append(f"lowpass=f={float(args.lowpass_hz):g}")

    if args.denoise == "afftdn":
        filters.append(
            "afftdn="
            f"nr={float(args.afftdn_noise_reduction):g}:"
            f"nf={float(args.afftdn_noise_floor):g}:"
            "tn=1"
        )
    elif args.denoise == "arnndn":
        if not args.arnndn_model:
            raise ValueError("--arnndn-model is required when --denoise arnndn is used.")
        filters.append(
            "arnndn="
            f"m={_ffmpeg_filter_escape(str(args.arnndn_model))}:"
            f"mix={float(args.arnndn_mix):g}"
        )

    if args.trim_silence:
        filters.append(
            "silenceremove="
            "start_periods=1:"
            f"start_duration={float(args.trim_start_duration):g}:"
            f"start_threshold={_format_db(args.trim_start_threshold_db)}:"
            "stop_periods=-1:"
            f"stop_duration={float(args.trim_stop_duration):g}:"
            f"stop_threshold={_format_db(args.trim_stop_threshold_db)}"
        )

    if args.compressor:
        filters.append(
            "acompressor="
            f"threshold={float(args.compressor_threshold):g}:"
            f"ratio={float(args.compressor_ratio):g}:"
            "attack=5:"
            "release=70:"
            "makeup=1:"
            "detection=rms:"
            "mix=1"
        )

    if args.loudnorm:
        filters.append(
            "loudnorm="
            f"I={float(args.target_lufs):g}:"
            f"LRA={float(args.target_lra):g}:"
            f"TP={float(args.true_peak_db):g}:"
            "linear=true"
        )

    if args.limiter:
        filters.append(
            "alimiter="
            f"limit={float(args.limiter_limit):g}:"
            "attack=5:"
            "release=50:"
            "level=true"
        )

    return ",".join(filters)


def build_output_path(
    input_path: Path,
    output_dir: Path | None,
    output_file: Path | None,
    suffix: str,
) -> Path:
    if output_file is not None:
        return output_file
    target_dir = output_dir if output_dir is not None else input_path.parent
    return target_dir / f"{input_path.stem}{suffix}.wav"


def resolve_ffmpeg(bin_name: str) -> str:
    ffmpeg_path = shutil.which(bin_name)
    if ffmpeg_path:
        return ffmpeg_path
    candidate = Path(bin_name)
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(f"ffmpeg not found: {bin_name}")


def build_ffmpeg_command(
    ffmpeg_bin: str,
    input_path: Path,
    output_path: Path,
    filter_chain: str,
    args: argparse.Namespace,
) -> list[str]:
    cmd: list[str] = [
        ffmpeg_bin,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if args.overwrite else "-n",
        "-i",
        str(input_path),
        "-vn",
        "-ac",
        str(int(args.channels)),
        "-ar",
        str(int(args.target_sample_rate)),
        "-sample_fmt",
        "s16",
    ]
    if filter_chain:
        cmd.extend(["-af", filter_chain])
    cmd.extend(
        [
            "-map_metadata",
            "-1",
            "-codec:a",
            "pcm_s16le",
            str(output_path),
        ]
    )
    return cmd


def main() -> int:
    args = parse_args()

    if args.output_file and len(args.inputs) != 1:
        print("ERROR: --output-file can only be used with a single input.", file=sys.stderr)
        return 2

    if args.output_file and args.output_file.suffix.lower() != ".wav":
        print("ERROR: --output-file must end in .wav.", file=sys.stderr)
        return 2

    try:
        ffmpeg_bin = resolve_ffmpeg(args.ffmpeg_bin)
        filter_chain = build_filter_chain(args)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    if args.print_filter:
        print(f"filter_chain={filter_chain}")

    failures = 0
    for raw_input in args.inputs:
        input_path = Path(raw_input).expanduser().resolve()
        if not input_path.exists():
            print(f"ERROR: input not found: {input_path}", file=sys.stderr)
            failures += 1
            continue

        output_path = build_output_path(
            input_path=input_path,
            output_dir=args.output_dir,
            output_file=args.output_file,
            suffix=args.suffix,
        ).expanduser().resolve()

        if output_path.suffix.lower() != ".wav":
            print(f"ERROR: output must be .wav: {output_path}", file=sys.stderr)
            failures += 1
            continue

        output_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = build_ffmpeg_command(ffmpeg_bin, input_path, output_path, filter_chain, args)
        print(f"[preprocess] {input_path.name} -> {output_path}")
        if args.dry_run:
            print(subprocess.list2cmdline(cmd))
            continue
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            print(f"ERROR: ffmpeg failed for {input_path}: exit {exc.returncode}", file=sys.stderr)
            failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
