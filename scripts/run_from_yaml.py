from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path
from typing import Any

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - import is validated at runtime
    yaml = None


AUDIOBOOK_PATH_KEYS = {
    "text_file",
    "reference_audio",
    "speaker2_reference_audio",
    "reference_text_file",
    "speaker2_reference_text_file",
    "resume_state",
    "output",
    "run_root",
}

PREPROCESS_PATH_KEYS = {"arnndn_model", "input", "output", "output_dir"}

PREPROCESS_META_KEYS = {"input", "output", "assign_to", "enabled", "name", "notes"}
AUDIOBOOK_META_KEYS = {"enabled", "script", "python", "args", "notes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run audiobook_qwen3.py from a YAML config and optionally preprocess "
            "reference audio files first."
        )
    )
    parser.add_argument(
        "config",
        type=Path,
        nargs="?",
        default=Path("run_settings.yaml"),
        help="Path to YAML config file (default: run_settings.yaml).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print generated commands without running them.",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip preprocess stage even if enabled in YAML.",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Skip audiobook generation stage (run preprocess only).",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required for scripts/run_from_yaml.py. Install with `pip install PyYAML`."
        )
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Config file not found: {path}") from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise RuntimeError(f"Config root must be a mapping/object: {path}")
    return data


def looks_like_url_or_data(value: str) -> bool:
    lowered = str(value).lower()
    return (
        "://" in lowered
        or lowered.startswith("data:")
        or lowered.startswith("hf://")
        or lowered.startswith("s3://")
    )


def resolve_config_path_value(config_dir: Path, key: str, value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if key not in AUDIOBOOK_PATH_KEYS and key not in PREPROCESS_PATH_KEYS:
        return value
    if key in {"reference_audio", "speaker2_reference_audio"} and looks_like_url_or_data(value):
        return value
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    return str((config_dir / candidate).resolve())


def build_flag(flag_name: str) -> str:
    return f"--{flag_name.replace('_', '-')}"


def append_option(
    command: list[str],
    key: str,
    value: Any,
    *,
    config_dir: Path,
    path_keys: set[str],
) -> None:
    if value is None:
        return
    flag = build_flag(key)

    if isinstance(value, bool):
        if value:
            command.append(flag)
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, (dict, list, tuple)):
                raise RuntimeError(f"Unsupported nested list value for '{key}'.")
            command.extend([flag, str(resolve_config_path_value(config_dir, key, item))])
        return

    if isinstance(value, dict):
        raise RuntimeError(f"Unsupported nested mapping value for '{key}'.")

    resolved = (
        resolve_config_path_value(config_dir, key, value) if key in path_keys else value
    )
    command.extend([flag, str(resolved)])


def command_str(cmd: list[str]) -> str:
    return subprocess.list2cmdline(cmd)


def run_command(cmd: list[str], *, cwd: Path, dry_run: bool) -> int:
    print(command_str(cmd))
    if dry_run:
        return 0
    completed = subprocess.run(cmd, cwd=str(cwd))
    return int(completed.returncode)


def build_preprocess_command(
    repo_root: Path,
    config_dir: Path,
    defaults: dict[str, Any],
    item: dict[str, Any],
) -> tuple[list[str], str | None, str]:
    if "input" not in item:
        raise RuntimeError("Each preprocess file entry must include 'input'.")

    merged = copy.deepcopy(defaults)
    for key, value in item.items():
        if key in PREPROCESS_META_KEYS:
            continue
        merged[key] = value

    input_value = resolve_config_path_value(config_dir, "input", item["input"])
    input_path = Path(str(input_value))

    output_value = item.get("output")
    if output_value is not None:
        output_path_str = str(resolve_config_path_value(config_dir, "output", output_value))
    else:
        output_dir_value = merged.get("output_dir")
        suffix_value = str(merged.get("suffix", "_prep"))
        output_dir = (
            Path(str(resolve_config_path_value(config_dir, "output_dir", output_dir_value)))
            if output_dir_value is not None
            else input_path.parent
        )
        output_path_str = str((output_dir / f"{input_path.stem}{suffix_value}.wav").resolve())

    script_path = repo_root / "scripts" / "preprocess_reference_audio.py"
    cmd: list[str] = [sys.executable, str(script_path), str(input_path), "--output-file", output_path_str]

    for key, value in merged.items():
        if key in {"output_dir", "suffix", "dry_run"}:
            continue
        append_option(
            cmd,
            key,
            value,
            config_dir=config_dir,
            path_keys=PREPROCESS_PATH_KEYS,
        )

    assign_to = item.get("assign_to")
    if assign_to is not None:
        assign_to = str(assign_to)
    return cmd, assign_to, output_path_str


def resolve_audiobook_config(config: dict[str, Any]) -> dict[str, Any]:
    if "audiobook" in config:
        section = config["audiobook"]
    elif "run" in config:
        section = config["run"]
    else:
        section = {}
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise RuntimeError("'audiobook' config must be a mapping/object.")
    if "args" in section:
        args_obj = section.get("args")
        if args_obj is None:
            return {}
        if not isinstance(args_obj, dict):
            raise RuntimeError("'audiobook.args' must be a mapping/object.")
        merged = dict(section)
        merged["args"] = dict(args_obj)
        return merged
    return {"args": dict(section), "enabled": True}


def main() -> int:
    cli = parse_args()
    config_path = cli.config.expanduser().resolve()
    config_dir = config_path.parent
    repo_root = Path(__file__).resolve().parents[1]

    try:
        config = load_yaml(config_path)
    except RuntimeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    audiobook_section = resolve_audiobook_config(config)
    audiobook_args = dict(audiobook_section.get("args") or {})
    preprocess_section = config.get("preprocess") or {}
    if preprocess_section and not isinstance(preprocess_section, dict):
        print("ERROR: 'preprocess' config must be a mapping/object.", file=sys.stderr)
        return 2

    if not cli.skip_preprocess and bool(preprocess_section.get("enabled", False)):
        defaults = preprocess_section.get("defaults") or {}
        files = preprocess_section.get("files") or []
        if not isinstance(defaults, dict):
            print("ERROR: 'preprocess.defaults' must be a mapping/object.", file=sys.stderr)
            return 2
        if not isinstance(files, list):
            print("ERROR: 'preprocess.files' must be a list.", file=sys.stderr)
            return 2
        for index, item in enumerate(files, start=1):
            if not isinstance(item, dict):
                print(f"ERROR: preprocess.files[{index}] must be a mapping/object.", file=sys.stderr)
                return 2
            if item.get("enabled", True) is False:
                continue
            try:
                cmd, assign_to, output_path_str = build_preprocess_command(
                    repo_root=repo_root,
                    config_dir=config_dir,
                    defaults=defaults,
                    item=item,
                )
            except RuntimeError as exc:
                print(f"ERROR: preprocess.files[{index}]: {exc}", file=sys.stderr)
                return 2
            name = item.get("name") or Path(str(item.get("input", ""))).name or f"file {index}"
            print(f"[preprocess {index}] {name}")
            code = run_command(cmd, cwd=config_dir, dry_run=cli.dry_run)
            if code != 0:
                return code
            if assign_to:
                audiobook_args[assign_to] = output_path_str

    if cli.skip_run:
        return 0

    if audiobook_section.get("enabled", True) is False:
        print("audiobook stage disabled in YAML.")
        return 0

    if not audiobook_args:
        print("ERROR: No audiobook arguments configured under 'audiobook.args'.", file=sys.stderr)
        return 2

    script_value = audiobook_section.get("script", "audiobook_qwen3.py")
    script_path = Path(str(resolve_config_path_value(config_dir, "script", script_value)))
    if not script_path.is_absolute():
        script_path = (repo_root / script_path).resolve()

    python_exec = str(audiobook_section.get("python") or sys.executable)
    cmd: list[str] = [python_exec, str(script_path)]
    for key, value in audiobook_args.items():
        append_option(
            cmd,
            key,
            value,
            config_dir=config_dir,
            path_keys=AUDIOBOOK_PATH_KEYS,
        )

    print("[audiobook]")
    return run_command(cmd, cwd=config_dir, dry_run=cli.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
