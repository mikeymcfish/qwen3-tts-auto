# MOSS/Qwen Audiobook Generator

CLI + Gradio app that turns a `.txt` file into an audiobook with voice cloning.

Default backend is **MOSS-TTS Delay** (`OpenMOSS-Team/MOSS-TTS`) for long-form narration stability.  
Qwen3-TTS is still supported as a fallback backend.

## Features

- Voice cloning with either:
  - `moss-delay` (default, production/long-form focused)
  - `moss-local` (smaller model)
  - `moss-ttsd` (native 2-speaker dialogue/conversation model)
  - `qwen` (legacy compatibility)
- Paragraph-aware batching with `[BREAK]` and `[CHAPTER]` control tags.
- Auto-tuned inference grouping for MOSS (`--inference-batch-size 0`).
- OOM-aware MOSS retry: automatically halves inference group size on CUDA OOM.
- Optional `--continuation-chain` mode for strongest cross-chunk continuity.
- Pause controls (`--pause-ms`, `--chapter-pause-ms`).
- Optional MP3 chapter metadata (`--use-chapters`).
- Resume/continue assets on early stop.
- Defrag-style progress UI in terminal and a Gradio UI (`gradio_app.py`).

## Install

```bash
pip install -r requirements.txt
```

This installs the **MOSS backend environment** (default).

Qwen currently has a hard dependency conflict with MOSS:
- `qwen-tts` pins `transformers==4.57.3`
- MOSS uses `transformers>=5.0.0`

Use a separate virtual environment for Qwen:

```bash
pip install -r requirements-qwen.txt
```

For Linux NVIDIA setup:

```bash
chmod +x scripts/install_linux_nvidia.sh
./scripts/install_linux_nvidia.sh
```

Installer behavior:
- It now skips PyTorch reinstall if a CUDA-enabled `torch` is already working in the target `.venv`.
- Set `FORCE_TORCH_INSTALL=1` to force reinstall from the selected CUDA wheel index.

## Quick Start (Recommended: MOSS Delay)

```bash
python audiobook_qwen3.py \
  --text-file /path/book.txt \
  --reference-audio /path/voice_ref.wav \
  --tts-backend moss-delay \
  --model-id OpenMOSS-Team/MOSS-TTS \
  --inference-batch-size 0 \
  --max-chars-per-batch 1800 \
  --output /path/book_audiobook.mp3
```

## Qwen Fallback Example

```bash
python audiobook_qwen3.py \
  --text-file /path/book.txt \
  --reference-audio /path/voice_ref.wav \
  --reference-text-file /path/voice_ref.txt \
  --tts-backend qwen \
  --model-id Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --x-vector-only-mode \
  --output /path/book_audiobook.mp3
```

Note: run this in your Qwen-specific environment if MOSS deps are also installed elsewhere.

## Native Dialogue Example (MOSS-TTSD)

This mode uses MOSS's built-in dialogue/conversation handling (native `[S1]` / `[S2]` tags).

```bash
python audiobook_qwen3.py \
  --text-file /path/dialogue.txt \
  --tts-backend moss-ttsd \
  --model-id OpenMOSS-Team/MOSS-TTSD-v1.0 \
  --reference-audio /path/speaker1.wav \
  --reference-text-file /path/speaker1.txt \
  --speaker2-reference-audio /path/speaker2.wav \
  --speaker2-reference-text-file /path/speaker2.txt \
  --output /path/dialogue_audiobook.mp3
```

Example `dialogue.txt`:

```text
[S1] Hello, are you there?
[S2] Yes, I'm here.
[S1] Great. Let's begin.
```

Transcript auto-load behavior (same as single-speaker mode):
- If `--reference-text-file` / `--reference-text` is omitted, the app will try `<reference-audio-basename>.txt`.
- If `--speaker2-reference-text-file` / `--speaker2-reference-text` is omitted, the app will try `<speaker2-reference-audio-basename>.txt`.

## Best-Quality Long-Form Mode (MOSS)

For maximum continuity (slower), enable continuation chaining:

```bash
python audiobook_qwen3.py \
  --text-file /path/book.txt \
  --reference-audio /path/voice_ref.wav \
  --reference-text-file /path/voice_ref.txt \
  --tts-backend moss-delay \
  --continuation-chain \
  --continuation-anchor-seconds 8 \
  --inference-batch-size 1 \
  --max-chars-per-batch 1600 \
  --output /path/book_audiobook.mp3
```

## Gradio UI

```bash
python gradio_app.py
```

Then open `http://127.0.0.1:7860`.

## YAML Run Config (No Long CLI Commands)

Use the included `run_settings.yaml` with the wrapper script:

```bash
python scripts/run_from_yaml.py
```

This can:

- load your audiobook run settings from YAML (`audiobook.args`)
- optionally preprocess reference clips first (`preprocess.files`)
- inject the cleaned output paths back into the audiobook command automatically

Dry-run preview (prints commands only):

```bash
python scripts/run_from_yaml.py --dry-run
```

## Reference Audio Preprocess Pipeline (Clone Prep)

Conservative ffmpeg-based speech cleanup is included:

```bash
python scripts/preprocess_reference_audio.py /path/ref.mp3 --output-file /path/ref_prep.wav
```

Default pipeline (tuned to be gentle):

- mono + resample to 24kHz
- high-pass filter (70 Hz)
- light denoise (`afftdn`)
- trim leading/trailing silence
- loudness normalize (`loudnorm`)
- final limiter

Notes:

- Requires `ffmpeg` in `PATH`.
- Keep denoise/enhancement conservative; over-processing can hurt voice-clone identity.
- `arnndn` is supported if your ffmpeg build includes it and you provide a model file.

## Key Arguments

- `--tts-backend`: `moss-delay`, `moss-local`, `moss-ttsd`, `qwen`, or `auto`.
- `--model-id`: model id/path for selected backend.
- `--reference-audio`: reference speech file/URL/path.
  - For MOSS backends, local compressed files (e.g. `.mp3`, `.m4a`) are auto-converted to WAV via `ffmpeg`.
- `--reference-text` / `--reference-text-file`:
  - Required for `qwen` unless `--x-vector-only-mode` is set.
  - Optional for MOSS backends.
- `--speaker2-reference-audio`: second speaker reference audio (used by `moss-ttsd` native dialogue mode).
- `--speaker2-reference-text` / `--speaker2-reference-text-file`:
  - Used by `moss-ttsd` native dialogue mode.
  - If omitted, app will try matching `<speaker2-reference-audio-basename>.txt`.
- `--max-chars-per-batch`: text chunk size.
- `--inference-batch-size`:
  - `0` = auto (recommended for MOSS).
  - `>=1` fixed group size.
  - Qwen backend is always forced to `1`.
- `--max-new-tokens`: max generated tokens per MOSS inference call.
- `--continuation-chain`: MOSS-only sequential continuation mode.
  Requires transcript anchor and forces sequential inference (`--inference-batch-size=1`).
- `--continuation-anchor-seconds`: in continuation mode, cap previous-batch audio context
  (and matching text suffix) to this duration. Lower values reduce VRAM use and can reduce
  over-pausing inherited from long trailing silence.
- `--pause-ms`: silence between generated chunks.
- `--chapter-pause-ms`: extra silence before chapter-start chunks.
- `--use-chapters`: embed MP3 chapter metadata from `[CHAPTER]`.
- `--no-defrag-ui`: verbose text progress mode.

## Control Tags

- `[BREAK]`: force a hard batch split.
- `[CHAPTER]`: force split and mark next spoken batch as chapter start.
- `[S1]` / `[S2]`: native dialogue speaker tags for `moss-ttsd`.

If `--use-chapters` is enabled, chapter times are embedded in final MP3 metadata.

## Early Stop + Continue

Press `Ctrl+C` once to stop after the current running inference group.
Press `Ctrl+C` again to request abort of the current group.

On stop, app writes:

- `continue_from_batch_XXXXX.txt`
- `continue_from_batch_XXXXX.sh`
- `continue_from_batch_XXXXX.ps1`
- updated `session_state.json`

## Notes

- Default model/backend:
  - backend: `moss-delay`
  - model: `OpenMOSS-Team/MOSS-TTS`
- MOSS backends support real grouped inference in this app.
- Qwen backend remains single-item generation for stability.
