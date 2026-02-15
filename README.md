# MOSS/Qwen Audiobook Generator

CLI + Gradio app that turns a `.txt` file into an audiobook with voice cloning.

Default backend is **MOSS-TTS Delay** (`OpenMOSS-Team/MOSS-TTS`) for long-form narration stability.  
Qwen3-TTS is still supported as a fallback backend.

## Features

- Voice cloning with either:
  - `moss-delay` (default, production/long-form focused)
  - `moss-local` (smaller model)
  - `qwen` (legacy compatibility)
- Paragraph-aware batching with `[BREAK]` and `[CHAPTER]` control tags.
- Auto-tuned inference grouping for MOSS (`--inference-batch-size 0`).
- OOM-aware MOSS retry: automatically halves inference group size on CUDA OOM.
- Pause controls (`--pause-ms`, `--chapter-pause-ms`).
- Optional MP3 chapter metadata (`--use-chapters`).
- Resume/continue assets on early stop.
- Defrag-style progress UI in terminal and a Gradio UI (`gradio_app.py`).

## Install

```bash
pip install -r requirements.txt
```

For Linux NVIDIA setup:

```bash
chmod +x scripts/install_linux_nvidia.sh
./scripts/install_linux_nvidia.sh
```

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

## Gradio UI

```bash
python gradio_app.py
```

Then open `http://127.0.0.1:7860`.

## Key Arguments

- `--tts-backend`: `moss-delay`, `moss-local`, `qwen`, or `auto`.
- `--model-id`: model id/path for selected backend.
- `--reference-audio`: reference speech file/URL/path.
- `--reference-text` / `--reference-text-file`:
  - Required for `qwen` unless `--x-vector-only-mode` is set.
  - Optional for MOSS backends.
- `--max-chars-per-batch`: text chunk size.
- `--inference-batch-size`:
  - `0` = auto (recommended for MOSS).
  - `>=1` fixed group size.
  - Qwen backend is always forced to `1`.
- `--max-new-tokens`: max generated tokens per MOSS inference call.
- `--pause-ms`: silence between generated chunks.
- `--chapter-pause-ms`: extra silence before chapter-start chunks.
- `--use-chapters`: embed MP3 chapter metadata from `[CHAPTER]`.
- `--no-defrag-ui`: verbose text progress mode.

## Control Tags

- `[BREAK]`: force a hard batch split.
- `[CHAPTER]`: force split and mark next spoken batch as chapter start.

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
