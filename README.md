# Qwen3-TTS Audiobook Generator

Simple CLI app that turns a `.txt` file into an audiobook with Qwen3-TTS voice cloning.

## Features

- Uses Qwen3-TTS voice cloning (`create_voice_clone_prompt` + `generate_voice_clone`).
- Splits text into paragraph-aware batches.
- Combines short paragraphs until `--max-chars-per-batch` is reached.
- Batch boundaries only happen between paragraphs.
- Combines all generated parts into one final WAV with a configurable pause between batches.
- Defrag-style live progress UI in terminal.
- Graceful stop: press `Ctrl+C` once, current batch finishes, then app exits.
- Auto-generated resume assets on early stop:
  - new `.txt` with remaining text
  - `continue_*.sh`
  - `continue_*.ps1`
  - `session_state.json`
- Linux NVIDIA auto-installer script.

## Linux NVIDIA Auto-Install

```bash
cd qwen3-tts-auto
chmod +x scripts/install_linux_nvidia.sh
./scripts/install_linux_nvidia.sh
```

This installer:
- checks for `nvidia-smi`
- creates `.venv`
- installs CUDA PyTorch wheels
- installs Python dependencies
- attempts optional `flash-attn`

## Basic Usage

```bash
python audiobook_qwen3.py \
  --text-file /path/book.txt \
  --reference-audio /path/voice_ref.wav \
  --reference-text-file /path/voice_ref_transcript.txt \
  --output /path/book_audiobook.wav \
  --max-chars-per-batch 1800 \
  --pause-ms 300 \
  --language Auto
```

## Important Arguments

- `--text-file`: input text file.
- `--reference-audio`: reference audio for cloning.
- `--reference-text` or `--reference-text-file`: transcript for reference audio.
- `--x-vector-only-mode`: allow cloning without transcript.
- `--max-chars-per-batch`: batch size control in characters.
- `--pause-ms`: silence inserted between batch outputs.
- `--output`: final combined WAV file.
- `--resume-state`: continue from an existing `session_state.json`.
- `--no-defrag-ui`: fallback to plain logs.

## Early Stop + Continue

Press `Ctrl+C` once while running.

Behavior:
1. Current batch finishes.
2. Existing parts are still combined into current output WAV.
3. The app writes:
   - `continue_from_batch_XXXXX.txt`
   - `continue_from_batch_XXXXX.sh`
   - `continue_from_batch_XXXXX.ps1`
   - updated `session_state.json`

Run the generated continue script to finish remaining text.

## Notes

- Default model is `Qwen/Qwen3-TTS-12Hz-0.6B-Base`.
- This project follows Qwen3-TTS API patterns from the official repository:
  - https://github.com/QwenLM/Qwen3-TTS
