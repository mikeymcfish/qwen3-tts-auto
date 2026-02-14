# Qwen3-TTS Audiobook Generator

Simple CLI app that turns a `.txt` file into an audiobook with Qwen3-TTS voice cloning.

## Features

- Uses Qwen3-TTS voice cloning (`create_voice_clone_prompt` + `generate_voice_clone`).
- Splits text into paragraph-aware batches.
- Combines short paragraphs until `--max-chars-per-batch` is reached.
- Keeps paragraph boundaries by default, but splits oversized paragraphs at sentence endings.
- Supports control tags in text:
  - `[BREAK]`: force a batch boundary
  - `[CHAPTER]`: force a batch boundary and mark a chapter start
- Optional MP3 chapter embedding with `--use-chapters` (based on `[CHAPTER]` tags).
- Combines all generated parts with pause spacing, then outputs high-quality MP3 by default.
- Defrag-style live progress UI where each batch uses one block per 200 chars:
  - red: pending
  - blue: currently processing
  - green: just completed
  - white: completed
- Cooler animated terminal style with scan beam + pulse effects.
- Graceful stop controls:
  - `Ctrl+C` once: stop after current batch
  - `Ctrl+C` twice: attempt to cancel current batch; if not interruptible, output is discarded and batch is kept for resume
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
- installs system audio tools (`ffmpeg`, `sox`)
- installs Python dependencies
- attempts optional `flash-attn`

## Basic Usage

```bash
python audiobook_qwen3.py \
  --text-file /path/book.txt \
  --reference-audio /path/voice_ref.wav \
  --output /path/book_audiobook.mp3 \
  --max-chars-per-batch 1800 \
  --pause-ms 300 \
  --language Auto
```

## Important Arguments

- `--text-file`: input text file.
- `--reference-audio`: reference audio for cloning.
- If `--reference-audio` is omitted in interactive mode, the app scans the text file's folder for audio files that have matching `.txt` transcripts and prompts you to pick one.
- `--reference-text` or `--reference-text-file`: transcript for reference audio.
- If transcript args are omitted, the app auto-looks for `/path/to/<reference-audio-basename>.txt`.
- `--x-vector-only-mode`: allow cloning without transcript.
- `--max-chars-per-batch`: batch size control in characters.
- `--pause-ms`: silence inserted between batch outputs.
- `--mp3-quality`: MP3 VBR quality for final encode (`0` best, `9` smallest).
- `--use-chapters`: embed MP3 chapter metadata from `[CHAPTER]` markers.
- `--inference-batch-size`: compatibility option; batched mode is disabled and forced to `1`.
- `--max-inference-chars`: compatibility option retained for old scripts; ignored.
- `--output`: final path (`.mp3` recommended, `.wav` supported).
- `--resume-state`: continue from an existing `session_state.json`.
- `--attn-implementation`: attention backend (`sdpa` default, `flash_attention_2` optional).
- `--dtype`: model precision (`bfloat16` default, auto-fallback to `float16` if unsupported).
- `--no-defrag-ui`: disable defrag UI and print detailed text status/progress logs.

## Text Control Tags

You can place these tags anywhere in your input `.txt`:

- `[BREAK]`: forces a hard batch split at that point.
- `[CHAPTER]`: forces a hard batch split and marks the next spoken batch as a chapter start.

Use `--use-chapters` to write those chapter starts into final MP3 chapter metadata.
Chapter times are computed from actual combined audio, including configured `--pause-ms`, so they align with playback.

## Early Stop + Continue

Press `Ctrl+C` once while running.

Behavior:
1. Current batch is completed, or cancel is attempted for current batch on second `Ctrl+C`.
2. Existing completed parts are combined into current output audio.
3. The app writes:
   - `continue_from_batch_XXXXX.txt`
   - `continue_from_batch_XXXXX.sh`
   - `continue_from_batch_XXXXX.ps1`
   - updated `session_state.json`

Run the generated continue script to finish remaining text.

## Notes

- Default model is `Qwen/Qwen3-TTS-12Hz-1.7B-Base`.
- Default attention is `sdpa` for reliability. If `flash-attn` is installed, you can use `--attn-implementation flash_attention_2`.
- This project follows Qwen3-TTS API patterns from the official repository:
  - https://github.com/QwenLM/Qwen3-TTS

## Troubleshooting (RunPod)

If you see:
- `CUDA error: no kernel image is available for execution on the device`
- `CUDA error: device-side assert triggered`

Then your GPU architecture is likely not supported by the currently installed PyTorch CUDA wheel.

Try:

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("gpu:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none")
print("capability:", torch.cuda.get_device_capability(0) if torch.cuda.is_available() else "n/a")
print("arches:", torch.cuda.get_arch_list() if torch.cuda.is_available() else "n/a")
PY
```

If your GPU capability is missing from `arches`, reinstall torch with a newer index URL:

```bash
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Note: on some GPUs (for example RTX 4090 `sm_89`), torch may list `sm_86` without `sm_89`.  
That can still work through same-major forward compatibility.

And run with safe settings:

```bash
python audiobook_qwen3.py ... --dtype bfloat16 --attn-implementation sdpa
```

Batched inference was removed for stability after repeated CUDA assert failures on some GPU setups.

If CUDA asserts still happen:

```bash
python audiobook_qwen3.py ... --device cpu --dtype float32 --attn-implementation sdpa
```

If CPU works but CUDA fails, the issue is in the CUDA stack (torch/cuda/driver image), not your text batching.

If CPU works and CUDA still asserts, try this explicit GPU-safe combo first:

```bash
python audiobook_qwen3.py ... --dtype bfloat16 --attn-implementation sdpa --language Auto
```

If you see:
- `/bin/sh: 1: sox: not found`

Install SoX on the pod:

```bash
apt-get update && apt-get install -y sox
```

If you see:
- `flash_attn_2_cuda ... undefined symbol ...`

Then installed `flash-attn` is ABI-mismatched with your torch build. Use SDPA or reinstall:

```bash
python -m pip uninstall -y flash-attn flash_attn
python audiobook_qwen3.py ... --attn-implementation sdpa
```
