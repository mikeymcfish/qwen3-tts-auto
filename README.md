# Qwen3-TTS Audiobook Generator

Simple CLI app that turns a `.txt` file into an audiobook with Qwen3-TTS voice cloning.

## Features

- Uses Qwen3-TTS voice cloning (`create_voice_clone_prompt` + `generate_voice_clone`).
- Splits text into paragraph-aware batches.
- Combines short paragraphs until `--max-chars-per-batch` is reached.
- Batch boundaries only happen between paragraphs.
- Combines all generated parts into one final WAV with a configurable pause between batches.
- Defrag-style live progress UI where each batch uses one block per 200 chars:
  - red: pending
  - blue: currently processing
  - green: just completed
  - white: completed
- Graceful stop: press `Ctrl+C` once, current inference call finishes, then app exits.
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
- `--inference-batch-size`: compatibility option; batched mode is disabled and forced to `1`.
- `--max-inference-chars`: compatibility option retained for old scripts; ignored.
- `--output`: final combined WAV file.
- `--resume-state`: continue from an existing `session_state.json`.
- `--attn-implementation`: attention backend (`sdpa` default, `flash_attention_2` optional).
- `--dtype`: model precision (`float16` default for broad GPU compatibility).
- `--no-defrag-ui`: fallback to plain logs.

## Early Stop + Continue

Press `Ctrl+C` once while running.

Behavior:
1. Current inference call finishes.
2. Existing parts are still combined into current output WAV.
3. The app writes:
   - `continue_from_batch_XXXXX.txt`
   - `continue_from_batch_XXXXX.sh`
   - `continue_from_batch_XXXXX.ps1`
   - updated `session_state.json`

Run the generated continue script to finish remaining text.

## Notes

- Default model is `Qwen/Qwen3-TTS-12Hz-0.6B-Base`.
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
python audiobook_qwen3.py ... --dtype float16 --attn-implementation sdpa
```

Batched inference was removed for stability after repeated CUDA assert failures on some GPU setups.
