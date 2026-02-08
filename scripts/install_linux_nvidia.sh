#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "This installer supports Linux only."
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "NVIDIA GPU/driver not detected (nvidia-smi is missing)."
  echo "Install NVIDIA drivers first, then rerun this installer."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_DIR}"

# Keep pip build/cache/temp on the same filesystem to avoid cross-device rename issues.
export PIP_CACHE_DIR="${REPO_DIR}/.pip-cache"
export TMPDIR="${REPO_DIR}/.pip-tmp"
mkdir -p "${PIP_CACHE_DIR}" "${TMPDIR}"

echo "[1/6] Installing base system packages..."
if command -v apt-get >/dev/null 2>&1; then
  APT_PREFIX=()
  APT_SKIP="0"
  if [[ "$(id -u)" -ne 0 ]]; then
    if command -v sudo >/dev/null 2>&1; then
      APT_PREFIX=(sudo)
    else
      echo "apt-get is available, but this shell is not root and sudo is missing."
      echo "Skipping system package install step."
      APT_SKIP="1"
    fi
  fi
  if [[ "${APT_SKIP}" == "0" ]]; then
    "${APT_PREFIX[@]}" apt-get update
    "${APT_PREFIX[@]}" apt-get install -y \
      python3 \
      python3-venv \
      python3-pip \
      ffmpeg \
      git \
      build-essential \
      libsndfile1
  fi
else
  echo "apt-get not found."
  echo "Install manually: python3 python3-venv python3-pip ffmpeg build-essential libsndfile1"
fi

echo "[2/6] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "[3/6] Upgrading pip tooling..."
python -m pip install --upgrade pip setuptools wheel

GPU_CC_RAW="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n1 | xargs || true)"
GPU_CC_MAJOR="${GPU_CC_RAW%%.*}"
GPU_CC_MINOR="${GPU_CC_RAW##*.}"
if [[ -z "${GPU_CC_MAJOR}" || "${GPU_CC_MAJOR}" == "${GPU_CC_RAW}" ]]; then
  GPU_CC_MAJOR="0"
fi
if [[ -z "${GPU_CC_MINOR}" || "${GPU_CC_MINOR}" == "${GPU_CC_RAW}" ]]; then
  GPU_CC_MINOR="0"
fi

TORCH_INDEX_URL_DEFAULT="https://download.pytorch.org/whl/cu121"
if [[ "${GPU_CC_MAJOR}" -ge 12 ]]; then
  TORCH_INDEX_URL_DEFAULT="https://download.pytorch.org/whl/cu128"
fi
TORCH_INDEX_URL="${TORCH_INDEX_URL:-${TORCH_INDEX_URL_DEFAULT}}"

echo "[4/6] Installing PyTorch CUDA wheels from ${TORCH_INDEX_URL}..."
python -m pip install --upgrade torch torchvision torchaudio --index-url "${TORCH_INDEX_URL}"

echo "[5/6] Installing project Python dependencies..."
python -m pip install --upgrade -r requirements.txt

echo "[6/6] Installing optional flash-attn..."
INSTALL_FLASH_ATTN="1"
if [[ "${GPU_CC_MAJOR}" -lt 8 ]]; then
  INSTALL_FLASH_ATTN="0"
  echo "Skipping flash-attn: GPU compute capability ${GPU_CC_RAW} is below 8.0."
fi
if [[ "${GPU_CC_MAJOR}" -ge 12 ]]; then
  INSTALL_FLASH_ATTN="0"
  echo "Skipping flash-attn on ${GPU_CC_RAW}; use sdpa unless you have a known-good build."
fi

if [[ "${INSTALL_FLASH_ATTN}" == "1" ]] && python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  if python -m pip install --upgrade flash-attn --no-build-isolation --no-clean; then
    echo "flash-attn installed."
  else
    echo "flash-attn installation failed; continuing without it."
    echo "You can still run the app; it will use sdpa attention by default."
  fi
else
  echo "Torch CUDA is unavailable or flash-attn was skipped."
fi

cat <<'EOF'

Install complete.

Next:
  source .venv/bin/activate
  python audiobook_qwen3.py --help

EOF
