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

echo "[1/6] Installing base system packages..."
if command -v apt-get >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y \
    python3 \
    python3-venv \
    python3-pip \
    ffmpeg \
    git \
    build-essential \
    libsndfile1
else
  echo "apt-get not found."
  echo "Install manually: python3 python3-venv python3-pip ffmpeg build-essential libsndfile1"
fi

echo "[2/6] Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate

echo "[3/6] Upgrading pip tooling..."
python -m pip install --upgrade pip setuptools wheel

echo "[4/6] Installing PyTorch CUDA wheels..."
python -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

echo "[5/6] Installing project Python dependencies..."
python -m pip install --upgrade -r requirements.txt

echo "[6/6] Installing optional flash-attn..."
if python -c "import torch; import sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
  if python -m pip install --upgrade flash-attn --no-build-isolation; then
    echo "flash-attn installed."
  else
    echo "flash-attn installation failed; continuing without it."
  fi
else
  echo "Torch CUDA is unavailable; skipping flash-attn."
fi

cat <<'EOF'

Install complete.

Next:
  source .venv/bin/activate
  python audiobook_qwen3.py --help

EOF
