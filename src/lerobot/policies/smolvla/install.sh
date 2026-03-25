#!/usr/bin/env bash
set -euo pipefail

# ===== Config =====
ENV_NAME="smol"
REPO_DIR="$HOME/VPR_model_tests/dino/lerobot_smolVLA"

# If you want a fully fresh env every time, uncomment:
# conda env remove -n "${ENV_NAME}" -y || true

echo "[1/6] Create conda env"
conda create -n "${ENV_NAME}" -y python=3.12

echo "[2/6] Install FFmpeg runtime (needed by torchcodec)"
conda install -n "${ENV_NAME}" -y -c conda-forge "ffmpeg>=6,<7"

echo "[3/6] Upgrade pip tooling inside env"
conda run -n "${ENV_NAME}" python -m pip install --upgrade pip setuptools wheel

echo "[4/6] Install PyTorch CUDA 12.8 stack"
PYTHONNOUSERSITE=1 conda run -n "${ENV_NAME}" python -m pip install \
  torch==2.9.1+cu128 torchvision==0.24.1+cu128 torchaudio==2.9.1+cu128 \
  --index-url https://download.pytorch.org/whl/cu128

echo "[5/6] Install LeRobot + SmolVLA extras (editable)"
cd "${REPO_DIR}"
PYTHONNOUSERSITE=1 conda run -n "${ENV_NAME}" python -m pip install -e ".[smolvla]"

echo "[6/6] Verify runtime"
PYTHONNOUSERSITE=1 conda run -n "${ENV_NAME}" python - <<'PY'
import torch, torchcodec
import importlib.metadata as m
print("torch:", torch.__version__)
print("torchcodec:", m.version("torchcodec"))
print("OK: torch/torchcodec import")
PY

PYTHONNOUSERSITE=1 conda run -n "${ENV_NAME}" bash -lc 'lerobot-train --help >/dev/null && echo "OK: lerobot-train available"'

echo
echo "Done. Use this for training:"
echo "conda activate ${ENV_NAME}"
echo "cd ${REPO_DIR}"
echo "PYTHONNOUSERSITE=1 lerobot-train --policy.path=lerobot/smolvla_base --policy.push_to_hub=false --dataset.repo_id=lerobot/svla_so100_pickplace --rename_map='{\"observation.images.top\":\"observation.images.camera1\",\"observation.images.wrist\":\"observation.images.camera2\"}' --batch_size=64 --steps=20000 --output_dir=outputs/train/my_smolvla --job_name=my_smolvla_training --policy.device=cuda"