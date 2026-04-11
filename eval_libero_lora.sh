#!/bin/bash
# Evaluate SmolVLA + LoRA checkpoints on LIBERO.
#
# Fixes applied vs original:
#   1. Uses eval CLI (lerobot_eval.py), not the training CLI with --steps=0.
#   2. Loads lerobot/smolvla_base as the pretrained base, then applies LoRA
#      delta weights.  (Training intended this; the random-init base at eval
#      time produces garbage outputs.)
#   3. Adds --rename_map so the policy receives camera1/camera2 (as trained)
#      instead of image/image2 (what LIBERO env names them).  Without this
#      the VLA is blind and gets 0 reward every episode.

eval "$(conda shell.bash hook)"
conda activate smol

BASE_SCRIPT="eval_lora_smolvla.py"
DEVICE="cuda"
VLM_MODEL="HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

TASK_IDS="[0]"
EPISODES=5       # keep low for 6 GB VRAM

RANKS=(4 8 16)
DEMO_COUNTS=(20 50 100)   # all trained variants

# Rename map: LIBERO env key -> policy training key
RENAME_MAP='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'

for R in "${RANKS[@]}"; do
  for DEMO in "${DEMO_COUNTS[@]}"; do

    JOB_NAME="exp2_r${R}_d${DEMO}"
    CKPT_PATH="outputs/train/${JOB_NAME}/checkpoints/last/pretrained_model"

    if [ ! -d "$CKPT_PATH" ]; then
      echo "Skipping $JOB_NAME — checkpoint not found at $CKPT_PATH"
      continue
    fi

    echo "=========================================================="
    echo "EVALUATING | Job: $JOB_NAME | Rank: $R | Demos: $DEMO"
    echo "Checkpoint: $CKPT_PATH"
    echo "=========================================================="

    python "$BASE_SCRIPT" \
      --policy.type=smolvla \
      --policy.pretrained_path="$CKPT_PATH" \
      --policy.use_peft=true \
      --policy.vlm_model_name="$VLM_MODEL" \
      --policy.device=$DEVICE \
      --policy.use_amp=true \
      --env.type=libero \
      --env.task_ids="$TASK_IDS" \
      --eval.n_episodes=$EPISODES \
      --eval.batch_size=1 \
      --rename_map="$RENAME_MAP" \
      --output_dir="outputs/eval/${JOB_NAME}"

    EXIT_CODE=$?
    echo "=========================================================="
    if [ $EXIT_CODE -eq 0 ]; then
      echo "DONE: $JOB_NAME"
      # Print the success rate immediately
      python -c "
import json, sys
try:
    d = json.load(open('outputs/eval/${JOB_NAME}/eval_info.json'))
    print(f'  Success rate: {d[\"overall\"][\"pc_success\"]:.1f}%')
except: pass
" 2>/dev/null
    else
      echo "FAILED: $JOB_NAME (exit code $EXIT_CODE)"
    fi
    echo ""

    # Free GPU memory between experiments
    python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

  done
done

echo "All evaluations complete. Results are in outputs/eval/."
