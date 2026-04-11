#!/bin/bash
# Minimal eval for the ALOHA LoRA checkpoint (RTX 3050 6 GB).

eval "$(conda shell.bash hook)"
conda activate smol

CKPT="outputs/train/aloha_r32_d50_20260411_201928/checkpoints/last/pretrained_model"
OUT_DIR="outputs/eval/aloha_r32_d50"

# ALOHA camera keys -> policy training names (same as training rename_map)
RENAME_MAP='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'

# 5 episodes is enough for a quick sanity check; increase when you have time
N_EPISODES=5

if [ ! -d "$CKPT" ]; then
    echo "ERROR: checkpoint not found at $CKPT"
    exit 1
fi

echo "Checkpoint : $CKPT"
echo "Episodes   : $N_EPISODES"
echo "Output     : $OUT_DIR"
echo ""

python eval_lora_smolvla.py \
  --policy.type=smolvla \
  --policy.pretrained_path="$CKPT" \
  --policy.use_peft=true \
  --policy.vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct" \
  --policy.device=cuda \
  --policy.use_amp=true \
  --env.type=aloha \
  --env.task=AlohaTransferCube-v0 \
  --eval.n_episodes=$N_EPISODES \
  --eval.batch_size=1 \
  --rename_map="$RENAME_MAP" \
  --output_dir="$OUT_DIR"

EXIT=$?
echo ""
if [ $EXIT -eq 0 ]; then
    python -c "
import json
try:
    d = json.load(open('${OUT_DIR}/eval_info.json'))
    sr = d.get('overall', d).get('pc_success', 'n/a')
    print(f'Success rate: {sr}%')
except Exception as e:
    print(f'Could not read result: {e}')
"
else
    echo "Eval failed (exit code $EXIT)"
fi
