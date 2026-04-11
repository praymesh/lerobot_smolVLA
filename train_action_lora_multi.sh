#!/bin/bash

# Activating environment just in case it's not active
eval "$(conda shell.bash hook)"
conda activate smol

# --- Experiment Configuration ---
BASE_POLICY="lerobot/smolvla_base"
DATASET="lerobot/aloha_sim_transfer_cube_human"

# ALOHA camera keys -> SmolVLA expected names
# Verify with: python -c "from lerobot.datasets.lerobot_dataset import LeRobotDataset; d=LeRobotDataset('lerobot/aloha_sim_transfer_cube_human'); print(d.meta.camera_keys)"
RENAME_MAP='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'

STEPS=40000
BATCH_SIZE=2
SAVE_FREQ=4000   # checkpoint every 4k steps -> 10 checkpoints total
LOG_FREQ=100     # log every 100 steps -> 400 loss points for clean curves

# configuration: rank 32, 50 demos
R=32
DEMO_COUNT=50

# --- Setup ---
EPISODES="[$(seq -s, 0 $((DEMO_COUNT-1)))]"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
JOB_NAME="aloha_r${R}_d${DEMO_COUNT}_${TIMESTAMP}"
OUT_DIR="outputs/train/${JOB_NAME}"

echo "LAUNCHING TRAINING | Job: $JOB_NAME | Rank: $R | Demos: $DEMO_COUNT"
echo "Dataset: $DATASET"
echo "Steps: $STEPS | Save every: $SAVE_FREQ | Log every: $LOG_FREQ"
echo "Output: $OUT_DIR"

export LORA_RANK=$R

# --- Pre-download dataset (skips automatically if already cached) ---
echo "Checking/downloading dataset to local cache..."
python - <<'PY'
import sys, os
sys.path.insert(0, os.path.abspath("src"))
from huggingface_hub import snapshot_download
from lerobot.utils.constants import HF_LEROBOT_HOME

repo_id = "lerobot/aloha_sim_transfer_cube_human"
local_dir = HF_LEROBOT_HOME / repo_id
print(f"  Cache path: {local_dir}")
snapshot_download(
    repo_id,
    repo_type="dataset",
    local_dir=local_dir,
    local_files_only=False,
)
print("  Dataset ready.")
PY

if [ $? -ne 0 ]; then
    echo "ERROR: Dataset download failed. Check your internet connection and HuggingFace access."
    exit 1
fi
echo ""

# Log to a staging file alongside the script; lerobot creates OUT_DIR itself
# so we must NOT mkdir it first (validate() raises FileExistsError if it exists).
LOG_STAGING="train_${JOB_NAME}.log"

# RUN THE TRAINING SCRIPT
# stdout+stderr go to both terminal and staging log file
python run_exp2_lora.py \
  --policy.path=$BASE_POLICY \
  --policy.push_to_hub=false \
  --policy.use_peft=true \
  --dataset.repo_id=$DATASET \
  --dataset.episodes="$EPISODES" \
  --rename_map="$RENAME_MAP" \
  --batch_size=$BATCH_SIZE \
  --num_workers=0 \
  --steps=$STEPS \
  --save_freq=$SAVE_FREQ \
  --log_freq=$LOG_FREQ \
  --output_dir=$OUT_DIR \
  --job_name=$JOB_NAME \
  --policy.device=cuda \
  2>&1 | tee "$LOG_STAGING"

# Move log into the output dir once lerobot has created it
if [ -d "$OUT_DIR" ]; then
    mv "$LOG_STAGING" "${OUT_DIR}/train.log"
    LOG_FILE="${OUT_DIR}/train.log"
else
    LOG_FILE="$LOG_STAGING"
fi

echo "=========================================================="
echo "Training complete."
echo "  Checkpoints : $OUT_DIR/checkpoints/  (every ${SAVE_FREQ} steps)"
echo "  Raw log     : $LOG_FILE"
echo "=========================================================="

# --- Extract clean loss CSV from the raw log ---
LOSS_CSV="${LOG_FILE%.log}_loss.csv"
python - "$LOG_FILE" "$LOSS_CSV" <<'PY'
import sys, re, csv

log_path, out_path = sys.argv[1], sys.argv[2]

def parse_num(s):
    """Handle format_big_number suffixes: 1K -> 1000, 1M -> 1e6, etc."""
    s = s.strip()
    if s.endswith("K"): return float(s[:-1]) * 1_000
    if s.endswith("M"): return float(s[:-1]) * 1_000_000
    if s.endswith("B"): return float(s[:-1]) * 1_000_000_000
    return float(s)

step_re  = re.compile(r"\bstep:([\d.KMB]+)")
loss_re  = re.compile(r"\bloss:([\d.eE+\-]+)")

rows = []
with open(log_path) as f:
    for line in f:
        sm = step_re.search(line)
        lm = loss_re.search(line)
        if sm and lm:
            rows.append((int(parse_num(sm.group(1))), float(lm.group(1))))

with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["step", "loss"])
    w.writerows(rows)

print(f"Loss curve: {out_path}  ({len(rows)} points)")
PY

echo ""
echo "Quick plot (requires matplotlib):"
echo "  python -c \""
echo "  import csv, matplotlib.pyplot as plt"
echo "  rows = list(csv.DictReader(open('${LOSS_CSV}')))"
echo "  plt.plot([r['step'] for r in rows], [r['loss'] for r in rows])"
echo "  plt.xlabel('step'); plt.ylabel('loss'); plt.title('Training loss')"
echo "  plt.savefig('${LOSS_CSV%.csv}.png', dpi=150); plt.show()\""
