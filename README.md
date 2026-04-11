# LoRA Fine-Tuning of SmolVLA Action Expert

Fine-tuning only the **action expert** of [SmolVLA](https://huggingface.co/lerobot/smolvla_base)
using LoRA (Low-Rank Adaptation), while keeping the SmolVLM-500M-Instruct backbone fully frozen.

---

## Architecture: What Gets Trained

SmolVLA consists of two parts:

| Component | Description | Frozen? |
|---|---|---|
| SmolVLM-500M-Instruct backbone | Vision encoder + language text model | **Yes** |
| Action expert (`lm_expert`) | Smaller transformer that processes noisy actions | **No — LoRA here** |
| Action projections | `action_in_proj`, `action_out_proj`, `action_time_mlp_in/out` | **No — LoRA here** |

The LoRA adapters target:
- `model.vlm_with_expert.lm_expert.*.q_proj` and `*.v_proj` — attention layers of the action expert
- `model.action_in_proj`, `model.action_out_proj`, `model.action_time_mlp_in`, `model.action_time_mlp_out` — action I/O projections

Defined in `src/lerobot/policies/smolvla/modeling_smolvla.py` (lines 484–493).

---

## Setup

```bash
# 1. Install base LeRobot + SmolVLA dependencies
pip install -e ".[smolvla]"

# 2. Install our additional requirements
pip install -r requirements.txt
```

For faster dataset downloads (recommended — avoids network hangs):
```bash
pip install hf-transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
```

---

## Training

### ALOHA (default config in the script)

```bash
./train_action_lora_multi.sh
```

This runs a single experiment: **rank 32, 50 demonstrations, 40 000 steps**,
with checkpoints saved every 4 000 steps.

Training output lands in:
```
outputs/train/aloha_r32_d50_<timestamp>/
  checkpoints/          # one folder per 4k steps + "last" symlink
  train_loss.csv        # step,loss CSV for plotting (generated at end)
```

The raw training log is at `train_aloha_r32_d50_<timestamp>.log` in the repo
root during training, then moved into the output directory on completion.

### LIBERO

Change the following variables at the top of `train_action_lora_multi.sh`:

```bash
DATASET="lerobot/libero"
RENAME_MAP='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
```

The LIBERO episodes already use the same `top`/`wrist` → `camera1`/`camera2`
mapping convention, just with different source key names.

### Changing the LoRA rank

Edit `train_action_lora_multi.sh`:

```bash
R=32        # change to 4, 8, 16, 64, etc.
```

The rank is passed to the training script via the `LORA_RANK` environment
variable. `lora_alpha` is automatically set to `2 × rank` (standard default).

### Sweeping multiple ranks / demo counts

To run a grid sweep (e.g. ranks 4, 16, 32 × demo counts 20, 50), replace the
fixed variables with loops:

```bash
RANKS=(4 16 32)
DEMO_COUNTS=(20 50)

for R in "${RANKS[@]}"; do
  for DEMO_COUNT in "${DEMO_COUNTS[@]}"; do
    # ... rest of the training block
  done
done
```

---

## Key Training Files

| File | Purpose |
|---|---|
| `train_action_lora_multi.sh` | Main training script (edit to change dataset / rank) |
| `run_exp2_lora.py` | Python entry-point: patches model loading and applies LoRA |

`run_exp2_lora.py` operates in two modes:

- **Training** (no `adapter_config.json` found): applies a fresh LoRA using
  `get_peft_model` with the specified rank. All base model weights are frozen.
- **Eval** (`adapter_config.json` found at `pretrained_path`): loads saved
  LoRA delta weights via `PeftModel.from_pretrained` for inference.

---

## Plotting the Loss Curve

After training completes, `train_loss.csv` is auto-generated. To plot:

```bash
python plot_loss.py outputs/train/aloha_r32_d50_<timestamp>/train_loss.csv
```

Options:
```
--smooth 0.95    # stronger EMA smoothing (default 0.9, range 0–1)
--no-smooth      # show raw values only
```

The PNG is saved next to the CSV automatically.

If training was interrupted or the CSV was not generated, extract it manually:

```bash
python plot_loss.py <path/to/train.log>   # works on the raw log too
```

Or regenerate the CSV from an existing log:

```bash
python - train_<jobname>.log outputs/train/<jobname>/train_loss.csv <<'PY'
import re, sys, csv
def p(s):
    if s.endswith("K"): return int(float(s[:-1])*1000)
    return int(float(s))
rows = []
for line in open(sys.argv[1]):
    sm = re.search(r"\bstep:([\d.KMB]+)", line)
    lm = re.search(r"\bloss:([\d.eE+\-]+)", line)
    if sm and lm: rows.append((p(sm.group(1)), float(lm.group(1))))
w = csv.writer(open(sys.argv[2], "w"))
w.writerow(["step", "loss"]); w.writerows(rows)
print(f"{len(rows)} rows written")
PY
```

---

## Evaluation

### ALOHA

```bash
./eval_aloha_lora.sh
```

Runs 5 episodes of `AlohaTransferCube-v0` using the trained checkpoint at
`outputs/train/aloha_r32_d50_<timestamp>/checkpoints/last/pretrained_model`.

To change the checkpoint, edit `CKPT` in `eval_aloha_lora.sh`.
To run more episodes (for a reliable success rate), change `N_EPISODES`.

### LIBERO

```bash
./eval_libero_lora.sh
```

Sweeps all trained LIBERO checkpoints under `outputs/train/exp2_r*/`.

### How evaluation loads LoRA

Evaluation uses `eval_lora_smolvla.py`, which:

1. Reads `train_config.json` from the adapter checkpoint to find the base
   model path (`lerobot/smolvla_base`).
2. Loads the full pretrained base model.
3. Applies the saved LoRA delta weights via `PeftModel.from_pretrained`.
4. Runs the lerobot eval loop.

---

## Dataset Camera Key Mapping

SmolVLA expects camera observations named `camera1` (and optionally `camera2`).
Both ALOHA and LIBERO use different names in their environments, so a
`rename_map` is passed at both training and eval time:

| Dataset | Raw key | Policy key |
|---|---|---|
| ALOHA | `observation.images.top` | `observation.images.camera1` |
| ALOHA | `observation.images.wrist` | `observation.images.camera2` |
| LIBERO | `observation.images.image` | `observation.images.camera1` |
| LIBERO | `observation.images.image2` | `observation.images.camera2` |
