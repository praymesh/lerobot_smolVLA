# SmolVLA with QLoRA (4-bit Quantized VLM + LoRA)

**QLoRA** fine-tuning of SmolVLA: the VLM backbone (SmolVLM2-500M) is loaded
in **4-bit NF4** quantization, then **LoRA adapters** are added on top. This
cuts VLM memory by ~4× compared to full BF16, and allows the language model to
adapt to the robot domain instead of staying fully frozen.

---

## What Changed vs Base SmolVLA

### Standard SmolVLA fine-tuning (lora branch)
```
VLM backbone  → BF16 weights, fully frozen
Action expert → full precision, trainable
```

### This branch (QLoRA)
```
VLM backbone  → 4-bit NF4 weights (BitsAndBytes), LoRA adapters trainable
Action expert → full precision, trainable
Vision encoder → frozen (no LoRA)
```

Both the action expert **and** the VLM text model learn — QLoRA makes the VLM
update affordable by keeping base weights at 4-bit and only storing the small
LoRA delta matrices in BF16.

---

## Key Files Changed

| File | What changed |
|---|---|
| `src/lerobot/policies/smolvla/smolvlm_with_expert.py` | `use_qlora` path: loads VLM in 4-bit NF4 via BnB, wraps with LoRA via PEFT; `get_vlm_model()` updated to traverse PeftModel wrappers; `set_requires_grad()` updated for LoRA-only training; `get_compute_dtype()` helper added |
| `src/lerobot/policies/smolvla/configuration_smolvla.py` | New fields: `use_qlora`, `lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_modules`; validation that `use_qlora` and `train_expert_only` are not both True |
| `src/lerobot/policies/smolvla/modeling_smolvla.py` | `SmolVLAPolicy.from_pretrained()` overridden to skip `.to(device)` after load; `SmolVLAPolicy.to()` overridden to no-op if VLM is already on GPU; all projection layers explicitly moved to the VLM's device after `device_map="auto"` |
| `src/lerobot/scripts/lerobot_eval.py` | Policy is loaded **before** the environment to give the 4-bit model priority on GPU memory |
| `test_qlora_sanity.py` | End-to-end sanity check: loads in QLoRA mode, reports trainable params, VRAM usage, runs a forward pass, verifies LoRA placement |

---

## New Configuration Fields

Add these to your training command via `--policy.*`:

| Field | Default | Description |
|---|---|---|
| `use_qlora` | `False` | Enable 4-bit NF4 quantization + LoRA on the VLM |
| `lora_r` | `16` | LoRA rank |
| `lora_alpha` | `32` | LoRA alpha (scaling = alpha / r) |
| `lora_dropout` | `0.05` | Dropout on LoRA layers |
| `lora_target_modules` | `["q_proj","k_proj","v_proj","o_proj","proj"]` | Which linear layers get LoRA (full attention + connector) |

> `use_qlora=True` is **incompatible** with `train_expert_only=True`.
> QLoRA trains LoRA params in the VLM text model, so the VLM must not be
> fully frozen. Setting both raises a `ValueError` at startup.

---

## Engineering Details

### 4-bit quantization config (BitsAndBytes)
```python
BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # NormalFloat4 — best accuracy at 4-bit
    bnb_4bit_use_double_quant=True,      # quantize the quantization constants too (~0.4 bit extra savings)
    bnb_4bit_compute_dtype=torch.bfloat16,  # activations and LoRA remain BF16
)
```

### Device placement problem — and the fix
`device_map="auto"` (used by BnB) places the VLM on GPU automatically.
Calling `.to(device)` after the fact causes OOM by attempting to duplicate
the already-placed model. Two overrides prevent this:

- **`SmolVLAPolicy.from_pretrained()`** — loads weights to CPU via
  `safetensors`, then lets PyTorch's in-place `copy_` move them to wherever
  each parameter already lives (GPU for VLM, CPU for the rest). Skips the
  final `policy.to(device)`.
- **`SmolVLAPolicy.to()`** — checks if the VLM is already on CUDA; if so,
  returns `self` immediately instead of calling `super().to()`.

All projection layers (`state_proj`, `action_in_proj`, etc.) are explicitly
moved to the VLM's device after initialization to keep everything on the same
device.

### `get_compute_dtype` helper
BnB 4-bit layers store weights as `uint8` (not a floating-point dtype).
The existing code that does `hidden_states.to(dtype=layer.weight.dtype)` would
fail. `get_compute_dtype()` returns `bfloat16` for quantized layers and the
actual weight dtype otherwise.

### `prepare_model_for_kbit_training`
Called before wrapping with LoRA. This:
- Enables gradient checkpointing on the VLM (saves activation memory at the
  cost of a small compute overhead)
- Casts non-quantized parameters (LayerNorm, embeddings) to `float32` for
  numerical stability

### Eval order swap
`lerobot_eval.py` now loads the policy **before** creating the simulation
environment. On a 6 GB GPU, the BnB 4-bit model claims its memory first;
the env renderer then uses whatever remains, preventing OOM during eval.

---

## Setup

```bash
# 1. Install base LeRobot + SmolVLA
pip install -e ".[smolvla]"

# 2. Install QLoRA requirements
pip install -r requirements.txt
```

---

## Sanity Check

Run before training to verify QLoRA loads correctly and VRAM fits:

```bash
python test_qlora_sanity.py
```

Expected output (RTX 3050 6 GB):
```
GPU : NVIDIA GeForce RTX 3050 ...
VRAM: 6.0 GB

Loading SmolVLM2-500M in 4-bit QLoRA mode ...
QLoRA enabled: VLM loaded in 4-bit NF4, LoRA r=16 on [q_proj, k_proj, ...]

--- Trainable Parameters ---
  Expert (action head) :  X.XX M
  VLM LoRA adapters    :  X.XX M
  Total trainable      :  X.XX M / XXX.XX M  (X.XX %)

--- GPU Memory After Load ---
  Allocated : ~1.5–2.0 GB
  Remaining : ~4.0 GB free for activations / optimizer

--- LoRA Adapter Check ---
  Vision encoder LoRA : N layers  frozen=True  ✓
  Text model LoRA     : N layers  trainable=True  ✓

✓  All checks passed — QLoRA is set up correctly.
```

---

## Training

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.use_qlora=true \
  --policy.lora_r=16 \
  --policy.lora_alpha=32 \
  --dataset.repo_id=lerobot/aloha_sim_transfer_cube_human \
  --dataset.episodes="[$(seq -s, 0 49)]" \
  --rename_map='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}' \
  --batch_size=2 \
  --steps=40000 \
  --save_freq=4000 \
  --log_freq=100 \
  --output_dir=outputs/train/qlora_run \
  --job_name=qlora_smolvla \
  --policy.device=cuda \
  --policy.use_amp=true \
  --num_workers=0
```

For **LIBERO**:
```bash
--dataset.repo_id=lerobot/libero
--rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
```

> **RTX 3050 6 GB:** QLoRA reduces VLM memory from ~2 GB (BF16) to ~0.5 GB
> (4-bit), freeing headroom for larger batches or longer sequences. Keep
> `--batch_size=2` and `--num_workers=0`.

---

## Architecture Summary

```
SmolVLMWithExpertModel
├── vlm  (PeftModel wrapping SmolVLMForConditionalGeneration)
│   └── base_model
│       └── model  (SmolVLMModel)
│           ├── vision_model  (SigLIP)         [FROZEN — no LoRA]
│           ├── connector     (Linear + GELU)  [LoRA adapters, trainable]
│           └── text_model    (SmolLM)         [4-bit NF4 weights]
│               └── layers[i].self_attn.*_proj [LoRA adapters, trainable]
└── lm_expert  (action expert, full BF16 precision) [trainable]
```

---

## Comparison: LoRA vs QLoRA

| | LoRA branch | QLoRA branch (this) |
|---|---|---|
| VLM backbone precision | BF16 (~2 GB) | 4-bit NF4 (~0.5 GB) |
| VLM text model trainable | No (fully frozen) | Yes (via LoRA adapters) |
| Vision encoder trainable | No | No |
| Action expert trainable | Yes | Yes |
| Extra dependencies | `peft` | `peft`, `bitsandbytes` |
| Min VRAM | ~4 GB | ~3 GB |
