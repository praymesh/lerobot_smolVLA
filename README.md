# SmolVLA with Florence-2 Vision Backbone

Replacing the default SigLIP vision encoder inside SmolVLA with
**Florence-2-base** (Microsoft), while keeping the SmolVLM-500M language model
and action expert untouched.

---

## What Changed

### Original SmolVLA architecture

```
Image → SigLIP encoder → SmolVLM connector → SmolLM text model
                                              + action expert (lm_expert)
```

### This branch (Florence-2 swap)

```
Image → Florence-2-base (DaViT) → new connector (Linear + LayerNorm) → SmolLM text model
                                                                         + action expert (lm_expert)
```

### Dimension bridge

Florence-2-base's DaViT outputs **768-dimensional** patch features.
SmolVLM2-500M's text model has a **960-dimensional** hidden size.
The original SmolVLM connector assumed its own encoder's output dimensions, so
it was replaced with:

```python
nn.Sequential(
    nn.Linear(768, 960, dtype=torch.bfloat16),
    nn.LayerNorm(960, dtype=torch.bfloat16),
)
```

Weights are Xavier-initialized to keep activations stable at the start of training.

### Token count

Florence-2 processes images at 768×768 with patch size 32, yielding 24×24 = 576
spatial tokens. An extra CLS token (index 0) is stripped, leaving **576 tokens**
passed to the language model.

---

## Key Files Modified

| File | What changed |
|---|---|
| `src/lerobot/policies/smolvla/smolvlm_with_expert.py` | `FlorenceVisionEncoder` class added; `SmolVLMWithExpertModel.__init__` swaps the encoder and connector |
| `src/lerobot/policies/smolvla/configuration_smolvla.py` | `vision_backbone` parameter added; default VLM downsized to `SmolVLM2-236M-Video-Instruct` |
| `eval_florence.sh` | Eval driver for LIBERO (Spatial / Goal / Long) and MetaWorld |
| `florence_sanity.py` | Standalone sanity-check for `FlorenceVisionEncoder` |
| `sanity.py` | Minimal check that Florence-2 loads and `_encode_image` runs |

---

## Compatibility Patches

Florence-2 requires `trust_remote_code=True` and conflicts with several
assumptions in modern `transformers`. The following patches are applied inside
`FlorenceVisionEncoder.__init__` before loading:

| Patch | Reason |
|---|---|
| `transformers.dynamic_module_utils.check_imports = lambda: []` | Florence remote code triggers import checks that fail on some systems |
| `PretrainedConfig.forced_bos_token_id = None` | Missing attribute in newer transformers breaks Florence config |
| `is_flash_attn_greater_or_equal_2_10 = lambda: False` | Flash attention not compatible with DaViT; forces eager attention |
| `PreTrainedModel._supports_sdpa = False` | Same — disables SDPA path |
| `torch.linspace` patched to `device='cpu'` | DaViT calls `.item()` on a linspace tensor during `__init__`; meta-device init in transformers causes this to crash — temporarily forces CPU |
| `low_cpu_mem_usage=False`, `_fast_init=False`, `device_map=None` | Blocks meta-tensor context managers that interfere with the linspace patch |

---

## Setup

```bash
# 1. Install base LeRobot + SmolVLA
pip install -e ".[smolvla]"

# 2. Install additional requirements
pip install -r requirements.txt
```

Florence-2 is downloaded automatically on first use via `trust_remote_code=True`.
For faster download:

```bash
export HF_HUB_ENABLE_HF_TRANSFER=1
```

---

## Sanity Checks

Run these before any training or eval to confirm Florence-2 loads correctly:

```bash
# Check Florence-2 _encode_image works standalone
python sanity.py

# Check FlorenceVisionEncoder output shape
python florence_sanity.py
```

Expected output from `florence_sanity.py`:
```
Loading microsoft/Florence-2-base for vision encoder...
Testing FlorenceVisionEncoder forward pass...
Input images shape: torch.Size([2, 3, 768, 768])
Output visual tokens shape: torch.Size([2, 64, 2048])
Passed!
```

---

## Training

Training uses the same LeRobot training pipeline as the base SmolVLA.
The Florence backbone is **frozen by default** (`freeze_vision_encoder=True`);
only the new connector, the language model, and the action expert are updated.

```bash
lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --policy.push_to_hub=false \
  --policy.vision_backbone=microsoft/Florence-2-base \
  --dataset.repo_id=<your_dataset> \
  --rename_map='{"observation.images.top":"observation.images.camera1"}' \
  --batch_size=2 \
  --steps=40000 \
  --output_dir=outputs/train/florence_run \
  --job_name=florence_smolvla \
  --policy.device=cuda \
  --policy.use_amp=true
```

> **RTX 3050 6 GB tip:** keep `--batch_size=2`, `--policy.use_amp=true`, and
> `--num_workers=0`. Florence-2 adds ~1.5 GB of VRAM versus SigLIP due to the
> larger DaViT model staying in memory even when frozen.

To train on **LIBERO** or **ALOHA**, set the matching dataset and rename map:

```bash
# LIBERO
--dataset.repo_id=lerobot/libero
--rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'

# ALOHA sim
--dataset.repo_id=lerobot/aloha_sim_transfer_cube_human
--rename_map='{"observation.images.top":"observation.images.camera1","observation.images.wrist":"observation.images.camera2"}'
```

---

## Evaluation

```bash
./eval_florence.sh
```

The script evaluates on four benchmarks sequentially:

| Suite | Tasks | Episodes / task |
|---|---|---|
| LIBERO-Spatial | 10 | 10 |
| LIBERO-Goal | 10 | 10 |
| LIBERO-Long (`libero_10`) | 10 | 10 |
| MetaWorld (easy / medium / hard / very hard) | 4 groups | 10 |

Each task runs with `--eval.batch_size=1` to avoid OOM on 4–6 GB GPUs.
EGL/rendering noise is filtered from stdout to keep output readable.

To point at your own Florence-trained checkpoint, edit the top of `eval_florence.sh`:

```bash
CHECKPOINT_LIBERO="outputs/train/florence_run/checkpoints/last/pretrained_model"
```

Results are aggregated at the end showing per-suite and overall success rates.

---

## Architecture Summary

```
SmolVLMWithExpertModel
├── vlm  (SmolVLMForConditionalGeneration)
│   └── model
│       ├── vision_model  ← FlorenceVisionEncoder  [FROZEN]
│       │   └── base_model  (Florence-2-base, trust_remote_code)
│       │       └── vision_tower  (DaViT)  → [B, 576, 768]
│       ├── connector  ← NEW: Linear(768→960) + LayerNorm(960)  [trainable]
│       └── text_model  (SmolLM, 960-dim hidden)  [trainable]
└── lm_expert  (action expert transformer, 720-dim hidden)  [trainable]
```

---

## Notes

- **DaViT vs SigLIP**: Florence-2 uses DaViT pretrained with grounding +
  captioning objectives, which may provide stronger spatial/object understanding
  for manipulation tasks compared to SigLIP's contrastive pretraining.
- **Image resolution**: Florence expects **768×768** input. Set
  `resize_imgs_with_padding: (768, 768)` in `SmolVLAConfig` when training to
  avoid rescaling artifacts.
- **VLM size**: This branch defaults to `SmolVLM2-236M-Video-Instruct` to save
  VRAM. Switch to `SmolVLM2-500M-Video-Instruct` in `configuration_smolvla.py`
  for full capacity.
- **Connector init**: Xavier uniform on the linear weight + zero bias ensures
  the new connector does not produce large activations at the start of training,
  which would otherwise destabilize the pretrained text model.
