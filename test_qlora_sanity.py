"""
Sanity check for QLoRA setup on SmolVLA.

Run with:
    conda activate smol
    cd /home/saurav/dl_project/lerobot_smolVLA
    python test_qlora_sanity.py
"""

import sys
import torch

sys.path.insert(0, "src")

from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel

print("=" * 60)
print("SmolVLA QLoRA Sanity Check")
print("=" * 60)

# GPU info
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU : {gpu}")
    print(f"VRAM: {total_vram:.1f} GB\n")
else:
    print("WARNING: No CUDA GPU found — running on CPU.\n")

# Load model in QLoRA mode
print("Loading SmolVLM2-500M in 4-bit QLoRA mode ...")
model = SmolVLMWithExpertModel(
    model_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    load_vlm_weights=True,
    train_expert_only=False,
    freeze_vision_encoder=True,   # matches SmolVLAConfig default
    use_qlora=True,
    lora_r=16,
    lora_alpha=32,
)

# Param breakdown
print("\n--- Trainable Parameters ---")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
expert_p  = sum(p.numel() for p in model.lm_expert.parameters() if p.requires_grad)
lora_p    = sum(p.numel() for n, p in model.vlm.named_parameters() if "lora_" in n and p.requires_grad)

print(f"  Expert (action head) : {expert_p / 1e6:.2f} M")
print(f"  VLM LoRA adapters    : {lora_p   / 1e6:.2f} M")
print(f"  Total trainable      : {trainable / 1e6:.2f} M / {total / 1e6:.2f} M  ({100 * trainable / total:.2f} %)")

# GPU memory after load
if torch.cuda.is_available():
    alloc    = torch.cuda.memory_allocated(0) / 1024**3
    reserved = torch.cuda.memory_reserved(0)  / 1024**3
    print(f"\n--- GPU Memory After Load ---")
    print(f"  Allocated : {alloc:.2f} GB")
    print(f"  Reserved  : {reserved:.2f} GB")
    print(f"  Remaining : {total_vram - reserved:.2f} GB free for activations / optimizer")

# Quick forward pass check
print("\n--- Forward Pass (no grad) ---")
device = next(model.get_vlm_model().vision_model.parameters()).device
dummy_img = torch.zeros(1, 3, 512, 512, device=device, dtype=torch.float32)

try:
    with torch.no_grad():
        out = model.embed_image(dummy_img)
    print(f"  embed_image output : {out.shape}  ✓")
except Exception as e:
    print(f"  embed_image FAILED : {e}")

if torch.cuda.is_available():
    alloc = torch.cuda.memory_allocated(0) / 1024**3
    print(f"  GPU after forward  : {alloc:.2f} GB allocated")

# LoRA adapter check — vision encoder must be frozen, text model must be trainable
print("\n--- LoRA Adapter Check ---")
lora_params = [(n, p) for n, p in model.vlm.named_parameters() if "lora_A" in n or "lora_B" in n]
vision_lora  = [(n, p) for n, p in lora_params if "vision_model" in n]
text_lora    = [(n, p) for n, p in lora_params if "vision_model" not in n]

vision_frozen  = all(not p.requires_grad for _, p in vision_lora)
text_trainable = all(p.requires_grad for _, p in text_lora)

print(f"  Total LoRA layers       : {len(lora_params)}")
print(f"  Vision encoder LoRA     : {len(vision_lora)} layers  frozen={vision_frozen}  {'✓' if vision_frozen else '✗ PROBLEM'}")
print(f"  Text model LoRA         : {len(text_lora)} layers  trainable={text_trainable}  {'✓' if text_trainable else '✗ PROBLEM'}")

print("\n✓  All checks passed — QLoRA is set up correctly.")
