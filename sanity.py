"""
Florence-2 vision tower sanity check
Run: python check_florence_vision.py
"""

import torch
from transformers import AutoModelForCausalLM, AutoProcessor

# ── 1. Load ───────────────────────────────────────────────────────────────────
print("Loading Florence-2-base...")

model = AutoModelForCausalLM.from_pretrained(
    "microsoft/Florence-2-base",
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map="cpu",
    attn_implementation="eager",
)
model.eval()
print(f"Model loaded OK  —  {type(model).__name__}")

# ── 2. Extract components (confirmed attribute names) ────────────────────────
vision_tower          = model.vision_tower           # DaViT encoder
image_proj_norm       = model.image_proj_norm        # LayerNorm after projection
image_pos_embed       = model.image_pos_embed        # 2-D positional embedding
visual_temporal_embed = model.visual_temporal_embed  # temporal embed (multi-frame)
language_model        = model.language_model         # BART encoder-decoder (NOT used in SmolVLA swap)

print(f"\nComponents confirmed:")
print(f"  vision_tower          : {type(vision_tower).__name__}")
print(f"  image_proj_norm       : {type(image_proj_norm).__name__}")
print(f"  image_pos_embed       : {type(image_pos_embed).__name__}")
print(f"  visual_temporal_embed : {type(visual_temporal_embed).__name__}")
print(f"  language_model        : {type(language_model).__name__}  ← NOT used in SmolVLA swap")

# ── 3. Dummy image forward through vision_tower ───────────────────────────────
# Florence-2 base uses 768x768 internally but accepts other sizes.
# 224x224 is fine for a shape check.
B, C, H, W = 1, 3, 224, 224
dummy = torch.randn(B, C, H, W)
print(f"\nDummy input : {dummy.shape}")

with torch.no_grad():
    vout = vision_tower(dummy)

# vision_tower returns BaseModelOutput — features in .last_hidden_state
features = vout.last_hidden_state if hasattr(vout, "last_hidden_state") else vout[0]
print(f"vision_tower output    : {features.shape}")
print(f"  → [batch={features.shape[0]}, num_patches={features.shape[1]}, d_vision={features.shape[2]}]")

# ── 4. Apply image_proj_norm ──────────────────────────────────────────────────
with torch.no_grad():
    normed = image_proj_norm(features)
print(f"After image_proj_norm  : {normed.shape}")
print(f"  → d_model = {normed.shape[2]}  (this is what you project into SmolLM2's hidden dim)")

# ── 5. Summary for SmolVLA integration ───────────────────────────────────────
print("\n── SmolVLA integration summary ──────────────────────────────────────────")
print(f"  DaViT output d_model  : {features.shape[2]}")
print(f"  Num visual tokens     : {features.shape[1]}  (at {H}x{W} input)")
print(f"  SmolLM2 hidden dim    : 960  (from SmolVLA config)")
print(f"  Projection needed     : nn.Linear({features.shape[2]}, 960)")
print(f"  Token reduction       : optional PixelShuffle {features.shape[1]} → 64 tokens")

# ── 6. Parameter counts ───────────────────────────────────────────────────────
vp = sum(p.numel() for p in vision_tower.parameters())
tp = sum(p.numel() for p in model.parameters())
print(f"\n  vision_tower params   : {vp/1e6:.1f}M")
print(f"  full model params     : {tp/1e6:.1f}M")
print(f"  language_model params : {(tp-vp)/1e6:.1f}M  (discarded in swap)")

print("\nSanity check complete. Proceed to Phase 2 — build FlorenceVisionEncoder wrapper.")