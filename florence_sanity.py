import torch
import transformers.utils
transformers.utils.is_flash_attn_greater_or_equal_2_10 = lambda: False
transformers.utils.is_flash_attn_2_available = lambda: False

import transformers.dynamic_module_utils as dyn_utils
dyn_utils.check_imports = lambda filename: []

import transformers.configuration_utils as config_utils
if not hasattr(config_utils.PretrainedConfig, 'forced_bos_token_id'):
    config_utils.PretrainedConfig.forced_bos_token_id = None

from transformers.modeling_utils import PreTrainedModel
if not hasattr(PreTrainedModel, '_supports_sdpa'):
    PreTrainedModel._supports_sdpa = False

from transformers import AutoModelForCausalLM

import warnings
warnings.filterwarnings('ignore')

model_id = 'microsoft/Florence-2-base'
print('Loading model...')
florence = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, attn_implementation='eager')

if hasattr(florence, 'model') and hasattr(florence.model, 'vision_tower'):
    vision_tower = florence.model.vision_tower
else:
    vision_tower = florence.vision_tower

print('Vision tower isolated:', type(vision_tower).__name__)

# Freeze the vision tower
for param in vision_tower.parameters():
    param.requires_grad = False

class TokenReducer(torch.nn.Module):
    def __init__(self, in_features=768, out_features=768, factor=3):
        super().__init__()
        # PixelUnshuffle decreases spatial dim by factor, increases channels by factor^2
        self.unshuffle = torch.nn.PixelUnshuffle(factor)
        self.proj = torch.nn.Linear(in_features * (factor ** 2), out_features)

    def forward(self, x):
        # x is [B, L, C] where L = 577 (1 CLS + 576 patches for 24x24)
        B, L, C = x.shape
        num_patches = L - 1
        H = int(num_patches ** 0.5)
        W = H
        
        # Isolate patches and reshape to [B, C, H, W]
        patches = x[:, 1:, :].transpose(1, 2).view(B, C, H, W)
        
        # Apply PixelUnshuffle -> [B, C * 9, H/3, W/3] (e.g., 8x8 spatial = 64 tokens)
        reduced = self.unshuffle(patches)
        
        # Flatten spatial dims to sequence -> [B, 64, C * 9]
        reduced = reduced.flatten(2).transpose(1, 2)
        
        return self.proj(reduced)

token_reducer = TokenReducer(in_features=768, out_features=768, factor=3)

dummy = torch.randn(1, 3, 768, 768)

with torch.no_grad():
    res = florence._encode_image(dummy)
    print('Original output shape:', res.shape)
    
    reduced_res = token_reducer(res)
    print('Reduced token output shape:', reduced_res.shape)
