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


class FlorenceVisionEncoder(torch.nn.Module):
    def __init__(
        self, 
        model_id='microsoft/Florence-2-base', 
        hidden_size=2048,  # Target LLM dimension 
        use_pixel_shuffle=True, 
        factor=3
    ):
        super().__init__()
        print(f'Loading {model_id} for vision encoder...')
        self.florence = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True, 
            attn_implementation='eager'
        )
        
        # Freeze the entire Florence model (including vision tower, embeddings, logic)
        for param in self.florence.parameters():
            param.requires_grad = False
            
        self.use_pixel_shuffle = use_pixel_shuffle
        
        # The output of Florence's `_encode_image` is typically 768 for Florence-2-base
        davit_out_dim = 768
        
        if self.use_pixel_shuffle:
            self.token_reducer = TokenReducer(
                in_features=davit_out_dim, 
                out_features=davit_out_dim, 
                factor=factor
            )
            
        # Linear projection to LLM hidden dimension
        self.proj = torch.nn.Linear(davit_out_dim, hidden_size)
        
        # Final LayerNorm
        self.norm = torch.nn.LayerNorm(hidden_size)

    def forward(self, pixel_values):
        """
        Args:
            pixel_values: Tensor of shape [B, C, H, W] representing images
        Returns:
            Tensor of shape [B, T, D_lm] with projected visual tokens
        """
        # Get DaViT tokens + positional embeddings via Florence's official _encode_image
        with torch.no_grad():
            x = self.florence._encode_image(pixel_values)
            
        # x is originally [B, 577, 768] (with factor=3 and 24x24 patches)
        if self.use_pixel_shuffle:
            x = self.token_reducer(x)
            
        # Linear projection to [B, T, D_lm]
        x = self.proj(x)
        
        # LayerNorm
        x = self.norm(x)
        
        return x

if __name__ == "__main__":
    # Test FlorenceVisionEncoder
    encoder = FlorenceVisionEncoder(hidden_size=2048)
    
    dummy_images = torch.randn(2, 3, 768, 768)
    
    print('Testing FlorenceVisionEncoder forward pass...')
    out = encoder(dummy_images)
    
    print('Input images shape:', dummy_images.shape)
    print('Output visual tokens shape:', out.shape)
    assert out.shape == (2, 64, 2048), f"Expected shape (2, 64, 2048), got {out.shape}"
    print("Passed!")
