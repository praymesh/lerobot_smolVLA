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

vision_tower = florence.model.vision_tower
print('Vision tower isolated:', type(vision_tower).__name__)
# Freeze the vision tower
for param in vision_tower.parameters():
    param.requires_grad = False

dummy = torch.randn(1, 3, 768, 768)

with torch.no_grad():
    res = florence._encode_image(dummy)
    print('Output shape:', res.shape)
