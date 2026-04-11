import sys
import os

# Make sure Python sees src/
sys.path.insert(0, os.path.abspath("src"))

# ==============================================================================
# 1. PATCH FACTORY
#
# Two modes depending on whether a LoRA adapter checkpoint exists at
# pretrained_path:
#
#   TRAINING  (no adapter_config.json at pretrained_path):
#     - Load base model fresh (use_peft=False, pretrained_path=base hub id)
#     - Wrap with a brand-new LoRA using LORA_RANK env var (default 8)
#     - Return trainable PEFT model
#
#   EVAL  (adapter_config.json present at pretrained_path):
#     - Load base model fresh
#     - Load saved LoRA weights from pretrained_path (frozen)
#     - Return frozen PEFT model for inference
# ==============================================================================
from lerobot.policies import factory
original_make_policy = factory.make_policy

def custom_make_policy(cfg, ds_meta=None, env_cfg=None, rename_map=None):
    lora_r = int(os.environ.get('LORA_RANK', 8))

    original_use_peft = getattr(cfg, 'use_peft', False)
    original_pretrained_path = getattr(cfg, 'pretrained_path', None)

    # Step 1: load the bare base model (no PEFT, no checkpoint)
    cfg.use_peft = False
    cfg.pretrained_path = None
    policy = original_make_policy(cfg, ds_meta, env_cfg, rename_map)

    # Restore cfg so downstream lerobot code sees the original values
    cfg.use_peft = original_use_peft
    cfg.pretrained_path = original_pretrained_path

    if not original_use_peft:
        return policy

    # Step 2: decide training vs eval
    from pathlib import Path
    adapter_config_path = None
    if original_pretrained_path is not None:
        candidate = Path(str(original_pretrained_path)) / "adapter_config.json"
        if candidate.exists():
            adapter_config_path = candidate

    if adapter_config_path is not None:
        # ------------------------------------------------------------------
        # EVAL: load saved LoRA adapter weights
        # ------------------------------------------------------------------
        from peft import PeftModel
        print(f"\n>>> [EVAL] Loading LoRA adapters from: {original_pretrained_path} <<<\n")
        policy = PeftModel.from_pretrained(
            policy,
            str(original_pretrained_path),
            is_trainable=False,
        )
    else:
        # ------------------------------------------------------------------
        # TRAINING: apply a fresh LoRA with the requested rank
        # ------------------------------------------------------------------
        from peft import LoraConfig, get_peft_model

        common_projections = (
            "state_proj|action_in_proj|action_out_proj"
            "|action_time_mlp_in|action_time_mlp_out"
        )
        target_modules = (
            rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj"
            rf"|model\.({common_projections}))"
        )

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,   # common default: alpha = 2 * r
            target_modules=target_modules,
            modules_to_save=[],
            bias="none",
        )

        # Freeze entire base model; LoRA params will be the only trainables
        for p in policy.parameters():
            p.requires_grad_(False)

        policy = get_peft_model(policy, lora_config)

        print(f"\n>>> [TRAIN] Applied fresh LoRA (rank={lora_r}) <<<")
        policy.print_trainable_parameters()
        print()

    return policy

factory.make_policy = custom_make_policy


# ==============================================================================
# 2. PATCH DATASET  (fixes missing camera stats key on first load)
# ==============================================================================
from lerobot.datasets.lerobot_dataset import LeRobotDataset
original_init = LeRobotDataset.__init__

def patched_init(self, *args, **kwargs):
    original_init(self, *args, **kwargs)
    if hasattr(self, 'meta') and hasattr(self.meta, 'stats'):
        for key in self.meta.camera_keys:
            if key not in self.meta.stats:
                self.meta.stats[key] = {}

LeRobotDataset.__init__ = patched_init


# ==============================================================================
# 3. IMPORT TRAINER
# ==============================================================================
try:
    from lerobot.scripts.lerobot_train import main as train_cli
except ImportError:
    from lerobot.scripts.train import main as train_cli


# ==============================================================================
# 4. RUN
# ==============================================================================
if __name__ == "__main__":
    train_cli()
