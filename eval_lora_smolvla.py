"""
Evaluation script for SmolVLA + LoRA adapters on LIBERO.

Background
----------
Training was run with --policy.path=lerobot/smolvla_base, which loads the
pretrained SmolVLA base (full VLM + pretrained action expert) and then wraps it
with LoRA via policy.wrap_with_peft().  The saved adapter_config.json has
base_model_name_or_path=null because the config's pretrained_path was cleared
before wrap_with_peft was called — but the model weights were already loaded
from smolvla_base.

So for correct eval we need to:
  1. Load lerobot/smolvla_base as the full base model (VLM + action expert).
  2. Apply the trained LoRA delta weights on top.

The base model path is read from train_config.json inside the adapter checkpoint
so we automatically replicate whatever base was used at training time.

Usage (driven by auto_eval.sh):
  python eval_lora_smolvla.py \\
    --policy.type=smolvla \\
    --policy.pretrained_path=outputs/train/exp2_r4_d20/checkpoints/last/pretrained_model \\
    --policy.use_peft=true \\
    --policy.vlm_model_name=HuggingFaceTB/SmolVLM2-500M-Video-Instruct \\
    --policy.device=cuda \\
    --policy.use_amp=true \\
    --env.type=libero \\
    --env.task_ids="[0]" \\
    --eval.n_episodes=5 \\
    --eval.batch_size=1 \\
    --rename_map='{"observation.images.image":"observation.images.camera1","observation.images.image2":"observation.images.camera2"}'
"""

import json
import os
import sys

sys.path.insert(0, os.path.abspath("src"))

# ==============================================================================
# PATCH 1: factory.make_policy
#
# The adapter checkpoint does NOT store the base model path in adapter_config
# (base_model_name_or_path=null).  We recover the correct base from the
# train_config.json saved alongside the adapter weights.
#
# Steps:
#   a) Read the base model path from train_config.json (defaults to
#      lerobot/smolvla_base if not found).
#   b) Load the full pretrained base model from that path.
#   c) Apply the trained LoRA adapters on top with PeftModel.from_pretrained.
# ==============================================================================
from lerobot.policies import factory

_orig_make_policy = factory.make_policy

# Fallback base model if train_config.json is absent or has no pretrained_path
_DEFAULT_BASE = "lerobot/smolvla_base"


def _read_base_model_path(adapter_path: str) -> str:
    """Read the base model path that was used during training."""
    train_cfg_file = os.path.join(adapter_path, "train_config.json")
    if os.path.isfile(train_cfg_file):
        try:
            with open(train_cfg_file) as f:
                train_cfg = json.load(f)
            base = train_cfg.get("policy", {}).get("pretrained_path")
            if base:
                return str(base)
        except Exception:
            pass
    return _DEFAULT_BASE


def _load_input_features_from_train_config(adapter_path: str):
    """
    Read the exact input_features dict that was used at training time.
    Returns a dict of {key: PolicyFeature} or None if unavailable.

    Why: the factory derives input_features from raw env feature names
    (image/image2) without applying the rename_map, so the policy ends up
    looking for the wrong keys in the batch.  Loading from train_config gives
    the renamed names (camera1/camera2) that the policy was actually trained on.
    """
    from lerobot.configs.types import FeatureType, PolicyFeature

    train_cfg_file = os.path.join(adapter_path, "train_config.json")
    if not os.path.isfile(train_cfg_file):
        return None
    try:
        with open(train_cfg_file) as f:
            train_cfg = json.load(f)
        raw = train_cfg.get("policy", {}).get("input_features")
        if not raw:
            return None
        return {
            k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"]))
            for k, v in raw.items()
        }
    except Exception as e:
        print(f"[eval_lora_smolvla] Warning: could not load input_features from train_config: {e}")
        return None


def _lora_make_policy(cfg, ds_meta=None, env_cfg=None, rename_map=None):
    """
    Build the pretrained SmolVLA base, then apply the saved LoRA adapters.
    """
    from peft import PeftModel

    adapter_path = str(getattr(cfg, "pretrained_path", "") or "")
    original_use_peft = getattr(cfg, "use_peft", False)

    # ------------------------------------------------------------------ #
    # Step 0 – override input_features from training config               #
    #                                                                      #
    # The factory sets input_features from raw env feature names, which   #
    # do NOT include the rename_map.  We pre-populate cfg.input_features  #
    # with the training-time names (camera1/camera2) so the factory's     #
    # "if not cfg.input_features:" branch is skipped.                     #
    # ------------------------------------------------------------------ #
    if adapter_path:
        train_feats = _load_input_features_from_train_config(adapter_path)
        if train_feats is not None:
            cfg.input_features = train_feats

    # ------------------------------------------------------------------ #
    # Step 1 – load the FULL pretrained base (e.g. lerobot/smolvla_base)  #
    # ------------------------------------------------------------------ #
    base_path = _read_base_model_path(adapter_path) if adapter_path else _DEFAULT_BASE
    print(f"\n>>> Base model : {base_path}")
    print(f">>> LoRA path  : {adapter_path}\n")

    # Tell the factory to load the pretrained base; disable PEFT so it doesn't
    # try to interpret the adapter checkpoint as a base model.
    if hasattr(cfg, "pretrained_path"):
        cfg.pretrained_path = base_path
    if hasattr(cfg, "use_peft"):
        cfg.use_peft = False

    policy = _orig_make_policy(cfg, ds_meta, env_cfg, rename_map)

    # ------------------------------------------------------------------ #
    # Step 2 – apply saved LoRA adapters                                  #
    # ------------------------------------------------------------------ #
    # Restore the original values in cfg so callers see a consistent state.
    if hasattr(cfg, "pretrained_path"):
        cfg.pretrained_path = adapter_path
    if hasattr(cfg, "use_peft"):
        cfg.use_peft = original_use_peft

    if original_use_peft and adapter_path:
        policy = PeftModel.from_pretrained(
            policy,
            adapter_path,
            is_trainable=False,
        )
        try:
            policy.print_trainable_parameters()
        except Exception:
            pass

    return policy


factory.make_policy = _lora_make_policy


# ==============================================================================
# PATCH 2: LeRobotDataset — fill in missing camera stats (HF dataset quirk)
# ==============================================================================
from lerobot.datasets.lerobot_dataset import LeRobotDataset

_orig_dataset_init = LeRobotDataset.__init__


def _patched_dataset_init(self, *args, **kwargs):
    _orig_dataset_init(self, *args, **kwargs)
    if hasattr(self, "meta") and hasattr(self.meta, "stats"):
        for key in self.meta.camera_keys:
            if key not in self.meta.stats:
                self.meta.stats[key] = {}


LeRobotDataset.__init__ = _patched_dataset_init


# ==============================================================================
# Run the eval CLI  (NOT the training CLI — steps=0 runs zero eval iterations)
# ==============================================================================
from lerobot.scripts.lerobot_eval import main as eval_main

if __name__ == "__main__":
    eval_main()
