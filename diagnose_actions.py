"""
Quick diagnostic: load the LoRA policy and run a few env steps to see what
actions are being generated. This tells us if the policy is frozen, random, or reasonable.
"""
import json, os, sys
sys.path.insert(0, os.path.abspath("src"))

# ── same patches as eval_lora_smolvla.py ──────────────────────────────────────
from lerobot.policies import factory
_orig_make_policy = factory.make_policy
_DEFAULT_BASE = "lerobot/smolvla_base"

def _read_base_model_path(adapter_path):
    try:
        with open(os.path.join(adapter_path, "train_config.json")) as f:
            tc = json.load(f)
        base = tc.get("policy", {}).get("pretrained_path")
        if base: return str(base)
    except Exception:
        pass
    return _DEFAULT_BASE

def _load_input_features_from_train_config(adapter_path):
    from lerobot.configs.types import FeatureType, PolicyFeature
    try:
        with open(os.path.join(adapter_path, "train_config.json")) as f:
            tc = json.load(f)
        raw = tc.get("policy", {}).get("input_features")
        if not raw: return None
        return {k: PolicyFeature(type=FeatureType(v["type"]), shape=tuple(v["shape"])) for k, v in raw.items()}
    except Exception as e:
        print(f"Warning: {e}")
        return None

def _lora_make_policy(cfg, ds_meta=None, env_cfg=None, rename_map=None):
    from peft import PeftModel
    adapter_path = str(getattr(cfg, "pretrained_path", "") or "")
    original_use_peft = getattr(cfg, "use_peft", False)
    if adapter_path:
        train_feats = _load_input_features_from_train_config(adapter_path)
        if train_feats is not None:
            cfg.input_features = train_feats
    base_path = _read_base_model_path(adapter_path) if adapter_path else _DEFAULT_BASE
    print(f"Base model: {base_path}")
    print(f"LoRA path:  {adapter_path}")
    if hasattr(cfg, "pretrained_path"): cfg.pretrained_path = base_path
    if hasattr(cfg, "use_peft"):        cfg.use_peft = False
    policy = _orig_make_policy(cfg, ds_meta, env_cfg, rename_map)
    if hasattr(cfg, "pretrained_path"): cfg.pretrained_path = adapter_path
    if hasattr(cfg, "use_peft"):        cfg.use_peft = original_use_peft
    if original_use_peft and adapter_path:
        policy = PeftModel.from_pretrained(policy, adapter_path, is_trainable=False)
        print("LoRA adapters applied.")
    return policy

factory.make_policy = _lora_make_policy

from lerobot.datasets.lerobot_dataset import LeRobotDataset
_orig_init = LeRobotDataset.__init__
def _patched_init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    if hasattr(self, "meta") and hasattr(self.meta, "stats"):
        for k in self.meta.camera_keys:
            if k not in self.meta.stats: self.meta.stats[k] = {}
LeRobotDataset.__init__ = _patched_init

# ── actual diagnostic ─────────────────────────────────────────────────────────
import argparse
import numpy as np
import torch
from contextlib import nullcontext

parser = argparse.ArgumentParser()
parser.add_argument("--adapter_path", default="outputs/train/exp2_r4_d20/checkpoints/last/pretrained_model")
parser.add_argument("--n_steps", type=int, default=3)
parser.add_argument("--no_lora", action="store_true", help="Load base model only (no LoRA)")
args = parser.parse_args()

from lerobot.configs.eval import EvalPipelineConfig
from lerobot.configs import parser as lerobot_parser
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation, add_envs_task
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.utils.device_utils import get_safe_torch_device

# Build a minimal policy config
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig

adapter_path = args.adapter_path if not args.no_lora else None

if args.no_lora:
    # For base model test: load from smolvla_base but override input features
    # to use the renamed keys (camera1, camera2) from the adapter's train_config.
    cfg_policy = SmolVLAConfig(
        pretrained_path="lerobot/smolvla_base",
        use_peft=False,
        device="cuda",
        use_amp=True,
        vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    )
    # Override input features to use training-time key names
    from lerobot.configs.types import FeatureType, PolicyFeature
    cfg_policy.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(6,)),
        "observation.images.camera1": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
        "observation.images.camera2": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 256, 256)),
    }
    preprocessor_pretrained_path = args.adapter_path  # use adapter's normalizer stats
else:
    cfg_policy = SmolVLAConfig(
        pretrained_path=adapter_path,
        use_peft=True,
        device="cuda",
        use_amp=True,
        vlm_model_name="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    )
    preprocessor_pretrained_path = args.adapter_path

rename_map = {
    "observation.images.image": "observation.images.camera1",
    "observation.images.image2": "observation.images.camera2",
}

from lerobot.envs.configs import LiberoEnv as LiberoEnvCfg
env_cfg = LiberoEnvCfg(task="libero_10", task_ids=[0])

print("\n=== Loading policy ===")
policy = make_policy(cfg=cfg_policy, env_cfg=env_cfg, rename_map=rename_map)
policy.eval()

device = get_safe_torch_device("cuda")

print("\n=== Setting up preprocessor ===")
preprocessor_overrides = {
    "device_processor": {"device": "cuda"},
    "rename_observations_processor": {"rename_map": rename_map},
}
preprocessor, postprocessor = make_pre_post_processors(
    policy_cfg=cfg_policy,
    pretrained_path=preprocessor_pretrained_path,
    preprocessor_overrides=preprocessor_overrides,
)

print("\n=== Creating env ===")
from lerobot.envs.factory import make_env
envs = make_env(env_cfg, n_envs=1, use_async_envs=False)

env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg=env_cfg, policy_cfg=cfg_policy)

print("\n=== Running diagnostic steps ===")
# make_env returns {suite: {task_id: vec_env}}
vec_env = list(list(envs.values())[0].values())[0]

policy.reset()
obs, info = vec_env.reset(seed=[42])

for step in range(args.n_steps):
    obs = preprocess_observation(obs)
    obs = add_envs_task(vec_env, obs)
    obs = env_preprocessor(obs)
    obs = preprocessor(obs)

    print(f"\n--- Step {step} ---")
    # Print state info
    if "observation.state" in obs:
        s = obs["observation.state"]
        print(f"  State shape: {tuple(s.shape)}, mean={s.float().mean().item():.4f}, std={s.float().std().item():.4f}")

    with torch.inference_mode(), torch.autocast(device_type="cuda") if cfg_policy.use_amp else nullcontext():
        action = policy.select_action(obs)

    action_post = postprocessor(action)
    from lerobot.utils.constants import ACTION
    raw_action = action_post[ACTION] if isinstance(action_post, dict) else action_post
    raw_np = raw_action.cpu().numpy()
    print(f"  Raw action (post-unnorm): shape={raw_np.shape}")
    print(f"    mean={raw_np.mean():.4f}, std={raw_np.std():.4f}, min={raw_np.min():.4f}, max={raw_np.max():.4f}")
    print(f"    values={np.round(raw_np.flatten()[:7], 3).tolist()}")

    # Step env
    action_np = raw_np
    if action_np.ndim == 1:
        action_np = action_np[np.newaxis, :]
    obs, reward, terminated, truncated, info = vec_env.step(action_np)
    print(f"  Reward: {reward}, Done: {terminated}")

print("\nDiagnostic complete.")
