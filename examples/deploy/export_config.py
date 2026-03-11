"""
Export deployment artifacts from a trained Go2 locomotion policy.

Produces two files in the log directory:
  deploy_config.json  -- observation/action pipeline config (no Genesis required at deploy time)
  policy.pt           -- TorchScript-traced policy (no RSL-RL required at deploy time)

Usage::

    python export_config.py
    python export_config.py -e go2-simple -d cpu
"""

import os
import glob
import argparse
import pickle
from importlib import metadata

import torch
import genesis as gs

from genesis_forge.wrappers import RslRlWrapper
from genesis_forge.deploy import export
from environment import Go2Env

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib").startswith("1."):
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install 'rsl-rl-lib>=2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

EXPERIMENT_NAME = "go2-simple"

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-d", "--device", type=str, default="gpu")
parser.add_argument("-e", "--exp_name", type=str, default=EXPERIMENT_NAME)
args = parser.parse_args()


def get_latest_model(log_dir: str) -> str:
    model_checkpoints = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if not model_checkpoints:
        print(f"No model files found at '{log_dir}'. Train first with train.py.")
        exit(1)
    return sorted(
        model_checkpoints,
        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]),
    )[-1]


def main():
    backend = gs.gpu if args.device == "gpu" else gs.cpu
    if args.device == "cpu":
        torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend)

    log_path = f"./logs/{args.exp_name}"
    model_path = get_latest_model(log_path)
    [cfg] = pickle.load(open(f"{log_path}/cfgs.pkl", "rb"))

    # ------------------------------------------------------------------
    # Build environment and load the trained policy
    # ------------------------------------------------------------------
    env = Go2Env(num_envs=1, headless=True)
    env = RslRlWrapper(env)
    env.build()

    runner = OnPolicyRunner(env, cfg, log_path, device=gs.device)
    runner.load(model_path)
    policy = runner.get_inference_policy(device=gs.device)

    # ------------------------------------------------------------------
    # 1. Export the deployment config
    #    This captures the full observation/action pipeline so the robot
    #    can reconstruct it without any Genesis dependency.
    # ------------------------------------------------------------------
    config_path = os.path.join(log_path, "deploy_config.json")
    deploy_config = export(env, path=config_path)
    print(f"Deploy config written to: {config_path}")
    print(
        f"  {len(deploy_config.observations)} observation slots, "
        f"total dim={deploy_config.num_observations}"
    )
    print(
        f"  {len(deploy_config.action_managers)} action manager(s), "
        f"total actions={deploy_config.num_actions}"
    )
    for am in deploy_config.action_managers:
        print(f"    [{am.type}] joints: {am.params['joint_names']}")

    # ------------------------------------------------------------------
    # 2. Export the policy as TorchScript
    #    Always trace on CPU so the saved file is device-agnostic and
    #    can be loaded on hardware without a GPU (e.g. Raspberry Pi,
    #    Jetson in CPU mode, onboard ARM SoC).
    # ------------------------------------------------------------------
    policy_cpu = policy.cpu()
    dummy_obs = torch.zeros(1, deploy_config.num_observations)  # CPU tensor
    with torch.no_grad():
        traced = torch.jit.trace(policy_cpu, dummy_obs)

    policy_path = os.path.join(log_path, "policy.pt")
    traced.save(policy_path)
    print(f"TorchScript policy written to: {policy_path}  (CPU, device-agnostic)")
    print()
    print("Ready to deploy. Run:")
    print(f"  python robot.py -e {args.exp_name}")


if __name__ == "__main__":
    main()
