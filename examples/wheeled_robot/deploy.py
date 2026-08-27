"""Export this trained policy for a real robot.

Produces a deployment bundle -- the observation and action pipelines captured from
the built environment, plus the policy as ONNX -- that the simulation-free
``genesis-forge-deploy`` runtime replays on the robot::

    uv run python deploy.py                  # after training

Before anything is written, the export runs the deployment classes against the live
training pipeline and refuses to write a bundle if they disagree.
"""

import argparse
import glob
import os
import pickle
import sys

import genesis as gs
import torch
from environment import WheeledRobotCommandDirectionEnv
from genesis_forge_deploy import load_bundle
from rsl_rl.runners import OnPolicyRunner

from genesis_forge.deployment import export
from genesis_forge.wrappers import RslRlWrapper

EXPERIMENT_NAME = "wheeled-robot-command"

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-e", "--exp_name", type=str, default=EXPERIMENT_NAME)
parser.add_argument(
    "-o", "--output", type=str, default="./deploy_bundle", help="Bundle directory."
)
args = parser.parse_args()


def get_latest_model(log_dir: str) -> str:
    """
    Get the last model checkpoint from the log directory
    """
    model_checkpoints = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if len(model_checkpoints) == 0:
        print(
            f"Warning: No model files found at '{log_dir}' (you might need to train more)."
        )
        sys.exit(1)
    # Sort by the file with the highest number
    sorted_models = sorted(
        model_checkpoints,
        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]),
    )
    return sorted_models[-1]


def get_onnx_policy(env, log_path: str):
    """Load the trained policy and export it to ONNX.

    Returns the ONNX path, a callable for the parity check, and the checkpoint used.
    """
    with open(f"{log_path}/cfgs.pkl", "rb") as f:
        [cfg] = pickle.load(f)
    checkpoint = get_latest_model(log_path)

    print("🎬 Loading last model...")
    runner = OnPolicyRunner(env, cfg, log_path, device=gs.device)
    runner.load(checkpoint)

    # Export an onnx policy directly from rsl_rl
    policy_filename = "policy.onnx"
    runner.export_policy_to_onnx(path=log_path, filename=policy_filename)
    onnx_path = os.path.join(log_path, policy_filename)

    # Get the trained reference policy
    reference_policy = runner.get_inference_policy(device="cpu").as_onnx(verbose=False)

    return onnx_path, reference_policy, checkpoint


def main():
    # Initialize Genesis
    backend = gs.cpu
    torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend)


    # A single environment is enough: export reads the pipeline configuration, not
    # a rollout, and a bundle describes one robot.
    env = WheeledRobotCommandDirectionEnv(num_envs=1, headless=True)
    env = RslRlWrapper(env)
    env.build()

    # Export the trained policy to ONNX
    onnx_path = None
    reference_policy = None
    checkpoint = None
    log_path = f"./logs/{args.exp_name}"
    onnx_path, reference_policy, checkpoint = get_onnx_policy(env, log_path)

    # `export` wants the environment itself, not the training-framework wrapper.
    bundle_path = export(
        env.unwrapped,
        args.output,
        policy_path=onnx_path,
        reference_policy=reference_policy,
        checkpoint=checkpoint,
        overwrite=True,
    )

    print()
    print(load_bundle(bundle_path).describe())
    print()
    print("Copy this folder to the robot, then: pip install genesis-forge-deploy[onnx]")


if __name__ == "__main__":
    main()
