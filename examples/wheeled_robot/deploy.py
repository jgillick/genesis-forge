"""Export a trained policy for use on a real robot.

Produces a deployment bundle -- the observation and action pipelines captured from
the built environment, plus the policy as ONNX -- that the simulation-free
``genesis-forge-runtime`` runtime replays on the robot::

    uv run python deploy.py  # after training
"""

import argparse
import os
from importlib.metadata import version

import genesis as gs
import numpy as np
from pathlib import Path
import onnxruntime
import torch
from environment import WheeledRobotCommandDirectionEnv
from rsl_rl.runners import OnPolicyRunner

from genesis_forge.deployment import export
from genesis_forge.wrappers import RslRlWrapper

from utils import get_latest_model, load_config_pickle

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-e", "--exp_name", type=str, default="wheeled-robot-command")
parser.add_argument(
    "-o",
    "--output",
    type=str,
    default="./on_robot/trained_bundle",
    help="Where to write the bundle.",
)
args = parser.parse_args()


def create_onnx_policy(env, log_path: str):
    """Load the trained policy and export it to ONNX.

    Returns the policy's file(s), the policy runner, and the
    checkpoint used.
    """
    cfg = load_config_pickle(log_path)
    checkpoint = get_latest_model(log_path)

    print("🎬 Loading last model...")
    runner = OnPolicyRunner(env, cfg, log_path, device=gs.device)
    runner.load(checkpoint)

    # Export an onnx policy from rsl_rl
    policy_filename = "policy.onnx"
    runner.export_policy_to_onnx(path=log_path, filename=policy_filename)
    onnx_path = os.path.join(log_path, policy_filename)
    policy_files = [onnx_path]

    # Check if onnx exported a companion data file
    weights = f"{onnx_path}.data"
    if os.path.isfile(weights):
        policy_files.append(weights)

    return policy_files, runner, checkpoint


def verify_onnx_policy(bundle, runner: OnPolicyRunner, rtol=1e-4, atol=1e-5):
    """Check the bundle's ONNX graph against the policy it was exported from.

    Genesis Forge proves the observation and action pipelines match, but whether
    the graph itself survived export depends on your training framework, so it is
    left to you. A normalizer that silently failed to make it into the graph is
    the classic sim-to-real failure, and nothing else catches it.

    Runs against the copy inside the bundle, so a companion file left out of
    ``policy_path`` fails here rather than on the robot. The tolerance is
    relative because export reorders floating-point accumulation, and that drift
    scales with the size of the actions.
    """
    worst = 0.0
    golden_observations = bundle.golden["observations"]
    reference_policy = runner.get_inference_policy(device="cpu").as_onnx(verbose=False)
    with bundle.unpacked() as directory:
        # Load the policy
        onnx_path = directory / "policy" / bundle.policy_files[0]
        session = onnxruntime.InferenceSession(
            str(onnx_path), providers=["CPUExecutionProvider"]
        )
        input_name = session.get_inputs()[0].name

        for golden_item in golden_observations:
            golden_obs = golden_item[None, :].astype("float32")

            # Policy vs golden actions
            policy_actions = np.asarray(
                session.run(None, {input_name: golden_obs})[0]
            ).ravel()
            with torch.no_grad():
                golden_reference = (
                    reference_policy(torch.from_numpy(golden_obs)).cpu().numpy().ravel()
                )

            difference = np.abs(policy_actions - golden_reference)
            worst = max(worst, float(difference.max()))
            if np.any(difference > atol + rtol * np.abs(golden_reference)):
                raise SystemExit(
                    f"The exported ONNX graph disagrees with the trained policy "
                    f"(largest difference {worst:.3e}). The usual causes are a stale "
                    f"file from an earlier run, or an observation normalizer that did "
                    f"not make it into the graph."
                )

    print(f"  onnx graph matches the trained policy (within {worst:.2e})")


def main():
    # Initialize Genesis
    torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=gs.cpu)

    # Setup environment
    env = WheeledRobotCommandDirectionEnv(num_envs=1, headless=True)
    env = RslRlWrapper(env)
    env.build()

    # Export the trained policy to ONNX
    log_path = Path("./logs") / args.exp_name
    policy_files, policy_runner, checkpoint = create_onnx_policy(env, log_path)

    # Create deployable bundle
    bundle = export(
        env.unwrapped,
        args.output,
        policy_path=policy_files,
        additional_provenance={
            "checkpoint": checkpoint,
            "framework": "rsl_rl",
            "framework_version": version("rsl-rl-lib"),
        },
    )

    # Verify the bundle's golden values against the trained policy
    verify_onnx_policy(bundle, policy_runner)

    print()
    print(bundle.describe())
    print()
    print(
        f"Copy the on_robot directory to the robot, then "
        f"follow the instructions in examples/wheeled_robot/on_robot/README.md."
    )


if __name__ == "__main__":
    main()
