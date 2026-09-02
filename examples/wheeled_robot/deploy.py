"""Export this trained policy for a real robot.

Produces a deployment bundle -- the observation and action pipelines captured from
the built environment, plus the policy as ONNX -- that the simulation-free
``genesis-forge-runtime`` runtime replays on the robot::

    uv run python deploy.py                  # after training

Before anything is written, the export runs the deployment classes against the live
training pipeline and refuses to write a bundle if they disagree.

Checking that the exported ONNX file matches the policy it came from is this
script's job rather than the library's -- how you export a policy is up to your
training framework, so verifying it belongs next to that code. See
``verify_onnx_policy`` below.
"""

import argparse
import glob
import os
import pickle
import sys
from importlib.metadata import version

import genesis as gs
import numpy as np
import onnxruntime
import torch
from environment import WheeledRobotCommandDirectionEnv
from rsl_rl.runners import OnPolicyRunner

from genesis_forge.deployment import export
from genesis_forge.wrappers import RslRlWrapper

EXPERIMENT_NAME = "wheeled-robot-command"

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-e", "--exp_name", type=str, default=EXPERIMENT_NAME)
parser.add_argument(
    "-o", "--output", type=str, default="./deploy_bundle", help="Where to write the bundle."
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

    Returns the policy's file(s), a callable for the parity check, and the
    checkpoint used.
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

    # Once the weights exceed onnx's inline threshold they are written to a
    # companion file instead, and the graph is useless without it. Both belong in
    # the bundle, so hand over both.
    policy_files = [onnx_path]
    weights = f"{onnx_path}.data"
    if os.path.isfile(weights):
        policy_files.append(weights)

    # Get the trained reference policy
    reference_policy = runner.get_inference_policy(device="cpu").as_onnx(verbose=False)

    return policy_files, reference_policy, checkpoint


def verify_onnx_policy(bundle, reference_policy, rtol=1e-4, atol=1e-5):
    """Check the bundle's ONNX graph against the policy it was exported from.

    This runs against the copy *inside the bundle*, not the file we handed to
    ``export``. That is the artifact going to the robot, and it is the only way to
    notice a companion file we forgot to list -- the source graph would load
    perfectly, with its weights still sitting next to it in the log directory.

    Genesis Forge packages the file and proves the observation and action
    pipelines match; confirming the *graph* matches is left to us, because it
    depends on how the framework exported it. This is worth doing: an observation
    normalizer that silently failed to make it into the graph is the classic
    sim-to-real failure, and no other check would see it.

    The bundle's golden.npz carries the observation vectors the parity gate ran
    on, which are exactly the right inputs to compare against.

    The tolerance is relative because export reorders floating-point
    accumulation, and that drift grows with how large the actions are -- this
    robot's wheel velocities run to ~50, so an absolute bound would have to be
    loose enough to be useless on a policy emitting joint angles around 1.
    """
    observations = bundle.golden["observations"]
    # An archive has to be on disk before onnxruntime can resolve a graph's
    # external weights. This unpacks to a temp directory and clears it after.
    with bundle.unpacked() as directory:
        worst = _compare_policies(
            directory / bundle.policy_file,
            reference_policy,
            observations,
            rtol=rtol,
            atol=atol,
        )
    print(f"  onnx graph matches the trained policy (within {worst:.2e})")


def _compare_policies(onnx_path, reference_policy, observations, *, rtol, atol):
    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    input_name = session.get_inputs()[0].name

    worst = 0.0
    for observation in observations:
        batched = observation[None, :].astype("float32")
        exported = np.asarray(session.run(None, {input_name: batched})[0]).ravel()
        with torch.no_grad():
            reference = (
                reference_policy(torch.from_numpy(batched)).cpu().numpy().ravel()
            )
        difference = np.abs(exported - reference)
        worst = max(worst, float(difference.max()))
        if np.any(difference > atol + rtol * np.abs(reference)):
            raise SystemExit(
                f"The exported ONNX graph disagrees with the trained policy "
                f"(largest difference {worst:.3e}). The usual causes are a stale "
                f"file from an earlier run, or an observation normalizer that did "
                f"not make it into the graph."
            )
    return worst


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
    log_path = f"./logs/{args.exp_name}"
    policy_files, reference_policy, checkpoint = get_onnx_policy(env, log_path)

    # `export` wants the environment itself, not the training-framework wrapper.
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

    verify_onnx_policy(bundle, reference_policy)

    print()
    print(bundle.describe())
    print()
    print(f"Copy {bundle.path.name} to the robot, then: "
          f"pip install genesis-forge-runtime[onnx]")


if __name__ == "__main__":
    main()
