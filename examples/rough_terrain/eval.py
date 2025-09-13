import os
import glob
import torch
import pickle
import argparse
from importlib import metadata
import genesis as gs

from genesis_forge.wrappers import RslRlWrapper
from environment import Go2RoughTerrainEnv

try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib").startswith("1."):
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please install install 'rsl-rl-lib>=2.2.4'.") from e
from rsl_rl.runners import OnPolicyRunner

EXPERIMENT_NAME = "go2-terrain"

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-d", "--device", type=str, default="gpu")
parser.add_argument("-e", "--exp_name", type=str, default=EXPERIMENT_NAME)
args = parser.parse_args()


def get_latest_model(log_dir: str) -> str:
    """
    Get the last model from the log directory
    """
    model_checkpoints = glob.glob(os.path.join(log_dir, "model_*.pt"))
    if len(model_checkpoints) == 0:
        print(
            f"Warning: No model files found at '{log_dir}' (you might need to train more)."
        )
        exit(1)
    model_checkpoints.sort()
    return model_checkpoints[-1]


def main():
    # Processor backend (GPU or CPU)
    backend = gs.gpu
    if args.device == "cpu":
        backend = gs.cpu
        torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend)

    # Load training configuration
    log_path = f"./logs/{args.exp_name}"
    [cfg] = pickle.load(open(f"{log_path}/cfgs.pkl", "rb"))
    model = get_latest_model(log_path)

    # Setup environment
    env = Go2RoughTerrainEnv(num_envs=1, headless=False)
    env = RslRlWrapper(env)
    env.build()

    # Make changes for eval
    # Pick a new direction every 2 seconds
    env.unwrapped.velocity_command.resample_time_s = 2.0

    # Eval
    print("🎬 Loading last model...")
    runner = OnPolicyRunner(env, cfg, log_path, device=gs.device)
    runner.load(model)
    policy = runner.get_inference_policy(device=gs.device)
    try:
        obs, _ = env.reset()
        with torch.no_grad():
            while True:
                actions = policy(obs)
                obs, _rews, _dones, _infos = env.step(actions)
    except KeyboardInterrupt:
        pass
    except gs.GenesisException as e:
        if e.message != "Viewer closed.":
            raise e
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
