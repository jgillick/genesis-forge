import argparse
import pickle
import sys
from pathlib import Path

import genesis as gs
import torch
from environment import Go2CommandDirectionEnv
from rsl_rl.runners import OnPolicyRunner

from genesis_forge.wrappers import RslRlWrapper

EXPERIMENT_NAME = "go2-command"

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-d", "--device", type=str, default="gpu")
parser.add_argument("-e", "--exp_name", type=str, default=EXPERIMENT_NAME)
args = parser.parse_args()


def get_latest_model(log_dir: Path) -> str:
    """
    Get the last model from the log directory
    """
    checkpoints = list(log_dir.glob("model_*.pt"))
    if not checkpoints:
        print(f"Error: No model files found at '{log_dir}'.")
        sys.exit(1)
    # Sort by the file with the highest number
    latest = max(checkpoints, key=lambda p: int(p.stem.split("_")[1]))
    return str(latest)


def main():
    # Processor backend (GPU or CPU)
    backend = gs.gpu
    if args.device == "cpu":
        backend = gs.cpu
        torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend)

    # Load training configuration
    log_path = Path("./logs") / args.exp_name
    with open(log_path / "cfgs.pkl", "rb") as f:
        [cfg] = pickle.load(f)
    model = get_latest_model(log_path)

    # Setup environment
    env = Go2CommandDirectionEnv(num_envs=1, headless=False)
    env = RslRlWrapper(env)
    env.build()

    # Eval
    print("🎬 Loading last model...")
    runner = OnPolicyRunner(env, cfg, log_path, device=gs.device)
    runner.load(model)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()
    try:
        with torch.no_grad():
            while True:
                actions = policy(obs)
                obs, _rews, _dones, _infos = env.step(actions)
    except KeyboardInterrupt:
        pass
    except gs.GenesisException as e:
        if str(e) != "Viewer closed.":
            raise
    except Exception:
        raise


if __name__ == "__main__":
    main()
