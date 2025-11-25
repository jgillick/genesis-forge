import os
import torch
import shutil
import pickle
import argparse
import genesis as gs
from skrl.utils.runner.torch import Runner

from genesis_forge.wrappers import (
    VideoWrapper,
    SkrlEnvWapper,
)
from environment import Go2SimpleEnv

EXPERIMENT_NAME = "go2-simple-skrl"
SKRL_CONFIG_PATH = "./skrl_ppo.yaml"

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-n", "--num_envs", type=int, default=4096)
parser.add_argument("-s", "--steps", type=int, default=6000)
parser.add_argument("-d", "--device", type=str, default="gpu")
parser.add_argument("-e", "--exp_name", type=str, default=EXPERIMENT_NAME)
args = parser.parse_args()


def training_cfg(log_base_dir: str, experiment_name: str, steps: int) -> dict:
    """
    Load the SKRL training configuration from the yaml file.
    """
    cfg = Runner.load_cfg_from_yaml(SKRL_CONFIG_PATH)
    cfg["agent"]["experiment"]["directory"] = log_base_dir
    cfg["agent"]["experiment"]["experiment_name"] = experiment_name
    cfg["trainer"]["timesteps"] = steps
    return cfg


def main():
    # Initialize Genesis
    # Processor backend (GPU or CPU)
    backend = gs.gpu
    if args.device == "cpu":
        backend = gs.cpu
        torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend)

    # Logging directory
    log_base_dir = "./logs"
    experiment_name = args.exp_name
    log_path = os.path.join(log_base_dir, experiment_name)
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path, exist_ok=True)
    print(f"Logging to: {log_path}")

    # Load training configuration and save snapshot of training configs
    cfg = training_cfg(log_base_dir, experiment_name, args.steps)
    pickle.dump(
        [cfg],
        open(os.path.join(log_path, "cfgs.pkl"), "wb"),
    )

    # Create environment
    env = Go2SimpleEnv(num_envs=args.num_envs, headless=True)

    # Record videos in regular intervals
    env = VideoWrapper(
        env,
        logging=False,
        video_length_sec=6,
        out_dir=os.path.join(log_path, "videos"),
        episode_trigger=lambda episode_id: episode_id % 5 == 0,
    )

    # Build the environment
    env = SkrlEnvWapper(env)
    env.build()
    env.reset()

    # Train
    print("💪 Training model...")
    runner = Runner(env, cfg)
    runner.run("train")
    env.close()


if __name__ == "__main__":
    main()
