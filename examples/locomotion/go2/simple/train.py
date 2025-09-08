import os
import argparse
import torch
from skrl.utils.runner.torch import Runner

import genesis as gs

from genesis_forge import (
    create_skrl_env,
    VideoWrapper,
)
from genesis_forge.rl.skrl.utils import (
    load_training_config,
    save_env_snapshots,
    get_latest_checkpoint,
)
from environment import Go2Env

SKRL_CONFIG = "./ppo.yaml"

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-n", "--num_envs", type=int, default=600)
parser.add_argument("--max_iterations", type=int, default=500)
parser.add_argument("-d", "--device", type=str, default="gpu")
args = parser.parse_args()


def train(
    cfg: dict,
    num_envs: int,
    video_path: str,
):
    """
    Train the agent.
    """

    #  Create environment
    env = Go2Env(num_envs=num_envs, headless=True)

    # Record videos in regular intervals
    env = VideoWrapper(
        env,
        video_length_sec=12,
        out_dir=video_path,
        episode_trigger=lambda episode_id: episode_id % 5 == 0,
    )

    # Build the environment
    env.build()

    # Setup training runner and train
    env = create_skrl_env(env)
    runner = Runner(env, cfg)
    print("💪 Training model...")
    runner.run("train")
    env.close()


def record_video(cfg: dict, log_path: str, video_path: str):
    """Record a video of the best performing episode."""
    # Recording environment
    env = Go2Env(num_envs=1)
    env = VideoWrapper(
        env,
        out_dir=video_path,
        filename="best.mp4",
        video_length_sec=15,
    )
    env.build()

    # Update timesteps to only record the final video
    cfg["trainer"]["timesteps"] = env._video_length_steps
    cfg["agent"]["experiment"]["write_interval"] = 0  # don't log to TensorBoard
    cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    # Load best checkpoint
    checkpoint_path = get_latest_checkpoint(log_path)
    if checkpoint_path is None:
        print(f"ERROR: No checkpoint found in '{log_path}'.")
        return

    # Setup runner
    env = create_skrl_env(env)
    runner = Runner(env, cfg)
    runner.agent.load(checkpoint_path)

    # Eval
    print("🎬 Recording video of best model...")
    runner.run("eval")
    env.close()
    env = None


def main():
    # Processor backend (GPU or CPU)
    backend = gs.gpu
    if args.device == "cpu":
        backend = gs.cpu
        torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend)

    # Load training configuration
    cfg, log_path = load_training_config(SKRL_CONFIG, args.max_iterations)
    video_path = os.path.join(log_path, "videos")
    print(f"Logging to: {log_path}")

    # Save a snapshot of the training configuration and local git changes to the environment
    save_env_snapshots(log_path, cfg)

    # Train agent
    train(cfg, args.num_envs, video_path)

    # Record a video of the final episode
    record_video(cfg, log_path, video_path)


if __name__ == "__main__":
    main()
