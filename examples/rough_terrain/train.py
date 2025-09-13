import os
import copy
import glob
import torch
import shutil
import pickle
import argparse
from importlib import metadata
import genesis as gs

from genesis_forge.wrappers import (
    VideoWrapper,
    RslRlWrapper,
)
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
parser.add_argument("-n", "--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=1200)
parser.add_argument("-d", "--device", type=str, default="gpu")
parser.add_argument("-e", "--exp_name", type=str, default=EXPERIMENT_NAME)
args = parser.parse_args()


def training_cfg(exp_name: str, max_iterations: int):
    return {
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1.0e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 1.0,
            "actor_obs_normalization": False,
            "critic_obs_normalization": False,
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 1,
            "max_iterations": max_iterations,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
    }


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


def train(cfg: dict, num_envs: int, log_dir: str, max_iterations: int):
    """
    Train the agent.
    """

    #  Create environment
    env = Go2RoughTerrainEnv(num_envs=num_envs, headless=True)

    # Record videos in regular intervals
    env = VideoWrapper(
        env,
        video_length_sec=12,
        out_dir=os.path.join(log_dir, "videos"),
        episode_trigger=lambda episode_id: episode_id % 5 == 0,
    )

    # Build the environment
    env = RslRlWrapper(env)
    env.build()
    env.reset()

    # Setup training runner and train
    print("💪 Training model...")
    runner = OnPolicyRunner(env, copy.deepcopy(cfg), log_dir, device=gs.device)
    runner.git_status_repos = ["."]
    runner.learn(num_learning_iterations=max_iterations, init_at_random_ep_len=False)
    env.close()


def record_video(cfg: dict, log_dir: str):
    """Record a video of the trained model."""
    # Recording environment
    env = Go2RoughTerrainEnv(num_envs=1, headless=True)
    env = VideoWrapper(
        env,
        out_dir=log_dir,
        filename="trained.mp4",
        video_length_sec=15,
    )
    video_length_steps = env.video_length_steps
    env = RslRlWrapper(env)
    env.build()

    # Eval
    print("🎬 Recording video of last model...")
    runner = OnPolicyRunner(env, copy.deepcopy(cfg), log_dir, device=gs.device)
    resume_path = get_latest_model(log_dir)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    obs, _ = env.reset()
    with torch.no_grad():
        i = 0
        while i < video_length_steps:
            i += 1
            actions = policy(obs)
            obs, _rews, _dones, _infos = env.step(actions)

    print(f"Saving video to {log_dir}/trained.mp4")
    env.close()


def main():
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

    # Load training configuration
    cfg = training_cfg(experiment_name, args.max_iterations)

    # Save config snapshot
    pickle.dump(
        [cfg],
        open(os.path.join(log_path, "cfgs.pkl"), "wb"),
    )

    # Train agent
    train(cfg, args.num_envs, log_path, args.max_iterations)

    # Record a video of best episode
    # record_video(cfg, log_path)


if __name__ == "__main__":
    main()
