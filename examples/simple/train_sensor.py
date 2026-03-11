"""
Train script using VideoSensorWrapper (Approach A).

Frames are captured with ``sensor.read()`` and accumulated in memory during each
recording window.  When the window ends the full frame list is submitted to a
background ``ThreadPoolExecutor`` for encoding -- the training loop is never blocked
by video I/O.

Compare with train_recorder.py (Approach C) which streams frames to a background
thread immediately as they arrive, keeping peak memory constant regardless of clip length.

Usage::

    python train_sensor.py
    python train_sensor.py --num_envs 1 --max_iterations 10 --device cpu
"""

import os
import copy
import torch
import shutil
import pickle
import argparse
from importlib import metadata
import genesis as gs

from genesis_forge.wrappers import (
    VideoSensorWrapper,
    RslRlWrapper,
)
from environment import Go2SimpleSensorEnv

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

EXPERIMENT_NAME = "go2-simple-sensor"

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-n", "--num_envs", type=int, default=4096)
parser.add_argument("--max_iterations", type=int, default=101)
parser.add_argument("-d", "--device", type=str, default="gpu")
parser.add_argument("-e", "--exp_name", type=str, default=EXPERIMENT_NAME)
args = parser.parse_args()


def training_cfg(exp_name: str, max_iterations: int):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
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
        "runner_class_name": "OnPolicyRunner",
        "seed": 1,
        "num_steps_per_env": 24,
        "save_interval": 100,
        "empirical_normalization": None,
        "obs_groups": {"policy": ["policy"], "critic": ["policy"]},
    }


def main():
    # Initialize Genesis
    # macOS has two limitations with the sensor camera:
    #   1. Genesis dlpack does not support Apple MPS (device_type 8) -- force CPU backend.
    #   2. RasterizerCameraSensor requires OpenGL 4.2 for n_envs > 1, which Metal lacks --
    #      cap num_envs to 1.
    import platform
    if platform.system() == "Darwin":
        if args.device == "gpu":
            print("macOS detected: switching to CPU backend (MPS is not supported by Genesis).")
            args.device = "cpu"
        if args.num_envs > 1:
            print(
                "macOS detected: capping num_envs to 1 "
                "(RasterizerCameraSensor requires OpenGL 4.2 for n_envs > 1, "
                "which Metal does not support)."
            )
            args.num_envs = 1

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

    # Load training configuration and save snapshot
    cfg = training_cfg(experiment_name, args.max_iterations)
    pickle.dump([cfg], open(os.path.join(log_path, "cfgs.pkl"), "wb"))

    # Create environment (includes a RasterizerCameraSensor at self.sensor_camera)
    env = Go2SimpleSensorEnv(num_envs=args.num_envs, headless=True)

    # Approach A: VideoSensorWrapper
    #
    # Frames captured by sensor.read() are accumulated in a list during the clip.
    # When the clip ends, the entire list is submitted to a ThreadPoolExecutor for
    # encoding -- training continues immediately without waiting for the file write.
    #
    # Peak memory ~ (video_length_sec * fps * H * W * 3) bytes per active clip.
    # For 12s @ 60fps at 1280x720: ~1.9 GB (consider a lower fps for long clips).
    env = VideoSensorWrapper(
        env,
        sensor_attr="sensor_camera",
        video_length_sec=12,
        out_dir=os.path.join(log_path, "videos"),
        episode_trigger=lambda episode_id: episode_id % 5 == 0,
    )

    env = RslRlWrapper(env)
    env.build()
    env.reset()

    print("Training with VideoSensorWrapper (Approach A: ThreadPoolExecutor)...")
    runner = OnPolicyRunner(env, copy.deepcopy(cfg), log_path, device=gs.device)
    runner.git_status_repos = ["."]
    runner.learn(
        num_learning_iterations=args.max_iterations, init_at_random_ep_len=False
    )
    # executor.shutdown(wait=True) is called inside VideoSensorWrapper.close(),
    # so all pending encodes finish before the process exits.
    env.close()


if __name__ == "__main__":
    main()
