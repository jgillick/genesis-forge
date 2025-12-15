import os
import glob
import torch
import pickle
import argparse
from importlib import metadata

import rclpy
from ros2_interface import RosInterface

from environment import Go2SimpleEnv

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

EXPERIMENT_NAME = "go2-simple"

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
    # Sort by the file with the highest number
    sorted_models = sorted(
        model_checkpoints,
        key=lambda x: int(os.path.basename(x).split("_")[1].split(".")[0]),
    )
    return sorted_models[-1]


def setup_observations(env: Go2SimpleEnv, ros_interface: RosInterface):
    # Assign a function to each observation that will return real sensor data
    obs = env.observation_manager.cfg
    obs["angle_velocity"].fn = lambda env: ros_interface.get_angular_velocity()
    obs["linear_velocity"].fn = lambda env: ros_interface.get_linear_velocity()
    obs["projected_gravity"].fn = lambda env: torch.zeros(3)
    obs["dof_position"].fn = lambda env: ros_interface.get_dofs_position()
    obs["dof_velocity"].fn = lambda env: ros_interface.get_dofs_velocity()
    # No need to update the actions observation, as that will be handled by the environment automatically


def main():
    # Processor backend (GPU or CPU)
    rclpy.init()
    if args.device == "cpu":
        device = torch.device("cpu")
        torch.set_default_device("cpu")
    elif args.device == "gpu":
        device = torch.device("cpu")
        torch.set_default_device("cuda:0")

    # Load training configuration
    log_path = f"./logs/{args.exp_name}"
    [cfg] = pickle.load(open(f"{log_path}/cfgs.pkl", "rb"))
    model = get_latest_model(log_path)

    # Setup environment
    env = Go2SimpleEnv(num_envs=1, headless=False, mode="real")
    env.build()
    pos_joints = []
    vel_joints = []
    force_joints = []
    ros_interface = RosInterface(pos_joints, vel_joints, force_joints)
    if rclpy.ok():
        rclpy.spin_once(ros_interface, timeout_sec=0.1)

    # Update observations to use real sensors
    setup_observations(env, ros_interface=ros_interface)

    # Load the trained policy
    print("🎬 Loading last model...")
    runner = OnPolicyRunner(env, cfg, log_path, device=device)
    runner.load(model)
    policy = runner.get_inference_policy(device=device)

    try:
        obs, _ = env.reset()
        with torch.no_grad():
            while True and rclpy.ok():
                rclpy.spin_once(ros_interface, timeout_sec=0.1)
                actions = policy(obs)
                obs, _rews, _dones, _infos = env.step(actions)
                # Get actions to send to the ros_interface
                ros_interface._pos_actions = env.action_manager.get_actions()

    except KeyboardInterrupt:
        pass
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
