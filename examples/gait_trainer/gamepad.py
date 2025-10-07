import os
import glob
import torch
import pickle
import argparse
import genesis as gs

from rsl_rl.runners import OnPolicyRunner
from genesis_forge.wrappers import RslRlWrapper
from genesis_forge.gamepads import Gamepad
from environment import Go2GaitTrainingEnv

EXPERIMENT_NAME = "go2-gait"

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
    env = Go2GaitTrainingEnv(num_envs=1, max_episode_length_s=None, gamepad_control=True)
    env.build()

    # Connect to gamepad
    print("🎮 Connecting to gamepad...")
    gamepad = Gamepad()
    env.velocity_command.use_gamepad(
        gamepad,
        ang_vel_z_axis=0,
        lin_vel_x_axis=1,
        lin_vel_y_axis=3,
    )
    env.gait_command_manager.use_gamepad(gamepad)

    # Eval
    print("Loading environment...")
    env = RslRlWrapper(env)
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
        if e.message != "Viewer closed.":
            raise e
    except Exception as e:
        raise e


if __name__ == "__main__":
    main()
