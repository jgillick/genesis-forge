import argparse

import genesis as gs
import torch
from pathlib import Path
from environment import WheeledRobotCommandDirectionEnv
from rsl_rl.runners import OnPolicyRunner

from genesis_forge.gamepads import Gamepad
from genesis_forge.wrappers import RslRlWrapper

from utils import get_latest_model, load_config_pickle

parser = argparse.ArgumentParser(add_help=True)
parser.add_argument("-d", "--device", type=str, default="gpu")
parser.add_argument("-e", "--exp_name", type=str, default="wheeled-robot-command")
args = parser.parse_args()


def main():
    # Processor backend (GPU or CPU)
    backend = gs.gpu
    if args.device == "cpu":
        backend = gs.cpu
        torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend)

    # Load training configuration
    log_path = Path("./logs") / args.exp_name
    cfg = load_config_pickle(log_path)
    model = get_latest_model(log_path)

    # Setup environment
    env = WheeledRobotCommandDirectionEnv(
        num_envs=1, headless=False, max_episode_length_s=None
    )
    env.build()

    # Connect to gamepad
    print("🎮 Connecting to gamepad...")
    gamepad = Gamepad()
    env.velocity_command.use_gamepad(
        gamepad,
        lin_vel_y_axis=0,  # left stick forward/backward
        ang_vel_z_axis=0,  # left stick left/right
    )

    # Run program
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
        if str(e) != "Viewer closed.":
            raise
    except Exception:
        raise


if __name__ == "__main__":
    main()
