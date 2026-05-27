"""Train Go2 velocity tracking with SKRL MAPPO (MASQ-style per-leg agents)."""

from __future__ import annotations

import argparse
import os
import shutil

import genesis as gs
import torch

from genesis_forge.wrappers import VideoWrapper
from skrl.memories.torch import RandomMemory
from skrl.multi_agents.torch import ExperimentCfg
from skrl.multi_agents.torch.mappo import MAPPO
from skrl.multi_agents.torch.mappo.mappo_cfg import MAPPO_CFG
from skrl.trainers.torch import SequentialTrainer

from env_wrapper import SkrlMasqWrapper
from environment import Go2MasqLocomotionEnv
from models import MasqGaussianPolicy, MasqValue

EXPERIMENT_NAME = "go2-multi-agent"

# Transitions per agent before each MAPPO update (buffer ≈ rollouts x num_envs)
ROLLOUTS = 16

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_envs", type=int, default=4096)
parser.add_argument(
    "-t",
    "--timesteps",
    type=int,
    default=20_000,
    help="Trainer timesteps",
)
parser.add_argument("-d", "--device", type=str, default="gpu", choices=("gpu", "cpu"))
parser.add_argument("-e", "--exp_name", type=str, default=EXPERIMENT_NAME)
args = parser.parse_args()

def main() -> None:
    # Initialize Genesis
    # Processor backend (GPU or CPU)
    backend = gs.gpu
    if args.device == "cpu":
        backend = gs.cpu
        torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend, performance_mode=True)

    # Logging directory
    log_base_dir = "./logs"
    experiment_name = args.exp_name
    log_path = os.path.join(log_base_dir, experiment_name)
    if os.path.exists(log_path):
        shutil.rmtree(log_path)
    os.makedirs(log_path, exist_ok=True)
    print(f"Logging to: {log_path}")

    # Create environment
    env = Go2MasqLocomotionEnv(num_envs=args.num_envs, headless=True)
    env = VideoWrapper(
        env,
        camera_attr="camera",
        video_length_sec=12,
        out_dir=os.path.join(log_path, "videos"),
        episode_trigger=lambda episode_id: episode_id % 2 == 0,
        logging=False
    )
    env.build()
    env.reset()

    # Add SKRL wrapper
    wrapped = SkrlMasqWrapper(env)
    agents = list(wrapped.possible_agents)


    # MAPPO learning configuration
    def for_each_agent(v):
        return {uid: v for uid in agents}
    cfg = MAPPO_CFG(
        rollouts=ROLLOUTS,
        learning_epochs=5,
        mini_batches=4,
        entropy_loss_scale=0.01,
        discount_factor=0.99,
        gae_lambda=0.95,
        learning_rate=3e-4,
        random_timesteps=0,
        learning_starts=0,
        value_loss_scale=1.0,
        grad_norm_clip=0.5,
        learning_rate_scheduler_kwargs=for_each_agent({}),
        observation_preprocessor_kwargs=for_each_agent({}),
        state_preprocessor_kwargs=for_each_agent({}),
        value_preprocessor_kwargs=for_each_agent({}),
        experiment=ExperimentCfg(
            directory=log_base_dir,
            experiment_name=args.exp_name,
            write_interval=max(1, args.timesteps // 100),
            checkpoint_interval=max(1000, args.timesteps // 10),
        ),
    )

    # SKRL memory is (rollout_steps, num_envs, …); use rollouts as the time axis (see skrl Runner).
    memories = {
        uid: RandomMemory(
            memory_size=ROLLOUTS,
            num_envs=args.num_envs,
        )
        for uid in agents
    }

    # Create agent models
    models: dict[str, dict] = {}
    obs_space = wrapped.observation_spaces[agents[0]]
    action_space = wrapped.action_spaces[agents[0]]
    state_space = wrapped.state_spaces[agents[0]]
    for uid in agents:
        models[uid] = {
            "policy": MasqGaussianPolicy(obs_space, action_space, clip_actions=False),
            "value": MasqValue(state_space),
        }
    agent = MAPPO(
        possible_agents=agents,
        models=models,
        memories=memories,
        observation_spaces=wrapped.observation_spaces,
        state_spaces=wrapped.state_spaces,
        action_spaces=wrapped.action_spaces,
        cfg=cfg,
    )
    print(agent)

    # Train
    trainer_cfg = {
        "timesteps": args.timesteps,
        "headless": True,
        "close_environment_at_exit": True,
        "environment_info": "episode",
    }
    SequentialTrainer(env=wrapped, agents=agent, cfg=trainer_cfg).train()
    wrapped.close()


if __name__ == "__main__":
    main()
