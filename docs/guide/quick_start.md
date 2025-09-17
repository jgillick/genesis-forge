# 👋 Quick Start

This guide will help you create your first robotic RL environment with Genesis Forge and train an agent in just a few minutes.

## Installation

First, install Genesis Forge and its dependencies:

```bash
pip install genesis-forge
```

For training, you'll also need an RL library. We recommend RSL-RL:

```bash
pip install tensorboard rsl-rl-lib>=2.2.4
```

## Genesis Simulator

This framework is built around the [Genesis Simulator](https://genesis-world.readthedocs.io/en/latest/user_guide/overview/what_is_genesis.html). If you're new to Genesis, we recommend starting with the [Hello, Genesis](https://genesis-world.readthedocs.io/en/latest/user_guide/getting_started/hello_genesis.html) guide to get familiar with the fundamentals.

## Your First Environment

Let's create a simple locomotion environment where a quadruped robot learns to walk forward. Here's a complete example:

```{code-block} python
:caption: environment.py

""" A simple Go2 robot locomotion environment"""

import torch
import genesis as gs
from genesis_forge import ManagedEnvironment
from genesis_forge.managers import (
    RewardManager,
    TerminationManager,
    EntityManager,
    ObservationManager,
    PositionActionManager,
)
from genesis_forge.managers.entity import reset
from genesis_forge.mdp import rewards, terminations


class MyFirstEnv(ManagedEnvironment):
    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1/50,  # 50Hz control frequency
        max_episode_length_s: int = 20,
        headless: bool = True,
    ):
        super().__init__(
            num_envs=num_envs,
            dt=dt,
            max_episode_length_sec=max_episode_length_s,
        )

        # Build the simulation scene
        self.scene = gs.Scene(
            show_viewer=not headless,
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf", # this is loaded from the genesis lib
                pos=[0.0, 0.0, 0.4],
                quat=[1.0, 0.0, 0.0, 0.0],
            ),
        )

    def config(self):
        """Configure all the managers that handle different aspects of the environment."""

        # Set target velocity (forward at 0.5 m/s)
        self.target_vel = torch.zeros((self.num_envs, 3), device=gs.device)
        self.target_vel[:, 0] = 0.5  # X-axis velocity

        # Entity Manager - Handles robot resets
        self.robot_manager = EntityManager(
            self,
            entity_attr="robot",
            on_reset={
                "position": {
                    "fn": reset.position,
                    "params": {
                        "position": [0.0, 0.0, 0.4],
                        "quat": [1.0, 0.0, 0.0, 0.0],
                    },
                },
            },
        )

        # Action Manager - Maps RL actions to joint positions
        self.action_manager = PositionActionManager(
            self,
            joint_names=[".*"],  # Control all joints
            default_pos={
                ".*_hip_joint": 0.0,
                "FL_thigh_joint": 0.8,
                "FR_thigh_joint": 0.8,
                "RL_thigh_joint": 1.0,
                "RR_thigh_joint": 1.0,
                ".*_calf_joint": -1.5,
            },
            scale=0.25,  # Scale actions
            use_default_offset=True, # DOF actions are relative to the default_pos positions

            # PD controller gains
            pd_kp=20,
            pd_kv=0.5,
        )

        # Reward Manager - Defines what behaviors to encourage
        RewardManager(
            self,
            logging_enabled=True,
            cfg={
                # Maintain target height
                "base_height": {
                    "weight": -50.0,
                    "fn": rewards.base_height,
                    "params": {"target_height": 0.3},
                },
                # Track forward velocity
                "velocity_tracking": {
                    "weight": 1.0,
                    "fn": rewards.command_tracking_lin_vel,
                    "params": {"command": self.target_vel[:, :2]},
                },
                # Minimize vertical motion
                "vertical_velocity": {
                    "weight": -1.0,
                    "fn": rewards.lin_vel_z_l2,
                },
                # Encourage smooth actions
                "action_smoothness": {
                    "weight": -0.005,
                    "fn": rewards.action_rate_l2,
                },
            },
        )

        # Termination Manager - Defines when episodes end
        self.termination_manager = TerminationManager(
            self,
            logging_enabled=True,
            term_cfg={
                # Episode should end (timeout) when it hits the max episode length
                "timeout": {
                    "fn": terminations.timeout,
                    "time_out": True,
                },
                # Terminate if the robot is falling over
                "fall_over": {
                    "fn": terminations.bad_orientation,
                    "params": {"limit_angle": 0.174},  # ~10 degrees
                },
            },
        )

        # Observation Manager - Defines what the agent observes
        ObservationManager(
            self,
            cfg={
                "angular_velocity": {
                    "fn": lambda env: self.robot_manager.get_angular_velocity(),
                    "scale": 0.25,
                },
                "linear_velocity": {
                    "fn": lambda env: self.robot_manager.get_linear_velocity(),
                    "scale": 2.0,
                },
                "projected_gravity": {
                    "fn": lambda env: self.robot_manager.get_projected_gravity(),
                },
                "joint_positions": {
                    "fn": lambda env: self.action_manager.get_dofs_position(),
                },
                "joint_velocities": {
                    "fn": lambda env: self.action_manager.get_dofs_velocity(),
                    "scale": 0.05,
                },
                "last_actions": {
                    "fn": lambda env: self.action_manager.get_actions(),
                },
            },
        )
```

## Training Your Agent

Now let's train a policy using RSL-RL:

```{code-block} python
:caption: train.py

import genesis as gs
from genesis_forge.wrappers import RslRlWrapper, VideoWrapper
from rsl_rl.runners import OnPolicyRunner
from my_first_env import MyFirstEnv

# Initialize Genesis
gs.init(backend=gs.gpu)  # Use gs.cpu if no GPU available

# Create environment with 4096 parallel simulations
env = MyFirstEnv(num_envs=4096, headless=True)

# Add video recording during training (optional)
env = VideoWrapper(
    env,
    video_length_sec=10,
    out_dir="./videos",
    episode_trigger=lambda ep: ep % 10 == 0,  # Record every 10th episode
)

# Wrap for RSL-RL compatibility
env = RslRlWrapper(env)
env.build()
env.reset()

# PPO training configuration
train_cfg = {
    "algorithm": {
        "class_name": "PPO",
        "learning_rate": 0.001,
        "num_learning_epochs": 5,
        "num_mini_batches": 4,
        "gamma": 0.99,
        "clip_param": 0.2,
    },
    "policy": {
        "class_name": "ActorCritic",
        "activation": "elu",
          "actor_hidden_dims": [512, 256, 128],
          "critic_hidden_dims": [512, 256, 128],
    },
    "runner": {
        "max_iterations": 300,
        "save_interval": 100,
        "experiment_name": "my_first_robot",
    },
    "seed": 42,
    "num_steps_per_env": 24,
}

# Train the agent
runner = OnPolicyRunner(env, train_cfg, "./logs", device=gs.device)
runner.learn(num_learning_iterations=300)

print("Training complete! Check ./logs for results and ./videos for recordings.")
```

Run the training with:

```bash
python train.py
```

## Evaluating Your Trained Policy

After training, you can visualize your trained policy:

```python
import torch
import genesis as gs
from genesis_forge.wrappers import RslRlWrapper
from rsl_rl.runners import OnPolicyRunner
from my_first_env import MyFirstEnv

# Initialize Genesis with visualization
gs.init(backend=gs.gpu)

# Create a single environment with visualization
env = MyFirstEnv(num_envs=1, headless=False)
env = RslRlWrapper(env)
env.build()
env.reset()

# Load the trained policy
runner = OnPolicyRunner(env, train_cfg, "./logs", device=gs.device)
runner.load("./logs/my_first_robot/model_300.pt")
policy = runner.get_inference_policy(device=gs.device)

# Run the policy
obs, _ = env.reset()
with torch.no_grad():
    while True:
        actions = policy(obs)
        obs, _, _, _ = env.step(actions)
```

## Other examples

Explore the complete examples in the repository:

- **[Simple Locomotion](https://github.com/jgillick/genesis-forge/tree/main/examples/simple)**: Basic forward walking
- **[Command Following](https://github.com/jgillick/genesis-forge/tree/main/examples/command_direction)**: Follow velocity commands
- **[Rough Terrain](https://github.com/jgillick/genesis-forge/tree/main/examples/rough_terrain)**: Navigate challenging terrain

## Troubleshooting

### GPU Memory Issues

If you run out of GPU memory, reduce the number of parallel environments:

```python
env = MyFirstEnv(num_envs=1024)  # Instead of 4096
```

### Poor Policy Performance

- Increase training iterations
- Tune reward weights
- Add curriculum learning for complex tasks
- Check that observations are properly scaled

## Next Steps

- Browse the [API Documentation](../api/index.md) for detailed reference
- Join our [Discord community](https://discord.gg/genesis-forge) for help and discussions

Happy training! 🚀
