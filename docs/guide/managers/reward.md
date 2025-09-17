# Reward Manager

The Reward Manager handles computing, combining, and logging reward components in your RL environment. It provides a clean way to define multi-objective rewards with automatic tracking and tensorboard logging.

You can see a full example using the reward manager in [examples/simple](https://github.com/jgillick/genesis-forge/tree/main/examples/simple).

## Overview

The Reward Manager allows you to:

- Define multiple reward components with individual weights
- Automatically sum rewards and track individual contributions
- Log rewards to tensorboard for analysis
- Dynamically adjust rewards during training (curriculum learning)
- Reuse common reward functions from the MDP library

## Basic Usage

```python
from genesis_forge.managers import RewardManager
from genesis_forge.mdp import rewards

class MyEnv(ManagedEnvironment):
    def config(self):
        RewardManager(
            self,
            cfg={
                "height": {
                    "weight": -1.0,            # Weight/scale
                    "fn": rewards.base_height, # Reward function
                    "params": {                # Params to the reward function
                        "target_height": 0.3
                    }
                },
                "flat_orientation": {
                    "fn": rewards.flat_orientation_l2,
                    "weight": -1.0,
                },
            },
        )
```

## Reward Configuration

Each reward config item requires:

- **fn**: A function that computes the reward
- **weight**: Multiplier for this component (can be negative for penalties)
- **params** (optional): Additional parameters to pass to the function

```python
RewardManager(
    self,
    cfg={
        "height_tracking": {
            "weight": -10.0,  # Strong penalty for wrong height
            "fn": rewards.base_height,
            "params": {
                "target_height": 0.35,  # Pass target to function
            },
        },
    },
)
```

## Built-in Reward Functions

Genesis Forge provides many common reward functions in [`genesis_forge.mdp.rewards`](../../api/mdp/rewards):

## Custom Reward Functions

A custom reward function takes in the environment as the first parameter, as well as any other parameter which will be defined in the `params` dict at the RewardManager. The returned value should be a tensor (shape: `(num_envs,)`) with a `float` value for each environment.

### Simple Custom Rewards

```python
def my_custom_reward(env):
    """Reward for staying near origin."""
    distance = torch.norm(env.robot.get_pos()[:, :2], dim=1)
    return torch.exp(-distance)

RewardManager(
    self,
    cfg={
        "stay_centered": {
            "fn": my_custom_reward,
            "weight": 0.5,
        },
    },
)
```

### Rewards with Parameters

```python
def target_height_reward(env, target_height: float):
    """Reward for reaching a target height."""
    base_pos = robot.get_pos()
    return torch.square(base_pos[:, 2] - target_height)

RewardManager(
    self,
    cfg={
        "height": {
            "weight": -5.0,
            "fn": target_height_reward,
            "params": {
                "target_height": 0.3
            },
        },
    },
)
```

### Lambda Functions

For simple one-liners, use lambda functions:

```python
RewardManager(
    self,
    cfg={
        # Penalize high angular velocity
        "spin_penalty": {
            "fn": lambda env: torch.abs(env.robot.get_ang_vel()[:, 2]),
            "weight": -0.2,
        },
    },
)
```

## Dynamic Reward Adjustment

### Curriculum Learning

Adjust rewards based on training progress:

```python
class MyEnv(ManagedEnvironment):
    def config(self):
        self.reward_manager = RewardManager(self, cfg={...})

    def step(self):
        self.update_curriculum()
        return super().step(actions)

    def update_curriculum(self):
        """Called periodically during training."""
        if self.step_count === 100:
            # Mid training: increase speed focus
            self.reward_manager.cfg["upright"]["weight"] = -2.0
            self.reward_manager.cfg["forward_vel"]["weight"] = 2.0
        elif self.step_count === 300:
            # Late training: add efficiency
            self.reward_manager.cfg["upright"]["weight"] = -1.0
            self.reward_manager.cfg["forward_vel"]["weight"] = 3.0
            self.reward_manager.cfg["energy"]["weight"] = -0.01
```

## Logging and Analysis

By default, individual reward components are logged to the `episode` item in the extras/infos dict. For many RL frameworks, like rsl_rl and skrl, items there will automatically be logged to tensorboard, or simular system. Rewards will be placed under the "Rewards" section.

```{figure} _images/reward_tensorboard.png
:alt: tensor board
Example tensorboard reward logging
```

To disable logging, set `logging_enabled` to `False`. To change the extras dict key that reward items are logged to, set the `extras_logging_key` param on the [environment](../../api/environments/genesis.md).
