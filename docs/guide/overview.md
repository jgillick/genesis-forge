# 💡 Overview

Genesis Forge is a modular framework for RL robot environments, inspired by [Isaac Lab](https://github.com/isaac-sim/IsaacLab/tree/main),
and built on top of the awesome [Genesis Simulator](https://github.com/Genesis-Embodied-AI/Genesis/tree/main). Our goal
is to provide the tools to get your training library up and running quickly, with less of the upfront boilerplate work, and a structure
that makes growing projects feel more manageable.

The preferred way to use Genesis Forge is through its management architecture. A manager is a Python class dedicated to a very specific part
of your environment system. There are managers for the reward system, processing actions, defining observations and the observation space, etc.
When you use managers, your main environment script becomes more of a flexible configuration center and should contain less logic.

Here's an example of defining the RewardManager:

```python
self.reward_manager = RewardManager(
    self,
    cfg={
        "base_height_target": {
            "weight": -50.0,
            "fn": rewards.base_height,
            "params": {
                "target_height": 0.3,
            },
        },
        "lin_vel_z": {
            "weight": -1.0,
            "fn": rewards.lin_vel_z,
        },
        "action_rate": {
            "weight": -0.005,
            "fn": rewards.action_rate,
        },
        "similar_to_default": {
            "weight": -0.1,
            "fn": rewards.dof_similar_to_default,
        },
    },
)
```

As you can see, this defines 4 rewards, and references the functions used to generate those rewards. This manager will also log the individual rewards to the `extras["episode"]` dict, for logging through a framework like rsl_rl or skrl. If your environment uses curriculum learning, you can
dynamically change the reward configurations as the training program progresses.

For example, to adjust the weights:

```python
self.reward_manager.cfg["base_height_target"]["weight"] = -25.0
self.reward_manager.cfg["similar_to_default"]["weight"] = -0.05
```

## Available Managers

Genesis Forge provides several built-in managers to handle different aspects of your RL environment:

- **ActionManager**: Processes raw actions from your RL policy and send them to your actuators.
- **ObservationManager**: Defines observations and manages the observation space
- **RewardManager**: Computes and tracks individual reward components
- **TerminationManager**: Handles episode termination conditions
- **CommandManager**: Manages high-level commands for goal-conditioned tasks
- **EntityManager**: Handles robot spawning, resets, and state management
- **ContactManager**: Tracks contacts and collisions between robots and the environment
- **TerrainManager**: Provides useful functions for working with various terrains.

## Next Steps

- Check out the [Quick Start Guide](quick_start.md) to build your first environment
- Browse the [Examples](https://github.com/jgillick/genesis-forge/tree/main/examples) to see complete implementations
- Review the [API Documentation](../api/index.md) for detailed manager configurations
