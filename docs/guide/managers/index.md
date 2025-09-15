# Managers

This guide provides practical examples and best practices for using Genesis Forge's manager-based architecture. Each manager handles a specific aspect of your RL environment, making your code more modular and maintainable.

## Why Use Managers?

Traditional RL environment code often becomes a monolithic mess as features are added. Genesis Forge's manager architecture provides:

1. **Separation of Concerns**: Each manager handles one specific aspect
2. **Reusability**: Managers can be shared across environments
3. **Configurability**: Easy to modify behavior through configuration
4. **Logging**: Automatic tensorboard logging support
5. **Less Boilerplate**: Common patterns are handled for you

## Basic Example

When your environment class inherits from `ManagedEnvironment`, managers are automatically coordinated throughout your environment's lifecycles.

```python
from genesis_forge import ManagedEnvironment

class MyEnv(ManagedEnvironment):
    def __init__(self, ...):
        super().__init__(...)
        # Build your scene here

    def config(self):
        """Configure all managers here"""
        self.action_manager = PositionActionManager(self, ...)
        RewardManager(self, ...)
        TerminationManager(self, ...)
        ObservationManager(self, ...)
```

## Manager Lifecycle

All managers follow a consistent lifecycle:

2. **Build** (`build()`): Called before the very first step to initialize manager and its buffers.
3. **Step** (`step()`): Called at each environment step
4. **Reset** (`reset()`): Called when environments reset

Explore the managers to learn more.

```{toctree}
:maxdepth: 1

action
command
contact
entity
observation
reward
termination
terrain
```
