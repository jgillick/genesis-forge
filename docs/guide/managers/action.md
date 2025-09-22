# Action Manager

The Action Manager is responsible for mapping actions from your RL policy to robot actuator inputs, or other behaviors. It handles scaling, offsets, joint limits, and PD controller configuration.

You can see a full example using the action manager in [examples/simple](https://github.com/jgillick/genesis-forge/tree/main/examples/simple).

## PositionActionManager

The most common action manager is `PositionActionManager`. This sets an unbounded action space (`+/-inf`) and the received actions are scaled to values relative to the defined offset.

```{math}
position = offset + scaling * action
```

By setting the offset to the default stable position (via `default_pos` and `use_default_offset` params), the policy will learn what is stable early, which can lead to faster convergence.

```python
from genesis_forge.managers import PositionActionManager

class MyEnv(ManagedEnvironment):
    def config(self):
        self.action_manager = PositionActionManager(
            self,
            joint_names=[".*"],      # Control all joints
            default_pos={            # Set a default stable position
                ".*_hip_joint": 0.0,
                ".*_thigh_joint": 0.8,
                ".*_calf_joint": -1.5,
            },
            scale=0.25,              # Scale actions by 0.25
            use_default_offset=True, # Add default positions as offset
            pd_kp=20,                # Proportional gain
            pd_kv=0.5,               # Derivative gain
        )
```

### Scaling and Offsets

Scale actions to appropriate ranges:

```python
# Uniform scaling
self.action_manager = PositionActionManager(
    self,
    scale=0.5,  # All actions multiplied by 0.5
)

# Per-joint scaling
self.action_manager = PositionActionManager(
    self,
    scale={
        ".*_hip_joint": 0.3,    # Hip joints have smaller range
        ".*_thigh_joint": 0.5,  # Thigh joints have medium range
        ".*_calf_joint": 0.7,   # Calf joints have larger range
    },
)
```

Control how actions are offset:

```python
# Option 1: Use default positions as offset
self.action_manager = PositionActionManager(
    self,
    default_pos={            # Neutral stable position for all joints
        ".*_hip_joint": 0.0,
        ".*_thigh_joint": 0.8,
        ".*_calf_joint": -1.5,
    }
    use_default_offset=True,  # action = default_pos + scale * raw_action
)

# Option 2: Use custom offset
self.action_manager = PositionActionManager(
    self,
    offset=0.2, # Apply this offset to all joints
    use_default_offset=False,
)
```

## Custom values per joint

Any parameter typed with `DofValue` can either be a value for all joints, or be a dict of joint name/patterns to values.

For example, if all your actuators are the same, you can set the PD controller values like this:

```python
self.action_manager = PositionActionManager(
    self,
    pd_kp=50,  # Proportional gain (stiffness)
    pd_kv=1.0, # Derivative gain (damping)
)
```

However, if different joints have different gains:

```python
self.action_manager = PositionActionManager(
    self,
    pd_kp={
      ".*_hip_joint": 20,
      ".*_thigh_joint": 50,
      ".*_calf_joint": 42,
    }
    pd_kv={
      ".*_hip_joint": 1.0,
      ".*_thigh_joint": 0.5,
      ".*_calf_joint": 0.5,
    }
)
```

## Sim2Real

Add attributes like damping, stiffness and friction to the joints.

```python
self.action_manager = PositionActionManager(
    self,
    # ... other params...
    damping={".*": 0.1},
    stiffness={".*": 100},
    frictionloss={".*": 0.01},
)
```

Setting the noise value will scale all values with a bit of random noise:

```python
self.action_manager = PositionActionManager(
    self,
    # ... other params...
    noise=0.02
)
```

## Accessing Joint Information

### Get Current Joint States

```python
# In your reward or observation functions
positions = self.action_manager.get_dofs_position()
velocities = self.action_manager.get_dofs_velocity()
forces = self.action_manager.get_dofs_force()

# Add noise for training robustness
noisy_pos = self.action_manager.get_dofs_position(noise=0.01)
```

### Get Action Information

```python
# Get the last actions sent to the robot
last_actions = self.action_manager.get_actions()

# Get the default positions
default_pos = self.action_manager.default_dofs_pos

# Get the number of controlled DOFs
num_actions = self.action_manager.num_actions

# Get the action space
action_space = self.action_manager.action_space
```

## PositionWithinLimitsActionManager

This action manager is similar to PositionActionManager, but sets the action space range to `-1.0 - 1.0` and converts the received action from that range to an absolute position within the limits of your actuator. This is useful if you need your policy to use the full range of your actuators, however, it might take longer to learn the default stable position.

```python
from genesis_forge.managers import PositionWithinLimitsActionManager

self.action_manager = PositionWithinLimitsActionManager(
    self,
    joint_names=[".*"],
    position_limits={
        ".*_hip_joint": (-0.3, 0.3),
        ".*_thigh_joint": (0.5, 1.5),
        ".*_calf_joint": (-2.0, -1.0),
    },
    pd_kp=30,
    pd_kv=1.0,
)
```
