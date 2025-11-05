# Action Manager

The Action Manager is responsible for mapping actions from your RL policy to robot actuator inputs, or other behaviors. The action manager uses the actuator
manager to send the action positions to the actuators.

You can see a full example using the action manager in [examples/simple](https://github.com/jgillick/genesis-forge/tree/main/examples/simple).

## PositionActionManager

The most common action manager is `PositionActionManager`. This sets an unbounded action space (`+/-inf`) and the received actions are scaled to values relative to the defined offset.

```{math}
position = offset + scaling * action
```

By setting the offset to the default stable position (via `default_pos` and `use_default_offset` params), the policy will learn what is stable early, which can lead to faster convergence.

```python
from genesis_forge.managers import PositionActionManager, ActuatorManager

class MyEnv(ManagedEnvironment):
    def config(self):
        self.actuator_manager = ActuatorManager(
            self,
            joint_names=[".*"],      # Control all joints
            default_pos={            # Set a default stable position
                ".*_hip_joint": 0.0,
                ".*_thigh_joint": 0.8,
                ".*_calf_joint": -1.5,
            },
            kp=20, # PD controller positional gains
            kv=0.5, # PD controller velocity gains
        )
        self.action_manager = PositionActionManager(
            self,
            scale=0.25,              # Scale actions by 0.25
            use_default_offset=True, # Add default positions as offset
            actuator_manager=self.actuator_manager,
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
self.actuator_manager = ActuatorManager(
    self,
    default_pos={ # Set a default stable position
        ".*_hip_joint": 0.0,
        ".*_thigh_joint": 0.8,
        ".*_calf_joint": -1.5,
    },
    # ...
)
self.action_manager = PositionActionManager(
    self,
    use_default_offset=True,  # action = actuator_manager.default_pos + scale * raw_action
    actuator_manager=self.actuator_manager,
)

# Option 2: Use custom offset
self.action_manager = PositionActionManager(
    self,
    offset=0.2, # Apply this offset to all joints
    use_default_offset=False,
    # ...
)
```

## PositionWithinLimitsActionManager

This action manager is similar to PositionActionManager, but sets the action space range to `-1.0 - 1.0` and converts the received action from that range to an absolute position within the limits of your actuator. This is useful if you need your policy to use the full range of your actuators, however, it might take longer to learn the default stable position.

```python
from genesis_forge.managers import PositionWithinLimitsActionManager, ActuatorManager

self.actuator_manager = ActuatorManager(
    # ...
)
self.action_manager = PositionWithinLimitsActionManager(
    self,
    actuator_manager=self.actuator_manager,
)
```

## Sim2Real - Action latency

In most robots, there is some latency between when the action is received, and when it is acted upon by the actuator. To roughly emulate this, you can set the `delay_step` parameter.

```python
self.action_manager = PositionActionManager(
    self,
    delay_step=1 # Delay sending actions to actuators by one step (dt)
    scale=0.25,
    use_default_offset=True,
    actuator_manager=self.actuator_manager,
)
```

This parameter lets use delay sending the actions to the actuators by a specific number of training steps.

## Get Action Information

```python
# Get the last actions sent to the robot
actions = self.action_manager.get_actions()

# Get the last raw action values, before they were converted to joint positions
actions = self.action_manager.raw_actions

# Get the number of controlled DOFs
num_actions = self.action_manager.num_actions

# Get the action space
action_space = self.action_manager.action_space
```
