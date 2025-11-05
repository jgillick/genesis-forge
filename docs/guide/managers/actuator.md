# Actuator Manager

The Actuator Manager configures and manages the DOF actuators (joints) of your robot. It handles setting PD controller gains, default positions, force limits, and physical properties like damping and stiffness. The actuator manager is generally used in concert with the [Action Manager](action.md) to control the robot.

You can see a full example using the actuator manager in [examples/simple](https://github.com/jgillick/genesis-forge/tree/main/examples/simple).

## Basic Setup

The simplest way to use the ActuatorManager is to configure it with basic PD controller gains:

```python
from genesis_forge import ManagedEnvironment
from genesis_forge.managers import ActuatorManager, PositionActionManager

class MyEnv(ManagedEnvironment):
    def config(self):
        # Configure actuators
        self.actuator_manager = ActuatorManager(
            self,
            joint_names=[".*"],      # Control all joints
            default_pos={            # Default position for all joints.
                ".*_hip_joint": 0.0,
                ".*_thigh_joint": 0.8,
                ".*_calf_joint": -1.5,
            },
            kp=20,                   # Proportional gain (stiffness)
            kv=0.5,                  # Derivative gain (damping)
        )

        # Use with action manager
        self.action_manager = PositionActionManager(
            self,
            actuator_manager=self.actuator_manager,
            # ... other settings ...
        )
```

## Joint Selection

You first need to tell the actuator manager which joints will be controlled, as actuators, by it.
Set `joint_names` to either a single regex pattern, a list of joint names, or a list of regex patterns.

```python
# Control all joints
self.actuator_manager = ActuatorManager(
    self,
    joint_names=".*",
)

# Control specific joints by name pattern
self.actuator_manager = ActuatorManager(
    self,
    joint_names=[
        "FL_.*_joint",  # All front-left joints
        "FR_.*_joint",  # All front-right joints
        "RL_.*_joint",  # All rear-left joints
        "RR_.*_joint",  # All rear-right joints
    ],
)

# Control specific joints by exact name
self.actuator_manager = ActuatorManager(
    self,
    joint_names=[
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
    ],
)
```

## Default Positions

Set default positions for joints that will be applied during environment reset.
These generally create a stable initial position for the robot. Similar to the `joint_names`
parameter, these can be regular expressions or exact names.

```python
self.actuator_manager = ActuatorManager(
    self,
    joint_names=[".*"],
    default_pos={
        ".*_hip_joint": 0.0,
        ".*_thigh_joint": 0.8,
        ".*_calf_joint": -1.5,
    },
    kp=20,
    kv=0.5,
)
```

You can also set different default positions per leg:

```python
self.actuator_manager = ActuatorManager(
    self,
    joint_names=[".*"],
    default_pos={
        ".*_hip_joint": 0.0,
        "FL_thigh_joint": 0.8,
        "FR_thigh_joint": 0.8,
        "RL_thigh_joint": 1.0,  # Rear legs slightly different
        "RR_thigh_joint": 1.0,
        ".*_calf_joint": -1.5,
    },
    kp=20,
    kv=0.5,
)
```

## PD Controller Gains

Configure proportional (kp) and derivative (kv) gains for the PD controller:

```python
# Uniform gains for all joints
self.actuator_manager = ActuatorManager(
    self,
    joint_names=[".*"],
    kp=20, # Positional gains
    kv=0.5, # Velocity gains
)

# Per-joint gains
self.actuator_manager = ActuatorManager(
    self,
    joint_names=[".*"],
    kp={
        ".*_hip_joint": 30,
        ".*_thigh_joint": 20,
        ".*_calf_joint": 20,
    },
    kv={
        ".*_hip_joint": 1.0,
        ".*_thigh_joint": 0.5,
        ".*_calf_joint": 0.5,
    },
)
```

## Force Limits

Constrain the maximum force that actuators can apply:

```python
# Uniform force limit
self.actuator_manager = ActuatorManager(
    self,
    joint_names=[".*"],
    max_force=8.0,  # +/- 8.0 N
    kp=20,
    kv=0.5,
)

# Per-joint force limits
self.actuator_manager = ActuatorManager(
    self,
    joint_names=[".*"],
    max_force={
        ".*_hip_joint": 10.0,
        ".*_thigh_joint": 8.0,
        ".*_calf_joint": 6.0,
    },
    kp=20,
    kv=0.5,
)

# Asymmetric force limits (min, max)
self.actuator_manager = ActuatorManager(
    self,
    joint_names=[".*"],
    max_force={
        ".*_hip_joint": (-12.0, 10.0),   # Different limits for each direction
        ".*_thigh_joint": (-8.0, 8.0),
        ".*_calf_joint": (-6.0, 6.0),
    },
    kp=20,
    kv=0.5,
)
```

## Sim2Real - Physical Properties

Set physical properties like damping, stiffness, friction loss, and armature, to match your real actuators.

```python
self.actuator_manager = ActuatorManager(
    self,
    joint_names=[".*"],
    kp=20,
    kv=0.5,
    damping=0.1,        # Joint damping
    stiffness=100,      # Joint stiffness
    frictionloss=0.01,  # Friction loss
    armature=0.01,      # Rotor inertia
)
```

These can also be set per-joint using dictionary patterns:

```python
self.actuator_manager = ActuatorManager(
    self,
    joint_names=[".*"],
    kp=20,
    kv=0.5,
    damping={
        ".*_hip_joint": 0.2,
        ".*_thigh_joint": 0.1,
        ".*_calf_joint": 0.1,
    },
)
```

## Domain Randomization

Add domain randomization to any actuator value using `NoisyValue`.

```python
from genesis_forge.managers import ActuatorManager, NoisyValue

self.actuator_manager = ActuatorManager(
    self,
    joint_names=[".*"],
    kp=NoisyValue(20, noise=0.02),   # 20 +/- 0.02
    kv=NoisyValue(0.5, noise=0.01),  # 0.5 +/- 0.01
    max_force=NoisyValue(8.0, noise=0.05),  # 8.0 +/- 0.05
)
```

The noise value has a base number, and the noise to add to it. For example:

```python
NoisyValue(20, noise=0.5)
```

In this case, the base value is `20` and random noise will be added to it in the range of `-0.5` to `+0.5`.
The noise is randomized across each join at environment reset.

## Accessing Joint Information

The ActuatorManager provides several methods to access joint states:

### Get Joint Information

```python
# Get joint properties
num_dofs = self.actuator_manager.num_dofs
dof_names = self.actuator_manager.dofs_names
dof_indices = self.actuator_manager.dofs_idx
default_positions = self.actuator_manager.default_dofs_pos
```

### Get Current Joint States

```python
# In your observation or reward functions
positions = self.actuator_manager.get_dofs_position()
velocities = self.actuator_manager.get_dofs_velocity()
forces = self.actuator_manager.get_dofs_force()

# Add noise for training robustness
noisy_pos = self.actuator_manager.get_dofs_position(noise=0.01)
noisy_vel = self.actuator_manager.get_dofs_velocity(noise=0.02)
```

### Set Joint States

```python
# Set joint positions directly
positions = torch.zeros((self.num_envs, self.actuator_manager.num_dofs))
self.actuator_manager.set_dofs_position(positions)

# Control joint positions (PD controller)
target_positions = torch.zeros((self.num_envs, self.actuator_manager.num_dofs))
self.actuator_manager.control_dofs_position(target_positions)
```
