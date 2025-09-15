# Observation Manager

The Observation Manager defines what your RL agent observes from the environment. It handles observation space creation, data collection, scaling, and noise injection for training robustness.

You can see a full example using the observation manager in [examples/simple](https://github.com/jgillick/genesis-forge/tree/main/examples/simple).

## Basic Usage

```python
from genesis_forge.managers import ObservationManager

class MyEnv(ManagedEnvironment):
    def config(self):
        ObservationManager(
            self,
            cfg={
                "projected_gravity": {
                    "fn": observations.entity_projected_gravity,
                    "noise": 0.1,  # Add noise for robustness
                },
                "joint_positions": {
                    "fn": observations.entity_dofs_position,
                    "params": {
                        "action_manager": self.action_manager
                    }
                },
                "joint_velocities": {
                    "fn": observations.entity_dofs_velocity,
                    "params": {
                        "action_manager": self.action_manager
                    }
                    "scale": 0.05,  # Scale down velocities
                },
            },
        )
```

## Observation Configuration

Each observation configuration dict can have:

- **fn**: Function that returns observation values
- **params**: Additional parameters to be passed to that function
- **scale**: Multiplier to normalize values
- **noise**: Random noise scale for training robustness

```python
ObservationManager(
    self,
    cfg={
        "robot_velocity": {
            "fn": observations.entity_linear_velocity,
            "scale": 2.0,    # Scale up small values
            "noise": 0.05,   # Add 5% noise
        },
        "contact_forces": {
            "fn": observations.entity_dofs_force,
            "params": {   # Pass parameters to entity_dofs_force
                "action_manager": self.action_manager
                "clip_to_max_force": True
            },
        },
        "actions": { # Use lambda for simple data returns
            "fn": lambda env: self.action_manager.get_actions(),
        },
    },
)
```

## Scaling and Normalization

### Why Scale?

Neural networks work best with inputs roughly in [-1, 1] range:

```python
# Without scaling: values vary wildly
"joint_velocity": {
    "fn": get_joint_vel,  # Returns values in [-100, 100]
}

# With scaling: normalized to reasonable range
"joint_velocity": {
    "fn": get_joint_vel,
    "scale": 0.01,  # Now in [-1, 1] range
}
```

### Common Scaling Values

- positions: `1.0` - Often already in radians
- velocities: `0.05`
- accelerations: `0.01`
- forces : `0.001` - Can be 1000s of N
- distances: `1.0` - Usually in meters
- angles: `1.0` - Already in radians

## Adding Noise

Add noise to observations for better Sim2Real robustness.

You can set the default noise value for all observations at the top of the config, which can be overridden in each config.

```python
ObservationManager(
    self,
    noise=0.1 # set this value for all observations
    cfg={
        "projected_gravity": {
            "fn": observations.entity_projected_gravity,
        },
        "joint_positions": {
            "fn": observations.entity_dofs_position,
            "params": {
                "action_manager": self.action_manager
            },
            "scale": 1.0 # supersedes the default setting
        },
        "joint_velocities": {
            "fn": observations.entity_dofs_velocity,
            "params": {
                "action_manager": self.action_manager
            }
        },
    },
)
```

### Different Noise Levels

Some observations are noisier in reality:

```python
cfg={
    # Encoders are precise
    "joint_pos": {
        "fn": get_joint_pos,
        "noise": 0.001,  # Very little noise
    },

    # IMU is moderately noisy
    "orientation": {
        "fn": get_orientation,
        "noise": 0.05,
    },

    # Force sensors are very noisy
    "contact_forces": {
        "fn": get_forces,
        "noise": 0.2,
    },
}
```

## Custom Observation Functions

A custom observation function takes in the environment as the first parameter, as well as any other parameter defined in the `params` dict at the ObservationManager. The returned value should be a tensor with a value, or list of values, for each environment.

### Simple Functions

```python
def get_height_above_ground(env):
    """Distance from base to ground."""
    base_height = env.robot.get_pos()[:, 2]
    terrain_height = env.terrain.get_height_at(env.robot.get_pos())
    return base_height - terrain_height

ObservationManager(
    self,
    cfg={
        "height": {
            "fn": get_height_above_ground,
            "scale": 1.0,
        },
    },
)
```

### Complex Observations

```python
def feet_in_contact(env, contact_manager: ContactManager, threshold=1.0):
    """Return the number of feet in contact with the ground with at least `threshold` force."""
    has_contact = contact_manager.contacts[:, :].norm(dim=-1) > threshold
    return has_contact.sum(dim=1)

ObservationManager(
    self,
    cfg={
        "foot_contact": {
            "fn": feet_in_contact,
            "params": {"contact_manager": self.contact_manager, "threshold": 5.0},
        },
    },
)
```
