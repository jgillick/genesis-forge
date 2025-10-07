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

A custom observation function takes in the environment as the first parameter, as well as any other parameter defined in the `params` dict at the ObservationManager. The returned value should be a tensor with a float value for each environment.

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
            "params": {
                "contact_manager": self.contact_manager,
                "threshold": 5.0,
            },
        },
    },
)
```

## Asymmetrical observations (policy v.s. critic)

In some cases, you might want to have different observation sets for specific components of your algorithm. For example, in an actor-critic model, you might want to have a set of "privileged" observations specifically for the critic that the policy doesn't have access to.

By giving the critic privileged information (like ground truth contact forces or internal robot states), the value estimates can become more accurate. This provides a better training signal for the actor, leading to faster and more stable policy learning. The policy still learns to act based only on realistic sensor data, while the critic can make better value predictions.

To do this, just define the component name on the observation manager.

:::{important}
You must at least define a nameless, or "policy", observation set. This represents the observations that will be available to your policy during deployment (e.g., real sensor data).
:::

```python
# Policy observations - what the robot can actually sense during deployment
ObservationManager(
    self,
    name="policy",
    noise=0.1,  # Add noise to simulate real sensor noise
    cfg={
        "velocity_cmd": { "fn": self.velocity_command.observation },
        "angle_velocity": { "fn": lambda env: self.robot_manager.get_angular_velocity() },
        "linear_velocity": { "fn": lambda env: self.robot_manager.get_linear_velocity() },
        "projected_gravity": { "fn": lambda env: self.robot_manager.get_projected_gravity() },
        "dof_position": { "fn": lambda env: self.action_manager.get_dofs_position() },
        "dof_velocity": {
            "fn": lambda env: self.action_manager.get_dofs_velocity(),
            "scale": 0.05,
        },
        "actions": { "fn": lambda env: self.action_manager.get_actions() },
    },
)

# Critic observations - includes privileged information for better value estimation
ObservationManager(
    self,
    name="critic",
    noise=0.0,  # Critic observations should not be noisy
    cfg={
        # Privileged observations (not available to policy at runtime)
        "foot_contact_force": {
            "fn": observations.contact_force,
            "params": {
                "contact_manager": self.foot_contact_manager,
            },
        },
        "dof_force": {
            "fn": lambda env: self.action_manager.get_dofs_force(),
            "scale": 0.1,
        },
        # Same observations as policy (for consistency)
        "velocity_cmd": { "fn": self.velocity_command.observation },
        "angle_velocity": { "fn": lambda env: self.robot_manager.get_angular_velocity() },
        "linear_velocity": { "fn": lambda env: self.robot_manager.get_linear_velocity() },
        "projected_gravity": { "fn": lambda env: self.robot_manager.get_projected_gravity() },
        "dof_position": { "fn": lambda env: self.action_manager.get_dofs_position() },
        "dof_velocity": {
            "fn": lambda env: self.action_manager.get_dofs_velocity(),
            "scale": 0.05,
        },
        "actions": { "fn": lambda env: self.action_manager.get_actions() },
    },
)
```

By default, the `policy` observation set will be returned as the main observations in the step function. However, all named sets will be set in a [TensorDict](https://docs.pytorch.org/tensordict/stable/index.html) and assigned to `extras["observations"]`.

```
obs, rewards, terminations, truncations, extras = env.step(actions)

print(obs) # <- Policy observations
print(extras["observations"]) # <- TensorDict with all observations
```

The [rsl_rl wrapper](project:/api/wrappers/rsl_rl.md) will automatically route these to the RSL_RL algorithm. But be sure to set the rsl_rl `obs_groups` config properly:

```
"obs_groups": {"policy": ["policy"], "critic": ["critic"]},
```
