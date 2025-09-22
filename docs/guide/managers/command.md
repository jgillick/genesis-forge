# Command Manager

The Command Manager generates high-level commands for goal-conditioned RL tasks. It handles command generation, resampling, visualization, and gamepad control for interactive testing.

- Generates random commands from specified ranges
- Resamples commands at regular intervals
- Provides debug visualization
- Supports gamepad control for testing

You can see a full example using the command manager in [examples/command_direction](https://github.com/jgillick/genesis-forge/tree/main/examples/command_direction).

```{eval-rst}
.. video:: _images/command_manager.mp4
```

## Velocity Command Manager

The most common command manager is `VelocityCommandManager` for locomotion tasks:

```python
from genesis_forge.managers.command import VelocityCommandManager

class MyEnv(ManagedEnvironment):
    def config(self):
        self.velocity_command = VelocityCommandManager(
            self,
            range={
                "lin_vel_x": [-1.0, 1.0],  # Forward/backward
                "lin_vel_y": [-0.5, 0.5],  # Left/right
                "ang_vel_z": [-1.0, 1.0],  # Rotation
            },
            resample_time_sec=10,  # Resample new commands every 10 seconds
            debug_visualizer=True, # Show command arrows
        )
```

In this example, ever 10 seconds, the `VelocityCommandManager` will generate random new X/Y/Z commands from the dict of ranges provided, for all environments.

When `debug_visualizer` is `True`, an arrow will be displayed above your robot showing which direction is commanded (green) and which direction your robot is actually moving (blue).

### Standing Probability

Include periods where the robot should stand still:

```python
VelocityCommandManager(
    self,
    range={...},
    standing_probability=0.2,  # 20% chance of zero command
)
```

### Velocity Command Visualization

Enable visual feedback of commands vs actual velocity:

```python
VelocityCommandManager(
    self,
    range={...},
    debug_visualizer=True,
    debug_visualizer_cfg={
        "envs_idx": [0],  # Only add this to environment zero
        "arrow_offset": 0.5,  # Height above robot
    },
)
```

Understanding the arrows:

- **Green Arrow**: Commanded velocity (robot-relative, shown in world frame)
- **Blue Arrow**: Actual robot velocity (world frame)

:::{caution}
The debug arrows can slow down the simulation since they need to be calculated and rendered for each environment on every step.

It's recommended to only enable them for a small number of environments at a time with the `envs_idx` configuration setting.
:::

### Using Velocity Commands in Rewards

Track how well the robot follows commands:

```python
from genesis_forge.mdp import rewards

RewardManager(
    self,
    cfg={
        "track_lin_vel": {
            "fn": rewards.command_tracking_lin_vel,
            "params": {"vel_cmd_manager": self.velocity_command},
            "weight": 2.0,
        },
        "track_ang_vel": {
            "fn": rewards.command_tracking_ang_vel,
            "params": {"vel_cmd_manager": self.velocity_command},
            "weight": 1.0,
        },
    },
)
```

### Using Commands in Observations

Include commands in the observation space:

```python
ObservationManager(
    self,
    cfg={
        "velocity_command": {
            "fn": lambda env: self.velocity_command.observation(),
        },
        # Or access raw command
        "raw_command": {
            "fn": lambda env: self.velocity_command.command,
        },
    },
)
```

### Gamepad Control

Control commands with a physical gamepad for testing:

```{code-block} python
:caption: train.py

from genesis_forge.gamepads import Gamepad

#...

# Setup your environment
env = MyEnv(num_envs=1, headless=False)
env.build()

# Add your gamepad to the velocity command manager
gamepad = Gamepad()
env.velocity_command.use_gamepad(gamepad)

# Run policy...
```

## Custom Command Manager

You can also create arbitrary commands with the basic `CommandManager`.

```python
# Create a random target height between 0.1 and 0.2
self.height_command = CommandManager(self, range=(0.1, 0.2))
```

```python
# Arbitrary number of ranges to support your command
self.target_command = CommandManager(self, range={
  "target_x": range=(-1.0, 1.0),
  "target_y": range=(-1.0, 1.0),
  "gait": range=(0.0, 5.0),
})
```

### Gamepad Control

Using the height command, from above, as an example:

```{code-block} python
:caption: train.py

from genesis_forge.gamepads import Gamepad

#...

# Setup your environment
env = MyEnv(num_envs=1, headless=False)
env.build()

# Connect joystick axis 3 to the height command value
gamepad = Gamepad()
env.command_manager.use_gamepad(gamepad_controller, range_axis=3)

# Run policy...
```
