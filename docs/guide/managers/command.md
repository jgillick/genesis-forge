# Command Manager

The Command Manager generates high-level commands for goal-conditioned RL tasks. It handles command generation, resampling, visualization, and gamepad control for interactive testing.

- Generates random commands from specified ranges
- Resamples commands at regular intervals
- Provides debug visualization
- Supports gamepad control for testing

You can see a full example using the command manager in [examples/command_direction](https://github.com/jgillick/genesis-forge/tree/main/examples/command_direction).

<video autoplay="" muted="" loop="" playsinline="" controls="" src="../../media/command_manager.mp4"></video>

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
            debug_visualizer_cfg={
                "envs_idx": [0],  # only debug env 0
            }
        )
```

In this example, the `VelocityCommandManager` will generate random new X/Y/Z commands from the dict of ranges every 10 seconds.

### Debug visualization

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

When `debug_visualizer` is `True`, arrows will be displayed above your robot showing which direction is commanded v.s. which direction your robot is actually moving.

- **Green Arrow**: Commanded velocity (robot-relative, shown in world frame)
- **Blue Arrow**: Actual robot velocity (world frame)

!!! warning "Caution"

    The debug arrows can slow down the simulation since they need to be calculated and rendered for each environment on every step.

    It's recommended to only enable them for a small number of environments at a time with the `envs_idx` configuration setting.

### Standing Probability

Include periods where the robot should stand still:

```python
VelocityCommandManager(
    self,
    range={...},
    stopped_probability=0.2,  # 20% chance of zero command
)
```

### Using Velocity Commands in Rewards

Track how well the robot follows commands:

```python
from genesis_forge.mdp import rewards

RewardManager(
    self,
    cfg={
        "track_lin_vel": {
            "fn": rewards.command_tracking_lin_vel(
                vel_cmd_manager=self.velocity_command,
            ),
            "weight": 2.0,
        },
        "track_ang_vel": {
            "fn": rewards.command_tracking_ang_vel(
                vel_cmd_manager=self.velocity_command,
            ),
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
            "fn": self.velocity_command.observation,
        },
    },
)
```

### Gamepad Control

After your policy is trained, you can control the commanded values with a physical game controller:

```python title="train.py"

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

## Pose Command Manager

`Pose2dCommand` commands a goal *pose* rather than a velocity: a point to drive to, and the direction to be facing once there. Use it for navigation tasks, where the policy is told where to end up and has to choose its own route and speed to get there.

```python
import math
from genesis_forge.managers.command import Pose2dCommand

class MyEnv(ManagedEnvironment):
    def config(self):
        self.pose_command = Pose2dCommand(
            self,
            range={
                "x": (-2.5, 2.5),
                "y": (-2.5, 2.5),
                "heading": (-math.pi, math.pi),
            },
            goal_reached_threshold=0.15,  # how close counts as arrived
            resample_on_reached=True,     # reaching a goal earns a new one
            debug_visualizer=True,        # draw the goal in the scene
        )
```

The position and the heading are drawn independently, so the goal heading is not simply the direction the robot happened to approach from — it has to both get there and turn to face the right way. This is what a real arrival often needs: backing into a charging dock, or pulling up to a shelf facing it.

Goals are sampled in the environment's local frame, and a new one is drawn whenever the environment resets, whenever the goal is reached (with `resample_on_reached`), and on a timer if you set `resample_time_sec`. Left unset, `resample_time_sec` means a goal never expires — the robot keeps working at it until it arrives or the episode ends.

By default a goal counts as reached on position alone, whichever way the robot ends up facing. Set `heading_reached_threshold` if the robot must also be lined up before it has arrived.

### Keeping goals clear of the scene

A goal is never placed on top of anything else in the scene. Every entity — the obstacles, the robot itself, anything else that was added — gets a circle of clear space around it, sized from its own footprint plus the reach threshold. If a drawn goal lands inside one of those circles it is drawn again.

This is automatic and there is nothing to configure. It is also why a fresh goal never starts out already reached: the robot's own footprint is one of the things goals are kept clear of.

### Using Pose Commands in Observations

The observation is the goal from the robot's own point of view, in seven numbers: the goal vector (ahead, left), the distance, the cosine/sine of the bearing (which way to drive), and the cosine/sine of the heading error (which way to turn to face the goal heading).

The goal vector alone would locate the goal, but it mixes up *how far* with *which way* — a few centimeters out it is a tiny vector, so the steering signal fades exactly where steering has to be most precise. Reporting distance and bearing separately keeps the direction at full strength all the way in.

```python
ObservationManager(
    self,
    cfg={
        "goal_pose": {"fn": self.pose_command.observation},
    },
)
```

### Using Pose Commands in Rewards

Goal-reaching usually wants both a reward for *being* at the goal and one for *getting closer*, since the first is nearly flat when the robot starts far away. Add `heading_tracking` if you also care which way the robot is facing:

```python
from genesis_forge.mdp import rewards

RewardManager(
    self,
    cfg={
        # Strongest at the goal: makes the robot settle there rather than drive past
        "position_tracking": {
            "fn": rewards.position_tracking(
                pose_cmd_manager=self.pose_command,
            ),
            "weight": 1.0,
        },
        # Pays for closing the distance at any range: gets the robot moving
        "position_progress": {
            "fn": rewards.position_progress(
                pose_cmd_manager=self.pose_command,
            ),
            "weight": 1.0,
        },
        # Turning toward the heading the goal asks for
        "heading_progress": {
            "fn": rewards.heading_progress(
                pose_cmd_manager=self.pose_command,
            ),
            "weight": 0.5,
        },
        # A bonus for arriving
        "reached_goal": {
            "fn": rewards.reached_goal(
                pose_cmd_manager=self.pose_command,
            ),
            "weight": 10.0,
        },
    },
)
```

Like the velocity tracking rewards, `position_tracking` derives its sensitivity from the command range, so widening the range in a curriculum loosens the reward to match.

`heading_progress` asks for the goal heading at every distance by default, which suits a robot that can travel one way while facing another — a legged or omnidirectional robot. A robot that has to point where it is going cannot chase the goal heading from far away without driving sideways to reach the goal, so set `lines_up_within` to have it steer toward the goal while there is ground to cover and line up with the goal heading only on the final approach.

`position_progress` and `heading_progress` pay for *changing* rather than for *being*: an entity that stands still earns exactly nothing from either. That matters when `resample_on_reached` is set, because a reward paid every step for sitting near the goal can be worth far more than the one-off bonus for arriving — the entity learns to park just outside the reach threshold and hold the reward instead. `position_tracking` and `heading_tracking` are the "being there" versions, and are the better choice when the goal is *not* replaced on arrival; their docstrings carry the warning.

If you don't care which way the robot faces, simply leave out the heading reward; the heading is still drawn and observed, but nothing scores it.

The manager also exposes `distance_to_goal`, `heading_error`, and `goal_reached` for writing your own reward or termination functions.

See [examples/wheeled_robot_navigation](https://github.com/jgillick/genesis-forge/tree/main/examples/wheeled_robot_navigation) for a full navigation environment.

## Custom Command Manager

You can also create arbitrary commands with the basic `CommandManager`.

```python
# Create a random target height between 0.1 and 0.2
self.height_command = CommandManager(self, range=(0.1, 0.2))
```

```python
# Arbitrary number of ranges to support your command
self.target_command = CommandManager(self, range={
  "target_x": (-1.0, 1.0),
  "target_y": (-1.0, 1.0),
  "gait": (0.0, 5.0),
})
```

```python title="train.py"

# Connect gamepad axis 3 to the height command value
from genesis_forge.gamepads import Gamepad
gamepad = Gamepad()
env.command_manager.use_gamepad(gamepad, range_axis=3)

# Run policy...
```
