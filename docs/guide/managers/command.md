# Command Manager

The Command Manager generates high-level commands for goal-conditioned RL tasks. It handles command generation, resampling, visualization, and gamepad control for interactive testing.

- Generates random commands from specified ranges
- Resamples commands at regular intervals
- Provides debug visualization
- Supports gamepad control for testing

You can see a full example using the command manager in [examples/command_direction](https://github.com/jgillick/genesis-forge/tree/main/examples/command_direction).

## Velocity Command Manager

The most common command manager is `VelocityCommandManager` for locomotion tasks:

<video autoplay="" muted="" loop="" playsinline="" controls="" src="../../media/cmd_velocity.mp4"></video>

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

`Pose2dCommand` commands a goal _pose_ rather than a velocity: a point to move to, and the direction to be facing once there. Use it for navigation tasks, where the policy is told where to end up and has to choose its own route and speed to get there.

<video autoplay="" muted="" loop="" playsinline="" controls="" src="../../media/cmd_pose2d.mp4"></video>

You can see a full example in [examples/wheeled_robot_navigation](https://github.com/jgillick/genesis-forge/tree/main/examples/wheeled_robot_navigation).

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
            goal_reached_threshold=0.15,                # how close counts as arrived
            heading_reached_threshold=math.radians(30),  # and how well lined up
            resample_on_reached=True,                    # reaching a goal earns a new one
            entity_manager=self.robot_manager,           # faster than reading the solver
            debug_visualizer=True,                       # draw the goal in the scene
        )
```

The position and the heading are sampled independently, so the goal heading is not simply the direction the robot happened to approach from — it has to both get there and turn to face the right way. This is what a real arrival often needs: backing into a charging dock, or pulling up to a shelf facing it.

A goal counts as reached once the robot is within `goal_reached_threshold` of the position _and_ within `heading_reached_threshold` of the goal heading. A new goal is drawn on reset, on arrival (with `resample_on_reached`), and — if you set `resample_time_sec` — once the robot has spent too long on the one it has.

If you only care about the position, leave the `heading` range out (or set it to `None`). The goal is then just a point to reach, arrived at facing any direction, and the heading drops out of the command, the observation, and the arrival check.

```python
self.pose_command = Pose2dCommand(
    self,
    range={"x": (-2.5, 2.5), "y": (-2.5, 2.5)},
)
```

### Visualizing the goal

With `debug_visualizer=True`, a marker is drawn at each goal, turning from green to red when the goal has been reached: an arrow pointing the way to face on arrival, or a ball for a position-only goal. Pass `terrain_manager` if the ground isn't flat, so the marker sits above the terrain.

```python
self.pose_command = Pose2dCommand(
    self,
    range={"x": (-2.5, 2.5), "y": (-2.5, 2.5), "heading": (-math.pi, math.pi)},
    terrain_manager=self.terrain_manager,
    debug_visualizer=True,
    debug_visualizer_cfg={
        "envs_idx": [0],       # only draw the first environment's goal
        "marker_height": 0.05, # how far above the ground the marker floats
    },
)
```

### Using Pose Commands in Observations

The observation is the goal from the robot's own point of view, in seven numbers: the goal vector (ahead, left), the distance, the cosine/sine of the bearing (which way to drive), and the cosine/sine of the heading error (which way to turn to face the goal heading). A position-only goal has no heading error, so its observation is the first five.

```python
ObservationManager(
    self,
    cfg={
        "goal_pose": {"fn": self.pose_command.observation},
    },
)
```

### Using Pose Commands in Rewards

Arriving is a rare event, so the bonus for it is far too sparse to learn from on its own. Pair it with a reward for _getting closer_, and — if you care which way the robot ends up facing — one for turning the right way:

```python
from genesis_forge.mdp import rewards

RewardManager(
    self,
    cfg={
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
                # How close to the goal the robot should switch from steering
                # toward the goal to lining up with the goal heading.
                lines_up_within=0.75,
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

By default, `heading_progress` asks for the goal heading at every distance — right for a legged or omnidirectional robot, which can travel one way while facing another. A robot that has to point where it is going (like a car) should set the `lines_up_within` arg, so it steers toward the goal until it is close, and only then lines up with the goal heading.

`position_progress` and `heading_progress` pay for _changing_ rather than for _being_: an entity that stands still earns exactly nothing from either.

If you don't care which way the robot faces, leave the `heading` range out of the command entirely; there is then no heading to reward, observe, or arrive lined up with.

The manager also exposes `distance_to_goal`, `bearing_error`, `heading_error`, and `goal_reached` for writing your own reward or termination functions.

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
