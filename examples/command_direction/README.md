# Go2 - Commanded direction

Train a Go2 robot to move in a commanded direction, controlled programmatically or through a gamepad controller.

This builds on the [simple example](../simple/) example. The only thing necessary to convert that example from a robot that walks in a straight line on the X-axis, is adding the `VelocityCommandManager`, and related rewards and observations.

```python
def config(self):
    # ...

    # Control manager
    # Sample random directions from the X/Y ranges, as well as a rotation velocity around the Z axis.
    self.velocity_command = VelocityCommandManager(
        self,
        range={
            "lin_vel_x": [-1.0, 1.0],
            "lin_vel_y": [-1.0, 1.0],
            "ang_vel_z": [-1.0, 1.0],
        },
        standing_probability=0.05,
        resample_time_sec=5.0,
        debug_visualizer=True,
        debug_visualizer_cfg={
            "envs_idx": [0],
        },
    )

    # Add command tracking to the reward manager
    RewardManager(
        self,
        logging_enabled=True,
        cfg={
            "tracking_lin_vel": {
                "weight": 1.0,
                "fn": rewards.command_tracking_lin_vel(
                    vel_cmd_manager=self.velocity_command,
                    entity_manager=self.robot_manager,
                ),
            },
            "tracking_ang_vel": {
                "weight": 0.2,
                "fn": rewards.command_tracking_ang_vel(
                    vel_cmd_manager=self.velocity_command,
                    entity_manager=self.robot_manager,
                ),
            },
            # ... other rewards ...
        },
    )

    # Add command to observations
    ObservationManager(
        self,
        cfg={
            "velocity_cmd": {"fn": self.velocity_command.observation, "scale": 0.5},
            # ... other observations ...
        },
    )
```

## Training

### With [uv](https://docs.astral.sh/uv/) (recommended)

Training:

```shell
uv run ./train.py
```

Evaluation:

```shell
uv run ./eval.py
```

### Without uv:

Install dependencies

```shell
pip install -e ../../ "rsl-rl-lib~=5.0" tensorboard
```

Train:

```shell
python ./train.py
```

Evaluation:

```shell
python ./eval.py
```

## Monitor training status

You can view the training progress with:

```shell
tensorboard --logdir ./logs/
```

## Training videos

The Genesis Forge training environment will also save videos while training that can be viewed in `./logs/go2-command/videos`.

## Gamepad control

You can use a game controller (Xbox, PlayStation, Nintendo Switch Pro, Logitech F310/F710, etc.) to control the robot in the trained policy yourself.

Simply connect your gamepad and run:

****

```shell
# With uv
uv run ./gamepad.py

# Without uv
python ./gamepad.py
```

You should now be able to use the joysticks to control the Go2 robot.
