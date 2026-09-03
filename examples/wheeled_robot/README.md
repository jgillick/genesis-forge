# Wheeled Robot - Commanded direction

Train a simplified [Freenove 4WD car](https://store.freenove.com/products/fnk0043) platform, to move in a commanded direction, controlled programmatically or through a gamepad controller.

This builds on the [command_direction](../command_direction/) example, but demonstrates driving four continuously-rotating wheels instead of legged position control.

The velocity action manager setup looks like this:

```python
def config(self):
    # ...

    self.wheel_motors = ActuatorManager(
        self,
        joint_names=[
            "TT_Motor-[1-4]_axel",
        ],
        kv=1.0,
    )
    self.action_manager = VelocityActionManager(
        self,
        scale=5.0,
        actuator_manager=self.wheel_motors,
    )
```

This is a ["differential steering"](https://en.wikipedia.org/wiki/Differential_steering) robot, which means, all 4 wheels are facing the same direction and turning is done by changing the speed each wheel is rotating at. As such, the velocity command can only direct the robot to go forwards/backwards, and turn along the center axis.

```python
    self.velocity_command = VelocityCommandManager(
        self,
        range={
            "lin_vel_x": (-0.1, 0.1), # forward/backward
            "lin_vel_y": (-0.0, 0.0), # cannot move side-to-side
            "ang_vel_z": (-0.2, 0.2), # turning
        },
        ...
    )
```

See [`environment.py`](./environment.py) for the full configuration, and the [`VelocityActionManager` guide](../../docs/guide/managers/action.md) for how the manager generalizes to any continuously-rotating-joint robot.

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

The Genesis Forge training environment will save videos while training that can be viewed in `./logs/wheeled-robot-command/videos`.

## Gamepad control

You can use a game controller (Xbox, PlayStation, Nintendo Switch Pro, Logitech F310/F710, etc.) to control the robot in the trained policy yourself.

Simply connect your gamepad and run:

```shell
# With uv
uv run ./gamepad.py

# Without uv
python ./gamepad.py
```

You should now be able to use the joysticks to control the wheeled robot.
