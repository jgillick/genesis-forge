# Wheeled Robot - Commanded direction

Train the [LeKiwi](https://github.com/SIGRobotics-UIUC/LeKiwi) platform, a 3-wheeled omnidirectional mobile base from the LeRobot ecosystem, to move in a commanded direction, controlled programmatically or through a gamepad controller.

This builds on the [command_direction](../command_direction/) example, but demonstrates driving continuously-rotating wheel joints instead of legged position control. The main thing that differs from a legged example:

**`VelocityActionManager`** instead of `PositionActionManager` — the policy's actions are raw target velocities for the 3 wheel joints, not joint positions. Wheels rotate continuously, so there's no bounded "position" to control toward.

```python
def config(self):
    # ...

    self.actuator_manager = ActuatorManager(
        self,
        joint_names=[
            "base_back_wheel_joint",
            "base_left_wheel_joint",
            "base_right_wheel_joint"
        ],
        kv=1.0,
    )
    self.action_manager = VelocityActionManager(
        self,
        scale=2.0,
        actuator_manager=self.actuator_manager,
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
