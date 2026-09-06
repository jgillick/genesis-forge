# Wheeled Robot - Commanded direction

Train a simplified [Freenove 4WD car](https://store.freenove.com/products/fnk0043) platform, to move in a commanded direction, controlled programmatically or through a gamepad controller.

This is a ["differential steering"](https://en.wikipedia.org/wiki/Differential_steering) robot, which means, all 4 wheels are facing the same direction and turning is done by changing the speed of the wheels on either side.

## Driving: two actions, not four

The robot has four wheels, in a ["differential steering"](https://en.wikipedia.org/wiki/Differential_steering) configuration, which means, turning is done by changing the speed of the wheels on the left and right side. This also means that the wheels on each side (left or right) act like a synchronized pair, each turning the same speed.

Asking the model to provide four actions for a car that only truly needs two, would be a waste. So we can group the actuators together using action_groups.

```python
self.wheel_action_manager = VelocityActionManager(
    self,
    action_groups=[
        ["TT_Motor-3_axel", "TT_Motor-4_axel"],  # left side
        ["TT_Motor-1_axel", "TT_Motor-2_axel"],  # right side
    ],
    ...
)
```

The other peculiarity is that the front motors are mounted in reverse orientation to the rear motors. So if you applied the same positive velocity to all four motors, the front and rear motors would be moving in opposite directions and the car would move nowhere.

<img src="./motors.png" width="200" alt="Motor orientation" />

To solve this, we can use negative action scaling to reverse the target actuator actions on the front motors.

```python
self.wheel_action_manager = VelocityActionManager(
    self,
    action_groups=[
        ["TT_Motor-3_axel", "TT_Motor-4_axel"],  # left side
        ["TT_Motor-1_axel", "TT_Motor-2_axel"],  # right side
    ],
    scale={
        "TT_Motor-1_axel": -1,  # right front (reverse action sign)
        "TT_Motor-2_axel": +1,  # right rear
        "TT_Motor-3_axel": -1,  # left front (reverse action sign)
        "TT_Motor-4_axel": +1,  # left rear
    },
    ...
)
```

Now, if the model sends an action of `5` to the left motors, the left/front action will be translated to `-5` and the left/rear action will remain as `+5`.

Lastly, since the robot can only move forwards and backwards, but not side-to-side, the commanded linear velocity range is set to zero.

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
