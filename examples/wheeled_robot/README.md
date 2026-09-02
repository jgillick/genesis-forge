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

## Deploy to a real robot

Export the trained policy along with the observation and action pipelines it was
trained against:

```shell
# With uv
uv run ./deploy.py

# Without uv
python ./deploy.py
```

This writes `./deploy_bundle.gfb` — a single file holding a readable
`manifest.json`, the policy as `policy.onnx` (plus the companion file its weights
live in), and recorded input/output pairs for an on-robot smoke test. Before writing
anything it runs the deployment code against the live training pipeline and refuses
to produce a bundle if the two disagree, then checks the packaged ONNX graph against
the policy it came from.

The script prints exactly what to wire up on the robot:

```
Bundle: ./.deploy_bundle
  control rate: 50.0 Hz (dt=0.02)
  observation vector: 18 values (18 per tick x 1 history)
  values you supply each tick:
    - velocity_cmd (3 values)
    - angle_velocity (3 values)
    ...
    - actions (3 values)
  joint targets produced (3):
    - [velocity] base_back_wheel_joint, base_right_wheel_joint, base_left_wheel_joint
```

Note the `[velocity]` tag: this robot's action manager produces wheel *velocities*,
so those targets go to a velocity command rather than a position one.

Copy that one file to the robot and install just the runtime — it needs numpy only,
no simulator:

```shell
pip install genesis-forge-runtime[onnx]
```

See the [deployment guide](https://genesis-forge.readthedocs.io/en/latest/guide/deployment/)
for the full control loop.

To export the pipeline contract before you have a trained checkpoint — useful for
wiring up the robot side early — skip the policy:

```shell
uv run ./deploy.py --skip-policy
```
