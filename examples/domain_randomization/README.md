# Go2 - Domain Randomization

A common way to ensure a smoother [Sim2Real](https://medium.com/@sim30217/sim2real-fa835321342a) transition is using "domain randomization" during training. Which basically means adding noise to the environment, so that the robot adapts and learns a
more general and robust policy. Due to this noise, the robot can take longer to train and rewards might need more tuning to
encourage the desired behavior.

This builds on the [command_direction](../command_direction/) example, and introduces random noise in the following places:

- Actuator values (gains, default positions, damping, etc) - No actuator performs as perfect as the data sheet, so the policy learns to adapt.
- Action latency - actions are delivered to the robot between 0 - 2 steps after the policy provides them, to emulate system latency.
- Observations - No sensors return perfectly clean data.
- Robot mass - Makes the robot heavier or lighter on reset to change its balance and loading.
- Random pushes - Shoves the robot every 4 - 8 seconds, so it learns to recover.

Here are the relevant snippets:

```python
    # Randomly add/subtract mass to the robot's body on reset
    self.robot_manager = EntityManager(
        self,
        entity=self.robot,
        on_reset={
            "mass_randomization": {
                "fn": reset.randomize_link_mass_shift(
                    link_name="base",
                    mass_range=(-0.5, 1.0),  # kg
                    # Changing mass is slow, so change it at most once every 100 steps
                    min_step_count_between_reset=100,
                ),
            },
        },
    )

    # Push and twist the robot for 0.1-0.3 seconds, every 4-8 seconds
    self.event_manager = EventManager(
        self,
        cfg={
            "push": {
                "fn": events.push_force(
                    entity_manager=self.robot_manager,
                    force=(80.0, 150.0),  # newtons
                    torque=(5.0, 10.0),  # newton-meters, around a random axis
                ),
                "interval_range_s": (4.0, 8.0),
                "duration_range_s": (0.1, 0.3),
            },
        },
    )

    # Actuator settings with random noise
    self.actuator_manager = ActuatorManager(
        self,
        joint_names=[".*"],
        default_pos={
            # Randomize the default positions by +/- 0.02 radians
            ".*_hip_joint": NoisyValue(0.0, 0.02),
            "FL_thigh_joint": NoisyValue(0.8, 0.02),
            "FR_thigh_joint": NoisyValue(0.8, 0.02),
            "RL_thigh_joint": NoisyValue(1.0, 0.02),
            "RR_thigh_joint": NoisyValue(1.0, 0.02),
            ".*_calf_joint": NoisyValue(-1.5, 0.02),
        },
        kp=NoisyValue(25, 1.0),  # +/- 1.0
        kv=NoisyValue(0.5, 0.05),  # +/- 0.05
        damping=NoisyValue(0.5, 0.05),  # +/- 0.05
        frictionloss=NoisyValue(0.1, 0.01),  # +/- 0.01
        armature=NoisyValue(0.01, 0.001),  # +/- 0.001
    )

    # Delay each robot's actions by 0-2 steps, redrawn at each reset
    self.action_manager = PositionActionManager(
        self,
        # ...
        delay_step=(0, 2),
    )

    # Add noise to the sensor observations
    ObservationManager(
        self,
        history_len=3,  # history helps the policy adapt to delayed actions and infer acceleration
        cfg={
            # ...
            "angle_velocity": {
                "fn": lambda env: self.robot_manager.get_angular_velocity(),
                "noise": 0.1,  # IMU gyroscope noise (rad/s)
            },
            # ...
        }
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
pip install -e ../../packages/genesis-forge "rsl-rl-lib~=5.0" tensorboard
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

The Genesis Forge training environment will also save videos while training that can be viewed in `./logs/go2-randomization/videos`.
