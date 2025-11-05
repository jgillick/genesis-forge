# Go2 - Domain Randomization

A common way to ensure a smoother [Sim2Real](https://medium.com/@sim30217/sim2real-fa835321342a) transition is using "domain randomization" during training.
This basically means adding noise to the environment, so that the robot adapts and learns a more general and robust policy.

This builds on the [command_direction](../command_direction/) example, and introduces random noise in these places:

- Actuator values (gains, default positions, damping, etc) - No actuator performs as perfect as the data sheet, so the policy learns to adapt.
- Observations - No sensors return perfectly clean data.
- Robot mass - Makes the robot heavier or lighter at each reset to change it's balance and loading.

Here are the relevant snippets:

```python
    # Randomly add/subtract mass to the robot's body
    self.robot_manager = EntityManager(
        self,
        entity_attr="robot",
        on_reset={
            "mass": {
                "fn": reset.randomize_link_mass_shift,
                "params": {
                    "link_name": "base",
                    "add_mass_range": [-100.0, 100.0],
                },
            },
        },
    )

    # Actuator settings with random noise
    self.actuator_manager = ActuatorManager(
        self,
        joint_names=[".*"],
        default_pos={
            # Randomize the default positions by +/- 0.1 radians
            ".*_hip_joint": NoisyValue(0.0, 0.05),
            "FL_thigh_joint": NoisyValue(0.8, 0.05),
            "FR_thigh_joint": NoisyValue(0.8, 0.05),
            "RL_thigh_joint": NoisyValue(1.0, 0.05),
            "RR_thigh_joint": NoisyValue(1.0, 0.05),
            ".*_calf_joint": NoisyValue(-1.5, 0.05),
        },
        kp=NoisyValue(20, 3.0),  # +/- 3.0
        kv=NoisyValue(0.5, 0.15),  # +/- 0.15
        damping=NoisyValue(0.5, 0.25),  # +/- 0.25
        frictionloss=NoisyValue(0.15, 0.1),  # +/- 0.1
    )

    # Set 0.01 noise on all observations
    ObservationManager(
        self,
        cfg={
            # ...
            "angle_velocity": {
                "fn": lambda env: self.robot_manager.get_angular_velocity(),
                "noise": 0.05, # IMU gyroscope noise: ~0.05-0.2 rad/s
            },
            # ...
        }
    }
```

## Training

We will be training the robot with the [rsl_rl](https://github.com/leggedrobotics/rsl_rl) training library. So first, we need to install that and tensorboard:

```bash
pip install tensorboard rsl-rl-lib>=2.2.4
```

Now you can run the training with:

```bash
python ./train.py
```

You can view the training progress with:

```bash
tensorboard --logdir ./logs/
```

The Genesis Forge training environment will also save videos while training that can be viewed in `./logs/go2-randomization/videos`.

## Evaluation

Now you can view the trained policy:

```bash
python ./eval.py ./logs/go2-randomization/
```
