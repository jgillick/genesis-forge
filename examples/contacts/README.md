# Go2 - Contact detection and foot air time rewards

This builds on the [command direction example](../command_direction/), and we add the contact manager to track
foot step "air time", so we can reward the robot for taking longer steps.

```python
def config(self):
    # ...

    # Contact manager to track foot steps
    self.foot_contact_manager = ContactManager(
        self,
        link_names=[".*_foot"],
        track_air_time=True,
        air_time_contact_threshold=1.0, # How much contact force is considered a step
    )

    # Add command tracking to the reward manager
    RewardManager(
        self,
        logging_enabled=True,
        cfg={
            "foot_air_time": {
                "weight": 1.25,
                "fn": rewards.feet_air_time,
                "params": {
                    "time_threshold": 0.5, # Target air-time, in seconds
                    "contact_manager": self.foot_contact_manager,
                    "vel_cmd_manager": self.velocity_command, # reduces the penalty if the the velocity command is close to zero
                },
            },
            # ... other rewards ...
        },
    )
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

The Genesis Forge training environment will also save videos while training that can be viewed in `./logs/go2-foot-step/videos`.

## Evaluation

Now you can view the trained policy:

```bash
python ./eval.py ./logs/go2-foot-step/
```

## Gamepad control

You can use a game controller (Xbox, PlayStation, Nintendo Switch Pro, Logitech F310/F710, etc.) to control the robot in the trained policy yourself.

Simply connect your gamepad and run:

```python
python ./gamepad.py
```

You should now be able to use the joysticks to control the Go2 robot.
