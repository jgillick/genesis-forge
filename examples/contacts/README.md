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

The Genesis Forge training environment will also save videos while training that can be viewed in `./logs/go2-foot-step/videos`.
