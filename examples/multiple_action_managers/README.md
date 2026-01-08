# Multiple Action Managers

Demonstrates using multiple action managers for a single robot. This can be useful if some actuators need to be controlled differently
from each other. Currently, Genesis Forge only has two types of positional action managers, but in the future, it will likely be expanded
to include velocity or torque based action managers.

To use multiple action managers, you pass a filter to each, specifying which actuators it should control:

```python
# First define your actuator manager
self.actuator_manager = ActuatorManager(self, joint_names=".*", kp=20, kv=0.5)

# Then define your action managers
self.hip_action_manager = PositionWithinLimitsActionManager(
    self,
    actuator_manager=self.actuator_manager,
    actuator_joints=[".*_hip_joint"],
    limit=(-0.8, 0.8),
)
self.leg_action_manager = PositionActionManager(
    self,
    actuator_manager=self.actuator_manager,
    actuator_joints=[
        ".*_thigh_joint",
        ".*_calf_joint",
    ],
    scale=0.25,
    clip=(-100.0, 100.0),
)
```

## Training

This will be trained using the [rsl_rl](https://github.com/leggedrobotics/rsl_rl) training library. So first, we need to install that and tensorboard:

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

The Genesis Forge training environment will also save videos while training that can be viewed in `./logs/go2-walking/videos`.

https://github.com/user-attachments/assets/be46df1b-35e5-4b5b-9bbc-f543210dd463

## Evaluation

Now you can view the trained policy:

```bash
python ./eval.py ./logs/go2-walking/
```
