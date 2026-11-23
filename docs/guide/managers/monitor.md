# Monitors

Monitors calculate values about your robot or the environment on every step and add them to your training logs (tensorboard, wandb, or wherever your training library writes to). Their main job is to help you see what's going on during training: how much power the robot draws, how hard each joint works, how high the body rides.

Every value a monitor calculates can also be read by your rewards, terminations, and observations. So when you notice something in the logs worth penalizing, the value is already there.

There are three monitors:

- `Monitor` - log anything, from your own functions
- `PowerMonitor` - estimates the electrical power your motors would draw
- `ActuatorMonitor` - tracks torque, peaks, and how often joints hit their force limit

## Basic Usage

Each config item is a function that returns one value per environment, shape `(num_envs,)`:

```python
from genesis_forge.managers import Monitor

class MyEnv(ManagedEnvironment):
    def config(self):
        self.body_monitor = Monitor(
            self,
            logging_tag="Body",  # The section name in your logs
            cfg={
                "height": {
                    "fn": lambda env: self.robot.get_pos()[:, 2],
                },
            },
        )
```

This logs the average height across all your environments, as `Body / height`.

## Multiple reducers

Set `reduce` to choose how the values from all your environments are combined for the log. The mean is logged under the item's name, and the others add a suffix:

```python
"height": {
    "fn": lambda env: self.robot.get_pos()[:, 2],
    "reduce": ["mean", "min", "p99"],  # <~ Body / height, Body / height_min, Body / height_p99
},
```

You can use `"mean"`, `"min"`, `"max"`, `"std"`, `"rms"`, or any percentile, like `"p95"`. A single one doesn't need a list: `"reduce": "max"`.

## Single numbers

Some values might be the same for every environment. Return a plain number and it's logged as-is:

```python
self.curriculum_level = 5
...
Monitor(
    self,
    logging_tag="Curriculum",
    cfg={
        "level": {
            "fn": lambda env: self.curriculum_level,  # <~ Curriculum / level
        },
    },
)
```

## Values with multiple dimensions

An item can also return several values per environment, shape (num_envs, N). Give each column a label, and each one is logged on its own.

```python
Monitor(
    self,
    logging_tag="Body",
    cfg={
        "velocity": {
            "fn": lambda env: self.robot.get_vel(),
            "labels": ["x", "y", "z"],  # <~ Body / velocity_x, Body / velocity_y, Body / velocity_z
        },
    }
)
```

If your labels aren't known until the scene is built, like link or joint names, pass a function that returns them instead. A stateful [MDP function](../../api/mdp/index.md) can also set `self.labels` in its `build()` method.

## Using values in rewards

Read the latest value by name. You get the raw tensor:

```python
RewardManager(
    self,
    cfg={
        "height": {
            "weight": -10.0,
            "fn": lambda env: torch.square(self.body_monitor["height"] - 0.3),
        },
    },
)
```

For a value with column labels, use `get()` to return one column:

```python
self.body_monitor.get("velocity", "z")
```

Monitors step right after the contact managers, before terminations and rewards, and update again after a reset, so the values are always current. That means an item can run more than once in a step, so calculate it from the current state rather than adding to a running total. An item can read monitors created before its own, and items listed before it.

## Power Monitor

The power monitor estimates the electrical power your motors would draw. It adds up the heat in each motor's windings and the mechanical work it does. The numbers come from your motor's datasheet or controller config:

```python
from genesis_forge.managers import MotorSpec, PowerMonitor

MY_MOTOR = MotorSpec(
    torque_constant=0.43,   # N·m/A, at the joint (multiply by the gear ratio if needed)
    phase_resistance=0.436, # ohms
)

class MyEnv(ManagedEnvironment):
    def config(self):
        self.actuator_manager = ActuatorManager(self, ...)

        self.power_monitor = PowerMonitor(
            self,
            actuator_manager=self.actuator_manager,
            motors=MY_MOTOR,
            supply_voltage=44.4,
            dof_groups={
                "hip": ".*_Hip",
                "knee": ".*_Knee",
            },
        )
```

It logs the robot's total power, its copper loss and mechanical power, the average power per motor in each joint group, and the battery current.

If your robot uses more than one kind of motor, map joint name patterns to their specs:

```python
motors={
    ".*_Hip": BIG_MOTOR,
    ".*_Knee": SMALL_MOTOR,
},
```

!!! note

    Energy put back into the battery while braking is assumed lost, so the estimate is a little high. It also assumes your motor drivers are perfectly efficient.

### Training with power -- rewards and terminations

There are a couple of rewards built on the power monitor. `electrical_power` penalizes watts directly:

```python
"energy": {
    "weight": -1e-4,
    "fn": rewards.electrical_power(power_monitor=self.power_monitor),
},
```

Moving fast takes more power, so a strong energy penalty works against velocity tracking. Too strong too early and standing still wins. Start small and grow the weight with a curriculum.

To stay inside your battery's limits, `terminations.bus_current_limit` ends an episode when the estimated current goes over `max_current`. And `observations.dof_power` gives your critic each motor's power, as a privileged observation.

## Actuator Monitor

The actuator monitor tracks how hard your joints are working, from the torque actually applied:

```python
from genesis_forge.managers import ActuatorMonitor

self.actuator_monitor = ActuatorMonitor(
    self,
    actuator_manager=self.actuator_manager,
    dof_groups={
        "hip": ".*_Hip",
        "knee": ".*_Knee",
    },
    saturation_threshold=0.95, # <~ at 95% of max_force, a joint counts as saturated
)
```

For each joint group, it logs:

- `torque` - the RMS torque, which drives motor heat and battery use. It's logged as `torque_rms`, or `torque_<group>_rms` for each group.
- `torque_peak` - the largest torque
- `saturation_frac` - how many joints are at their force limit

`rewards.torque_saturation` penalizes joints that sit at their force limit, which is often where simulation and real hardware start to differ.

## DOF groups

Both the power and actuator monitors take `dof_groups`: names for joint name patterns. A pattern can also be a list, and a joint can be in more than one group:

```python
dof_groups={
    "hip": ".*_Hip",
    "legs": [".*_Femur", ".*_Tibia"],
    "front": "[RL][12]_.*",
}
```

Leave it out, and the values are for all the joints together.

## Your own monitor items

The power and actuator monitors also take a `cfg`, to add your own items next to theirs:

```python
self.power_monitor = PowerMonitor(
    self,
    actuator_manager=self.actuator_manager,
    motors=MY_MOTOR,
    cfg={
        "watts_per_kg": {
            "fn": lambda env: self.power_monitor["total_w"] / ROBOT_MASS,
        },
    },
)
```
