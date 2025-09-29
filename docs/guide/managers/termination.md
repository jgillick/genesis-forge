# Termination Manager

The Termination Manager handles episode termination conditions in your RL environment. It determines when episodes should end, distinguishes between timeouts and failures, and provides automatic logging of termination reasons.

You can see a full example using the reward manager in [examples/rough_terrain](https://github.com/jgillick/genesis-forge/tree/main/examples/rough_terrain).

## Basic Usage

```python
from genesis_forge.managers import TerminationManager
from genesis_forge.mdp import terminations

class MyEnv(ManagedEnvironment):
    def config(self):
        self.termination_manager = TerminationManager(
            self,
            logging_enabled=True,
            term_cfg={
                "timeout": {
                    "fn": terminations.timeout, # Ends the episode when it reaches the maximum steps (env.max_episode_length)
                    "time_out": True,  # This is a timeout, not failure
                },
                "fall_over": {
                    "fn": terminations.bad_orientation, # Terminate if the robot is falling over
                    "params": {"limit_angle": 0.5},  # 28 degrees
                },
            },
        )
```

## Termination Configuration

Each termination condition has:

- **fn**: Function that returns boolean tensor indicating termination
- **params**: Optional parameters for the function
- **time_out**: Whether this is a timeout (`True`) or failure (`False`, default)

```python
TerminationManager(
    self,
    term_cfg={
        "max_episode_length": {
            "fn": terminations.timeout,
            "time_out": True,  # Normal episode end
        },
        "robot_fell": {
            "fn": terminations.bad_orientation,
            "params": {"limit_angle": 0.3},
        },
        "out_of_bounds": {
            "fn": lambda env: torch.norm(env.robot.get_pos()[:, :2], dim=1) > 5.0,
        },
    },
)
```

## Built-in Termination Functions

Genesis Forge provides common termination conditions in [`genesis_forge.mdp.terminations`](../../api/mdp/terminations):

```python
term_cfg={
    "timeout": {
        "fn": terminations.base_height_below_minimum,
        "params": {
            "minimum_height": 0.05,
        }
    },
}
```

## Custom Termination Functions

A custom termination function takes in the environment as the first parameter, as well as any other parameter which will be defined in the `params` dict at the TerminationManager. The returned value should be a tensor (shape: `(num_envs,)`) with a `bool` value for each environment.

```python
def velocity_limit(env, max_velocity=10.0):
    """Terminate if robot moves too fast."""
    velocity = torch.norm(env.robot.get_vel(), dim=1)
    return velocity > max_velocity

TerminationManager(
    self,
    term_cfg={
        "too_fast": {
            "fn": velocity_limit,
            "params": {"max_velocity": 8.0},
        },
    },
)
```

## Timeout vs Termination

Understanding the distinction is important for RL algorithms:

- **Timeout** (`time_out=True`): Natural episode end, not a failure

  - Episode reached max length
  - Task successfully completed
  - Training scenario ended

- **Termination** (`time_out=False`): Episode ended due to failure
  - Robot fell over
  - Violated safety constraints
  - Task failed

## Curriculum-Based Termination

Adjust termination criteria during training:

```python
class MyEnv(ManagedEnvironment):
    def config(self):
        self.termination_manager = TerminationManager(self, term_cfg={
            "timeout": {
                "fn": terminations.timeout,
                "time_out": True,
            },
            "bad_orientation": {
                "fn": terminations.bad_orientation,
                "params": {"angle_limit": 25},
            },
            "too_low": {
                "fn": terminations.base_height_below_minimum,
                "params": {
                    "minimum_height": 0.05
                }
            }
        })

    def step(self):
        self.update_curriculum()
        return super().step(actions)

    def update_curriculum(self):
        """Make termination criteria stricter over time."""
        if self.step_count > 200:
            # Mid: moderate
            angle_limit = 20
            height_threshold = 0.10
        else:
            # Late: strict
            angle_limit = 17
            height_threshold = 0.15

        # Update termination parameters
        self.termination_manager.term_cfg["bad_orientation"].params["angle_limit"] = angle_limit
        self.termination_manager.term_cfg["too_low"].params["minimum_height"] = height_threshold
```

## Logging and Analysis

By default, individual termination averages are logged to the `episode` item in the extras/infos dict. For many RL frameworks, like rsl_rl and skrl, items there will automatically be logged to tensorboard, or simular system. Terminations will be placed under the "Terminations" section.

```{figure} _images/termination_tensorboard.png
:alt: tensor board
Example tensorboard termination logging
```

To disable logging, set `logging_enabled` to `False`. To change the extras dict key that termination items are logged to, set the `extras_logging_key` param on the [environment](../../api/environments/genesis.md).
