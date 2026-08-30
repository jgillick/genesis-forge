# Wheeled Robot - Navigating to a goal

Train the [Freenove 4WD car](https://store.freenove.com/products/fnk0043) to drive to a goal position while avoiding obstacles, using its ultrasonic range sensor.

This builds on [wheeled_robot_obstacles](../wheeled_robot_obstacles/), which shares the sensor, obstacle, and collision setup — read that one first for how the ultrasonic sensor is mounted and configured.

The difference is what the robot is told. In the previous examples the policy is handed a velocity to follow, so *how to move* is given and only the execution is learned. Here the policy is told only **where to end up**, and has to pick its own heading and speed to get there.

## The goal command

`PositionCommandManager` samples a goal position for each environment and reports how to get there:

```python
self.position_command = PositionCommandManager(
    self,
    range={"x": (-2.5, 2.5), "y": (-2.5, 2.5)},
    goal_reached_threshold=0.15,
    resample_on_reached=True,
    debug_visualizer=True,
)
```

Its observation is the vector from the robot to the goal, rotated into the **robot's own frame**, so a single two-element observation carries both the direction to turn and how far there is left to go:

```python
"goal_vec": {"fn": self.position_command.observation},
```

Reaching a goal earns a new one immediately (`resample_on_reached=True`), so one episode is a string of navigation problems rather than a single drive. With `resample_time_sec` left unset, a goal never expires on a timer — the robot keeps working at it until it arrives or the episode ends.

Turn on `debug_visualizer` to see the goal drawn as a sphere in the scene, which turns red once it has been reached.

## Rewards

Goal-reaching needs both a reward for *being there* and a reward for *getting closer*, because the first one alone is nearly flat when the robot starts far away:

| Reward | Weight | What it does |
|---|---|---|
| `position_tracking` | 1.0 | Strongest right at the goal — makes the robot settle on it instead of driving past |
| `position_progress` | 1.0 | Pays for closing the distance, at any distance — what gets the robot moving at all |
| `reached_goal` | 10.0 | A bonus for arriving |
| `collision` | -10.0 | Hitting an obstacle, which also ends the episode |
| `action_rate` | -0.005 | Discourages twitchy steering |
| `body_acceleration_exp` | -0.1 | Discourages jerky motion |

`position_tracking` derives its own sensitivity from the goal range, the same way the velocity-tracking rewards do, so widening the range in a curriculum automatically loosens the reward instead of silently making it unreachable.

`position_progress` measures the speed at which the robot is approaching its goal, so driving away is penalized. It skips any step where the distance jumped for a reason the robot is not responsible for — the step after a reset, and the step where a goal was reached and replaced.

## Starting orientation

The robot starts each episode facing a random direction:

```python
"rotation": {"fn": reset.set_rotation(z=(0.0, 2 * math.pi))},
```

Without this the goal would always start out somewhere ahead of a forward-facing robot, and the policy could do well without ever really reading the goal vector.

## Episode length

Goals are sampled anywhere in a ±2.5m box, so the furthest is about 3.5m away diagonally. At the ~0.23 m/s this platform tops out at (see the [speed table](../wheeled_robot_obstacles/README.md#driving-speed)), that is roughly 15 seconds of driving for a single goal, so the episode runs for 30 seconds — long enough to reach a couple of goals in a row and actually benefit from `resample_on_reached`.

If you widen `GOAL_RANGE` or slow the robot down, raise `max_episode_length_s` to match. A goal the robot cannot physically reach before the episode ends is pure noise in the reward signal.

See [`environment.py`](./environment.py) for the full configuration.

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

Videos are saved while training and can be viewed in `./logs/wheeled-robot-goal-nav/videos`.

## Known limitations

Goals and obstacles are sampled independently, so a goal can land right next to — or underneath — an obstacle, which makes some episodes unwinnable. Rejecting goals that fall within a short distance of any obstacle position on reset is a good exercise, and would sharpen the training signal.
