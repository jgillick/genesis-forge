# Wheeled Robot - Navigation

Train the [Freenove 4WD car](https://store.freenove.com/products/fnk0043) to drive to a goal pose — a point, and the direction to be facing once there — while avoiding obstacles, using its ultrasonic range sensor.

This builds on the [wheeled_robot](../wheeled_robot/) example, which covers the car itself. There, the policy is handed a velocity to follow, so _how to move_ is given and only the execution is learned. Here it is told only **where to end up and which way to face there**, and has to pick its own route and speed, read its range sensor, and avoid obstacles on the way.

See [`environment.py`](./environment.py) for the full configuration.

## The goal: a 2D pose

`Pose2dCommand` samples a goal pose for each environment and reports how to get there:

```python
self.pose_command = Pose2dCommand(
    self,
    range={
        "x": (-2.5, 2.5),
        "y": (-2.5, 2.5),
        "heading": (-math.pi, math.pi),
    },
    goal_reached_threshold=0.2,
    heading_reached_threshold=math.radians(30),
    resample_on_reached=True,
    debug_visualizer=True,
)
```

Position and heading are drawn independently, so the robot cannot satisfy the heading just by arriving — it has to turn to face the right way. Reaching a goal earns a new one immediately (`resample_on_reached=True`), so one episode is a string of navigation problems rather than a single drive.

The observation is the goal seen from the **robot's own frame**, in seven numbers: the goal vector (ahead, left), the distance, the cosine/sine of the bearing (which way to drive), and the cosine/sine of the heading error (which way to face on arrival):

```python
"goal_pose": {"fn": self.pose_command.observation},
```

### Why a polar controller, not x/y

This robot is a _skid-steer_: it can drive and spin on the spot, but not slide sideways. That makes reaching a pose — a position **and** a heading together — a well-studied hard case: no fixed feedback rule can bring a robot like this smoothly to a pose ([Brockett's condition](https://arxiv.org/html/2607.26442)), which in practice shows up as shuffling near the goal. The classical fix is to think in polar terms — distance, bearing, heading error — which is exactly what the observation above reports, and the reward has to respect the same constraint:

```python
rewards.heading_progress(
    pose_cmd_manager=self.pose_command,
    lines_up_within=0.75
)
```

Further than `lines_up_within` from the goal, this pays for turning _toward the goal_ — the way the robot actually has to point to drive there. Closer in, it hands over to the goal heading. Asking for the goal heading the whole way round instead makes the robot line up early and then try to crab sideways into the goal, which it physically cannot do. A robot that _can_ travel one way while facing another — legged or omnidirectional — doesn't need this split.

Arrival also requires the heading (`heading_reached_threshold`), not just position: without it the goal would be replaced the instant the robot drove into range, and there would be nothing to line up for. The 30° tolerance isn't tight, on purpose — every degree shaved off costs extra shuffling to satisfy distance and heading together.

## Rewards

Every reward here pays for _doing_ something, never for _being_ somewhere. A robot that stops earns exactly zero.

| Reward                  | Weight | What it does                                                                    |
| ----------------------- | ------ | ------------------------------------------------------------------------------- |
| `position_progress`     | 1.0    | Pays for closing the distance, at any range — what gets the robot moving at all |
| `heading_progress`      | 0.5    | Pays for turning the right way, per the polar controller above                  |
| `reached_goal`          | 50.0   | A bonus for arriving: on the goal _and_ lined up with it                        |
| `keep_clear`            | -2.0   | Crowding an obstacle, growing from nothing at 0.3m to full on contact           |
| `collision`             | -50.0  | Hitting an obstacle, which also ends the episode                                |
| `action_rate`           | -0.005 | Discourages twitchy steering                                                    |
| `body_acceleration_exp` | -0.02  | Discourages jerky motion                                                        |

`position_progress` and `heading_progress` are both _rates_ — closing speed on distance and on heading error — not proximity. That distinction matters here: the goal is replaced the moment it's reached, so a reward that pays for merely _being_ near the goal (`position_tracking` / `heading_tracking` in this library) lets a robot park just outside the reach threshold and collect it forever, which is worth far more over an episode than the one-time arrival bonus. Progress rewards can't be farmed that way — standing still pays zero by construction, and over a whole goal they only ever add up to the distance the robot started with. This is [potential-based shaping](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf), which is provably incapable of changing the optimal policy, so it can't invent a camping strategy either. `position_tracking`/`heading_tracking` are still the right call for tasks where the goal isn't replaced on arrival.

`keep_clear` deliberately reads the true distance to each obstacle rather than the sensor reading. A penalty computed from the sensor would be one the robot could dodge by _pointing the sensor somewhere else_ — teaching it to look away from danger instead of avoiding it.

Note the collision termination below is _not_ marked `time_out: True` — a crash is a genuine failure, and marking it as a time-out would tell the learning algorithm to bootstrap value past it as though the episode had merely been cut short.

## The ultrasonic sensor

The model uses an HC-SR04 ultrasonic sensor to detect obstacles. This is modeled as a raycaster sensor that returns a single value of the closest object detected. Sensors must be added before the scene is built, so the sensor is created in `__init__` alongside the entities.

## The training algorithm: a recurrent policy

This training algorithm uses `RNNModel` (a GRU, 256-unit hidden state, `[256, 128]` MLP head) rather than plain feedforward networks:

```python
"actor": {
    "class_name": "RNNModel",
    "rnn_type": "gru",
    "rnn_hidden_dim": 256,
    "hidden_dims": [256, 128],
    ...
},
```

Avoiding obstacles depends on integrating information over time — a single ultrasonic reading is ambiguous, and disambiguating it takes watching how the reading changes as the robot drives, turns, and sweeps its head. A recurrent hidden state carries a running summary across the whole episode, which is a better fit for a signal whose relevant timescale isn't known in advance. PPO (via `rsl-rl`) trains through this the normal way, backpropagating through `num_steps_per_env` steps of hidden state per update.

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

Videos are saved while training and can be viewed in `./logs/wheeled-robot-navigation/videos`.
