# Wheeled Robot - Navigation

Train the [Freenove 4WD car](https://store.freenove.com/products/fnk0043) to drive to a goal pose — a point, and the direction to be facing once there — while avoiding obstacles, using its ultrasonic range sensor.

This builds on the [wheeled_robot](../wheeled_robot/) example, which covers the car itself. There, the policy is handed a velocity to follow, so _how to move_ is given and only the execution is learned. Here it is told only **where to end up and which way to face there**, and has to pick its own route and speed, read its range sensor, and stay out of trouble on the way.

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

Position and heading are drawn independently, so the robot cannot satisfy the heading just by arriving — it has to turn to face the right way. Reaching a goal earns a new one immediately (`resample_on_reached=True`), so one episode is a string of navigation problems rather than a single drive. Goals are also kept clear of every obstacle and of the robot itself, automatically, so a fresh goal never starts out already reached.

The observation is the goal seen from the **robot's own frame**, in seven numbers: the goal vector (ahead, left), the distance, the cosine/sine of the bearing (which way to drive), and the cosine/sine of the heading error (which way to face on arrival):

```python
"goal_pose": {"fn": self.pose_command.observation},
```

Splitting distance out from bearing keeps the steering signal at full strength all the way to the goal, instead of shrinking to nothing as the goal vector itself does.

### Why a polar controller, not x/y

This robot is a _skid-steer_: it can drive and spin on the spot, but not slide sideways. That makes reaching a pose — a position **and** a heading together — a well-studied hard case: no fixed feedback rule can bring a robot like this smoothly to a pose ([Brockett's condition](https://arxiv.org/html/2607.26442)), which in practice shows up as shuffling near the goal. The classical fix is to think in polar terms — distance, bearing, heading error — which is exactly what the observation above reports, and the reward has to respect the same constraint:

```python
rewards.heading_progress(pose_cmd_manager=self.pose_command, lines_up_within=0.75)
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

## Randomized obstacles

Obstacles are plain boxes that never move, but each environment gets its own layout: on every reset, each box is dropped somewhere new in a ring around the robot (`radius_range=(0.5, 3.0)`), whose inner radius keeps the spawn point clear. A `fixed=True` morph defaults to `batch_fixed_verts=True`, which is what allows a per-environment position — and those positions are reflected in the raycasts, so every environment sees its own course. Each box gets its own color, purely cosmetic, so runs are easy to tell apart in videos.

For collisions, the robot is always touching the ground, so "is anything touching the robot" isn't useful. The contact manager tracks every link but filters contacts down to the obstacles, so a wheel clipping a box mid-turn registers just like a head-on hit:

```python
self.collision_manager = ContactManager(self, entity=self.robot, with_entity=self.obstacles)
```

## The ultrasonic sensor

Genesis models a range sensor as a raycaster: a bundle of rays in a pattern, each reporting the distance to the first thing it hits. An ultrasonic sensor echoes off whatever is closest in its beam, so the nearest ray is the distance it reports. Settings come from the [HC-SR04 datasheet](https://www.sparkfun.com/datasheets/Sensors/Proximity/HCSR04.pdf) — a 15° cone, 2cm–400cm range, 0.3cm resolution, ~3mm accuracy.

Sensors must be added before the scene is built, so the sensor is created in `__init__` alongside the entities. Two details worth knowing if you mount a sensor on your own robot:

- **`euler_offset` orients the beam.** Ray patterns fire along the sensor frame's **+X** axis, but a link's frame is whatever the model gives it — here the board's own **+Z** is the one pointing out of the transducers, so the pattern is rotated onto it. `draw_debug=True` shows where the rays actually go.
- **`pos_offset` starts the rays clear of the robot.** Rays hit _everything_, including the robot they're mounted on, and `min_range` does not suppress those self-hits — without a small offset, every ray would stop on the transducer housing a few millimeters out.

`return_points=False` measures distances without building a point cloud, about four times cheaper.

### Why the head sweeps a little

A single beam distance carries no direction: an obstacle dead ahead and one significantly off to the side can report nearly the same reading with `reduce="min"`, since it just collapses the rays to the nearest echo. Without a way to tell those apart, swerving is a coin flip, and the safer bet ends up being to drive straight and accept the occasional crash.

The head's pan servo is given to the policy as a small action, letting it sweep the beam side to side:

```python
self.head_action_manager = PositionActionManager(
    self,
    scale=0.5,
    use_default_offset=True,
    actuator_manager=self.head_manager,
    actuator_joints=["servo-2"],  # pan only; tilt stays where build() puts it
    clip={"servo-2": (-math.radians(7.5), math.radians(7.5))},
)
```

The pan range is deliberately narrow. The beam is 15° wide, so watching straight ahead and looking off to the side are the _same_ 15° — the sensor can't do both at once. Measured against the robot's own collision corridor:

| head angle | corridor covered at 1m | at 2m |
| ---------- | ---------------------- | ----- |
| 0°         | 88%                    | 100%  |
| 7.5°       | 50%                    | 50%   |
| 15°+       | ~0%                    | 0%    |

Capping the pan at ±7.5° keeps the corridor mostly covered no matter which way the head is pointed, while still letting the policy nudge the beam enough to pick up parallax from its own sweeping — nothing in the reward argues for looking anywhere useful on its own (`keep_clear` and the collision penalty are computed from true positions, not the sensor), so an uncapped range lets the policy blind itself by parking the head off to one side.

Since a reading is meaningless without knowing where the sensor was pointing when it was taken, the head angle joins the observation alongside the sensor reading. The observation also carries a couple of frames of history (`history_len=2`) so a memoryless policy would have something to compare against — though here that job is mostly done by the recurrent policy itself, below.

## Driving: two actions, not four

The robot has four wheels, but the policy only commands two numbers — one per side:

```python
action_groups=[
    ["TT_Motor-3_axel", "TT_Motor-4_axel"],  # left side
    ["TT_Motor-1_axel", "TT_Motor-2_axel"],  # right side
]
```

The two wheels down each side are bolted to the same chassis, so steering them apart does nothing but scrub rubber — two actions is how the robot actually moves: same sign to drive, opposite signs to spin in place.

The two **front** gearboxes are mounted turned around in the model, so the same command spins a front wheel the opposite way to a rear one. Each wheel's scale carries a mounting sign to correct for it:

```python
"TT_Motor-1_axel": -MAX_WHEEL_VELOCITY,  # front right
"TT_Motor-2_axel": +MAX_WHEEL_VELOCITY,  # rear right
"TT_Motor-3_axel": -MAX_WHEEL_VELOCITY,  # front left
"TT_Motor-4_axel": +MAX_WHEEL_VELOCITY,  # rear left
```

Get this wrong and the robot still drives and still turns — nothing looks broken — it's just noticeably slower, because the front wheels spend the whole time being dragged backwards against the rear pair. If you group wheels on a new robot, check the signs by driving each pattern and comparing distance covered, not by watching whether it moves.

`MAX_WHEEL_VELOCITY` (20 rad/s, ~200 RPM) is set by the hardware — the real TT motors on the platform top out around there. The robot also starts each episode facing a random direction (`reset.set_rotation(z=(0.0, 2 * math.pi))`), so it can't do well just by driving toward whatever's in front of it.

## The training algorithm: a recurrent policy

The actor and critic are both `RNNModel`s (a GRU, 256-unit hidden state, `[256, 128]` MLP head) rather than plain feedforward networks:

```python
"actor": {
    "class_name": "RNNModel",
    "rnn_type": "gru",
    "rnn_hidden_dim": 256,
    "hidden_dims": [256, 128],
    ...
},
```

Avoidance here depends on integrating information over time — a single ultrasonic reading is ambiguous, and disambiguating it takes watching how the reading changes as the robot drives, turns, and sweeps its head. A fixed-length stack of recent observations (`history_len`) only covers a short window; a recurrent hidden state instead carries a running summary across the whole episode, which is a better fit for a signal whose relevant timescale isn't known in advance. PPO (via `rsl-rl`) trains through this the normal way, backpropagating through `num_steps_per_env` steps of hidden state per update.

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
