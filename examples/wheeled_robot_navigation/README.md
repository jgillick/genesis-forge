# Wheeled Robot - Navigation

Train the [Freenove 4WD car](https://store.freenove.com/products/fnk0043) to drive to a goal pose — a point, and the direction to be facing once there — while avoiding obstacles, using its ultrasonic range sensor.

This builds on the [wheeled_robot](../wheeled_robot/) example, which covers the car itself. There, the policy is handed a velocity to follow, so _how to move_ is given and only the execution is learned. Here it is told only **where to end up and which way to face there**, and has to pick its own route and speed, aim and read its range sensor, and avoid obstacles on the way.

Every reset lays out the obstacles afresh and points the robot in a random direction, so the goal is never reliably straight ahead.

<video autoplay="" muted="" loop="" playsinline="" controls="" src="../../docs/media/cmd_pose2d.mp4" style="max-width:500px"></video>

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
    entity_manager=self.robot_manager,
    debug_visualizer=True,
)
```

Position and heading are drawn independently, so the robot cannot satisfy the heading just by arriving — it has to turn to face the right way. Reaching a goal earns a new one immediately, so one episode is a string of navigation problems rather than a single drive.

The observation is the goal seen from the **robot's own frame**, in seven numbers: the goal vector (ahead, left), the distance, the cosine/sine of the bearing (which way to drive), and the cosine/sine of the heading error (which way to face on arrival):

```python
ObservationManager(
    self,
    cfg={
        "goal_pose": {"fn": self.pose_command.observation},
        ...
    }
)
```

### Lining up late, not early

Like most wheeled robots, this one can drive and spin on the spot, but it cannot move sideways. Holding the goal heading on the way in would mean crabbing sideways into the goal — a well-known hard case ([Brockett's condition](https://arxiv.org/abs/2607.26442)). The `lines_up_within` argument of the `heading_progress` reward pays for turning _toward_ the goal when far away, and on the final approach pays for the correct heading.

```python
rewards.heading_progress(
    pose_cmd_manager=self.pose_command,
    lines_up_within=0.75,
)
```

In practice, this means the robot moves directly to the target point, and when it get's close, adjusts it's body to also match the heading.

## Seeing: an aimed ultrasonic sensor

The robot's only view of the world is an HC-SR04 ultrasonic sensor, modeled as a raycaster over a 15° cone that reports the single closest hit. The sensor rides on a servo, that the robot controls to pan the sensor left and right looking for obstacles.

## Rewards

Every reward here pays for _doing_ something, never for _being_ somewhere. A robot that stops moving earns exactly zero.

| Reward                  | Weight | What it does                                                                    |
| ----------------------- | ------ | ------------------------------------------------------------------------------- |
| `position_progress`     | 1.0    | Pays for closing the distance, at any range — what gets the robot moving at all |
| `heading_progress`      | 0.5    | Pays for turning the right way                                                  |
| `reached_goal`          | 50.0   | A bonus for arriving: on the goal _and_ lined up with it                        |
| `keep_clear`            | -2.0   | Penalizes for crowding an obstacle                                              |
| `collision`             | -50.0  | Hitting an obstacle, which also ends the episode                                |
| `action_rate`           | -0.005 | Discourages twitchy steering (wheels only, see above)                           |
| `body_acceleration_exp` | -0.02  | Discourages jerky motion                                                        |

Paying for movement rather than position matters here because the goal is replaced the moment it is reached. A reward for merely _being_ near the goal would pay out every step, worth far more over an episode than the one-time arrival bonus, so the robot would learn to park just outside the threshold and collect it forever. Rates can't be farmed that way — this is [potential-based shaping](https://ai.stanford.edu/~ang/papers/shaping-icml99.pdf).

## The training algorithm: a recurrent policy

Training uses `RNNModel` (a GRU, 256-unit hidden state, `[256, 128]` MLP head) rather than plain feedforward networks:

```python
"actor": {
    "class_name": "RNNModel",
    "rnn_type": "gru",
    "rnn_hidden_dim": 256,
    "hidden_dims": [256, 128],
    ...
},
```

Avoiding obstacles depends on integrating information over time — a single ultrasonic reading is one number in one direction, and making sense of it takes watching how it changes as the robot drives, turns, and sweeps its head. A recurrent hidden state carries a running summary across the whole episode, which suits a signal whose relevant timescale isn't known in advance. PPO (via `rsl-rl`) trains through this the normal way, backpropagating through `num_steps_per_env` steps of hidden state per update.

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
