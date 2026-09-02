# Wheeled Robot - Navigation

Train the [Freenove 4WD car](https://store.freenove.com/products/fnk0043) to drive to a goal pose — a point, and the direction to be facing once there — while avoiding obstacles, using its ultrasonic range sensor.

This builds on the [wheeled_robot](../wheeled_robot/) example, which covers the car itself. The difference is what the robot is told. There, the policy is handed a velocity to follow, so *how to move* is given and only the execution is learned. Here it is told only **where to end up and which way to face there**, and has to pick its own route and speed, read its range sensor, and stay out of trouble on the way.

## The ultrasonic sensor

Genesis models a range sensor as a raycaster: a bundle of rays in a pattern, each reporting the distance to the first thing it hits. An ultrasonic sensor echoes off whatever is closest in its beam, so the nearest of those rays is the distance it reports.

The settings come from the [HC-SR04 datasheet](https://www.sparkfun.com/datasheets/Sensors/Proximity/HCSR04.pdf):

| Datasheet           | Setting                           |
| ------------------- | --------------------------------- |
| Measuring angle 15° | `fov=(15.0, 15.0)`                |
| Range 2cm – 400cm   | `min_range=0.02`, `max_range=4.0` |
| Resolution 0.3cm    | `resolution=0.003`                |
| Accuracy ~3mm       | `noise=0.003`                     |

Sensors must be added **before the scene is built**, which happens before `config()` is called, so the sensor is created in `__init__` alongside the entities. Two details are worth knowing if you mount a sensor on your own robot:

- **`euler_offset` orients the beam.** Raycast patterns fire along the sensor frame's **+X** axis, but a link's frame is whatever the model gives it. On this robot the ultrasonic board's own **+Z** axis is the one pointing out of the transducers, so the pattern is rotated onto it. If your beam comes out sideways, this is why — turn on `draw_debug=True` to see where the rays actually go.
- **`pos_offset` starts the rays clear of the robot.** Rays hit _everything_, including the robot they are mounted on, and `min_range` does not suppress those self-hits. Without the 3cm offset, every ray would stop on the transducer housings about 8mm out.

`return_points=False` measures distances without building a point cloud, which is about four times cheaper and all a range reading needs.

### Aiming the head above the floor

A 15° cone from a sensor mounted 10cm up is a problem: its lowest rays meet the ground at `0.105 / sin(7.5°) ≈ 0.8m`. Because the reading is the **nearest** echo, the floor then masks every obstacle past 0.8m, and the sensor reports ~0.79m on a completely clear road.

The fix is the one a real robot needs too — aim the head slightly up:

```python
HEAD_TILT = math.radians(7.0)
```

Tilting just under the cone's 7.5° half-angle lifts the lowest ray to about −0.5°, which does not reach the floor until roughly 12m, well past the sensor's range. The beam still sits low enough to catch obstacles.

The tilt is applied as the head actuator's default position, and `build()` holds exactly that pose, so the two cannot drift apart.

## The goal command

`Pose2dCommand` samples a goal pose for each environment and reports how to get there:

```python
self.pose_command = Pose2dCommand(
    self,
    range={"x": (-2.5, 2.5), "y": (-2.5, 2.5), "heading": (-math.pi, math.pi)},
    goal_reached_threshold=0.2,
    heading_reached_threshold=math.radians(30),
    resample_on_reached=True,
    debug_visualizer=True,
)
```

The position and the heading are drawn independently, so the robot cannot satisfy the heading just by arriving — it has to turn to face the right way. This platform can spin on the spot, so it is free to line itself up either on the way in or once it is in position.

Its observation is the goal seen from the **robot's own frame**, in seven numbers: the goal vector (ahead, left), the distance, the cosine/sine of the bearing (which way to drive), and the cosine/sine of the heading error (which way to face on arrival).

```python
"goal_pose": {"fn": self.pose_command.observation},
```

The goal vector on its own would be enough to locate the goal, but it mixes up *how far* with *which way*: at 3m it is a long vector, and a few centimeters out it is a tiny one. That makes the steering signal fade away exactly where steering has to be most precise — right at the goal. Splitting the distance out from the bearing keeps the direction at full strength all the way in.

Reaching a goal earns a new one immediately (`resample_on_reached=True`), so one episode is a string of navigation problems rather than a single drive. With `resample_time_sec` left unset, a goal never expires on a timer — the robot keeps working at it until it arrives or the episode ends.

A goal counts as reached only when the robot is both within `goal_reached_threshold` of the position and within `heading_reached_threshold` of the heading — see [why arrival requires the heading](#why-arrival-requires-the-heading) below. Leave `heading_reached_threshold` unset and arrival is judged on position alone, whichever way the robot ends up facing.

Turn on `debug_visualizer` to see the goal drawn as an arrow in the scene: it starts at the goal position and points the way to face on arrival, turning red once the goal has been reached.

## Where goals can land

Goals are never placed on top of anything else in the scene. Every entity — the obstacles and the robot itself — gets a circle of clear space around it, sized from its own footprint plus the reach threshold, and a goal that lands inside one is drawn again. This is automatic; there is nothing to configure.

Because the robot is one of the things goals are kept clear of, a fresh goal never starts out already reached.

## Rewards

Every reward here pays for *doing* something, never for *being* somewhere. A robot that stops earns exactly zero — which is the whole design, for reasons in the next section.

| Reward | Weight | What it does |
|---|---|---|
| `position_progress` | 1.0 | Pays for closing the distance, at any range — what gets the robot moving at all |
| `heading_progress` | 0.5 | Pays for turning the right way: toward the goal while travelling, toward the goal heading on the approach |
| `reached_goal` | 50.0 | A bonus for arriving: on the goal *and* lined up with it |
| `keep_clear` | -1.0 | Crowding an obstacle, growing from nothing at 0.35m to full on contact |
| `collision` | -10.0 | Hitting an obstacle, which also ends the episode |
| `action_rate` | -0.005 | Discourages twitchy steering |
| `body_acceleration_exp` | -0.02 | Discourages jerky motion |

`body_acceleration_exp` is worth a note: it is `1 - exp(-sensitivity · motion)`, and at the default sensitivity every plausible motion scores near its ceiling of 1. Saturated, it no longer tells smooth apart from jerky — it just taxes moving at all, at a size comparable to the reward for making progress. This example lowers both the sensitivity and the weight so it stays in the range where it actually discriminates, because arriving on a pose needs brisk turning and a flat tax on motion works against that.

`position_progress` measures the speed at which the robot is approaching its goal, so driving away is penalized. `heading_progress` is its mirror for angles, measuring how fast the robot is closing the angle it is being asked to close. Both skip any step where the value jumped for a reason the robot is not responsible for — the step after a reset, and the step where a goal was reached and replaced.

### Why the robot steers at the goal before lining up with it

This robot is a *skid-steer*: it can drive and it can spin on the spot, but it cannot slide sideways. That makes reaching a pose — a position **and** a heading at the same time — much harder than it looks, and it is a well-studied problem. A robot under this constraint [cannot be brought smoothly to a pose by any fixed feedback rule](https://arxiv.org/html/2607.26442) (Brockett's condition), which in practice shows up as shuffling back and forth near the goal. The classical answer is to think in polar terms — distance, the bearing to the goal, and the heading to arrive on — rather than in x/y, which is exactly what the observation above reports.

The reward has to respect the same constraint. `heading_progress` takes a `lines_up_within` distance:

```python
rewards.heading_progress(pose_cmd_manager=self.pose_command, lines_up_within=0.75)
```

Further out than that, it pays for turning *toward the goal* — the way the robot actually has to point to drive there. Closer in, it hands over to the goal heading. Asking for the goal heading the whole way round instead makes the robot line up early and then try to crab sideways into the goal, which it physically cannot do; the visible result is a robot driving parallel to its goal, inching in a few centimeters at a time.

For a robot that *can* travel one way while facing another — a legged or omnidirectional robot — none of this applies, and the default of asking for the goal heading everywhere is the right choice.

### Why there is no "distance to goal" reward

The obvious reward for a navigation task is one that grows as the robot nears the goal (`position_tracking` in this library, and a heading equivalent in `heading_tracking`). Both are deliberately **not** used here, because they break this particular task in a way that is worth understanding.

Rewards are scaled by the timestep, so a reward paid every step is worth its weight multiplied by however many steps it is collected over. In a 30 second episode that is 1500 steps. A proximity reward at weight 1.0 is therefore worth up to 30 over an episode — while arriving at a goal, even at weight 50, is worth 1.0 each time.

Now consider what the robot can do with that. Parking just outside the reach threshold pays the proximity rewards forever. Crossing the threshold pays the arrival bonus *once* and then replaces the goal with a new one several meters away, ending the income. Camping is worth roughly twenty times arriving, so the optimal policy is to drive up to the goal, stop just short of it, and sit there.

That is not hypothetical — it is what an earlier version of this example actually learned:

| iteration | position_tracking | position_progress | heading_tracking | reached_goal |
|---|---|---|---|---|
| 109 | 0.071 | 0.270 | 0.047 | **0.097** |
| 305 | 0.214 | 0.211 | 0.147 | **0.063** |
| 599 | 0.355 | 0.139 | 0.242 | **0.031** |

Total reward rose the whole way, while the number of goals actually reached fell to a third. The policy was optimizing exactly what it was told to.

Reweighting does not fix this; it only moves the parking spot. What fixes it is using rewards that cannot be collected by holding a pose. `position_progress` and `heading_progress` are both *rates* — standing still pays zero by construction, and over a whole goal they can only ever add up to the distance and angle the robot started with. This is [potential-based shaping](https://people.eecs.berkeley.edu/~pabbeel/cs287-fa09/readings/NgHaradaRussell-shaping-ICML1999.pdf), which is provably incapable of changing which policy is optimal, and therefore incapable of inventing a camping strategy.

`position_tracking` and `heading_tracking` are still fine for tasks where the goal is *not* replaced on arrival — their docstrings carry the warning.

### Why arrival requires the heading

`heading_reached_threshold` is set to 30°, so a goal counts as reached only when the robot is both in position and lined up. Without it the goal would be replaced the instant the robot drove into range, and there would be nothing to line up for — the heading would be decorative.

The tolerance is not tight on purpose. Since the robot cannot slide sideways, every degree it shaves off costs extra shuffling to satisfy the distance and the heading together. 30° leaves enough room to arrive in one movement.

This is only safe because nothing pays for standing still. If a proximity reward existed, requiring the heading would let a robot that reached the goal but could not align park on it and collect forever, which is a worse version of the same trap.

## Seeing: the robot has to look around

An earlier version of this example crashed out of roughly 78% of its episodes and never improved on it across 420 iterations — the collision penalty was flat from start to finish. It also barely tried to swerve, which turns out to be the giveaway.

The cause was that the sensor reading carried no direction. `raycaster_distance` defaults to `reduce="min"`, collapsing the sensor's rays to a single nearest echo, so an obstacle dead ahead and one 0.15m off to the side both read 0.80m:

| obstacle 1.0m ahead, offset | reading |
|---|---|
| 0.00m (dead centre) | 0.81m |
| 0.05m | 0.81m |
| 0.10m | 0.80m |
| 0.15m | 0.80m |

All four are on a collision course. Knowing *something* is ahead without knowing which side it is on makes swerving a coin flip, and a wrong guess costs goal progress for nothing — so driving straight through and accepting the crash really was the better bet. The robot was not failing to avoid obstacles; it was correctly declining to try.

There are two ways out of that, and `POLICY_AIMS_HEAD` switches between them so you can try both.

**Fixed head** (the default). The sensor stays pinned forwards, and the robot has to work out direction from how the readings change as it drives and turns — the parallax in its own motion. Fewer actions and a simpler problem, but the information is indirect.

**The policy aims the sensor.** The pan servo becomes an action. With one range sensor, looking around *is* how the robot finds out what is beside it:

```python
self.head_action_manager = PositionActionManager(
    self,
    scale=HEAD_PAN_SCALE,
    use_default_offset=True,
    actuator_manager=self.head_manager,
    actuator_joints=["servo-2"],  # pan only; the tilt stays where build() puts it
)
```

When it is switched on, the head angle joins the observation — a range reading is meaningless without knowing where the sensor was pointing when it was taken. With a fixed head that angle is a constant, so it is left out; a constant teaches nothing and still costs a slot in every stacked frame.

**Aiming was tried, and the fixed head won.** The reason is geometric, and it is worth understanding before switching it back on. The beam is 15° wide, so watching the path ahead and looking off to the side are the *same* 15° — the sensor cannot do both. Measured against the robot's own collision corridor:

| head angle | corridor covered at 1m | at 2m |
| ---------- | ---------------------- | ----- |
| 0° | 88% | 100% |
| 7.5° | 50% | 50% |
| 15° | 6% | 0% |
| 25°+ | **0%** | **0%** |

`HEAD_PAN_SCALE` allows ±46°, so the policy can blind itself completely — and it did, parking the head off to one side and driving into things. Nothing in the reward argues against that: `keep_clear` and the collision penalty are computed from true positions rather than from the sensor, so the head action has no gradient connecting it to any outcome. It drifts wherever initialization and noise take it.

Turning the pan range down far enough to keep the corridor covered (≈7.5°) leaves so little sweep that panning buys nothing, so the fixed head is both simpler and strictly better here. Making aiming work would need a reason for the policy to look where it is going — a wider beam, a second sensor, or a reward tied to what the sensor can actually see.

**The observation carries history**, either way. One reading in one direction cannot place an obstacle; the information is in how readings change over time — from a sweep when the policy aims the head, and from the robot's own motion when it doesn't. `history_len` stacks recent observations so a policy with no memory of its own can put that together. This is the main dial to turn if avoidance is still poor: too short and the pattern does not fit inside the window.

Together these let the robot find things it cannot see head-on. Measured with an obstacle 0.7m ahead and 0.5m to the left:

| pan action | head angle | range |
|---|---|---|
| 0.00 | −0.09 rad | nothing |
| 0.50 | +0.31 rad | nothing |
| 0.75 | +0.54 rad | **0.67m** |

**A note on `keep_clear`.** It measures the true distance to each obstacle rather than reading the sensor, which is deliberate. A penalty computed from the sensor would be a penalty the robot could escape by *pointing the sensor somewhere else* — teaching it to look away from danger rather than avoid it. Rewards are free to use information the observation withholds, and here that difference matters.

## Driving: two actions, not four

The robot has four wheels, but the policy only commands two numbers — one per side:

```python
action_groups=[
    ["TT_Motor-3_axel", "TT_Motor-4_axel"],  # left side
    ["TT_Motor-1_axel", "TT_Motor-2_axel"],  # right side
]
```

The two wheels down each side are bolted to the same chassis, so steering them apart does nothing except scrub rubber. Given four independent commands, a policy has to spend capacity discovering that for itself. Two actions describe how the robot actually moves: same sign to drive, opposite signs to spin on the spot.

There is a catch worth knowing about, because it is invisible until you measure it. The two **front** gearboxes are mounted turned around in the model, so the same command spins a front wheel the opposite way to a rear one. Each wheel therefore carries a mounting sign in its scale:

```python
WHEEL_MOUNTING_SIGN = {
    "TT_Motor-1_axel": -1.0,  # front right
    "TT_Motor-2_axel": +1.0,  # rear right
    "TT_Motor-3_axel": -1.0,  # front left
    "TT_Motor-4_axel": +1.0,  # rear left
}
```

Without it the robot still drives straight and still turns, so nothing looks broken — the front wheels just spend the whole time being dragged backwards. Measured over 1.2 seconds of full throttle, the difference is 0.166m against 0.381m, and in-place turns are two to three times slower. If you group wheels on a new robot, check the signs by driving each pattern and comparing distance covered, not by watching whether it moves.

## Starting orientation

The robot starts each episode facing a random direction:

```python
"rotation": {"fn": reset.set_rotation(z=(0.0, 2 * math.pi))},
```

Without this the goal would always start out somewhere ahead of a forward-facing robot, and the policy could do well without ever really reading the goal vector.

## Driving speed

The wheel scale is set by the hardware: 20 rad/s is about 200 RPM, which is what the TT motors on the real platform turn. Measured on the 6.9cm wheels, speed is linear in that scale with no sign of slip anywhere in the range:

| `VelocityActionManager` scale | wheel rad/s | robot speed |
| ----------------------------- | ----------- | ----------- |
| 5                             | 5.0         | 0.173 m/s   |
| 10                            | 10.0        | 0.346 m/s   |
| 15                            | 15.0        | 0.519 m/s   |
| **20**                        | 20.0        | **0.692 m/s** |
| 30                            | 30.0        | 1.036 m/s   |

That works out at 0.0346 m/s per rad/s throughout — a straight line.

Earlier versions of this example reported that slip capped the robot at ~0.26 m/s around a scale of 20, and that a scale of 30 measured *slower* than 20. Those measurements were real, but the cause was not slip: the two front wheels were being driven backwards against the rear pair (see the mounting signs above), so most of the motor effort went into scrubbing rubber. With the signs corrected the robot is about two and a half times faster.

It is worth knowing that when reading anything that was tuned before the fix, because reaction time scales directly with speed. At 0.69 m/s, in a field where obstacles average about a metre apart, the robot crosses the gap between them in roughly 1.4 seconds.

## Obstacles and collisions

Obstacles are plain boxes that never move, but each environment gets its own layout. A `fixed=True` morph defaults to `batch_fixed_verts=True`, which is what allows a per-environment position to be set — and those positions _are_ reflected in the raycasts, so every environment sees its own course. On every reset each box is dropped somewhere new in a ring around the robot, whose inner radius keeps the spawn point clear.

Each box gets its own color — evenly spaced hues, shuffled, so they are easy to tell apart in videos. It is purely cosmetic: the policy only ever sees a distance, never a color.

For collisions, the robot is always touching the ground, so "is anything touching the robot" is not a useful test. The contact manager tracks **every** link of the robot but filters contacts down to the obstacles:

```python
self.collision_manager = ContactManager(
    self,
    entity=self.robot,
    with_entity=self.obstacles,
)
```

Filtering by the obstacles rather than by link name is what separates a crash from normal driving, and it means the whole robot counts — clipping a box with a wheel mid-turn registers just like driving into one head-on.

Note that the collision termination is _not_ marked `time_out: True`. A crash is a genuine failure, and marking it as a time-out would tell the learning algorithm to bootstrap value past it, as though the episode had merely been cut short.

## Episode length

Goals are sampled anywhere in a ±2.5m box, so the furthest is about 3.5m away diagonally. At the ~0.26 m/s this platform tops out at (see [driving speed](#driving-speed) below), that is roughly 15 seconds of driving for a single goal, so the episode runs for 30 seconds — long enough to reach a couple of goals in a row and actually benefit from `resample_on_reached`.

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

Videos are saved while training and can be viewed in `./logs/wheeled-robot-navigation/videos`.
