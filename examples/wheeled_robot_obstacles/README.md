# Wheeled Robot - Obstacle avoidance with an ultrasonic sensor

Train the [Freenove 4WD car](https://store.freenove.com/products/fnk0043) to follow velocity commands through a field of obstacles, using a simulated ultrasonic range sensor to see what is in front of it.

This builds on the [wheeled_robot](../wheeled_robot/) example — same differential-steering platform and same velocity command task — and adds a sensor, obstacles, and a reason to care about them. It uses the `Freenove4WD_w_sensor.xml` model, which includes the pan/tilt head unit carrying the real robot's HC-SR04 ultrasonic sensor and camera.

## The ultrasonic sensor

Genesis models a range sensor as a raycaster: a bundle of rays in a pattern, each reporting the distance to the first thing it hits. The nearest of those rays is the distance the sensor reports — an ultrasonic sensor echoes off whatever is closest in its beam.

The settings come from the [HC-SR04 datasheet](https://www.sparkfun.com/datasheets/Sensors/Proximity/HCSR04.pdf):

| Datasheet           | Setting                           |
| ------------------- | --------------------------------- |
| Measuring angle 15° | `fov=(15.0, 15.0)`                |
| Range 2cm – 400cm   | `min_range=0.02`, `max_range=4.0` |
| Resolution 0.3cm    | `resolution=0.003`                |
| Accuracy ~3mm       | `noise=0.003`                     |

Sensors must be added **before the scene is built**, which happens before `config()` is called, so the sensor is created in `__init__` alongside the entities:

```python
ultrasonic_link = self.robot.get_link("Ultrasonic_HC-SR04_PCB")
self.ultrasonic = self.scene.add_sensor(
    gs.sensors.Raycaster(
        pattern=gs.sensors.SphericalPattern(fov=(15.0, 15.0), n_points=(5, 5)),
        entity_idx=self.robot.idx,
        link_idx_local=ultrasonic_link.idx_local,
        euler_offset=(0.0, -90.0, 0.0),
        pos_offset=(0.0, 0.0, 0.03),
        min_range=0.02,
        max_range=4.0,
        return_points=False,
        noise=0.003,
        resolution=0.003,
    )
)
```

A few details worth knowing if you mount a sensor on your own robot:

- **`euler_offset` orients the beam.** Raycast patterns fire along the sensor frame's **+X** axis, but a link's frame is whatever the model gives it. On this robot the ultrasonic board's own **+Z** axis is the one pointing out of the transducers, so the pattern is rotated onto it. If your beam comes out sideways, this is why — turn on `draw_debug=True` to see where the rays actually go.
- **`pos_offset` starts the rays clear of the robot.** Rays hit _everything_, including the robot they are mounted on, and `min_range` does not suppress those self-hits. Without the 3cm offset, every ray would stop on the `Ultrasonic_Can-1`/`Ultrasonic_Can-2` transducer housings about 8mm out.
- **`return_points=False`** measures distances without building a point cloud, which is about four times cheaper — all you need for a range reading.
- **`noise`, `resolution`** make the reading imperfect and quantized, like the real thing.

### Aiming the head above the floor

A 15° cone from a sensor mounted 10cm up is a problem: its lowest rays meet the ground at `0.105 / sin(7.5°) ≈ 0.8m`. Because the reading is the **nearest** echo, the floor then masks every obstacle past 0.8m, and the sensor reports ~0.79m on a completely clear road.

The fix is the one a real robot needs too — aim the head slightly up:

```python
HEAD_TILT = math.radians(7.0)
```

Tilting just under the cone's 7.5° half-angle lifts the lowest ray to about −0.5°, which does not reach the floor until roughly 12m, well past the sensor's range. The beam still sits low enough to catch obstacles: measured against a 20cm box, the reading tracks the true distance to within ~2cm from 0.3m all the way out to 3.3m, and a clear road now reads the full 4.0m.

The tilt is applied as the head actuator's default position, and the environment's `build()` holds exactly that pose, so the two can't drift apart:

```python
default_pos={
    "servo-2": 0.0,          # facing straight ahead
    "servo_horn-1": HEAD_TILT,  # aimed slightly up, to clear the floor
},
```

The reading becomes an observation through `raycaster_distance`, which reduces the cone to its nearest ray and scales it to `[0, 1]`, where `1.0` means nothing is within range:

```python
"ultrasonic": {
    "fn": observations.raycaster_distance(sensor=self.ultrasonic, normalize=True),
},
```

## Holding the head still

The head has two servos (`servo-2` for left/right, `servo_horn-1` for up/down). This example holds the sensor at a fixed pose by giving those joints an actuator manager with PD gains and **no action manager**, so the policy never drives them:

```python
self.head_manager = ActuatorManager(
    self,
    joint_names=["servo-2", "servo_horn-1"],
    default_pos={"servo-2": 0.0, "servo_horn-1": HEAD_TILT},
    kp=8.0,
    kv=0.4,
)
```

The actuator manager sets the gains and the starting pose but never commands a target, so the environment's `build()` sets one once — it persists for the whole run, because nothing else writes to these joints:

```python
self.head_manager.control_dofs_position(self.head_manager.default_dofs_pos)
```

Holding `default_dofs_pos` rather than a separately written angle means the pose the servos are reset to and the pose they are driven toward can never disagree.

Because the sensor is mounted on the head rather than the chassis, the beam follows the head automatically. Adding a `PositionActionManager` over `servo-2` would let a policy learn to sweep the sensor and scan for obstacles.

## Obstacles

Obstacles are plain boxes that never move, but each environment gets its own layout. A `fixed=True` morph defaults to `batch_fixed_verts=True`, which is what allows a per-environment position to be set — and those per-environment positions _are_ reflected in the raycasts, so every environment sees its own course:

```python
self.obstacles = [
    self.scene.add_entity(
        gs.morphs.Box(size=(0.15, 0.15, 0.2), fixed=True),
        surface=gs.surfaces.Rough(color=color),
    )
    for color in _obstacle_colors(6)
]
```

Each box gets its own color — evenly spaced hues, shuffled, so they are easy to tell apart in the viewer and in training videos. It is purely cosmetic: the policy only ever sees a distance, never a color.

On every reset each box is dropped somewhere new in a ring around the robot. The ring's inner radius keeps the robot's spawn point clear, so an episode never starts inside an obstacle:

```python
"position": {
    "fn": reset.randomize_annulus_position(radius_range=(0.6, 2.5), z=0.1),
},
```

## Driving speed

The wheels follow whatever speed they are commanded, but slip caps how fast that actually moves the robot. Measured on this platform, full throttle gives:

| `VelocityActionManager` scale | wheel rad/s | robot speed   |
| ----------------------------- | ----------- | ------------- |
| 5                             | 5.0         | 0.055 m/s     |
| 10                            | 10.0        | 0.119 m/s     |
| **20**                        | 20.0        | **0.257 m/s** |
| 30                            | 30.0        | 0.199 m/s     |
| 40                            | 40.0        | 0.260 m/s     |

Speed saturates around a scale of 20 — past that the wheels just spin, and 30 measured _slower_ than 20. So the example uses `scale=20`, and commands velocities up to 0.2 m/s, which leaves the robot a little headroom to actually track the fastest command it can be given (it reaches 0.23 m/s).

This matters for the episode length too. At ~0.23 m/s a 15 second episode covers about 3.5m, which is enough to drive through a field of obstacles scattered out to 2.5m. If you slow the robot down, lengthen the episode to match, or it will never reach anything.

## Detecting collisions

The robot is always touching the ground, so "is anything touching the robot" is not a useful collision test. Instead the contact manager tracks **every** link of the robot but filters contacts down to the obstacles:

```python
self.collision_manager = ContactManager(
    self,
    entity=self.robot,
    with_entity=self.obstacles,
)
```

Filtering by the obstacles rather than by link name is what separates a crash from normal driving, and it means the whole robot counts — clipping a box with a wheel mid-turn registers just like driving into one head-on. A collision both ends the episode and costs a one-time penalty.

Note that the collision termination is _not_ marked `time_out: True`. A crash is a genuine failure, and marking it as a time-out would tell the learning algorithm to bootstrap value past it, as though the episode had merely been cut short.

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

Videos are saved while training and can be viewed in `./logs/wheeled-robot-obstacles/videos`.

## Where to go next

- [wheeled_robot_goal_nav](../wheeled_robot_goal_nav/) — instead of following velocity commands, the robot is told where to end up and has to navigate there itself.
- **Active scanning** — add a `PositionActionManager` over `servo-2` and let the policy aim the sensor while it drives.
- **Depth camera** — the same head carries a `camera` link, and `gs.sensors.DepthCamera` casts rays in a camera pattern instead of a cone. It needs no renderer, and the same `raycaster_distance` function turns it into an observation with `reduce="flatten"`:

  ```python
  self.depth = self.scene.add_sensor(
      gs.sensors.DepthCamera(
          pattern=gs.sensors.DepthCameraPattern(res=(16, 12), fov_horizontal=60.0),
          entity_idx=self.robot.idx,
          link_idx_local=self.robot.get_link("camera").idx_local,
          max_range=3.0,
          return_points=False,
      )
  )
  # observation: 192 values, one per ray
  "depth": {
      "fn": observations.raycaster_distance(
          sensor=self.depth, reduce="flatten", normalize=True
      ),
  },
  ```

  A rendered RGB camera is a bigger step: it needs the batch renderer and a policy that can consume images.
