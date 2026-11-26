# Event Manager

The Event Manager runs functions in the middle of an episode, like randomly pushing your robot so it learns to recover.

You can see a full example in [examples/domain_randomization](https://github.com/jgillick/genesis-forge/tree/main/examples/domain_randomization).

## Basic Usage

```python
from genesis_forge import ManagedEnvironment
from genesis_forge.managers import EntityManager, EventManager
from genesis_forge.mdp import events

class MyEnv(ManagedEnvironment):
    # ...

    def config(self):
        self.robot_manager = EntityManager(self, entity=self.robot)

        self.event_manager = EventManager(
            self,
            cfg={
                # Instantly change the robot's velocity every 5-10 seconds
                "kick": {
                    "fn": events.push_velocity(
                        entity_manager=self.robot_manager,
                        linear={"x": (-0.5, 0.5), "y": (-0.5, 0.5)},  # m/s
                    ),
                    "interval_range_s": (5.0, 10.0),
                },
                # Push the robot with 10-40 newtons, for 0.1-0.3 seconds
                "shove": {
                    "fn": events.push_force(
                        entity_manager=self.robot_manager,
                        force=(10.0, 40.0),
                    ),
                    "interval_range_s": (4.0, 8.0),
                    "duration_range_s": (0.1, 0.3),
                },
            },
        )
```

!!! note "Setting the state on reset"

    Events happen during an episode. To set the robot's state at the start of one, use the `on_reset` config of the [Entity Manager](entity.md).

## Event Configuration

Each event has the following possible values:

- **fn**: The event function, constructed with its params
- **interval_range_s**: How long to wait before the event fires, in seconds. This can be a number or a `(min, max)` range. Leave it out to only fire the event manually.
- **duration_range_s**: How long the event lasts, in seconds. This can be a number or a `(min, max)` range. Only used by events that last, like `push_force`, and required for those.

Every environment picks its own random time from each range. The interval is counted from the end of the previous event, and starts over when the environment resets.

!!! warning "Caution"

    The event will never fire if the episode is shorter than the interval. Keep `interval_range_s` well below your episode length. With a 10 second episode, use something like `(2.0, 5.0)`.

!!! warning "Caution"

    Create a new function for each event. If two events share the same `push_force` instance, one can end the other's push.

## Built-in Event Functions

You can find these in [`genesis_forge.mdp.events`](project:/api/mdp/events.md):

| Function | What it does | Duration |
|----------|--------------|----------|
| `push_velocity` | Instantly adds to the velocity of the robot's base | None, it's instant |
| `push_force` | Pushes one of the robot's links in a random horizontal direction, never up or down | Required |

`push_velocity` is the easier one to tune, because it has the same effect on any robot, big or small. `push_force` is closer to a real push, but it depends on your robot's mass. You can estimate the change in velocity with:

```
change in velocity = force * duration / mass

30 N * 0.2 s / 15 kg = 0.4 m/s
```

## Changing Events During Training

It can help to start with small pushes, and make them stronger as your robot learns:

```python
self.event_manager["shove"].fn.force = (20.0, 60.0)

# Used the next time each environment starts waiting
self.event_manager["shove"].interval_range_s = (2.0, 5.0)
```

The axes of `push_velocity` are a dictionary, so set the whole thing instead of changing one of its values:

```python
self.event_manager["kick"].fn.linear = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
```

You can also turn all events off, for example in your evaluation script:

```python
env.event_manager.enabled = False
```

## Debug Visualization

See your events as they happen:

```python
EventManager(
    self,
    cfg={...},
    debug_visualizer=True,
    debug_visualizer_cfg={
        "min_display_s": 0.5,  # Show every event for at least half a second
    },
)
```

By default, events are shown in the environments that your scene renders, which you set with `rendered_envs_idx`:

```python
self.scene = gs.Scene(
    vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
    # ...
)
```

To show them in different environments, set `envs_idx` in `debug_visualizer_cfg`.

Each event function decides what to draw. The built-in pushes show a red arrow in the direction of the push. A longer arrow is a stronger push. Torque and angular velocity are not shown.

You can change how an event looks on the function itself:

```python
events.push_force(
    entity_manager=self.robot_manager,
    force=(10.0, 40.0),
    debug_style={
        "color": (0.0, 0.0, 1.0, 1.0),
        "arrow_max_length": 1.0,
    },
)
```

!!! note "Recording with a camera"

    The debug visuals will always appear in the interactive viewer, but a camera will only render them if it was created with `debug=True`:

    ```python
    self.camera = self.scene.add_camera(
        pos=(-2.5, -1.5, 1.0),
        lookat=(0.0, 0.0, 0.3),
        debug=True,  # <~ show the debug visuals
    )
    ```

!!! warning "Caution"

    Drawing can slow down the simulation. It's recommended to only render a small number of environments, or to set `envs_idx` to a few of them.

## Triggering an Event

Fire an event on the next step, without waiting for its timer:

```python
self.event_manager.trigger("shove")          # all environments
self.event_manager.trigger("shove", [0, 1])  # only some environments
```

Leave out `interval_range_s` for an event that only fires when you trigger it:

```python
"stop": {
    "fn": stop_moving(entity_manager=self.robot_manager),  # <~ a custom event (see below)
},
```

An event that lasts still needs `duration_range_s`, so it knows when to end.

## Privileged Observations

Your real robot can't sense how hard it's being pushed, but this can be a useful observation for the critic:

```python
ObservationManager(
    self,
    name="critic",
    cfg={
        # ...
        "push": {
            "fn": lambda env: self.event_manager["shove"].fn.current_force,
            "scale": 0.02,  # 50 N becomes 1.0
        },
    },
)
```

## Custom Event Functions

Let's create an event that suddenly stops the robot. An event that happens all at once extends `EventMdpFn`, and its `__call__` method runs when the event fires:

```python
from dataclasses import dataclass

import torch

from genesis_forge import GenesisEnv
from genesis_forge.managers import EntityManager, EventMdpFn

@dataclass(kw_only=True, eq=False)
class stop_moving(EventMdpFn):
    """Suddenly stop the robot."""

    entity_manager: EntityManager

    def __call__(self, env: GenesisEnv, envs_idx: torch.Tensor):
        self.entity_manager.entity.zero_all_dofs_velocity(envs_idx)
```

Then add it to the event manager, like any other event:

```python
self.event_manager = EventManager(
    self,
    cfg={
        "stop": {
            "fn": stop_moving(entity_manager=self.robot_manager),
            "interval_range_s": (5.0, 10.0),
        },
    },
)
```

An event that lasts for a length of time extends `DurationEventMdpFn`, with one or more of these methods:

- `start(envs_idx)`: Called once, when the event fires
- `step(envs_idx)`: Called on every step that the event is active, including the first
- `end(envs_idx)`: Called once, when the event is over or the environment resets

For example, let's lift the robot with a constant force:

```python
import genesis as gs

from genesis_forge.managers import DurationEventMdpFn
from genesis_forge.utils import apply_external_wrench

@dataclass(kw_only=True, eq=False)
class lift(DurationEventMdpFn):
    """Lift the robot."""

    entity_manager: EntityManager
    force: float = 10.0

    def build(self):
        self._force = torch.zeros((self.env.num_envs, 1, 3), device=gs.device)
        self._force[:, :, 2] = self.force

    def step(self, envs_idx: torch.Tensor):
        # Genesis clears external forces after every step
        apply_external_wrench(
            self.entity_manager.entity,
            links_idx_local=[0],  # base link
            envs_idx=envs_idx,
            force=self._force[envs_idx],
        )
```

An event that lasts also needs a duration:

```python
self.event_manager = EventManager(
    self,
    cfg={
        "lift": {
            "fn": lift(entity_manager=self.robot_manager, force=20.0),
            "interval_range_s": (5.0, 10.0),
            "duration_range_s": (0.5, 1.0),
        },
    },
)
```

To show your event in the debug visualizer, add a `debug_draw` method and return the objects you draw. The manager removes them for you before the next render:

```python
    def debug_draw(self, envs_idx: torch.Tensor):
        scene = self.env.scene
        pos = self.entity_manager.entity.get_pos(envs_idx=envs_idx).cpu().numpy()
        pos += scene.envs_offset[envs_idx.cpu().numpy()]
        return [
            scene.draw_debug_sphere(p, radius=0.05, color=(0.0, 1.0, 0.0, 0.5))
            for p in pos
        ]
```

## Next Steps

- [Entity Manager](entity.md) - Set the robot's state on reset
- [Observation Manager](observation.md) - Add privileged observations for the critic
- [Built-in event functions](project:/api/mdp/events.md)
