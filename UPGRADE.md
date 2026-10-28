# Upgrade Guide

This document collects upgrade notes for each breaking release of Genesis Forge. Find
the section for the version you're upgrading to.

## Upgrading to version 1.0

### Small items

- Deprecated parameters from previous versions have now been removed
- The action manager's deprecated `actuators` property has been removed. Use `actuator_manager`.
- `*_attr` params now take the object itself (e.g. `entity_attr="robot"` → `entity=self.robot`).
  See [below](#_attr-params-replaced-with-direct-object-references).
- `PositionActionManager` and `PositionWithinLimitsActionManager` no longer accept the `quiet_action_errors` argument.
- `BaseManager.__init__` no longer accepts the `enabled` argument -- it was already ignored.
- `genesis-world>=1.3.0` is now required (previously `>=1.0.0`).

### Built-in MDP function params

Every built-in MDP function (rewards, terminations, observations, etc) are now _constructed_ with params,
instead of referenced with a separate `"params"` dict.

**Before:**

```python
"foo": {
    "fn": rewards.base_height,
    "params": {"target_height": 0.3},
}
```

**After:**

```python
"foo": {
    "fn": rewards.base_height(target_height=0.3),
}
```

This includes built-in functions that take no params. Functions like
`rewards.is_alive`, `rewards.terminated`, `rewards.action_rate_l2`,
`terminations.timeout`, and `reset.zero_all_dofs_velocity` are now classes too, so they
must be called with `()`.

**Before:**

```python
"timeout": {"fn": terminations.timeout, "time_out": True},
```

**After:**

```python
"timeout": {"fn": terminations.timeout(), "time_out": True},
```

### How to change mdp params during training

Params are now attributes on the function instance, not separate dict entries.

**Before:**

```python
self.reward_manager.cfg["base_height"].params["target_height"] = 0.3
```

**After:**

```python
self.reward_manager.cfg["base_height"].fn.target_height = 0.3
```

Changing several params at once should go through `update(...)` so the function rebuilds
internal state once instead of once per assignment:

```python
self.reward_manager.cfg["has_contact"].fn.update(
    min_contacts=2,
    threshold=1.0,
)
```

`increment_param()` and `increment_weight()` still work the same.

---

### Custom MdpFnClass / ResetMdpFnClass subclasses

If you wrote your own reward, termination, observation, or reset functions as
a `MdpFnClass` or `ResetMdpFnClass` subclass, convert them to use `MdpFn` / `ResetMdpFn` data classes:

**Before:**

```python
class random_values(MdpFnClass):
    def __init__(self, env: GenesisEnv, size: int = 1, scale: float = 1.0):
        self.env = env
        self.size = size
        self.scale = scale
        self.build()

    def build(self):
        self._buffer = torch.zeros((self.env.num_envs, self.size), device=gs.device)

    def __call__(self, env, size, scale):
        return self._buffer.uniform_(-scale, scale)

...

"noise": {
    "fn": random_values,
    "params": {"size": 3, "scale": 0.5},
}
```

**After:**

```python
@dataclass(kw_only=True, eq=False)
class random_values(MdpFn):
    size: int = 1
    scale: float = 1.0

    def build(self):
        self._buffer = torch.zeros((self.env.num_envs, self.size), device=gs.device)

    def __call__(self, env):
        return self._buffer.uniform_(-self.scale, self.scale)

...

"noise": {
    "fn": random_values(size=3, scale=0.5),
}
```

Because `build()` runs again on every param change, setting
`fn.size = 5` mid-training reallocates the buffer at the new size automatically.

1. **Base class.** `MdpFnClass` → `MdpFn`. `ResetMdpFnClass` → `ResetMdpFn`.
2. **Add the decorator.** `@dataclass(kw_only=True, eq=False)` directly above the class.
3. **Delete `__init__`.** Its params become class-level field annotations with the
   same defaults.
4. **Simplify `__call__`.** Drop the repeated params from its signature — read them as
   `self.<param_name>` instead. It only ever takes `env` (plus `entity` and `envs_idx`
   for `ResetMdpFn`).
5. **Move setup into `build()`.** Anything your `__init__` computed from `env` /
   `entity` (buffers, derived values) belongs here instead — those aren't available
   until `build()`. It runs once at environment build, and again on every param
   change, so re-derive everything from the current field values each time.
6. **Drop any manual `self.build()` call.** The framework calls it for you now.

### Custom plain MDP functions

Any mdp function you defined as a plain function (not class-based function),
should still work the same as before (though, note, the params dict will likely be deprecated in the future):

```python
def my_custom_reward(env: GenesisEnv, target: float) -> torch.Tensor:
    # ... do reward calculations here ...

...

RewardManager(
    self,
    cfg={
        "my_reward": {
            "weight": 1.0,
            "fn": my_custom_reward,
            "params": {"target": 1.2},
        },
    },
)
```

### Legacy actuator kwargs on action managers.

`PositionActionManager` (and other
`BaseActionManager` subclasses) no longer accept actuator settings directly
(`joint_names`, `default_pos`, `pd_kp`, `pd_kv`, `max_force`, `damping`, `stiffness`,
`frictionloss`, `noise_scale`). Define these on an `ActuatorManager` instead:

**Before:**

```python
self.action_manager = PositionActionManager(
    self,
    joint_names=".*",
    default_pos={".*": 0.0},
    pd_kp=50,
    pd_kv=0.5,
)
```

**After:**

```python
self.actuator_manager = ActuatorManager(
    self,
    joint_names=".*",
    default_pos={".*": 0.0},
    kp=50,
    kv=0.5,
)
self.action_manager = PositionActionManager(
    self,
    actuator_manager=self.actuator_manager,
)
```

### Custom action managers

If you subclass `BaseActionManager`, the methods you override have changed:

- **`send_actions_to_simulation(self, actions)`** now receives the processed actions
  returned by `step()`, shape `(num_envs, num_dofs)`. Add the `actions` parameter to
  your override, or it fails with `TypeError: takes 1 positional argument but 2 were given`.
- **`process_actions(self, actions)`** no longer has a default. The base implementation
  used to return the actions unchanged, and now raises `NotImplementedError`. If you
  relied on that, override it with `return actions`. It now receives one action per
  DOF, `(num_envs, num_dofs)`, already fanned out from any `action_groups`.
- **`reset(envs_idx)`** now clears the raw, current, and last actions of the reset
  environments, along with the delay buffer. Call `super().reset(envs_idx)` if you
  override it.

For a scale/offset/clip action manager, consider subclassing the new
`AffineDofActionManager` instead, which implements `process_actions` for you.

**Before:**

```python
class MyActionManager(BaseActionManager):
    def send_actions_to_simulation(self):
        self.actuator_manager.control_dofs_position(self.get_actions(), self.dofs_idx)
```

**After:**

```python
class MyActionManager(BaseActionManager):
    def process_actions(self, actions):
        return actions

    def send_actions_to_simulation(self, actions):
        self.actuator_manager.control_dofs_position(actions, self.dofs_idx)
```

### Action manager param removed from some MDP functions

`rewards.dof_similar_to_default`, `observations.entity_dofs_position`,
`observations.entity_dofs_velocity`, and `observations.entity_dofs_force` no
longer accept `action_manager` -- pass
`actuator_manager` instead:

**Before:**

```python
"fn": rewards.dof_similar_to_default(action_manager=self.action_manager),
```

**After:**

```python
"fn": rewards.dof_similar_to_default(actuator_manager=self.actuator_manager),
```

### `*_attr` params replaced with direct object references

Parameters that referenced an environment object used to take the _name_ of the
attribute (`entity_attr="robot"`). Now they take the object itself (`entity=self.robot`).

This applies to everything except wrappers, like`VideoWrapper`'s `camera_attr`, since they're configured outside the environment and wont have a direct reference to the object.

**Before:**

```python
self.robot_manager = EntityManager(self, entity_attr="robot")
self.terrain_manager = TerrainManager(self, terrain_attr="terrain")
self.contact_manager = ContactManager(
    self,
    entity_attr="robot",
    with_entity_attr="terrain",
    ...
)
"fn": rewards.base_height(target_height=0.3, entity_attr="robot"),
```

**After:**

```python
self.robot_manager = EntityManager(self, entity=self.robot)
self.terrain_manager = TerrainManager(self, terrain=self.terrain)
self.contact_manager = ContactManager(
    self,
    entity=self.robot,
    with_entity=self.terrain,
    ...
)
"fn": rewards.base_height(target_height=0.3, entity=self.robot),
```

### `default_noise_scale` removed from `ActuatorManager`.

Use `NoisyValue` on individual values instead of a single global noise scale:

**Before:**

```python
self.actuator_manager = ActuatorManager(self, kp=50, default_noise_scale=0.02)
```

**After:**

```python
self.actuator_manager = ActuatorManager(self, kp=NoisyValue(50, 0.02))
```

### Observation noise is now applied before `scale`

Observation noise used to be added _after_ the value was scaled. It's now added
_before_, in the raw units the observation function returns, so it can match the real
sensor's noise regardless of the scale you choose for the policy. This applies to both
the per-entry `noise` and the `ObservationManager`'s default `noise`.

Nothing errors, but any entry with both `noise` and a `scale` other than `1.0` now gets
a different amount of noise. For example, `scale=0.05` with `noise=0.1` used to add
±0.1 to the policy input, and now adds ±0.005. To keep the old behavior, divide the
noise by the scale:

**Before:**

```python
"dof_vel": {
    "fn": observations.entity_dofs_velocity(actuator_manager=self.actuator_manager),
    "scale": 0.05,
    "noise": 0.1,
},
```

**After:**

```python
"dof_vel": {
    "fn": observations.entity_dofs_velocity(actuator_manager=self.actuator_manager),
    "scale": 0.05,
    "noise": 2.0,  # 0.1 / 0.05
},
```

### Non-finite observations now raise an error

The `ObservationManager` now raises a `ValueError` when any observation is `inf` or
NaN, naming the entry and how many environments are affected. Previously these values
were passed straight to the RL library, where a single physics blow-up could poison the
policy's normalizer and corrupt every environment.

To keep a rare blow-up from stopping training, give observations that can spike a
`clip` range. It's applied after noise and before `scale`. Make it several times wider
than any value a healthy simulation produces, so it only catches a simulation that has
come apart:

```python
"dof_vel": {
    "fn": observations.entity_dofs_velocity(actuator_manager=self.actuator_manager),
    "scale": 0.05,
    "clip": (-100.0, 100.0),
},
```

### Reset always receives a tensor of environment indices

`ManagedEnvironment.reset(None)` now resolves `None` to `env.all_envs_idx` (a
`torch.long` tensor of every environment) before calling `super().reset()` and each
manager's `reset()`. Managers and `reset()` overrides no longer receive `None` for a
full reset. If your custom code has an `if envs_idx is None:` branch for a full reset,
that branch will no longer run. Check `len(envs_idx) == self.env.num_envs` instead, or
make the code work the same for any set of indices.

Environment indices are also now typed as `torch.Tensor` rather than `list[int]`.

### `VideoWrapper` step and episode counts start at 1

The step and episode counts passed to `episode_trigger`, `step_trigger`, and the video
filenames now start at 1 instead of 0. Custom triggers fire one count later than
before. For example, `episode_trigger=lambda ep: ep % 100 == 0` used to record the
very first episode, and now first records episode 100. The default trigger now records
episodes 1, 8, 27, ... and no longer records episode 0.

### New deprecations

These still work in 1.0, but warn and will be removed in a future release:

| Deprecated                                         | Replacement                                       |
| -------------------------------------------------- | ------------------------------------------------- |
| `rewards.stand_still_joint_deviation_l1`           | `rewards.stopped_joint_deviation_l1`              |
| `VelocityCommandManager(standing_probability=...)` | `VelocityCommandManager(stopped_probability=...)` |
| `VelocityCommandManager.standing_probability`      | `VelocityCommandManager.stopped_probability`      |
| `VelocityCommandManager.standing_envs`             | `VelocityCommandManager.stopped_envs()`           |
| `debug_visualizer_cfg` key `standing_color`        | `stopped_color`                                   |
| `debug_visualizer_cfg` key `standing_ball_radius`  | `stopped_ball_radius`                             |
| `wrappers.skrl.SkrlEnvWapper`                      | `wrappers.SkrlEnvWrapper`                         |

`SkrlEnvWapper` is no longer exported from `genesis_forge.wrappers`, so
`from genesis_forge.wrappers import SkrlEnvWapper` now fails with an `ImportError`.
Import `SkrlEnvWrapper` instead.
