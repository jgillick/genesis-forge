# Upgrade Guide

This document collects upgrade notes for each breaking release of Genesis Forge. Find
the section for the version you're upgrading to.

## Upgrading to version 1.0

### Small items

- Deprecated parameters from previous versions have now been removed
- The action manager's `actuators` property was been renamed to `actuator_manager`.
- `*\_attr params` to point to the actual instances (e.g. robot_attr="robot" to robot=self.robot)

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
def my_custom_reward(env: GenesisEnv, target: float):
    # ... do reward calculations here ...

...

RewardManager(
    self,
    cfg={
        "fn": my_custom_reward,
        "params": {
            "target": 1.2
        },
    }
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
