# Upgrade Guide

This document collects upgrade notes for each breaking release of Genesis Forge. Find
the section for the version you're upgrading to.

## Upgrading to version 1.0

`MdpFnClass` / `ResetMdpFnClass` have been removed, and MDP function params are now
typed dataclass fields instead of a `params` dict. See the release notes for why.

### Built-inMDP function params

Every built-in MDP function (rewards, terminations, observations, etc) that takes params
is now **constructed** with them, instead of referenced with a separate `"params"` dict.

**Before:**

```python
"fn": rewards.base_height,
"params": {"target_height": 0.3},
```

**After:**

```python
"fn": rewards.base_height(target_height=0.3),
```

This applies to every built-in in `genesis_forge.mdp` (`rewards`, `terminations`,
`observations`, `reset`) mdp function.

### Changing mdp function params during training

Params are now attributes on the function instance, not separate dict entries.

**Before:**

```python
self.reward_manager.cfg["base_height"].params["target_height"] = 0.3
```

**After:**

```python
self.reward_manager.cfg["base_height"].fn.target_height = 0.3
```

Changing several params at once should go through `update()` so the function rebuilds
once instead of once per assignment:

```python
self.reward_manager.cfg["smooth"].fn.update(sensitivity=0.3, entity_attr="robot")
```

`increment_param()` and `increment_weight()` need no code changes.

---

### Custom MdpFnClass / ResetMdpFnClass subclasses

If you wrote your own stateful reward, termination, observation, or reset function as
a class, convert it to `MdpFn` / `ResetMdpFn`:

**Before:**

```python
class randomize_link_mass_shift(ResetMdpFnClass):
    def __init__(
        self,
        env: GenesisEnv,
        entity: RigidEntity,
        link_name: str,
        mass_range: tuple[float, float],
    ):
        self.env = env
        self._entity = entity
        self._link_name = link_name
        self._links_idx_local = []
        self._mass_shift_buffer = None
        self.build()

    def build(self):
        self._links_idx_local = []
        if self._link_name is not None:
            links = links_by_name_pattern(self._entity, self._link_name)
            if len(links) > 0:
                self._links_idx_local = [link.idx_local for link in links]
                self._mass_shift_buffer = torch.zeros(
                    (self.env.num_envs, len(self._links_idx_local)), device=gs.device
                )
            else:
                raise ValueError(f"No links found with name/pattern '{self._link_name}'")

    def __call__(self, env, entity, envs_idx, link_name, mass_range):
        self._mass_shift_buffer[envs_idx, :].uniform_(*mass_range)
        self._entity.set_mass_shift(
            self._mass_shift_buffer[envs_idx],
            links_idx_local=self._links_idx_local,
            envs_idx=envs_idx,
        )
```

**After:**

```python
@dataclass(kw_only=True, eq=False)
class randomize_link_mass_shift(ResetMdpFn):
    link_name: str = None
    mass_range: tuple[float, float] = None

    def build(self):
        self._links_idx_local = []
        if self.link_name is not None:
            links = links_by_name_pattern(self.entity, self.link_name)
            if len(links) > 0:
                self._links_idx_local = [link.idx_local for link in links]
                self._mass_shift_buffer = torch.zeros(
                    (self.env.num_envs, len(self._links_idx_local)), device=gs.device
                )
            else:
                raise ValueError(f"No links found with name/pattern '{self.link_name}'")

    def __call__(self, env, entity, envs_idx):
        self._mass_shift_buffer[envs_idx, :].uniform_(*self.mass_range)
        entity.set_mass_shift(
            self._mass_shift_buffer[envs_idx],
            links_idx_local=self._links_idx_local,
            envs_idx=envs_idx,
        )
```

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

Any custom mdp function you defined as a plain function (not class-based function),
should still work the same as before:

```python
 RewardManager(
    self,
    cfg={
        "fn": my_custom_reward,
        "params": {
            "target": 1.2
        },
    }
}

...

def my_custom_reward(env: GenesisEnv, target: float):
    # ... do reward calculations here ...
```
