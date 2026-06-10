# Genesis Forge — Developer Guide for AI Tools

A modular robotics RL training framework built on the Genesis physics simulator.
Provides a manager-based architecture for building Gymnasium-compatible parallel environments.

## Setup

```bash
pip install -e .
```

No test suite exists — verify changes manually using the examples:

```bash
python examples/stand_up/environment.py
```

Rebuild docs and regenerate `llms.txt` / `llms-full.txt`:

```bash
make llms        # builds docs + copies llms files to repo root
cd docs && make html  # docs only
```

Build and publish the package:

```bash
make build       # runs uv build into dist/
make deploy      # uploads to PyPI via twine
```

## Architecture

### Manager pattern

`ManagedEnvironment` is the primary base class. Users subclass it and override
`config()` to register managers. The environment calls `config()` → `build()` →
loop of `step()` / `reset(envs_idx)`.

```python
class MyEnv(ManagedEnvironment):
    def config(self):
        self.actuator_mgr = ActuatorManager(self, ...)
        self.reward_mgr = RewardManager(self, cfg={...})
```

### Manager registration

Every manager calls `super().__init__(env, type="<type>")` — this triggers
`env.add_manager(type, self)` automatically. **Never call `env.add_manager()`
directly.** The `type` string must match one of the `ManagerType` literals in
`genesis_forge/managers/base.py`:

```
"action" | "actuator" | "reward" | "termination" |
"contact" | "terrain" | "entity" | "command" | "observation"
```

Constraints:
- Only **one** `RewardManager` and one `TerminationManager` per env
- Multiple managers of all other types are allowed

### MDP functions

All MDP functions (rewards, terminations, observations, resets) share this signature:

```python
def my_fn(env: GenesisEnv, **params) -> torch.Tensor: ...
```

Return shape conventions:
- Rewards and terminations: `(num_envs,)` — one scalar per parallel env
- Observations: `(num_envs, N)` — one vector per parallel env

Always use `gs.device` (not `"cuda"` or `"cpu"`) for tensor device placement.

### Stateful MDP functions

Use `MdpFnClass` for functions that need per-env state across steps:

```python
class my_reward(MdpFnClass):
    def __init__(self, env, my_param=1.0):
        super().__init__(env)
        # allocate buffers here, e.g.:
        self.buf = torch.zeros(env.num_envs, device=gs.device)

    def reset(self, envs_idx):
        self.buf[envs_idx] = 0.0

    def __call__(self, env, my_param=1.0) -> torch.Tensor:
        ...
```

Use `ResetMdpFnClass` for entity reset functions that need initialization:

```python
class my_reset(ResetMdpFnClass):
    def __init__(self, env, entity, my_param): ...
    def __call__(self, env, entity, envs_idx, my_param): ...
```

`MdpFnClass` and `ResetMdpFnClass` are instantiated automatically during
`build()`. Params in the config dict are passed to both `__init__` and `__call__`.

## What to avoid

- **Non-tensor per-env state** — always pre-allocate `torch.Tensor` buffers of
  shape `(num_envs, ...)`. Python scalars or lists break multi-env parallelism.
- **Calling `env.add_manager()` manually** — managers self-register in `__init__`.
- **Hardcoding device** — use `gs.device`, not `"cuda"` or `torch.device("cuda")`.

## Naming and file conventions

| What | Where |
|------|-------|
| Manager classes | `genesis_forge/managers/` (flat layout) |
| MDP functions | `genesis_forge/mdp/rewards.py`, `terminations.py`, `observations.py`, `reset.py` |
| Config TypedDicts | Subclass `ConfigItemDict` from `genesis_forge/managers/config/` |
| Wrappers | `genesis_forge/wrappers/` |
| Examples | `examples/<name>/environment.py` |

## Key classes at a glance

| Class | File | Purpose |
|-------|------|---------|
| `GenesisEnv` | `genesis_env.py` | Base env, subclass for full manual control |
| `ManagedEnvironment` | `managed_env.py` | Config-driven env, override `config()` |
| `BaseManager` | `managers/base.py` | Abstract base; all managers inherit from this |
| `MdpFnClass` | `managers/config/mdp_fn_class.py` | Base for stateful reward/termination/obs functions |
| `ResetMdpFnClass` | `managers/config/mdp_fn_class.py` | Base for stateful entity reset functions |
