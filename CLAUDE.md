# Genesis Forge — Developer Guide for AI Tools

A modular robotics RL training framework built on the Genesis physics simulator.
Provides a manager-based architecture for building Gymnasium-compatible parallel environments.

## Setup

```bash
pip install -e .
```

Run the tests, which cover the pure-Python framework (config item dispatch, MDP
function lifecycle) against a fake environment — no Genesis scene, no GPU:

```bash
uv run pytest
```

Anything touching the simulator is verified by running an example briefly:

```bash
uv run --directory examples/stand_up python train.py -n 16 --max_iterations 1
```

Each example has its own venv holding a **non-editable** copy of genesis-forge,
so repo changes are not picked up by default. Either shadow it:

```bash
PYTHONPATH=$PWD uv run --directory examples/stand_up python train.py -n 16 --max_iterations 1
```

or refresh the copy with `uv sync --directory examples/stand_up --reinstall-package genesis-forge`.

Rebuild docs and regenerate `llms.txt` / `llms-full.txt`:

```bash
make docs        # builds docs + copies llms files to repo root
make serve       # live-reloading docs preview
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

An MDP function that takes **no params** is a plain function:

```python
def my_fn(env: GenesisEnv) -> torch.Tensor: ...
```

An MDP function **with params** subclasses `MdpFn` and declares them as
keyword-only dataclass fields. That single declaration is the constructor
signature, the attributes read in `__call__`, and the surface a curriculum
mutates — so params are type-checked and autocompleted at every one of them:

```python
@dataclass(kw_only=True, eq=False)
class my_reward(MdpFn):
    target_height: float = 0.3
    entity_manager: EntityManager = None

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        pos = self.entity_manager.entity.get_pos()
        return torch.square(pos[:, 2] - self.target_height)
```

In a manager config the function is **constructed**, not referenced, and there
is no `params` dict:

```python
"height": {"weight": -50.0, "fn": my_reward(target_height=0.35)},
```

`eq=False` is part of the convention: MDP functions carry per-env buffers, so
two instances with equal params are not interchangeable, and the generated
`__eq__` would also make them unhashable.

Always write `@dataclass(kw_only=True, eq=False)` yourself — `MdpFn` also
applies it automatically via `__init_subclass__` as a backstop for a forgotten
decorator, but that backstop is invisible to griffe (which builds the
published API reference from source): an undecorated class still works
correctly, but its rendered constructor signature comes out empty.

Return shape conventions:
- Rewards and terminations: `(num_envs,)` — one scalar per parallel env
- Observations: `(num_envs, N)` — one vector per parallel env

Always use `gs.device` (not `"cuda"` or `"cpu"`) for tensor device placement.

### Stateful MDP functions

Allocate buffers in `build()`, never in `__init__` — `self.env` does not exist
until the manager binds the function during the environment build phase:

```python
@dataclass(kw_only=True, eq=False)
class my_reward(MdpFn):
    my_param: float = 1.0

    def build(self):
        self.buf = torch.zeros(self.env.num_envs, device=gs.device)

    def reset(self, envs_idx):
        self.buf[envs_idx] = 0.0

    def __call__(self, env: GenesisEnv) -> torch.Tensor:
        ...
```

`build()` is called once by the manager, and again whenever a declared param is
assigned. It must be **idempotent** and re-derive everything from the current
field values. Never call it yourself.

A buffer `build()` does not reallocate survives a param change — that is how a
function keeps per-env history across a curriculum step. Reallocating is
equally valid when the change invalidates that history; the choice belongs to
the function.

Only declared params trigger a rebuild. Anything else assigned on `self`
(per-step scratch state, buffers created in `build()`) does not.

Use `ResetMdpFn` for entity reset functions, which are bound to an entity as
well as the environment:

```python
@dataclass(kw_only=True, eq=False)
class my_reset(ResetMdpFn):
    my_param: float = 1.0

    def build(self):
        ...  # both self.env and self.entity are available here

    def __call__(self, env, entity, envs_idx): ...
```

`MdpFnClass` and `ResetMdpFnClass` were the previous way to declare a stateful MDP
function (params via a config `params` dict, constructed by the manager, received
again at `__call__`). They have been removed. See `UPGRADE.md` for the conversion
recipe if you have code still using them.

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
| `MdpFn` | `managers/config/mdp_fn.py` | Base for reward/termination/obs functions with params |
| `ResetMdpFn` | `managers/config/mdp_fn.py` | Base for entity reset functions with params |
