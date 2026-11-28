"""
MDP function libraries for rewards, terminations, observations, events, and resets.

All functions follow the signature ``(env: GenesisEnv, **params) -> torch.Tensor``
and return batched tensors of shape ``(num_envs,)`` (scalar per env) or
``(num_envs, N)`` (vector per env for observations).

Submodules:
- :mod:`rewards` — reward functions (e.g. ``base_height``, ``command_tracking_lin_vel``)
- :mod:`terminations` — termination conditions (e.g. ``bad_orientation``, ``timeout``)
- :mod:`observations` — observation functions (e.g. ``entity_linear_velocity``)
- :mod:`events` — events during an episode (e.g. ``push_force``)
- :mod:`reset` — entity reset functions (e.g. ``randomize_terrain_position``)
"""

from . import events, observations, reset, rewards, terminations

__all__ = ["events", "observations", "reset", "rewards", "terminations"]
