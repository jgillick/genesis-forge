"""Shared by the MDP function tests."""

import torch


def built(fn, env, **context):
    """Bind an MDP function to the environment and build it, as its manager would."""
    fn.context(env, **context)
    fn.safe_build()
    return fn


class FakeVelCmd:
    """Stands in for a VelocityCommandManager, holding a fixed command."""

    def __init__(self, command, range=None):
        self.command = command
        self.range = range or {
            "lin_vel_x": (-1.0, 1.0),
            "lin_vel_y": (-1.0, 1.0),
            "ang_vel_z": (-1.0, 1.0),
        }

    def stopped_envs(self, threshold: float = 0.01) -> torch.Tensor:
        return torch.norm(self.command, dim=1) <= threshold
