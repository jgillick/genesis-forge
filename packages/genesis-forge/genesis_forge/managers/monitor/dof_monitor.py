from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import genesis as gs
import torch

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.actuator import ActuatorManager

from .monitor import Monitor
from .monitor_config import MonitorConfig

_GROUP_REDUCTIONS = {
    "mean": torch.mean,
    "max": torch.amax,
    "min": torch.amin,
    "sum": torch.sum,
}


class DofMonitor(Monitor):
    """
    The base for monitors of an actuator manager's DOFs, which can report on named groups
    of joints.

    Args:
        env: The environment to monitor.
        actuator_manager: The actuator manager whose DOFs are monitored.
        dof_groups: Named joint name patterns to report on, like `{"hip": ".*_Hip"}`.
                    If None, the values are for all the DOFs together.
        cfg: Additional monitor items.
        logging_enabled: Whether to add the values to the training logs.
        logging_tag: The section name the values are logged under.
    """

    _reset_this_step: torch.Tensor
    """Which environments were reset this step."""

    def __init__(
        self,
        env: GenesisEnv,
        actuator_manager: ActuatorManager,
        dof_groups: Mapping[str, str | list[str]] | None = None,
        cfg: dict[str, MonitorConfig] | None = None,
        logging_enabled: bool = True,
        logging_tag: str = "Monitor",
    ):
        super().__init__(
            env,
            cfg=cfg,
            logging_enabled=logging_enabled,
            logging_tag=logging_tag,
        )
        self.actuator_manager = actuator_manager
        self._dof_groups_cfg = dof_groups
        self._dof_groups: dict[str, torch.Tensor] | None = None

    @property
    def dof_groups(self) -> dict[str, torch.Tensor] | None:
        """
        Each group's DOF tensor column positions from the actuator manager, or
        None if no groups were configured. Resolved at build.
        """
        return self._dof_groups

    def build(self):
        """Resolve the DOF groups into tensor columns, then build the monitor items."""
        if self._dof_groups_cfg is not None:
            groups = self.actuator_manager.resolve_dof_groups(self._dof_groups_cfg)

            # Monitor tensors have a column per actuator DOF, in the actuator manager's order
            dofs_idx = self.actuator_manager.dofs_idx
            self._dof_groups = {}
            for group, group_dofs_idx in groups.items():
                columns = [dofs_idx.index(idx) for idx in group_dofs_idx.tolist()]
                self._dof_groups[group] = torch.tensor(columns, device=gs.device)
        self._reset_this_step = torch.zeros(
            self.env.num_envs, dtype=torch.bool, device=gs.device
        )
        super().build()

    def step(self):
        """Calculate and log the monitor items for a new step."""
        self._reset_this_step[:] = False
        super().step()

    def reset(self, envs_idx: torch.Tensor | None = None):
        """
        Reset the monitor items, then recalculate them, with zero torque for the
        reset environments until their next step.
        """
        if envs_idx is None:
            envs_idx = self.env.all_envs_idx
        self._reset_this_step[envs_idx] = True
        super().reset(envs_idx)

    def group_labels(self) -> list[str] | None:
        """The DOF group names, used as the labels of per-group monitor items."""
        if self._dof_groups is None:
            return None
        return list(self._dof_groups)

    def by_group(
        self, values: torch.Tensor, reduce: Literal["mean", "max", "min", "sum"]
    ) -> torch.Tensor:
        """
        Combine the per-DOF values within each DOF group.

        Args:
            values: Per-DOF values, shape (num_envs, num_dofs).
            reduce: How to combine each group's values.

        Returns:
            Shape (num_envs, n_groups), or (num_envs,) for all DOFs when there are no groups.

        Example::

            rms_torque = self.by_group(self.dof_torque.square(), "mean").sqrt()
        """
        reduce_fn = _GROUP_REDUCTIONS[reduce]
        if self._dof_groups is None:
            return reduce_fn(values, dim=-1)
        groups = [
            reduce_fn(values[:, cols], dim=-1) for cols in self._dof_groups.values()
        ]
        return torch.stack(groups, dim=-1)

    def _read_dof_torque(self) -> torch.Tensor:
        """
        Read the torque each DOF applied this step, clipped to its force limit.

        Environments reset this step read zero. Their new episode hasn't applied any
        force yet, and the simulator's value would be calculated from the new pose
        against the last episode's target.
        """
        torque = self.actuator_manager.get_dofs_control_force(clip_to_max_force=True)
        return torque.masked_fill(self._reset_this_step.unsqueeze(-1), 0.0)
