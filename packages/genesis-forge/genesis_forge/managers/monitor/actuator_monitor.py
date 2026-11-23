from __future__ import annotations

from collections.abc import Mapping

import genesis as gs
import torch

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.actuator import ActuatorManager

from .dof_monitor import DofMonitor
from .monitor_config import MonitorConfig


class ActuatorMonitor(DofMonitor):
    """
    Tracks how hard your robot's actuators are working, from the torque actually
    applied (clipped to the actuator force limits).

    Monitor items, each for every DOF group:

    - `torque`: The RMS torque of the group's joints, logged as the RMS across
      environments, `torque_rms`. Drives motor heat and battery use.
    - `torque_peak`: The largest torque among the group's joints.
    - `saturation_frac`: The fraction of the group's joints at or near their force limit.

    Args:
        env: The environment to monitor.
        actuator_manager: The actuator manager whose DOFs are monitored.
        dof_groups: Named joint name patterns to report on, like `{"hip": ".*_Hip"}`.
        saturation_threshold: A joint counts as saturated when its torque is at or above
                              this fraction of its force limit.
        cfg: Additional monitor items.
        logging_enabled: Whether to add the values to the training logs.
        logging_tag: The section name the values are logged under.

    Example::

        class MyEnv(ManagedEnvironment):
            def config(self):
                self.actuator_manager = ActuatorManager(self, max_force=8.0, ...)

                self.actuator_monitor = ActuatorMonitor(
                    self,
                    actuator_manager=self.actuator_manager,
                    dof_groups={"hip": ".*_Hip", "knee": ".*_Knee"},
                )
    """

    _dof_torque: torch.Tensor
    _dof_saturated: torch.Tensor
    _force_range: tuple[torch.Tensor, torch.Tensor]

    def __init__(
        self,
        env: GenesisEnv,
        actuator_manager: ActuatorManager,
        dof_groups: Mapping[str, str | list[str]] | None = None,
        saturation_threshold: float = 0.95,
        cfg: dict[str, MonitorConfig] | None = None,
        logging_enabled: bool = True,
        logging_tag: str = "Actuators",
    ):
        self.saturation_threshold = saturation_threshold

        items: dict[str, MonitorConfig] = {
            "torque": {
                "fn": self._torque,
                "labels": self.group_labels,
                "reduce": "rms",
            },
            "torque_peak": {
                "fn": self._torque_peak,
                "labels": self.group_labels,
                "reduce": ["mean", "p99"],
            },
            "saturation_frac": {
                "fn": self._saturation_frac,
                "labels": self.group_labels,
            },
        }

        super().__init__(
            env,
            actuator_manager=actuator_manager,
            dof_groups=dof_groups,
            cfg={**items, **(cfg or {})},
            logging_enabled=logging_enabled,
            logging_tag=logging_tag,
        )

    """
    Properties
    """

    @property
    def dof_torque(self) -> torch.Tensor:
        """The torque applied to each DOF this step. Shape (num_envs, num_dofs)."""
        return self._dof_torque

    @property
    def dof_saturated(self) -> torch.Tensor:
        """Whether each DOF is at or near its force limit. Shape (num_envs, num_dofs)."""
        return self._dof_saturated

    """
    Lifecycle Operations
    """

    def build(self):
        """
        Initialize the torque buffers, then build the monitor items.
        """
        per_dof_shape = (self.env.num_envs, self.actuator_manager.num_dofs)
        self._dof_torque = torch.zeros(
            per_dof_shape, device=gs.device, dtype=gs.tc_float
        )
        self._dof_saturated = torch.zeros(
            per_dof_shape, device=gs.device, dtype=torch.bool
        )

        self._read_force_range()
        super().build()

    def reset(self, envs_idx: torch.Tensor | None = None):
        """
        Read the force limits, which may be randomized on reset, then reset the items.
        """
        self._read_force_range()
        super().reset(envs_idx)

    def update(self):
        """
        Read the applied torque, then calculate the monitor items.
        """
        self._dof_torque[:] = self._read_dof_torque()

        lower, upper = self._force_range
        threshold = self.saturation_threshold
        self._dof_saturated[:] = (self._dof_torque >= threshold * upper) | (
            self._dof_torque <= threshold * lower
        )
        super().update()

    def _read_force_range(self):
        """
        Read the simulator's current force limits for each DOF, which can differ by
        direction and per environment.
        """
        lower, upper = self.actuator_manager.get_dofs_force_range()
        self._force_range = (lower.to(gs.device), upper.to(gs.device))

    """
    Monitor items
    """

    def _torque(self, env: GenesisEnv) -> torch.Tensor:
        """The RMS torque of each group's joints."""
        return self.by_group(self.dof_torque.square(), "mean").sqrt()

    def _torque_peak(self, env: GenesisEnv) -> torch.Tensor:
        """The largest torque among each group's joints."""
        return self.by_group(self.dof_torque.abs(), "max")

    def _saturation_frac(self, env: GenesisEnv) -> torch.Tensor:
        """The fraction of each group's joints at or near their force limit."""
        return self.by_group(self.dof_saturated.float(), "mean")
