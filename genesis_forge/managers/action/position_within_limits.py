from __future__ import annotations

import torch

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.actuator import ActuatorManager
from genesis_forge.utils import assign_by_pattern
from genesis_forge.values import ensure_dof_pattern

from .position_action_manager import PositionActionManager


class PositionWithinLimitsActionManager(PositionActionManager):
    """
    This is similar to `PositionActionManager` but converts actions from the range -1.0 - 1.0 to DOF positions within the limits of the actuators.

    Args:
        env: The environment to manage the DOF actuators for.
        actuator_manager: The actuator manager which is used to setup and control the DOF joints.
        actuator_joints: Which joints of the actuator manager that this action manager will control.
                         These can be full names or regular expressions.
        limit: A dictionary of DOF name patterns and their position limits.
               If omitted, the limits will be set to the limits of the actuators defined in the model.
        soft_limit_scale_factor: Scales the range of all limits by this factor to establish a safety region within the limits. Defaults to 1.0.
        delay_step: The number of steps to delay the actions for.
                    This is an easy way to emulate the latency in the system.

    Simple example using the limits defined in the model::

        class MyEnv(ManagedEnvironment):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def config(self):
                self.actuator_manager = ActuatorManager(
                    self,
                    joint_names=".*",
                    default_pos={".*": 0.0},
                    kp={".*": 50},
                    kv={".*": 0.5},
                )
                self.action_manager = PositionalActionManager(
                    self,
                    actuator_manager=self.actuator_manager,
                    actuator_joints=[".*"], # optional joint filter
                )

    Example defining custom limits::

        class MyEnv(ManagedEnvironment):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def config(self):
                self.actuator_manager = ActuatorManager(
                    self,
                    joint_names=".*",
                    default_pos={".*": 0.0},
                    kp={".*": 50},
                    kv={".*": 0.5},
                )
                self.action_manager = PositionalActionManager(
                    self,
                    actuator_manager=self.actuator_manager,
                    limit = {
                        ".*_Hip": (-1.0, 1.0),
                        ".*_Femur": (-1.5, 1.2),
                    },
                )

    """

    def __init__(
        self,
        env: GenesisEnv,
        actuator_manager: ActuatorManager | None = None,
        actuator_joints: list[str] | str = ".*",
        limit: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
        soft_limit_scale_factor: float = 1.0,
        delay_step: int = 0,
    ):
        super().__init__(
            env,
            actuator_manager=actuator_manager,
            actuator_joints=actuator_joints,
            delay_step=delay_step,
        )
        self._limit_cfg = ensure_dof_pattern(limit if limit is not None else {})
        self._soft_limit_scale_factor = soft_limit_scale_factor

    """
    Lifecycle Operations
    """

    def build(self):
        """
        Builds the manager and initialized all the buffers.
        """
        super().build()

        lower, upper = self._get_dof_limits()
        lower = lower.unsqueeze(0).expand(self.env.num_envs, -1)
        upper = upper.unsqueeze(0).expand(self.env.num_envs, -1)
        self._offset = (upper + lower) * 0.5
        self._scale = (upper - lower) * 0.5 * self._soft_limit_scale_factor

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Convert the actions to position commands within the limits.

        Args:
            actions: The incoming step actions to handle.

        Returns:
            The actions as position commands.
        """
        # Convert the action from -1 to 1, to absolute position within the actuator limits
        actions = actions.clamp(-1.0, 1.0)
        actions = actions * self._scale + self._offset
        return actions

    """
    Internal methods
    """

    def _get_dof_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Define the position limits for the DOFs
        """
        lower, upper = self.get_dofs_limits()
        dof_names = list[str](self.dofs.keys())
        for i, limits in enumerate(assign_by_pattern(dof_names, self._limit_cfg)):
            if limits is None:
                continue
            lower[i], upper[i] = limits
        return lower, upper
