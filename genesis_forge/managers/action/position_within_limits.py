from __future__ import annotations
import re
import torch
from genesis import gs

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.values import ensure_dof_pattern
from .position_action_manager import PositionActionManager
from genesis_forge.managers.actuator import ActuatorManager


class PositionWithinLimitsActionManager(PositionActionManager):
    """
    This is similar to `PositionActionManager` but converts actions from the range -1.0 - 1.0 to DOF positions within the limits of the actuators.

    Args:
        env: The environment to manage the DOF actuators for.
        actuator_manager: The actuator manager which is used to setup and control the DOF joints.
        actuator_joints: Which joints of the actuator manager that this action manager will control.
                         These can be full names or regular expressions.
        limit: (optional) A dictionary of DOF name patterns and their position limits.
               If omitted, the limits will be set to the limits of the actuators defined in the model.
        quiet_action_errors: Whether to quiet action errors.
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
        quiet_action_errors: bool = False,
        limit: tuple[float, float] | dict[str, tuple[float, float]] = {},
        delay_step: int = 0,
        **kwargs,
    ):
        super().__init__(
            env,
            actuator_manager=actuator_manager,
            actuator_joints=actuator_joints,
            quiet_action_errors=quiet_action_errors,
            delay_step=delay_step,
            **kwargs,
        )
        self._limit_cfg = ensure_dof_pattern(limit)

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
        self._scale = (upper - lower) * 0.5

    def handle_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Converts the actions to position commands, and send them to the DOF actuators.
        Override this function if you want to change the action handling logic.

        Args:
            actions: The incoming step actions to handle.

        Returns:
            The processed and handled actions.
        """
        # Convert the action from -1 to 1, to absolute position within the actuator limits
        actions.clamp_(-1.0, 1.0)
        actions = actions * self._scale + self._offset

        # Set target positions
        self.actuator_manager.control_dofs_position(actions, self.dofs_idx)

        return actions

    """
    Internal methods
    """

    def _get_dof_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Define the position limits for the DOFs
        """
        lower, upper = self.get_dofs_limits()
        is_set = [False] * self.num_actions
        dof_names = list[str](self.dofs.keys())
        for pattern, value in self._limit_cfg.items():
            found = False
            for i, name in enumerate[str](dof_names):
                if not is_set[i] and re.match(f"^{pattern}$", name):
                    is_set[i] = True
                    lower[i] = value[0]
                    upper[i] = value[1]
                    found = True
            if not found:
                raise RuntimeError(f"Joint DOF '{pattern}' not found.")
        return lower, upper
