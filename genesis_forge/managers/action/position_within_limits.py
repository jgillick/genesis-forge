from __future__ import annotations
import torch
from typing import Callable

from genesis_forge.genesis_env import GenesisEnv
from .position_action_manager import PositionActionManager
from genesis_forge.managers.actuator import ActuatorManager


class PositionWithinLimitsActionManager(PositionActionManager):
    """
    This is similar to `PositionActionManager` but converts actions from the range -1.0 - 1.0 to DOF positions within the limits of the actuators.

    Args:
        env: The environment to manage the DOF actuators for.
        actuator_manager: The actuator manager which is used to setup and control the DOF joints.
        action_handler: A function to handle the actions.
        quiet_action_errors: Whether to quiet action errors.
        delay_step: The number of steps to delay the actions for.
                    This is an easy way to emulate the latency in the system.

    Example::

        class MyEnv(ManagedEnvironment):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def config(self):
                self.actuator_manager = ActuatorManager(
                    self,
                    joint_names=".*",
                    default_pos={
                        # Hip joints
                        "Leg[1-2]_Hip": -1.0,
                        "Leg[3-4]_Hip": 1.0,
                        # Femur joints
                        "Leg[1-4]_Femur": 0.5,
                        # Tibia joints
                        "Leg[1-4]_Tibia": 0.6,
                    },
                    kp={".*": 50},
                    kv={".*": 0.5},
                    max_force={".*": 8.0},
                )
                self.action_manager = PositionalActionManager(
                    self,
                    actuator_manager=self.actuator_manager,
                )

    """

    def __init__(
        self,
        env: GenesisEnv,
        actuator_manager: ActuatorManager | None = None,
        action_handler: Callable[[torch.Tensor], None] = None,
        quiet_action_errors: bool = False,
        delay_step: int = 0,
        **kwargs,
    ):
        super().__init__(
            env,
            action_handler=action_handler,
            quiet_action_errors=quiet_action_errors,
            delay_step=delay_step,
            actuator_manager=actuator_manager,
            **kwargs,
        )

    """
    Lifecycle Operations
    """

    def build(self):
        """
        Builds the manager and initialized all the buffers.
        """
        lower, upper = self._actuator_manager.get_dofs_limits()
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
        self._actions = actions * self._scale + self._offset

        # Set target positions
        self._actuator_manager.control_dofs_position(self._actions)

        return self._actions
