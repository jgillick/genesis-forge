from __future__ import annotations
import torch
from typing import Callable

from genesis_forge.genesis_env import GenesisEnv
from .position_action_manager import PositionActionManager, DofValue


class PositionWithinLimitsActionManager(PositionActionManager):
    """
    This is similar to `PositionActionManager` but converts actions from the range -1.0 - 1.0 to DOF positions within the limits of the actuators.

    Args:
        env: The environment to manage the DOF actuators for.
        joint_names: The joint names to manage.
        default_pos: The default DOF positions.
        pd_kp: The PD kp values.
        pd_kv: The PD kv values.
        max_force: The max force values.
        damping: The damping values.
        stiffness: The stiffness values.
        frictionloss: The frictionloss values.
        reset_random_scale: Scale all DOF values on reset by this amount +/-.
        action_handler: A function to handle the actions.
        quiet_action_errors: Whether to quiet action errors.
        randomization_cfg: The randomization configuration used to randomize the DOF values across all environments and between resets.
        resample_randomization_s: The time interval to resample the randomization values.
        delay_step: The number of steps to delay the actions for.
                    This is an easy way to emulate the latency in the system.

    Example::

        class MyEnv(ManagedEnvironment):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def config(self):
                self.action_manager = PositionalActionManager(
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
                    pd_kp={".*": 50},
                    pd_kv={".*": 0.5},
                    max_force={".*": 8.0},
                )

            @property
            def action_space(self):
                return self.action_manager.action_space

    """

    def __init__(
        self,
        env: GenesisEnv,
        joint_names: list[str] | str = ".*",
        default_pos: DofValue = {".*": 0.0},
        pd_kp: DofValue = None,
        pd_kv: DofValue = None,
        max_force: DofValue = None,
        damping: DofValue = None,
        stiffness: DofValue = None,
        frictionloss: DofValue = None,
        noise_scale: float = 0.0,
        action_handler: Callable[[torch.Tensor], None] = None,
        quiet_action_errors: bool = False,
        delay_step: int = 0,
    ):
        super().__init__(
            env,
            joint_names=joint_names,
            default_pos=default_pos,
            pd_kp=pd_kp,
            pd_kv=pd_kv,
            max_force=max_force,
            damping=damping,
            stiffness=stiffness,
            frictionloss=frictionloss,
            noise_scale=noise_scale,
            action_handler=action_handler,
            quiet_action_errors=quiet_action_errors,
            delay_step=delay_step,
        )

        _pos_limit_lower: torch.Tensor = None
        _pos_limit_upper: torch.Tensor = None

    """
    Operations
    """

    def build(self):
        """
        Builds the manager and initialized all the buffers.
        """
        super().build()

        dofs_idx = list(self._enabled_dof.values())
        lower, upper = self.env.robot.get_dofs_limit(dofs_idx)
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
        self.env.robot.control_dofs_position(self._actions, self.dofs_idx)

        return self._actions
