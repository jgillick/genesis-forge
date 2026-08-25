from __future__ import annotations

import genesis as gs
import torch

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.action.affine_dof_action_manager import (
    AffineDofActionManager,
)
from genesis_forge.managers.actuator import ActuatorManager
from genesis_forge.values import ensure_dof_pattern


class VelocityActionManager(AffineDofActionManager):
    """
    Converts actions to DOF target velocities, using affine transformations (scale and offset).

    .. math::

       velocity = offset + scaling * action

    Args:
        env: The environment to manage the DOF actuators for.
        clip: Clip the action values to this velocity range. Either a single ``(min, max)`` tuple applied
              to every controlled DOF, or a dict of ``{<joint name or regex>: (min, max)}``.
              A DOF not covered by the dict (or when `clip` is omitted entirely) is left
              unbounded.
        actuator_manager: The actuator manager which is used to setup and control the DOF joints.
        actuator_joints: Which joints of the actuator manager that this action manager will control.
                         These can be full names or regular expressions.
        scale: How much to scale the action.
        offset: Offset factor for the action.
        delay_step: The number of steps to delay the actions for.
                    This is an easy way to emulate the latency in the system.

    Example::

        class MyEnv(ManagedEnvironment):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # ...define scene and robot...

            def config(self):
                self.actuator_manager = ActuatorManager(
                    self,
                    joint_names=["wheel1", "wheel2"],
                    kv=5.0,
                )
                self.action_manager = VelocityActionManager(
                    self,
                    clip=(-16.0, 16.0),
                    actuator_manager=self.actuator_manager,
                )
    """

    def __init__(
        self,
        env: GenesisEnv,
        clip: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
        actuator_manager: ActuatorManager | None = None,
        actuator_joints: list[str] | str = ".*",
        scale: float | dict[str, float] = 1.0,
        offset: float | dict[str, float] = 0.0,
        delay_step: int = 0,
    ):
        super().__init__(
            env,
            delay_step=delay_step,
            actuator_manager=actuator_manager,
            actuator_joints=actuator_joints,
        )
        self._clip_cfg = ensure_dof_pattern(clip if clip is not None else {})
        self._scale_cfg = ensure_dof_pattern(scale)
        self._offset_cfg = ensure_dof_pattern(offset)

    """
    Lifecycle Operations
    """

    def build(self):
        """
        Builds the manager and initializes all the buffers.
        """
        super().build()

        # Clip value
        if self._clip_cfg is not None:
            # Default clip values to +/-inf (unbounded)
            self._clip_values = torch.full(
                (self.num_actions, 2), float("inf"), device=gs.device, dtype=gs.tc_float
            )
            self._clip_values[:, 0] = float("-inf")

            self._get_dof_value_tensor(
                self._clip_cfg,
                output=self._clip_values,
            )

        # Scale
        if self._scale_cfg is not None:
            self._scale_values = self._get_dof_value_tensor(
                self._scale_cfg, default_value=1.0
            )

        # Offset
        if self._offset_cfg is not None:
            self._offset_values = self._get_dof_value_tensor(
                self._offset_cfg, default_value=0.0
            )

    def send_actions_to_simulation(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Sends the actions as velocity commands to the actuators in the simulation.
        """
        self.actuator_manager.control_dofs_velocity(self.get_actions(), self.dofs_idx)
