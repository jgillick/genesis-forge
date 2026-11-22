from __future__ import annotations

import torch

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.action.affine_dof_action_manager import (
    AffineDofActionManager,
)
from genesis_forge.managers.actuator import ActuatorManager
from genesis_forge.values import ensure_dof_pattern


class TorqueActionManager(AffineDofActionManager):
    """
    Converts actions to DOF torques, using affine transformations (scale and offset).

    .. math::

       torque = offset + scaling * action

    The torques are applied directly to the joints, bypassing the PD controller, so the
    actuator manager's `kp` and `kv` gains have no effect on the DOFs this manager controls.

    Args:
        env: The environment to manage the DOF actuators for.
        clip: Clip the action values to this torque range. Either a single ``(min, max)`` tuple applied
              to every controlled DOF, or a dict of ``{<joint name or regex>: (min, max)}``.
              A DOF not defined is clipped to the actuator manager's `max_force` or the value defined in the robot model file.
        actuator_manager: The actuator manager which is used to setup and control the DOF joints.
        actuator_joints: Which joints of the actuator manager that this action manager will control.
                         These can be full names or regular expressions.
        action_groups: Drive several joints from a single action, for joints that are
                       mechanically tied together. See `BaseActionManager`.
        scale: How much to scale the action.
        offset: Offset factor for the action.
        delay_step: Steps to delay actions by, to emulate actuator/bus latency. This is
                    either a fixed step value or a min/max range to create random delays
                    from. See `ActionDelayBuffer`.

    Example::

        class MyEnv(ManagedEnvironment):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # ...define scene and robot...

            def config(self):
                self.actuator_manager = ActuatorManager(
                    self,
                    joint_names=[".*"],
                    max_force=8.0,
                )
                self.action_manager = TorqueActionManager(
                    self,
                    scale=4.0,  # clipped to max_force, since no clip is set
                    actuator_manager=self.actuator_manager,
                )
    """

    deploy_type: str = "torque"

    def __init__(
        self,
        env: GenesisEnv,
        clip: tuple[float, float] | dict[str, tuple[float, float]] | None = None,
        actuator_manager: ActuatorManager | None = None,
        actuator_joints: list[str] | str = ".*",
        action_groups: list[list[str] | str] | None = None,
        scale: float | dict[str, float] = 1.0,
        offset: float | dict[str, float] = 0.0,
        delay_step: int | tuple[int, int] = 0,
    ):
        super().__init__(
            env,
            delay_step=delay_step,
            actuator_manager=actuator_manager,
            actuator_joints=actuator_joints,
            action_groups=action_groups,
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

        # Clip value, defaulting to the actuator's max_force, or the simulator's force range
        lower, upper = self.actuator_manager.get_dofs_nominal_force_range()
        self._clip_values = torch.stack(
            [lower[self.actuator_dof_filter], upper[self.actuator_dof_filter]], dim=1
        )
        self._get_dof_value_tensor(self._clip_cfg, output=self._clip_values)

        self._scale_values = self._get_dof_value_tensor(
            self._scale_cfg, default_value=1.0
        )
        self._offset_values = self._get_dof_value_tensor(
            self._offset_cfg, default_value=0.0
        )

    def send_actions_to_simulation(self, actions: torch.Tensor) -> None:
        """
        Sends the actions as torque commands to the actuators in the simulation.
        """
        self.actuator_manager.control_dofs_force(actions, self.dofs_idx)
