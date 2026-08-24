from __future__ import annotations

import re
from typing import Any, TypeVar

import genesis as gs
import torch

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.action.base import BaseActionManager
from genesis_forge.managers.actuator import ActuatorManager
from genesis_forge.values import ensure_dof_pattern

T = TypeVar("T")


class VelocityActionManager(BaseActionManager):
    """
    Converts actions to DOF target velocities, using affine transformations (scale and offset).

    .. math::

       velocity = offset + scaling * action

    Unlike `PositionActionManager`, this manager has no notion of a "default" value to
    offset from, and does not fall back to the actuator's DOF position limits for clipping --
    continuously-rotating DOFs (e.g. wheels) typically report unbounded position limits in
    Genesis, which would silently disable clipping (or produce NaN clip bounds if a
    soft-limit-style scale factor were ever applied). `clip` is therefore a required argument,
    and every controlled DOF must be covered by it.

    Args:
        env: The environment to manage the DOF actuators for.
        clip: Clip the action values to this range. Either a single ``(min, max)`` tuple applied
              to every controlled DOF, or a dict of ``{<joint name or regex>: (min, max)}``.
              Required -- there is no limits-based fallback.
        actuator_manager: The actuator manager which is used to setup and control the DOF joints.
        actuator_joints: Which joints of the actuator manager that this action manager will control.
                         These can be full names or regular expressions.
        scale: How much to scale the action.
        offset: Offset factor for the action.
        quiet_action_errors: Whether to quiet action errors.
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
                    joint_names=["wheel1_speed", "wheel2_speed", "wheel3_speed"],
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
        clip: tuple[float, float] | dict[str, tuple[float, float]],
        actuator_manager: ActuatorManager | None = None,
        actuator_joints: list[str] | str = ".*",
        scale: float | dict[str, float] = 1.0,
        offset: float | dict[str, float] = 0.0,
        quiet_action_errors: bool = False,
        delay_step: int = 0,
    ):
        super().__init__(
            env,
            delay_step=delay_step,
            actuator_manager=actuator_manager,
            actuator_joints=actuator_joints,
        )
        if clip is None:
            raise ValueError(
                "VelocityActionManager requires an explicit `clip` value -- continuously-rotating "
                "DOFs report unbounded position limits, so there is no safe limits-based fallback."
            )
        self._clip_cfg = ensure_dof_pattern(clip)
        self._scale_cfg = ensure_dof_pattern(scale)
        self._offset_cfg = ensure_dof_pattern(offset)
        self._quiet_action_errors = quiet_action_errors

        self._clip_values: torch.Tensor = None
        self._scale_values: torch.Tensor = None
        self._offset_values: torch.Tensor = None

    """
    Lifecycle Operations
    """

    def build(self):
        """
        Builds the manager and initializes all the buffers.
        """
        super().build()

        # `clip` has no limits-based fallback, so start from a NaN sentinel: any DOF the
        # clip config doesn't cover is caught explicitly here rather than silently unclipped.
        self._clip_values = torch.full(
            (self.num_actions, 2), float("nan"), device=gs.device, dtype=gs.tc_float
        )
        self._get_dof_value_tensor(self._clip_cfg, output=self._clip_values)
        if torch.isnan(self._clip_values).any():
            missing = [
                name
                for i, name in enumerate(self.dofs.keys())
                if torch.isnan(self._clip_values[i]).any()
            ]
            raise ValueError(
                f"VelocityActionManager `clip` does not cover DOF(s): {missing}. "
                "Every controlled DOF must have an explicit clip range."
            )

        self._scale_values = self._get_dof_value_tensor(
            self._scale_cfg, default_value=1.0
        )
        self._offset_values = self._get_dof_value_tensor(
            self._offset_cfg, default_value=0.0
        )

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Convert the actions to velocity commands, and clamp them to `clip`.

        Args:
            actions: The incoming step actions to handle.

        Returns:
            The actions as velocity commands.
        """
        if not self._quiet_action_errors:
            if torch.isnan(actions).any():
                print(f"ERROR: NaN actions received! Actions: {actions}")
            if torch.isinf(actions).any():
                print(f"ERROR: Infinite actions received! Actions: {actions}")

        actions = actions * self._scale_values + self._offset_values
        actions = torch.clamp(
            actions,
            min=self._clip_values[:, 0],
            max=self._clip_values[:, 1],
        )
        return actions

    def send_actions_to_simulation(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Sends the actions as velocity commands to the actuators in the simulation.
        """
        self.actuator_manager.control_dofs_velocity(self.get_actions(), self.dofs_idx)

    """
    Internal methods
    """

    def _get_dof_value_tensor(
        self,
        values: float | dict,
        default_value: T = 0.0,
        output: torch.Tensor | list[Any] | None = None,
    ) -> torch.Tensor:
        """
         Given a DofValue dict, loop over the entries, and set the value to the DOF indices (from the actuator) that match the pattern.

        Args:
            values: The DOF value to convert (for example: `{".*": 50}`).

        Returns:
            A list of values for the DOF indices.
            For example, for 4 DOFs: [50, 50, 50, 50]
        """
        is_set = [False] * self.num_actions
        dof_names = list(self.dofs.keys())
        if output is None:
            output = torch.zeros(
                self.num_actions, device=gs.device, dtype=gs.tc_float
            ).fill_(default_value)
        for pattern, value in values.items():
            found = False
            for i, name in enumerate[str](dof_names):
                if not is_set[i] and re.match(f"^{pattern}$", name):
                    if isinstance(value, (list, tuple)):
                        value = torch.tensor(value, device=gs.device)
                    is_set[i] = True
                    output[i] = value
                    found = True
            if not found:
                raise RuntimeError(f"Joint DOF '{pattern}' not found.")
        return output
