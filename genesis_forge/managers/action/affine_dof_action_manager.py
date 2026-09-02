from __future__ import annotations

import re
from typing import Any, TypeVar

import genesis as gs
import torch

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.action.base import BaseActionManager
from genesis_forge.managers.actuator import ActuatorManager

T = TypeVar("T")


class AffineDofActionManager(BaseActionManager):
    """
    Base for action managers that convert actions to per-DOF values via an affine
    transform, followed by a clip.

    .. math::

       value = offset + scaling * action

    Shared by `PositionActionManager` and `VelocityActionManager`. A subclass's
    `build()` is responsible for populating `self._scale_values`,
    `self._offset_values`, and `self._clip_values` (each shape `(num_actions,)`,
    except `_clip_values` which is `(num_actions, 2)`).

    Args:
        env: The environment to manage the DOF actuators for.
        actuator_manager: The actuator manager which is used to setup and control the DOF joints.
        actuator_joints: Which joints of the actuator manager that this action manager will control.
                         These can be full names or regular expressions.
        action_groups: Drive several joints from a single action. See `BaseActionManager`.
        delay_step: The number of steps to delay the actions for.
                    This is an easy way to emulate the latency in the system.
    """

    def __init__(
        self,
        env: GenesisEnv,
        actuator_manager: ActuatorManager | None = None,
        actuator_joints: list[str] | str = ".*",
        action_groups: list[list[str] | str] | None = None,
        delay_step: int = 0,
    ):
        super().__init__(
            env,
            actuator_manager=actuator_manager,
            actuator_joints=actuator_joints,
            action_groups=action_groups,
            delay_step=delay_step,
        )
        self._scale_values: torch.Tensor | None = None
        self._offset_values: torch.Tensor | None = None
        self._clip_values: torch.Tensor | None = None

    """
    Lifecycle Operations
    """

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Apply the affine scale/offset transform and clip to `self._scale_values` /
        `self._offset_values` / `self._clip_values` (populated by a subclass's `build()`).

        Args:
            actions: The incoming step actions to handle.

        Returns:
            The processed and converted actions.
        """
        if self._scale_values is None or self._offset_values is None or self._clip_values is None:
            raise RuntimeError(
                "AffineDofActionManager: _scale_values, _offset_values, and _clip_values must be set by a subclass's build() before calling process_actions()"
            )

        # Validate actions.
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
            default_value: The value to fill unset DOFs with, when `output` is not provided.
            output: An existing tensor to fill in place, instead of allocating a new one.

        Returns:
            A list of values for the DOF indices.
            For example, for 4 DOFs: [50, 50, 50, 50]
        """
        is_set = [False] * self.num_dofs
        dof_names = list(self.dofs.keys())
        if output is None:
            output = torch.zeros(
                self.num_dofs, device=gs.device, dtype=gs.tc_float
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
