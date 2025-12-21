from __future__ import annotations
import re
import torch
import genesis as gs
import numpy as np
from gymnasium import spaces
from typing import Any, Callable, TypeVar
from deprecated import deprecated
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.action.base import BaseActionManager
from genesis_forge.values import ensure_dof_pattern
from genesis_forge.managers.actuator import ActuatorManager


T = TypeVar("T")


class ForceActionManager(BaseActionManager):
    """
    Converts actions to DOF forces, using affine transformations (scale).

    .. math::

       force = scaling * action

    Args:
        env: The environment to manage the DOF actuators for.
        actuator_manager: The actuator manager which is used to setup and control the DOF joints.
        actuator_filter: Which joints of the actuator manager that this action manager will control.
                   These can be full names or regular expressions.
        scale: How much to scale the action.
        clip: Clip the action values to the range. If omitted, the action values will automatically be clipped to the joint limits.
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
                    joint_names=".*",
                    default_pos={".*": 0.0},
                    kp=50,
                    kv=0.5,
                    max_force=8.0,
                )
                self.action_manager = ForceActionManager(
                    self,
                    scale=0.5,
                    actuator_manager=self.actuator_manager,
                )

    Example using the manager directly::

        class MyEnv(GenesisEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # ...define scene and robot...

                self.actuator_manager = ActuatorManager(
                    self,
                    joint_names=".*",
                    default_pos={".*": 0.0},
                    kp=50,
                    kv=0.5,
                    max_force=8.0,
                )
                self.action_manager = ForceActionManager(
                    self,
                    scale=0.5,
                    use_default_offset=True,
                )

            def build(self):
                super().build()
                self.actuator_manager.build()
                self.action_manager.build()

            step(self, actions: torch.Tensor) -> None:
                super().step(actions)
                self.action_manager.step(actions)

                # ...do other step things...

            reset(self, envs_idx: list[int] = None) -> None:
                super().reset(envs_idx)
                self.actuator_manager.reset(envs_idx)
                self.action_manager.reset(envs_idx)

                # ...do other reset things...


    """

    def __init__(
        self,
        env: GenesisEnv,
        actuator_manager: ActuatorManager | None = None,
        actuator_filter: list[str] | str = ".*",
        scale: float | dict[str, float] = 1.0,
        clip: tuple[float, float] | dict[str, tuple[float, float]] = None,
        quiet_action_errors: bool = False,
        delay_step: int = 0,
        **kwargs,
    ):
        super().__init__(
            env,
            delay_step=delay_step,
            actuator_manager=actuator_manager,
            actuator_filter=actuator_filter,
            **kwargs,
        )
        self._scale_cfg = ensure_dof_pattern(scale)
        self._clip_cfg = ensure_dof_pattern(clip)
        self._quiet_action_errors = quiet_action_errors
        self._enabled_dof = None

        self._dofs_force_buffer: torch.Tensor = None

    """
    Lifecycle Operations
    """

    def build(self):
        """
        Builds the manager and initialized all the buffers.
        """
        super().build()

        # Define the clip values
        lower_limit, upper_limit = self.actuators.get_dofs_force_limits(self.dofs_idx)
        self._clip_values = torch.stack([lower_limit, upper_limit], dim=1)
        if self._clip_cfg is not None:
            self._get_dof_value_tensor(self._clip_cfg, output=self._clip_values)

        # Scale
        self._scale_values = None
        if self._scale_cfg is not None:
            self._scale_values = self._get_dof_value_tensor(self._scale_cfg)

    def step(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Take the incoming actions for this step and handle them.

        Args:
            actions: The incoming step actions to handle.
        """
        if not self.enabled:
            return
        actions = super().step(actions)
        self._actions = self.handle_actions(actions)
        return self._actions

    def handle_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Converts the actions to force commands, and send them to the DOF actuators.
        Override this function if you want to change the action handling logic.

        Args:
            actions: The incoming step actions to handle.

        Returns:
            The processed and handled actions.
        """

        # Validate actions
        if not self._quiet_action_errors:
            if torch.isnan(actions).any():
                print(f"ERROR: NaN actions received! Actions: {actions}")
            if torch.isinf(actions).any():
                print(f"ERROR: Infinite actions received! Actions: {actions}")

        # Process actions
        actions = actions * self._scale_values
        actions = torch.clamp(
            actions,
            min=self._clip_values[:, 0],
            max=self._clip_values[:, 1],
        )

        # Set target forces
        self.actuators.control_dofs_force(actions, self.dofs_idx)

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
