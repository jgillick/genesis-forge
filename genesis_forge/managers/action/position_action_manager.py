from __future__ import annotations
import re
import torch
import genesis as gs
from typing import Any, TypeVar
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.action.base import BaseActionManager
from genesis_forge.values import ensure_dof_pattern
from genesis_forge.managers.actuator import ActuatorManager


T = TypeVar("T")


class PositionActionManager(BaseActionManager):
    """
    Converts actions to DOF positions, using affine transformations (scale and offset).

    .. math::

       position = offset + scaling * action

    If `use_default_offset` is `True`, the `offset` will be set to the `default_pos` value for each DOF/joint.

    Args:
        env: The environment to manage the DOF actuators for.
        actuator_manager: The actuator manager which is used to setup and control the DOF joints.
        actuator_joints: Which joints of the actuator manager that this action manager will control.
                         These can be full names or regular expressions.
        scale: How much to scale the action.
        offset: Offset factor for the action.
        use_default_offset: Whether to use default joint positions configured in the articulation asset as offset. Defaults to True.
        clip: Clip the action values to the range. If omitted, the action values will automatically be clipped to the joint limits.
        soft_limit_scale_factor: Scales the clip range of all limits by this factor around the midpoint 
                                 of each joint's limits to establish a safety region within the limits. 
                                 Defaults to 1.0.
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
                self.action_manager = PositionActionManager(
                    self,
                    scale=0.5,
                    use_default_offset=True,
                    actuator_manager=self.actuator_manager,
                    actuator_joints=[".*"], # optional joint filter
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
                self.action_manager = PositionActionManager(
                    self,
                    scale=0.5,
                    offset=0.0,
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
        actuator_joints: list[str] | str = ".*",
        scale: float | dict[str, float] = 1.0,
        offset: float | dict[str, float] = 0.0,
        clip: tuple[float, float] | dict[str, tuple[float, float]] = None,
        soft_limit_scale_factor: float = 1.0,
        use_default_offset: bool = True,
        quiet_action_errors: bool = False,
        delay_step: int = 0,
        **kwargs,
    ):
        super().__init__(
            env,
            delay_step=delay_step,
            actuator_manager=actuator_manager,
            actuator_joints=actuator_joints,
            **kwargs,
        )
        self._offset_cfg = ensure_dof_pattern(offset)
        self._scale_cfg = ensure_dof_pattern(scale)
        self._clip_cfg = ensure_dof_pattern(clip)
        self._soft_limit_scale_factor = soft_limit_scale_factor
        self._quiet_action_errors = quiet_action_errors
        self._enabled_dof = None
        self._use_default_offset = use_default_offset

        self._dofs_pos_buffer: torch.Tensor = None

        if use_default_offset and offset != 0.0:
            raise ValueError("Cannot set both use_default_offset and offset")

    """
    Properties
    """

    @property
    def default_dofs_pos(self) -> torch.Tensor:
        """
        Return the default DOF positions.
        """
        return self.actuator_manager.default_dofs_pos[:, self.actuator_dof_filter]

    """
    Lifecycle Operations
    """

    def build(self):
        """
        Builds the manager and initialized all the buffers.
        """
        super().build()

        # Define the clip values
        lower_limit, upper_limit = self.actuator_manager.get_dofs_limits(self.dofs_idx)
        self._clip_values = torch.stack([lower_limit, upper_limit], dim=1)
        if self._clip_cfg is not None:
            self._get_dof_value_tensor(self._clip_cfg, output=self._clip_values)
        if self._soft_limit_scale_factor != 1.0:
            midpoint = (self._clip_values[:, 0] + self._clip_values[:, 1]) * 0.5
            half_range = (self._clip_values[:, 1] - self._clip_values[:, 0]) * 0.5 * self._soft_limit_scale_factor
            self._clip_values[:, 0] = midpoint - half_range
            self._clip_values[:, 1] = midpoint + half_range

        # Scale
        self._scale_values = None
        if self._scale_cfg is not None:
            self._scale_values = self._get_dof_value_tensor(self._scale_cfg)

        # Offset
        self._offset_values = None
        if self._use_default_offset:
            self._offset_values = self.default_dofs_pos
        else:
            offset = self._offset_cfg if self._offset_cfg is not None else 0.0
            self._offset_values = self._get_dof_value_tensor(offset)

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Convert the actions to position commands, and clamp them to the limits.

        Args:
            actions: The incoming step actions to handle.

        Returns:
            The actions as position commands.
        """
        # Validate actions
        if not self._quiet_action_errors:
            if torch.isnan(actions).any():
                print(f"ERROR: NaN actions received! Actions: {actions}")
            if torch.isinf(actions).any():
                print(f"ERROR: Infinite actions received! Actions: {actions}")

        # Process actions
        actions = actions * self._scale_values + self._offset_values
        actions = torch.clamp(
            actions,
            min=self._clip_values[:, 0],
            max=self._clip_values[:, 1],
        )
        return actions

    def send_actions_to_simulation(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Sends the actions as position commands to the actuators in the simulation.
        """
        actions = self.get_actions()
        self.actuator_manager.control_dofs_position(actions, self.dofs_idx)

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
