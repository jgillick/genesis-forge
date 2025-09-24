from __future__ import annotations
import re
import torch
import genesis as gs
import numpy as np
from gymnasium import spaces
from typing import Any, Callable, TypeVar

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager
from genesis_forge.managers.action.base import BaseActionManager

T = TypeVar("T")
DofValue = dict[str, T] | T
"""Mapping of DOF name (literal or regex) to value."""


def _ensure_dof_pattern(value: DofValue) -> dict[str, Any] | None:
    """
    Ensures the value is a dictionary in the form: {<joint name or regex>: <value>}.

    Example:
        >>> ensure_dof_pattern(50)
        {".*": 50}
        >>> ensure_dof_pattern({".*": 50})
        {".*": 50}
        >>> ensure_dof_pattern({"knee_joint": 50})
        {"knee_joint": 50}

    Args:
        value: The value to convert.

    Returns:
        A dictionary of DOF name pattern to value.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    return {".*": value}


class PositionActionManager(BaseActionManager):
    """
    Converts actions to DOF positions, using affine transformations (scale and offset).

    .. math::

       position = offset + scaling * action

    If `use_default_offset` is `True`, the `offset` will be set to the `default_pos` value for each DOF/joint.

    Args:
        env: The environment to manage the DOF actuators for.
        joint_names: The joint names to manage.
        default_pos: The default DOF positions.
        scale: How much to scale the action.
        offset: Offset factor for the action.
        use_default_offset: Whether to use default joint positions configured in the articulation asset as offset. Defaults to True.
        clip: Clip the action values to the range. If omitted, the action values will automatically be clipped to the joint limits.
        pd_kp: The PD kp values.
        pd_kv: The PD kv values.
        max_force: The max force values.
        damping: The damping values.
        stiffness: The stiffness values.
        frictionloss: The frictionloss values.
        reset_random_scale: Scale all DOF values on reset by this amount +/-.
        quiet_action_errors: Whether to quiet action errors.
        randomization_cfg: The randomization configuration used to randomize the DOF values across all environments and between resets.
        resample_randomization_s: The time interval to resample the randomization values.

    Example::

        class MyEnv(ManagedEnvironment):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # ...define scene and robot...

            def config(self):
                self.action_manager = PositionActionManager(
                    self,
                    joint_names=".*",
                    scale=0.5,
                    use_default_offset=True,
                    default_pos={
                        # Hip joints
                        "Leg[1-2]_Hip": -1.0,
                        "Leg[3-4]_Hip": 1.0,
                        # Femur joints
                        "Leg[1-4]_Femur": 0.5,
                        # Tibia joints
                        "Leg[1-4]_Tibia": 0.6,
                    },
                    pd_kp=50,
                    pd_kv=0.5,
                    max_force=8.0,
                )

    Example using the manager directly::

        class MyEnv(GenesisEnv):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                # ...define scene and robot...

            def config(self):
                self.action_manager = PositionActionManager(
                    self,
                    joint_names=".*",
                    action_scale=0.5,
                    action_offset=0.0,
                    use_default_offset=True,
                    default_pos={
                        # Hip joints
                        "Leg[1-2]_Hip": -1.0,
                        "Leg[3-4]_Hip": 1.0,
                        # Femur joints
                        "Leg[1-4]_Femur": 0.5,
                        # Tibia joints
                        "Leg[1-4]_Tibia": 0.6,
                    },
                    pd_kp=50,
                    pd_kv=0.5,
                    max_force=8.0,
                )

            def build(self):
                super().build()
                config()

            step(self, actions: torch.Tensor) -> None:
                super().step(actions)
                self.action_manager.step(actions)

                # ...do other step things...

            reset(self, envs_idx: list[int] = None) -> None:
                super().reset(envs_idx)
                self.action_manager.reset(envs_idx)

                # ...do other reset things...


    """

    def __init__(
        self,
        env: GenesisEnv,
        joint_names: list[str] | str = ".*",
        default_pos: DofValue[float] = {".*": 0.0},
        scale: DofValue[float] = 1.0,
        clip: DofValue[tuple[float, float]] = None,
        offset: DofValue[float] = 0.0,
        use_default_offset: bool = True,
        pd_kp: DofValue[float] = None,
        pd_kv: DofValue[float] = None,
        max_force: DofValue[float | tuple[float, float]] = None,
        damping: DofValue[float] = None,
        stiffness: DofValue[float] = None,
        frictionloss: DofValue[float] = None,
        noise_scale: float = 0.0,
        action_handler: Callable[[torch.Tensor], None] = None,
        quiet_action_errors: bool = False,
    ):
        super().__init__(env)
        self._has_initialized = False
        self._default_pos_cfg = _ensure_dof_pattern(default_pos)
        self._offset_cfg = _ensure_dof_pattern(offset)
        self._scale_cfg = _ensure_dof_pattern(scale)
        self._clip_cfg = _ensure_dof_pattern(clip)
        self._pd_kp_cfg = _ensure_dof_pattern(pd_kp)
        self._pd_kv_cfg = _ensure_dof_pattern(pd_kv)
        self._max_force_cfg = _ensure_dof_pattern(max_force)
        self._damping_cfg = _ensure_dof_pattern(damping)
        self._stiffness_cfg = _ensure_dof_pattern(stiffness)
        self._frictionloss_cfg = _ensure_dof_pattern(frictionloss)
        self._quiet_action_errors = quiet_action_errors
        self._enabled_dof = None
        self._noise_scale = noise_scale
        self._use_default_offset = use_default_offset

        self._default_dofs_pos: torch.Tensor = None
        self._dofs_pos_buffer: torch.Tensor = None

        if use_default_offset and offset != 0.0:
            raise ValueError("Cannot set both use_default_offset and offset")

        if isinstance(joint_names, str):
            self._joint_name_cfg = [joint_names]
        elif isinstance(joint_names, list):
            self._joint_name_cfg = joint_names
        else:
            raise TypeError(f"Invalid joint_names type: {type(joint_names)}")

    """
    Properties
    """

    @property
    def action_space(self) -> tuple[float, float]:
        """
        Returns the actions space for the environment, based on the number of DOFs defined in this action manager.
        """
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.num_actions,),
            dtype=np.float32,
        )

    @property
    def num_actions(self) -> int:
        """
        Get the number of actions.
        """
        assert (
            self._enabled_dof is not None
        ), "PositionalActionManager not initialized. You may need to add <PositionalActionManager>.reset() in your environment's reset method."

        return len(self._enabled_dof)

    @property
    def dofs_idx(self) -> list[int]:
        """
        Get the indices of the DOF that are enabled (via joint_names).
        """
        return list(self._enabled_dof.values())

    @property
    def default_dofs_pos(self) -> torch.Tensor:
        """
        Return the default DOF positions.
        """
        return self._default_dofs_pos

    """
    DOF Getters
    """

    def get_dofs_position(self, noise: float = 0.0):
        """
        Return the current position of the enabled DOFs.
        This is a wrapper for `RigidEntity.get_dofs_position`.

        Args:
            noise: The maximum amount of random noise to add to the position values returned.
        """
        pos = self.env.robot.get_dofs_position(self.dofs_idx)
        if noise > 0.0:
            pos = self._add_random_noise(pos, noise)
        return pos

    def get_dofs_velocity(self, noise: float = 0.0, clip: tuple[float, float] = None):
        """
        Return the current velocity of the enabled DOFs.
        This is a wrapper for `RigidEntity.get_dofs_velocity`.

        Args:
            noise: The maximum amount of random noise to add to the velocity values returned.
            clip: Clip the velocity returned.
        """
        vel = self.env.robot.get_dofs_velocity(self.dofs_idx)
        if noise > 0.0:
            vel = self._add_random_noise(vel, noise)
        if clip is not None:
            vel = vel.clamp(**clip)
        return vel

    def get_dofs_force(self, noise: float = 0.0, clip_to_max_force: bool = False):
        """
        Return the force experienced by the enabled DOFs.
        This is a wrapper for `RigidEntity.get_dofs_force`.

        Args:
            noise: The maximum amount of random noise to add to the force values returned.
            clip_to_max_force: Clip the force returned to the maximum force defined by the `max_force` parameter.

        Returns:
            The force experienced by the enabled DOFs.
        """
        force = self.env.robot.get_dofs_force(self.dofs_idx)
        if noise > 0.0:
            force = self._add_random_noise(force, noise)
        if clip_to_max_force and self._force_range is not None:
            force = force.clamp(self._force_range[0], self._force_range[1])
        return force

    """
    Operations
    """

    def build(self):
        """
        Builds the manager and initialized all the buffers.
        """

        # Find all enabled joints by names/patterns
        self._enabled_dof = dict()
        for joint in self.env.robot.joints:
            if joint.type != gs.JOINT_TYPE.REVOLUTE:
                continue
            name = joint.name
            for pattern in self._joint_name_cfg:
                if re.match(f"^{pattern}$", name):
                    self._enabled_dof[name] = joint.dof_start
                    break

        # Default DOF positions
        if self._default_pos_cfg is not None:
            self._default_dofs_pos = self._get_dof_value_tensor(self._default_pos_cfg)
        else:
            self._default_dofs_pos = torch.zeros(self.num_actions, device=gs.device)
        self._default_dofs_pos = self._default_dofs_pos.unsqueeze(0).expand(
            self.env.num_envs, -1
        )

        # Get the joint limits
        lower_limit, upper_limit = self.env.robot.get_dofs_limit(self.dofs_idx)

        # Map config params to the DOF indices
        self._scale_values = None
        self._offset_values = None
        self._kp_values = None
        self._kv_values = None
        self._damping_values = None
        self._stiffness_values = None
        self._frictionloss_values = None
        self._clip_values = torch.stack([lower_limit, upper_limit], dim=1)
        if self._scale_cfg is not None:
            self._scale_values = self._get_dof_value_tensor(self._scale_cfg)
        if self._clip_cfg is not None:
            self._get_dof_value_tensor(self._clip_cfg, output=self._clip_values)
        if self._pd_kp_cfg is not None:
            self._kp_values = self._get_dof_value_tensor(self._pd_kp_cfg)
        if self._pd_kv_cfg is not None:
            self._kv_values = self._get_dof_value_tensor(self._pd_kv_cfg)
        if self._damping_cfg is not None:
            self._damping_values = self._get_dof_value_tensor(self._damping_cfg)
        if self._stiffness_cfg is not None:
            self._stiffness_values = self._get_dof_value_tensor(self._stiffness_cfg)
        if self._frictionloss_cfg is not None:
            self._frictionloss_values = self._get_dof_value_tensor(
                self._frictionloss_cfg
            )
        if self._use_default_offset:
            self._offset_values = self._default_dofs_pos
        else:
            offset = self._offset_cfg if self._offset_cfg is not None else 0.0
            self._offset_values = self._get_dof_value_tensor(offset)

        # Max force
        # The value can either be a single float or a tuple range
        self._force_range = None
        if self._max_force_cfg is not None:
            max_force = self._get_dof_value_array(self._max_force_cfg)

            # Convert values to upper and lower arrays
            force_upper = [0.0] * self.num_actions
            force_lower = [0.0] * self.num_actions
            for i, value in enumerate(max_force):
                if isinstance(max_force[0], (list, tuple)):
                    force_lower[i] = value[0]
                    force_upper[i] = value[1]
                else:
                    force_lower[i] = -value
                    force_upper[i] = value

            self._force_range = (
                torch.tensor(force_lower, device=gs.device),
                torch.tensor(force_upper, device=gs.device),
            )

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
        Converts the actions to position commands, and send them to the DOF actuators.
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
        actions = actions * self._scale_values + self._offset_values
        actions = torch.clamp(
            actions,
            min=self._clip_values[:, 0],
            max=self._clip_values[:, 1],
        )

        # Set target positions
        self.env.robot.control_dofs_position(actions, self.dofs_idx)

        return actions

    def reset(
        self,
        envs_idx: list[int] = None,
    ):
        """Reset the DOF positions."""
        if not self.enabled:
            return
        if envs_idx is None:
            envs_idx = torch.arange(self.env.num_envs, device=gs.device)

        # Set DOF values with random scaling
        if self._kp_values is not None:
            kp = self._add_random_noise(self._kp_values, self._noise_scale)
            self.env.robot.set_dofs_kp(kp, self.dofs_idx, envs_idx)
        if self._kv_values is not None:
            kv = self._add_random_noise(self._kv_values, self._noise_scale)
            self.env.robot.set_dofs_kv(kv, self.dofs_idx, envs_idx)
        if self._damping_values is not None:
            damping = self._add_random_noise(self._damping_values, self._noise_scale)
            self.env.robot.set_dofs_damping(damping, self.dofs_idx, envs_idx)
        if self._stiffness_values is not None:
            stiffness = self._add_random_noise(
                self._stiffness_values, self._noise_scale
            )
            self.env.robot.set_dofs_stiffness(stiffness, self.dofs_idx, envs_idx)
        if self._frictionloss_values is not None:
            frictionloss = self._add_random_noise(
                self._frictionloss_values, self._noise_scale
            )
            self.env.robot.set_dofs_frictionloss(frictionloss, self.dofs_idx, envs_idx)
        if self._force_range is not None:
            lower = self._add_random_noise(self._force_range[0], self._noise_scale)
            upper = self._add_random_noise(self._force_range[1], self._noise_scale)
            self.env.robot.set_dofs_force_range(lower, upper, self.dofs_idx, envs_idx)

        # Reset DOF positions with random scaling
        position = self._add_random_noise(
            self._default_dofs_pos[envs_idx], self._noise_scale
        )
        self.env.robot.set_dofs_position(
            position=position,
            dofs_idx_local=self.dofs_idx,
            envs_idx=envs_idx,
        )

    """
    Implementation
    """

    def _get_dof_value_array(
        self,
        values: DofValue[T],
        default_value: T = 0.0,
        output: torch.Tensor | list[Any] | None = None,
    ) -> torch.Tensor | list[Any]:
        """
        Given a DofValue dict, loop over the entries, and set the value to the DOF indices (from dofs_idx) that match the pattern.

        Args:
            values: The DOF value to convert (for example: `{".*": 50}`).

        Returns:
            A list of values for the DOF indices.
            For example, for 4 DOFs: [50, 50, 50, 50]
        """
        is_set = [False] * self.num_actions
        if output is None:
            output = [default_value] * self.num_actions
        for pattern, value in values.items():
            found = False
            for i, name in enumerate(self._enabled_dof.keys()):
                if not is_set[i] and re.match(f"^{pattern}$", name):
                    if isinstance(output, torch.Tensor) and not isinstance(
                        value, torch.Tensor
                    ):
                        value = torch.tensor(value, device=gs.device)
                    is_set[i] = True
                    output[i] = value
                    found = True
            if not found:
                raise RuntimeError(f"Joint DOF '{pattern}' not found.")
        return output

    def _get_dof_value_tensor(
        self,
        values: DofValue[T],
        default_value: T = 0.0,
        output: torch.Tensor | list[Any] | None = None,
    ) -> torch.Tensor:
        """
        Wrapper for _get_dof_value_array that returns a tensor.
        """
        values = self._get_dof_value_array(values, default_value, output)
        return torch.tensor(values, device=gs.device, dtype=gs.tc_float)

    def _add_random_noise(
        self, values: torch.Tensor, noise_scale: float = 0.0
    ) -> torch.Tensor:
        """
        Add random noise to the tensor values
        """
        if noise_scale == 0.0:
            return values
        noise_value = torch.empty_like(values).uniform_(-1, 1) * noise_scale
        return values + noise_value
