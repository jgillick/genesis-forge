import re
import torch
import genesis as gs
import numpy as np
from gymnasium import spaces
from typing import Any, Callable

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager
from genesis_forge.managers.action.base import BaseActionManager

DofValue = dict[str, float] | float
"""Mapping of DOF name (literal or regex) to value."""


def ensure_dof_pattern(value: DofValue) -> dict[str, Any] | None:
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


class PositionalActionManager(BaseActionManager):
    """
    Manages converting the actions to actuator positions.

    This manager converts actions from the range -1.0 - 1.0 to DOF positions within the limits of the actuators.

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

    _default_dofs_pos: torch.Tensor = None
    _dofs_pos_buffer: torch.Tensor = None
    _pos_limit_lower: torch.Tensor = None
    _pos_limit_upper: torch.Tensor = None

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
    ):
        super().__init__(env)
        self._has_initialized = False
        self._default_pos_cfg = ensure_dof_pattern(default_pos)
        self._pd_kp_cfg = ensure_dof_pattern(pd_kp)
        self._pd_kv_cfg = ensure_dof_pattern(pd_kv)
        self._max_force_cfg = ensure_dof_pattern(max_force)
        self._damping_cfg = ensure_dof_pattern(damping)
        self._stiffness_cfg = ensure_dof_pattern(stiffness)
        self._frictionloss_cfg = ensure_dof_pattern(frictionloss)
        self._quiet_action_errors = quiet_action_errors
        self._action_handler = action_handler
        self._enabled_dof = None
        self._noise_scale = noise_scale if self.env.mode != "play" else 0.0

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
    def num_actions(self) -> int:
        """
        Get the number of actions.
        """
        assert (
            self._enabled_dof is not None
        ), "PositionalActionManager not initialized. You may need to add <PositionalActionManager>.reset() in your environment's reset method."

        return len(self._enabled_dof)

    @property
    def action_space(self) -> tuple[float, float]:
        """
        If using the default action handler, the action space is [-1, 1].
        """
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.num_actions,),
            dtype=np.float32,
        )

    @property
    def dofs_idx(self) -> list[int]:
        """
        Get the indices of the DOF that are enabled (via joint_names).
        """
        return list(self._enabled_dof.values())

    @property
    def default_dofs_pos(self) -> torch.Tensor:
        """Return the default DOF positions."""
        return self._default_dofs_pos

    """
    DOF Getters
    """

    def get_dofs_position(self, noise: float = 0.0):
        """Return the position of the enabled DOFs."""
        pos = self.env.robot.get_dofs_position(self.dofs_idx)
        if noise > 0.0:
            pos = self._add_random_noise(pos, noise)
        return pos

    def get_dofs_velocity(self, noise: float = 0.0, clip: tuple[float, float] = None):
        """Return the velocity of the enabled DOFs."""
        vel = self.env.robot.get_dofs_velocity(self.dofs_idx)
        if noise > 0.0:
            vel = self._add_random_noise(vel, noise)
        if clip is not None:
            vel = vel.clamp(**clip)
        return vel

    def get_dofs_force(self, noise: float = 0.0, clip_to_max_force: bool = False):
        """Return the force of the enabled DOFs."""
        force = self.env.robot.get_dofs_force(self.dofs_idx)
        if noise > 0.0:
            force = self._add_random_noise(force, noise)
        if clip_to_max_force:
            force = force.clamp(self._force_range[0], self._force_range[1])
        return force

    """
    Operations
    """

    def build(self):
        """Initialize the buffers."""
        self._init_buffers()

    def step(self, actions: torch.Tensor) -> None:
        """
        Convert the actions into DOF positions and set the DOF actuators.
        """
        if not self.enabled:
            return
        if self._action_handler is not None:
            self._action_handler(actions)
        else:
            self._step_action_handler(actions)

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

    def _init_buffers(self):
        """Define the buffers for the DOF values."""
        self._enabled_dof = dict()
        for joint in self.env.robot.joints:
            if joint.type != gs.JOINT_TYPE.REVOLUTE:
                continue
            name = joint.name
            for pattern in self._joint_name_cfg:
                if re.match(pattern, name):
                    self._enabled_dof[name] = joint.dof_start
                    break
        dofs_idx = list(self._enabled_dof.values())

        # Get position Limits and convert to shape (num_envs, limit)
        lower, upper = self.env.robot.get_dofs_limit(dofs_idx)
        self._pos_limit_lower = lower.unsqueeze(0).expand(self.env.num_envs, -1)
        self._pos_limit_upper = upper.unsqueeze(0).expand(self.env.num_envs, -1)

        # Set DOF values
        self._kp_values = None
        self._kv_values = None
        self._damping_values = None
        self._stiffness_values = None
        self._frictionloss_values = None
        self._max_force_values = None
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

        # Default DOF positions
        if self._default_pos_cfg is not None:
            self._default_dofs_pos = self._get_dof_value_tensor(self._default_pos_cfg)
            self._default_dofs_pos = self._default_dofs_pos.unsqueeze(0).expand(
                self.env.num_envs, -1
            )

    def _step_action_handler(self, actions: torch.Tensor):
        """Convert actions to position commands, and send them to the DOF actuators."""

        # Validate actions
        if not self._quiet_action_errors:
            if torch.isnan(actions).any():
                print(f"ERROR: NaN actions received! Actions: {actions}")
            if torch.isinf(actions).any():
                print(f"ERROR: Infinite actions received! Actions: {actions}")

        actions = actions.clamp(-1.0, 1.0)
        self.actions = actions

        # Convert the action from -1 to 1, to absolute position within the actuator limits
        lower = self._pos_limit_lower
        upper = self._pos_limit_upper
        offset = (upper + lower) * 0.5
        target_positions = actions * (upper - lower) * 0.5 + offset

        # Set target positions
        self.env.robot.control_dofs_position(target_positions, self.dofs_idx)

    def _get_dof_value_array(self, values: DofValue) -> list[Any]:
        """
        Given a DofValue dict, loop over the entries, and set the value to the DOF indices (from dofs_idx) that match the pattern.

        Args:
            values: The DOF value to convert (for example: `{".*": 50}`).

        Returns:
            A list of values for the DOF indices.
            For example, for 4 DOFs: [50, 50, 50, 50]
        """
        is_set = [False] * self.num_actions
        value_arr = [0.0] * self.num_actions
        for pattern, value in values.items():
            for i, name in enumerate(self._enabled_dof.keys()):
                if not is_set[i] and re.match(pattern, name):
                    is_set[i] = True
                    value_arr[i] = value
        return value_arr

    def _get_dof_value_tensor(self, values: DofValue) -> torch.Tensor:
        """
        Wrapper for _get_dof_value_array that returns a tensor.
        """
        values = self._get_dof_value_array(values)
        return torch.tensor(values, device=gs.device, dtype=gs.tc_float)

    def _add_random_noise(
        self, values: torch.Tensor, noise_scale: float = 0.0
    ) -> torch.Tensor:
        """
        Add random noise to the tensor values
        """
        noise_value = torch.empty_like(values).uniform_(-1, 1) * noise_scale
        return values + noise_value
