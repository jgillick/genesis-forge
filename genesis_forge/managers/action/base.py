import re
import torch
import numpy as np
from gymnasium import spaces
import genesis as gs
from deprecated import deprecated
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.actuator import ActuatorManager
from genesis_forge.managers.base import BaseManager

deprecated_arg_names = [
    "joint_names",
    "default_pos",
    "pd_kp",
    "pd_kv",
    "max_force",
    "damping",
    "stiffness",
    "frictionloss",
    "noise_scale",
]


class BaseActionManager(BaseManager):
    """
    Base for managers that handle actions.

    Args:
        env: The environment to manage the DOF actuators for.
        actuator_manager: The actuator manager which is used to setup and control the DOF joints.
        actuator_joints: Which joints of the actuator manager that this action manager will control.
                         These can be full names or regular expressions.
        delay_step: The number of steps to delay the actions for.
                    This is an easy way to emulate the latency in the system.
    """

    def __init__(
        self,
        env: GenesisEnv,
        actuator_manager: ActuatorManager | None = None,
        actuator_joints: list[str] | str = ".*",
        delay_step: int = 0,
        **kwargs,
    ):
        super().__init__(env, type="action")
        self._raw_actions = None
        self._actions = None
        self._delay_step = delay_step
        self._action_delay_buffer = []
        self._actuator_manager = actuator_manager
        self._actuator_joints = (
            [actuator_joints] if isinstance(actuator_joints, str) else actuator_joints
        )
        self._dofs: dict[int, str] = {}
        self._actuator_dof_filter: torch.Tensor = None

        # Deprecated actuator parameters
        deprecated_actuator_args = {
            key: kwargs[key] for key in deprecated_arg_names if key in kwargs
        }
        if len(deprecated_actuator_args) > 0:
            dep_list = ", ".join(deprecated_actuator_args.keys())
            if self._actuator_manager is not None:
                raise ValueError(
                    f"Cannot set both actuator_manager and deprecated actuator parameters: {dep_list}"
                )
            print(
                f"Actuator arguments are deprecated in the action manager, instead define an ActuatorManager ({dep_list})"
            )
            self._actuator_manager = ActuatorManager(
                env,
                joint_names=kwargs.get("joint_names", ".*"),
                default_pos=kwargs.get("default_pos", {".*": 0.0}),
                kp=kwargs.get("pd_kp", None),
                kv=kwargs.get("pd_kv", None),
                max_force=kwargs.get("max_force", None),
                damping=kwargs.get("damping", None),
                stiffness=kwargs.get("stiffness", None),
                frictionloss=kwargs.get("frictionloss", None),
                default_noise_scale=kwargs.get("noise_scale", 0.0),
            )
        if self._actuator_manager is None:
            raise ValueError("No ActuatorManager provided.")

    """
    Properties
    """

    @property
    def actuator_manager(self) -> ActuatorManager:
        """
        Get the actuator manager.
        """
        return self._actuator_manager

    @property
    @deprecated(version="0.4.0", reason="Use `actuator_manager` instead")
    def actuators(self) -> ActuatorManager:
        return self._actuator_manager

    @property
    def num_actions(self) -> int:
        """
        Get the number of actions.
        """
        return len(self.dofs_idx)

    @property
    def dofs_idx(self) -> list[int]:
        """
        Get the indices of the DOFs that this action manager controls.
        """
        return list[int](self._dofs.values())

    @property
    def dofs(self) -> dict[str, int]:
        """
        Get a dictionary of the DOF names and their indices
        """
        return self._dofs

    @property
    def actuator_dof_filter(self) -> torch.Tensor:
        """
        An index filter for the actuator DOF buffer values.
        """
        return self._actuator_dof_filter

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
    def actions(self) -> torch.Tensor:
        """
        The actions for for the current step.
        """
        if self._actions is None:
            return torch.zeros((self.env.num_envs, self.num_actions))
        return self._actions

    @property
    def raw_actions(self) -> torch.Tensor:
        """
        The actions received from the policy, before being converted.
        """
        if self._raw_actions is None:
            return torch.zeros((self.env.num_envs, self.num_actions))
        return self._raw_actions

    """
    DOF convenience wrappers
    """

    def get_dofs_position(self):
        """
        A wrapper for `RigidEntity.get_dofs_limits` that returns the position limits of the controlled DOFs.

        Returns:
            position: tuple[torch.Tensor, torch.Tensor]
                      The position of the DOFs managed by this action manager.
        """
        return self.actuator_manager.get_dofs_position(dofs_idx=self.dofs_idx)

    def get_dofs_limits(self):
        """
        A wrapper for `RigidEntity.get_dofs_limit` that returns the limits of the controlled DOFs.

        Returns:
            lower_limit: torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
                         The lower limit of the positional limits for the entity's dofs.
            upper_limit: torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
                         The upper limit of the positional limits for the entity's dofs.
        """
        return self.actuator_manager.get_dofs_limits(dofs_idx=self.dofs_idx)

    def get_dofs_velocity(self, clip: tuple[float, float] = None):
        """
        A wrapper for `RigidEntity.get_dofs_velocity` that returns the current velocity of the controlled DOFs.

        Args:
            clip: Range to clip the velocity to.

        Returns:
            velocity: torch.Tensor, shape (n_envs, n_dofs)
            The velocity of the enabled DOFs managed by this action manager.
        """
        return self.actuator_manager.get_dofs_velocity(
            clip=clip, dofs_idx=self.dofs_idx
        )

    def get_dofs_force(self, clip_to_max_force: bool = False):
        """
        A wrapper for `RigidEntity.get_dofs_force` that returns the force experienced by the controlled DOFs.

        Args:
            clip_to_max_force: Clip the force returned to the maximum force defined by the `max_force` parameter
                               defined in the actuator manager.

        Returns:
            force: torch.Tensor, shape (n_envs, n_dofs)
            The force experienced by the enabled DOFs.
        """
        return self.actuator_manager.get_dofs_force(
            clip_to_max_force=clip_to_max_force, dofs_idx=self.dofs_idx
        )

    """
    Lifecycle Operations
    """

    def build(self):
        """
        Builds the manager and initialized all the buffers.
        """
        # Filter the actuator DOFs that this action manager controls
        actuator_dofs = self._actuator_manager.dofs
        index_filter = []
        for filter in self._actuator_joints:
            for index, (name, dof_idx) in enumerate[tuple[str, int]](
                actuator_dofs.items()
            ):
                if name == filter or re.match(f"^{filter}$", name):
                    self._dofs[name] = dof_idx
                    index_filter.append(index)
        self._actuator_dof_filter = torch.tensor(
            index_filter, device=gs.device, dtype=gs.tc_int
        )

    def step(self, actions: torch.Tensor) -> None:
        """
        Handle the received actions.
        """
        # Action delay buffer
        if self._delay_step > 0:
            self._action_delay_buffer.insert(0, actions)
            actions = self._action_delay_buffer.pop()

        # Copy the actions into the manager buffer
        self._raw_actions = actions
        if self._actions is None:
            self._actions = torch.empty_like(actions, device=gs.device)
        self._actions[:] = self._raw_actions[:]
        return self._actions

    def reset(self, envs_idx: list[int] | None):
        """Reset environments."""
        if (
            self._delay_step > 0
            and len(self._action_delay_buffer) < self._delay_step
            and self.num_actions > 0
        ):
            while len(self._action_delay_buffer) < self._delay_step:
                self._action_delay_buffer.append(
                    torch.zeros((self.env.num_envs, self.num_actions), device=gs.device)
                )

    def get_actions(self) -> torch.Tensor:
        """
        Get the current actions for the environments.
        """
        if self._actions is None:
            return torch.zeros((self.env.num_envs, self.num_actions))
        return self._actions
