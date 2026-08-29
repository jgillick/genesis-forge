from __future__ import annotations

import re

import genesis as gs
import numpy as np
import torch
from gymnasium import spaces

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.actuator import ActuatorManager
from genesis_forge.managers.base import BaseManager


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
    ):
        super().__init__(env, type="action")
        self._raw_actions = None
        self._actions = None
        self._last_actions = None
        self._delay_step = delay_step
        self._action_delay_buffer = []
        self._actuator_manager = actuator_manager
        self._actuator_joints = (
            [actuator_joints] if isinstance(actuator_joints, str) else actuator_joints
        )
        self._dofs: dict[int, str] = {}
        self._actuator_dof_filter: torch.Tensor | None = None

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
        The processed actions for for the current step.
        """
        if self._actions is None:
            return torch.zeros((self.env.num_envs, self.num_actions))
        return self._actions

    @property
    def raw_actions(self) -> torch.Tensor:
        """
        The actions received from the policy, before being processed.
        """
        if self._raw_actions is None:
            return torch.zeros((self.env.num_envs, self.num_actions))
        return self._raw_actions

    @property
    def last_actions(self) -> torch.Tensor:
        """
        The processed actions for for the previous step.
        """
        if self._last_actions is None:
            return torch.zeros((self.env.num_envs, self.num_actions))
        return self._last_actions

    """
    DOF convenience wrappers
    """

    def get_dofs_position(self) ->  torch.Tensor:
        """
        A wrapper for `RigidEntity.get_dofs_limits` that returns the position limits of the controlled DOFs.

        Returns:
            position: torch.Tensor, shape (n_envs, n_dofs)
                      The position of the DOFs managed by this action manager.
        """
        return self.actuator_manager.get_dofs_position(dofs_idx=self.dofs_idx)

    def get_dofs_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        A wrapper for `RigidEntity.get_dofs_limit` that returns the limits of the controlled DOFs.

        Returns:
            lower_limit: torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
                         The lower limit of the positional limits for the entity's dofs.
            upper_limit: torch.Tensor, shape (n_dofs,) or (n_envs, n_dofs)
                         The upper limit of the positional limits for the entity's dofs.
        """
        return self.actuator_manager.get_dofs_limits(dofs_idx=self.dofs_idx)

    def get_dofs_velocity(self, clip: tuple[float, float] | None = None) -> torch.Tensor:
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

    def get_dofs_force(self, clip_to_max_force: bool = False) -> torch.Tensor:
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

    def get_actions(self) -> torch.Tensor:
        """
        Get the current actions for the environments.
        """
        if self._actions is None:
            return torch.zeros((self.env.num_envs, self.num_actions))
        return self._actions

    def get_actions_dict(self, env_idx: int = 0) -> dict[str, float]:
        """
        Get the latest actions for an environment as a dictionary of DOF names and values.
        """
        return {
            name: value.item()
            for name, value in zip(
                self.dofs.keys(), self._actions[env_idx, :]
            )
        }

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Convert the incoming step actions into the values to send to the simulation.
        Override this function to define how actions are processed -- for example,
        `AffineDofActionManager` applies a per-DOF scale/offset/clip transform.

        Args:
            actions: The incoming step actions to handle.

        Returns:
            The processed and converted actions.
        """
        raise NotImplementedError(
            "process_actions is not implemented for this action manager."
        )

    def send_actions_to_simulation(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Send the latest processed actions to the actuators in the simulation.
        Override this function to define how the actions are sent to the simulation.
        """
        raise NotImplementedError(
            "handle_actions is not implemented for this action manager."
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

        # Seed the action delay buffer with zero actions, so the first `delay_step`
        # steps send no-op actions while the real ones are still queued
        self._action_delay_buffer = [
            torch.zeros((self.env.num_envs, self.num_actions), device=gs.device)
            for _ in range(self._delay_step)
        ]

    def step(self, actions: torch.Tensor) -> None:
        """
        Handle actions received in this step.
        """
        # Action delay buffer: queue a copy of this step's actions and send the oldest
        if self._delay_step > 0:
            self._action_delay_buffer.insert(0, actions.clone())
            actions = self._action_delay_buffer.pop()

        # Copy the actions into the manager buffer
        self._raw_actions = actions
        if self._actions is None:
            self._actions = torch.zeros_like(actions, device=gs.device)
            self._last_actions = torch.zeros_like(actions, device=gs.device)
        self._last_actions[:] = self._actions[:]

        # Process the actions
        self._actions[:] = self.process_actions(self._raw_actions[:])

        return self._actions

    def reset(self, envs_idx: torch.Tensor | None = None):
        """
        Clear the action history of the reset environments, so the previous episode's
        actions are neither delivered by the delay buffer nor reported as the last actions.
        """
        if envs_idx is None:
            envs_idx = self.env.all_envs_idx
        for delayed_actions in self._action_delay_buffer:
            delayed_actions[envs_idx] = 0.0
        if self._last_actions is not None:
            self._last_actions[envs_idx] = 0.0
