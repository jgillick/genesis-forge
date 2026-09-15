from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import genesis as gs
import numpy as np
import torch
from gymnasium import spaces

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.actuator import ActuatorManager
from genesis_forge.managers.base import BaseManager
from genesis_forge.utils import name_matches


@dataclass
class DeploymentActionConfig:
    """How an action manager's process is reproduced on a robot.

    Returned by :meth:`BaseActionManager.get_deployment_config` and written into
    the deployment bundle. Everything here must be plain data -- no tensors, no
    references to the simulator -- because it is read back by a numpy-only
    runtime that never imports Genesis or torch.

    Args:
        deploy_type: Stable name for this manager's process. The runtime resolves
            its processor by this name, so it is the manifest's contract: keep it
            stable across refactors. Built-in names are ``"position"`` and
            ``"position_within_limits"``.
        config: Plain-data parameters the processor needs (numbers, lists of
            numbers, strings). The schema belongs to the processor, not to the
            exporter, so custom managers are free to define their own.
        processor_import_path: For custom managers, where the matching processor
            class lives, written as ``"my_package.processors:MyProcessor"``. Leave
            unset for built-in types, which the runtime already ships.
        joint_action_index: One action index per joint, in the manager's joint
            order, when ``action_groups`` has several joints sharing an action.
            Unset when every joint has its own action.
    """

    deploy_type: str
    config: dict[str, Any] = field(default_factory=dict)
    processor_import_path: str | None = None
    joint_action_index: list[int] | None = None


def to_nominal_array(
    tensor: torch.Tensor,
    *,
    name: str,
    num_joints: int,
    num_envs: int,
    manager_name: str,
) -> list[float]:
    """Reduce a possibly per-environment tensor to one nominal value per joint.

    Process parameters are stored per-environment, and domain randomization may
    perturb them differently in each one. A bundle describes a single robot, so
    export refuses to guess which environment is authoritative: if the values
    diverge across environments, this raises instead of silently baking in
    whatever environment 0 happened to hold.

    Args:
        tensor: The parameter to reduce, shaped ``(num_joints,)`` or
            ``(num_envs, num_joints)``.
        name: Parameter name, used in error messages.
        num_joints: How many joints this manager controls.
        num_envs: The environment's parallel environment count.
        manager_name: Manager name, used in error messages.

    Returns:
        One plain float per joint.

    Raises:
        ValueError: The values differ across parallel environments, or the shape
            is not one this reduction understands.
    """
    values = tensor.detach()

    if values.ndim == 1:
        if values.shape[0] != num_joints:
            raise ValueError(
                f"Cannot export '{name}' from action manager '{manager_name}': "
                f"expected {num_joints} value(s), found {values.shape[0]}."
            )
        return [float(item) for item in values.cpu().tolist()]

    if values.ndim == 2 and values.shape[1] == num_joints:
        if values.shape[0] > 1 and not bool((values == values[0]).all()):
            spread = float((values.max(dim=0).values - values.min(dim=0).values).max())
            raise ValueError(
                f"Cannot export '{name}' from action manager '{manager_name}': the "
                f"value differs across parallel environments (largest spread "
                f"{spread:g}). This usually means domain randomization is active. "
                f"Export from a build with randomization disabled so the bundle "
                f"records the nominal values the policy was trained against."
            )
        return [float(item) for item in values[0].cpu().tolist()]

    raise ValueError(
        f"Cannot export '{name}' from action manager '{manager_name}': unexpected "
        f"shape {tuple(values.shape)} for {num_joints} joint(s) across {num_envs} "
        f"environment(s)."
    )


class BaseActionManager(BaseManager):
    """
    Base for managers that handle actions.

    Args:
        env: The environment to manage the DOF actuators for.
        actuator_manager: The actuator manager which is used to setup and control the DOF joints.
        actuator_joints: Which joints of the actuator manager that this action manager will control.
                         These can be full names or regular expressions.
        action_groups: Drive several joints from a single action, given as a list of groups,
                       where each group is the joint names (or regular expressions) that one
                       action controls. For example the wheels down one side of a skid-steer robot,
                       where one action per side matches how the robot actually moves. Every controlled
                       joint must appear in exactly one group. Defaults to None: one action per joint.
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
        self._action_groups = action_groups
        self._dof_action_idx: torch.Tensor | None = None
        self._dofs: dict[int, str] = {}
        self._actuator_dof_filter: torch.Tensor = None

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
        The number of values the policy provides for this manager on each step. This is
        one per controlled DOF, unless `action_groups` ties several DOFs to one action.
        """
        if self._action_groups is not None:
            return len(self._action_groups)
        return self.num_dofs

    @property
    def num_dofs(self) -> int:
        """
        The number of DOFs this manager drives. The same as `num_actions` unless
        `action_groups` is used.
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
    def action_space(self) -> spaces.Space:
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
            return torch.zeros((self.env.num_envs, self.num_dofs), device=gs.device)
        return self._actions

    @property
    def raw_actions(self) -> torch.Tensor:
        """
        The actions received from the policy, before being processed.

        Note this is what the manager *consumed* this step: with a `delay_step`,
        it is the action taken off the delay buffer, not the one just received.
        """
        if self._raw_actions is None:
            return torch.zeros((self.env.num_envs, self.num_actions), device=gs.device)
        return self._raw_actions

    @property
    def joint_action_index(self) -> list[int] | None:
        """One action index per joint, in `dofs` order.

        None when every joint has its own action. Otherwise `action_groups` has
        joints sharing one, and this says which.
        """
        if self._dof_action_idx is None:
            return None
        return [int(index) for index in self._dof_action_idx.detach().cpu().tolist()]

    @property
    def last_actions(self) -> torch.Tensor:
        """
        The processed actions for for the previous step.
        """
        if self._last_actions is None:
            return torch.zeros((self.env.num_envs, self.num_dofs), device=gs.device)
        return self._last_actions

    """
    DOF convenience wrappers
    """

    def get_dofs_position(self) -> torch.Tensor:
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

    def get_dofs_velocity(
        self, clip: tuple[float, float] | None = None
    ) -> torch.Tensor:
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
        Get the current target actions for the environments.
        """
        if self._actions is None:
            return torch.zeros((self.env.num_envs, self.num_dofs), device=gs.device)
        return self._actions

    def get_actions_dict(self, env_idx: int = 0) -> dict[str, float]:
        """
        Get the latest actions for an environment as a dictionary of DOF names and values.
        """
        return {
            name: value.item()
            for name, value in zip(
                self.dofs.keys(), self._actions[env_idx, :], strict=True
            )
        }

    def actions_to_dof_targets(self, actions: torch.Tensor) -> torch.Tensor:
        """Turn this manager's slice of the policy output into per-DOF targets.

        Takes `(num_envs, num_actions)`: fans the actions out to the DOFs they
        drive, then processes them. Override `process_actions` rather than this.
        """
        return self.process_actions(self._actions_for_dofs(actions))

    def process_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Convert the incoming step actions into the values to send to the simulation.
        Override this function to define how actions are processed -- for example,
        `AffineDofActionManager` applies a per-DOF scale/offset/clip transform.

        Args:
            actions: One action per DOF, shape `(num_envs, num_dofs)`, already
                fanned out from the policy's slice by `actions_to_dof_targets`.

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
                if name_matches(name, filter):
                    self._dofs[name] = dof_idx
                    index_filter.append(index)
        self._actuator_dof_filter = torch.tensor(
            index_filter, device=gs.device, dtype=gs.tc_int
        )

        self._build_action_groups()

        # Seed the action delay buffer with zero actions, so the first `delay_step`
        # steps send no-op actions while the real ones are still queued
        self._action_delay_buffer = [
            torch.zeros((self.env.num_envs, self.num_actions), device=gs.device)
            for _ in range(self._delay_step)
        ]

    def _build_action_groups(self):
        """
        Work out which action drives each controlled DOF, so a step's actions can be
        handed out to the DOFs they belong to.
        """
        if self._action_groups is None:
            return

        dof_names = list(self._dofs.keys())
        action_of_dof: list[int | None] = [None] * len(dof_names)

        for action_idx, group in enumerate(self._action_groups):
            patterns = [group] if isinstance(group, str) else group
            for pattern in patterns:
                for dof_idx, name in enumerate(dof_names):
                    if not name_matches(name, pattern):
                        continue
                    if action_of_dof[dof_idx] is not None:
                        raise ValueError(
                            f"Joint '{name}' is in more than one action group. Each joint "
                            "must be driven by exactly one action."
                        )
                    action_of_dof[dof_idx] = action_idx

        ungrouped = [
            name for name, a in zip(dof_names, action_of_dof, strict=True) if a is None
        ]
        if ungrouped:
            raise ValueError(
                f"These joints are not in any action group, so nothing would drive them: "
                f"{ungrouped}. Every controlled joint must appear in exactly one group."
            )

        self._dof_action_idx = torch.tensor(
            action_of_dof, device=gs.device, dtype=torch.long
        )

    def _actions_for_dofs(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Distribute the actions to the DOFs they control.
        If `action_groups` is not defined, the actions are already one per DOF, otherwise,
        the actions are distributed to the DOFs according to the groups defined in `action_groups`.
        """
        if self._dof_action_idx is None:
            return actions
        return actions[:, self._dof_action_idx]

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
            shape = (actions.shape[0], self.num_dofs)
            self._actions = torch.zeros(shape, device=gs.device, dtype=actions.dtype)
            self._last_actions = torch.zeros(
                shape, device=gs.device, dtype=actions.dtype
            )
        self._last_actions[:] = self._actions[:]

        self._actions[:] = self.actions_to_dof_targets(actions)

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
        if self._actions is not None:
            self._raw_actions[envs_idx] = 0.0
            self._actions[envs_idx] = 0.0
            self._last_actions[envs_idx] = 0.0

    """
    Deployment
    """

    def get_deployment_config(self) -> DeploymentActionConfig:
        """Describe this manager's process so it can be reproduced on a robot.

        Called by :func:`genesis_forge.deployment.export` after the environment is
        built, when every process parameter has been resolved. Implementations
        return plain data only -- see :class:`DeploymentActionConfig`.

        Custom action managers opt in by overriding this method and shipping a
        matching :class:`~genesis_forge_runtime.ActionManagerProcessor` subclass.

        Raises:
            NotImplementedError: This manager has not opted in to deployment export.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support deployment export. Override "
            f"get_deployment_config() to return a DeploymentActionConfig describing "
            f"this manager's process as plain data, and ship a ActionManagerProcessor "
            f"subclass that replays it on the robot."
        )
