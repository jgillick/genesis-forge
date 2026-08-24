from __future__ import annotations

from typing import Any, TypedDict

import genesis as gs
import numpy as np
import torch
from gymnasium import spaces
from tensordict import TensorDict

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers import (
    ActuatorManager,
    CommandManager,
    ContactManager,
    EntityManager,
    ObservationManager,
    PositionActionManager,
    RewardManager,
    TerminationManager,
    TerrainManager,
)
from genesis_forge.managers.base import BaseManager, ManagerType


class ManagersDict(TypedDict):
    actuator: list[ActuatorManager]
    contact: list[ContactManager]
    entity: list[EntityManager]
    command: list[CommandManager]
    terrain: list[TerrainManager]
    action: list[PositionActionManager]
    observation: list[ObservationManager]
    reward: RewardManager | None
    termination: TerminationManager | None


class ManagedEnvironment(GenesisEnv):
    """
    An environment which moves a lot of the logic of the environment to manager classes.
    This helps to keep the environment code clean and modular.

    Args:
        num_envs: Number of parallel environments.
        dt: Simulation time step.
        max_episode_length_sec: Maximum episode length in seconds.
        max_episode_random_scaling: Randomly scale the maximum episode length by this amount (+/-) so that not all environments reset at the same time.
        extras_logging_key: The key used, in info/extras dict, which is returned by step and reset functions, to send data to tensorboard by the RL agent.

    Example::

        class MyEnv(ManagedEnvironment):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

                # ...Define scene here...

            def config(self):
                self.action_manager = PositionalActionManager(
                    self,
                    joint_names=".*",
                    pd_kp=50,
                    pd_kv=0.5,
                    max_force=8.0,
                    default_pos={
                        # Hip joints
                        "Leg[1-2]_Hip": -1.0,
                        "Leg[3-4]_Hip": 1.0,
                        # Femur joints
                        "Leg[1-4]_Femur": 0.5,
                        # Tibia joints
                        "Leg[1-4]_Tibia": 0.6,
                    },
                )
                self.reward_manager = RewardManager(
                    self,
                    term_cfg={
                        "Default pose": {
                            "weight": -1.0,
                            "fn": rewards.dof_similar_to_default,
                            "params": {
                                "dof_action_manager": self.action_manager,
                            },
                        },
                        "Base height": {
                            "fn": mdp.rewards.base_height,
                            "params": { "target_height": 0.135 },
                            "weight": -100.0,
                        },
                    },
                )
                ObservationManager(
                    self,
                    cfg={
                        "velocity_cmd": {"fn": self.velocity_command.observation},
                        "robot_ang_vel": {
                            "fn": utils.entity_ang_vel,
                            "params": {"entity": self.robot},
                            "noise": 0.1,
                        },
                        "robot_lin_vel": {
                            "fn": utils.entity_lin_vel,
                            "params": {"entity": self.robot},
                            "noise": 0.1,
                        },
                        "robot_projected_gravity": {
                            "fn": utils.entity_projected_gravity,
                            "params": {"entity": self.robot},
                            "noise": 0.1,
                        },
                        "robot_dofs_position": {
                            "fn": self.action_manager.get_dofs_position,
                            "noise": 0.01,
                        },
                        "actions": {"fn": lambda: env.actions},
                    },
                )
    """

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 100,
        max_episode_length_sec: int | None = 10,
        max_episode_random_scaling: float = 0.0,
        extras_logging_key: str = "episode",
    ):
        super().__init__(
            num_envs=num_envs,
            dt=dt,
            max_episode_length_sec=max_episode_length_sec,
            max_episode_random_scaling=max_episode_random_scaling,
            extras_logging_key=extras_logging_key,
        )

        self.managers: ManagersDict = {
            "contact": [],
            "entity": [],
            "command": [],
            "terrain": [],
            "actuator": [],
            "action": [],
            # there can only be one of each of these
            "observation": [],
            "reward": None,
            "termination": None,
        }

        self._action_space = None
        self._action_ranges: list[tuple[int, int]] = []
        self._observation_space = None
        self._reward_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_float
        )
        self._terminated_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_bool
        )
        self._truncated_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_bool
        )
        self._observations_buf = TensorDict({}, device=gs.device)

    """
    Properties
    """

    @property
    def action_space(self) -> torch.Tensor:
        """
        The action space, provided by the action manager(s), if any exist.
        """
        return self._action_space

    @action_space.setter
    def action_space(self, action_space: spaces.Space):
        """
        Set the action space.
        """
        self._action_space = action_space

    @property
    def observation_space(self) -> spaces.Space:
        """
        Observation space after :meth:`build`.

        If a manager is named ``"policy"``, that manager's space is used. Otherwise
        all observation managers are concatenated in registration order (same as
        :meth:`get_observations`).
        """
        if self._observation_space is not None:
            return self._observation_space
        if len(self.managers["observation"]) == 1:
            return self.managers["observation"][0].observation_space
        return None

    @observation_space.setter
    def observation_space(self, observation_space: spaces.Space):
        """
        Set the observation space.
        """
        self._observation_space = observation_space

    """
    Managers
    """

    def add_manager(self, manager_type: ManagerType, manager: BaseManager):
        """
        Adds a manager to the environment.
        This will automatically be called by the manager class.

        Args:
            manager_type: The type of manager to add.
            manager: The manager to add.
        """
        if manager_type not in self.managers:
            raise ValueError(f"'{manager_type}' is not a valid manager type.")

        # Append manager if the dict item is a list
        if isinstance(self.managers[manager_type], list):
            self.managers[manager_type].append(manager)
        elif self.managers[manager_type] is None:
            self.managers[manager_type] = manager
        else:
            raise ValueError(
                f"Manager type '{manager_type}' already has a manager, and an environment cannot have multiple {manager_type} managers."
            )

    """
    Operations
    """

    def config(self):
        """
        Override this method and initialize all your managers here.

        Example::

            def config(self):
                EntityManager(
                    self,
                    entity=self.robot,
                    on_reset={
                        "position": {
                            "fn": reset.position(
                                position=INITIAL_BODY_POSITION,
                                quat=INITIAL_QUAT,
                            ),
                        },
                    },
                )
        """

    def build(self):
        """
        Builds the environment before the first step.
        The Genesis scene and all the scene entities must be added before calling this method.
        """
        super().build()
        self.config()

        for terrain_manager in self.managers["terrain"]:
            terrain_manager.build()

        for actuator_manager in self.managers["actuator"]:
            actuator_manager.build()
        self._build_action_managers()

        for contact_manager in self.managers["contact"]:
            contact_manager.build()
        if self.managers["termination"] is not None:
            self.managers["termination"].build()
        if self.managers["reward"] is not None:
            self.managers["reward"].build()
        for command_manager in self.managers["command"]:
            command_manager.build()
        for entity_manager in self.managers["entity"]:
            entity_manager.build()
        self._build_observation_managers()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """
        Performs a step in all environments with the given actions.

        Args:
            actions: Batch of actions for each environment with the :attr:`action_space` shape.

        Returns:
            Batch of (observations, rewards, terminations, truncations, extras)
        """
        super().step(actions)

        # Execute the actions and a simulation step
        for i, action_manager in enumerate[PositionActionManager](
            self.managers["action"]
        ):
            (start, end) = self._action_ranges[i]
            processed_actions = action_manager.step(actions[:, start:end])
            action_manager.send_actions_to_simulation(processed_actions)
        self.scene.step()

        # Update entity managers
        for entity_manager in self.managers["entity"]:
            entity_manager.step()

        # Calculate contact forces
        for contact_manager in self.managers["contact"]:
            contact_manager.step()

        # Calculate termination and truncation
        reset_env_idx = None
        truncated = self._truncated_buf
        terminated = self._terminated_buf
        if self.managers["termination"] is not None:
            terminated, truncated = self.managers["termination"].step()
            reset_env_idx = (
                (terminated | truncated).nonzero(as_tuple=False).reshape((-1,)).detach()
            )

        # Calculate rewards
        rewards = self._reward_buf
        if self.managers["reward"] is not None:
            rewards = self.managers["reward"].step()

        # Command managers
        for command_manager in self.managers["command"]:
            command_manager.step()

        # Reset environments
        if reset_env_idx is not None and reset_env_idx.numel() > 0:
            self.reset(reset_env_idx)

        # Get observations
        obs = self.get_observations()

        return (
            obs,
            rewards,
            terminated,
            truncated,
            self.extras,
        )

    def reset(
        self, env_ids: list[int] | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """
        Reset one or more environments.
        Each of the registered managers will also be reset for those environments.

        Args:
            env_ids: The environment ids to reset. If None, all environments are reset.

        Returns:
            A batch of observations (if env_ids is None) and an info dictionary from the vectorized environment.
        """
        reset_all = env_ids is None
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=gs.device)
        elif not isinstance(env_ids, torch.Tensor):
            env_ids = torch.as_tensor(env_ids, device=gs.device, dtype=torch.long)

        (obs, _) = super().reset(env_ids)

        for actuator_manager in self.managers["actuator"]:
            actuator_manager.reset(env_ids)
        for action_manager in self.managers["action"]:
            action_manager.reset(env_ids)
        for entity_manager in self.managers["entity"]:
            entity_manager.reset(env_ids)
        for contact_manager in self.managers["contact"]:
            contact_manager.reset(env_ids)
        if self.managers["termination"] is not None:
            self.managers["termination"].reset(env_ids)
        if self.managers["reward"] is not None:
            self.managers["reward"].reset(env_ids)
        for command_manager in self.managers["command"]:
            command_manager.reset(env_ids)
        for obs_manager in self.managers["observation"]:
            obs_manager.reset(env_ids)

        # Only get observations when env_ids is None because this will be the initial reset called before the first step
        # Otherwise, the observations are ignored
        if reset_all:
            obs = self.get_observations()

        return obs, self.extras

    def get_observations(self) -> torch.Tensor:
        """
        Returns the current observations for this step.

        Named observations are stored in `extras["observations"]`. 
        If a manager named `"policy"` exists, only its tensor is returned; 
        otherwise all managers are concatenated in registration order.
        """
        self.extras["observations"] = TensorDict({}, device=gs.device)

        if len(self.managers["observation"]) == 0:
            return super().get_observations()

        # Make observations
        parts: list[torch.Tensor] = []
        policy_obs = None
        for obs_manager in self.managers["observation"]:
            obs = obs_manager.get_observations()
            self.extras["observations"][obs_manager.name] = obs
            parts.append(obs)
            if obs_manager.name == "policy":
                policy_obs = obs

        # If there is a "policy" observation manager, this is the one returned to the policy
        if policy_obs is not None:
            return policy_obs
        # Otherwise, concatenate the observation manager spaces
        return torch.cat(parts, dim=-1)

    """
    Internal methods
    """

    def _build_action_managers(self):
        """
        Build the action managers and combine the action spaces.
        """
        if len(self.managers["action"]) == 0:
            return

        low = []
        high = []
        size = 0
        self._action_ranges = []
        for action_manager in self.managers["action"]:
            action_manager.build()

            start = size
            size += action_manager.action_space.shape[0]
            end = size
            self._action_ranges.append((start, end))

            low.append(action_manager.action_space.low)
            high.append(action_manager.action_space.high)

        self._action_space = spaces.Box(
            low=np.concatenate(low),
            high=np.concatenate(high),
            shape=(size,),
            dtype=np.float32,
        )

    def _build_observation_managers(self) -> None:
        """
        Build observation managers and set :attr:`observation_space`.

        If any manager is named ``"policy"``, use that manager's space only. Otherwise
        concatenate all managers in registration order (same layout as
        :meth:`get_observations`).
        """
        obs_managers = self.managers["observation"]
        if len(obs_managers) == 0:
            self._observation_space = None
            return

        # Build observation managers
        policy_obs_mgr = None
        for obs_manager in obs_managers:
            obs_manager.build()
            if obs_manager.name == "policy":
                policy_obs_mgr = obs_manager

        # If ther is an observation manager named "policy", that is the primary observation
        # manager and what will be returned to the main policy
        if policy_obs_mgr is not None:
            self._observation_space = policy_obs_mgr.observation_space
            return

        # Merge the observation manager spaces
        low = []
        high = []
        size = 0
        for obs_manager in obs_managers:
            size += obs_manager.observation_space.shape[0]
            low.append(obs_manager.observation_space.low)
            high.append(obs_manager.observation_space.high)
        self._observation_space = spaces.Box(
            low=np.concatenate(low),
            high=np.concatenate(high),
            shape=(size,),
            dtype=np.float32,
        )
