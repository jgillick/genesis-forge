import torch
from typing import Any, TypedDict
from gymnasium import spaces
import genesis as gs

from genesis_forge.genesis_env import GenesisEnv, EnvMode
from genesis_forge.managers.base import BaseManager, ManagerType
from genesis_forge.managers import (
    ContactManager,
    EntityManager,
    CommandManager,
    TerrainManager,
    PositionActionManager,
    ObservationManager,
    RewardManager,
    TerminationManager,
)


class ManagersDict(TypedDict):
    contact: list[ContactManager]
    entity: list[EntityManager]
    command: list[CommandManager]
    terrain: list[TerrainManager]
    action: PositionActionManager | None
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
            # there can only be one of each of these
            "action": None,
            "observation": [],
            "reward": None,
            "termination": None,
        }

        self._action_space = None
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

    """
    Properties
    """

    @property
    def action_space(self) -> torch.Tensor:
        """
        The action space, provided by the action manager, if it exists.
        """
        if self.managers["action"] is not None:
            return self.managers["action"].action_space
        if self._action_space is not None:
            return self._action_space
        return None

    @action_space.setter
    def action_space(self, action_space: spaces.Space):
        """
        Set the action space.
        """
        self._action_space = action_space

    @property
    def observation_space(self) -> spaces.Space:
        """
        The observation space for the "policy" observation manager, if it exists.
        """
        if len(self.managers["observation"]) > 0:
            for obs in self.managers["observation"]:
                if obs.name == "policy":
                    return obs.observation_space
            return self.managers["observation"][0].observation_space
        if self._observation_space is not None:
            return self._observation_space
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
                    entity_attr="robot",
                    on_reset={
                        "position": {
                            "fn": reset.position,
                            "params": {
                                "position": INITIAL_BODY_POSITION,
                                "quat": INITIAL_QUAT,
                            },
                        },
                    },
                )
        """
        pass

    def build(self):
        """
        Builds the environment before the first step.
        The Genesis scene and all the scene entities must be added before calling this method.
        """
        super().build()
        self.config()

        for terrain_manager in self.managers["terrain"]:
            terrain_manager.build()
        if self.managers["action"] is not None:
            self.managers["action"].build()
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
        for obs in self.managers["observation"]:
            obs.build()

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
        self.extras["observations"] = {}

        # Execute the actions and a simulation step
        if self.managers["action"] is not None:
            self.managers["action"].step(actions)
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
        (obs, _) = super().reset(env_ids)

        if self.managers["action"] is not None:
            self.managers["action"].reset(env_ids)
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
        if env_ids is None:
            obs = self.get_observations()

        return obs, self.extras

    def get_observations(self) -> torch.Tensor:
        """
        Returns the current observations for this step.
        If you use the ObservationManager, this will be handled automatically.
        Otherwise, override this method to return the observations.
        """
        if len(self.managers["observation"]) > 0:
            # We already have observations for this step
            if (
                "observations" in self.extras
                and "policy" in self.extras["observations"]
            ):
                return self.extras["observations"]["policy"]
            if "observations" not in self.extras:
                self.extras["observations"] = {}

            # Get observations
            policy_obs = None
            for obs_manager in self.managers["observation"]:
                obs = obs_manager.get_observations()
                self.extras["observations"][obs_manager.name] = obs
                if obs_manager.name == "policy":
                    policy_obs = obs
            return policy_obs

        return super().get_observations()
