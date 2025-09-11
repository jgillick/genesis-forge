import torch
from typing import Any
from gymnasium import spaces
import genesis as gs

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers.base import BaseManager, ManagerType


class ManagedEnvironment(GenesisEnv):
    """
    An environment which moves a lot of the logic of the environment to manager classes.
    This helps to keep the environment code clean and modular.

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

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.managers: dict[ManagerType, BaseManager | list[BaseManager] | None] = {
            "action": None,  # there can only be one
            "observation": None,  # there can only be one
            "reward": [],
            "termination": [],
            "contact": [],
            "terrain": [],
            "entity": [],
            "command": [],
        }

        self._action_space = None
        self._observation_space = None
        self._reward_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=gs.tc_float
        )
        self._terminated_buf = torch.zeros(
            (self.num_envs,), device=gs.device, dtype=torch.bool
        )
        self._truncated_buf = torch.zeros_like(self._terminated_buf)

    """
    Properties
    """

    @property
    def action_space(self) -> torch.Tensor:
        """The action space, provided by the action manager, if it exists."""
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
        """The observation space, provided by the observation manager, if it exists."""
        if self.managers["observation"] is not None:
            return self.managers["observation"].observation_space
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
        """Configure the environment managers here."""
        pass

    def build(self):
        """Called when the scene is built"""
        super().build()
        self.config()

        for terrain_manager in self.managers["terrain"]:
            terrain_manager.build()
        if self.managers["action"] is not None:
            self.managers["action"].build()
        for contact_manager in self.managers["contact"]:
            contact_manager.build()
        for termination_manager in self.managers["termination"]:
            termination_manager.build()
        for reward_manager in self.managers["reward"]:
            reward_manager.build()
        for command_manager in self.managers["command"]:
            command_manager.build()
        for entity_manager in self.managers["entity"]:
            entity_manager.build()
        if self.managers["observation"] is not None:
            self.managers["observation"].build()

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """
        Performs a step in the environment.
        """
        super().step(actions)

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
        self._terminated_buf[:] = False
        self._truncated_buf[:] = False
        for termination_manager in self.managers["termination"]:
            terminated, truncated = termination_manager.step()
            self._terminated_buf[:] |= terminated
            self._truncated_buf[:] |= truncated
        dones = self._terminated_buf | self._truncated_buf
        reset_env_idx = dones.nonzero(as_tuple=False).reshape((-1,))

        # Calculate rewards
        self._reward_buf[:] = 0.0
        for reward_manager in self.managers["reward"]:
            self._reward_buf += reward_manager.step()

        # Command managers
        for command_manager in self.managers["command"]:
            command_manager.step()

        # Reset environments
        if reset_env_idx.numel() > 0:
            self.reset(reset_env_idx)

        # Get observation
        obs = self.get_observations()

        return (
            obs,
            self._reward_buf,
            self._terminated_buf,
            self._truncated_buf,
            self.extras,
        )

    def reset(
        self, env_ids: list[int] | None = None
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Reset managers."""
        (obs, _) = super().reset(env_ids)

        if self.managers["action"] is not None:
            self.action_manager.reset(env_ids)
        for entity_manager in self.managers["entity"]:
            entity_manager.reset(env_ids)
        for contact_manager in self.managers["contact"]:
            contact_manager.reset(env_ids)
        for termination_manager in self.managers["termination"]:
            termination_manager.reset(env_ids)
        for reward_manager in self.managers["reward"]:
            reward_manager.reset(env_ids)
        for command_manager in self.managers["command"]:
            command_manager.reset(env_ids)
        if self.managers["observation"] is not None:
            self.managers["observation"].reset(env_ids)

        # Only get observations when env_ids is None because this will be the initial reset called before the first step
        # Otherwise, the observations are ignored
        if env_ids is None:
            obs = self.get_observations()

        return obs, self.extras

    def get_observations(self) -> torch.Tensor:
        """Return the current observations for each environment."""
        if self.managers["observation"] is not None:
            return self.managers["observation"].get_observations()
        return super().get_observations()
