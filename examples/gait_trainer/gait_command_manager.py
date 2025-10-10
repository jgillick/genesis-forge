"""
Gait Command Manager for implementing the periodic reward composition method
from "Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition"
"""

import torch
import random
import genesis as gs
from typing import TypedDict, Literal
from genesis.engine.entities import RigidEntity
from genesis_forge.managers.command.command_manager import CommandManager, CommandRange
from genesis_forge.managers import ContactManager
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.gamepads import Gamepad

GAIT_PERIOD_RANGE = [0.3, 0.6]
FOOT_CLEARANCE_RANGE = [0.04, 0.12]
CURRICULUM_CHECK_EVERY_STEPS = 500

GaitName = Literal["walk", "trot", "pronk", "pace", "bound", "canter"]
FootName = Literal["FL", "FR", "RL", "RR"]

# Gait configuration
# Phase offsets for each foot in different gaits (0.0 = start of cycle, 0.5 = mid-cycle)
# Each gait defines when each foot contacts the ground relative each other in the gait cycle.
GAIT_OFFSETS: dict[GaitName, dict[FootName, float]] = {
    # "walk": {
    #     "FL": 0.0,
    #     "FR": 0.5,
    #     "RL": 0.75,
    #     "RR": 0.25,
    # },
    "trot": {
        "FL": 0.0, # Front-left foot
        "FR": 0.5,
        "RL": 0.5,
        "RR": 0.0, # Rear-right foot
    },
    "pace": {
        "FL": 0.5,
        "FR": 0.0,
        "RL": 0.5,
        "RR": 0.0,
    },
    "bound": {
        "FL": 0.0,
        "FR": 0.0,
        "RL": 0.5,
        "RR": 0.5,
    },
    "pronk": {
        "FL": 0.0,
        "FR": 0.0,
        "RL": 0.0,
        "RR": 0.0,
    },
    # "canter": {
    #     "FL": 0.67,
    #     "FR": 0.33,
    #     "RL": 0.33,
    #     "RR": 0.0,
    # },
}


class FootNames(TypedDict):
    FL: str
    """Front left foot"""
    FR: str
    """Front right foot"""
    RL: str
    """Rear left foot"""
    RR: str
    """Rear right foot"""


class GaitCommandManager(CommandManager):
    """
    Manages gait parameters for implementing different locomotion gaits for a quadruped robot.
    Based on the paper "Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition" (Siekmann et al., 2020)
    https://arxiv.org/abs/2011.01387
    """

    def __init__(
        self,
        env: GenesisEnv,
        foot_names: FootNames,
        resample_time_sec: float = 5.0,
        robot_entity_attr: str = "robot",
    ):
        super().__init__(env, range={}, resample_time_sec=resample_time_sec)

        self._robot_entity_attr = robot_entity_attr
        self._foot_names = foot_names
        self.foot_links = []
        self._gamepad: Gamepad | None = None
        self._gamepad_btn_pressed: bool = False
        self._gamepad_gait_idx = 0

        # Initial ranges - these will be expanded in the curriculum
        self._num_gaits = 1
        self._gait_period_range = [
            (GAIT_PERIOD_RANGE[0] + GAIT_PERIOD_RANGE[1]) / 2
        ] * 2
        self._foot_clearance_range = [FOOT_CLEARANCE_RANGE[0]] * 2
        self._all_gaits_learned = False

        # Buffers
        self.foot_offset = torch.zeros((env.num_envs, 4), device=gs.device)
        self.gait_period = torch.zeros((env.num_envs, 1), device=gs.device)
        self.foot_height = torch.zeros((env.num_envs, 1), device=gs.device)
        self.gait_time = torch.zeros(
            env.num_envs, 1, dtype=torch.float, device=gs.device
        )
        self.gait_phase = torch.zeros(
            env.num_envs, 1, dtype=torch.float, device=gs.device
        )
        self.clock_input = torch.zeros(
            env.num_envs,
            8,
            dtype=torch.float,
            device=gs.device,
        )
        self._gait_selected = torch.zeros(env.num_envs, dtype=torch.long, device=gs.device)

    @property
    def command(self) -> torch.Tensor:
        """
        The combined gait command
        """
        if self._gamepad is not None:
            self._process_gamepad_input()
        return torch.cat(
            [
                self.foot_offset,
                self.foot_height,
                self.gait_period,
            ],
            dim=-1,
        )

    """
    Curriculum operations
    """

    def increment_num_gaits(self):
        """
        If training is going well, increase the number of available gaits by 1.
        """
        if self._all_gaits_learned:
            return
        
        # All gaits have now been mastered
        if self._num_gaits == len(GAIT_OFFSETS):
            self._all_gaits_learned = True
            print("🎯 All gaits learned! Switching to uniform sampling.")
        else:
            self._num_gaits = min(self._num_gaits + 1, len(GAIT_OFFSETS))

    def increment_gait_period_range(self):
        """
        If training is going well, increase the possible gait period range by 0.05.
        """
        self._gait_period_range[0] = max(
            self._gait_period_range[0] - 0.05, GAIT_PERIOD_RANGE[0]
        )
        self._gait_period_range[1] = min(
            self._gait_period_range[1] + 0.05, GAIT_PERIOD_RANGE[1]
        )

    def increment_foot_clearance_range(self):
        """
        If training is going well, increase the possible foot clearance range by 0.05.
        """
        self._foot_clearance_range[0] = max(
            self._foot_clearance_range[0] - 0.01, FOOT_CLEARANCE_RANGE[0]
        )
        self._foot_clearance_range[1] = min(
            self._foot_clearance_range[1] + 0.01, FOOT_CLEARANCE_RANGE[1]
        )

    """
    Command lifecycle operations
    """

    def resample_command(self, env_ids: list[int]):
        """
        Resample the command for the given environments
        """
        # Do not resample if using gamepad control
        if self._gamepad is not None:
            return

        # Convert env_ids to tensor if it's a list
        if isinstance(env_ids, list):
            env_ids = torch.tensor(env_ids, device=gs.device, dtype=torch.long)
        
        gait_names = list(GAIT_OFFSETS.keys())[:self._num_gaits]
        if self._num_gaits == 1:
            # Only one gait available - set all to the same gait
            self._set_gait(gait_names[0], env_ids)
            self._gait_selected[env_ids] = 0
        else:
            # Generate a random list of gait indices
            gait_indices = self._generate_random_gait_indices(len(env_ids))
            for gait_idx in range(self._num_gaits):
                mask = gait_indices == gait_idx
                if mask.any():
                    selected_envs = env_ids[mask]
                    gait_name = gait_names[gait_idx]
                    self._set_gait(gait_name, selected_envs)
                    self._gait_selected[selected_envs] = gait_idx

    def build(self):
        """
        Get foot link indices
        """
        super().build()
        robot: RigidEntity = getattr(self.env, self._robot_entity_attr)
        for i, key in enumerate(("FL", "FR", "RL", "RR")):
            foot_link_name = self._foot_names[key]
            self.foot_links.insert(i, robot.get_link(foot_link_name))

    def step(self):
        """
        Increment the gait time and phase
        """
        super().step()
        self._log_metrics()

        # Update periodic gait values
        self.gait_time = (self.gait_time + self.env.dt) % self.gait_period
        self.gait_phase = self.gait_time / self.gait_period
        for i in range(4):  # FL, FR, RL, RR
            foot_phase = (self.gait_phase + self.foot_offset[:, i].unsqueeze(1)) % 1.0
            self.clock_input[:, i] = torch.sin(2 * torch.pi * foot_phase).squeeze(-1)
            self.clock_input[:, i + 4] = torch.cos(2 * torch.pi * foot_phase).squeeze(
                -1
            )

    def reset(self, env_ids: list[int] | None = None):
        """
        Reset environments
        """
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=gs.device)
        super().reset(env_ids)
        self.clock_input[env_ids, :] = 0.0
        self.gait_time[env_ids] = 0.0
        self.gait_phase[env_ids] = 0.0

    def observation(self, env: GenesisEnv) -> torch.Tensor:
        """
        Return command observations
        """
        return torch.cat(
            [
                self.command,
                self.clock_input,
            ],
            dim=-1,
        )

    def use_gamepad(self, gamepad: Gamepad):
        """
        Control the command using a gamepad.
        Pressing the A button will cycle through the gaits.
        """
        self._gamepad = gamepad
        self._num_gaits = len(GAIT_OFFSETS)
        self._gamepad_gait_idx = 0
        self._gamepad_select_gait(list(GAIT_OFFSETS.keys())[0])

    """
    Rewards
    """

    def foot_height_reward(
        self, env: GenesisEnv, sensitivity: float = 0.1
    ) -> torch.Tensor:
        """
        Calculate the reward for the feet reaching the target height during the swing phase
        """
        link_idx = [f.idx_local for f in self.foot_links]
        foot_vel = env.robot.get_links_vel(links_idx_local=link_idx)
        foot_pos = env.robot.get_links_pos(links_idx_local=link_idx)
        foot_vel_xy_norm = torch.norm(foot_vel[:, :, :2], dim=-1)
        clearance_error = torch.sum(
            foot_vel_xy_norm * torch.square(foot_pos[:, :, 2] - self.foot_height),
            dim=-1,
        )
        return torch.exp(-clearance_error / sensitivity)

    def gait_phase_reward(
        self, env: GenesisEnv, contact_manager: ContactManager
    ) -> torch.Tensor:
        """
        Calculate the reward for the feet being in the correct phase.
        """
        fl = self._foot_phase_reward(0, contact_manager)
        fr = self._foot_phase_reward(1, contact_manager)
        rl = self._foot_phase_reward(2, contact_manager)
        rr = self._foot_phase_reward(3, contact_manager)
        quad_reward = fl.flatten() + fr.flatten() + rl.flatten() + rr.flatten()
        return torch.exp(quad_reward)

    """
    Private methods
    """

    def _foot_phase_reward(
        self, foot_idx: int, contact_manager: ContactManager
    ) -> torch.Tensor:
        """
        Calculate the individual foot phase reward
        """
        link = self.foot_links[foot_idx]
        force_weight = torch.zeros(
            self.env.num_envs, 1, dtype=torch.float, device=gs.device
        )
        vel_weight = torch.zeros(
            self.env.num_envs, 1, dtype=torch.float, device=gs.device
        )

        # Force / velocity
        force = torch.norm(contact_manager.get_contact_forces(link.idx), dim=-1).view(
            -1, 1
        )
        velocity = torch.norm(link.get_vel(), dim=-1).view(-1, 1)

        # Phase
        phi = (self.gait_phase + self.foot_offset[:, foot_idx].unsqueeze(1)) % 1.0
        phi *= 2 * torch.pi

        swing_indices = (phi >= 0.0) & (phi < torch.pi)
        swing_indices = swing_indices.nonzero().flatten()
        stance_indices = (phi >= torch.pi) & (phi < 2 * torch.pi)
        stance_indices = stance_indices.nonzero().flatten()

        force_weight[swing_indices, :] = -1  # force is penalized during swing phase
        vel_weight[swing_indices, :] = 0  # speed is not penalized during swing phase
        force_weight[stance_indices, :] = 0  # force is not penalized during stance
        vel_weight[stance_indices, :] = -1  # speed is penalized during stance phase

        return vel_weight * velocity + force_weight * force
    
    def _set_gait(self, gait_name: GaitName, env_ids: torch.Tensor | None = None):
        """
        Set the gait for a batch of environments
        """
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=gs.device)
            
        gait_offsets = GAIT_OFFSETS[gait_name]
        
        # Define the foot offsets for the selected gait (vectorized assignment)
        self.foot_offset[env_ids, 0] = gait_offsets["FL"]
        self.foot_offset[env_ids, 1] = gait_offsets["FR"]
        self.foot_offset[env_ids, 2] = gait_offsets["RL"]
        self.foot_offset[env_ids, 3] = gait_offsets["RR"]

        # Foot clearance
        # Pronk and bound gait should be at minimum foot clearance
        if gait_name in ["pronk", "bound"]:
            self.foot_height[env_ids, 0] = self._foot_clearance_range[0]
        else:
            self.foot_height[env_ids, 0] = torch.empty(
                len(env_ids), device=gs.device
            ).uniform_(*self._foot_clearance_range)

        # Gait period
        self.gait_period[env_ids, 0] = torch.empty(
            len(env_ids), device=gs.device
        ).uniform_(*self._gait_period_range)
    
    def _generate_random_gait_indices(self, num: int) -> torch.Tensor:
        """
        Pick a list of random gait indices, with the most recent gaits having a higher probability of
        being picked.
        If self._all_gaits_learned is True, all gaits are equally weighted.
        """
        
        if not self._all_gaits_learned:
            # If we haven't learned all gaits yet, weight the recent gaits exponentially higher
            weights = torch.arange(self._num_gaits, device=gs.device).exp()
        else:
            # Otherwise, all gaits are equally weighted
            weights = torch.ones(self._num_gaits, device=gs.device)

        # Normalize: the sum of all weights should be 1
        weights /= weights.sum()
        weights = weights[:self._num_gaits].expand(num, -1)

        return torch.multinomial(weights, 1).squeeze(-1)

    def _process_gamepad_input(self):
        """
        Select a new gait when the A button is pressed.
        """
        if "A" in self._gamepad.state.buttons:
            self._gamepad_btn_pressed = True
        elif self._gamepad_btn_pressed:
            self._gamepad_btn_pressed = False
            self._gamepad_gait_idx = (self._gamepad_gait_idx + 1) % self._num_gaits
            gait_name = list(GAIT_OFFSETS.keys())[self._gamepad_gait_idx]
            self._gamepad_select_gait(gait_name)
    
    def _gamepad_select_gait(self, gait_name: GaitName):
        """
        Select a new gait when the A button is pressed.
        """
        print(f"💃 Selecting gait: {gait_name}")
        env_idx = 0
            
        # Define the foot offsets for the selected gait (vectorized assignment)
        gait_offsets = GAIT_OFFSETS[gait_name]
        self.foot_offset[env_idx, 0] = gait_offsets["FL"]
        self.foot_offset[env_idx, 1] = gait_offsets["FR"]
        self.foot_offset[env_idx, 2] = gait_offsets["RL"]
        self.foot_offset[env_idx, 3] = gait_offsets["RR"]

        # Foot clearance
        # Pronk and bound gait should be at minimum foot clearance, all others should be at the middle of the range
        if gait_name in ["pronk", "bound"]:
            self.foot_height[env_idx, 0] = FOOT_CLEARANCE_RANGE[0]
        else:
            mid_foot_clearance = (FOOT_CLEARANCE_RANGE[0] + FOOT_CLEARANCE_RANGE[1]) / 2
            self.foot_height[env_idx, 0] = mid_foot_clearance

        # Gait period
        mid_gait_period = (GAIT_PERIOD_RANGE[0] + GAIT_PERIOD_RANGE[1]) / 2
        self.gait_period[env_idx, 0] = mid_gait_period
    
    def _log_metrics(self):
        self.env.extras[self.env.extras_logging_key]["Metrics / num_gaits"] = self._num_gaits
        
        # Log gait distribution if multiple gaits are active
        gait_names = list(GAIT_OFFSETS.keys())
        for i, gait_name in enumerate(gait_names):
            count = (self._gait_selected == i).sum()
            self.env.extras[self.env.extras_logging_key][f"Metrics / gait_{gait_name}_envs"] = count
