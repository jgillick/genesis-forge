"""
Gait Command Manager for implementing the periodic reward composition method
from "Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition"
"""

import torch
import genesis as gs
from typing import Dict, Tuple, TypedDict, Literal
from genesis.engine.entities import RigidEntity
from genesis_forge.managers.command.command_manager import CommandManager, CommandRange
from genesis_forge.managers import ContactManager
from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.gamepads import Gamepad

GAIT_PERIOD_RANGE = [0.3, 0.6]
FOOT_CLEARANCE_RANGE = [0.04, 0.12]
CURRICULUM_CHECK_EVERY_STEPS = 500

GaitName = Literal["walk", "trot", "pronk", "pace", "bound"]
FootName = Literal["FL", "FR", "RL", "RR"]

# The foot/leg phase offsets relative to each other for each gait
GAIT_OFFSETS: dict[GaitName, dict[FootName, float]] = {
    "walk": {
        "FL": 0.0,  # Front-left foot starts first
        "RR": 0.25,  # Rear-right foot follows
        "FR": 0.5,  # Front-right foot follows
        "RL": 0.75,  # Rear-left foot follows
    },
    "trot": {
        "FL": 0.0,
        "FR": 0.5,
        "RL": 0.5,
        "RR": 0.0,
    },
    "pronk": {
        "FL": 0.0,
        "FR": 0.0,
        "RL": 0.0,
        "RR": 0.0,
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
        self.gait_phase_reward_sums = 0.0
        self.foot_height_reward_sums = 0.0

    @property
    def command(self) -> torch.Tensor:
        """
        The combined gait command
        """
        if self._gamepad is not None:
            return self._process_gamepad_input()
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
        # Select a random gait for these environments
        selected_gait_idx = torch.randint(0, self._num_gaits, (1,), device=gs.device)
        gait_name = list(GAIT_OFFSETS.keys())[selected_gait_idx]
        self._set_gait(gait_name, env_ids)

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

        self.gait_time = (self.gait_time + self.env.dt) % self.gait_period
        self.gait_phase = self.gait_time / self.gait_period

        # Populate clock input with foot-specific phase information
        for i in range(4):  # For each foot (FL, FR, RL, RR)
            # Calculate individual foot phase
            foot_phase = (self.gait_phase + self.foot_offset[:, i].unsqueeze(1)) % 1.0

            # Sine/Cosine components
            self.clock_input[:, i] = torch.sin(2 * torch.pi * foot_phase).squeeze(-1)
            self.clock_input[:, i + 4] = torch.cos(2 * torch.pi * foot_phase).squeeze(
                -1
            )

        # Metics
        self.env.extras[self.env.extras_logging_key]["Metrics / num_gaits"] = (
            torch.tensor(self._num_gaits, dtype=torch.float, device=gs.device)
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
        Pressing the X button will cycle through the gaits.
        """
        self._gamepad = gamepad
        self._num_gaits = len(GAIT_OFFSETS)
        self._gamepad_gait_idx = 0
        self._set_gait("walk")

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
        reward = torch.exp(-clearance_error / sensitivity)
        self.foot_height_reward_sums += reward.mean()
        return reward

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
        reward = torch.exp(quad_reward)
        self.gait_phase_reward_sums += reward.mean()
        return reward

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
        force_weight[stance_indices, :] = (
            0  # force is not penalized during stance phase
        )
        vel_weight[stance_indices, :] = -1  # speed is penalized during stance phase

        return vel_weight * velocity + force_weight * force

    def _set_gait(self, gait_name: GaitName, env_ids: list[int] | None = None):
        """
        Set the gait parameters for the given environments
        """
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=gs.device)

        gait_offsets = GAIT_OFFSETS[gait_name]

        # Define the foot offsets for the selected gait
        self.foot_offset[env_ids, 0] = gait_offsets["FL"]
        self.foot_offset[env_ids, 1] = gait_offsets["FR"]
        self.foot_offset[env_ids, 2] = gait_offsets["RL"]
        self.foot_offset[env_ids, 3] = gait_offsets["RR"]

        # Foot clearance is set in the gait command manager
        # pronk and bound gait should be at minimum foot clearance
        if gait_name in ["pronk", "bound"]:
            min_clearance = FOOT_CLEARANCE_RANGE[0]
            self.foot_height[env_ids, 0] = min_clearance
        else:
            self.foot_height[env_ids, 0] = torch.empty(
                len(env_ids), device=gs.device
            ).uniform_(*FOOT_CLEARANCE_RANGE)

        # Gait period
        self.gait_period[env_ids, 0] = torch.empty(
            len(env_ids), device=gs.device
        ).uniform_(*GAIT_PERIOD_RANGE)

    def _process_gamepad_input(self, gamepad: Gamepad):
        """
        Select a new gait when the X button is pressed.
        """
        if "X" in gamepad.state.buttons:
            self._gamepad_btn_pressed = True
        elif self._gamepad_btn_pressed:
            inc = 1 if self._gamepad_dpad_pressed == "up" else -1
            self._gamepad_gait_idx = (self._gamepad_gait_idx + inc) % self._num_gaits
            gait_name = list(GAIT_OFFSETS.keys())[self._gamepad_gait_idx]
            self._set_gait(gait_name)
            self._gamepad_btn_pressed = None
