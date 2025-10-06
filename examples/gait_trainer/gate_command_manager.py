"""
Gait Command Manager for implementing the periodic reward composition method
from "Sim-to-Real Learning of All Common Bipedal Gaits via Periodic Reward Composition"
"""

import torch
import genesis as gs
from typing import Dict, Tuple, TypedDict
from genesis.engine.entities import RigidEntity
from genesis_forge.managers.command.command_manager import CommandManager, CommandRange
from genesis_forge.managers import ContactManager
from genesis_forge.genesis_env import GenesisEnv

GAIT_PERIOD_RANGE = [0.3, 0.6]
FOOT_CLEARANCE_RANGE = [0.04, 0.12]
CURRICULUM_CHECK_EVERY_STEPS = 400

# The foot/leg phase offsets relative to each other for each gait
GAIT_OFFSETS = {
    "trot": {
        "FL": 0.0,  # Front-left foot
        "FR": 0.5,
        "RL": 0.5,
        "RR": 0.0,  # Rear-right foot
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
        self.gate_phase = torch.zeros(
            env.num_envs, 1, dtype=torch.float, device=gs.device
        )
        self.clock_input = torch.zeros(
            env.num_envs,
            8,
            dtype=torch.float,
            device=gs.device,
        )
        self.gait_phase_reward_sums = torch.tensor(0.0, device=gs.device)
        self.foot_height_reward_sums = torch.tensor(0.0, device=gs.device)

    @property
    def command(self) -> torch.Tensor:
        """
        The combined gait command
        """
        return torch.cat(
            [
                self.foot_offset,
                self.foot_height,
                self.gait_period,
            ],
            dim=-1,
        )

    def resample_command(self, env_ids: list[int]):
        """
        Resample the command for the given environments
        """
        # Select a random gait for these environments
        selected_gait_idx = torch.randint(0, len(GAIT_OFFSETS), (1,), device=gs.device)
        gait_name = list(GAIT_OFFSETS.keys())[selected_gait_idx]
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
        self._check_curriculum()
        super().step()

        self.gait_time = (self.gait_time + self.env.dt) % self.gait_period
        self.gate_phase = self.gait_time / self.gait_period

    def reset(self, env_ids: list[int] | None = None):
        """
        Reset environments
        """
        if env_ids is None:
            env_ids = torch.arange(self.env.num_envs, device=gs.device)
        super().reset(env_ids)
        self.clock_input[env_ids, :] = 0.0
        self.gait_time[env_ids] = 0.0
        self.gate_phase[env_ids] = 0.0

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
        phi = (self.gate_phase + self.foot_offset[:, foot_idx].unsqueeze(1)) % 1.0
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

    def _check_curriculum(self):
        """
        Check the robot's progress and increase the gaits and ranges if the robot is making progress.
        """

        # Only check every <CURRICULUM_CHECK_EVERY_STEPS> steps
        if (
            self.env.step_count > 0
            and self.env.step_count % CURRICULUM_CHECK_EVERY_STEPS == 0
        ):
            phase_reward_mean = (
                self.gait_phase_reward_sums / CURRICULUM_CHECK_EVERY_STEPS
            )
            foot_reward_mean = (
                self.foot_height_reward_sums / CURRICULUM_CHECK_EVERY_STEPS
            )
            print("Curriculum check!")
            print(f"Gait phase reward mean: {phase_reward_mean}")
            print(f"Foot height reward mean: {foot_reward_mean}")

            # Gait phase
            if phase_reward_mean > 0.8:
                self._num_gaits = min(self._num_gaits + 1, len(GAIT_OFFSETS))
                self._gait_period_range[0] = max(
                    self._gait_period_range[0] - 0.05, GAIT_PERIOD_RANGE[0]
                )
                self._gait_period_range[1] = min(
                    self._gait_period_range[1] + 0.05, GAIT_PERIOD_RANGE[1]
                )

            # Foot clearance
            if foot_reward_mean > 0.8:
                self._foot_clearance_range[0] = max(
                    self._foot_clearance_range[0] - 0.01, FOOT_CLEARANCE_RANGE[0]
                )
                self._foot_clearance_range[1] = min(
                    self._foot_clearance_range[1] + 0.01, FOOT_CLEARANCE_RANGE[1]
                )

            # Reset rewards and log metrics
            self.gait_phase_reward_sums = torch.tensor(0.0, device=gs.device)
            self.foot_height_reward_sums = torch.tensor(0.0, device=gs.device)
            self.env.extras[self.env.extras_logging_key]["Metrics / num_gaits"] = (
                torch.tensor(self._num_gaits, dtype=torch.float, device=gs.device)
            )
            print(self.env.extras["episode"]["Metrics / num_gaits"])
