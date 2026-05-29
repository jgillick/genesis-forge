"""Reset handlers for the stand-up training environment."""

from __future__ import annotations

import re

import genesis as gs
import torch
from typing import TYPE_CHECKING

from genesis_forge.genesis_env import GenesisEnv
from genesis_forge.managers import ResetMdpFnClass
from ground_positions import RANDOM_GROUND_POSES

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity

JOINT_PATTERNS = [
    "FL_.*_joint",
    "FR_.*_joint",
    "RL_.*_joint",
    "RR_.*_joint",
]

class random_ground_pose(ResetMdpFnClass):
    """
    Reset the robot to a random collapsed pose from a pre-generated pose list.

    Args:
        env: The environment
        entity: The robot entity (provided at build time)
        poses: List of dicts with pos, quat, joints; defaults to ground_positions.RANDOM_GROUND_POSES
    """

    def __init__(
        self,
        env: GenesisEnv,
        entity: RigidEntity,
    ):
        super().__init__(env, entity)
        self._entity = entity
        self._poses = RANDOM_GROUND_POSES
        self._dof_names: list[str] = []
        self._dofs_idx: list[int] = []
        self._pos: torch.Tensor | None = None
        self._quat: torch.Tensor | None = None
        self._dof_pos: torch.Tensor | None = None

    def build(self, entity: RigidEntity | None = None):
        if len(self._poses) == 0:
            raise RuntimeError(
                "No poses found. Generate them with the script: generate_random_ground_pos.py"
            )

        # Collect joints
        self._dof_names: list[str] = []
        self._dofs_idx: list[int] = []
        for joint in self._entity.joints:
            if joint.type != gs.JOINT_TYPE.REVOLUTE:
                continue
            for pattern in JOINT_PATTERNS:
                if pattern == joint.name or re.match(f"^{pattern}$", joint.name):
                    self._dof_names.append(joint.name)
                    self._dofs_idx.append(joint.dof_start)
                    break
        
        # Organize poses
        pos_list = []
        quat_list = []
        dof_list = []
        for pose in self._poses:
            pos_list.append(pose["pos"])
            quat_list.append(pose["quat"])
            dof_list.append([pose["joints"][name] for name in self._dof_names])

        # Setup buffers
        self._pos = torch.tensor(pos_list, device=gs.device, dtype=gs.tc_float)
        self._quat = torch.tensor(quat_list, device=gs.device, dtype=gs.tc_float)
        self._dof_pos = torch.tensor(dof_list, device=gs.device, dtype=gs.tc_float)


    def __call__(
        self,
        env: GenesisEnv,
        entity: RigidEntity,
        envs_idx: list[int],
        poses: list[dict] | None = None,
    ):
        n = len(envs_idx)
        idx = torch.randint(0, self._pos.shape[0], (n,), device=gs.device)

        entity.set_pos(
            self._pos[idx],
            envs_idx=envs_idx,
            zero_velocity=True,
        )
        entity.set_quat(
            self._quat[idx],
            envs_idx=envs_idx,
            zero_velocity=True,
        )
        entity.set_dofs_position(
            position=self._dof_pos[idx],
            dofs_idx_local=self._dofs_idx,
            envs_idx=envs_idx,
        )
        entity.zero_all_dofs_velocity(envs_idx)
