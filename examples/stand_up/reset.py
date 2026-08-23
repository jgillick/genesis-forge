"""Reset handlers for the stand-up training environment."""

from __future__ import annotations

import re
from dataclasses import dataclass

import genesis as gs
import torch
from typing import TYPE_CHECKING

from genesis_forge.managers import ResetMdpFn
from ground_positions import RANDOM_GROUND_POSES

if TYPE_CHECKING:
    from genesis.engine.entities import RigidEntity

JOINT_PATTERNS = [
    "FL_.*_joint",
    "FR_.*_joint",
    "RL_.*_joint",
    "RR_.*_joint",
]


@dataclass(kw_only=True, eq=False)
class random_ground_pose(ResetMdpFn):
    """
    Reset the robot to a random collapsed pose from a pre-generated pose list.
    """

    def build(self):
        poses = RANDOM_GROUND_POSES
        if len(poses) == 0:
            raise RuntimeError(
                "No poses found. Generate them with the script: generate_random_ground_pos.py"
            )

        # Collect joints
        dof_names: list[str] = []
        dofs_idx: list[int] = []
        for joint in self.entity.joints:
            if joint.type != gs.JOINT_TYPE.REVOLUTE:
                continue
            for pattern in JOINT_PATTERNS:
                if pattern == joint.name or re.match(f"^{pattern}$", joint.name):
                    dof_names.append(joint.name)
                    dofs_idx.append(joint.dof_start)
                    break
        self._dofs_idx = dofs_idx

        # Organize poses
        pos_list = [pose["pos"] for pose in poses]
        quat_list = [pose["quat"] for pose in poses]
        dof_list = [[pose["joints"][name] for name in dof_names] for pose in poses]

        # Setup buffers
        self._pos = torch.tensor(pos_list, device=gs.device, dtype=gs.tc_float)
        self._quat = torch.tensor(quat_list, device=gs.device, dtype=gs.tc_float)
        self._dof_pos = torch.tensor(dof_list, device=gs.device, dtype=gs.tc_float)

    def __call__(
        self,
        env,
        entity: RigidEntity,
        envs_idx: list[int],
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
