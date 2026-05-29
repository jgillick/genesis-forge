#!/usr/bin/env python3
"""
Drop the Go2 from random joint configurations and record settled ground poses.

Writes ground_positions.py with joints (name -> position), pos, and quat for
each stable resting pose (usable for exact training resets without settle time).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import genesis as gs
import torch

DT = 0.01
INITIAL_QUAT = [1.0, 0.0, 0.0, 0.0]
INITIAL_BASE_Z = 0.4  # nominal standing height; only a first guess before link alignment

# Drop / settle tuning (edit here instead of CLI flags)
DROP_MARGIN = 0.05  # m clearance between AABB bottom and ground before release
MAX_FALL_STEPS = 800  # max sim steps per trial (~8 s at 100 Hz)
STABLE_STEPS = 50  # consecutive stable steps required to record a pose
LIN_THRESH = 0.05  # base linear speed (m/s)
ANG_THRESH = 0.15  # base angular speed (rad/s)
DOF_THRESH = 0.05  # max leg joint speed (rad/s)
LIMIT_SCALE = 0.95  # fraction of URDF joint range when sampling pre-drop pose

DEFAULT_ITERATIONS = 100
DEFAULT_OUTPUT = "ground_positions.py"

JOINT_PATTERNS = [
    "FL_.*_joint",
    "FR_.*_joint",
    "RL_.*_joint",
    "RR_.*_joint",
]


@dataclass
class LegDofs:
    names: list[str]
    idx: list[int]


def collect_leg_dofs(robot) -> LegDofs:
    names: list[str] = []
    idx: list[int] = []
    for joint in robot.joints:
        if joint.type != gs.JOINT_TYPE.REVOLUTE:
            continue
        for pattern in JOINT_PATTERNS:
            if pattern == joint.name or re.match(f"^{pattern}$", joint.name):
                names.append(joint.name)
                idx.append(joint.dof_start)
                break
    if len(names) != 12:
        raise RuntimeError(f"Expected 12 leg DOFs, found {len(names)}: {names}")
    return LegDofs(names=names, idx=idx)


def set_passive_joints(robot, dofs_idx: list[int]) -> None:
    n = len(dofs_idx)
    zeros = torch.zeros((n,), device=gs.device, dtype=gs.tc_float)
    robot.set_dofs_kp(zeros, dofs_idx, [0])
    robot.set_dofs_kv(zeros, dofs_idx, [0])


def sample_joint_positions(robot, dofs_idx: list[int], limit_scale: float) -> torch.Tensor:
    lower, upper = robot.get_dofs_limit(dofs_idx)
    if lower.dim() == 1:
        lower = lower.unsqueeze(0)
        upper = upper.unsqueeze(0)
    span = (upper - lower) * limit_scale
    mid = (upper + lower) * 0.5
    return mid + (torch.rand((1, len(dofs_idx)), device=gs.device) - 0.5) * span


def align_base_to_ground_margin(robot, drop_margin: float) -> float:
    """
    Shift the base vertically so the robot AABB's lowest point is at drop_margin above z=0.
    Returns the applied delta-z on the base.
    """

    # Get the lowest point of the robot's bounding box
    aabb = robot.get_AABB()
    if aabb.ndim == 3:
        aabb = aabb[0]
    min_z = float(aabb[0, 2].item())

    # Set position
    delta_z = drop_margin - min_z
    pos = robot.get_pos().clone()
    pos[0, 2] += delta_z
    robot.set_pos(pos, envs_idx=[0], zero_velocity=True)
    return float(delta_z)


def place_above_ground(
    robot,
    dofs_idx: list[int],
    joint_pos: torch.Tensor,
    drop_margin: float,
) -> None:
    quat = torch.tensor([INITIAL_QUAT], device=gs.device, dtype=gs.tc_float)

    robot.set_dofs_position(
        position=joint_pos,
        dofs_idx_local=dofs_idx,
        envs_idx=[0],
    )
    pos = torch.tensor(
        [[0.0, 0.0, INITIAL_BASE_Z]], device=gs.device, dtype=gs.tc_float
    )
    robot.set_pos(pos, envs_idx=[0], zero_velocity=True)
    robot.set_quat(quat, envs_idx=[0], zero_velocity=True)
    robot.zero_all_dofs_velocity([0])

    align_base_to_ground_margin(robot, drop_margin)


def is_stable(
    robot,
    dofs_idx: list[int],
    lin_thresh: float,
    ang_thresh: float,
    dof_thresh: float,
) -> bool:
    lin = torch.linalg.norm(robot.get_vel()[0]).item()
    ang = torch.linalg.norm(robot.get_ang()[0]).item()
    dof_vel = robot.get_dofs_velocity(dofs_idx)
    max_dof = float(torch.max(torch.abs(dof_vel[0])).item())
    return lin < lin_thresh and ang < ang_thresh and max_dof < dof_thresh


def run_drop_until_stable(
    robot,
    scene: gs.Scene,
    dofs_idx: list[int],
    max_fall_steps: int,
    stable_steps: int,
    lin_thresh: float,
    ang_thresh: float,
    dof_thresh: float,
) -> bool:
    consecutive = 0
    for _ in range(max_fall_steps):
        scene.step()
        if is_stable(robot, dofs_idx, lin_thresh, ang_thresh, dof_thresh):
            consecutive += 1
            if consecutive >= stable_steps:
                return True
        else:
            consecutive = 0
    return False


def record_pose(robot, dof_names: list[str], dofs_idx: list[int]) -> dict:
    positions = robot.get_dofs_position(dofs_idx)[0].cpu().tolist()
    joints = {name: pos for name, pos in zip(dof_names, positions, strict=True)}
    return {
        "joints": joints,
        "pos": robot.get_pos()[0].cpu().tolist(),
        "quat": robot.get_quat()[0].cpu().tolist(),
    }


def write_pose_list(path: Path, poses: list[dict], iterations: int) -> None:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines = [
        '"""Auto-generated resting poses for Go2 stand-up resets. Do not edit by hand."""',
        "",
        f"# Generated: {ts}",
        f"# Requested iterations: {iterations}",
        f"# Recorded poses: {len(poses)}",
        "",
        "RANDOM_GROUND_POSES = [",
    ]
    for pose in poses:
        lines.append("    {")
        lines.append(f"        \"pos\": {pose['pos']!r},")
        lines.append(f"        \"quat\": {pose['quat']!r},")
        lines.append(f"        \"joints\": {pose['joints']!r},")
        lines.append("    },")
    lines.append("]")
    lines.append("")
    path.write_text("\n".join(lines))


def build_scene(headless: bool) -> tuple[gs.Scene, object]:
    scene = gs.Scene(
        show_viewer=not headless,
        sim_options=gs.options.SimOptions(dt=DT, substeps=2),
        viewer_options=gs.options.ViewerOptions(
            max_FPS=int(0.5 / DT),
            camera_pos=(2.0, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=40,
        ),
        vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
        rigid_options=gs.options.RigidOptions(
            dt=DT,
            constraint_solver=gs.constraint_solver.Newton,
            enable_collision=True,
            enable_joint_limit=True,
            max_collision_pairs=30,
        ),
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(
            file="urdf/go2/urdf/go2.urdf",
            pos=[0.0, 0.0, 0.0],
            quat=INITIAL_QUAT,
        ),
    )
    scene.build(n_envs=1)
    return scene, robot


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Drop the Go2 from random joint poses and record settled ground states "
            "(joints, pos, quat) to a Python module for stand-up training resets."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Output poses can be applied in a training env without settle time: "
            "set_pos, set_quat, set_dofs_position from the recorded values, then "
            "zero_all_dofs_velocity."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=DEFAULT_ITERATIONS,
        help="Number of drop trials to run. Timeouts are not counted toward the output.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Path to the generated Python module.",
    )
    parser.add_argument(
        "--show-viewer",
        action="store_true",
        help="Open the Genesis viewer to watch each drop (default: headless).",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="gpu",
        choices=["gpu", "cpu"],
        help="Genesis compute backend.",
    )
    args = parser.parse_args()

    backend = gs.gpu if args.device == "gpu" else gs.cpu
    if args.device == "cpu":
        torch.set_default_device("cpu")
    gs.init(logging_level="warning", backend=backend)

    scene, robot = build_scene(headless=not args.show_viewer)
    leg = collect_leg_dofs(robot)
    set_passive_joints(robot, leg.idx)

    poses: list[dict] = []
    skipped = 0

    for i in range(args.iterations):
        joint_pos = sample_joint_positions(robot, leg.idx, LIMIT_SCALE)
        place_above_ground(robot, leg.idx, joint_pos, DROP_MARGIN)

        if run_drop_until_stable(
            robot,
            scene,
            leg.idx,
            MAX_FALL_STEPS,
            STABLE_STEPS,
            LIN_THRESH,
            ANG_THRESH,
            DOF_THRESH,
        ):
            pose = record_pose(robot, leg.names, leg.idx)
            poses.append(pose)
            sample_joints = list(pose["joints"].values())[:3]
            print(
                f"[{i + 1}/{args.iterations}] recorded  "
                f"z={pose['pos'][2]:.4f}  "
                f"joints=[{', '.join(f'{v:.3f}' for v in sample_joints)}, ...]"
            )
        else:
            skipped += 1
            print(f"[{i + 1}/{args.iterations}] skipped (did not stabilize in time)")

    out_path = Path(args.output)
    write_pose_list(out_path, poses, args.iterations)
    print(f"\nWrote {len(poses)} poses to {out_path.resolve()} ({skipped} skipped)")


if __name__ == "__main__":
    main()
