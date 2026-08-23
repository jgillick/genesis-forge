"""
Go2 stand-up environment: learn to rise from random collapsed ground poses.
"""

from __future__ import annotations

import genesis as gs

from genesis_forge import ManagedEnvironment
from genesis_forge.managers import (
    RewardManager,
    TerminationManager,
    EntityManager,
    ObservationManager,
    ActuatorManager,
    PositionActionManager,
)
from genesis_forge.mdp import rewards, terminations

from reset import random_ground_pose
from rewards import stand_and_balance_reward

class Go2StandUpEnv(ManagedEnvironment):
    """Train the Go2 to stand up from random ground poses."""

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 50,
        max_episode_length_s: int | None = 3,
        headless: bool = True,
    ):
        super().__init__(
            num_envs=num_envs,
            dt=dt,
            max_episode_length_sec=max_episode_length_s,
            max_episode_random_scaling=0.1,
        )

        self.scene = gs.Scene(
            show_viewer=not headless,
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(min(num_envs, 1)))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                max_collision_pairs=30,
            ),
        )

        self.scene.add_entity(gs.morphs.Plane())

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=[0.0, 0.0, 0.4],
                quat=[1.0, 0.0, 0.0, 0.0],
            ),
        )

        self.camera = self.scene.add_camera(
            pos=(-2.5, -1.5, 1.0),
            lookat=(0.0, 0.0, 0.0),
            res=(1280, 720),
            fov=40,
            env_idx=0,
            debug=True,
        )
        self.camera.follow_entity(self.robot)

    def config(self):
        self.actuator_manager = ActuatorManager(
            self,
            joint_names=[
                "FL_.*_joint",
                "FR_.*_joint",
                "RL_.*_joint",
                "RR_.*_joint",
            ],
            default_pos={
                ".*_hip_joint": 0.0,
                "FL_thigh_joint": 0.8,
                "FR_thigh_joint": 0.8,
                "RL_thigh_joint": 1.0,
                "RR_thigh_joint": 1.0,
                ".*_calf_joint": -1.5,
            },
            kp=20,
            kv=0.5,
            max_force=20.0,
        )
        self.action_manager = PositionActionManager(
            self,
            scale=0.1,
            use_default_offset=True,
            actuator_manager=self.actuator_manager,
        )

        self.robot_manager = EntityManager(
            self,
            entity_attr="robot",
            on_reset={
                "random_ground_pose": {
                    "fn": random_ground_pose(),
                },
            },
        )

        RewardManager(
            self,
            logging_enabled=True,
            cfg={
                "base_height": {
                    "weight": -50.0,
                    "fn": rewards.base_height(
                        target_height=0.25,
                        entity_manager=self.robot_manager,
                    ),
                },
                "stand_and_balance": {
                    "weight": 2.0,
                    "fn": stand_and_balance_reward,
                    "params": {
                        "entity_manager": self.robot_manager,
                        "target_height": 0.28,
                        "max_tilt_deg": 20.0,
                    },
                },
                "flat_orientation": {
                    "weight": -0.5,
                    "fn": rewards.flat_orientation_l2(
                        entity_manager=self.robot_manager,
                    ),
                },
                "lin_vel_xy": {
                    "weight": -0.2,
                    "fn": rewards.lin_vel_xy_l2(entity_manager=self.robot_manager),
                },
                "lin_vel_z": {
                    "weight": -3.0,
                    "fn": rewards.lin_vel_z_l2(entity_manager=self.robot_manager),
                },
                "dof_vel": {
                    "weight": -0.02,
                    "fn": rewards.dof_velocity_l2(action_manager=self.action_manager),
                },
                "action_rate": {
                    "weight": -0.1,
                    "fn": rewards.action_rate_l2(),
                },
                "action_accel": {
                    "weight": -0.02,
                    "fn": rewards.action_acceleration_l2(
                        action_manager=self.action_manager
                    ),
                },
                "torque_l2": {
                    "weight": -0.0003,
                    "fn": rewards.dof_torque_l2(actuator_manager=self.actuator_manager),
                },
                "body_acceleration": {
                    "weight": -0.2,
                    "fn": rewards.body_acceleration_exp(
                        entity_manager=self.robot_manager
                    ),
                },
                "is_alive": {
                    "weight": 0.05,
                    "fn": rewards.is_alive(),
                },
            },
        )

        self.termination_manager = TerminationManager(
            self,
            logging_enabled=True,
            term_cfg={
                "timeout": {
                    "fn": terminations.timeout(),
                    "time_out": True,
                },
                "is_upsidedown": {
                    "fn": terminations.is_upsidedown(
                        entity_manager=self.robot_manager,
                        threshold=0.5,
                    ),
                },
            },
        )

        ObservationManager(
            self,
            history_len=4,
            cfg={
                "angle_velocity": {
                    "fn": lambda env: self.robot_manager.get_angular_velocity(),
                    "scale": 0.25,
                },
                "linear_velocity": {
                    "fn": lambda env: self.robot_manager.get_linear_velocity(),
                    "scale": 2.0,
                },
                "projected_gravity": {
                    "fn": lambda env: self.robot_manager.get_projected_gravity(),
                },
                "dof_position": {
                    "fn": lambda env: self.action_manager.get_dofs_position(),
                },
                "dof_velocity": {
                    "fn": lambda env: self.action_manager.get_dofs_velocity(),
                    "scale": 0.05,
                },
                "dof_torque": {
                    "fn": lambda env: self.actuator_manager.get_dofs_force(),
                    "scale": 0.05,
                },
                "actions": {
                    "fn": lambda env: self.action_manager.get_actions(),
                },
            },
        )

