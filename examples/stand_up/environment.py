"""
Go2 stand-up environment: learn to rise from random collapsed ground poses.
"""

from __future__ import annotations

import genesis as gs
from reset import random_ground_pose

from genesis_forge import ManagedEnvironment
from genesis_forge.managers import (
    ActuatorManager,
    EntityManager,
    ObservationManager,
    PositionActionManager,
    RewardManager,
    TerminationManager,
    ContactManager,
)
from genesis_forge.mdp import observations, rewards, terminations


class Go2StandUpEnv(ManagedEnvironment):
    """Train the Go2 to stand up from random ground poses."""

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 50,
        max_episode_length_s: int | None = 5,
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
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(
                rendered_envs_idx=list(range(min(num_envs, 1)))
            ),
            rigid_options=gs.options.RigidOptions(
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
                links_to_keep=[
                    "Head_upper",
                    "Head_lower",
                    "FL_foot",
                    "FR_foot",
                    "RL_foot",
                    "RR_foot",
                ],
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
        ##
        # Robot manager
        # i.e. what to do with the robot when it is reset
        self.robot_manager = EntityManager(
            self,
            entity=self.robot,
            on_reset={
                "random_ground_pose": {
                    "fn": random_ground_pose(),
                },
            },
        )

        ##
        # Joint Actuators/Actions
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
            max_force=23.5,
        )
        self.action_manager = PositionActionManager(
            self,
            scale=0.25,
            use_default_offset=True,
            actuator_manager=self.actuator_manager,
        )

        ##
        # Contact managers
        #

        # Track feet on the ground
        self.foot_contact_manager = ContactManager(
            self, link_names=[".*_foot"], with_entity=self.terrain
        )

        # Track body parts on the ground
        self.body_contact_manager = ContactManager(
            self,
            link_names=["base", "Head_.*", ".*_thigh", ".*_calf"],
        )

        ##
        # Rewards
        RewardManager(
            self,
            logging_enabled=True,
            cfg={
                # Make sure the robot stays standing at a reasonable height
                "height_target": {
                    "weight": -10.0,
                    "fn": rewards.base_height(
                        target_height=0.3,
                        entity_manager=self.robot_manager,
                    ),
                },
                # Encourage the robot to stay level
                "flat_orientation": {
                    "weight": -0.15,
                    "fn": rewards.flat_orientation_l2(
                        entity_manager=self.robot_manager,
                    ),
                },
                # Penalize excessive angular velocity in the x and y axes (roll and pitch)
                "angular_velocity_penalty": {
                    "weight": -0.05,
                    "fn": rewards.ang_vel_xy_l2(entity_manager=self.robot_manager),
                },
                # Discourage the robot from bounding up and down (z-axis linear velocity)
                "linear_velocity_penalty": {
                    "weight": -8.0,
                    "fn": rewards.lin_vel_z_l2(entity_manager=self.robot_manager),
                },
                # Discourage the robot from making jittery actuator movements
                # Penalizes actions that change back-and-forth a lot
                "action_rate": {
                    "weight": -0.025,
                    "fn": rewards.action_rate_l2(),
                },
                # Discourage abrupt joint velocity changes (e.g. jerky joint movements)
                "joint_acceleration": {
                    "weight": -2.0e-5,
                    "fn": rewards.dof_acc_l2(
                        actuator_manager=self.actuator_manager,
                    ),
                },
                # Discourage joints from exceeding a velocity limits
                "joint_velocity_limit": {
                    "weight": -0.02,
                    "fn": rewards.dof_velocity_l2(
                        threshold=3.0,
                        actuator_manager=self.actuator_manager,
                    ),
                },
                # Discourage any body part (beside feet) being in contact with the ground
                "on_ground": {
                    "weight": -0.5,
                    "fn": rewards.has_contact(
                        contact_manager=self.body_contact_manager,
                    ),
                },
                # Encourage the feet being in contact with the ground
                "feet_on_ground": {
                    "weight": 0.2,
                    "fn": rewards.contact_fraction(
                        contact_manager=self.foot_contact_manager,
                    ),
                },
                # Penalizing terminations adds a real cost to falling over
                "terminated": {
                    "weight": -150.0,
                    "fn": rewards.terminated(),
                },
            },
        )

        ##
        # Termination conditions
        TerminationManager(
            self,
            logging_enabled=True,
            term_cfg={
                "timeout": {
                    "fn": terminations.timeout(),
                    "time_out": True,
                },
                # Terminate early if the robot falls over
                "fall_over": {
                    "fn": terminations.is_upsidedown(
                        entity_manager=self.robot_manager,
                        threshold=-0.26,  # 75 degrees
                    ),
                },
            },
        )

        ##
        # Observations
        ObservationManager(
            self,
            history_len=5,
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
                    "fn": observations.current_actions(),
                },
            },
        )
