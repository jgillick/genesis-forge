"""
Simplified Go2 Locomotion Environment using managers to handle everything.
"""

import torch
import genesis as gs

from genesis_forge import ManagedEnvironment
from genesis_forge.managers import (
    RewardManager,
    TerminationManager,
    EntityManager,
    ObservationManager,
    PositionActionManager,
    VelocityCommandManager,
    ContactManager,
)
from genesis_forge.managers.entity import reset
from genesis_forge.mdp import rewards, terminations


INITIAL_BODY_POSITION = [0.0, 0.0, 0.35]
INITIAL_QUAT = [1.0, 0.0, 0.0, 0.0]


class Go2CommandDirectionEnv(ManagedEnvironment):
    """
    Example training environment for the Go2 robot.
    """

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 50,  # control frequency on real robot is 50hz
        max_episode_length_s: int | None = 20,
        headless: bool = True,
    ):
        super().__init__(
            num_envs=num_envs,
            dt=dt,
            max_episode_length_sec=max_episode_length_s,
            max_episode_random_scaling=0.1,
        )

        # Construct the scene
        self.scene = gs.Scene(
            show_viewer=not headless,
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                # for this locomotion policy there are usually no more than 30 collision pairs
                # set a low value can save memory
                max_collision_pairs=30,
            ),
        )

        # Create terrain
        self.terrain = self.scene.add_entity(gs.morphs.Plane())

        # Robot
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=INITIAL_BODY_POSITION,
                quat=INITIAL_QUAT,
            ),
        )

        # Camera, for headless video recording
        self.camera = self.scene.add_camera(
            pos=(-2.5, -1.5, 1.0),
            lookat=(0.0, 0.0, 0.0),
            res=(1280, 720),
            fov=40,
            env_idx=0,
            debug=True,
        )

    def config(self):
        """
        Configure the environment managers
        """
        ##
        # Robot manager
        # i.e. what to do with the robot when it is reset
        self.robot_manager = EntityManager(
            self,
            entity_attr="robot",
            on_reset={
                # Reset the robot's initial position
                "position": {
                    "fn": reset.position,
                    "params": {
                        "position": INITIAL_BODY_POSITION,
                        "quat": INITIAL_QUAT,
                    },
                },
            },
        )

        ##
        # Joint Actions
        self.action_manager = PositionActionManager(
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
            scale=0.5,
            use_default_offset=True,
            pd_kp=20,
            pd_kv=0.5,
        )

        ##
        # Commanded direction
        self.velocity_command = VelocityCommandManager(
            self,
            range={
                "lin_vel_x": [-1.0, 1.0],
                "lin_vel_y": [0, 0],
                "ang_vel_z": [-0.5, 0.5],
            },
            standing_probability=0.0,
            resample_time_s=5.0,
            debug_visualizer=True,
            debug_visualizer_cfg={
                "envs_idx": [0],
            },
        )

        ##
        # Contact manager
        self.foot_contact_manager = ContactManager(
            self,
            link_names=[".*_calf"],
            track_air_time=True,
            air_time_contact_threshold=5.0,
        )
        self.thigh_contact_manager = ContactManager(
            self,
            link_names=[".*_thigh"],
        )

        ##
        # Rewards
        RewardManager(
            self,
            logging_enabled=True,
            cfg={
                "foot_air_time": {
                    "weight": 0.25,
                    "fn": rewards.feet_air_time,
                    "params": {
                        "contact_manager": self.foot_contact_manager,
                        "vel_cmd_manager": self.velocity_command,
                        "time_threshold": 0.5,
                    },
                },
                "tracking_lin_vel": {
                    "weight": 1.0,
                    "fn": rewards.command_tracking_lin_vel,
                    "params": {
                        "vel_cmd_manager": self.velocity_command,
                        "entity_manager": self.robot_manager,
                    },
                },
                "tracking_ang_vel": {
                    "weight": 0.5,
                    "fn": rewards.command_tracking_ang_vel,
                    "params": {
                        "vel_cmd_manager": self.velocity_command,
                        "entity_manager": self.robot_manager,
                    },
                },
                "lin_vel_z": {
                    "weight": -2.0,
                    "fn": rewards.lin_vel_z_l2,
                    "params": {
                        "entity_manager": self.robot_manager,
                    },
                },
                "ang_vel_xy": {
                    "weight": -0.05,
                    "fn": rewards.ang_vel_xy_l2,
                    "params": {
                        "entity_manager": self.robot_manager,
                    },
                },
                "undesired_contacts": {
                    "weight": -1.0,
                    "fn": rewards.has_contact,
                    "params": {
                        "contact_manager": self.thigh_contact_manager,
                        "threshold": 1.0,
                    },
                },
                "action_rate": {
                    "weight": -0.01,
                    "fn": rewards.action_rate_l2,
                },
                "similar_to_default": {
                    "weight": -0.1,
                    "fn": rewards.dof_similar_to_default,
                    "params": {
                        "action_manager": self.action_manager,
                    },
                },
                "flat_orientation": {
                    "weight": -2.5,
                    "fn": rewards.flat_orientation_l2,
                },
            },
        )

        ##
        # Termination conditions
        self.termination_manager = TerminationManager(
            self,
            logging_enabled=True,
            term_cfg={
                # The episode ended
                "timeout": {
                    "fn": terminations.timeout,
                    "time_out": True,
                },
                # Terminate if the robot's pitch and yaw angles are too large
                "fall_over": {
                    "fn": terminations.bad_orientation,
                    "params": {
                        "limit_angle": 20.0,
                        "entity_manager": self.robot_manager,
                    },
                },
            },
        )

        ##
        # Observations
        ObservationManager(
            self,
            cfg={
                "velocity_cmd": {"fn": self.velocity_command.observation},
                "angle_velocity": {
                    "fn": lambda env: self.robot_manager.get_angular_velocity(),
                },
                "linear_velocity": {
                    "fn": lambda env: self.robot_manager.get_linear_velocity(),
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
                "actions": {
                    "fn": lambda env: self.action_manager.get_actions(),
                },
            },
        )

    def build(self):
        super().build()
        self.camera.follow_entity(self.robot)
