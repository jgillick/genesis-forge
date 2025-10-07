import torch
import os
import numpy as np
from PIL import Image
import genesis as gs

from genesis_forge import ManagedEnvironment
from genesis_forge.managers import (
    RewardManager,
    TerminationManager,
    EntityManager,
    ObservationManager,
    PositionActionManager,
    VelocityCommandManager,
    CommandManager,
    TerrainManager,
    ContactManager,
)
from genesis_forge.mdp import reset, rewards, terminations, observations

from gait_command_manager import GaitCommandManager


HEIGHT_OFFSET = 0.4
INITIAL_BODY_POSITION = [0.0, 0.0, HEIGHT_OFFSET]
INITIAL_QUAT = [1.0, 0.0, 0.0, 0.0]
CURRICULUM_CHECK_EVERY_STEPS = 300


class Go2GaitTrainingEnv(ManagedEnvironment):
    """
    Example training environment for the Go2 robot.
    """

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 50,  # control frequency on real robot is 50hz
        max_episode_length_s: int | None = 8,
        headless: bool = True,
    ):
        super().__init__(
            num_envs=num_envs,
            dt=dt,
            max_episode_length_sec=max_episode_length_s,
            max_episode_random_scaling=0.5,
        )
        self._next_curriculum_check_step = CURRICULUM_CHECK_EVERY_STEPS

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
                links_to_keep=["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
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
                        "zero_velocity": True,
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
            scale=0.25,
            use_default_offset=True,
            pd_kp=20,
            pd_kv=0.5,
        )

        ##
        # Contact manager
        self.foot_contact_manager = ContactManager(
            self,
            link_names=[".*_foot"],
            air_time_contact_threshold=1.0,
        )

        ##
        # Commanded direction
        self.velocity_command = VelocityCommandManager(
            self,
            range={
                "lin_vel_x": [-1.0, 1.0],
                "lin_vel_y": [-0.0, 0.0],
                "ang_vel_z": [-1.0, 1.0],
            },
            standing_probability=0.00,
            resample_time_sec=5.0,
            debug_visualizer=True,
            debug_visualizer_cfg={
                "envs_idx": [0],
            },
        )

        ##
        # Gait command manager
        self.gait_command_manager = GaitCommandManager(
            self,
            foot_names={
                "FL": "FL_foot",
                "FR": "FR_foot",
                "RL": "RL_foot",
                "RR": "RR_foot",
            },
        )

        ##
        # Rewards
        self.reward_manager = RewardManager(
            self,
            logging_enabled=True,
            cfg={
                "gait_phase_reward": {
                    "weight": 1.5,
                    "fn": self.gait_command_manager.gait_phase_reward,
                    "params": {
                        "contact_manager": self.foot_contact_manager,
                    },
                },
                "foot_height_reward": {
                    "weight": 0.9,
                    "fn": self.gait_command_manager.foot_height_reward,
                },
                "base_height_target": {
                    "weight": -25.0,
                    "fn": rewards.base_height,
                    "params": {
                        "target_height": 0.35,
                        "entity_manager": self.robot_manager,
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
                "body_acceleration": {
                    "weight": -0.1,
                    "fn": rewards.body_acceleration_exp,
                    "params": {
                        "entity_manager": self.robot_manager,
                    },
                },
                "lin_vel_z": {
                    "weight": -0.1,
                    "fn": rewards.lin_vel_z_l2,
                    "params": {
                        "entity_manager": self.robot_manager,
                    },
                },
                "action_rate": {
                    "weight": -0.01,
                    "fn": rewards.action_rate_l2,
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
            name="policy",
            history_len=5,
            cfg={
                "gait_command": {
                    "fn": self.gait_command_manager.observation,
                },
                "velocity_cmd": {
                    "fn": self.velocity_command.observation,
                },
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
        ObservationManager(
            self,
            name="critic",
            history_len=5,
            cfg={
                "foot_contact_force": {
                    "fn": observations.contact_force,
                    "params": {
                        "contact_manager": self.foot_contact_manager,
                    },
                },
                "dof_force": {
                    "fn": observations.entity_dofs_force,
                    "params": {
                        "action_manager": self.action_manager,
                    },
                    "scale": 0.1,
                },
            },
        )

    def build(self):
        super().build()
        self.camera.follow_entity(self.robot)

    def step(self, actions: torch.Tensor):
        # Keep the camera fixed on the robot
        self.camera.set_pose(lookat=self.robot.get_pos()[0])
        return super().step(actions)

    def reset(self, envs_idx: list[int] | None = None):
        reset = super().reset(envs_idx)
        if envs_idx is not None:
            self.update_curriculum()
        return reset

    def update_curriculum(self):
        """
        Check the curriculum
        """
        # Limit how often we check/update the curriculum
        if self.step_count < self._next_curriculum_check_step:
            return
        self._next_curriculum_check_step = (
            self.step_count + CURRICULUM_CHECK_EVERY_STEPS
        )

        # Gait phase
        # Increase gaits and period range if the base gait reward is over 0.7
        gait_phase_reward = self.reward_manager.last_episode_mean_reward(
            "gait_phase_reward", before_weight=True
        )
        if gait_phase_reward > 0.75:
            self.gait_command_manager.increment_num_gaits()
            self.gait_command_manager.increment_gait_period_range()

        # Foot clearance
        # Increase foot clearance range, if the base reward is over 0.8
        foot_height_reward = self.reward_manager.last_episode_mean_reward(
            "foot_height_reward", before_weight=True
        )
        if foot_height_reward > 0.8:
            self.gait_command_manager.increment_foot_clearance_range()
