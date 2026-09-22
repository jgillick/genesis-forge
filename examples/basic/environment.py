"""
Simplified Go2 Locomotion Environment using managers to handle everything.
"""

import genesis as gs
import torch

from genesis_forge import ManagedEnvironment
from genesis_forge.managers import (
    ActuatorManager,
    EntityManager,
    ObservationManager,
    PositionActionManager,
    RewardManager,
    TerminationManager,
)
from genesis_forge.mdp import observations, reset, rewards, terminations

INITIAL_BODY_POSITION = (0.0, 0.0, 0.4)
INITIAL_QUAT = (1.0, 0.0, 0.0, 0.0)
TARGET_X_VELOCITY = 0.5


class Go2BasicEnv(ManagedEnvironment):
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

        # Set the target robot direction, along the X axis
        self.target_linear_velocity = torch.tensor(
            [TARGET_X_VELOCITY, 0.0], device=gs.device, dtype=gs.tc_float
        ).repeat(self.num_envs, 1)

        # Construct the scene
        self.scene = gs.Scene(
            show_viewer=not headless,
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
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
            entity=self.robot,
            on_reset={
                # Reset the robot's initial position
                "position": {
                    "fn": reset.position(
                        position=INITIAL_BODY_POSITION,
                        quat=INITIAL_QUAT,
                        zero_velocity=True,
                    ),
                },
            },
        )

        ##
        # Joint Actions
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
        )
        self.action_manager = PositionActionManager(
            self,
            scale=0.25,
            clip=(-100.0, 100.0),
            use_default_offset=True,
            actuator_manager=self.actuator_manager,
        )

        ##
        # Rewards
        RewardManager(
            self,
            logging_enabled=True,
            cfg={
                # Make sure the robot stays standing at a reasonable height (0.3 meters)
                "height_target": {
                    "weight": -50.0,
                    "fn": rewards.base_height(
                        target_height=0.3,
                    ),
                },
                # Encourage the robot to follow the target linear velocity
                "target_linear_velocity": {
                    "weight": 1.0,
                    "fn": rewards.command_tracking_lin_vel(
                        command=self.target_linear_velocity,
                        entity_manager=self.robot_manager,
                    ),
                },
                # Discourage the robot from bounding up and down (z-axis linear velocity)
                "linear_velocity_penalty": {
                    "weight": -2.0,
                    "fn": rewards.lin_vel_z_l2(entity_manager=self.robot_manager),
                },
                # Penalize excessive angular velocity in the x and y axes (roll and pitch)
                "angular_velocity_penalty": {
                    "weight": -0.05,
                    "fn": rewards.ang_vel_xy_l2(entity_manager=self.robot_manager),
                },
                # Discourage the robot from making jittery actuator movements
                # Penalizes actions that change back-and-forth a lot
                "action_rate": {
                    "weight": -0.01,
                    "fn": rewards.action_rate_l2(),
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
                    "fn": terminations.timeout(),
                    "time_out": True,
                },
                # Terminate if the robot's pitch and yaw angles are too large
                "fall_over": {
                    "fn": terminations.bad_orientation(
                        limit_angle=20.0,
                        entity_manager=self.robot_manager,
                    ),
                },
            },
        )

        ##
        # Observations
        ObservationManager(
            self,
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
                "actions": {
                    "fn": observations.current_actions(),
                },
            },
        )

    def build(self):
        super().build()
        self.camera.follow_entity(self.robot)
