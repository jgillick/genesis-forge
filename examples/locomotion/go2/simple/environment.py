"""
Simplified Go2 Locomotion Environment using managers to handle everything.
"""

import math
import genesis as gs
from genesis_forge import ManagedEnvironment
from genesis_forge.managers import (
    VelocityCommandManager,
    RewardManager,
    TerminationManager,
    TerrainManager,
    EntityManager,
    ObservationManager,
    PositionActionManager,
)
from genesis_forge.managers.entity import reset
from genesis_forge.utils import (
    entity_projected_gravity,
    entity_ang_vel,
    entity_lin_vel,
)
from genesis_forge.mdp import rewards, terminations


from genesis.utils.geom import (
    quat_to_xyz,
    transform_by_quat,
    inv_quat,
    transform_quat_by_quat,
)

INITIAL_BODY_POSITION = [0.0, 0.0, 0.32]
INITIAL_QUAT = [1.0, 0.0, 0.0, 0.0]


class Go2Env(ManagedEnvironment):
    """
    Example training environment for the Go2 robot.
    """

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 100,
        max_episode_length_s: int | None = 6,
        headless: bool = True,
    ):
        super().__init__(
            num_envs=num_envs,
            dt=dt,
            max_episode_length_sec=max_episode_length_s,
            max_episode_random_scaling=0.1,
            headless=headless,
        )

        # Construct the scene
        self.scene = gs.Scene(
            show_viewer=not self.headless,
            sim_options=gs.options.SimOptions(dt=self.dt),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(-2.5, -1.5, 1.0),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=40,
                max_FPS=60,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=True,
            ),
        )

        # Camera, for headless video recording
        self.camera = self.scene.add_camera(
            pos=(-2.5, -1.5, 1.0),
            lookat=(0.0, 0.0, 0.0),
            res=(1280, 960),
            fov=40,
            env_idx=0,
            debug=True,
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

    def config(self):
        """
        Configure the environment managers
        """
        self.terrain_manager = TerrainManager(self, terrain_attr="terrain")

        ##
        # Robot management
        # i.e. what to do with the robot when it is reset
        EntityManager(
            self,
            entity_attr="robot",
            on_reset={
                # Reset all DOF velocities
                "zero_dof": {
                    "fn": reset.zero_all_dofs_velocity,
                },
                # Reset the robot's initial position
                "position": {
                    "fn": reset.position,
                    "params": {
                        "position": INITIAL_BODY_POSITION,
                    },
                },
                # Randomize the robot's initial orientation
                "orientation": {
                    "fn": reset.set_rotation,
                    "params": {
                        "z": (0, 2 * math.pi),
                    },
                },
            },
        )

        ##
        # Joint Actions
        self.action_manager = PositionActionManager(
            self,
            joint_names=".*",
            default_pos={
                ".*_hip_joint": 0.0,
                "F[L|R]_thigh_joint": 0.8,
                "R[L|R]_thigh_joint": 1.0,
                ".*_calf_joint": -1.5,
            },
            scale=0.25,
            use_default_offset=True,
            pd_kp=20,
            pd_kv=0.5,
        )

        ##
        # Velocity command
        self.command = VelocityCommandManager(
            self,
            # Starting ranges should be small, while robot is learning to stand
            range={
                "lin_vel_x": [-0.5, 0.5],
                "lin_vel_y": [-0.5, 0.5],
                "ang_vel_z": [-0.5, 0.5],
            },
            standing_probability=0.02,
            resample_time_s=5.0,
            debug_visualizer=True,
            debug_visualizer_cfg={
                "envs_idx": [0],
            },
        )

        ##
        # Rewards
        RewardManager(
            self,
            logging_enabled=True,
            cfg={
                "base_height_target": {
                    "weight": -50.0,
                    "fn": rewards.base_height,
                    "params": {
                        "target_height": 0.3,
                    },
                },
                "tracking_lin_vel": {
                    "weight": 1.0,
                    "fn": rewards.command_tracking_lin_vel,
                    "params": {
                        "vel_cmd_manager": self.command,
                    },
                },
                "tracking_ang_vel": {
                    "weight": 0.2,
                    "fn": rewards.command_tracking_ang_vel,
                    "params": {
                        "vel_cmd_manager": self.command,
                    },
                },
                "lin_vel_z": {
                    "weight": -1.0,
                    "fn": rewards.lin_vel_z,
                },
                "action_rate": {
                    "weight": -0.005,
                    "fn": rewards.action_rate,
                },
                "similar_to_default": {
                    "weight": -0.1,
                    "fn": rewards.dof_similar_to_default,
                    "params": {
                        "dof_action_manager": self.action_manager,
                    },
                },
            },
        )

        ##
        # Termination conditions
        self.termination_manager = TerminationManager(
            self,
            logging_enabled=True,
            term_cfg={
                "timeout": {
                    "fn": terminations.timeout,
                    "time_out": True,
                },
                "fall_over": {
                    # "fn": self.termination_roll,
                    "fn": terminations.bad_orientation,
                    "params": {
                        "limit_angle": 0.35,  # ~20 degrees
                    },
                },
            },
        )

        ##
        # Observations
        ObservationManager(
            self,
            cfg={
                "command": {"fn": self.command.observation},
                "angle_velocity": {
                    "fn": entity_ang_vel,
                    "params": {"entity": self.robot},
                    "noise": 0.01,
                },
                "linear_vel": {
                    "fn": entity_lin_vel,
                    "params": {"entity": self.robot},
                    "noise": 0.01,
                },
                "projected_gravity": {
                    "fn": entity_projected_gravity,
                    "params": {"entity": self.robot},
                    "noise": 0.01,
                },
                "dof_position": {
                    "fn": self.action_manager.get_dofs_position,
                    "noise": 0.01,
                },
                "dof_velocity": {
                    "fn": self.action_manager.get_dofs_velocity,
                    "noise": 0.1,
                },
                "actions": {"fn": self.action_manager.get_actions},
            },
        )

    # def termination_roll(self, env):
    #     base_quat = self.robot.get_quat()
    #     base_init_quat = torch.tensor(INITIAL_QUAT, device=gs.device)
    #     inv_base_init_quat = inv_quat(base_init_quat)
    #     base_euler = quat_to_xyz(
    #         transform_quat_by_quat(
    #             torch.ones_like(base_quat) * inv_base_init_quat,
    #             base_quat,
    #         ),
    #         rpy=True,
    #         degrees=True,
    #     )

    #     term = torch.zeros((self.num_envs,), device=gs.device, dtype=gs.tc_int)
    #     term |= torch.abs(base_euler[:, 1]) > 10.0
    #     term |= torch.abs(base_euler[:, 0]) > 10.0
    #     return term.bool()
