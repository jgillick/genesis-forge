"""
Simplified Go2 Locomotion Environment using managers to handle everything.
"""

import os
import math
import torch
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
    TerrainManager,
)
from genesis_forge.managers.entity import reset
from genesis_forge.mdp import rewards, terminations

HEIGHT_OFFSET = 0.4  # How high above the terrain the robot should be placed
INITIAL_BODY_POSITION = [0.0, 0.0, HEIGHT_OFFSET]
INITIAL_QUAT = [1.0, 0.0, 0.0, 0.0]


class Go2RoughTerrainEnv(ManagedEnvironment):
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
            headless=headless,
        )

        # Construct the scene
        self.scene = gs.Scene(
            show_viewer=not self.headless,
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
        self.terrain = self.create_terrain(self.scene)

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
            res=(1280, 960),
            fov=40,
            env_idx=0,
            debug=True,
        )

    def create_terrain(self, scene: gs.Scene):
        """
        Create a random terrain map entity
        """

        # Create random terrain height map
        height_field = np.zeros([40, 40])
        heights_range = np.arange(-10, 20, 10)
        height_field[5:35, 5:35] = np.random.choice(heights_range, (30, 30))

        # Create a tiled terrain surface texture
        # Load a checker image, and tile it 24 times in X and Y directions
        this_dir = os.path.dirname(os.path.abspath(__file__))
        tile_path = os.path.join(this_dir, "checker.png")
        checker_image = np.array(Image.open(tile_path))
        tiled_image = np.tile(checker_image, (24, 24, 1))

        return scene.add_entity(
            surface=gs.surfaces.Default(
                diffuse_texture=gs.textures.ImageTexture(
                    image_array=tiled_image,
                )
            ),
            morph=gs.morphs.Terrain(
                horizontal_scale=0.25,
                vertical_scale=0.005,
                height_field=height_field,
                pos=(-20, -20, 0),
            ),
        )

    def config(self):
        """
        Configure the environment managers
        """
        ##
        # Terrain manager helps the EntityManager safetly place the robot above the terrain on reset
        self.terrain_manager = TerrainManager(self)

        ##
        # Robot manager
        # i.e. what to do with the robot when it is reset
        self.robot_manager = EntityManager(
            self,
            entity_attr="robot",
            on_reset={
                # Randomize the robot's rotation after reset
                "rotation": {
                    "fn": reset.set_rotation,
                    "params": {"z": (0, 2 * math.pi)},
                },
                # Randomize the robot's position on the terrain after reset
                "position": {
                    "fn": reset.randomize_terrain_position,
                    "params": {
                        "terrain_manager": self.terrain_manager,
                        "height_offset": HEIGHT_OFFSET,
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
            clip=(-100.0, 100.0),
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
                "lin_vel_y": [-1.0, 1.0],
                "ang_vel_z": [-1.0, 1.0],
            },
            standing_probability=0.05,
            resample_time_s=5.0,
            debug_visualizer=True,
            debug_visualizer_cfg={
                "envs_idx": [0],
                "arrow_offset": 0.02,
            },
        )

        ##
        # Rewards
        RewardManager(
            self,
            logging_enabled=True,
            cfg={
                "base_height_target": {
                    "weight": -25.0,
                    "fn": rewards.base_height,
                    "params": {
                        "target_height": 0.3,
                        "terrain_manager": self.terrain_manager,
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
                "lin_vel_z": {
                    "weight": -1.0,
                    "fn": rewards.lin_vel_z,
                    "params": {
                        "entity_manager": self.robot_manager,
                    },
                },
                "action_rate": {
                    "weight": -0.005,
                    "fn": rewards.action_rate,
                },
                "similar_to_default": {
                    "weight": -0.05,
                    "fn": rewards.dof_similar_to_default,
                    "params": {
                        "action_manager": self.action_manager,
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
                # The episode ended
                "timeout": {
                    "fn": terminations.timeout,
                    "time_out": True,
                },
                # Terminate if the robot's pitch and yaw angles are too large
                "fall_over": {
                    "fn": terminations.bad_orientation,
                    "params": {
                        "limit_angle": 0.174,  # ~10 degrees
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

    def step(self, actions: torch.Tensor):
        # Keep the camera fixed on the robot
        self.camera.set_pose(lookat=self.robot.get_pos())
        return super().step(actions)
