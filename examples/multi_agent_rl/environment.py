from __future__ import annotations

import os
from PIL import Image
import numpy as np
import genesis as gs
from gymnasium import spaces

from genesis_forge import ManagedEnvironment
from genesis_forge.managers import (
    ActuatorManager,
    ContactManager,
    EntityManager,
    ObservationManager,
    PositionActionManager,
    RewardManager,
    TerminationManager,
    TerrainManager,
    VelocityCommandManager,
)
from genesis_forge.mdp import reset, rewards, terminations

OBS_SHARED_KEY = "shared"
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

HEIGHT_OFFSET = 0.4
INITIAL_BODY_POSITION = [0.0, 0.0, HEIGHT_OFFSET]
INITIAL_QUAT = [1.0, 0.0, 0.0, 0.0]

_LEG_THIGH_DEFAULT = {"FL": 0.8, "FR": 0.8, "RL": 1.0, "RR": 1.0}


class Go2MasqLocomotionEnv(ManagedEnvironment):
    """
    Go2 multi-agent locomotion training with one-agent per leg.
    """

    AGENTS = ("FL", "FR", "RL", "RR")

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 50,
        max_episode_length_s: int | None = 20,
        headless: bool = True,
    ):
        super().__init__(
            num_envs=num_envs,
            dt=dt,
            max_episode_length_sec=max_episode_length_s,
            max_episode_random_scaling=0.1,
        )

        self.leg_actuator_managers: dict[str, ActuatorManager] = {}
        self.leg_action_managers: dict[str, PositionActionManager] = {}

        self.scene = gs.Scene(
            show_viewer=not headless,
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                max_FPS=int(0.5 / self.dt),
                camera_pos=(-2.5, -1.5, 1.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                max_collision_pairs=30,
            ),
        )

        # Terrain
        checker_image = np.array(Image.open(os.path.join(THIS_DIR, "checker.png")))
        tiled_image = np.tile(checker_image, (24, 24, 1))
        self.terrain = self.scene.add_entity(
            surface=gs.surfaces.Default(
                diffuse_texture=gs.textures.ImageTexture(
                    image_array=tiled_image,
                )
            ),
            morph=gs.morphs.Terrain(
                pos=(-12, -12, 0),
                n_subterrains=(1, 1),
                subterrain_size=(24, 24),
                vertical_scale=0.001,
                subterrain_types=[["random_uniform_terrain"]],
                subterrain_parameters={
                    "random_uniform_terrain": {
                        "min_height": 0.0,
                        "max_height": 0.08,
                        "step": 0.04,
                        "downsampled_scale": 0.25,
                    },
                },
            ),
        )

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=INITIAL_BODY_POSITION,
                quat=INITIAL_QUAT,
                links_to_keep=list(["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
            ),
        )

        # Update the main viewer to follow the robot
        if self.scene.viewer is not None:
            self.scene.viewer.follow_entity(self.robot)

        self.camera = self.scene.add_camera(
            pos=(-2.5, -1.5, 1.0),
            lookat=(0.0, 0.0, 0.0),
            res=(1280, 720),
            fov=40,
            env_idx=0,
            debug=True,
        )
        self.camera.follow_entity(self.robot)

    @property
    def agents(self) -> list[str]:
        return list(self.AGENTS)

    @property
    def observation_spaces(self) -> dict[str, spaces.Box]:
        """Per-leg MAPPO observation space: which is the shared space + the leg's local space."""
        space_by_name = { manager.name: manager.observation_space for manager in self.managers["observation"] }
        shared = space_by_name[OBS_SHARED_KEY]
        named_spaces = {}
        for agent in self.AGENTS:
            # Concatenate the shared space with the agent's local space
            leg = space_by_name[agent]
            low = np.concatenate(
                [shared.low.astype(np.float32), leg.low.astype(np.float32)]
            )
            high = np.concatenate(
                [shared.high.astype(np.float32), leg.high.astype(np.float32)]
            )
            named_spaces[agent] = spaces.Box(low, high, (int(low.shape[0]),), dtype=np.float32)
        return named_spaces

    @property
    def action_spaces(self) -> dict[str, spaces.Box]:
        """Per-leg posture commands (one :class:`~genesis_forge.managers.PositionActionManager` each)."""
        return {
            agent: self.leg_action_managers[agent].action_space
            for agent in self.AGENTS
        }

    @property
    def state_spaces(self) -> dict[str, spaces.Box]:
        """Centralized MAPPO critic on Forge's fused observation vector."""
        fused = self.observation_space
        if fused is None:
            raise RuntimeError("Call build() before reading state_spaces.")
        return {agent: fused for agent in self.AGENTS}

    def config(self) -> None:
        self.terrain_manager = TerrainManager(self, terrain_attr="terrain")
        
        self.robot_manager = EntityManager(
            self,
            entity_attr="robot",
            on_reset={
                # Randomize the robot's position on the terrain after reset
                "position": {
                    "fn": reset.randomize_terrain_position,
                    "params": {
                        "height_offset": HEIGHT_OFFSET,
                        "terrain_manager": self.terrain_manager,
                    },
                },
            },
        )

        for agent in self.AGENTS:
            self.leg_actuator_managers[agent] = ActuatorManager(
                self,
                joint_names=[
                    f"{agent}_hip_joint",
                    f"{agent}_thigh_joint",
                    f"{agent}_calf_joint",
                ],
                default_pos={
                    f"{agent}_hip_joint": 0.0,
                    f"{agent}_thigh_joint": _LEG_THIGH_DEFAULT[agent],
                    f"{agent}_calf_joint": -1.5,
                },
                kp=20,
                kv=0.5,
            )
            self.leg_action_managers[agent] = PositionActionManager(
                self,
                scale=0.25,
                use_default_offset=True,
                actuator_manager=self.leg_actuator_managers[agent],
            )

        self.velocity_command = VelocityCommandManager(
            self,
            range={
                "lin_vel_x": [-1.0, 1.0],
                "lin_vel_y": [-1.0, 1.0],
                "ang_vel_z": [-1.0, 1.0],
            },
            standing_probability=0.02,
            resample_time_sec=5.0,
            debug_visualizer=True,
            debug_visualizer_cfg={"envs_idx": [0]},
        )

        self.foot_contact_manager = ContactManager(
            self,
            link_names=[".*_foot"],
            track_air_time=True,
            air_time_contact_threshold=1.0,
        )

        RewardManager(
            self,
            logging_enabled=True,
            cfg={
                "base_height_target": {
                    "weight": -30.0,
                    "fn": rewards.base_height,
                    "params": {
                        "target_height": 0.3,
                        "entity_attr": "robot",
                        "terrain_manager": self.terrain_manager,
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
                    "fn": rewards.lin_vel_z_l2,
                    "params": {"entity_manager": self.robot_manager},
                },
                "action_rate": {
                    "weight": -0.005,
                    "fn": rewards.action_rate_l2,
                },
                "similar_to_default": {
                    "weight": -0.05,
                    "fn": rewards.dof_similar_to_default,
                    "params": {
                        "actuator_manager": [
                            self.leg_actuator_managers[a] for a in self.AGENTS
                        ],
                    },
                },
                # "foot_air_time": {
                #     "weight": 1.0,
                #     "fn": rewards.feet_air_time,
                #     "params": {
                #         "time_threshold": 1.0,
                #         "contact_manager": self.foot_contact_manager,
                #         "vel_cmd_manager": self.velocity_command,
                #     },
                # },
            },
        )

        self.termination_manager = TerminationManager(
            self,
            logging_enabled=True,
            term_cfg={
                "timeout": {
                    "fn": terminations.timeout,
                    "time_out": True,
                },
                "fall_over": {
                    "fn": terminations.bad_orientation,
                    "params": {
                        "limit_angle": 30.0,
                        "entity_manager": self.robot_manager,
                        "grace_steps": 10,
                    },
                },
            },
        )

        # Shared observations with the overall robot state
        ObservationManager(
            self,
            name=OBS_SHARED_KEY,
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
            },
        )

        # Observations for each leg (bind mgr= per iteration — bare action_manager is late-bound to RR)
        for agent in self.AGENTS:
            action_manager = self.leg_action_managers[agent]
            ObservationManager(
                self,
                name=agent,
                cfg={
                    "dof_position": {
                        "fn": lambda env, mgr=action_manager: mgr.get_dofs_position(),
                    },
                    "dof_velocity": {
                        "fn": lambda env, mgr=action_manager: mgr.get_dofs_velocity(),
                        "scale": 0.05,
                    },
                    "actions": {"fn": lambda env, mgr=action_manager: mgr.get_actions()},
                },
            )

    
