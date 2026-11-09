from collections.abc import Sequence

import genesis as gs
import torch

from genesis_forge import EnvMode, ManagedEnvironment, terrain
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
from genesis_forge.mdp import observations, reset, rewards, terminations

HEIGHT_OFFSET = 0.4  # How high above the terrain the robot should be placed

# The terrain levels, from flat ground to big bumps
TERRAIN_LEVELS = 5

# How well (0 to 1) a robot must follow its velocity commands to move up, or below which it moves down
MOVE_UP_TRACKING = 0.8
MOVE_DOWN_TRACKING = 0.5

INITIAL_BODY_POSITION = [0.0, 0.0, HEIGHT_OFFSET]
INITIAL_QUAT = [1.0, 0.0, 0.0, 0.0]


class Go2TerrainCurriculumEnv(ManagedEnvironment):
    """
    Example training environment for the Go2 robot, with a terrain curriculum.

    Every robot starts on flat ground. When its episode ends, it moves up a level if it
    followed its velocity commands well, or down a level if it followed them poorly or fell.
    """

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 50,
        max_episode_length_s: int | None = 20,
        headless: bool = True,
        mode: EnvMode = "train",
    ):
        super().__init__(
            num_envs=num_envs,
            dt=dt,
            max_episode_length_sec=max_episode_length_s,
            max_episode_random_scaling=0.1,
        )
        self._mode = mode

        # Track the subterrain level for each environment
        self.env_subterrain_idx = torch.full(
            (self.num_envs,),
            0 if mode == "train" else TERRAIN_LEVELS - 1,
            device=gs.device,
            dtype=torch.long,
        )
        self.env_subterrain_max = self.env_subterrain_idx.clone()

        # Construct the scene
        self.scene = gs.Scene(
            show_viewer=not headless,
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(-2.5, -1.5, 1.0),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
                enable_self_collision=False,
            ),
        )

        # Create terrain
        self.terrain_layout = self.create_terrain(self.scene)

        # Robot
        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                file="urdf/go2/urdf/go2.urdf",
                pos=INITIAL_BODY_POSITION,
                quat=INITIAL_QUAT,
            ),
        )

        # Update the main viewer to follow the robot
        if self.scene.viewer is not None:
            self.scene.viewer.follow_entity(self.robot)

        # Camera, for headless video recording
        self.camera = self.scene.add_camera(
            pos=(-2.5, -1.5, 1.6),
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
        self.terrain_manager = TerrainManager(self, layout=self.terrain_layout)

        ##
        # Robot manager
        # i.e. what to do with the robot when it is reset
        self.robot_manager = EntityManager(
            self,
            entity=self.robot,
            on_reset={
                # Place each robot on its level's subterrain
                "position": {
                    "fn": reset.randomize_terrain_position(
                        height_offset=HEIGHT_OFFSET,
                        terrain_manager=self.terrain_manager,
                        subterrain=self.env_subterrain_idx,
                    ),
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
        # Commanded direction
        self.velocity_command = VelocityCommandManager(
            self,
            range={
                "lin_vel_x": (-1.0, 1.0),
                "lin_vel_y": (-1.0, 1.0),
                "ang_vel_z": (-0.5, 0.5),
            },
            stopped_probability=0.05,
            resample_time_sec=5.0,
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
        self.undesired_contacts = ContactManager(
            self,
            link_names=[".*_thigh", "base"],
        )

        ##
        # Rewards
        self.reward_manager = RewardManager(
            self,
            logging_enabled=True,
            cfg={
                # Encourage the robot to follow the commanded linear velocity
                "command_linear_velocity": {
                    "weight": 1.0,
                    "fn": rewards.command_tracking_lin_vel(
                        vel_cmd_manager=self.velocity_command,
                        entity_manager=self.robot_manager,
                    ),
                },
                # Encourage the robot to follow the commanded angular velocity
                "commanded_angular_velocity": {
                    "weight": 0.5,
                    "fn": rewards.command_tracking_ang_vel(
                        vel_cmd_manager=self.velocity_command,
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
                # Penalize non-foot body parts from contacting the terrain
                "undesired_contacts": {
                    "weight": -1.0,
                    "fn": rewards.has_contact(
                        contact_manager=self.undesired_contacts,
                        threshold=5.0,
                    ),
                },
                # Discourage the robot from making jittery actuator movements
                # Penalizes actions that change back-and-forth a lot
                "action_rate": {
                    "weight": -0.005,
                    "fn": rewards.action_rate_l2(),
                },
                # Encourage longer steps
                "foot_air_time": {
                    "weight": 1.0,
                    "fn": rewards.feet_air_time(
                        contact_manager=self.foot_contact_manager,
                        vel_cmd_manager=self.velocity_command,
                        time_threshold=0.5,
                    ),
                },
                # Discourage the robot from tilting too much
                "flat_orientation": {
                    "weight": -1.5,
                    "fn": rewards.flat_orientation_l2(),
                },
                # Encourage the robot to maintain a pose similar to it's default stable pose
                "similar_to_default": {
                    "weight": -0.05,
                    "fn": rewards.dof_similar_to_default(
                        actuator_manager=self.actuator_manager,
                    ),
                },
                # Penalize early termination
                "terminated": {
                    "weight": -100.0,
                    "fn": rewards.terminated(),
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
                # If the robot wandered out of the subterrain bounds, it's not a failure
                "out_of_bounds": {
                    "fn": terminations.out_of_bounds(
                        terrain_manager=self.terrain_manager,
                        subterrain=True,
                    ),
                    "time_out": True,
                },
                # Terminate if the robot's pitch and yaw angles are too large
                "bad_orientation": {
                    "fn": terminations.bad_orientation(
                        limit_angle=30.0,
                        entity_manager=self.robot_manager,
                        grace_steps=20,
                    ),
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
                    "scale": 0.02,
                },
                "actions": {
                    "fn": observations.current_actions(),
                },
            },
        )

    def create_terrain(self, scene: gs.Scene) -> terrain.TerrainLayout:
        """
        Create the terrain levels, and add them to the scene
        """
        # One subterrain per level, getting rougher with each level.
        grid = []
        for level in range(TERRAIN_LEVELS):
            rough = terrain.RandomUniform(
                name=f"level_{level}",
                min_height=0.0,
                max_height=0.05 * level,
                step=0.01,
                downsampled_scale=0.25,
            )
            grid.append([rough])

        return terrain.TerrainLayout(
            scene,
            subterrain_size=(12.0, 12.0),
            vertical_scale=0.001,  # the Go2 robot is small
            grid=grid,
        )

    def build(self):
        super().build()
        self.camera.follow_entity(self.robot)

    def reset(
        self, env_ids: torch.Tensor | Sequence[int] | None = None
    ) -> tuple[torch.Tensor, dict]:
        if env_ids is None:
            envs_idx = self.all_envs_idx
        elif isinstance(env_ids, torch.Tensor):
            envs_idx = env_ids
        else:
            envs_idx = torch.as_tensor(env_ids, device=gs.device, dtype=torch.long)

        # Move the robots whose episodes just ended up or down a level
        if self._mode == "train" and self.step_count > 0:
            self.update_terrain_levels(envs_idx)

        obs, extras = super().reset(env_ids)

        extras[self.extras_logging_key][
            "Curriculum / terrain_level_avg"
        ] = self.env_subterrain_idx.float().mean()
        extras[self.extras_logging_key][
            "Curriculum / terrain_level_max"
        ] = self.env_subterrain_max.float().max()
        return obs, extras

    def update_terrain_levels(self, envs_idx: torch.Tensor):
        """
        Move each robot up a level if it followed its velocity commands well for the whole
        episode, or down a level if it followed them poorly or its episode ended early.
        """
        # How well each robot followed its velocity commands, from 0 to 1: its average
        # command_linear_velocity reward, which follows every command the episode had
        tracking = self.reward_manager.episode_mean_reward_for_envs(
            "command_linear_velocity", envs_idx
        )
        # Fell over. Walking off the level is a time out, so it isn't counted here.
        ended_early = self.termination_manager.terminated[envs_idx]

        move_up = (tracking >= MOVE_UP_TRACKING) & ~ended_early
        move_down = (tracking < MOVE_DOWN_TRACKING) | ended_early
        level = self.env_subterrain_idx[envs_idx] + move_up.long() - move_down.long()

        # Track the highest level each robot has reached
        self.env_subterrain_max[envs_idx] = torch.maximum(
            self.env_subterrain_max[envs_idx], level.clamp(max=TERRAIN_LEVELS - 1)
        )

        # Past the top level, start over at a random level, so the easier ones aren't forgotten
        random_level = torch.randint_like(level, TERRAIN_LEVELS)
        level = torch.where(level >= TERRAIN_LEVELS, random_level, level.clamp(min=0))

        self.env_subterrain_idx[envs_idx] = level
