import genesis as gs
import torch

from genesis_forge import ManagedEnvironment
from genesis_forge.managers import (
    ActuatorManager,
    EntityManager,
    ObservationManager,
    RewardManager,
    TerminationManager,
    VelocityActionManager,
    VelocityCommandManager,
)
from genesis_forge.managers.terrain_manager import TerrainManager
from genesis_forge.mdp import observations, reset, rewards, terminations

INITIAL_BODY_POSITION = (0.0, 0.0, 0.0458)
INITIAL_QUAT = (1.0, 0.0, 0.0, 0.0)
MAX_WHEEL_VELOCITY = 20.0  # ~200RPM


class WheeledRobotCommandDirectionEnv(ManagedEnvironment):
    """
    Example training environment for the Freenove 4WD raspberry pi platform.
    """

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 50,
        max_episode_length_s: int | None = 10,
        headless: bool = True,
    ):
        super().__init__(
            num_envs=num_envs,
            dt=dt,
            max_episode_length_sec=max_episode_length_s,
        )

        # Construct the scene
        self.scene = gs.Scene(
            show_viewer=not headless,
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(-0.5, 0.5, 0.5),
                camera_lookat=(0.0, 0.0, 0.0),
                camera_fov=40,
            ),
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(range(1))),
            rigid_options=gs.options.RigidOptions(
                dt=self.dt,
                constraint_solver=gs.constraint_solver.Newton,
                enable_collision=True,
                enable_joint_limit=True,
            ),
        )

        # Create terrain
        self.terrain = self.scene.add_entity(gs.morphs.Plane())

        # Robot
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file="./Freenove4WD/Freenove4WD.xml",
                pos=INITIAL_BODY_POSITION,
                quat=INITIAL_QUAT,
            ),
        )

        # Update the main viewer to follow the robot
        if self.scene.viewer is not None:
            self.scene.viewer.follow_entity(self.robot)

        # Camera, for headless video recording
        self.camera = self.scene.add_camera(
            pos=(-1.0, 1.0, 1.0),  # x, y, z
            lookat=(0.0, 0.0, 0.0),
            res=(1280, 720),
            fov=40,
            env_idx=0,
            debug=True,
        )
        self.camera.follow_entity(self.robot)

    def config(self):
        """
        Configure the environment managers
        """
        self.terrain_manager = TerrainManager(self, terrain=self.terrain)

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
        # Wheel actuation
        self.wheel_motors = ActuatorManager(
            self,
            joint_names=[
                "TT_Motor-[1-4]_axel",
            ],
            kv=1.0,
        )
        self.action_manager = VelocityActionManager(
            self,
            # Group the wheels on each side together, as one action
            # since they should be moving at the same velocity.
            action_groups=[
                ["TT_Motor-1_axel", "TT_Motor-2_axel"],  # right side
                ["TT_Motor-3_axel", "TT_Motor-4_axel"],  # left side
            ],
            scale={
                # The front and rear motors are mounted opposite of each other,
                # so their target velocities need to be reversed in order to be turning in the same direction
                "TT_Motor-1_axel": -MAX_WHEEL_VELOCITY,  # front right
                "TT_Motor-2_axel": +MAX_WHEEL_VELOCITY,  # rear right
                "TT_Motor-3_axel": -MAX_WHEEL_VELOCITY,  # front left
                "TT_Motor-4_axel": +MAX_WHEEL_VELOCITY,  # rear left
            },
            clip=(-MAX_WHEEL_VELOCITY, MAX_WHEEL_VELOCITY),
            actuator_manager=self.wheel_motors,
        )

        ##
        # Head servos
        self.head_sevos = ActuatorManager(
            self,
            joint_names=[
                "servo-2",  # left/right
                "servo_horn-1",  # up/down
            ],
            default_pos={
                "servo-2": 0.0,  # facing straight ahead
                "servo_horn-1": 0.0,
            },
            kp=8.0,
            kv=0.4,
        )

        ##
        # Commanded direction
        self.velocity_command = VelocityCommandManager(
            self,
            range={
                "lin_vel_x": (-0.5, 0.5),  # forward/backward
                "lin_vel_y": (-0.0, 0.0),  # cannot move side-to-side
                "ang_vel_z": (-0.5, 0.5),  # turning
            },
            stopped_probability=0.02,
            resample_time_sec=5.0,
            debug_visualizer=True,
            debug_visualizer_cfg={
                "envs_idx": [0],
                "arrow_radius": 0.01,
                "ang_arc_width": 0.015,
            },
        )

        ##
        # Rewards
        RewardManager(
            self,
            logging_enabled=True,
            cfg={
                "tracking_lin_vel": {
                    "weight": 1.0,
                    "fn": rewards.command_tracking_lin_vel(
                        vel_cmd_manager=self.velocity_command,
                        entity_manager=self.robot_manager,
                    ),
                },
                "tracking_ang_vel": {
                    "weight": 0.8,
                    "fn": rewards.command_tracking_ang_vel(
                        vel_cmd_manager=self.velocity_command,
                        entity_manager=self.robot_manager,
                    ),
                },
                "action_rate": {
                    "weight": -0.005,
                    "fn": rewards.action_rate_l2(),
                },
                "body_acceleration_exp": {
                    "weight": -0.1,
                    "fn": rewards.body_acceleration_exp(
                        entity_manager=self.robot_manager,
                    ),
                },
                # When the command is stopped, the wheels should not be moving
                "stopped_dof_velocity": {
                    "weight": -0.01,
                    "fn": rewards.stopped_dof_velocity_l2(
                        vel_cmd_manager=self.velocity_command,
                        actuator_manager=self.wheel_motors,
                    ),
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
                # The robot went out of the terrain
                "out_of_bounds": {
                    "time_out": True,
                    "fn": terminations.out_of_bounds(
                        terrain_manager=self.terrain_manager
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
                "dof_velocity": {
                    "fn": lambda env: self.action_manager.get_dofs_velocity(),
                    "scale": 0.05,
                },
                "actions": {
                    "fn": observations.current_actions(),
                },
            },
        )

    def step(self, actions: torch.Tensor):
        # Keep the head from drooping by keeping the servos at position 0.0
        # This is purely cosmetic, and does not affect the training at all
        self.head_sevos.control_dofs_position(
            torch.zeros((self.num_envs, 2), device=gs.device)
        )

        return super().step(actions)
