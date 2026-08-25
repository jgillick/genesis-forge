import genesis as gs

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
from genesis_forge.mdp import reset, rewards, terminations

INITIAL_BODY_POSITION = (0.0, 0.0, 0.035)
INITIAL_QUAT = (1.0, 0.0, 0.0, 0.0)

class WheeledRobotCommandDirectionEnv(ManagedEnvironment):
    """
    Example training environment for LeKiwi, a 3-wheeled omnidirectional robot
    base from the LeRobot ecosystem, trained to track a commanded body velocity.
    """

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

        # Construct the scene
        self.scene = gs.Scene(
            show_viewer=not headless,
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            viewer_options=gs.options.ViewerOptions(
                refresh_rate=int(0.5 / self.dt),
                camera_pos=(-0.5, -0.5, 0.5),
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

        # Robot -- LeKiwi's 3 driven wheels, each already modeled by its authors
        # as a single low-friction collision capsule (no passive rollers).
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file="./lekiwi/lekiwi.xml",
                pos=INITIAL_BODY_POSITION,
                quat=INITIAL_QUAT,
            ),
        )

        # Update the main viewer to follow the robot
        if self.scene.viewer is not None:
            self.scene.viewer.follow_entity(self.robot)

        # Camera, for headless video recording
        self.camera = self.scene.add_camera(
            pos=(-0.5, -0.5, 0.5),
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
        self.actuator_manager = ActuatorManager(
            self,
            joint_names=[
                "base_back_wheel_joint",
                "base_left_wheel_joint",
                "base_right_wheel_joint"
            ],
            kv=1.0,
        )
        self.action_manager = VelocityActionManager(
            self,
            scale=2.0,
            clip=(-6.28, 6.28),
            actuator_manager=self.actuator_manager,
        )

        ##
        # Commanded direction
        self.velocity_command = VelocityCommandManager(
            self,
            range={
                "lin_vel_x": (-0.2, 0.2),
                "lin_vel_y": (-0.2, 0.2),
                "ang_vel_z": (-0.2, 0.2),
            },
            stopped_probability=0.02,
            resample_time_sec=5.0,
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
                "tracking_lin_vel": {
                    "weight": 1.0,
                    "fn": rewards.command_tracking_lin_vel(
                        vel_cmd_manager=self.velocity_command,
                        entity_manager=self.robot_manager,
                    ),
                },
                "tracking_ang_vel": {
                    "weight": 0.5,
                    "fn": rewards.command_tracking_ang_vel(
                        vel_cmd_manager=self.velocity_command,
                        entity_manager=self.robot_manager,
                    ),
                },
                "lin_vel_z": {
                    "weight": -1.0,
                    "fn": rewards.lin_vel_z_l2(entity_manager=self.robot_manager),
                },
                "action_rate": {
                    "weight": -0.005,
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
                # Terminate if the robot tips over.
                "fall_over": {
                    "fn": terminations.bad_orientation(
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
                    "fn": lambda env: self.action_manager.get_actions(),
                    # Echoes the pipeline's own output rather than a sensor, so a
                    # deployed policy fills this in itself instead of asking the
                    # user for it.
                    "pipeline_state": "processed_actions",
                },
            },
        )
