import genesis as gs

from genesis_forge import ManagedEnvironment
from genesis_forge.managers import (
    ActuatorManager,
    EntityManager,
    ObservationManager,
    PositionActionManager,
    RewardManager,
    TerminationManager,
    VelocityCommandManager,
)
from genesis_forge.managers.actuator import NoisyValue
from genesis_forge.managers.contact.contact_manager import ContactManager
from genesis_forge.mdp import observations, reset, rewards, terminations

HEIGHT_OFFSET = 0.4
INITIAL_BODY_POSITION = (0.0, 0.0, HEIGHT_OFFSET)
INITIAL_QUAT = (1.0, 0.0, 0.0, 0.0)


class Go2DomainRandomizationEnv(ManagedEnvironment):
    """
    Example training environment for the Go2 robot with domain randomization.
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
                ##
                # Batch settings required for some domain randomization
                # and can slow down the simulation and training.
                batch_links_info=True,  # Required for randomize_link_mass_shift
                batch_dofs_info=True,  # Required for armature randomization
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
            pos=(-2.5, -1.5, 1.5),
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
                # Add/subtract a random amount of mass to the robot's body
                "mass_randomization": {
                    "fn": reset.randomize_link_mass_shift(
                        link_name="base",
                        mass_range=(-0.5, 1.0),  # kg
                    ),
                },
            },
        )

        ##
        # Joint Actions
        self.actuator_manager = ActuatorManager(
            self,
            joint_names=[".*"],
            default_pos={
                # Randomize the default positions by +/- 0.02 radians
                ".*_hip_joint": NoisyValue(0.0, 0.02),
                "FL_thigh_joint": NoisyValue(0.8, 0.02),
                "FR_thigh_joint": NoisyValue(0.8, 0.02),
                "RL_thigh_joint": NoisyValue(1.0, 0.02),
                "RR_thigh_joint": NoisyValue(1.0, 0.02),
                ".*_calf_joint": NoisyValue(-1.5, 0.02),
            },
            kp=NoisyValue(25, 1.0),  # +/- 1.0
            kv=NoisyValue(0.5, 0.05),  # +/- 0.05
            damping=NoisyValue(0.1, 0.025),  # +/- 0.025
            frictionloss=NoisyValue(0.1, 0.01),  # +/- 0.01
            armature=NoisyValue(0.01, 0.001),  # +/- 0.001
        )
        self.action_manager = PositionActionManager(
            self,
            scale=0.25,
            use_default_offset=True,
            actuator_manager=self.actuator_manager,
            # Each env's actions reach the actuators 0-2 steps late,
            # to emulate the variable latency of the real control loop
            delay_step=(0, 2),
        )

        ##
        # Commanded direction
        self.velocity_command = VelocityCommandManager(
            self,
            range={
                "lin_vel_x": (-1.0, 1.0),
                "lin_vel_y": (-1.0, 1.0),
                "ang_vel_z": (-1.0, 1.0),
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
                # Make sure the robot stays standing at a reasonable height (0.3 meters)
                "height_target": {
                    "weight": -50.0,
                    "fn": rewards.base_height(
                        target_height=0.3,
                    ),
                },
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
                    "weight": -1.0,
                    "fn": rewards.lin_vel_z_l2(entity_manager=self.robot_manager),
                },
                # Penalize excessive angular velocity in the x and y axes (roll and pitch)
                "angular_velocity_penalty": {
                    "weight": -0.05,
                    "fn": rewards.ang_vel_xy_l2(entity_manager=self.robot_manager),
                },
                # Penalizes actions that change back-and-forth a lot
                # which discourages the robot from making jittery actuator movements
                "action_rate": {
                    "weight": -0.01,
                    "fn": rewards.action_rate_l2(),
                },
                # Discourage abrupt joint velocity changes (e.g. jerky joint movements)
                "dof_acceleration": {
                    "weight": -5.0e-7,
                    "fn": rewards.dof_acc_l2(
                        actuator_manager=self.actuator_manager,
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
                # Terminate if the robot's pitch and yaw angles are too large
                "fall_over": {
                    "fn": terminations.bad_orientation(
                        limit_angle=30.0,
                        entity_manager=self.robot_manager,
                    ),
                },
            },
        )

        ##
        # Observations
        ObservationManager(
            self,
            history_len=3,  # History gives the policy additional context about the robot's state and velocity
            cfg={
                "velocity_cmd": {
                    "fn": self.velocity_command.observation,
                },
                "angle_velocity": {
                    "fn": lambda env: self.robot_manager.get_angular_velocity(),
                    "noise": 0.1,
                },
                "linear_velocity": {
                    "fn": lambda env: self.robot_manager.get_linear_velocity(),
                    "noise": 0.1,
                },
                "projected_gravity": {
                    "fn": lambda env: self.robot_manager.get_projected_gravity(),
                    "noise": 0.05,
                },
                "dof_position": {
                    "fn": lambda env: self.action_manager.get_dofs_position(),
                    "noise": 0.01,
                },
                "dof_velocity": {
                    "fn": lambda env: self.action_manager.get_dofs_velocity(),
                    "scale": 0.02,
                    "noise": 1.5,
                },
                "actions": {
                    "fn": observations.current_actions(),
                },
            },
        )
