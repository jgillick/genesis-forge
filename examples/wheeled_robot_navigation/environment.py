import colorsys
import math
import random

import genesis as gs

from genesis_forge import ManagedEnvironment
from genesis_forge.managers import (
    ActuatorManager,
    ContactManager,
    EntityManager,
    ObservationManager,
    Pose2dCommand,
    PositionActionManager,
    RewardManager,
    TerminationManager,
    VelocityActionManager,
)
from genesis_forge.managers.terrain_manager import TerrainManager
from genesis_forge.mdp import observations, reset, rewards, terminations

INITIAL_BODY_POSITION = (0.0, 0.0, 0.0458)
INITIAL_QUAT = (1.0, 0.0, 0.0, 0.0)
NUM_OBSTACLES = 30
OBSTACLE_SIZE = (0.1, 0.1, 0.1)
OBSTACLE_COLORS = [(0.95, 0.8, 0.3, 1.0), (0.75, 0.33, 0.95, 1.0), (0.33, 0.93, 0.95, 1.0), (0.75, 0.95, 0.33, 1.0), (0.95, 0.33, 0.43, 1.0)]
MAX_WHEEL_VELOCITY = 20.0 # ~200RPM

def obstacle_color(index: int) -> tuple[float, float, float, float]:
    return OBSTACLE_COLORS[index % len(OBSTACLE_COLORS)]


class WheeledRobotNavigationEnv(ManagedEnvironment):
    """
    The Freenove 4WD raspberry pi platform navigating to goal positions while avoiding
    obstacles, using its ultrasonic range sensor.

    Where the `wheeled_robot` example follows velocity commands it is given, this one is
    told only *where to end up and which way to face there*: the policy has to choose its
    own route and speed, turn to line up with the goal heading, and keep clear of the
    obstacles on the way using its ultrasonic range sensor. Reaching a goal immediately
    earns a new one, so a single episode is a string of navigation problems.
    """

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 10,
        max_episode_length_s: int | None = 30,
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
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=10),
            viewer_options=gs.options.ViewerOptions(
                refresh_rate=int(0.5 / self.dt),
                camera_pos=(-1.0, 1.0, 1.0),
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
                file="../wheeled_robot/model/Freenove4WD.xml",
                pos=INITIAL_BODY_POSITION,
                quat=INITIAL_QUAT,
            ),
        )

        # Obstacles: randomly repositioned on every reset
        self.obstacles = [
            self.scene.add_entity(
                gs.morphs.Box(
                    size=OBSTACLE_SIZE,
                    pos=(1.0 + i * 0.5, 0.0, OBSTACLE_SIZE[2] / 2),
                    fixed=True,
                ),
                surface=gs.surfaces.Rough(color=obstacle_color(i)),
            )
            for i in range(NUM_OBSTACLES)
        ]

        # Simulated HC-SR04 ultrasonic sensor, mounted on the head's ultrasonic board.
        ultrasonic_link = self.robot.get_link("Ultrasonic_HC-SR04_PCB")
        self.ultrasonic = self.scene.add_sensor(
            gs.sensors.Raycaster(
                pattern=gs.sensors.SphericalPattern(
                    fov=(15.0, 15.0), # 15 degreens cone
                    n_points=(5, 5),
                ),
                entity_idx=self.robot.idx,
                link_idx_local=ultrasonic_link.idx_local,
                euler_offset=(0.0, -90.0, 0.0),
                pos_offset=(0.0, 0.0, 0.03),
                min_range=0.02,
                max_range=4.0, # 2cm-400cm
                return_points=False,
                noise=0.003,
                resolution=0.003,
                draw_debug=True
            )
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

    def build(self):
        super().build()

        # Set both head servos to their default pose. The policy takes over the pan servo
        # from here; the tilt servo is never commanded again, so it holds this angle.
        self.head_manager.control_dofs_position(self.head_manager.default_dofs_pos)

    def config(self):
        """
        Configure the environment managers
        """
        self.terrain_manager = TerrainManager(self, terrain=self.terrain)

        ##
        # Robot manager
        self.robot_manager = EntityManager(
            self,
            entity=self.robot,
            on_reset={
                "position": {
                    "fn": reset.position(
                        position=INITIAL_BODY_POSITION,
                        quat=INITIAL_QUAT,
                        zero_velocity=True,
                    ),
                },
                # Start facing a random direction, so the goal is not reliably ahead
                # and the policy has to actually read the goal vector to turn correctly.
                "rotation": {
                    "fn": reset.set_rotation(z=(0.0, 2 * math.pi)),
                },
            },
        )

        ##
        # Obstacles: a new layout every reset
        self.obstacle_managers = [
            EntityManager(
                self,
                entity=obstacle,
                on_reset={
                    "position": {
                        "fn": reset.randomize_annulus_position(
                            radius_range=(0.5, 3.0),
                            z=OBSTACLE_SIZE[2] / 2,
                        ),
                    },
                },
            )
            for obstacle in self.obstacles
        ]

        ##
        # Wheel actuation with one action per side
        # This is a skid-steer robot, meaning that it turns by changing the speed/direction
        # of the wheels on either side of the robot.
        self.wheel_motors = ActuatorManager(
            self,
            joint_names=[
                "TT_Motor-[1-4]_axel",
            ],
            kv=1.0,
        )
        self.wheel_action_manager = VelocityActionManager(
            self,
            # Reduce the actions from 4 to 2 by grouping
            # the motors on each side together.
            action_groups=[
                ["TT_Motor-3_axel", "TT_Motor-4_axel"],  # left side
                ["TT_Motor-1_axel", "TT_Motor-2_axel"],  # right side
            ],
            scale={
                # The front and rear motors are mounted opposite of each other,
                # so their actions need to be reversed in order to be turning in the same direction 
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
        self.head_manager = ActuatorManager(
            self,
            joint_names=[
                "servo-2",  # left/right
                "servo_horn-1",  # up/down
            ],
            default_pos={
                "servo-2": 0.0,  # facing straight ahead
                "servo_horn-1": 0.0,  # aimed slightly up, to clear the floor
            },
            kp=8.0,
            kv=0.4,
        )
        self.head_action_manager = PositionActionManager(
            self,
            scale=0.5,
            use_default_offset=True,
            actuator_manager=self.head_manager,
            # Only selecting the side-to-side servo, the up-down servo is kept at the default position
            actuator_joints=["servo-2"],
            clip={
                # Sweeping 7.5 degrees to the lef/right gives full body clearance coverage
                "servo-2": (-math.radians(7.5), math.radians(7.5)),
            }
        )

        ##
        # The goal pose to drive to: a point, and the direction to face once there.
        # Reaching a goal earns a new one, so the robot keeps navigating for the whole
        # episode rather than parking on its first goal. Goals are automatically kept
        # clear of the obstacles and of the robot itself.
        self.pose_command = Pose2dCommand(
            self,
            range={"x": (-2.5, 2.5), "y": (-2.5, 2.5), "heading": (-math.pi, math.pi)},
            goal_reached_threshold=0.2,
            heading_reached_threshold=math.radians(30),
            resample_on_reached=True,
            debug_visualizer=True,
            debug_visualizer_cfg={
                "envs_idx": [0],
            },
        )

        ##
        # Track collisions between the robot and the obstacles
        self.collision_manager = ContactManager(
            self,
            entity=self.robot,
            with_entity=self.obstacles,
        )

        ##
        # Rewards
        #
        # Every reward here pays for *doing* something, never for *being*
        # somewhere: a robot that stops earns exactly zero. See the README for
        # why that matters when the goal is replaced the moment it is reached.
        RewardManager(
            self,
            logging_enabled=True,
            cfg={
                # Closing the distance, measured as the speed the robot is approaching
                # its goal. This is what gets it moving, at any distance from the goal.
                "position_progress": {
                    "weight": 1.0,
                    "fn": rewards.position_progress(
                        pose_cmd_manager=self.pose_command,
                    ),
                },
                # Turning the right way: toward the goal while there is still ground to
                # cover, then toward the goal heading on the final approach. Asking for
                # the goal heading the whole way would have the robot line up early and
                # then try to crab sideways into the goal, which it cannot do.
                "heading_progress": {
                    "weight": 0.5,
                    "fn": rewards.heading_progress(
                        pose_cmd_manager=self.pose_command,
                        lines_up_within=0.75, # How close to the goal the robot should switch from steering
                                              # toward the goal to lining up with the goal heading.
                    ),
                },
                # Arriving: on the goal *and* lined up with it. This is the task, and the
                # only reward paid for reaching a state rather than for making progress.
                "reached_goal": {
                    "weight": 50.0,
                    "fn": rewards.reached_goal(
                        pose_cmd_manager=self.pose_command,
                    ),
                },
                # Twitchy steering is discouraged, but only for the wheels. 
                "action_rate": {
                    "weight": -0.005,
                    "fn": rewards.action_rate_l2(
                        action_manager=self.wheel_action_manager,
                    ),
                },
                # Discourages jerky motion.
                "body_acceleration_exp": {
                    "weight": -0.02,
                    "fn": rewards.body_acceleration_exp(
                        entity_manager=self.robot_manager,
                        sensitivity=0.02,
                    ),
                },
                # Leaving room around an obstacle. Hitting an obstacle ends the episode, 
                # but that only tells the robot it got something wrong once it is far too late to steer. 
                # This gives it something to follow on the way in.
                "keep_clear": {
                    "weight": -2.0,
                    "fn": rewards.keep_clear(
                        entities=self.obstacles,
                        clearance=0.3,
                        entity_manager=self.robot_manager,
                    ),
                },
                "collision": {
                    "weight": -50.0,
                    "fn": rewards.has_contact(
                        contact_manager=self.collision_manager,
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
                "timeout": {
                    "fn": terminations.timeout(),
                    "time_out": True,
                },
                "out_of_bounds": {
                    "time_out": True,
                    "fn": terminations.out_of_bounds(
                        terrain_manager=self.terrain_manager
                    ),
                },
                "collision": {
                    "fn": terminations.has_contact(
                        contact_manager=self.collision_manager,
                        threshold=0.25,
                    ),
                },
            },
        )

        ObservationManager(
            self,
            history_len=2,
            cfg={
                # The goal from the robot's own point of view: how far ahead it is, how
                # far to the left, and how far to turn to face the right way on arrival.
                "goal_pose": {"fn": self.pose_command.observation},
                "angle_velocity": {
                    "fn": lambda env: self.robot_manager.get_angular_velocity(),
                },
                "linear_velocity": {
                    "fn": lambda env: self.robot_manager.get_linear_velocity(),
                },
                "dof_velocity": {
                    "fn": lambda env: self.wheel_action_manager.get_dofs_velocity(),
                    "scale": 0.05,
                },
                "head_pan": {
                    "fn": lambda env: self.head_action_manager.get_dofs_position(),
                },
                "actions": {
                    "fn": observations.current_actions(),
                },
                "ultrasonic": {
                    "fn": observations.raycaster_distance(
                        sensor=self.ultrasonic,
                        reduce="min",
                        normalize=True,
                    ),
                },
            },
        )
