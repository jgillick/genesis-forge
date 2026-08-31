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
    PositionCommandManager,
    RewardManager,
    TerminationManager,
    VelocityActionManager,
)
from genesis_forge.managers.terrain_manager import TerrainManager
from genesis_forge.mdp import observations, reset, rewards, terminations

INITIAL_BODY_POSITION = (0.0, 0.0, 0.0458)
INITIAL_QUAT = (1.0, 0.0, 0.0, 0.0)

NUM_OBSTACLES = 8
OBSTACLE_SIZE = (0.15, 0.15, 0.2)
OBSTACLE_RADIUS_RANGE = (0.6, 2.5)

GOAL_RANGE = {"x": (-2.5, 2.5), "y": (-2.5, 2.5)}
GOAL_REACHED_THRESHOLD = 0.15

GOAL_OBSTACLE_MARGIN = math.hypot(OBSTACLE_SIZE[0], OBSTACLE_SIZE[1]) / 2 + GOAL_REACHED_THRESHOLD
"""
Minimum distance a goal must keep from every obstacle's center: the obstacle's own
half-diagonal (so the goal doesn't land inside its footprint) plus the reach threshold
(so reaching the goal doesn't require clipping the obstacle).
"""

WHEEL_VELOCITY_SCALE = 10.0
"""
Maximum wheel speed, in rad/s, that a full-throttle action commands.

Measured on this platform: the wheels track whatever they are told, but slip caps the
robot at ~0.26 m/s, which it reaches around a scale of 20. Larger scales just spin the
wheels -- a scale of 30 actually measured *slower* than 20.
"""

ULTRASONIC_MAX_RANGE = 4.0
"""The HC-SR04's rated maximum range (2cm - 400cm)."""

ULTRASONIC_BEAM_ANGLE = 15.0
"""The HC-SR04's measuring angle, as a full cone width."""

HEAD_TILT = math.radians(7.0)
"""Aim the head slightly up so the beam cone clears the floor.
See the wheeled_robot_obstacles example for why this is needed."""


def _obstacle_colors(count: int) -> list[tuple[float, float, float, float]]:
    """
    A shuffled set of evenly-spaced hues, so the obstacles are easy to tell apart in
    the viewer and in training videos. Purely cosmetic -- the policy never sees color.
    """
    hues = [(i / count + random.random() / count) % 1.0 for i in range(count)]
    random.shuffle(hues)
    return [(*colorsys.hsv_to_rgb(h, 0.65, 0.95), 1.0) for h in hues]


class WheeledRobotGoalNavEnv(ManagedEnvironment):
    """
    The Freenove 4WD raspberry pi platform navigating to goal positions while avoiding
    obstacles, using its ultrasonic range sensor.

    Where the `wheeled_robot_obstacles` example follows velocity commands it is given,
    this one is told only *where to end up*: the policy has to choose its own heading
    and speed. Reaching a goal immediately earns a new one, so a single episode is a
    string of navigation problems.
    """

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 50,
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
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
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

        # Robot: the platform with the pan/tilt head, which carries the ultrasonic sensor
        self.robot = self.scene.add_entity(
            gs.morphs.MJCF(
                file="../wheeled_robot/model/Freenove4WD_w_sensor.xml",
                pos=INITIAL_BODY_POSITION,
                quat=INITIAL_QUAT,
            ),
        )

        # Obstacles: immovable, but repositioned per environment on every reset
        self.obstacles = [
            self.scene.add_entity(
                gs.morphs.Box(
                    size=OBSTACLE_SIZE,
                    pos=(1.0 + i * 0.5, 0.0, OBSTACLE_SIZE[2] / 2),
                    fixed=True,
                ),
                surface=gs.surfaces.Rough(color=color),
            )
            for i, color in enumerate(_obstacle_colors(NUM_OBSTACLES))
        ]

        # Simulated HC-SR04 ultrasonic sensor, mounted on the head's ultrasonic board.
        # See the wheeled_robot_obstacles example for what the offsets are doing.
        ultrasonic_link = self.robot.get_link("Ultrasonic_HC-SR04_PCB")
        self.ultrasonic = self.scene.add_sensor(
            gs.sensors.Raycaster(
                pattern=gs.sensors.SphericalPattern(
                    fov=(ULTRASONIC_BEAM_ANGLE, ULTRASONIC_BEAM_ANGLE),
                    n_points=(5, 5),
                ),
                entity_idx=self.robot.idx,
                link_idx_local=ultrasonic_link.idx_local,
                euler_offset=(0.0, -90.0, 0.0),
                pos_offset=(0.0, 0.0, 0.03),
                min_range=0.02,
                max_range=ULTRASONIC_MAX_RANGE,
                return_points=False,
                noise=0.003,
                resolution=0.003,
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

        # Hold the head at its configured pose (see wheeled_robot_obstacles)
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
                            radius_range=OBSTACLE_RADIUS_RANGE,
                            z=OBSTACLE_SIZE[2] / 2,
                        ),
                    },
                },
            )
            for obstacle in self.obstacles
        ]

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
            scale=WHEEL_VELOCITY_SCALE,
            clip=(-WHEEL_VELOCITY_SCALE, WHEEL_VELOCITY_SCALE),
            actuator_manager=self.wheel_motors,
        )

        ##
        # Head servos, held facing forward and tilted slightly up
        # (see wheeled_robot_obstacles)
        self.head_manager = ActuatorManager(
            self,
            joint_names=[
                "servo-2",  # left/right
                "servo_horn-1",  # up/down
            ],
            default_pos={
                "servo-2": 0.0,  # facing straight ahead
                "servo_horn-1": HEAD_TILT,  # aimed slightly up, to clear the floor
            },
            kp=8.0,
            kv=0.4,
        )

        ##
        # The goal position to drive to.
        # Reaching a goal earns a new one, so the robot keeps navigating for the whole
        # episode rather than parking on its first goal.
        self.position_command = PositionCommandManager(
            self,
            range=GOAL_RANGE,
            goal_reached_threshold=GOAL_REACHED_THRESHOLD,
            resample_on_reached=True,
            avoid_entities=self.obstacles,
            avoid_margin=GOAL_OBSTACLE_MARGIN,
            debug_visualizer=True,
            debug_visualizer_cfg={
                "envs_idx": [0],
            },
        )

        ##
        # Collisions with the obstacles (see wheeled_robot_obstacles)
        self.collision_manager = ContactManager(
            self,
            entity=self.robot,
            with_entity=self.obstacles,
        )

        ##
        # Rewards
        RewardManager(
            self,
            logging_enabled=True,
            cfg={
                # Being near the goal. This is strongest right at the goal, and is what
                # makes the robot settle there instead of driving past it.
                "position_tracking": {
                    "weight": 1.0,
                    "fn": rewards.position_tracking(
                        position_cmd_manager=self.position_command,
                    ),
                },
                # Closing the distance. Unlike the reward above, this pays off even when
                # the goal is far away, which is what gets the robot moving at all.
                "position_progress": {
                    "weight": 1.0,
                    "fn": rewards.position_progress(
                        position_cmd_manager=self.position_command,
                    ),
                },
                # Arriving
                "reached_goal": {
                    "weight": 10.0,
                    "fn": rewards.reached_goal(
                        position_cmd_manager=self.position_command,
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
                "collision": {
                    "weight": -10.0,
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
                    ),
                },
            },
        )

        ##
        # Observations
        ObservationManager(
            self,
            cfg={
                # Where the goal is, relative to the robot: direction and distance in
                # one vector, in the robot's own frame.
                "goal_vec": {"fn": self.position_command.observation},
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
                # The distance to whatever is ahead, scaled to [0, 1]
                "ultrasonic": {
                    "fn": observations.raycaster_distance(
                        sensor=self.ultrasonic,
                        normalize=True,
                    ),
                },
            },
        )
