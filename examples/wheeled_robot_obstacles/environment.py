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
    RewardManager,
    TerminationManager,
    VelocityActionManager,
    VelocityCommandManager,
)
from genesis_forge.managers.terrain_manager import TerrainManager
from genesis_forge.mdp import observations, reset, rewards, terminations

INITIAL_BODY_POSITION = (0.0, 0.0, 0.0458)
INITIAL_QUAT = (1.0, 0.0, 0.0, 0.0)

NUM_OBSTACLES = 6
OBSTACLE_SIZE = (0.15, 0.15, 0.2)
"""Boxes tall enough to be seen by the ultrasonic sensor at its ~10cm mounting height."""

OBSTACLE_RADIUS_RANGE = (0.6, 2.5)
"""Obstacles are scattered in a ring around the robot. The inner radius keeps the
robot's spawn point clear, so an episode never starts inside an obstacle."""

WHEEL_VELOCITY_SCALE = 20.0
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
"""
Aim the head slightly above horizontal.

The sensor sits only ~10cm off the ground, so a level 15-degree cone puts its lowest
rays into the floor about 0.8m out. Since a reading is the *nearest* echo, the floor
would then mask every obstacle past that. Tilting up by just under the cone's
half-angle lifts the lowest ray to roughly -0.5 degrees, which does not reach the
floor until ~12m -- far beyond the sensor's range -- while staying low enough to still
catch obstacles at any distance. Real robots need this same trick.
"""


def _obstacle_colors(count: int) -> list[tuple[float, float, float, float]]:
    """
    A shuffled set of evenly-spaced hues, so the obstacles are easy to tell apart in
    the viewer and in training videos. Purely cosmetic -- the policy never sees color.
    """
    hues = [(i / count + random.random() / count) % 1.0 for i in range(count)]
    random.shuffle(hues)
    return [(*colorsys.hsv_to_rgb(h, 0.65, 0.95), 1.0) for h in hues]


class WheeledRobotObstaclesEnv(ManagedEnvironment):
    """
    The Freenove 4WD raspberry pi platform following velocity commands through a field
    of obstacles, using its ultrasonic range sensor to avoid them.

    This is the `wheeled_robot` example plus three things: obstacles scattered around
    the arena, a simulated ultrasonic sensor mounted on the robot's head, and an
    episode-ending collision penalty.
    """

    def __init__(
        self,
        num_envs: int = 1,
        dt: float = 1 / 50,
        max_episode_length_s: int | None = 15,
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

        # Obstacles. These are immovable, but each environment gets its own layout:
        # `fixed=True` morphs default to `batch_fixed_verts=True`, which is what allows
        # a per-environment position to be set on reset.
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

        # Simulated HC-SR04 ultrasonic sensor.
        #
        # Sensors have to be added before the scene is built, which happens before
        # `config()` is called, so this belongs here rather than with the managers.
        #
        # It is mounted on the head's ultrasonic board, so the beam follows the head if
        # the servos are ever driven. That link's own +Z axis points out of the
        # transducers, so `euler_offset` rotates the raycast pattern (which fires along
        # +X) onto it, and `pos_offset` starts the rays just beyond the transducer
        # housings, which the beam would otherwise hit.
        # The beam is a symmetric cone, like the real sensor's. The head is aimed
        # slightly up (see HEAD_TILT) so the cone clears the floor.
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
                # A real ultrasonic reading is neither exact nor perfectly fine-grained:
                # the HC-SR04 resolves to 0.3cm and is accurate to a few millimeters.
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

        # Give the head servos a position target to hold. The actuator manager sets the
        # PD gains and the starting pose, but never commands a target, and a target set
        # once here persists for the whole run, since nothing else drives these joints.
        self.head_manager.control_dofs_position(self.head_manager.default_dofs_pos)

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
        # Obstacles: each one lands somewhere new in the ring on every reset,
        # so the policy sees a different course each episode.
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
            actuator_manager=self.wheel_motors,
        )

        ##
        # Head servos.
        # No action manager is attached, so the policy does not control the head: the
        # PD controller simply holds it at a fixed pose: facing forward, tilted up by
        # HEAD_TILT so the beam clears the floor. Add a PositionActionManager here to
        # let a policy aim the sensor itself.
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
        # Commanded direction
        self.velocity_command = VelocityCommandManager(
            self,
            range={
                "lin_vel_x": (-0.2, 0.2),  # forward/backward
                "lin_vel_y": (-0.0, 0.0),  # cannot move side-to-side
                "ang_vel_z": (-1.0, 1.0),  # turning
            },
            stopped_probability=0.02,
            resample_time_sec=5.0,
            debug_visualizer=True,
            debug_visualizer_cfg={
                "envs_idx": [0],
            },
        )

        ##
        # Collisions with the obstacles.
        # Filtering by the obstacles, rather than by link name, is what separates a
        # collision from the wheels' constant contact with the ground. That also means
        # every part of the robot counts, so clipping an obstacle with a wheel while
        # turning is caught just like driving into one head-on.
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
                # Hitting an obstacle also ends the episode, so this is the cost of
                # the crash itself rather than an ongoing penalty.
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
                # The robot hit an obstacle. This is a failure, not a time-out, so the
                # value bootstrapping treats it as the dead end that it is.
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
                # The distance to whatever is in front of the robot, scaled to [0, 1].
                # A reading of 1.0 means nothing is within range.
                "ultrasonic": {
                    "fn": observations.raycaster_distance(
                        sensor=self.ultrasonic,
                        normalize=True,
                    ),
                },
            },
        )
