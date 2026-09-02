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
OBSTACLE_RADIUS_RANGE = (0.5, 3.0)

GOAL_RANGE = {"x": (-2.5, 2.5), "y": (-2.5, 2.5), "heading": (-math.pi, math.pi)}
GOAL_REACHED_THRESHOLD = 0.2
GOAL_HEADING_THRESHOLD = math.radians(30)
"""
How closely the robot has to be facing the goal heading to have arrived. Arrival needs
both this and the distance: without the heading, the goal would be replaced the instant
the robot drove into range, and there would be nothing to line up for.

A car-like robot cannot slide sideways to fix its position without turning, so the
tighter this gets, the more shuffling back and forth it takes to satisfy the distance and
the heading at the same time. 30 degrees leaves room to arrive in one movement.
"""

GOAL_LINEUP_DISTANCE = 0.75
"""
How close to the goal the robot should switch from steering *toward* the goal to lining
up with the goal heading. Further out than this, chasing the goal heading would mean
driving sideways, which this robot cannot do.
"""

WHEEL_MOUNTING_SIGN = {
    "TT_Motor-1_axel": -1.0,  # front right
    "TT_Motor-2_axel": +1.0,  # rear right
    "TT_Motor-3_axel": -1.0,  # front left
    "TT_Motor-4_axel": +1.0,  # rear left
}
"""
Which way each wheel has to be driven to roll the robot forwards.

The two front gearboxes are mounted turned around in the model, so the same command spins
a front wheel the opposite way to a rear one. Flipping them here means "positive drives
this wheel forwards" holds for all four, which is what lets one action drive a whole side.
Get this wrong and the front wheels fight the rear ones: the robot still drives, just
noticeably slower, while the tires scrub.
"""

WHEEL_VELOCITY_SCALE = 20.0
"""
Maximum wheel speed, in rad/s, that a full-throttle action commands. 20 rad/s is about
200 RPM, which is what the TT motors on the real platform turn, so this is set by the
hardware rather than chosen.
"""

POLICY_AIMS_HEAD = False
"""
Whether the policy steers the sensor, or it stays pinned facing forwards.

With a single range sensor, one reading says *something is this far away* without saying
which side of the nose it is on -- so a robot that can only look straight ahead has to
work direction out some other way, from how the readings change as it drives and turns.
Aiming the sensor is the direct alternative: look left, look right, find out.

Aiming was tried and it made things worse, for a reason worth understanding before trying
it again. The beam is only 15 degrees wide, so the sensor cannot watch the path ahead and
look to the side at the same time -- those are the same 15 degrees. Measured against the
robot's own collision corridor, a head pointed straight ahead covers 88% of it at 1m and
all of it at 2m; by 15 degrees off-axis that is down to 6%, and past 25 degrees the robot
is driving completely blind.

Nothing rewards the policy for looking anywhere useful either: the collision penalties are
worked out from true positions, not from the sensor, so the head action has no gradient
tying it to outcomes. It drifts, parks off to one side, and the robot drives blind. With
the head held forwards the path is always the thing being watched.
"""



def _obstacle_colors(count: int) -> list[tuple[float, float, float, float]]:
    """
    A shuffled set of evenly-spaced hues, so the obstacles are easy to tell apart in
    the viewer and in training videos. Purely cosmetic -- the policy never sees color.
    """
    hues = [(i / count + random.random() / count) % 1.0 for i in range(count)]
    random.shuffle(hues)
    return [(*colorsys.hsv_to_rgb(h, 0.65, 0.95), 1.0) for h in hues]


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
                file="../wheeled_robot/model/Freenove4WD.xml",
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
        # The settings come from the datasheet: a 15 degree measuring angle, a 2cm-400cm
        # range, 0.3cm resolution and ~3mm accuracy.
        #
        # Two offsets are worth knowing about if you mount a sensor on your own robot:
        #  - `euler_offset` orients the beam. Ray patterns fire along the sensor frame's
        #    +X axis, but this board's +Z is the axis pointing out of the transducers, so
        #    the pattern is rotated onto it. If your beam comes out sideways, this is why.
        #  - `pos_offset` starts the rays clear of the robot. Rays hit everything,
        #    including the robot carrying them, and `min_range` does not suppress those
        #    self-hits -- without the 3cm offset every ray would stop on the transducer
        #    housings about 8mm out.
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
        # One action per side, not one per wheel. This is a skid-steer robot: the two
        # wheels down each side are bolted to the same chassis and can only usefully turn
        # together, so a policy with four independent wheel commands spends its capacity
        # learning not to make them fight each other. Two actions -- left side, right
        # side -- is how the robot actually moves: same sign to drive, opposite to spin.
        #
        # The per-wheel scale carries the mounting sign, so that within a side both wheels
        # roll the same way for one action (see WHEEL_MOUNTING_SIGN).
        self.wheel_action_manager = VelocityActionManager(
            self,
            scale={
                name: sign * WHEEL_VELOCITY_SCALE
                for name, sign in WHEEL_MOUNTING_SIGN.items()
            },
            clip=(-WHEEL_VELOCITY_SCALE, WHEEL_VELOCITY_SCALE),
            actuator_manager=self.wheel_motors,
            action_groups=[
                ["TT_Motor-3_axel", "TT_Motor-4_axel"],  # left side
                ["TT_Motor-1_axel", "TT_Motor-2_axel"],  # right side
            ],
        )

        ##
        # Head servos. The tilt is always held at a fixed angle. The pan servo is either
        # held facing forwards too, or handed to the policy -- see POLICY_AIMS_HEAD.
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
        # `use_default_offset` centers the action range on "straight ahead", so a zero
        # action points the sensor where a fixed head would sit, and the policy learns a
        # deflection from there. The tilt servo is left out, so it keeps the target set in
        # `build()` and stays where it is put -- and so does the pan servo when the policy
        # isn't given it.
        self.head_action_manager = None
        if POLICY_AIMS_HEAD:
            self.head_action_manager = PositionActionManager(
                self,
                scale=0.5,
                use_default_offset=True,
                actuator_manager=self.head_manager,
                actuator_joints=["servo-2"],
                clip={
                    "servo-2": (-math.radians(30), math.radians(30)),
                }
            )

        ##
        # The goal pose to drive to: a point, and the direction to face once there.
        # Reaching a goal earns a new one, so the robot keeps navigating for the whole
        # episode rather than parking on its first goal. Goals are automatically kept
        # clear of the obstacles and of the robot itself.
        self.pose_command = Pose2dCommand(
            self,
            range=GOAL_RANGE,
            goal_reached_threshold=GOAL_REACHED_THRESHOLD,
            heading_reached_threshold=GOAL_HEADING_THRESHOLD,
            resample_on_reached=True,
            debug_visualizer=True,
            debug_visualizer_cfg={
                "envs_idx": [0],
            },
        )

        ##
        # Collisions. The robot is always touching the ground, so "is anything touching
        # the robot" is not a useful test. Tracking every link but filtering the contacts
        # down to the obstacles is what separates a crash from normal driving -- and it
        # means the whole robot counts, so clipping a box with a wheel mid-turn registers
        # just like driving into one head-on.
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
                # Every reward here pays for *doing* something, never for *being*
                # somewhere: a robot that stops earns exactly zero. See the README for
                # why that matters when the goal is replaced the moment it is reached.
                #
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
                        lines_up_within=GOAL_LINEUP_DISTANCE,
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
                # Twitchy steering is discouraged, but only for the wheels. When the
                # policy aims the head, sweeping the sensor around is exactly the
                # behavior wanted, and counting its changes here would tax it for looking.
                "action_rate": {
                    "weight": -0.005,
                    "fn": rewards.action_rate_l2(
                        action_manager=self.wheel_action_manager,
                    ),
                },
                # Discourages jerky motion. The sensitivity matters as much as the weight:
                # at the default this penalty saturates -- every plausible motion scores
                # near its ceiling, so it stops telling smooth apart from jerky and just
                # taxes moving at all. That is the opposite of what this task needs, since
                # arriving on a pose takes brisk turning.
                "body_acceleration_exp": {
                    "weight": -0.02,
                    "fn": rewards.body_acceleration_exp(
                        entity_manager=self.robot_manager,
                        sensitivity=0.02,
                    ),
                },
                # Leaving room. Hitting an obstacle ends the episode, but that only tells
                # the robot it got something wrong once it is far too late to steer. This
                # gives it something to follow on the way in.
                "keep_clear": {
                    "weight": -2.0,
                    "fn": rewards.keep_clear(
                        entities=self.obstacles,
                        clearance=0.25, # The robot is about 0.25m across and the obstacles 0.15m
                                        # so they touch at around 0.2m -- this leaves a modest band of warning before that.
                        entity_manager=self.robot_manager,
                    ),
                },
                "collision": {
                    "weight": -25.0,
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
        observation_cfg = {
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
            "actions": {
                "fn": observations.current_actions(),
            },
        }

        # Where the sensor is currently pointing. Only worth observing when the policy
        # can aim it: with a fixed head this is the same number every step, and a
        # constant teaches nothing while still costing a slot in every stacked frame.
        if POLICY_AIMS_HEAD:
            observation_cfg["head_pan"] = {
                "fn": lambda env: self.head_action_manager.get_dofs_position(),
            }

        ObservationManager(
            self,
            # One range reading only means something alongside where the sensor was
            # pointing when it was taken, and a single reading cannot place an obstacle
            # on its own. The history is what lets those readings add up to a picture --
            # from a sweep when the policy aims the head, and from the robot's own motion
            # when it doesn't.
            history_len=2,
            cfg={
                **observation_cfg,
                # The distance to whatever the sensor is pointed at, scaled to [0, 1]
                "ultrasonic": {
                    "fn": observations.raycaster_distance(
                        sensor=self.ultrasonic,
                        # A real HC-SR04 reports one nearest echo, not a picture of the
                        # cone, so the rays collapse to their minimum
                        reduce="min",
                        normalize=True,
                    ),
                },
            },
        )
