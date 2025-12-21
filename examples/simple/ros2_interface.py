import rclpy
from rclpy.node import Node
from rclpy.clock import Clock
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Twist, Pose
import torch


class RosInterface(Node):
    def __init__(self, pos_joints, vel_joints, force_joints):
        super().__init__("ros_interface")
        self._ros_node = self
        self._ros_clock = Clock()

        self._pos_joints = pos_joints
        self._vel_joints = vel_joints
        self._force_joints = force_joints
        self._all_joints = pos_joints + vel_joints + force_joints

        self._num_pos_joints = len(pos_joints)
        self._num_vel_joints = len(vel_joints)
        self._num_force_joints = len(force_joints)
        self._num_joints = (
            self._num_pos_joints + self._num_vel_joints + self._num_force_joints
        )

        self._pos_state = None
        self._vel_state = None
        self._force_state = None

        self._robot_lin_vel = None
        self._robot_ang_vel = None
        self._robot_pos = None
        self._robot_quat = None

        self.setup_ros()

    def setup_ros(self):
        """
        Setup ROS publishers and subscribers.
        Should be called after env.build() so that joint names are available.
        """
        # Initialize action buffers
        self._pos_actions = torch.zeros(self._num_pos_joints)
        self._vel_actions = torch.zeros(self._num_vel_joints)
        self._force_actions = torch.zeros(self._num_force_joints)

        # Setup subscribers and publishers
        self._setup_joint_action_publisher()
        self._setup_robot_twist_subscriber()
        self._setup_robot_pose_subscriber()
        self._setup_joint_state_subscriber()

    def _current_timestep(self):
        """
        Get the current sim time
        """
        return self._ros_clock.now().to_msg()

    def _setup_joint_action_publisher(self):
        print("Joint actions Publisher started")

        def joint_action_callback():
            joint_state_msg = JointState()
            joint_state_msg.header.stamp = self._current_timestep()
            joint_state_msg.name = self._all_joints

            pos_actions = []
            vel_actions = []
            force_actions = []
            for pos_idx in range(self._num_pos_joints):
                pos_actions.append(self._pos_actions[pos_idx].item())
                vel_actions.append(None)
                force_actions.append(None)
            for vel_idx in range(self._num_vel_joints):
                pos_actions.append(None)
                vel_actions.append(self._vel_actions[vel_idx].item())
                force_actions.append(None)
            for force_idx in range(self._num_force_joints):
                pos_actions.append(None)
                vel_actions.append(None)
                force_actions.append(self._force_actions[force_idx].item())

            joint_state_msg.position = pos_actions
            joint_state_msg.velocity = vel_actions
            joint_state_msg.effort = force_actions
            self.joint_state_publisher.publish(joint_state_msg)

        self.joint_state_publisher = self._ros_node.create_publisher(
            JointState, f"/joint_commands", 50
        )
        self.timer = self._ros_node.create_timer(0.01, joint_action_callback)

    def _setup_robot_twist_subscriber(self):
        print("Robot twist Subscriber started")

        def robot_twist_callback(msg):
            self._robot_lin_vel = msg.linear
            self._robot_ang_vel = msg.angular

        self.robot_twist_subscriber = self._ros_node.create_subscription(
            Twist, f"/robot_twist", robot_twist_callback, 100
        )

    def _setup_robot_pose_subscriber(self):
        print("Robot pose Subscriber started")

        def robot_pose_callback(msg):
            self._robot_pos = torch.tensor(
                [msg.position.x, msg.position.y, msg.position.z]
            )
            self._robot_quat = torch.tensor(
                [
                    msg.orientation.w,
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z,
                ]
            )

        self.robot_pose_subscriber = self._ros_node.create_subscription(
            Pose, f"/robot_pose", robot_pose_callback, 100
        )

    def _setup_joint_state_subscriber(self):
        print("Joint state Subscriber started")

        def joint_state_callback(msg):
            pos_vals = []
            vel_vals = []
            eff_vals = []
            # We need to fill values in the order of self.dofs_idx / self.joint_names
            for joint_index in range(len(list(msg.name))):
                if list(msg.position)[joint_index] is not None:
                    pos_vals.append(list(msg.position)[joint_index])
                else:
                    pos_vals.append(0.0)
                if list(msg.velocity)[joint_index] is not None:
                    vel_vals.append(list(msg.velocity)[joint_index])
                else:
                    vel_vals.append(0.0)
                if list(msg.effort)[joint_index] is not None:
                    eff_vals.append(list(msg.effort)[joint_index])
                else:
                    eff_vals.append(0.0)

            self._pos_state = torch.tensor(pos_vals)
            self._vel_state = torch.tensor(vel_vals)
            self._force_state = torch.tensor(eff_vals)

        self._ros_node.create_subscription(
            JointState, f"joint_states", joint_state_callback, 100
        )

    def get_angular_velocity(self):
        if self._robot_ang_vel:
            return torch.tensor(
                [self._robot_ang_vel.x, self._robot_ang_vel.y, self._robot_ang_vel.z]
            ).unsqueeze(0)
        return torch.zeros((1, 3))

    def get_linear_velocity(self):
        if self._robot_lin_vel:
            return torch.tensor(
                [self._robot_lin_vel.x, self._robot_lin_vel.y, self._robot_lin_vel.z]
            ).unsqueeze(0)
        return torch.zeros((1, 3))

    def get_dofs_position(self):
        if self._pos_state is not None:
            return self._pos_state.unsqueeze(0)
        return torch.zeros((1, self._num_joints))

    def get_dofs_velocity(self):
        if self._vel_state is not None:
            return self._vel_state.unsqueeze(0)
        return torch.zeros((1, self._num_joints))

    def get_dofs_force(self):
        if self._force_state is not None:
            return self._force_state.unsqueeze(0)
        return torch.zeros((1, self._num_joints))
