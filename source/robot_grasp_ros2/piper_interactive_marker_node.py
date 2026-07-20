from __future__ import annotations

from contextlib import suppress
from typing import Optional

import rclpy
from geometry_msgs.msg import Pose, PoseStamped
from interactive_markers.interactive_marker_server import InteractiveMarkerServer
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import InteractiveMarker
from visualization_msgs.msg import InteractiveMarkerControl
from visualization_msgs.msg import Marker

from src.robot.client import RobotArmClientConfig


class PiperInteractiveMarkerNode(Node):
    """RViz interactive marker that publishes Piper TCP target poses."""

    def __init__(self) -> None:
        super().__init__("piper_interactive_marker")

        defaults = RobotArmClientConfig()
        self.declare_parameter("frame_id", "base_link")
        self.declare_parameter("ee_link", "gripper_base")
        self.declare_parameter("marker_scale", 0.18)
        self.declare_parameter("publish_topic", defaults.interactive_target_pose_topic)
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("default_position_xyz", [0.30, 0.0, 0.30])

        self._frame_id = str(self.get_parameter("frame_id").value)
        self._ee_link = str(self.get_parameter("ee_link").value)
        self._marker_scale = float(self.get_parameter("marker_scale").value)
        self._publish_topic = str(self.get_parameter("publish_topic").value)
        self._publish_rate_hz = max(1.0, float(self.get_parameter("publish_rate_hz").value))
        default_xyz = [float(v) for v in self.get_parameter("default_position_xyz").value]

        self._pose_pub = self.create_publisher(PoseStamped, self._publish_topic, 10)
        self._server = InteractiveMarkerServer(self, "piper_target_marker")
        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._marker_name = "piper_goal"
        self._current_pose: Optional[Pose] = None
        self._synced_from_link = False

        self._default_pose = Pose()
        self._default_pose.position.x = default_xyz[0]
        self._default_pose.position.y = default_xyz[1]
        self._default_pose.position.z = default_xyz[2]
        self._default_pose.orientation.w = 1.0

        self._create_interactive_marker()

        timer_dt = 1.0 / self._publish_rate_hz
        self._publish_timer = self.create_timer(timer_dt, self._publish_pose_timer)
        self._init_pose_timer = self.create_timer(0.2, self._try_sync_initial_pose_from_tf)

        self.get_logger().info(
            f"Piper interactive marker ready: {self._publish_topic} in frame {self._frame_id}"
        )

    def _create_interactive_marker(self) -> None:
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self._frame_id
        int_marker.name = self._marker_name
        int_marker.description = "Piper Target Pose"
        int_marker.scale = self._marker_scale
        int_marker.pose = self._default_pose

        body_control = InteractiveMarkerControl()
        body_control.always_visible = True
        body_control.interaction_mode = InteractiveMarkerControl.NONE
        body_control.markers.append(self._make_axis_marker("x"))
        body_control.markers.append(self._make_axis_marker("y"))
        body_control.markers.append(self._make_axis_marker("z"))
        int_marker.controls.append(body_control)

        int_marker.controls.append(self._make_control("rotate_x", 1.0, 0.0, 0.0, True))
        int_marker.controls.append(self._make_control("move_x", 1.0, 0.0, 0.0, False))
        int_marker.controls.append(self._make_control("rotate_y", 0.0, 1.0, 0.0, True))
        int_marker.controls.append(self._make_control("move_y", 0.0, 1.0, 0.0, False))
        int_marker.controls.append(self._make_control("rotate_z", 0.0, 0.0, 1.0, True))
        int_marker.controls.append(self._make_control("move_z", 0.0, 0.0, 1.0, False))

        self._server.insert(int_marker, feedback_callback=self._on_feedback)
        self._server.applyChanges()
        self._current_pose = int_marker.pose

    @staticmethod
    def _make_control(
        name: str,
        x: float,
        y: float,
        z: float,
        is_rotate: bool,
    ) -> InteractiveMarkerControl:
        control = InteractiveMarkerControl()
        control.name = name
        control.orientation.w = 1.0
        control.orientation.x = x
        control.orientation.y = y
        control.orientation.z = z
        control.interaction_mode = (
            InteractiveMarkerControl.ROTATE_AXIS
            if is_rotate
            else InteractiveMarkerControl.MOVE_AXIS
        )
        return control

    def _make_axis_marker(self, axis: str) -> Marker:
        marker = Marker()
        marker.type = Marker.ARROW
        marker.scale.x = self._marker_scale * 0.45
        marker.scale.y = self._marker_scale * 0.08
        marker.scale.z = self._marker_scale * 0.08
        marker.color.a = 0.95

        if axis == "x":
            marker.color.r = 1.0
            marker.color.g = 0.1
            marker.color.b = 0.1
            marker.pose.orientation.w = 1.0
        elif axis == "y":
            marker.color.r = 0.1
            marker.color.g = 1.0
            marker.color.b = 0.1
            marker.pose.orientation.w = 0.70710678
            marker.pose.orientation.z = 0.70710678
        else:
            marker.color.r = 0.1
            marker.color.g = 0.3
            marker.color.b = 1.0
            marker.pose.orientation.w = 0.70710678
            marker.pose.orientation.y = -0.70710678

        return marker

    def _on_feedback(self, feedback) -> None:
        self._current_pose = feedback.pose
        self._publish_pose(feedback.pose)

    def _publish_pose_timer(self) -> None:
        if self._current_pose is None:
            return
        self._publish_pose(self._current_pose)

    def _publish_pose(self, pose: Pose) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.pose = pose
        self._pose_pub.publish(msg)

    def _try_sync_initial_pose_from_tf(self) -> None:
        if self._synced_from_link:
            return
        try:
            tf_msg = self._tf_buffer.lookup_transform(
                self._frame_id,
                self._ee_link,
                Time(),
            )
        except TransformException:
            return

        pose = Pose()
        pose.position.x = tf_msg.transform.translation.x
        pose.position.y = tf_msg.transform.translation.y
        pose.position.z = tf_msg.transform.translation.z
        pose.orientation = tf_msg.transform.rotation

        self._server.setPose(self._marker_name, pose)
        self._server.applyChanges()
        self._current_pose = pose
        self._synced_from_link = True
        self.get_logger().info(
            f"Marker initial pose synced from TF: {self._frame_id} -> {self._ee_link}"
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PiperInteractiveMarkerNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        with suppress(KeyboardInterrupt):
            node.destroy_node()
        if rclpy.ok():
            with suppress(KeyboardInterrupt):
                rclpy.shutdown()


if __name__ == "__main__":
    main()
