from __future__ import annotations

from contextlib import suppress
import math
import threading
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from std_msgs.msg import String
from tf2_ros import Buffer, TransformException, TransformListener

from src.robot import FakeRobotArmClient, Ros2PiperClient
from src.robot.client import RobotArmClient, RobotArmClientConfig
from src.robot.types import EndPoseMMDeg


class PiperPoseBridgeNode(Node):
    """Bridge RViz target poses into the existing RobotArmClient pose API."""

    def __init__(self) -> None:
        super().__init__("piper_pose_bridge")

        defaults = RobotArmClientConfig()
        self.declare_parameter("robot_backend", "fake")
        self.declare_parameter("auto_enable", True)
        self.declare_parameter("command_frame", "base_link")
        self.declare_parameter("target_pose_topic", defaults.interactive_target_pose_topic)
        self.declare_parameter("command_pose_topic", defaults.interactive_command_pose_topic)
        self.declare_parameter("speed_percent", 25.0)
        self.declare_parameter("command_period_s", 0.20)
        self.declare_parameter("min_translation_delta_mm", 2.0)
        self.declare_parameter("min_rotation_delta_deg", 1.0)
        self.declare_parameter("tf_lookup_timeout_s", 0.10)

        self._backend = str(self.get_parameter("robot_backend").value or "fake").strip().lower()
        self._auto_enable = bool(self.get_parameter("auto_enable").value)
        self._command_frame = str(self.get_parameter("command_frame").value)
        self._target_pose_topic = str(self.get_parameter("target_pose_topic").value)
        self._command_pose_topic = str(self.get_parameter("command_pose_topic").value)
        self._speed_percent = float(self.get_parameter("speed_percent").value)
        self._command_period_s = max(0.05, float(self.get_parameter("command_period_s").value))
        self._min_translation_delta_mm = float(self.get_parameter("min_translation_delta_mm").value)
        self._min_rotation_delta_deg = float(self.get_parameter("min_rotation_delta_deg").value)
        self._tf_lookup_timeout_s = max(0.01, float(self.get_parameter("tf_lookup_timeout_s").value))

        self._status_pub = self.create_publisher(String, "~/status", 20)
        self._command_pose_pub = self.create_publisher(PoseStamped, self._command_pose_topic, 10)
        self._target_sub = self.create_subscription(
            PoseStamped,
            self._target_pose_topic,
            self._on_target_pose,
            10,
        )

        self._tf_buffer = Buffer()
        self._tf_listener = TransformListener(self._tf_buffer, self)

        self._robot = self._create_robot_client()
        self._robot_ready = False

        self._lock = threading.Lock()
        self._pending_target: Optional[EndPoseMMDeg] = None
        self._last_sent_target: Optional[EndPoseMMDeg] = None

        self._connect_robot()
        self._command_timer = self.create_timer(self._command_period_s, self._flush_pending_target)

        self._publish_status(
            f"ready backend={self._backend} topic={self._target_pose_topic} frame={self._command_frame}"
        )

    def _publish_status(self, text: str) -> None:
        self.get_logger().info(text)
        self._status_pub.publish(String(data=text))

    def _create_robot_client(self) -> RobotArmClient:
        if self._backend == "fake":
            return FakeRobotArmClient()
        if self._backend == "ros2":
            client = Ros2PiperClient()
            client.attach_ros_node(self)
            return client
        raise RuntimeError(f"unsupported robot backend: {self._backend}")

    def _connect_robot(self) -> None:
        self._robot.connect()
        if self._auto_enable and not self._robot.enable():
            raise RuntimeError("robot enable failed")
        self._robot_ready = True

    def _on_target_pose(self, msg: PoseStamped) -> None:
        try:
            transformed = self._transform_pose_to_command_frame(msg)
            target = self._pose_msg_to_end_pose(transformed)
        except Exception as exc:
            self._publish_status(f"target rejected: {exc}")
            return

        self._command_pose_pub.publish(transformed)
        with self._lock:
            self._pending_target = target

    def _flush_pending_target(self) -> None:
        if not self._robot_ready:
            return

        with self._lock:
            target = self._pending_target
            last_sent = self._last_sent_target

        if target is None:
            return
        if last_sent is not None and not self._target_changed(last_sent, target):
            return

        try:
            self._robot.move_end_pose_mm_deg(
                x_mm=target.x_mm,
                y_mm=target.y_mm,
                z_mm=target.z_mm,
                roll_deg=target.roll_deg,
                pitch_deg=target.pitch_deg,
                yaw_deg=target.yaw_deg,
                speed_percent=self._speed_percent,
            )
        except Exception as exc:
            self._publish_status(f"command failed: {exc}")
            return

        with self._lock:
            self._last_sent_target = target
        self._publish_status(
            "commanded "
            f"({target.x_mm:.1f}, {target.y_mm:.1f}, {target.z_mm:.1f}, "
            f"{target.roll_deg:.1f}, {target.pitch_deg:.1f}, {target.yaw_deg:.1f}) mm/deg"
        )

    def _target_changed(self, previous: EndPoseMMDeg, current: EndPoseMMDeg) -> bool:
        dx = current.x_mm - previous.x_mm
        dy = current.y_mm - previous.y_mm
        dz = current.z_mm - previous.z_mm
        dpos_mm = math.sqrt(dx * dx + dy * dy + dz * dz)

        droll = self._angle_diff_deg(current.roll_deg, previous.roll_deg)
        dpitch = self._angle_diff_deg(current.pitch_deg, previous.pitch_deg)
        dyaw = self._angle_diff_deg(current.yaw_deg, previous.yaw_deg)
        drot_deg = math.sqrt(droll * droll + dpitch * dpitch + dyaw * dyaw)

        return (
            dpos_mm >= self._min_translation_delta_mm
            or drot_deg >= self._min_rotation_delta_deg
        )

    def _transform_pose_to_command_frame(self, msg: PoseStamped) -> PoseStamped:
        source_frame = msg.header.frame_id or self._command_frame
        if source_frame == self._command_frame:
            normalized = PoseStamped()
            normalized.header = msg.header
            normalized.header.frame_id = self._command_frame
            normalized.pose = msg.pose
            return normalized

        stamp = (
            Time.from_msg(msg.header.stamp)
            if (msg.header.stamp.sec or msg.header.stamp.nanosec)
            else Time()
        )
        tf_msg = self._tf_buffer.lookup_transform(
            self._command_frame,
            source_frame,
            stamp,
            timeout=Duration(seconds=self._tf_lookup_timeout_s),
        )
        T_command_source = self._transform_stamped_to_matrix(tf_msg)
        T_source_target = self._pose_to_matrix(msg.pose)
        T_command_target = T_command_source @ T_source_target

        normalized = PoseStamped()
        normalized.header.stamp = self.get_clock().now().to_msg()
        normalized.header.frame_id = self._command_frame
        normalized.pose = self._matrix_to_pose(T_command_target)
        return normalized

    @classmethod
    def _pose_msg_to_end_pose(cls, msg: PoseStamped) -> EndPoseMMDeg:
        roll_deg, pitch_deg, yaw_deg = cls._quat_to_euler_deg(
            msg.pose.orientation.x,
            msg.pose.orientation.y,
            msg.pose.orientation.z,
            msg.pose.orientation.w,
        )
        return EndPoseMMDeg(
            x_mm=msg.pose.position.x * 1000.0,
            y_mm=msg.pose.position.y * 1000.0,
            z_mm=msg.pose.position.z * 1000.0,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
        )

    @staticmethod
    def _quat_to_rot(x: float, y: float, z: float, w: float) -> np.ndarray:
        norm = x * x + y * y + z * z + w * w
        if norm < 1e-12:
            raise ValueError("invalid quaternion norm")
        s = 2.0 / norm
        xx, yy, zz = x * x * s, y * y * s, z * z * s
        xy, xz, yz = x * y * s, x * z * s, y * z * s
        wx, wy, wz = w * x * s, w * y * s, w * z * s
        rot = np.eye(3, dtype=float)
        rot[0, 0] = 1.0 - (yy + zz)
        rot[0, 1] = xy - wz
        rot[0, 2] = xz + wy
        rot[1, 0] = xy + wz
        rot[1, 1] = 1.0 - (xx + zz)
        rot[1, 2] = yz - wx
        rot[2, 0] = xz - wy
        rot[2, 1] = yz + wx
        rot[2, 2] = 1.0 - (xx + yy)
        return rot

    @classmethod
    def _quat_to_euler_deg(
        cls,
        x: float,
        y: float,
        z: float,
        w: float,
    ) -> tuple[float, float, float]:
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2.0 * (w * y - z * x)
        if abs(sinp) >= 1.0:
            pitch = math.copysign(math.pi / 2.0, sinp)
        else:
            pitch = math.asin(sinp)

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        return tuple(math.degrees(v) for v in (roll, pitch, yaw))

    @staticmethod
    def _rot_to_quat(rot: np.ndarray) -> tuple[float, float, float, float]:
        trace = float(rot[0, 0] + rot[1, 1] + rot[2, 2])
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (rot[2, 1] - rot[1, 2]) / s
            y = (rot[0, 2] - rot[2, 0]) / s
            z = (rot[1, 0] - rot[0, 1]) / s
            return x, y, z, w
        if rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
            return x, y, z, w
        if rot[1, 1] > rot[2, 2]:
            s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
            return x, y, z, w
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s
        return x, y, z, w

    @classmethod
    def _pose_to_matrix(cls, pose) -> np.ndarray:
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = cls._quat_to_rot(
            pose.orientation.x,
            pose.orientation.y,
            pose.orientation.z,
            pose.orientation.w,
        )
        transform[0, 3] = float(pose.position.x)
        transform[1, 3] = float(pose.position.y)
        transform[2, 3] = float(pose.position.z)
        return transform

    @classmethod
    def _matrix_to_pose(cls, transform: np.ndarray):
        from geometry_msgs.msg import Pose

        pose = Pose()
        pose.position.x = float(transform[0, 3])
        pose.position.y = float(transform[1, 3])
        pose.position.z = float(transform[2, 3])
        qx, qy, qz, qw = cls._rot_to_quat(transform[:3, :3])
        pose.orientation.x = qx
        pose.orientation.y = qy
        pose.orientation.z = qz
        pose.orientation.w = qw
        return pose

    @classmethod
    def _transform_stamped_to_matrix(cls, tf_msg) -> np.ndarray:
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = cls._quat_to_rot(
            tf_msg.transform.rotation.x,
            tf_msg.transform.rotation.y,
            tf_msg.transform.rotation.z,
            tf_msg.transform.rotation.w,
        )
        transform[0, 3] = float(tf_msg.transform.translation.x)
        transform[1, 3] = float(tf_msg.transform.translation.y)
        transform[2, 3] = float(tf_msg.transform.translation.z)
        return transform

    @staticmethod
    def _angle_diff_deg(a_deg: float, b_deg: float) -> float:
        return (a_deg - b_deg + 180.0) % 360.0 - 180.0

    def close(self) -> None:
        try:
            self._robot.disconnect()
        finally:
            self._robot_ready = False


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PiperPoseBridgeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        with suppress(KeyboardInterrupt):
            node.close()
        with suppress(KeyboardInterrupt):
            node.destroy_node()
        if rclpy.ok():
            with suppress(KeyboardInterrupt):
                rclpy.shutdown()


if __name__ == "__main__":
    main()
