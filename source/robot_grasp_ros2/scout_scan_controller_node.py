from __future__ import annotations

import math
import threading
import time

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from robot_grasp_msgs.srv import MoveBaseRelative


def _normalize_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


def _yaw_from_odometry(message: Odometry) -> float:
    q = message.pose.pose.orientation
    return math.atan2(
        2.0 * ((q.w * q.z) + (q.x * q.y)),
        1.0 - (2.0 * ((q.y * q.y) + (q.z * q.z))),
    )


class ScoutScanControllerNode(Node):
    """Bounded odometry-closed-loop adapter for placement-area scanning."""

    def __init__(self) -> None:
        super().__init__("base_scan_controller")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("odom_topic", "/odom")
        self.declare_parameter("command_rate_hz", 20.0)
        self.declare_parameter("odom_stale_timeout_s", 0.5)
        self.declare_parameter("max_distance_m", 0.45)
        self.declare_parameter("max_speed_mps", 0.06)
        self.declare_parameter("min_speed_mps", 0.02)
        self.declare_parameter("position_tolerance_m", 0.012)
        self.declare_parameter("max_lateral_error_m", 0.05)
        self.declare_parameter("max_yaw_error_deg", 3.0)

        callback_group = ReentrantCallbackGroup()
        self._cmd_pub = self.create_publisher(
            Twist,
            str(self.get_parameter("cmd_vel_topic").value),
            10,
        )
        self._status_pub = self.create_publisher(String, "~/status", 10)
        self.create_subscription(
            Odometry,
            str(self.get_parameter("odom_topic").value),
            self._on_odometry,
            20,
            callback_group=callback_group,
        )
        self.create_service(
            MoveBaseRelative,
            "~/move_relative",
            self._on_move_relative,
            callback_group=callback_group,
        )
        self.create_service(
            Trigger,
            "~/stop",
            self._on_stop,
            callback_group=callback_group,
        )
        self._odom_lock = threading.Lock()
        self._latest_odom: tuple[float, float, float, float] | None = None
        self._motion_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._publish_status("idle")

    def _publish_status(self, text: str) -> None:
        self.get_logger().info(text)
        self._status_pub.publish(String(data=text))

    def _on_odometry(self, message: Odometry) -> None:
        position = message.pose.pose.position
        with self._odom_lock:
            self._latest_odom = (
                float(position.x),
                float(position.y),
                _yaw_from_odometry(message),
                time.monotonic(),
            )

    def _odom_snapshot(self) -> tuple[float, float, float]:
        with self._odom_lock:
            snapshot = self._latest_odom
        if snapshot is None:
            raise RuntimeError("Scout /odom has not been received")
        x, y, yaw, received_at = snapshot
        stale_limit = float(self.get_parameter("odom_stale_timeout_s").value)
        if time.monotonic() - received_at > stale_limit:
            raise RuntimeError("Scout /odom is stale")
        return x, y, yaw

    def _publish_velocity(self, linear_x: float) -> None:
        command = Twist()
        command.linear.x = float(linear_x)
        self._cmd_pub.publish(command)

    def _stop_base(self) -> None:
        for _ in range(5):
            self._publish_velocity(0.0)
            time.sleep(0.02)

    def _on_stop(self, _request, response):
        self._stop_event.set()
        self._stop_base()
        response.success = True
        response.message = "Scout scan motion stop requested; zero velocity published"
        self._publish_status("stopped")
        return response

    def _on_move_relative(self, request, response):
        if not self._motion_lock.acquire(blocking=False):
            response.success = False
            response.message = "another Scout scan motion is active"
            return response
        self._stop_event.clear()
        traveled = 0.0
        lateral = 0.0
        yaw_error = 0.0
        try:
            distance = float(request.distance_m)
            requested_speed = abs(float(request.speed_mps))
            timeout_s = max(1.0, float(request.timeout_s))
            max_distance = float(self.get_parameter("max_distance_m").value)
            max_speed = float(self.get_parameter("max_speed_mps").value)
            min_speed = float(self.get_parameter("min_speed_mps").value)
            tolerance = float(self.get_parameter("position_tolerance_m").value)
            if abs(distance) < tolerance:
                response.success = True
                response.message = "target is already within tolerance"
                return response
            if abs(distance) > max_distance:
                raise RuntimeError(
                    f"requested distance {distance:.3f}m exceeds safety limit {max_distance:.3f}m"
                )
            if not 0.0 < requested_speed <= max_speed:
                raise RuntimeError(
                    f"requested speed {requested_speed:.3f}m/s is outside (0,{max_speed:.3f}]"
                )

            start_x, start_y, start_yaw = self._odom_snapshot()
            direction = 1.0 if distance > 0.0 else -1.0
            rate_hz = max(5.0, float(self.get_parameter("command_rate_hz").value))
            deadline = time.monotonic() + timeout_s
            self._publish_status(
                f"moving: distance={distance:.3f}m speed<={requested_speed:.3f}m/s"
            )
            while time.monotonic() < deadline:
                if self._stop_event.is_set():
                    raise RuntimeError("Scout scan motion stopped by operator")
                x, y, yaw = self._odom_snapshot()
                dx, dy = x - start_x, y - start_y
                traveled = (dx * math.cos(start_yaw)) + (dy * math.sin(start_yaw))
                lateral = (-dx * math.sin(start_yaw)) + (dy * math.cos(start_yaw))
                yaw_error = math.degrees(_normalize_angle(yaw - start_yaw))
                if abs(lateral) > float(self.get_parameter("max_lateral_error_m").value):
                    raise RuntimeError(f"Scout lateral drift too large: {lateral:.3f}m")
                if abs(yaw_error) > float(self.get_parameter("max_yaw_error_deg").value):
                    raise RuntimeError(f"Scout yaw drift too large: {yaw_error:.2f}deg")
                remaining = distance - traveled
                if direction * remaining <= tolerance:
                    response.success = True
                    response.message = (
                        f"Scout relative move complete: target={distance:.3f}m "
                        f"traveled={traveled:.3f}m"
                    )
                    break
                speed = min(
                    requested_speed,
                    max(min_speed, abs(remaining) * 0.8),
                )
                self._publish_velocity(direction * speed)
                time.sleep(1.0 / rate_hz)
            else:
                raise RuntimeError(
                    f"Scout relative move timed out after {timeout_s:.1f}s "
                    f"(traveled={traveled:.3f}m target={distance:.3f}m)"
                )
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        finally:
            self._stop_base()
            response.traveled_m = float(traveled)
            response.lateral_error_m = float(lateral)
            response.yaw_error_deg = float(yaw_error)
            self._publish_status(
                "idle"
                if bool(response.success)
                else f"failed: {response.message}"
            )
            self._motion_lock.release()
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ScoutScanControllerNode()
    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_base()
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
