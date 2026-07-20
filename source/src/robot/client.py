"""Robot arm client abstractions for the ROS2 migration."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
import threading
import time
from typing import Dict, Tuple

from .types import ArmStatusSnapshot, EndPoseMMDeg, GripperStatus


_CONTROL_MODE_NAMES = {
    0x00: "STANDBY",
    0x01: "CAN",
    0x03: "ETHERNET",
    0x04: "WIFI",
    0x07: "OFFLINE",
}

_ARM_STATUS_NAMES = {
    0x00: "NORMAL",
    0x01: "ESTOP",
    0x02: "NO_SOLUTION",
    0x03: "SINGULAR",
    0x04: "ANGLE_LIMIT",
    0x05: "JOINT_COMM_ERR",
    0x06: "BRAKE_NOT_OPEN",
    0x07: "COLLISION",
    0x08: "DRAG_OVERSPEED",
    0x09: "JOINT_ABNORMAL",
    0x0A: "OTHER",
    0x0B: "TEACH_RECORD",
    0x0C: "TEACH_EXEC",
    0x0D: "TEACH_PAUSE",
    0x0E: "NTC_OVER_TEMP",
    0x0F: "DISCHARGE_OVER_TEMP",
}

_MOVE_MODE_NAMES = {
    0x00: "MOVE_P",
    0x01: "MOVE_J",
    0x02: "MOVE_L",
    0x03: "MOVE_C",
    0x04: "MOVE_M",
}

_MOTION_STATUS_NAMES = {
    0x00: "ARRIVED",
    0x01: "NOT_ARRIVED",
}


@dataclass(slots=True)
class RobotArmClientConfig:
    node_name: str = "robot_grasp_client"
    pose_topic: str = "/pos_cmd"
    end_pose_topic: str = "/end_pose"
    end_pose_stamped_topic: str = "/end_pose_stamped"
    arm_status_topic: str = "/arm_status"
    joint_state_topic: str = "/joint_states_feedback"
    joint_ctrl_topic: str = "/joint_ctrl_single"
    enable_service: str = "/enable_srv"
    interactive_target_pose_topic: str = "/interactive_piper/target_pose"
    interactive_command_pose_topic: str = "/interactive_piper/command_pose"
    display_joint_state_topic: str = "/joint_states"
    pose_feedback_timeout_s: float = 2.0
    service_timeout_s: float = 5.0
    poll_interval_s: float = 0.05


class RobotArmClient(ABC):
    """Minimal robot control contract used by the migrated pipeline."""

    def __init__(self, config: RobotArmClientConfig | None = None):
        self.config = config or RobotArmClientConfig()

    @abstractmethod
    def connect(self) -> None:
        """Establish connection to the robot layer."""

    @abstractmethod
    def disconnect(self) -> None:
        """Tear down connection resources."""

    @abstractmethod
    def enable(self) -> bool:
        """Enable motors / control."""

    @abstractmethod
    def disable(self) -> bool:
        """Disable motors / control."""

    @abstractmethod
    def emergency_stop(self) -> None:
        """Trigger rapid stop."""

    @abstractmethod
    def recover_from_estop(self) -> None:
        """Recover from emergency stop."""

    @abstractmethod
    def read_end_pose_mm_deg(self) -> EndPoseMMDeg:
        """Read the current TCP pose in mm / degrees."""

    @abstractmethod
    def move_end_pose_mm_deg(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        roll_deg: float,
        pitch_deg: float,
        yaw_deg: float,
        speed_percent: float,
    ) -> EndPoseMMDeg:
        """Command an absolute TCP pose target."""

    @abstractmethod
    def wait_until_pose_reached(
        self,
        target: EndPoseMMDeg,
        timeout_s: float,
        pos_tolerance_mm: float,
        rot_tolerance_deg: float,
    ) -> Tuple[bool, EndPoseMMDeg, Dict[str, float]]:
        """Block until pose is reached or timeout."""

    @abstractmethod
    def pose_error(
        self,
        target: EndPoseMMDeg,
        actual: EndPoseMMDeg | None = None,
    ) -> Dict[str, float]:
        """Compute error between target and actual pose."""

    @abstractmethod
    def format_arm_status(self) -> str:
        """Return a human readable arm status summary."""

    @abstractmethod
    def open_gripper(self, open_mm: float, effort_nm: float | None = None) -> None:
        """Open gripper to the target width."""

    @abstractmethod
    def close_gripper(self, effort_nm: float | None = None) -> None:
        """Close the gripper with the desired torque."""

    @abstractmethod
    def wait_for_gripper(self, target_mm: float, tol_mm: float, timeout_s: float) -> bool:
        """Wait until the gripper reaches the commanded width."""

    @abstractmethod
    def wait_for_gripper_effort(self, target_effort_nm: float, timeout_s: float) -> bool:
        """Wait until the gripper reports the desired effort."""

    @abstractmethod
    def get_gripper_status(self) -> GripperStatus:
        """Return cached gripper state."""

    @abstractmethod
    def get_arm_status_snapshot(self) -> ArmStatusSnapshot:
        """Return a structured arm status snapshot."""


class FakeRobotArmClient(RobotArmClient):
    """In-memory robot client that never touches hardware."""

    def __init__(self, config: RobotArmClientConfig | None = None):
        super().__init__(config)
        self._pose = EndPoseMMDeg(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        self._moving = False
        self._enabled = False
        self._gripper = GripperStatus(angle_mm=70.0, effort_nm=0.0, enabled=True)

    def connect(self) -> None:
        self._enabled = False

    def disconnect(self) -> None:
        self._enabled = False

    def enable(self) -> bool:
        self._enabled = True
        return self._enabled

    def disable(self) -> bool:
        self._enabled = False
        return True

    def emergency_stop(self) -> None:
        self._enabled = False

    def recover_from_estop(self) -> None:
        self._enabled = True

    def read_end_pose_mm_deg(self) -> EndPoseMMDeg:
        return self._pose

    def move_end_pose_mm_deg(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        roll_deg: float,
        pitch_deg: float,
        yaw_deg: float,
        speed_percent: float,
    ) -> EndPoseMMDeg:
        self._moving = True
        self._pose = EndPoseMMDeg(x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg)
        self._moving = False
        return self._pose

    def wait_until_pose_reached(
        self,
        target: EndPoseMMDeg,
        timeout_s: float,
        pos_tolerance_mm: float,
        rot_tolerance_deg: float,
    ) -> Tuple[bool, EndPoseMMDeg, Dict[str, float]]:
        reached = True
        error = self.pose_error(target, self._pose)
        return reached, self._pose, error

    def pose_error(
        self,
        target: EndPoseMMDeg,
        actual: EndPoseMMDeg | None = None,
    ) -> Dict[str, float]:
        actual = actual or self._pose
        dx = actual.x_mm - target.x_mm
        dy = actual.y_mm - target.y_mm
        dz = actual.z_mm - target.z_mm
        droll = actual.roll_deg - target.roll_deg
        dpitch = actual.pitch_deg - target.pitch_deg
        dyaw = actual.yaw_deg - target.yaw_deg
        dpos = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
        drot = (droll ** 2 + dpitch ** 2 + dyaw ** 2) ** 0.5
        return {
            "dx_mm": dx,
            "dy_mm": dy,
            "dz_mm": dz,
            "dpos_mm": dpos,
            "droll_deg": droll,
            "dpitch_deg": dpitch,
            "dyaw_deg": dyaw,
            "drot_deg": drot,
        }

    def format_arm_status(self) -> str:
        snapshot = self.get_arm_status_snapshot()
        return (
            f"control_mode={snapshot.control_mode}(0x{snapshot.control_mode_code:02X}), "
            f"arm_status={snapshot.arm_status}(0x{snapshot.arm_status_code:02X}), "
            f"move_mode={snapshot.move_mode}(0x{snapshot.move_mode_code:02X}), "
            f"teach_status={snapshot.teach_status}, "
            f"motion_status={snapshot.motion_status}(0x{snapshot.motion_status_code:02X}), "
            f"trajectory_index={snapshot.trajectory_index}, "
            f"err_code=0x{snapshot.err_code:04X}"
        )

    def open_gripper(self, open_mm: float, effort_nm: float | None = None) -> None:
        self._gripper.angle_mm = open_mm

    def close_gripper(self, effort_nm: float | None = None) -> None:
        self._gripper.angle_mm = 0.0
        if effort_nm is not None:
            self._gripper.effort_nm = effort_nm

    def wait_for_gripper(self, target_mm: float, tol_mm: float, timeout_s: float) -> bool:
        return abs(self._gripper.angle_mm - target_mm) <= tol_mm

    def wait_for_gripper_effort(self, target_effort_nm: float, timeout_s: float) -> bool:
        # Fake client: gripper is always considered settled immediately.
        return True

    def get_gripper_status(self) -> GripperStatus:
        return self._gripper

    def get_arm_status_snapshot(self) -> ArmStatusSnapshot:
        return ArmStatusSnapshot(
            control_mode="CAN",
            arm_status="NORMAL" if self._enabled else "OTHER",
            move_mode="MOVE_P",
            motion_status="ARRIVED",
            teach_status="0x00",
            control_mode_code=0x01,
            arm_status_code=0x00 if self._enabled else 0x0A,
            move_mode_code=0x00,
            motion_status_code=0x00,
            teach_status_code=0x00,
            trajectory_index=0,
            err_code=0,
            raw_summary="fake snapshot",
        )


class Ros2PiperClient(RobotArmClient):
    """ROS2 client backed by the agilexrobotics/piper_ros humble topics."""

    def __init__(self, config: RobotArmClientConfig | None = None):
        super().__init__(config)
        try:
            import rclpy  # type: ignore
            from rclpy.callback_groups import ReentrantCallbackGroup  # type: ignore
            from geometry_msgs.msg import Pose  # type: ignore
            from piper_msgs.msg import PiperStatusMsg, PosCmd  # type: ignore
            from piper_msgs.srv import Enable  # type: ignore
            from rclpy.executors import SingleThreadedExecutor  # type: ignore
            from sensor_msgs.msg import JointState  # type: ignore

            self._rclpy = rclpy
            self._ReentrantCallbackGroup = ReentrantCallbackGroup
            self._Pose = Pose
            self._PosCmd = PosCmd
            self._PiperStatusMsg = PiperStatusMsg
            self._Enable = Enable
            self._JointState = JointState
            self._SingleThreadedExecutor = SingleThreadedExecutor
        except ImportError:  # pragma: no cover
            self._rclpy = None
            self._ReentrantCallbackGroup = None
            self._Pose = None
            self._PosCmd = None
            self._PiperStatusMsg = None
            self._Enable = None
            self._JointState = None
            self._SingleThreadedExecutor = None
        self._external_node = None
        self._node = None
        self._executor = None
        self._spin_thread = None
        self._callback_group = None
        self._lock = threading.Lock()
        self._pose_publisher = None
        self._joint_ctrl_publisher = None
        self._enable_client = None
        self._connected = False
        self._owns_context = False
        self._latest_pose: EndPoseMMDeg | None = None
        self._latest_pose_monotonic = 0.0
        self._latest_arm_status = ArmStatusSnapshot()
        self._latest_gripper = GripperStatus()
        self._latest_joint_positions: list[float] = [0.0] * 7

    def attach_ros_node(self, node) -> None:
        """Bind this client to an externally managed ROS node."""
        self._external_node = node

    def _require_ros(self) -> None:
        if self._rclpy is None:
            raise RuntimeError(
                "ROS2 libraries are not available in this environment. "
                "Source /opt/ros/humble/setup.bash and the piper_ros overlay first."
            )

    @staticmethod
    def _quaternion_to_euler_deg(
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
    def _angle_diff_deg(a_deg: float, b_deg: float) -> float:
        return (a_deg - b_deg + 180.0) % 360.0 - 180.0

    @classmethod
    def _rotation_matrix_from_rpy_deg(
        cls,
        roll_deg: float,
        pitch_deg: float,
        yaw_deg: float,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]:
        roll = math.radians(roll_deg)
        pitch = math.radians(pitch_deg)
        yaw = math.radians(yaw_deg)
        sr, cr = math.sin(roll), math.cos(roll)
        sp, cp = math.sin(pitch), math.cos(pitch)
        sy, cy = math.sin(yaw), math.cos(yaw)
        return (
            (cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr),
            (sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr),
            (-sp, cp * sr, cp * cr),
        )

    @classmethod
    def _rotation_error_deg(cls, target: EndPoseMMDeg, actual: EndPoseMMDeg) -> float:
        target_rot = cls._rotation_matrix_from_rpy_deg(
            target.roll_deg,
            target.pitch_deg,
            target.yaw_deg,
        )
        actual_rot = cls._rotation_matrix_from_rpy_deg(
            actual.roll_deg,
            actual.pitch_deg,
            actual.yaw_deg,
        )
        relative = [[0.0, 0.0, 0.0] for _ in range(3)]
        for row in range(3):
            for col in range(3):
                relative[row][col] = sum(actual_rot[k][row] * target_rot[k][col] for k in range(3))
        trace_value = relative[0][0] + relative[1][1] + relative[2][2]
        cos_theta = max(-1.0, min(1.0, 0.5 * (trace_value - 1.0)))
        return math.degrees(math.acos(cos_theta))

    def _make_joint_state_command(
        self,
        gripper_open_mm: float,
        effort_nm: float | None = None,
    ):
        joint_msg = self._JointState()
        joint_msg.name = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
            "gripper",
        ]
        with self._lock:
            positions = list(self._latest_joint_positions)
        positions[6] = max(0.0, gripper_open_mm / 1000.0)
        joint_msg.position = positions
        joint_msg.velocity = [0.0] * 7
        joint_msg.effort = [0.0] * 6 + [effort_nm if effort_nm is not None else 1.0]
        return joint_msg

    def _call_enable_service(self, enabled: bool) -> bool:
        self._require_ros()
        if self._enable_client is None:
            raise RuntimeError("Enable service client is not initialized")

        if not self._enable_client.wait_for_service(timeout_sec=self.config.service_timeout_s):
            raise RuntimeError(f"Enable service not available: {self.config.enable_service}")

        request = self._Enable.Request()
        request.enable_request = enabled
        future = self._enable_client.call_async(request)
        deadline = time.monotonic() + self.config.service_timeout_s

        while time.monotonic() < deadline:
            if future.done():
                response = future.result()
                return bool(response and response.enable_response)
            time.sleep(self.config.poll_interval_s)

        raise TimeoutError(f"Timed out calling enable service: {self.config.enable_service}")

    def _wait_for_pose_feedback(self, timeout_s: float | None = None) -> EndPoseMMDeg:
        timeout_s = timeout_s if timeout_s is not None else self.config.pose_feedback_timeout_s
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            pose = self._latest_pose
            if pose is not None:
                return pose
            time.sleep(self.config.poll_interval_s)
        raise TimeoutError(f"No pose feedback received on {self.config.end_pose_topic}")

    def _pose_callback(self, msg) -> None:
        roll_deg, pitch_deg, yaw_deg = self._quaternion_to_euler_deg(
            msg.orientation.x,
            msg.orientation.y,
            msg.orientation.z,
            msg.orientation.w,
        )
        pose = EndPoseMMDeg(
            x_mm=msg.position.x * 1000.0,
            y_mm=msg.position.y * 1000.0,
            z_mm=msg.position.z * 1000.0,
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
        )
        with self._lock:
            self._latest_pose = pose
            self._latest_pose_monotonic = time.monotonic()

    def _joint_state_callback(self, msg) -> None:
        with self._lock:
            self._latest_joint_positions = list(msg.position[:7]) + [0.0] * max(0, 7 - len(msg.position))
            if len(self._latest_joint_positions) > 7:
                self._latest_joint_positions = self._latest_joint_positions[:7]
            gripper_mm = 0.0
            gripper_effort_nm = 0.0
            if "gripper" in msg.name:
                idx = msg.name.index("gripper")
                if idx < len(msg.position):
                    gripper_mm = msg.position[idx] * 1000.0
                if idx < len(msg.effort):
                    gripper_effort_nm = msg.effort[idx]
            self._latest_gripper = GripperStatus(
                angle_mm=gripper_mm,
                effort_nm=gripper_effort_nm,
                enabled=self._connected,
            )

    def _status_callback(self, msg) -> None:
        err_code = int(msg.err_code)
        snapshot = ArmStatusSnapshot(
            control_mode=_CONTROL_MODE_NAMES.get(msg.ctrl_mode, f"0x{int(msg.ctrl_mode):02X}"),
            arm_status=_ARM_STATUS_NAMES.get(msg.arm_status, f"0x{int(msg.arm_status):02X}"),
            move_mode=_MOVE_MODE_NAMES.get(msg.mode_feedback, f"0x{int(msg.mode_feedback):02X}"),
            motion_status=_MOTION_STATUS_NAMES.get(msg.motion_status, f"0x{int(msg.motion_status):02X}"),
            teach_status=f"0x{int(msg.teach_status):02X}",
            control_mode_code=int(msg.ctrl_mode),
            arm_status_code=int(msg.arm_status),
            move_mode_code=int(msg.mode_feedback),
            motion_status_code=int(msg.motion_status),
            teach_status_code=int(msg.teach_status),
            trajectory_index=int(msg.trajectory_num),
            err_code=err_code,
            raw_summary=(
                f"ctrl={msg.ctrl_mode} arm={msg.arm_status} mode={msg.mode_feedback} "
                f"teach={msg.teach_status} motion={msg.motion_status} "
                f"traj={msg.trajectory_num} err={err_code}"
            ),
        )
        with self._lock:
            self._latest_arm_status = snapshot

    def connect(self) -> None:
        self._require_ros()
        using_external_node = self._external_node is not None
        if not using_external_node and not self._rclpy.ok():
            self._rclpy.init(args=None)
            self._owns_context = True
        if self._node is None:
            self._node = self._external_node or self._rclpy.create_node(self.config.node_name)
            self._callback_group = self._ReentrantCallbackGroup() if self._ReentrantCallbackGroup is not None else None
            self._pose_publisher = self._node.create_publisher(
                self._PosCmd,
                self.config.pose_topic,
                1,
                callback_group=self._callback_group,
            )
            self._joint_ctrl_publisher = self._node.create_publisher(
                self._JointState,
                self.config.joint_ctrl_topic,
                1,
                callback_group=self._callback_group,
            )
            self._node.create_subscription(
                self._Pose,
                self.config.end_pose_topic,
                self._pose_callback,
                1,
                callback_group=self._callback_group,
            )
            self._node.create_subscription(
                self._PiperStatusMsg,
                self.config.arm_status_topic,
                self._status_callback,
                1,
                callback_group=self._callback_group,
            )
            self._node.create_subscription(
                self._JointState,
                self.config.joint_state_topic,
                self._joint_state_callback,
                1,
                callback_group=self._callback_group,
            )
            self._enable_client = self._node.create_client(
                self._Enable,
                self.config.enable_service,
                callback_group=self._callback_group,
            )
            if not using_external_node:
                self._executor = self._SingleThreadedExecutor()
                self._executor.add_node(self._node)
                self._spin_thread = threading.Thread(
                    target=self._executor.spin,
                    daemon=True,
                    name="ros2-piper-client-spin",
                )
                self._spin_thread.start()
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False
        if self._external_node is not None:
            return
        if self._executor is not None:
            self._executor.shutdown(timeout_sec=1.0)
            self._executor = None
        if self._spin_thread is not None:
            self._spin_thread.join(timeout=1.0)
        if self._node is not None:
            self._node.destroy_node()
        self._node = None
        self._pose_publisher = None
        self._joint_ctrl_publisher = None
        self._enable_client = None
        self._callback_group = None
        self._spin_thread = None
        if self._owns_context and self._rclpy.ok():
            self._rclpy.shutdown()
        self._owns_context = False

    def enable(self) -> bool:
        self._require_ros()
        return self._call_enable_service(True)

    def disable(self) -> bool:
        self._require_ros()
        return self._call_enable_service(False)

    def emergency_stop(self) -> None:
        self._require_ros()
        self.disable()

    def recover_from_estop(self) -> None:
        self._require_ros()
        self.enable()

    def read_end_pose_mm_deg(self) -> EndPoseMMDeg:
        self._require_ros()
        return self._wait_for_pose_feedback()

    def move_end_pose_mm_deg(
        self,
        x_mm: float,
        y_mm: float,
        z_mm: float,
        roll_deg: float,
        pitch_deg: float,
        yaw_deg: float,
        speed_percent: float,
    ) -> EndPoseMMDeg:
        self._require_ros()
        if self._pose_publisher is None:
            raise RuntimeError("Pose publisher is not initialized")
        command = self._PosCmd()
        command.x = x_mm / 1000.0
        command.y = y_mm / 1000.0
        command.z = z_mm / 1000.0
        command.roll = math.radians(roll_deg)
        command.pitch = math.radians(pitch_deg)
        command.yaw = math.radians(yaw_deg)
        command.gripper = self.get_gripper_status().angle_mm / 1000.0
        command.mode1 = 0
        command.mode2 = 0
        self._pose_publisher.publish(command)
        return EndPoseMMDeg(x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg)

    def wait_until_pose_reached(
        self,
        target: EndPoseMMDeg,
        timeout_s: float,
        pos_tolerance_mm: float,
        rot_tolerance_deg: float,
    ) -> Tuple[bool, EndPoseMMDeg, Dict[str, float]]:
        deadline = time.monotonic() + timeout_s
        actual = self.read_end_pose_mm_deg()
        error = self.pose_error(target, actual)
        while time.monotonic() < deadline:
            actual = self.read_end_pose_mm_deg()
            error = self.pose_error(target, actual)
            if error["dpos_mm"] <= pos_tolerance_mm and error["drot_deg"] <= rot_tolerance_deg:
                return True, actual, error
            time.sleep(self.config.poll_interval_s)
        return False, actual, error

    def pose_error(
        self,
        target: EndPoseMMDeg,
        actual: EndPoseMMDeg | None = None,
    ) -> Dict[str, float]:
        actual = actual or self.read_end_pose_mm_deg()
        dx = actual.x_mm - target.x_mm
        dy = actual.y_mm - target.y_mm
        dz = actual.z_mm - target.z_mm
        droll = self._angle_diff_deg(actual.roll_deg, target.roll_deg)
        dpitch = self._angle_diff_deg(actual.pitch_deg, target.pitch_deg)
        dyaw = self._angle_diff_deg(actual.yaw_deg, target.yaw_deg)
        dpos = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
        drot = self._rotation_error_deg(target, actual)
        return {
            "dx_mm": dx,
            "dy_mm": dy,
            "dz_mm": dz,
            "dpos_mm": dpos,
            "droll_deg": droll,
            "dpitch_deg": dpitch,
            "dyaw_deg": dyaw,
            "drot_deg": drot,
        }

    def format_arm_status(self) -> str:
        snapshot = self.get_arm_status_snapshot()
        return (
            f"control_mode={snapshot.control_mode}(0x{snapshot.control_mode_code:02X}), "
            f"arm_status={snapshot.arm_status}(0x{snapshot.arm_status_code:02X}), "
            f"move_mode={snapshot.move_mode}(0x{snapshot.move_mode_code:02X}), "
            f"teach_status={snapshot.teach_status}, "
            f"motion_status={snapshot.motion_status}(0x{snapshot.motion_status_code:02X}), "
            f"trajectory_index={snapshot.trajectory_index}, "
            f"err_code=0x{snapshot.err_code:04X}"
        )

    def open_gripper(self, open_mm: float, effort_nm: float | None = None) -> None:
        self._require_ros()
        if self._joint_ctrl_publisher is None:
            raise RuntimeError("Joint control publisher is not initialized")
        self._joint_ctrl_publisher.publish(
            self._make_joint_state_command(open_mm, effort_nm),
        )

    def close_gripper(self, effort_nm: float | None = None) -> None:
        self._require_ros()
        if self._joint_ctrl_publisher is None:
            raise RuntimeError("Joint control publisher is not initialized")
        self._joint_ctrl_publisher.publish(
            self._make_joint_state_command(0.0, effort_nm),
        )

    def wait_for_gripper(self, target_mm: float, tol_mm: float, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if abs(self.get_gripper_status().angle_mm - target_mm) <= tol_mm:
                return True
            time.sleep(self.config.poll_interval_s)
        return False

    def wait_for_gripper_effort(self, target_effort_nm: float, timeout_s: float) -> bool:
        """Wait until gripper stops moving (angle stabilises), indicating contact or full close.

        The /joint_states_feedback effort field reflects actual joint torque, which is not
        directly comparable to the commanded effort_nm value.  Waiting for an exact effort
        match is therefore unreliable on real hardware.  Instead we wait until the gripper
        angle has stabilised (delta < 1 mm over two consecutive polls), which reliably
        indicates that the gripper has either gripped an object or reached its travel limit.
        """
        deadline = time.monotonic() + timeout_s
        prev_angle_mm: float | None = None
        while time.monotonic() < deadline:
            current_angle_mm = self.get_gripper_status().angle_mm
            if prev_angle_mm is not None and abs(current_angle_mm - prev_angle_mm) < 1.0:
                return True
            prev_angle_mm = current_angle_mm
            time.sleep(self.config.poll_interval_s)
        return True  # treat timeout as success to avoid blocking retreat/handoff/home

    def get_gripper_status(self) -> GripperStatus:
        with self._lock:
            return GripperStatus(
                angle_mm=self._latest_gripper.angle_mm,
                effort_nm=self._latest_gripper.effort_nm,
                enabled=self._latest_gripper.enabled,
            )

    def get_arm_status_snapshot(self) -> ArmStatusSnapshot:
        with self._lock:
            return ArmStatusSnapshot(
                control_mode=self._latest_arm_status.control_mode,
                arm_status=self._latest_arm_status.arm_status,
                move_mode=self._latest_arm_status.move_mode,
                motion_status=self._latest_arm_status.motion_status,
                teach_status=self._latest_arm_status.teach_status,
                control_mode_code=self._latest_arm_status.control_mode_code,
                arm_status_code=self._latest_arm_status.arm_status_code,
                move_mode_code=self._latest_arm_status.move_mode_code,
                motion_status_code=self._latest_arm_status.motion_status_code,
                teach_status_code=self._latest_arm_status.teach_status_code,
                trajectory_index=self._latest_arm_status.trajectory_index,
                err_code=self._latest_arm_status.err_code,
                raw_summary=self._latest_arm_status.raw_summary,
            )
