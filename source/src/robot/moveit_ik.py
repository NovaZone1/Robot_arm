from __future__ import annotations

from dataclasses import dataclass
import math
import threading
import time

from .types import EndPoseMMDeg


ARM_JOINT_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")
FULL_JOINT_NAMES = (*ARM_JOINT_NAMES, "gripper")
MOVEIT_GRIPPER_JOINT_NAME = "joint7"


@dataclass(slots=True)
class MoveItIkConfig:
    ik_service: str = "/compute_ik"
    joint_state_topic: str = "/joint_states_feedback"
    joint_command_topic: str = "/joint_ctrl_single"
    group_name: str = "arm"
    ik_link_name: str = "link6"
    base_frame: str = "base_link"
    timeout_s: float = 5.0
    response_timeout_padding_s: float = 2.0
    avoid_collisions: bool = True
    default_gripper_effort: float = 1.0


def ensure_moveit_pose_mode_supported(robot_backend: str) -> None:
    if str(robot_backend or "").strip().lower() != "ros2":
        raise RuntimeError("pose_execution_mode=moveit_ik requires robot_backend=ros2")


def clamp_joint_command_speed(speed_percent: float) -> float:
    return max(1.0, min(100.0, float(speed_percent)))


def extract_arm_joint_positions(names: list[str], positions: list[float]) -> list[float]:
    mapping: dict[str, float] = {}
    for index, name in enumerate(names):
        if index < len(positions):
            mapping[str(name)] = float(positions[index])
    missing = [name for name in ARM_JOINT_NAMES if name not in mapping]
    if missing:
        raise RuntimeError(f"IK solution missing joints: {', '.join(missing)}")
    return [mapping[name] for name in ARM_JOINT_NAMES]


def normalize_joint_state_for_moveit(
    names: list[str],
    positions: list[float],
) -> tuple[list[str], list[float]]:
    normalized_names: list[str] = []
    normalized_positions: list[float] = []
    for index, name in enumerate(names):
        if index >= len(positions):
            break
        normalized_names.append(MOVEIT_GRIPPER_JOINT_NAME if str(name) == "gripper" else str(name))
        normalized_positions.append(float(positions[index]))
    return normalized_names, normalized_positions


def extract_moveit_arm_seed_state(
    names: list[str],
    positions: list[float],
) -> tuple[list[str], list[float]]:
    normalized_names, normalized_positions = normalize_joint_state_for_moveit(names, positions)
    mapping = dict(zip(normalized_names, normalized_positions))
    missing = [name for name in ARM_JOINT_NAMES if name not in mapping]
    if missing:
        raise RuntimeError(f"joint feedback missing MoveIt arm joints: {', '.join(missing)}")
    return list(ARM_JOINT_NAMES), [float(mapping[name]) for name in ARM_JOINT_NAMES]


def make_joint_command_payload(
    *,
    arm_joint_positions: list[float],
    gripper_position: float,
    speed_percent: float,
    gripper_effort: float,
) -> dict[str, list[float] | list[str]]:
    if len(arm_joint_positions) != len(ARM_JOINT_NAMES):
        raise ValueError(f"expected {len(ARM_JOINT_NAMES)} arm joints, got {len(arm_joint_positions)}")
    return {
        "name": list(FULL_JOINT_NAMES),
        "position": [float(value) for value in arm_joint_positions] + [float(gripper_position)],
        "velocity": [clamp_joint_command_speed(speed_percent)] * len(ARM_JOINT_NAMES)
        + [clamp_joint_command_speed(speed_percent)],
        "effort": [0.0] * len(ARM_JOINT_NAMES) + [float(gripper_effort)],
    }


def pose_to_quaternion_xyzw(pose: EndPoseMMDeg) -> tuple[float, float, float, float]:
    roll = math.radians(float(pose.roll_deg))
    pitch = math.radians(float(pose.pitch_deg))
    yaw = math.radians(float(pose.yaw_deg))

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    w = cr * cp * cy + sr * sp * sy
    return (x, y, z, w)


class MoveItIkExecutor:
    def __init__(self, node, config: MoveItIkConfig | None = None):
        self._node = node
        self.config = config or MoveItIkConfig()
        try:
            from rclpy.callback_groups import ReentrantCallbackGroup  # type: ignore
            from moveit_msgs.srv import GetPositionIK  # type: ignore
            from sensor_msgs.msg import JointState  # type: ignore
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "MoveIt ROS interfaces are not available. "
                "Install/source moveit_msgs before enabling pose_execution_mode=moveit_ik."
            ) from exc

        self._ReentrantCallbackGroup = ReentrantCallbackGroup
        self._GetPositionIK = GetPositionIK
        self._JointState = JointState
        self._callback_group = self._ReentrantCallbackGroup()
        self._lock = threading.Lock()
        self._latest_joint_names: list[str] = []
        self._latest_joint_positions: list[float] = []
        self._latest_joint_efforts: list[float] = []
        self._locked_gripper_command: tuple[float, float] | None = None

        self._joint_state_sub = self._node.create_subscription(
            self._JointState,
            self.config.joint_state_topic,
            self._joint_state_callback,
            1,
            callback_group=self._callback_group,
        )
        self._joint_cmd_pub = self._node.create_publisher(
            self._JointState,
            self.config.joint_command_topic,
            1,
        )
        self._ik_client = self._node.create_client(
            self._GetPositionIK,
            self.config.ik_service,
            callback_group=self._callback_group,
        )

    def _joint_state_callback(self, msg) -> None:
        with self._lock:
            self._latest_joint_names = list(msg.name)
            self._latest_joint_positions = [float(value) for value in msg.position]
            self._latest_joint_efforts = [float(value) for value in msg.effort]

    def _joint_state_snapshot(self) -> tuple[list[str], list[float], list[float]]:
        deadline = time.monotonic() + max(0.1, float(self.config.timeout_s))
        while time.monotonic() < deadline:
            with self._lock:
                names = list(self._latest_joint_names)
                positions = list(self._latest_joint_positions)
                efforts = list(self._latest_joint_efforts)
            if names and positions:
                return names, positions, efforts
            time.sleep(0.01)
        raise RuntimeError(f"no joint feedback received on {self.config.joint_state_topic}")

    @staticmethod
    def _split_duration(timeout_s: float) -> tuple[int, int]:
        safe_timeout = max(0.0, float(timeout_s))
        seconds = int(safe_timeout)
        nanoseconds = int((safe_timeout - seconds) * 1_000_000_000)
        return seconds, nanoseconds

    def _service_response_timeout_s(self) -> float:
        return max(0.1, float(self.config.timeout_s)) + max(
            0.0,
            float(getattr(self.config, "response_timeout_padding_s", 0.0)),
        )

    def _build_ik_request(self, pose: EndPoseMMDeg):
        names, positions, _efforts = self._joint_state_snapshot()
        request = self._GetPositionIK.Request()
        request.ik_request.group_name = self.config.group_name
        request.ik_request.ik_link_name = self.config.ik_link_name
        request.ik_request.avoid_collisions = bool(self.config.avoid_collisions)
        request.ik_request.pose_stamped.header.frame_id = self.config.base_frame
        request.ik_request.pose_stamped.header.stamp = self._node.get_clock().now().to_msg()
        request.ik_request.pose_stamped.pose.position.x = float(pose.x_mm) / 1000.0
        request.ik_request.pose_stamped.pose.position.y = float(pose.y_mm) / 1000.0
        request.ik_request.pose_stamped.pose.position.z = float(pose.z_mm) / 1000.0
        qx, qy, qz, qw = pose_to_quaternion_xyzw(pose)
        request.ik_request.pose_stamped.pose.orientation.x = qx
        request.ik_request.pose_stamped.pose.orientation.y = qy
        request.ik_request.pose_stamped.pose.orientation.z = qz
        request.ik_request.pose_stamped.pose.orientation.w = qw
        arm_seed_names, arm_seed_positions = extract_moveit_arm_seed_state(names, positions)
        request.ik_request.robot_state.joint_state.name = arm_seed_names
        request.ik_request.robot_state.joint_state.position = arm_seed_positions
        request.ik_request.robot_state.joint_state.velocity = [0.0] * len(arm_seed_positions)
        request.ik_request.robot_state.joint_state.effort = [0.0] * len(arm_seed_positions)
        seconds, nanoseconds = self._split_duration(self.config.timeout_s)
        request.ik_request.timeout.sec = seconds
        request.ik_request.timeout.nanosec = nanoseconds
        return request

    def compute_ik(self, pose: EndPoseMMDeg) -> list[float]:
        if not self._ik_client.wait_for_service(timeout_sec=self.config.timeout_s):
            raise RuntimeError(f"MoveIt IK service not available: {self.config.ik_service}")

        request = self._build_ik_request(pose)
        future = self._ik_client.call_async(request)
        deadline = time.monotonic() + self._service_response_timeout_s()

        while time.monotonic() < deadline:
            if future.done():
                response = future.result()
                if response is None:
                    raise RuntimeError(f"MoveIt IK request returned no response: {self.config.ik_service}")
                error_code = int(response.error_code.val)
                if error_code != 1:
                    detail = str(getattr(response.error_code, "message", "") or "").strip()
                    suffix = f" message={detail}" if detail else ""
                    raise RuntimeError(f"MoveIt IK failed: code={error_code}{suffix}")
                return extract_arm_joint_positions(
                    list(response.solution.joint_state.name),
                    list(response.solution.joint_state.position),
                )
            time.sleep(0.01)

        raise TimeoutError(f"MoveIt IK request timed out: {self.config.ik_service}")

    def lock_gripper_command(self, *, position_m: float, effort_nm: float) -> None:
        self._locked_gripper_command = (float(position_m), float(effort_nm))

    def clear_gripper_command(self) -> None:
        self._locked_gripper_command = None

    def publish_joint_command(self, arm_joint_positions: list[float], speed_percent: float) -> None:
        names, positions, efforts = self._joint_state_snapshot()
        gripper_position = 0.0
        gripper_effort = self.config.default_gripper_effort

        if self._locked_gripper_command is not None:
            gripper_position, gripper_effort = self._locked_gripper_command
        elif "gripper" in names:
            gripper_index = names.index("gripper")
            if gripper_index < len(positions):
                gripper_position = float(positions[gripper_index])
            if gripper_index < len(efforts):
                gripper_effort = float(efforts[gripper_index]) or self.config.default_gripper_effort

        payload = make_joint_command_payload(
            arm_joint_positions=arm_joint_positions,
            gripper_position=gripper_position,
            speed_percent=speed_percent,
            gripper_effort=gripper_effort,
        )
        msg = self._JointState()
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.name = list(payload["name"])
        msg.position = list(payload["position"])
        msg.velocity = list(payload["velocity"])
        msg.effort = list(payload["effort"])
        self._joint_cmd_pub.publish(msg)

    def execute_pose(self, pose: EndPoseMMDeg, speed_percent: float) -> list[float]:
        arm_joint_positions = self.compute_ik(pose)
        self.publish_joint_command(arm_joint_positions, speed_percent)
        return arm_joint_positions
