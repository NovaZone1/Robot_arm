from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
from geometry_msgs.msg import Point, Quaternion, TransformStamped
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image

from robot_grasp_msgs.msg import GraspCandidate as GraspCandidateMsg
from robot_grasp_msgs.msg import GraspPlan as GraspPlanMsg
from robot_grasp_msgs.msg import PerceptionSummary as PerceptionSummaryMsg
from robot_grasp_msgs.msg import PipelineStatus as PipelineStatusMsg
from robot_grasp_msgs.msg import Pose6D as Pose6DMsg
from src.grasping.models import GraspCandidate, GraspPlan
from src.robot.types import EndPoseMMDeg
from src.run_grasp_pipeline_ros2 import DEFAULT_HAND_EYE_CONFIG, build_config, build_parser
from src.utils.calibration import load_camera_to_tcp_transform
from src.utils.transforms import make_transform_xyz_rpy_mm_deg, rpy_deg_from_rotation_matrix

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


@dataclass(slots=True)
class SimpleIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float


def new_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{int(time.time() * 1000)}"


def json_dumps(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)


def make_latched_qos(depth: int = 1) -> QoSProfile:
    qos = QoSProfile(depth=max(1, int(depth)))
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    qos.reliability = ReliabilityPolicy.RELIABLE
    return qos


def parse_options_json(text: str) -> dict[str, Any]:
    if not text or not text.strip():
        return {}
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("options_json must decode to an object")
    return payload


def build_args_from_options(options: dict[str, Any]) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args([])
    extra_cli_args = []
    for key, value in options.items():
        if key == "extra_cli_args":
            if isinstance(value, list):
                extra_cli_args = [str(item) for item in value]
            continue
        if hasattr(args, key):
            setattr(args, key, value)
    if extra_cli_args:
        args = parser.parse_args(extra_cli_args, namespace=args)
    return args


def build_runtime_config(options: dict[str, Any]):
    args = build_args_from_options(options)
    return build_config(args)


def load_hand_eye_matrix(config_path: str | Path) -> np.ndarray:
    resolved = Path(config_path).expanduser().resolve()
    if yaml is None:
        raise RuntimeError("yaml is unavailable; cannot load hand-eye config")
    with open(resolved, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    calib = payload.get("calibration", {})
    transform, _ = load_camera_to_tcp_transform(calib, allow_legacy=False)
    return np.asarray(transform, dtype=np.float64).reshape(4, 4)


def default_runtime_options() -> dict[str, Any]:
    args = build_parser().parse_args([])
    payload = vars(args).copy()
    payload["hand_eye_config"] = str(DEFAULT_HAND_EYE_CONFIG)
    return payload


def matrix_to_quaternion_xyzw(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace_value = float(np.trace(matrix))
    if trace_value > 0.0:
        s = np.sqrt(trace_value + 1.0) * 2.0
        qw = 0.25 * s
        qx = (matrix[2, 1] - matrix[1, 2]) / s
        qy = (matrix[0, 2] - matrix[2, 0]) / s
        qz = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        qw = (matrix[2, 1] - matrix[1, 2]) / s
        qx = 0.25 * s
        qy = (matrix[0, 1] + matrix[1, 0]) / s
        qz = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        qw = (matrix[0, 2] - matrix[2, 0]) / s
        qx = (matrix[0, 1] + matrix[1, 0]) / s
        qy = 0.25 * s
        qz = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        qw = (matrix[1, 0] - matrix[0, 1]) / s
        qx = (matrix[0, 2] + matrix[2, 0]) / s
        qy = (matrix[1, 2] + matrix[2, 1]) / s
        qz = 0.25 * s
    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm > 1e-8:
        quat /= norm
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def quaternion_xyzw_to_matrix(quaternion: Quaternion) -> np.ndarray:
    qx = float(quaternion.x)
    qy = float(quaternion.y)
    qz = float(quaternion.z)
    qw = float(quaternion.w)
    norm = np.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    if norm <= 1e-8:
        return np.eye(3, dtype=np.float64)
    qx /= norm
    qy /= norm
    qz /= norm
    qw /= norm
    return np.array(
        [
            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qz * qw), 2.0 * (qx * qz + qy * qw)],
            [2.0 * (qx * qy + qz * qw), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qx * qw)],
            [2.0 * (qx * qz - qy * qw), 2.0 * (qy * qz + qx * qw), 1.0 - 2.0 * (qx * qx + qy * qy)],
        ],
        dtype=np.float64,
    )


def point_from_xyz(xyz: tuple[float, float, float] | np.ndarray) -> Point:
    values = np.asarray(xyz, dtype=np.float64).reshape(3)
    msg = Point()
    msg.x = float(values[0])
    msg.y = float(values[1])
    msg.z = float(values[2])
    return msg


def pose6d_from_end_pose(pose: EndPoseMMDeg) -> Pose6DMsg:
    msg = Pose6DMsg()
    msg.x_mm = float(pose.x_mm)
    msg.y_mm = float(pose.y_mm)
    msg.z_mm = float(pose.z_mm)
    msg.roll_deg = float(pose.roll_deg)
    msg.pitch_deg = float(pose.pitch_deg)
    msg.yaw_deg = float(pose.yaw_deg)
    return msg


def pose6d_to_end_pose(msg: Pose6DMsg) -> EndPoseMMDeg:
    return EndPoseMMDeg(
        x_mm=float(msg.x_mm),
        y_mm=float(msg.y_mm),
        z_mm=float(msg.z_mm),
        roll_deg=float(msg.roll_deg),
        pitch_deg=float(msg.pitch_deg),
        yaw_deg=float(msg.yaw_deg),
    )


def pose6d_from_position_m_rpy_deg(position_m: tuple[float, float, float], rpy_deg: tuple[float, float, float]) -> Pose6DMsg:
    msg = Pose6DMsg()
    msg.x_mm = float(position_m[0]) * 1000.0
    msg.y_mm = float(position_m[1]) * 1000.0
    msg.z_mm = float(position_m[2]) * 1000.0
    msg.roll_deg = float(rpy_deg[0])
    msg.pitch_deg = float(rpy_deg[1])
    msg.yaw_deg = float(rpy_deg[2])
    return msg


def grasp_candidate_to_msg(candidate: GraspCandidate) -> GraspCandidateMsg:
    msg = GraspCandidateMsg()
    msg.instance_index = int(candidate.instance_index)
    msg.score = float(candidate.score)
    msg.width_m = float(candidate.width_m)
    msg.depth_m = float(candidate.depth_m)
    msg.translation_camera_m = point_from_xyz(candidate.translation_camera_m)
    qx, qy, qz, qw = matrix_to_quaternion_xyzw(candidate.rotation_camera)
    msg.orientation_camera.x = qx
    msg.orientation_camera.y = qy
    msg.orientation_camera.z = qz
    msg.orientation_camera.w = qw
    msg.has_object_center = candidate.object_center_camera_m is not None
    if candidate.object_center_camera_m is not None:
        msg.object_center_camera_m = point_from_xyz(candidate.object_center_camera_m)
    msg.center_offset_m = float(candidate.center_offset_m or 0.0)
    return msg


def grasp_candidate_from_msg(msg: GraspCandidateMsg) -> GraspCandidate:
    object_center = None
    if bool(msg.has_object_center):
        object_center = (
            float(msg.object_center_camera_m.x),
            float(msg.object_center_camera_m.y),
            float(msg.object_center_camera_m.z),
        )
    return GraspCandidate(
        instance_index=int(msg.instance_index),
        score=float(msg.score),
        width_m=float(msg.width_m),
        depth_m=float(msg.depth_m),
        translation_camera_m=(
            float(msg.translation_camera_m.x),
            float(msg.translation_camera_m.y),
            float(msg.translation_camera_m.z),
        ),
        rotation_camera=quaternion_xyzw_to_matrix(msg.orientation_camera),
        object_center_camera_m=object_center,
        center_offset_m=(float(msg.center_offset_m) if bool(msg.has_object_center) else None),
        raw_grasp=None,
    )


def grasp_plan_to_msg(plan: GraspPlan) -> GraspPlanMsg:
    msg = GraspPlanMsg()
    msg.candidate = grasp_candidate_to_msg(plan.candidate)
    msg.target_pose = pose6d_from_position_m_rpy_deg(plan.target_base_m, plan.target_rpy_deg)
    msg.pregrasp_pose = pose6d_from_position_m_rpy_deg(plan.pregrasp_base_m, plan.target_rpy_deg)
    msg.grasp_pose = pose6d_from_position_m_rpy_deg(plan.grasp_base_m, plan.target_rpy_deg)
    msg.retreat_pose = pose6d_from_position_m_rpy_deg(plan.retreat_base_m, plan.target_rpy_deg)
    msg.within_workspace = bool(plan.within_workspace)
    msg.workspace_violations = list(plan.workspace_violations)
    has_geometry = (
        plan.target_contact_point_base_m is not None
        and plan.tool_contact_offset_tool_m is not None
    )
    msg.has_tool_contact_geometry = has_geometry
    if has_geometry:
        msg.target_contact_point_base_m.x = float(plan.target_contact_point_base_m[0])
        msg.target_contact_point_base_m.y = float(plan.target_contact_point_base_m[1])
        msg.target_contact_point_base_m.z = float(plan.target_contact_point_base_m[2])
        msg.tool_contact_offset_tool_m.x = float(plan.tool_contact_offset_tool_m[0])
        msg.tool_contact_offset_tool_m.y = float(plan.tool_contact_offset_tool_m[1])
        msg.tool_contact_offset_tool_m.z = float(plan.tool_contact_offset_tool_m[2])
    return msg


def grasp_plan_from_msg(msg: GraspPlanMsg) -> GraspPlan:
    candidate = grasp_candidate_from_msg(msg.candidate)

    def _pose_msg_to_position(pose_msg: Pose6DMsg) -> tuple[float, float, float]:
        return (
            float(pose_msg.x_mm) / 1000.0,
            float(pose_msg.y_mm) / 1000.0,
            float(pose_msg.z_mm) / 1000.0,
        )

    target_rpy_deg = (
        float(msg.target_pose.roll_deg),
        float(msg.target_pose.pitch_deg),
        float(msg.target_pose.yaw_deg),
    )
    target_contact_point_base_m = None
    tool_contact_offset_tool_m = None
    if bool(msg.has_tool_contact_geometry):
        target_contact_point_base_m = (
            float(msg.target_contact_point_base_m.x),
            float(msg.target_contact_point_base_m.y),
            float(msg.target_contact_point_base_m.z),
        )
        tool_contact_offset_tool_m = (
            float(msg.tool_contact_offset_tool_m.x),
            float(msg.tool_contact_offset_tool_m.y),
            float(msg.tool_contact_offset_tool_m.z),
        )
    return GraspPlan(
        candidate=candidate,
        target_base_m=_pose_msg_to_position(msg.target_pose),
        target_rpy_deg=target_rpy_deg,
        pregrasp_base_m=_pose_msg_to_position(msg.pregrasp_pose),
        grasp_base_m=_pose_msg_to_position(msg.grasp_pose),
        retreat_base_m=_pose_msg_to_position(msg.retreat_pose),
        within_workspace=bool(msg.within_workspace),
        workspace_violations=list(msg.workspace_violations),
        target_contact_point_base_m=target_contact_point_base_m,
        tool_contact_offset_tool_m=tool_contact_offset_tool_m,
    )


def matrix_to_transform_msg(matrix: np.ndarray, *, parent_frame: str, child_frame: str, stamp=None) -> TransformStamped:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    qx, qy, qz, qw = matrix_to_quaternion_xyzw(matrix[:3, :3])
    msg = TransformStamped()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = parent_frame
    msg.child_frame_id = child_frame
    msg.transform.translation.x = float(matrix[0, 3])
    msg.transform.translation.y = float(matrix[1, 3])
    msg.transform.translation.z = float(matrix[2, 3])
    msg.transform.rotation.x = qx
    msg.transform.rotation.y = qy
    msg.transform.rotation.z = qz
    msg.transform.rotation.w = qw
    return msg


def transform_msg_to_matrix(msg: TransformStamped) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quaternion_xyzw_to_matrix(msg.transform.rotation)
    matrix[0, 3] = float(msg.transform.translation.x)
    matrix[1, 3] = float(msg.transform.translation.y)
    matrix[2, 3] = float(msg.transform.translation.z)
    return matrix


def base_to_camera_from_tcp_and_hand_eye(tcp_pose: EndPoseMMDeg, hand_eye: np.ndarray) -> np.ndarray:
    base_to_tcp = make_transform_xyz_rpy_mm_deg(
        xyz_mm=(tcp_pose.x_mm, tcp_pose.y_mm, tcp_pose.z_mm),
        rpy_deg=(tcp_pose.roll_deg, tcp_pose.pitch_deg, tcp_pose.yaw_deg),
    )
    return np.asarray(base_to_tcp, dtype=np.float64).reshape(4, 4) @ np.asarray(hand_eye, dtype=np.float64).reshape(4, 4)


def color_image_to_msg(image_bgr: np.ndarray, *, frame_id: str, stamp=None) -> Image:
    array = np.asarray(image_bgr, dtype=np.uint8)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("color image must be HxWx3 uint8")
    msg = Image()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(array.shape[0])
    msg.width = int(array.shape[1])
    msg.encoding = "bgr8"
    msg.step = int(array.shape[1] * 3)
    msg.is_bigendian = False
    msg.data = array.tobytes()
    return msg


def depth_image_to_msg(depth_meters: np.ndarray, *, frame_id: str, stamp=None) -> Image:
    array = np.asarray(depth_meters, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("depth image must be HxW float32")
    msg = Image()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.height = int(array.shape[0])
    msg.width = int(array.shape[1])
    msg.encoding = "32FC1"
    msg.step = int(array.shape[1] * 4)
    msg.is_bigendian = False
    msg.data = array.tobytes()
    return msg


def color_msg_to_bgr(msg: Image) -> np.ndarray:
    if msg.encoding != "bgr8":
        raise ValueError(f"unsupported color encoding: {msg.encoding}")
    array = np.frombuffer(msg.data, dtype=np.uint8)
    return array.reshape((int(msg.height), int(msg.width), 3)).copy()


def depth_msg_to_meters(msg: Image) -> np.ndarray:
    if msg.encoding != "32FC1":
        raise ValueError(f"unsupported depth encoding: {msg.encoding}")
    array = np.frombuffer(msg.data, dtype=np.float32)
    return array.reshape((int(msg.height), int(msg.width))).copy()


def intrinsics_to_camera_info(intrinsics: Any, *, frame_id: str, stamp=None) -> CameraInfo:
    msg = CameraInfo()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.width = int(intrinsics.width)
    msg.height = int(intrinsics.height)
    msg.k = [
        float(intrinsics.fx),
        0.0,
        float(intrinsics.ppx),
        0.0,
        float(intrinsics.fy),
        float(intrinsics.ppy),
        0.0,
        0.0,
        1.0,
    ]
    msg.p = [
        float(intrinsics.fx),
        0.0,
        float(intrinsics.ppx),
        0.0,
        0.0,
        float(intrinsics.fy),
        float(intrinsics.ppy),
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
    ]
    return msg


def camera_info_to_intrinsics(msg: CameraInfo) -> SimpleIntrinsics:
    return SimpleIntrinsics(
        width=int(msg.width),
        height=int(msg.height),
        fx=float(msg.k[0]),
        fy=float(msg.k[4]),
        ppx=float(msg.k[2]),
        ppy=float(msg.k[5]),
    )


def make_perception_summary_msg(
    *,
    scene_id: str,
    prompt: str,
    camera_frame: str,
    instance_count: int,
    scene_grasp_count: int,
    scene_point_count: int,
    object_point_counts: list[int],
    debug_lines: list[str],
) -> PerceptionSummaryMsg:
    msg = PerceptionSummaryMsg()
    msg.scene_id = scene_id
    msg.prompt = prompt
    msg.camera_frame = camera_frame
    msg.instance_count = int(instance_count)
    msg.scene_grasp_count = int(scene_grasp_count)
    msg.scene_point_count = int(scene_point_count)
    msg.object_point_counts = [int(value) for value in object_point_counts]
    msg.debug_lines = list(debug_lines)
    return msg


def make_pipeline_status_msg(*, stamp, run_id: str, node_name: str, stage: str, state: str, detail: str) -> PipelineStatusMsg:
    msg = PipelineStatusMsg()
    msg.stamp = stamp
    msg.run_id = run_id
    msg.node_name = node_name
    msg.stage = stage
    msg.state = state
    msg.detail = detail
    return msg


def pose6d_to_tuple_mm_deg(msg: Pose6DMsg) -> tuple[float, float, float, float, float, float]:
    return (
        float(msg.x_mm),
        float(msg.y_mm),
        float(msg.z_mm),
        float(msg.roll_deg),
        float(msg.pitch_deg),
        float(msg.yaw_deg),
    )


def pose6d_to_position_rpy(msg: Pose6DMsg) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    return (
        (float(msg.x_mm) / 1000.0, float(msg.y_mm) / 1000.0, float(msg.z_mm) / 1000.0),
        (float(msg.roll_deg), float(msg.pitch_deg), float(msg.yaw_deg)),
    )


def candidate_debug_dict(candidate: GraspCandidate) -> dict[str, Any]:
    return {
        "instance_index": candidate.instance_index,
        "score": candidate.score,
        "width_m": candidate.width_m,
        "depth_m": candidate.depth_m,
        "translation_camera_m": list(candidate.translation_camera_m),
        "object_center_camera_m": (
            list(candidate.object_center_camera_m)
            if candidate.object_center_camera_m is not None
            else None
        ),
        "center_offset_m": candidate.center_offset_m,
    }


def plan_debug_dict(plan: GraspPlan) -> dict[str, Any]:
    return {
        "target_base_m": list(plan.target_base_m),
        "target_rpy_deg": list(plan.target_rpy_deg),
        "pregrasp_base_m": list(plan.pregrasp_base_m),
        "grasp_base_m": list(plan.grasp_base_m),
        "retreat_base_m": list(plan.retreat_base_m),
        "within_workspace": plan.within_workspace,
        "workspace_violations": list(plan.workspace_violations),
        "target_contact_point_base_m": (
            list(plan.target_contact_point_base_m)
            if plan.target_contact_point_base_m is not None
            else None
        ),
        "tool_contact_offset_tool_m": (
            list(plan.tool_contact_offset_tool_m)
            if plan.tool_contact_offset_tool_m is not None
            else None
        ),
    }
