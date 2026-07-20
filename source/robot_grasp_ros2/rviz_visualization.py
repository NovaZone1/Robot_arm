from __future__ import annotations

from copy import deepcopy
import math

import numpy as np
from geometry_msgs.msg import Point, Pose, PoseArray, TransformStamped
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import ColorRGBA, Header
from visualization_msgs.msg import Marker, MarkerArray

from src.grasping.models import GraspCandidate, GraspPlan, PerceptionResult
from src.grasping.planning import PureGraspPlanner
from src.utils.transforms import rotation_matrix_from_rpy_deg

try:
    from tf2_ros import TransformBroadcaster
except Exception:  # pragma: no cover
    TransformBroadcaster = None


_INSTANCE_COLORS = [
    (0x39, 0x8C, 0xFF),
    (0xFF, 0x7A, 0x2F),
    (0x33, 0xB5, 0x63),
    (0xE6, 0x4A, 0x5E),
    (0x8A, 0x57, 0xFF),
    (0x1C, 0xC7, 0xC9),
]

_AXIS_COLORS = (
    (0.95, 0.20, 0.20, 1.0),
    (0.20, 0.85, 0.25, 1.0),
    (0.20, 0.45, 0.95, 1.0),
)

_VALIDATION_STATUS_COLORS = {
    "rejected_by_robot_validation": (0.92, 0.30, 0.24, 0.95),
    "rejected_during_plan_build": (0.70, 0.32, 0.18, 0.95),
    "selected_for_execution": (0.16, 0.78, 0.38, 0.95),
    "accepted_not_selected": (0.95, 0.76, 0.18, 0.95),
}


def _make_latched_qos() -> QoSProfile:
    qos = QoSProfile(depth=1)
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
    qos.reliability = ReliabilityPolicy.RELIABLE
    return qos


def _matrix_to_quaternion_xyzw(rotation: np.ndarray) -> tuple[float, float, float, float]:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    trace_value = float(np.trace(matrix))
    if trace_value > 0.0:
        s = math.sqrt(trace_value + 1.0) * 2.0
        qw = 0.25 * s
        qx = (matrix[2, 1] - matrix[1, 2]) / s
        qy = (matrix[0, 2] - matrix[2, 0]) / s
        qz = (matrix[1, 0] - matrix[0, 1]) / s
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        qw = (matrix[2, 1] - matrix[1, 2]) / s
        qx = 0.25 * s
        qy = (matrix[0, 1] + matrix[1, 0]) / s
        qz = (matrix[0, 2] + matrix[2, 0]) / s
    elif matrix[1, 1] > matrix[2, 2]:
        s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        qw = (matrix[0, 2] - matrix[2, 0]) / s
        qx = (matrix[0, 1] + matrix[1, 0]) / s
        qy = 0.25 * s
        qz = (matrix[1, 2] + matrix[2, 1]) / s
    else:
        s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        qw = (matrix[1, 0] - matrix[0, 1]) / s
        qx = (matrix[0, 2] + matrix[2, 0]) / s
        qy = (matrix[1, 2] + matrix[2, 1]) / s
        qz = 0.25 * s

    quat = np.array([qx, qy, qz, qw], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm > 1e-8:
        quat /= norm
    return (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))


def _make_pose(translation_m: tuple[float, float, float] | np.ndarray, rotation: np.ndarray) -> Pose:
    xyz = np.asarray(translation_m, dtype=np.float64).reshape(3)
    qx, qy, qz, qw = _matrix_to_quaternion_xyzw(rotation)
    pose = Pose()
    pose.position.x = float(xyz[0])
    pose.position.y = float(xyz[1])
    pose.position.z = float(xyz[2])
    pose.orientation.x = qx
    pose.orientation.y = qy
    pose.orientation.z = qz
    pose.orientation.w = qw
    return pose


def _make_point(x: float, y: float, z: float) -> Point:
    point = Point()
    point.x = float(x)
    point.y = float(y)
    point.z = float(z)
    return point


def _make_color(r: float, g: float, b: float, a: float = 1.0) -> ColorRGBA:
    color = ColorRGBA()
    color.r = float(r)
    color.g = float(g)
    color.b = float(b)
    color.a = float(a)
    return color


def _make_empty_pointcloud(header: Header) -> PointCloud2:
    return _make_xyz_pointcloud(header, np.empty((0, 3), dtype=np.float32))


def _make_safe_empty_xyz_points() -> np.ndarray:
    # RViz on this workstation can abort on truly empty PointCloud2.data.
    # A single finite origin point keeps the byte payload non-empty while
    # avoiding crashes observed with NaN-filled fallback points.
    return np.array([[0.0, 0.0, 0.0]], dtype=np.float32)


def _make_xyz_pointcloud(header: Header, points: np.ndarray | None) -> PointCloud2:
    pts = np.asarray(points if points is not None else np.empty((0, 3), dtype=np.float32), dtype=np.float32).reshape(-1, 3)
    if pts.size > 0:
        finite_mask = np.isfinite(pts).all(axis=1)
        pts = pts[finite_mask]
    if len(pts) == 0:
        pts = _make_safe_empty_xyz_points()

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = int(len(pts))
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = 12
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = False
    msg.data = pts.astype(np.float32, copy=False).tobytes()
    return msg


def _pack_rgb_u32(r: int, g: int, b: int) -> np.uint32:
    return np.uint32((int(r) << 16) | (int(g) << 8) | int(b))


def _make_xyzrgb_pointcloud(header: Header, points: np.ndarray, rgb_values: np.ndarray) -> PointCloud2:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    rgb = np.asarray(rgb_values, dtype=np.uint32).reshape(-1)
    if len(pts) != len(rgb):
        raise ValueError("points and rgb_values must have the same length")
    if len(pts) > 0:
        finite_mask = np.isfinite(pts).all(axis=1)
        pts = pts[finite_mask]
        rgb = rgb[finite_mask]
    if len(pts) == 0:
        pts = _make_safe_empty_xyz_points()
        rgb = np.zeros((1,), dtype=np.uint32)

    payload = np.empty(
        len(pts),
        dtype=[
            ("x", np.float32),
            ("y", np.float32),
            ("z", np.float32),
            ("rgb", np.uint32),
        ],
    )
    if len(pts) > 0:
        payload["x"] = pts[:, 0]
        payload["y"] = pts[:, 1]
        payload["z"] = pts[:, 2]
        payload["rgb"] = rgb

    msg = PointCloud2()
    msg.header = header
    msg.height = 1
    msg.width = int(len(payload))
    msg.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.UINT32, count=1),
    ]
    msg.is_bigendian = False
    msg.point_step = int(payload.dtype.itemsize)
    msg.row_step = msg.point_step * msg.width
    msg.is_dense = False
    msg.data = payload.tobytes()
    return msg


def _merge_instance_clouds(pointclouds: list) -> tuple[np.ndarray, np.ndarray]:
    merged_points: list[np.ndarray] = []
    merged_rgb: list[np.ndarray] = []
    for index, pointcloud in enumerate(pointclouds):
        if pointcloud is None:
            continue
        points = np.asarray(pointcloud.points, dtype=np.float32).reshape(-1, 3)
        if len(points) == 0:
            continue
        color = _INSTANCE_COLORS[index % len(_INSTANCE_COLORS)]
        rgb = np.full(len(points), _pack_rgb_u32(*color), dtype=np.uint32)
        merged_points.append(points)
        merged_rgb.append(rgb)

    if not merged_points:
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.uint32)
    return np.concatenate(merged_points, axis=0), np.concatenate(merged_rgb, axis=0)


def _candidate_to_pose(candidate: GraspCandidate) -> Pose:
    raw_rotation = np.asarray(candidate.rotation_camera, dtype=np.float64).reshape(3, 3)
    # Keep the RViz pose aligned with the same grasp/tool convention used by the planner.
    adjusted_rotation = PureGraspPlanner._normalize_columns(raw_rotation @ PureGraspPlanner._R_ADJUST)
    return _make_pose(candidate.translation_camera_m, adjusted_rotation)


def _build_pose_array(header: Header, poses: list[Pose]) -> PoseArray:
    msg = PoseArray()
    msg.header = header
    msg.poses = poses
    return msg


def _build_transform_stamped(
    *,
    stamp,
    parent_frame: str,
    child_frame: str,
    transform: np.ndarray,
) -> TransformStamped:
    matrix = np.asarray(transform, dtype=np.float64).reshape(4, 4)
    qx, qy, qz, qw = _matrix_to_quaternion_xyzw(matrix[:3, :3])
    tf_msg = TransformStamped()
    tf_msg.header.stamp = stamp
    tf_msg.header.frame_id = parent_frame
    tf_msg.child_frame_id = child_frame
    tf_msg.transform.translation.x = float(matrix[0, 3])
    tf_msg.transform.translation.y = float(matrix[1, 3])
    tf_msg.transform.translation.z = float(matrix[2, 3])
    tf_msg.transform.rotation.x = qx
    tf_msg.transform.rotation.y = qy
    tf_msg.transform.rotation.z = qz
    tf_msg.transform.rotation.w = qw
    return tf_msg


def _delete_all_marker_array(header: Header) -> MarkerArray:
    marker = Marker()
    marker.header = header
    marker.action = Marker.DELETEALL
    return MarkerArray(markers=[marker])


def _build_axes_marker_array(
    *,
    header: Header,
    namespace: str,
    frames: list[tuple[str, tuple[float, float, float] | np.ndarray, np.ndarray]],
    axis_length: float,
    axis_thickness: float,
    text_height: float,
) -> MarkerArray:
    if not frames:
        return _delete_all_marker_array(header)

    line_marker = Marker()
    line_marker.header = header
    line_marker.ns = namespace
    line_marker.id = 0
    line_marker.type = Marker.LINE_LIST
    line_marker.action = Marker.ADD
    line_marker.pose.orientation.w = 1.0
    line_marker.scale.x = float(axis_thickness)

    markers = [line_marker]
    marker_id = 1

    for label, translation, rotation in frames:
        origin = np.asarray(translation, dtype=np.float64).reshape(3)
        axes = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
        for axis_index, rgba in enumerate(_AXIS_COLORS):
            axis_vector = axes[:, axis_index]
            endpoint = origin + axis_vector * float(axis_length)
            line_marker.points.append(_make_point(origin[0], origin[1], origin[2]))
            line_marker.points.append(_make_point(endpoint[0], endpoint[1], endpoint[2]))
            color = _make_color(*rgba)
            line_marker.colors.append(color)
            line_marker.colors.append(color)

        if label:
            text_marker = Marker()
            text_marker.header = header
            text_marker.ns = namespace
            text_marker.id = marker_id
            marker_id += 1
            text_marker.type = Marker.TEXT_VIEW_FACING
            text_marker.action = Marker.ADD
            text_marker.pose.position = _make_point(origin[0], origin[1], origin[2] + float(axis_length) * 1.15)
            text_marker.pose.orientation.w = 1.0
            text_marker.scale.z = float(text_height)
            text_marker.color = _make_color(1.0, 1.0, 1.0, 0.95)
            text_marker.text = label
            markers.append(text_marker)

    return MarkerArray(markers=markers)


def _build_path_marker_array(
    *,
    header: Header,
    namespace: str,
    waypoints: list[tuple[str, tuple[float, float, float]]],
    line_thickness: float,
) -> MarkerArray:
    if not waypoints:
        return _delete_all_marker_array(header)

    markers: list[Marker] = []
    path_marker = Marker()
    path_marker.header = header
    path_marker.ns = namespace
    path_marker.id = 0
    path_marker.type = Marker.LINE_STRIP
    path_marker.action = Marker.ADD
    path_marker.pose.orientation.w = 1.0
    path_marker.scale.x = float(line_thickness)
    path_marker.color = _make_color(1.0, 0.85, 0.15, 0.9)
    path_marker.points = [
        _make_point(position[0], position[1], position[2])
        for _name, position in waypoints
    ]
    markers.append(path_marker)

    points_marker = Marker()
    points_marker.header = header
    points_marker.ns = namespace
    points_marker.id = 1
    points_marker.type = Marker.SPHERE_LIST
    points_marker.action = Marker.ADD
    points_marker.pose.orientation.w = 1.0
    points_marker.scale.x = 0.015
    points_marker.scale.y = 0.015
    points_marker.scale.z = 0.015
    points_marker.color = _make_color(1.0, 0.95, 0.35, 0.95)
    points_marker.points = [
        _make_point(position[0], position[1], position[2])
        for _name, position in waypoints
    ]
    markers.append(points_marker)

    marker_id = 2
    for name, position in waypoints:
        text_marker = Marker()
        text_marker.header = header
        text_marker.ns = namespace
        text_marker.id = marker_id
        marker_id += 1
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position = _make_point(position[0], position[1], position[2] + 0.03)
        text_marker.pose.orientation.w = 1.0
        text_marker.scale.z = 0.03
        text_marker.color = _make_color(1.0, 1.0, 1.0, 0.95)
        text_marker.text = name
        markers.append(text_marker)

    return MarkerArray(markers=markers)


def _validation_label(record: dict[str, object]) -> str:
    candidate_index = int(record.get("candidate_index", -1))
    selection_result = str(record.get("selection_result") or "")
    stage = str(record.get("robot_validation_stage") or "").strip()
    error_type = str(record.get("ik_error_type") or "").strip()

    if selection_result == "selected_for_execution":
        return (
            f"cand{candidate_index} selected fallback"
            if candidate_index > 0
            else f"cand{candidate_index} selected"
        )
    if selection_result == "rejected_by_robot_validation":
        if stage and error_type:
            return f"cand{candidate_index} rejected@{stage} {error_type}"
        if stage:
            return f"cand{candidate_index} rejected@{stage}"
        return f"cand{candidate_index} rejected"
    if selection_result == "rejected_during_plan_build":
        return f"cand{candidate_index} plan-build-error"
    return f"cand{candidate_index} {selection_result or 'validation'}"


def build_candidate_validation_marker_array(
    *,
    validation_records: list[dict[str, object]],
    camera_frame: str,
    stamp,
) -> MarkerArray:
    header = Header(frame_id=camera_frame)
    if stamp is not None:
        header.stamp = stamp
    if not validation_records:
        return _delete_all_marker_array(header)

    markers: list[Marker] = []
    line_marker = Marker()
    line_marker.header = header
    line_marker.ns = "candidate_validation_path"
    line_marker.id = 0
    line_marker.type = Marker.LINE_STRIP
    line_marker.action = Marker.ADD
    line_marker.pose.orientation.w = 1.0
    line_marker.scale.x = 0.004
    line_marker.color = _make_color(1.0, 0.95, 0.35, 0.85)

    marker_id = 1
    path_points: list[tuple[float, float, float]] = []

    for record in validation_records:
        translation = record.get("translation_camera_m")
        if not isinstance(translation, (list, tuple)) or len(translation) != 3:
            continue
        point = np.asarray(translation, dtype=np.float64).reshape(3)
        path_points.append((float(point[0]), float(point[1]), float(point[2])))

        selection_result = str(record.get("selection_result") or "")
        rgba = _VALIDATION_STATUS_COLORS.get(selection_result, (0.80, 0.80, 0.80, 0.95))

        sphere_marker = Marker()
        sphere_marker.header = header
        sphere_marker.ns = "candidate_validation"
        sphere_marker.id = marker_id
        marker_id += 1
        sphere_marker.type = Marker.SPHERE
        sphere_marker.action = Marker.ADD
        sphere_marker.pose.position = _make_point(point[0], point[1], point[2])
        sphere_marker.pose.orientation.w = 1.0
        sphere_marker.scale.x = 0.020
        sphere_marker.scale.y = 0.020
        sphere_marker.scale.z = 0.020
        sphere_marker.color = _make_color(*rgba)
        markers.append(sphere_marker)

        text_marker = Marker()
        text_marker.header = header
        text_marker.ns = "candidate_validation"
        text_marker.id = marker_id
        marker_id += 1
        text_marker.type = Marker.TEXT_VIEW_FACING
        text_marker.action = Marker.ADD
        text_marker.pose.position = _make_point(point[0], point[1], point[2] + 0.035)
        text_marker.pose.orientation.w = 1.0
        text_marker.scale.z = 0.028
        text_marker.color = _make_color(1.0, 1.0, 1.0, 0.98)
        text_marker.text = _validation_label(record)
        markers.append(text_marker)

    if len(path_points) >= 2:
        line_marker.points = [_make_point(x, y, z) for x, y, z in path_points]
        markers.insert(0, line_marker)

    return MarkerArray(markers=markers)


class PipelineRvizPublisher:
    """Publish RViz-friendly topics for point clouds, poses, markers, and TF."""

    def __init__(self, node) -> None:
        self._node = node
        qos = _make_latched_qos()
        self._scene_cloud_pub = node.create_publisher(PointCloud2, "~/rviz/scene_pointcloud", qos)
        self._instance_cloud_pub = node.create_publisher(PointCloud2, "~/rviz/instance_pointcloud", qos)
        self._candidate_grasps_pub = node.create_publisher(PoseArray, "~/rviz/candidate_grasps", qos)
        self._selected_grasp_pub = node.create_publisher(PoseArray, "~/rviz/selected_grasp", qos)
        self._plan_waypoints_pub = node.create_publisher(PoseArray, "~/rviz/plan_waypoints", qos)
        self._candidate_markers_pub = node.create_publisher(MarkerArray, "~/rviz/candidate_markers", qos)
        self._selected_markers_pub = node.create_publisher(MarkerArray, "~/rviz/selected_grasp_markers", qos)
        self._plan_markers_pub = node.create_publisher(MarkerArray, "~/rviz/plan_markers", qos)
        self._camera_transform_pub = node.create_publisher(TransformStamped, "~/rviz/camera_transform", qos)
        self._latest_visual_msgs: list[tuple[object, object]] = []
        self._tf_broadcaster = None
        if TransformBroadcaster is not None:
            try:
                self._tf_broadcaster = TransformBroadcaster(node)
            except Exception as exc:  # pragma: no cover
                node.get_logger().warning(
                    "TF broadcaster is unavailable in this Python 3.10 overlay; "
                    f"falling back to ~/rviz/camera_transform only. cause={exc}"
                )
        self._latest_camera_transform: TransformStamped | None = None
        self._tf_timer = node.create_timer(0.5, self._rebroadcast_latest_transform)
        self._visual_timer = node.create_timer(1.0, self._rebroadcast_latest_visuals)

    def _publish_cached(self, publisher, msg) -> None:
        publisher.publish(msg)
        cached_msg = deepcopy(msg)
        for index, (existing_publisher, _existing_msg) in enumerate(self._latest_visual_msgs):
            if existing_publisher is publisher:
                self._latest_visual_msgs[index] = (publisher, cached_msg)
                return
        self._latest_visual_msgs.append((publisher, cached_msg))

    def _refresh_header_stamp(self, msg, stamp):
        refreshed = deepcopy(msg)
        if hasattr(refreshed, "header"):
            refreshed.header.stamp = stamp
        if isinstance(refreshed, MarkerArray):
            for marker in refreshed.markers:
                marker.header.stamp = stamp
        return refreshed

    def _rebroadcast_latest_visuals(self) -> None:
        if not self._latest_visual_msgs:
            return
        stamp = self._node.get_clock().now().to_msg()
        for publisher, msg in self._latest_visual_msgs:
            publisher.publish(self._refresh_header_stamp(msg, stamp))

    def _rebroadcast_latest_transform(self) -> None:
        if self._latest_camera_transform is None or self._tf_broadcaster is None:
            return
        tf_msg = TransformStamped()
        tf_msg.header.frame_id = self._latest_camera_transform.header.frame_id
        tf_msg.child_frame_id = self._latest_camera_transform.child_frame_id
        tf_msg.header.stamp = self._node.get_clock().now().to_msg()
        tf_msg.transform = self._latest_camera_transform.transform
        self._tf_broadcaster.sendTransform(tf_msg)

    def _update_camera_transform(
        self,
        *,
        base_to_camera: np.ndarray,
        base_frame: str,
        camera_frame: str,
    ) -> None:
        tf_msg = _build_transform_stamped(
            stamp=self._node.get_clock().now().to_msg(),
            parent_frame=base_frame,
            child_frame=camera_frame,
            transform=base_to_camera,
        )
        self._latest_camera_transform = tf_msg
        self._camera_transform_pub.publish(tf_msg)
        if self._tf_broadcaster is not None:
            self._tf_broadcaster.sendTransform(tf_msg)

    def clear(self, camera_frame: str, base_frame: str) -> None:
        stamp = self._node.get_clock().now().to_msg()
        camera_header = Header(frame_id=camera_frame, stamp=stamp)
        base_header = Header(frame_id=base_frame, stamp=stamp)
        self._publish_cached(self._scene_cloud_pub, _make_empty_pointcloud(camera_header))
        self._publish_cached(self._instance_cloud_pub, _make_empty_pointcloud(camera_header))
        self._publish_cached(self._candidate_grasps_pub, _build_pose_array(camera_header, []))
        self._publish_cached(self._selected_grasp_pub, _build_pose_array(camera_header, []))
        self._publish_cached(self._plan_waypoints_pub, _build_pose_array(base_header, []))
        self._publish_cached(self._candidate_markers_pub, _delete_all_marker_array(camera_header))
        self._publish_cached(self._selected_markers_pub, _delete_all_marker_array(camera_header))
        self._publish_cached(self._plan_markers_pub, _delete_all_marker_array(base_header))

    def publish_result(
        self,
        result: dict[str, object],
        *,
        camera_frame: str,
        base_frame: str,
        candidate_topk: int,
    ) -> None:
        stamp = self._node.get_clock().now().to_msg()
        camera_header = Header(frame_id=camera_frame, stamp=stamp)
        base_header = Header(frame_id=base_frame, stamp=stamp)
        base_to_camera = result.get("base_to_camera")
        if isinstance(base_to_camera, np.ndarray):
            self._update_camera_transform(
                base_to_camera=base_to_camera,
                base_frame=base_frame,
                camera_frame=camera_frame,
            )

        perception = result.get("perception")
        if isinstance(perception, PerceptionResult):
            self._publish_cached(self._scene_cloud_pub, _make_xyz_pointcloud(camera_header, perception.scene_points))
            instance_points, instance_rgb = _merge_instance_clouds(perception.pointclouds)
            self._publish_cached(
                self._instance_cloud_pub,
                _make_xyzrgb_pointcloud(camera_header, instance_points, instance_rgb),
            )
        else:
            self._publish_cached(self._scene_cloud_pub, _make_empty_pointcloud(camera_header))
            self._publish_cached(self._instance_cloud_pub, _make_empty_pointcloud(camera_header))

        candidate_poses: list[Pose] = []
        candidate_frames: list[tuple[str, tuple[float, float, float] | np.ndarray, np.ndarray]] = []
        candidate_pool = result.get("candidate_pool") or []
        for candidate_index, item in enumerate(list(candidate_pool)[: max(0, int(candidate_topk))]):
            candidate = item[0] if isinstance(item, tuple) else item
            if isinstance(candidate, GraspCandidate):
                candidate_poses.append(_candidate_to_pose(candidate))
                rotation = PureGraspPlanner._normalize_columns(
                    np.asarray(candidate.rotation_camera, dtype=np.float64).reshape(3, 3) @ PureGraspPlanner._R_ADJUST
                )
                candidate_frames.append(
                    (
                        f"cand{candidate_index}: {candidate.score:.3f}",
                        candidate.translation_camera_m,
                        rotation,
                    )
                )
        self._publish_cached(self._candidate_grasps_pub, _build_pose_array(camera_header, candidate_poses))
        self._publish_cached(
            self._candidate_markers_pub,
            _build_axes_marker_array(
                header=camera_header,
                namespace="candidate_grasps",
                frames=candidate_frames,
                axis_length=0.045,
                axis_thickness=0.0035,
                text_height=0.025,
            )
        )

        selected_candidate = result.get("candidate")
        selected_poses = [_candidate_to_pose(selected_candidate)] if isinstance(selected_candidate, GraspCandidate) else []
        self._publish_cached(self._selected_grasp_pub, _build_pose_array(camera_header, selected_poses))
        selected_frames: list[tuple[str, tuple[float, float, float] | np.ndarray, np.ndarray]] = []
        if isinstance(selected_candidate, GraspCandidate):
            selected_rotation = PureGraspPlanner._normalize_columns(
                np.asarray(selected_candidate.rotation_camera, dtype=np.float64).reshape(3, 3) @ PureGraspPlanner._R_ADJUST
            )
            selected_frames.append(
                (
                    f"selected: {selected_candidate.score:.3f}",
                    selected_candidate.translation_camera_m,
                    selected_rotation,
                )
            )
        self._publish_cached(
            self._selected_markers_pub,
            _build_axes_marker_array(
                header=camera_header,
                namespace="selected_grasp",
                frames=selected_frames,
                axis_length=0.065,
                axis_thickness=0.005,
                text_height=0.03,
            )
        )

        plan = result.get("plan")
        use_pregrasp = bool(result.get("use_pregrasp", True))
        plan_poses: list[Pose] = []
        plan_frames: list[tuple[str, tuple[float, float, float] | np.ndarray, np.ndarray]] = []
        plan_waypoints: list[tuple[str, tuple[float, float, float]]] = []
        if isinstance(plan, GraspPlan):
            rotation = rotation_matrix_from_rpy_deg(*plan.target_rpy_deg)
            ordered_waypoints: list[tuple[str, tuple[float, float, float]]] = []
            if use_pregrasp:
                ordered_waypoints.append(("pregrasp", plan.pregrasp_base_m))
            ordered_waypoints.extend(
                [
                    ("target", plan.target_base_m),
                    ("grasp", plan.grasp_base_m),
                    ("retreat", plan.retreat_base_m),
                ]
            )
            plan_poses.extend([_make_pose(position, rotation) for _name, position in ordered_waypoints])
            plan_frames.extend([(name, position, rotation) for name, position in ordered_waypoints])
            plan_waypoints.extend(ordered_waypoints)
        self._publish_cached(self._plan_waypoints_pub, _build_pose_array(base_header, plan_poses))
        plan_markers = _build_axes_marker_array(
            header=base_header,
            namespace="plan_waypoints_axes",
            frames=plan_frames,
            axis_length=0.05,
            axis_thickness=0.004,
            text_height=0.03,
        )
        path_markers = _build_path_marker_array(
            header=base_header,
            namespace="plan_waypoints_path",
            waypoints=plan_waypoints,
            line_thickness=0.006,
        )
        self._publish_cached(
            self._plan_markers_pub,
            MarkerArray(markers=list(plan_markers.markers) + list(path_markers.markers)),
        )
