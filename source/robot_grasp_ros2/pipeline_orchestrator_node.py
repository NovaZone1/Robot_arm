from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import threading
import time
import traceback

import cv2
import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.exceptions import ParameterUninitializedException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rcl_interfaces.srv import SetParameters
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import MarkerArray

from robot_grasp_msgs.msg import PlacePlan
from robot_grasp_msgs.srv import (
    AnalyzeScene,
    CaptureScene,
    DetectTarget2D,
    ExecuteGraspPlan,
    ExecuteNamedPose,
    ExecutePlacePlan,
    GetRobotState,
    MatchItemLabel,
    MoveBaseRelative,
    StopRobot,
)
from robot_grasp_ros2.distributed_utils import (
    base_to_camera_from_tcp_and_hand_eye,
    build_runtime_config,
    camera_info_to_intrinsics,
    color_image_to_msg,
    candidate_debug_dict,
    color_msg_to_bgr,
    depth_msg_to_meters,
    grasp_candidate_from_msg,
    grasp_plan_to_msg,
    json_dumps,
    load_hand_eye_matrix,
    make_latched_qos,
    matrix_to_transform_msg,
    new_run_id,
    plan_debug_dict,
    pose6d_from_position_m_rpy_deg,
)
from robot_grasp_ros2.rviz_visualization import build_candidate_validation_marker_array
from src.grasping.planning import PureGraspPlanner
from src.perception.item_catalog import (
    ItemCatalog,
    LabelDetection,
    ReferenceLabelMatcher,
    default_item_catalog_path,
)
from src.perception.placement_uv_map import load_mapping_for_item
from src.robot.plan_validation import select_first_reachable_candidate


@dataclass(slots=True)
class PendingConfirmation:
    run_id: str
    prompt: str
    scene_id: str
    plan: object
    move_home_after: bool
    target_item_id: str
    hand_eye: np.ndarray
    request_payload: dict[str, object]
    cycle_records: list[dict[str, object]]
    result_payload: dict[str, object]


class PipelineOrchestratorNode(Node):
    """Service-compatible distributed orchestrator for the migrated grasp pipeline."""

    def __init__(self) -> None:
        super().__init__("grasp_pipeline")
        self._declare_parameters()

        text_qos = make_latched_qos(depth=10)
        diagnostics_qos = make_latched_qos(depth=20)
        self._status_pub = self.create_publisher(String, "~/status", text_qos)
        self._summary_pub = self.create_publisher(String, "~/summary", text_qos)
        self._diagnostics_pub = self.create_publisher(String, "~/diagnostics", diagnostics_qos)
        self._result_pub = self.create_publisher(String, "~/result_json", text_qos)
        self._candidate_validation_markers_pub = self.create_publisher(
            MarkerArray,
            "~/rviz/candidate_validation_markers",
            make_latched_qos(depth=5),
        )

        self.create_subscription(String, "~/run_prompt", self._handle_run_prompt, 10)
        self.create_service(Trigger, "~/run", self._handle_run_service)
        self.create_service(Trigger, "~/probe", self._handle_probe_service)
        self.create_service(
            Trigger,
            "~/scan_placement",
            self._handle_scan_placement_service,
        )
        self.create_service(
            Trigger,
            "~/scan_placement_multi_view",
            self._handle_scan_placement_multi_view_service,
        )
        self.create_service(
            Trigger,
            "~/align_placement_target",
            self._handle_align_placement_target_service,
        )
        self.create_service(
            Trigger,
            "~/scan_and_align_placement_target",
            self._handle_scan_and_align_placement_target_service,
        )
        self.create_service(
            Trigger,
            "~/execute_aligned_place",
            self._handle_execute_aligned_place_service,
        )
        self.create_service(Trigger, "~/stop", self._handle_stop_service)
        self.create_service(Trigger, "~/confirm", self._handle_confirm_service)
        self.create_service(Trigger, "~/reject", self._handle_reject_service)

        self._rpc_callback_group = ReentrantCallbackGroup()
        self._search_preview_lock = threading.Lock()
        self._search_preview_bgr: np.ndarray | None = None
        self._search_preview_sequence = 0
        self._search_preview_received_at = 0.0
        self._capture_client = self.create_client(
            CaptureScene,
            self._service_name("camera_capture_service"),
            callback_group=self._rpc_callback_group,
        )
        self._analyze_client = self.create_client(
            AnalyzeScene,
            self._service_name("vision_analyze_service"),
            callback_group=self._rpc_callback_group,
        )
        self._detect_target_2d_client = self.create_client(
            DetectTarget2D,
            self._service_name("vision_detect_target_2d_service"),
            callback_group=self._rpc_callback_group,
        )
        self._match_label_client = self.create_client(
            MatchItemLabel,
            self._service_name("vision_match_label_service"),
            callback_group=self._rpc_callback_group,
        )
        self._executor_set_parameters_client = self.create_client(
            SetParameters,
            "/robot_executor/set_parameters",
            callback_group=self._rpc_callback_group,
        )
        self._get_state_client = self.create_client(
            GetRobotState,
            self._service_name("robot_state_service"),
            callback_group=self._rpc_callback_group,
        )
        self._named_pose_client = self.create_client(
            ExecuteNamedPose,
            self._service_name("robot_named_pose_service"),
            callback_group=self._rpc_callback_group,
        )
        self._execute_plan_client = self.create_client(
            ExecuteGraspPlan,
            self._service_name("robot_execute_plan_service"),
            callback_group=self._rpc_callback_group,
        )
        self._execute_place_client = self.create_client(
            ExecutePlacePlan,
            self._service_name("robot_execute_place_service"),
            callback_group=self._rpc_callback_group,
        )
        self._stop_robot_client = self.create_client(
            StopRobot,
            self._service_name("robot_stop_service"),
            callback_group=self._rpc_callback_group,
        )
        self._move_base_client = self.create_client(
            MoveBaseRelative,
            self._service_name("base_move_service"),
            callback_group=self._rpc_callback_group,
        )
        self._stop_base_client = self.create_client(
            Trigger,
            self._service_name("base_stop_service"),
            callback_group=self._rpc_callback_group,
        )
        self.create_subscription(
            Odometry,
            str(self.get_parameter("base_odom_topic").value),
            self._handle_base_odometry,
            20,
            callback_group=self._rpc_callback_group,
        )
        self.create_subscription(
            Image,
            str(self.get_parameter("continuous_search_image_topic").value),
            self._handle_search_preview,
            5,
            callback_group=self._rpc_callback_group,
        )

        self._run_lock = threading.Lock()
        self._run_thread: threading.Thread | None = None
        self._run_id: str | None = None
        self._stop_requested = False
        self._scan_active = False
        self._pending_confirmation: PendingConfirmation | None = None
        self._item_catalog_cache: ItemCatalog | None = None
        self._target_card_matcher_cache: ReferenceLabelMatcher | None = None
        self._last_card_search_travel_m = 0.0
        self._last_placement_scan_travel_m = 0.0
        self._grasp_exclude_roi_norm: list[float] | None = None
        self._grasp_search_roi_norm: list[float] | None = None
        self._base_odom_lock = threading.Lock()
        self._latest_base_odom: tuple[float, float, float, float] | None = None

        self._auto_start_armed = bool(self.get_parameter("auto_start").value)
        self._auto_start_timer = self.create_timer(0.5, self._maybe_auto_start)
        self._publish_status("idle")

    def _declare_parameters(self) -> None:
        self.declare_parameter("prompt", "")
        self.declare_parameter("target_item_id", "")
        self.declare_parameter("item_catalog_path", "")
        self.declare_parameter("execute", False)
        self.declare_parameter("place_after_grasp", False)
        self.declare_parameter("move_to_placement_observation_after_grasp", True)
        self.declare_parameter("dynamic_box_localization", True)
        self.declare_parameter("move_home_after", True)
        self.declare_parameter("enable_pregrasp", False)
        self.declare_parameter("show_pointcloud", False)
        self.declare_parameter("precenter", False)
        self.declare_parameter("confirm", False)
        self.declare_parameter("pointcloud_filter_mode", "bilateral")
        self.declare_parameter("pointcloud_backend", "sdk")
        self.declare_parameter("depth_fusion_frames", 8)
        # Search phases only need a current RGB image.  Reserve the full
        # multi-frame depth fusion for the stopped, final grasp/placement
        # validation frame.
        self.declare_parameter("search_depth_fusion_frames", 1)
        self.declare_parameter("continuous_search_enabled", True)
        self.declare_parameter(
            "continuous_search_image_topic", "/camera_server/latest/color"
        )
        self.declare_parameter("continuous_search_preview_max_age_s", 0.8)
        self.declare_parameter("continuous_search_preview_wait_s", 1.2)
        self.declare_parameter("continuous_search_stop_on_center", True)
        self.declare_parameter("continuous_search_poll_s", 0.08)
        self.declare_parameter("speed", 25)
        self.declare_parameter("observation_speed", 25)
        self.declare_parameter("home_speed", 25)
        self.declare_parameter("graspnet_checkpoint", "checkpoint.tar")
        self.declare_parameter("hand_eye_config", "")
        self.declare_parameter("apply_npoint_tool_offset", False)
        self.declare_parameter("npoint_tool_offset_file", "")
        self.declare_parameter("tool_contact_offset_scale", 1.0)
        self.declare_parameter("safe_top_down_candidate_filter", False)
        self.declare_parameter("use_object_center_contact", False)
        self.declare_parameter("object_center_contact_max_offset_m", 0.08)
        self.declare_parameter("manual_target_bias_x_mm", 0.0)
        self.declare_parameter("manual_target_bias_y_mm", 0.0)
        self.declare_parameter("manual_target_bias_z_mm", 0.0)
        self.declare_parameter("table_z_m", 0.0)
        self.declare_parameter("min_gripper_table_clearance_m", 0.03)
        self.declare_parameter("color_block_center_height_m", 0.060)
        self.declare_parameter("pregrasp_offset_m", 0.0)
        self.declare_parameter("descend_offset_m", 0.0)
        self.declare_parameter("grasp_z_offset_m", 0.0)
        self.declare_parameter("retreat_offset_m", 0.0)
        self.declare_parameter("workspace_x", [0.10, 1.20])
        self.declare_parameter("workspace_y", [-0.50, 0.50])
        self.declare_parameter("workspace_z", [0.00, 0.60])
        self.declare_parameter("robot_validation_candidate_limit", 6)
        self.declare_parameter("robot_validation_variant_limit", 4)
        self.declare_parameter("extra_cli_args", [""])
        self.declare_parameter("auto_start", False)
        self.declare_parameter("skip_observation_move", False)
        self.declare_parameter("auto_target_from_card", False)
        self.declare_parameter("target_card_search_roi_norm", [0.0, 0.0, 1.0, 1.0])
        self.declare_parameter("target_card_match_threshold", 0.50)
        self.declare_parameter("target_card_min_confidence", 0.55)
        self.declare_parameter("target_card_min_margin", 0.08)
        self.declare_parameter("target_card_capture_frames", 3)
        self.declare_parameter("target_card_consensus_frames", 2)
        self.declare_parameter("target_card_capture_interval_s", 0.15)
        # If the arm observation pose is slightly behind the printed target
        # card, let Scout make a short, slow forward search instead of
        # repeatedly analyzing the same empty view.
        self.declare_parameter("target_card_base_search_enabled", False)
        self.declare_parameter("target_card_base_search_step_m", 0.07)
        self.declare_parameter("target_card_base_search_max_travel_m", 0.35)
        self.declare_parameter("target_card_base_search_speed_mps", 0.03)
        self.declare_parameter("target_card_base_search_timeout_s", 18.0)
        # First failure returns to the observation pose and retries. Two
        # repeats means three attempts total, then the run skips grasping.
        self.declare_parameter("target_card_max_retries", 2)
        self.declare_parameter("grasp_scan_max_retries", 2)
        self.declare_parameter("placement_scan_max_retries", 2)
        self.declare_parameter("observe_pose", [30.0, 0.0, 400.0, 0.0, 120.0, 0.0])
        self.declare_parameter("placement_observe_pose", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.declare_parameter(
            "placement_observe_joint_positions_rad",
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        self.declare_parameter("label_search_roi_norm", [0.0, 0.0, 1.0, 1.0])
        self.declare_parameter("label_match_threshold", 0.42)
        # The six box labels are printed catalog images.  Keep generic HSV
        # color/shape markers disabled by default: ground shadows, reflections
        # and a held blue block must not be sufficient to stop the base scan.
        self.declare_parameter("label_marker_detection_enabled", False)
        self.declare_parameter("camera_capture_service", "/camera_server/capture")
        self.declare_parameter("vision_analyze_service", "/vision_worker/analyze")
        self.declare_parameter(
            "vision_detect_target_2d_service",
            "/vision_worker/detect_target_2d",
        )
        self.declare_parameter("vision_match_label_service", "/vision_worker/match_item_label")
        self.declare_parameter("robot_state_service", "/robot_executor/get_state")
        self.declare_parameter("robot_named_pose_service", "/robot_executor/execute_named_pose")
        self.declare_parameter("robot_execute_plan_service", "/robot_executor/execute_grasp_plan")
        self.declare_parameter("robot_execute_place_service", "/robot_executor/execute_place_plan")
        self.declare_parameter("robot_stop_service", "/robot_executor/stop_robot")
        self.declare_parameter("artifact_root", "")
        self.declare_parameter("placement_scan_viz_dir", "")
        self.declare_parameter("base_multiview_enabled", False)
        self.declare_parameter("base_alignment_enabled", False)
        self.declare_parameter("base_target_alignment_enabled", False)
        self.declare_parameter("base_aligned_place_enabled", False)
        self.declare_parameter("base_grasp_scan_enabled", False)
        self.declare_parameter("base_multiview_offset_m", 0.15)
        self.declare_parameter("base_multiview_max_travel_m", 1.50)
        self.declare_parameter("base_multiview_max_views", 24)
        self.declare_parameter("base_multiview_speed_mps", 0.04)
        self.declare_parameter("base_multiview_move_timeout_s", 22.0)
        self.declare_parameter("base_multiview_settle_s", 0.8)
        self.declare_parameter("base_target_center_tolerance_norm", 0.18)
        self.declare_parameter("base_grasp_bottle_center_norm", [0.598, 0.485])
        self.declare_parameter("base_grasp_block_center_norm", [0.606, 0.619])
        self.declare_parameter("base_grasp_center_tolerance_u_norm", 0.18)
        self.declare_parameter("base_grasp_center_tolerance_v_norm", 0.24)
        self.declare_parameter("base_target_fine_step_m", 0.07)
        self.declare_parameter("grasp_scan_lost_frames_before_reverse", 2)
        self.declare_parameter("post_grasp_base_advance_m", 1.50)
        self.declare_parameter("post_grasp_base_advance_speed_mps", 0.10)
        self.declare_parameter("post_grasp_base_advance_timeout_s", 50.0)
        self.declare_parameter("post_place_base_advance_m", 1.50)
        self.declare_parameter("post_place_base_advance_speed_mps", 0.10)
        self.declare_parameter("post_place_base_advance_timeout_s", 50.0)
        self.declare_parameter("post_place_home_pose", [57.0, 0.0, 215.0, 0.0, 85.0, 0.0])
        self.declare_parameter("base_odom_topic", "/odom")
        self.declare_parameter(
            "base_move_service",
            "/base_scan_controller/move_relative",
        )
        self.declare_parameter("base_stop_service", "/base_scan_controller/stop")
        self.declare_parameter("use_cached_multiview_box_map", False)
        self.declare_parameter("cached_box_map_max_age_s", 600.0)
        self.declare_parameter("cached_box_map_position_tolerance_m", 0.025)
        self.declare_parameter("cached_box_map_yaw_tolerance_deg", 2.0)

    def _publish_status(self, text: str) -> None:
        self.get_logger().info(text)
        self._status_pub.publish(String(data=text))

    def _handle_base_odometry(self, message: Odometry) -> None:
        position = message.pose.pose.position
        q = message.pose.pose.orientation
        yaw = math.atan2(
            2.0 * ((q.w * q.z) + (q.x * q.y)),
            1.0 - (2.0 * ((q.y * q.y) + (q.z * q.z))),
        )
        with self._base_odom_lock:
            self._latest_base_odom = (
                float(position.x),
                float(position.y),
                float(yaw),
                time.monotonic(),
            )

    def _handle_search_preview(self, message: Image) -> None:
        try:
            image = color_msg_to_bgr(message).copy()
        except Exception as exc:
            self.get_logger().warning(f"continuous search preview ignored: {exc}")
            return
        with self._search_preview_lock:
            self._search_preview_bgr = image
            self._search_preview_sequence += 1
            self._search_preview_received_at = time.monotonic()

    def _wait_for_search_preview(
        self,
        *,
        after_sequence: int = -1,
        wait_s: float | None = None,
    ) -> tuple[np.ndarray, int] | None:
        if not bool(self.get_parameter("continuous_search_enabled").value):
            return None
        deadline = time.monotonic() + max(
            0.0,
            float(self.get_parameter("continuous_search_preview_wait_s").value)
            if wait_s is None
            else float(wait_s),
        )
        maximum_age = max(
            0.05,
            float(self.get_parameter("continuous_search_preview_max_age_s").value),
        )
        while time.monotonic() <= deadline:
            with self._search_preview_lock:
                image = self._search_preview_bgr
                sequence = self._search_preview_sequence
                age = time.monotonic() - self._search_preview_received_at
                if (
                    image is not None
                    and sequence > after_sequence
                    and age <= maximum_age
                ):
                    return image.copy(), sequence
            time.sleep(0.02)
        return None

    def _detect_target_2d_from_preview(
        self,
        *,
        run_id: str,
        prompt: str,
        after_sequence: int = -1,
        wait_s: float | None = None,
    ) -> tuple[object, int] | None:
        preview = self._wait_for_search_preview(
            after_sequence=after_sequence,
            wait_s=wait_s,
        )
        if preview is None:
            return None
        preview_bgr, preview_sequence = preview
        request = DetectTarget2D.Request()
        request.run_id = run_id
        request.prompt = prompt
        request.color_image = color_image_to_msg(
            preview_bgr,
            frame_id="camera_color_optical_frame",
            stamp=self.get_clock().now().to_msg(),
        )
        if hasattr(request, "options_json"):
            request.options_json = self._detect_target_2d_options_json()
        response = self._call_client(
            self._detect_target_2d_client,
            request,
            timeout_s=30.0,
        )
        if not response.success:
            raise RuntimeError(response.message)
        return response, int(preview_sequence)

    def _base_odom_snapshot(self) -> tuple[float, float, float]:
        with self._base_odom_lock:
            snapshot = self._latest_base_odom
        if snapshot is None:
            raise RuntimeError("Scout /odom has not been received")
        x, y, yaw, received_at = snapshot
        if time.monotonic() - received_at > 0.75:
            raise RuntimeError("Scout /odom is stale")
        return x, y, yaw

    def _publish_summary(self, text: str) -> None:
        self._summary_pub.publish(String(data=text))

    def _publish_diagnostics(self, lines: list[str]) -> None:
        for line in lines:
            self._diagnostics_pub.publish(String(data=line))

    def _publish_result(self, payload: dict[str, object]) -> None:
        self._result_pub.publish(String(data=json_dumps(payload)))

    def _publish_candidate_validation_markers(
        self,
        *,
        validation_records: list[dict[str, object]],
        camera_frame: str,
    ) -> None:
        marker_array = build_candidate_validation_marker_array(
            validation_records=validation_records,
            camera_frame=str(camera_frame or "camera_color_optical_frame"),
            stamp=self.get_clock().now().to_msg(),
        )
        self._candidate_validation_markers_pub.publish(marker_array)

    def _service_name(self, parameter_name: str) -> str:
        return str(self.get_parameter(parameter_name).value)

    def _peek_pending_confirmation(self) -> PendingConfirmation | None:
        with self._run_lock:
            return self._pending_confirmation

    def _set_pending_confirmation(self, pending: PendingConfirmation | None) -> None:
        with self._run_lock:
            self._pending_confirmation = pending

    @staticmethod
    def _append_summary_line(summary: str, line: str) -> str:
        base = str(summary or "").strip()
        extra = str(line).strip()
        if not base:
            return extra
        if not extra:
            return base
        return f"{base}\n{extra}"

    def _artifact_root_dir(self) -> Path:
        configured = str(self.get_parameter("artifact_root").value or "").strip()
        if configured:
            return Path(configured).expanduser().resolve()
        return Path(__file__).resolve().parents[2] / "log" / "distributed_runs"

    def _run_artifact_dir(self, run_id: str) -> Path:
        return self._artifact_root_dir() / run_id

    def _placement_scan_viz_dir(self) -> Path:
        configured = str(self.get_parameter("placement_scan_viz_dir").value or "").strip()
        if configured:
            return Path(configured).expanduser().resolve()
        return Path(__file__).resolve().parents[2] / "ros_ws" / "viz" / "placement_scan"

    @staticmethod
    def _pose_debug_dict(pose) -> dict[str, float]:
        return {
            "x_mm": float(pose.x_mm),
            "y_mm": float(pose.y_mm),
            "z_mm": float(pose.z_mm),
            "roll_deg": float(pose.roll_deg),
            "pitch_deg": float(pose.pitch_deg),
            "yaw_deg": float(pose.yaw_deg),
        }

    @staticmethod
    def _capture_debug_dict(capture_response) -> dict[str, object]:
        return {
            "scene_id": str(capture_response.scene_id),
            "camera_frame": str(capture_response.camera_frame),
            "color_width": int(capture_response.color_image.width),
            "color_height": int(capture_response.color_image.height),
            "depth_width": int(capture_response.depth_image.width),
            "depth_height": int(capture_response.depth_image.height),
            "camera_info_width": int(capture_response.camera_info.width),
            "camera_info_height": int(capture_response.camera_info.height),
            "camera_intrinsics": [
                float(value)
                for value in list(capture_response.camera_info.k)
            ],
        }

    @staticmethod
    def _analyze_debug_dict(analyze_response) -> dict[str, object]:
        perception = analyze_response.perception
        candidate_count = int(len(analyze_response.candidate_pool))
        selected_candidate = (
            candidate_debug_dict(grasp_candidate_from_msg(analyze_response.selected_candidate))
            if analyze_response.has_selected_candidate
            else None
        )
        candidate_pool = [
            candidate_debug_dict(grasp_candidate_from_msg(item))
            for item in list(analyze_response.candidate_pool)[:20]
        ]
        diagnostics = json.loads(analyze_response.diagnostics_json).get("diagnostics", [])
        return {
            "message": str(analyze_response.message),
            "summary": str(analyze_response.summary),
            "diagnostics": diagnostics,
            "candidate_count": candidate_count,
            "has_selected_candidate": bool(analyze_response.has_selected_candidate),
            "selected_candidate": selected_candidate,
            "candidate_pool": candidate_pool,
            "perception": {
                "scene_id": str(perception.scene_id),
                "prompt": str(perception.prompt),
                "camera_frame": str(perception.camera_frame),
                "instance_count": int(perception.instance_count),
                "scene_grasp_count": int(perception.scene_grasp_count),
                "scene_point_count": int(perception.scene_point_count),
                "object_point_counts": [int(value) for value in list(perception.object_point_counts)],
                "debug_lines": [str(line) for line in list(perception.debug_lines)],
            },
        }

    @staticmethod
    def _write_json_file(path: Path, payload: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _write_run_artifacts(
        self,
        *,
        run_id: str,
        request_payload: dict[str, object],
        cycle_records: list[dict[str, object]],
        result_payload: dict[str, object],
    ) -> Path:
        artifact_dir = self._run_artifact_dir(run_id)
        result_payload_for_file = dict(result_payload)
        execution_trace: list[object] = []
        candidate_validation: list[object] = []
        execution_payload = result_payload_for_file.get("execution")
        if isinstance(execution_payload, dict):
            execution_payload_for_file = dict(execution_payload)
            raw_trace = execution_payload_for_file.pop("execution_trace", None)
            if isinstance(raw_trace, list):
                execution_trace = list(raw_trace)
            for phase_name in ("grasp", "placement"):
                phase_payload = execution_payload_for_file.get(phase_name)
                if not isinstance(phase_payload, dict):
                    continue
                phase_payload_for_file = dict(phase_payload)
                if phase_name == "placement" and isinstance(phase_payload_for_file.get("execution"), dict):
                    nested = dict(phase_payload_for_file["execution"])
                    phase_trace = nested.pop("execution_trace", None)
                    phase_payload_for_file["execution"] = nested
                else:
                    phase_trace = phase_payload_for_file.pop("execution_trace", None)
                if isinstance(phase_trace, list):
                    execution_trace.extend(
                        [{"task_phase": phase_name, **dict(item)} for item in phase_trace]
                    )
                execution_payload_for_file[phase_name] = phase_payload_for_file
            result_payload_for_file["execution"] = execution_payload_for_file
        raw_candidate_validation = result_payload_for_file.pop("candidate_validation", None)
        if isinstance(raw_candidate_validation, list):
            candidate_validation = list(raw_candidate_validation)
        self._write_json_file(artifact_dir / "request.json", request_payload)
        self._write_json_file(
            artifact_dir / "cycles.json",
            {
                "run_id": run_id,
                "cycle_count": len(cycle_records),
                "cycles": cycle_records,
            },
        )
        if execution_trace:
            self._write_json_file(
                artifact_dir / "execution_trace.json",
                {
                    "run_id": run_id,
                    "execution_trace": execution_trace,
                },
            )
        if candidate_validation:
            self._write_json_file(
                artifact_dir / "candidate_validation.json",
                {
                    "run_id": run_id,
                    "candidate_validation": candidate_validation,
                },
            )
        self._write_json_file(artifact_dir / "final_result.json", result_payload_for_file)

        artifact_root = self._artifact_root_dir()
        artifact_root.mkdir(parents=True, exist_ok=True)
        (artifact_root / "latest_run.txt").write_text(str(artifact_dir) + "\n", encoding="utf-8")
        with (artifact_root / "index.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "run_id": run_id,
                        "artifact_dir": str(artifact_dir),
                        "status": str(result_payload_for_file.get("status", "")),
                        "prompt": str(result_payload_for_file.get("prompt", "")),
                        "scene_id": str(result_payload_for_file.get("scene_id", "")),
                        "summary": str(result_payload_for_file.get("summary", "")),
                        "timestamp_unix_s": time.time(),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
        return artifact_dir

    def _finalize_pending_confirmation(
        self,
        *,
        pending: PendingConfirmation,
        result_payload: dict[str, object],
        status: str,
        summary: str,
        diagnostics: list[str],
    ) -> None:
        final_payload = dict(result_payload)
        final_payload["status"] = status
        final_payload["summary"] = summary
        final_payload["diagnostics"] = diagnostics
        final_payload.setdefault("confirmed", False)
        final_payload.setdefault("execution", None)
        final_payload.setdefault("artifacts", {})
        final_payload["artifacts"]["artifact_root"] = str(self._artifact_root_dir())

        artifact_dir = self._write_run_artifacts(
            run_id=pending.run_id,
            request_payload=pending.request_payload,
            cycle_records=pending.cycle_records,
            result_payload=final_payload,
        )
        final_payload["artifacts"]["run_dir"] = str(artifact_dir)
        self._write_json_file(artifact_dir / "final_result.json", final_payload)

        final_status = f"{status}: {summary}" if summary else status
        self._publish_status(final_status)
        if summary:
            self._publish_summary(summary)
        if diagnostics:
            self._publish_diagnostics(diagnostics)
        self._publish_result(final_payload)

    def _options_payload(self) -> dict[str, object]:
        payload = {
            "prompt": str(self.get_parameter("prompt").value or ""),
            "target_item_id": str(self.get_parameter("target_item_id").value or ""),
            "execute": bool(self.get_parameter("execute").value),
            "place_after_grasp": bool(self.get_parameter("place_after_grasp").value),
            "move_to_placement_observation_after_grasp": bool(
                self.get_parameter(
                    "move_to_placement_observation_after_grasp"
                ).value
            ),
            "dynamic_box_localization": bool(
                self.get_parameter("dynamic_box_localization").value
            ),
            "base_grasp_scan_enabled": bool(
                self.get_parameter("base_grasp_scan_enabled").value
            ),
            "auto_target_from_card": self._auto_target_from_card_enabled(),
            "move_home_after": bool(self.get_parameter("move_home_after").value),
            "enable_pregrasp": bool(self.get_parameter("enable_pregrasp").value),
            "show_pointcloud": bool(self.get_parameter("show_pointcloud").value),
            "precenter": bool(self.get_parameter("precenter").value),
            "confirm": bool(self.get_parameter("confirm").value),
            "pointcloud_filter_mode": str(self.get_parameter("pointcloud_filter_mode").value or "bilateral"),
            "pointcloud_backend": str(self.get_parameter("pointcloud_backend").value or "sdk"),
            "depth_fusion_frames": int(self.get_parameter("depth_fusion_frames").value),
            "speed": int(self.get_parameter("speed").value),
            "observation_speed": int(self.get_parameter("observation_speed").value),
            "home_speed": int(self.get_parameter("home_speed").value),
            "prefer_object_center_candidates": bool(
                str(self.get_parameter("target_item_id").value or "").strip()
            ),
            "graspnet_checkpoint": str(self.get_parameter("graspnet_checkpoint").value or "checkpoint.tar"),
            "apply_npoint_tool_offset": bool(self.get_parameter("apply_npoint_tool_offset").value),
            "npoint_tool_offset_file": str(self.get_parameter("npoint_tool_offset_file").value or ""),
            "tool_contact_offset_scale": float(self.get_parameter("tool_contact_offset_scale").value),
            "safe_top_down_candidate_filter": bool(self.get_parameter("safe_top_down_candidate_filter").value),
            "use_object_center_contact": bool(self.get_parameter("use_object_center_contact").value),
            "object_center_contact_max_offset_m": float(
                self.get_parameter("object_center_contact_max_offset_m").value
            ),
            "manual_target_bias_x_mm": float(self.get_parameter("manual_target_bias_x_mm").value),
            "manual_target_bias_y_mm": float(self.get_parameter("manual_target_bias_y_mm").value),
            "manual_target_bias_z_mm": float(self.get_parameter("manual_target_bias_z_mm").value),
            "table_z_m": float(self.get_parameter("table_z_m").value),
            "min_gripper_table_clearance_m": float(self.get_parameter("min_gripper_table_clearance_m").value),
            "color_block_center_height_m": float(
                self.get_parameter("color_block_center_height_m").value
            ),
            "pregrasp_offset_m": float(self.get_parameter("pregrasp_offset_m").value),
            "descend_offset_m": float(self.get_parameter("descend_offset_m").value),
            "grasp_z_offset_m": float(self.get_parameter("grasp_z_offset_m").value),
            "retreat_offset_m": float(self.get_parameter("retreat_offset_m").value),
            "workspace_x": [float(value) for value in list(self.get_parameter("workspace_x").value or [0.10, 1.20])],
            "workspace_y": [float(value) for value in list(self.get_parameter("workspace_y").value or [-0.50, 0.50])],
            "workspace_z": [float(value) for value in list(self.get_parameter("workspace_z").value or [0.00, 0.60])],
            "robot_validation_candidate_limit": int(self.get_parameter("robot_validation_candidate_limit").value),
            "robot_validation_variant_limit": int(self.get_parameter("robot_validation_variant_limit").value),
        }
        hand_eye_config = str(self.get_parameter("hand_eye_config").value or "").strip()
        if hand_eye_config:
            payload["hand_eye_config"] = hand_eye_config
        try:
            extra_cli_args = list(self.get_parameter("extra_cli_args").value or [])
        except ParameterUninitializedException:
            extra_cli_args = []
        extra_cli_args = [str(item) for item in extra_cli_args if str(item).strip()]
        if extra_cli_args:
            payload["extra_cli_args"] = extra_cli_args
        return payload

    def _item_catalog(self) -> ItemCatalog:
        if self._item_catalog_cache is None:
            configured = str(self.get_parameter("item_catalog_path").value or "").strip()
            path = Path(configured).expanduser().resolve() if configured else default_item_catalog_path()
            self._item_catalog_cache = ItemCatalog.load(path)
        return self._item_catalog_cache

    def _resolve_target_item(self, prompt: str):
        requested = str(self.get_parameter("target_item_id").value or "").strip()
        item = self._item_catalog().resolve(requested or prompt)
        if requested and item is None:
            raise RuntimeError(f"unknown target_item_id: {requested}")
        return item

    def _target_card_matcher(self) -> ReferenceLabelMatcher:
        if self._target_card_matcher_cache is None:
            self._target_card_matcher_cache = ReferenceLabelMatcher(self._item_catalog())
        return self._target_card_matcher_cache

    def _auto_target_from_card_enabled(self) -> bool:
        """Return the mode flag, retaining compatibility with partial test nodes."""
        try:
            return bool(self.get_parameter("auto_target_from_card").value)
        except (KeyError, ParameterUninitializedException):
            return False

    @staticmethod
    def _select_target_card_detection(
        detections: tuple[LabelDetection, ...] | list[LabelDetection],
        *,
        minimum_confidence: float,
        minimum_margin: float,
    ) -> LabelDetection:
        ranked = sorted(detections, key=lambda value: float(value.confidence), reverse=True)
        if not ranked:
            raise RuntimeError("target card not recognized: no catalog image detected")
        best = ranked[0]
        if float(best.confidence) < float(minimum_confidence):
            raise RuntimeError(
                "target card confidence too low: "
                f"best={best.item_id} confidence={float(best.confidence):.3f} "
                f"required={float(minimum_confidence):.3f}"
            )
        if len(ranked) > 1:
            runner_up = ranked[1]
            margin = float(best.confidence) - float(runner_up.confidence)
            if margin < float(minimum_margin):
                raise RuntimeError(
                    "target card is ambiguous: "
                    f"best={best.item_id}:{float(best.confidence):.3f} "
                    f"second={runner_up.item_id}:{float(runner_up.confidence):.3f} "
                    f"margin={margin:.3f} required={float(minimum_margin):.3f}"
                )
        return best

    @classmethod
    def _select_target_card_consensus(
        cls,
        frame_detections: list[tuple[LabelDetection, ...]],
        *,
        minimum_confidence: float,
        minimum_margin: float,
        minimum_votes: int,
    ) -> tuple[LabelDetection, dict[str, object]]:
        """Require the same unambiguous catalog winner in multiple frames."""
        winners: list[LabelDetection] = []
        rejected_frames: list[dict[str, object]] = []
        for frame_index, detections in enumerate(frame_detections):
            try:
                winners.append(
                    cls._select_target_card_detection(
                        detections,
                        minimum_confidence=minimum_confidence,
                        minimum_margin=minimum_margin,
                    )
                )
            except RuntimeError as error:
                rejected_frames.append(
                    {"frame_index": frame_index, "reason": str(error)}
                )

        votes: dict[str, list[LabelDetection]] = {}
        for winner in winners:
            votes.setdefault(winner.item_id, []).append(winner)
        ranked_votes = sorted(
            votes.items(),
            key=lambda entry: (
                len(entry[1]),
                max(float(value.confidence) for value in entry[1]),
            ),
            reverse=True,
        )
        if not ranked_votes:
            reasons = "; ".join(
                str(value["reason"]) for value in rejected_frames
            )
            raise RuntimeError(
                "target card not stable in any frame"
                + (f": {reasons}" if reasons else "")
            )
        item_id, item_winners = ranked_votes[0]
        vote_count = len(item_winners)
        if vote_count < max(1, int(minimum_votes)):
            vote_summary = ", ".join(
                f"{candidate_id}={len(candidate_winners)}"
                for candidate_id, candidate_winners in ranked_votes
            )
            raise RuntimeError(
                "target card multi-frame consensus failed: "
                f"best={item_id} votes={vote_count} "
                f"required={max(1, int(minimum_votes))} all=[{vote_summary}]"
            )
        if len(ranked_votes) > 1 and len(ranked_votes[1][1]) == vote_count:
            raise RuntimeError(
                "target card multi-frame consensus tied: "
                f"{item_id}={vote_count} {ranked_votes[1][0]}={vote_count}"
            )
        selected = max(item_winners, key=lambda value: float(value.confidence))
        return selected, {
            "winning_item_id": item_id,
            "winning_votes": vote_count,
            "required_votes": max(1, int(minimum_votes)),
            "valid_frame_winners": [value.item_id for value in winners],
            "rejected_frames": rejected_frames,
        }

    def _identify_target_card_once(self, *, run_id: str) -> tuple[object, dict[str, object]]:
        self._publish_status(f"identifying_target_card: run_id={run_id}")
        roi_values = tuple(
            float(value)
            for value in list(self.get_parameter("target_card_search_roi_norm").value or [])
        )
        if len(roi_values) != 4:
            raise RuntimeError("target_card_search_roi_norm must contain 4 values")
        capture_frames = max(
            1, int(self.get_parameter("target_card_capture_frames").value)
        )
        consensus_frames = max(
            1, int(self.get_parameter("target_card_consensus_frames").value)
        )
        if consensus_frames > capture_frames:
            raise RuntimeError(
                "target_card_consensus_frames cannot exceed target_card_capture_frames"
            )
        capture_interval_s = max(
            0.0,
            float(self.get_parameter("target_card_capture_interval_s").value),
        )
        artifact_dir = self._run_artifact_dir(run_id) / "target_card"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        frame_records: list[dict[str, object]] = []
        frame_detections: list[tuple[LabelDetection, ...]] = []
        frame_images: list[np.ndarray] = []
        frame_overlays: list[np.ndarray] = []
        search_roi: tuple[int, int, int, int] | None = None
        last_preview_sequence = -1
        for frame_index in range(capture_frames):
            preview = self._wait_for_search_preview(
                after_sequence=last_preview_sequence
            )
            if preview is not None:
                image, last_preview_sequence = preview
                scene_id = f"continuous-preview-{last_preview_sequence}"
            else:
                capture = self._capture_scene_once(
                    run_id=run_id,
                    phase_label=f"target_card_{frame_index:02d}",
                    depth_fusion_frames=int(
                        self.get_parameter("search_depth_fusion_frames").value
                    ),
                )
                image = color_msg_to_bgr(capture.color_image)
                scene_id = str(capture.scene_id)
            detections, current_search_roi = self._target_card_matcher().match_all(
                image,
                roi_norm=roi_values,
                threshold=float(
                    self.get_parameter("target_card_match_threshold").value
                ),
                # A target card contains a catalog photograph, unlike the six
                # box labels where direct HSV marker detection is useful.
                # Template-only matching prevents chair/box reflections and
                # real objects from winning on color alone.
                marker_detection_enabled=False,
            )
            search_roi = current_search_roi
            overlay = image.copy()
            roi_x, roi_y, roi_w, roi_h = current_search_roi
            cv2.rectangle(
                overlay,
                (roi_x, roi_y),
                (roi_x + roi_w, roi_y + roi_h),
                (255, 255, 0),
                2,
            )
            for detection in detections:
                x, y, width, height = detection.bbox_xywh
                color = (0, 165, 255)
                cv2.rectangle(overlay, (x, y), (x + width, y + height), color, 2)
                cv2.putText(
                    overlay,
                    f"{detection.item_id} {float(detection.confidence):.2f}",
                    (x, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    color,
                    2,
                    cv2.LINE_AA,
                )
            frame_color_path = artifact_dir / f"frame_{frame_index:02d}_color.png"
            frame_overlay_path = artifact_dir / f"frame_{frame_index:02d}_overlay.png"
            cv2.imwrite(str(frame_color_path), image)
            cv2.imwrite(str(frame_overlay_path), overlay)
            frame_images.append(image)
            frame_overlays.append(overlay)
            frame_detections.append(tuple(detections))
            frame_records.append(
                {
                    "frame_index": frame_index,
                    "scene_id": scene_id,
                    "detections": [
                        {
                            "item_id": detection.item_id,
                            "confidence": float(detection.confidence),
                            "bbox_xywh": list(detection.bbox_xywh),
                            "method": detection.method,
                        }
                        for detection in sorted(
                            detections,
                            key=lambda value: float(value.confidence),
                            reverse=True,
                        )
                    ],
                    "color_path": str(frame_color_path),
                    "overlay_path": str(frame_overlay_path),
                }
            )

            # The normal policy is two matching, high-confidence photographs
            # out of at most three.  Do not incur a third camera transaction
            # once that policy has already been satisfied; retain the third
            # frame only as a tie/noise fallback.
            if len(frame_detections) >= consensus_frames:
                try:
                    self._select_target_card_consensus(
                        frame_detections,
                        minimum_confidence=float(
                            self.get_parameter("target_card_min_confidence").value
                        ),
                        minimum_margin=float(
                            self.get_parameter("target_card_min_margin").value
                        ),
                        minimum_votes=consensus_frames,
                    )
                except RuntimeError:
                    # A third frame may still resolve a weak or conflicting
                    # result, so continue until the configured maximum.
                    pass
                else:
                    break
            if frame_index + 1 < capture_frames and capture_interval_s > 0.0:
                time.sleep(capture_interval_s)

        # Fail closed unless the same clear winner repeats in multiple frames.
        selected, consensus = self._select_target_card_consensus(
            frame_detections,
            minimum_confidence=float(
                self.get_parameter("target_card_min_confidence").value
            ),
            minimum_margin=float(self.get_parameter("target_card_min_margin").value),
            minimum_votes=consensus_frames,
        )
        selected_frame_index = next(
            index
            for index, detections in enumerate(frame_detections)
            if selected in detections
        )
        image = frame_images[selected_frame_index]
        overlay = frame_overlays[selected_frame_index]
        color_path = artifact_dir / "color.png"
        overlay_path = artifact_dir / "overlay.png"
        cv2.imwrite(str(color_path), image)
        item = self._item_catalog().require(selected.item_id)
        x, y, width, height = selected.bbox_xywh
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 220, 0), 3)
        cv2.imwrite(str(overlay_path), overlay)
        payload = {
            "success": True,
            "scene_id": str(frame_records[selected_frame_index]["scene_id"]),
            "matched_item_id": item.item_id,
            "display_name": item.display_name,
            "grasp_prompt": item.grasp_prompt,
            "kind": item.kind,
            "confidence": float(selected.confidence),
            "bbox_xywh": list(selected.bbox_xywh),
            "method": selected.method,
            "search_roi_xywh": list(search_roi or (0, 0, 0, 0)),
            "consensus": consensus,
            "frames": frame_records,
            "color_path": str(color_path),
            "overlay_path": str(overlay_path),
            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
        }
        self._publish_status(
            f"target_card_identified: run_id={run_id} item={item.item_id} "
            f"confidence={float(selected.confidence):.3f}"
        )
        return item, payload

    def _probe_target_card_preview(self, *, after_sequence: int, wait_s: float):
        """Return catalog-photo detections from one fresh live preview frame."""
        preview = self._wait_for_search_preview(
            after_sequence=after_sequence,
            wait_s=wait_s,
        )
        if preview is None:
            return None
        image, sequence = preview
        roi_values = tuple(
            float(value)
            for value in list(self.get_parameter("target_card_search_roi_norm").value or [])
        )
        detections, _roi = self._target_card_matcher().match_all(
            image,
            roi_norm=roi_values,
            threshold=float(self.get_parameter("target_card_match_threshold").value),
            marker_detection_enabled=False,
        )
        return tuple(detections), sequence

    def _identify_target_card(self, *, run_id: str) -> tuple[object, dict[str, object]]:
        """Identify the card, advancing Scout slowly if it is initially out of view."""
        self._last_card_search_travel_m = 0.0
        try:
            return self._identify_target_card_once(run_id=run_id)
        except RuntimeError as initial_error:
            if not bool(self.get_parameter("target_card_base_search_enabled").value):
                raise
            failures = [str(initial_error)]

        step_m = max(0.01, float(self.get_parameter("target_card_base_search_step_m").value))
        max_travel_m = max(0.0, float(self.get_parameter("target_card_base_search_max_travel_m").value))
        speed_mps = max(0.01, float(self.get_parameter("target_card_base_search_speed_mps").value))
        timeout_s = max(3.0, float(self.get_parameter("target_card_base_search_timeout_s").value))
        traveled_m = 0.0
        commanded_m = 0.0
        movements: list[dict[str, object]] = []
        self._publish_status(
            f"target_card_not_visible: run_id={run_id}; "
            f"slow_forward_search max={max_travel_m:.2f}m speed={speed_mps:.2f}m/s"
        )

        while commanded_m + 1e-6 < max_travel_m:
            segment_m = min(step_m, max_travel_m - commanded_m)
            trigger = None
            if bool(self.get_parameter("continuous_search_enabled").value):
                movement, trigger = self._move_base_for_scan_until_preview_trigger(
                    segment_m,
                    run_id=run_id,
                    preview_probe=self._probe_target_card_preview,
                    should_stop=lambda detections: bool(detections),
                    timeout_s=timeout_s,
                    speed_mps=speed_mps,
                )
            else:
                movement = self._move_base_for_scan(
                    segment_m, timeout_s=timeout_s, speed_mps=speed_mps
                )
            movements.append(dict(movement))
            traveled_m += max(0.0, float(movement.get("traveled_m", segment_m)))
            # A visual early-stop can legitimately report zero odometry travel.
            # Count the bounded command budget separately so an ambiguous card
            # cannot cause an endless stop/retry loop at one physical position.
            commanded_m += segment_m
            time.sleep(max(0.0, float(self.get_parameter("base_multiview_settle_s").value)))
            try:
                item, payload = self._identify_target_card_once(run_id=run_id)
            except RuntimeError as error:
                failures.append(str(error))
                continue
            self._last_card_search_travel_m = traveled_m
            payload["base_target_card_search"] = {
                "enabled": True,
                "max_travel_m": max_travel_m,
                "speed_mps": speed_mps,
                "traveled_m": traveled_m,
                "commanded_search_m": commanded_m,
                "stopped_by_preview": bool(
                    trigger is not None
                    or any(bool(record.get("stopped_by_continuous_preview")) for record in movements)
                ),
                "moves": movements,
            }
            return item, payload

        self._last_card_search_travel_m = traveled_m
        raise RuntimeError(
            "target card not recognized after low-speed forward search "
            f"({traveled_m:.2f}m/{max_travel_m:.2f}m): {failures[-1]}"
        )

    def _recognition_max_retries(self, parameter_name: str) -> int:
        return max(0, int(self.get_parameter(parameter_name).value))

    @staticmethod
    def _norm_point_in_roi(
        u_norm: float,
        v_norm: float,
        roi: list[float] | tuple[float, float, float, float] | None,
    ) -> bool:
        if roi is None or len(roi) != 4:
            return False
        x0, y0, x1, y1 = (float(value) for value in roi)
        return x0 <= u_norm <= x1 and y0 <= v_norm <= y1

    def _clear_grasp_card_exclusion(self) -> None:
        self._grasp_exclude_roi_norm = None
        self._grasp_search_roi_norm = None

    def _remember_target_card_exclusion(self, card_payload: dict[str, object] | None) -> None:
        """Ignore only the printed photograph, not the real object window."""
        if not card_payload or not bool(card_payload.get("success")):
            self._clear_grasp_card_exclusion()
            return
        bbox = [int(value) for value in list(card_payload.get("bbox_xywh") or [])]
        image_width = int(card_payload.get("image_width") or 0)
        image_height = int(card_payload.get("image_height") or 0)
        if len(bbox) != 4 or image_width <= 1 or image_height <= 1:
            self._clear_grasp_card_exclusion()
            return
        pad = 0.04
        x, y, width, height = bbox
        self._grasp_exclude_roi_norm = [
            max(0.0, (x / image_width) - pad),
            max(0.0, (y / image_height) - pad),
            min(1.0, ((x + width) / image_width) + pad),
            min(1.0, ((y + height) / image_height) + pad),
        ]
        # Search the whole frame except the printed card. A bottle's
        # calibrated center sits near v=0.485; clipping below the card
        # search window previously hid the real object.
        self._grasp_search_roi_norm = None

    def _detect_target_2d_options_json(self) -> str:
        options: dict[str, object] = {}
        search_roi = getattr(self, "_grasp_search_roi_norm", None)
        exclude_roi = getattr(self, "_grasp_exclude_roi_norm", None)
        if search_roi and len(search_roi) == 4:
            options["search_roi_norm"] = list(search_roi)
        if exclude_roi and len(exclude_roi) == 4:
            options["exclude_roi_norm"] = list(exclude_roi)
        return json_dumps(options) if options else ""

    def _scan_reverse_speed_mps(self) -> float:
        return max(0.02, float(self.get_parameter("base_multiview_speed_mps").value))

    def _scan_reverse_timeout_s(self, distance_m: float, *, speed_mps: float) -> float:
        # Card-search 0.03 m/s / 18 s only covers ~0.5 m and cannot undo a
        # 1.5 m item scan. Size the timeout from the remaining distance.
        min_timeout = max(
            8.0,
            float(self.get_parameter("base_multiview_move_timeout_s").value),
        )
        return max(min_timeout, abs(float(distance_m)) / max(speed_mps, 0.02) + 8.0)

    def _reverse_scan_travel(self, reverse_m: float) -> dict[str, object]:
        """Drive the chassis back to the scan origin, retrying leftover distance."""
        requested_m = abs(float(reverse_m))
        remaining_m = requested_m
        traveled_m = 0.0
        if remaining_m <= 0.02:
            return {
                "requested_m": requested_m,
                "traveled_m": 0.0,
                "remaining_m": remaining_m,
                "complete": True,
                "moves": [],
            }
        speed_mps = self._scan_reverse_speed_mps()
        moves: list[dict[str, object]] = []
        failures = 0
        while remaining_m > 0.02 and failures < 3:
            timeout_s = self._scan_reverse_timeout_s(remaining_m, speed_mps=speed_mps)
            try:
                movement = self._move_base_for_scan(
                    -remaining_m,
                    timeout_s=timeout_s,
                    speed_mps=speed_mps,
                )
            except Exception as exc:
                failures += 1
                self.get_logger().warning(
                    f"scan reverse move failed "
                    f"({traveled_m:.3f}/{requested_m:.3f}m): {exc}"
                )
                continue
            step_m = abs(float(movement.get("traveled_m") or 0.0))
            moves.append(dict(movement))
            if step_m < 0.02:
                failures += 1
                continue
            traveled_m += step_m
            remaining_m = max(0.0, requested_m - traveled_m)
            failures = 0
        complete = remaining_m <= 0.05
        if not complete:
            self.get_logger().error(
                "failed to return to scan start: "
                f"reversed {traveled_m:.3f}m of {requested_m:.3f}m"
            )
        return {
            "requested_m": requested_m,
            "traveled_m": traveled_m,
            "remaining_m": remaining_m,
            "complete": complete,
            "speed_mps": speed_mps,
            "moves": moves,
        }

    def _return_to_observation_for_retry(
        self,
        *,
        run_id: str,
        reverse_m: float,
        reason: str,
    ) -> None:
        """Reverse any search travel, then put the arm back at the observation pose."""
        self._publish_status(
            f"recognition_retry_return: run_id={run_id} reason={reason} "
            f"reverse_m={float(reverse_m):.3f}"
        )
        reverse_payload = self._reverse_scan_travel(reverse_m)
        self._publish_status(
            f"recognition_retry_reversed: run_id={run_id} "
            f"traveled={float(reverse_payload['traveled_m']):.3f}m/"
            f"{float(reverse_payload['requested_m']):.3f}m "
            f"complete={bool(reverse_payload['complete'])}"
        )
        observe_pose = self._observe_pose_values()
        self._execute_named_pose(
            name="observation",
            position_m=(
                observe_pose[0] / 1000.0,
                observe_pose[1] / 1000.0,
                observe_pose[2] / 1000.0,
            ),
            rpy_deg=(observe_pose[3], observe_pose[4], observe_pose[5]),
            speed_percent=float(self.get_parameter("observation_speed").value),
            open_gripper_first=True,
            timeout_s=45.0,
        )

    def _identify_target_card_with_retries(
        self, *, run_id: str
    ) -> tuple[object, dict[str, object]]:
        max_retries = self._recognition_max_retries("target_card_max_retries")
        attempts: list[dict[str, object]] = []
        for attempt in range(1 + max_retries):
            if attempt > 0:
                self._return_to_observation_for_retry(
                    run_id=run_id,
                    reverse_m=self._last_card_search_travel_m,
                    reason=f"target_card_retry_{attempt + 1}",
                )
                self._last_card_search_travel_m = 0.0
            try:
                item, payload = self._identify_target_card(run_id=run_id)
                payload = dict(payload)
                payload["attempt"] = attempt + 1
                payload["max_attempts"] = 1 + max_retries
                payload["retry_history"] = attempts
                return item, payload
            except RuntimeError as error:
                attempts.append({"attempt": attempt + 1, "error": str(error)})
                self.get_logger().warning(
                    f"target card attempt {attempt + 1}/{1 + max_retries} failed: {error}"
                )
        return None, {
            "status": "skipped_no_target_card",
            "attempt": 1 + max_retries,
            "max_attempts": 1 + max_retries,
            "retry_history": attempts,
        }

    def _set_executor_strategy_for_item(self, item) -> str:
        strategy = "safe_top_down" if str(item.kind) == "block" else "center_horizontal"
        request = SetParameters.Request()
        request.parameters = [
            Parameter("execution_strategy", value=strategy).to_parameter_msg()
        ]
        response = self._call_client(
            self._executor_set_parameters_client,
            request,
            timeout_s=10.0,
        )
        results = list(response.results)
        if len(results) != 1 or not bool(results[0].successful):
            reason = str(results[0].reason) if results else "no parameter result"
            raise RuntimeError(f"failed to select {strategy} grasp strategy: {reason}")
        return strategy

    def _wait_for_client(self, client, *, timeout_s: float = 5.0) -> None:
        if client.wait_for_service(timeout_sec=timeout_s):
            return
        raise RuntimeError(f"service unavailable: {client.srv_name}")

    def _call_client(self, client, request, *, timeout_s: float = 30.0):
        self._wait_for_client(client, timeout_s=5.0)
        future = client.call_async(request)
        deadline = time.monotonic() + timeout_s
        while rclpy.ok() and not future.done():
            if self._stop_requested:
                raise RuntimeError("orchestrator stop requested")
            if time.monotonic() >= deadline:
                raise TimeoutError(f"service call timeout: {client.srv_name}")
            time.sleep(0.05)
        if future.cancelled():
            raise RuntimeError(f"service call cancelled: {client.srv_name}")
        exception = future.exception()
        if exception is not None:
            raise exception
        return future.result()

    def _build_runtime(self):
        options = self._options_payload()
        config, _summary = build_runtime_config(options)
        hand_eye_path = str(options.get("hand_eye_config") or config.hand_eye_config_path)
        hand_eye = load_hand_eye_matrix(hand_eye_path)
        online_bias = self._manual_online_bias_from_options(options)
        planner = PureGraspPlanner(config, hand_eye, online_bias)
        return options, config, hand_eye, planner

    @staticmethod
    def _manual_online_bias_from_options(options: dict[str, object]) -> dict[str, object] | None:
        bias_mm = {
            "x_mm": float(options.get("manual_target_bias_x_mm") or 0.0),
            "y_mm": float(options.get("manual_target_bias_y_mm") or 0.0),
            "z_mm": float(options.get("manual_target_bias_z_mm") or 0.0),
        }
        if not any(abs(value) > 1e-9 for value in bias_mm.values()):
            return None
        return {
            "reference_point_type": "dashboard_manual_base_bias",
            "bias_mm": bias_mm,
        }

    def _observe_pose_values(self) -> list[float]:
        values = list(self.get_parameter("observe_pose").value or [30.0, 0.0, 400.0, 0.0, 120.0, 0.0])
        if len(values) != 6:
            raise RuntimeError("observe_pose parameter must contain 6 values")
        return [float(value) for value in values]

    def _placement_observe_pose_values(self) -> list[float]:
        values = list(self.get_parameter("placement_observe_pose").value or [])
        if len(values) != 6 or not any(abs(float(value)) > 1e-9 for value in values):
            raise RuntimeError(
                "placement_observe_pose is not calibrated; configure 6 mm/deg values "
                "before enabling place_after_grasp"
            )
        return [float(value) for value in values]

    @staticmethod
    def _pose6d_msg_to_dataclass(pose_msg):
        return type(
            "Pose",
            (),
            {
                "x_mm": float(pose_msg.x_mm),
                "y_mm": float(pose_msg.y_mm),
                "z_mm": float(pose_msg.z_mm),
                "roll_deg": float(pose_msg.roll_deg),
                "pitch_deg": float(pose_msg.pitch_deg),
                "yaw_deg": float(pose_msg.yaw_deg),
            },
        )()

    def _execute_named_pose(
        self,
        *,
        name: str,
        position_m: tuple[float, float, float],
        rpy_deg: tuple[float, float, float],
        speed_percent: float,
        open_gripper_first: bool,
        timeout_s: float = 45.0,
    ):
        request = ExecuteNamedPose.Request()
        request.name = name
        request.pose = pose6d_from_position_m_rpy_deg(position_m, rpy_deg)
        request.speed_percent = float(speed_percent)
        request.open_gripper_first = bool(open_gripper_first)
        response = self._call_client(self._named_pose_client, request, timeout_s=timeout_s)
        if not response.success:
            raise RuntimeError(response.message)
        return response

    def _read_robot_state_snapshot(self, *, hand_eye: np.ndarray):
        robot_state = self._call_client(self._get_state_client, GetRobotState.Request(), timeout_s=5.0)
        if not robot_state.success:
            raise RuntimeError(robot_state.message)
        current_pose = self._pose6d_msg_to_dataclass(robot_state.current_pose)
        tcp_pose = pose6d_from_position_m_rpy_deg(
            (
                current_pose.x_mm / 1000.0,
                current_pose.y_mm / 1000.0,
                current_pose.z_mm / 1000.0,
            ),
            (
                current_pose.roll_deg,
                current_pose.pitch_deg,
                current_pose.yaw_deg,
            ),
        )
        base_to_camera = base_to_camera_from_tcp_and_hand_eye(current_pose, hand_eye)
        return {
            "robot_state": robot_state,
            "current_pose": current_pose,
            "tcp_pose": tcp_pose,
            "base_to_camera": base_to_camera,
        }

    def _capture_scene_once(
        self,
        *,
        run_id: str,
        phase_label: str,
        depth_fusion_frames: int | None = None,
    ):
        self._publish_status(f"capturing_scene: run_id={run_id} phase={phase_label}")
        capture_req = CaptureScene.Request()
        capture_req.run_id = run_id
        capture_req.depth_fusion_frames = max(
            1,
            int(
                self.get_parameter("depth_fusion_frames").value
                if depth_fusion_frames is None
                else depth_fusion_frames
            ),
        )
        capture_req.pointcloud_filter_mode = str(self.get_parameter("pointcloud_filter_mode").value)
        capture_req.pointcloud_backend = str(self.get_parameter("pointcloud_backend").value)
        capture_response = self._call_client(self._capture_client, capture_req, timeout_s=30.0)
        if not capture_response.success:
            raise RuntimeError(capture_response.message)
        return capture_response

    def _match_box_label_once(
        self,
        *,
        run_id: str,
        item_id: str,
        capture_response,
        base_to_camera: np.ndarray,
        require_complete: bool = True,
    ) -> dict[str, object]:
        self._publish_status(f"matching_box_label: run_id={run_id} item={item_id}")
        request = MatchItemLabel.Request()
        request.run_id = run_id
        request.scene_id = str(capture_response.scene_id)
        request.expected_item_id = item_id
        request.color_image = capture_response.color_image
        request.depth_image = capture_response.depth_image
        request.camera_info = capture_response.camera_info
        request.base_to_camera = matrix_to_transform_msg(
            base_to_camera,
            parent_frame="base_link",
            child_frame=str(capture_response.camera_frame or "camera_color_optical_frame"),
            stamp=self.get_clock().now().to_msg(),
        )
        request.options_json = json_dumps(
            {
                "label_search_roi_norm": [
                    float(value)
                    for value in list(self.get_parameter("label_search_roi_norm").value or [])
                ],
                "label_match_threshold": float(self.get_parameter("label_match_threshold").value),
                "label_marker_detection_enabled": bool(
                    self.get_parameter("label_marker_detection_enabled").value
                ),
                "table_z_m": float(self.get_parameter("table_z_m").value),
                # Target-alignment scans only need the photograph label and
                # its pixel center.  Avoid expensive transparent-box depth
                # localization until an actual placement plan needs it.
                "localize_box_row": bool(require_complete),
            }
        )
        response = self._call_client(self._match_label_client, request, timeout_s=30.0)
        payload = {
            "success": bool(response.success),
            "message": str(response.message),
            "expected_item_id": item_id,
            "matched_item_id": str(response.matched_item_id or ""),
            "confidence": float(response.confidence),
            "slot_index": int(response.slot_index),
            "detected_label_count": int(response.detected_label_count),
            "has_box_center": bool(response.has_box_center),
            "box_center_base_m": [
                float(response.box_center_base_m.x),
                float(response.box_center_base_m.y),
                float(response.box_center_base_m.z),
            ],
            "bbox_xywh": [
                int(response.bbox_x),
                int(response.bbox_y),
                int(response.bbox_width),
                int(response.bbox_height),
            ],
            "diagnostics": (
                json.loads(response.diagnostics_json)
                if str(response.diagnostics_json or "").strip()
                else {}
            ),
        }
        complete = not (
            not response.success
            or str(response.matched_item_id) != item_id
            or int(response.slot_index) < 0
            or int(response.detected_label_count) != 6
            or not bool(response.has_box_center)
        )
        payload["complete"] = bool(complete)
        if require_complete and not complete:
            raise RuntimeError(
                f"box label verification failed for {item_id}: {response.message}"
            )
        return payload

    def _match_box_label_from_preview(
        self,
        *,
        run_id: str,
        item_id: str,
        after_sequence: int = -1,
        wait_s: float | None = None,
    ) -> tuple[dict[str, object], np.ndarray, int] | None:
        """Match a target box label from RGB preview without depth localization."""
        preview = self._wait_for_search_preview(
            after_sequence=after_sequence,
            wait_s=wait_s,
        )
        if preview is None:
            return None
        color_bgr, preview_sequence = preview
        request = MatchItemLabel.Request()
        request.run_id = run_id
        request.scene_id = f"continuous-preview-{preview_sequence}"
        request.expected_item_id = item_id
        request.color_image = color_image_to_msg(
            color_bgr,
            frame_id="camera_color_optical_frame",
            stamp=self.get_clock().now().to_msg(),
        )
        request.options_json = json_dumps(
            {
                "label_search_roi_norm": [
                    float(value)
                    for value in list(
                        self.get_parameter("label_search_roi_norm").value or []
                    )
                ],
                "label_match_threshold": float(
                    self.get_parameter("label_match_threshold").value
                ),
                "label_marker_detection_enabled": bool(
                    self.get_parameter("label_marker_detection_enabled").value
                ),
                "localize_box_row": False,
            }
        )
        response = self._call_client(
            self._match_label_client,
            request,
            timeout_s=30.0,
        )
        payload = {
            "success": bool(response.success),
            "message": str(response.message),
            "expected_item_id": item_id,
            "matched_item_id": str(response.matched_item_id or ""),
            "confidence": float(response.confidence),
            "slot_index": int(response.slot_index),
            "detected_label_count": int(response.detected_label_count),
            "has_box_center": False,
            "box_center_base_m": [0.0, 0.0, 0.0],
            "bbox_xywh": [
                int(response.bbox_x),
                int(response.bbox_y),
                int(response.bbox_width),
                int(response.bbox_height),
            ],
            "diagnostics": (
                json.loads(response.diagnostics_json)
                if str(response.diagnostics_json or "").strip()
                else {}
            ),
            "complete": False,
            "source": "continuous_preview",
            "color_width": int(color_bgr.shape[1]),
            "color_height": int(color_bgr.shape[0]),
        }
        return payload, color_bgr, int(preview_sequence)

    @staticmethod
    def _pose6d_from_mm_deg(values: tuple[float, float, float, float, float, float]):
        return pose6d_from_position_m_rpy_deg(
            (values[0] / 1000.0, values[1] / 1000.0, values[2] / 1000.0),
            (values[3], values[4], values[5]),
        )

    def _build_place_plan_message(
        self,
        *,
        item_id: str,
        label_confidence: float,
        slot_index: int,
        box_center_base_m: tuple[float, float, float],
    ) -> PlacePlan:
        catalog = self._item_catalog()
        poses = catalog.build_place_poses_mm_deg(
            item_id,
            slot_index,
            slot_center_mm=tuple(float(value) * 1000.0 for value in box_center_base_m),
        )
        message = PlacePlan()
        message.item_id = item_id
        message.slot_index = int(slot_index)
        message.approach_pose = self._pose6d_from_mm_deg(poses["approach"])
        message.release_pose = self._pose6d_from_mm_deg(poses["release"])
        message.retreat_pose = self._pose6d_from_mm_deg(poses["retreat"])
        message.box_outer_size_m = [float(value) for value in catalog.box.outer_size_m]
        message.label_verified = True
        message.label_confidence = float(label_confidence)
        return message

    def _build_base_aligned_place_plan_message(
        self,
        *,
        item_id: str,
        label_confidence: float,
        slot_index: int,
    ) -> PlacePlan:
        catalog = self._item_catalog()
        poses = catalog.build_base_aligned_place_poses_mm_deg(item_id)
        message = PlacePlan()
        message.item_id = item_id
        message.slot_index = int(slot_index)
        message.approach_pose = self._pose6d_from_mm_deg(poses["approach"])
        message.release_pose = self._pose6d_from_mm_deg(poses["release"])
        message.retreat_pose = self._pose6d_from_mm_deg(poses["retreat"])
        message.box_outer_size_m = [
            float(value) for value in catalog.box.outer_size_m
        ]
        message.label_verified = True
        message.label_confidence = float(label_confidence)
        return message

    def _cached_multiview_label(
        self,
        item_id: str,
    ) -> dict[str, object] | None:
        if not bool(self.get_parameter("use_cached_multiview_box_map").value):
            return None
        path = self._placement_scan_viz_dir() / "latest.json"
        if not path.is_file():
            raise RuntimeError("multi-view box map is required but no scan exists")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if (
            not bool(payload.get("success"))
            or str(payload.get("scan_mode") or "") != "base_multiview"
            or not bool(payload.get("base_returned_to_start"))
        ):
            raise RuntimeError("latest placement scan is not a valid multi-view box map")
        created_at = float(payload.get("created_at_unix_s") or 0.0)
        max_age = float(self.get_parameter("cached_box_map_max_age_s").value)
        if created_at <= 0.0 or time.time() - created_at > max_age:
            raise RuntimeError("cached multi-view box map is stale; scan again")

        origin = dict(payload.get("base_odom_origin") or {})
        current = self._base_odom_snapshot()
        position_error = math.hypot(
            current[0] - float(origin["x_m"]),
            current[1] - float(origin["y_m"]),
        )
        yaw_error = abs(
            math.degrees(
                math.atan2(
                    math.sin(current[2] - float(origin["yaw_rad"])),
                    math.cos(current[2] - float(origin["yaw_rad"])),
                )
            )
        )
        position_limit = float(
            self.get_parameter("cached_box_map_position_tolerance_m").value
        )
        yaw_limit = float(
            self.get_parameter("cached_box_map_yaw_tolerance_deg").value
        )
        if position_error > position_limit or yaw_error > yaw_limit:
            raise RuntimeError(
                "Scout moved since multi-view scan: "
                f"position_error={position_error:.3f}m "
                f"yaw_error={yaw_error:.2f}deg; scan again"
            )

        fused = dict(payload.get("fused_map") or {})
        centers = dict(fused.get("item_to_box_center_base_m") or {})
        slots = dict(fused.get("item_to_slot_index") or {})
        confidences = dict(fused.get("item_to_confidence") or {})
        if item_id not in centers or item_id not in slots:
            raise RuntimeError(f"cached box map does not contain {item_id}")
        return {
            "success": True,
            "complete": True,
            "message": "using odometry-validated multi-view box map",
            "expected_item_id": item_id,
            "matched_item_id": item_id,
            "confidence": float(confidences.get(item_id) or 0.0),
            "slot_index": int(slots[item_id]),
            "detected_label_count": 6,
            "has_box_center": True,
            "box_center_base_m": [
                float(value) for value in list(centers[item_id])
            ],
            "diagnostics": fused,
            "source_scan_id": str(payload.get("scan_id") or ""),
        }

    def _run_placement_stage(
        self,
        *,
        run_id: str,
        item_id: str,
        move_home_after: bool,
        hand_eye: np.ndarray,
        advance_base_during_observation: bool = False,
    ) -> dict[str, object]:
        observe = self._placement_observe_pose_values()
        observe_response, post_grasp_movement = (
            self._move_to_placement_observation(
                run_id=run_id,
                observe=observe,
                advance_base=advance_base_during_observation,
                timeout_s=30.0,
                target_item_id=item_id,
            )
        )
        observe_pose = self._pose6d_msg_to_dataclass(observe_response.actual_pose)
        base_to_camera = base_to_camera_from_tcp_and_hand_eye(observe_pose, hand_eye)
        label = self._cached_multiview_label(item_id)
        capture_response = None
        if label is None:
            capture_response = self._capture_scene_once(
                run_id=run_id,
                phase_label="placement_label",
            )
            label = self._match_box_label_once(
                run_id=run_id,
                item_id=item_id,
                capture_response=capture_response,
                base_to_camera=base_to_camera,
            )
        plan_msg = self._build_place_plan_message(
            item_id=item_id,
            label_confidence=float(label["confidence"]),
            slot_index=int(label["slot_index"]),
            box_center_base_m=tuple(float(value) for value in label["box_center_base_m"]),
        )

        validation_request = ExecutePlacePlan.Request()
        validation_request.run_id = run_id
        validation_request.execute = False
        validation_request.move_home_after = bool(move_home_after)
        validation_request.plan = plan_msg
        validation_response = self._call_client(
            self._execute_place_client,
            validation_request,
            timeout_s=45.0,
        )
        if not validation_response.success:
            raise RuntimeError(f"place plan validation failed: {validation_response.message}")
        validation = json.loads(validation_response.execution_json)

        self._publish_status(f"placing_object: run_id={run_id} item={item_id}")
        execute_request = ExecutePlacePlan.Request()
        execute_request.run_id = run_id
        execute_request.execute = True
        execute_request.move_home_after = bool(move_home_after)
        execute_request.plan = plan_msg
        execute_response = self._call_client(
            self._execute_place_client,
            execute_request,
            timeout_s=180.0,
        )
        if not execute_response.success:
            raise RuntimeError(f"place execution failed: {execute_response.message}")
        placement_payload = {
            "item_id": item_id,
            "slot_index": int(label["slot_index"]),
            "placement_observe_actual_pose": self._pose_debug_dict(
                self._pose6d_msg_to_dataclass(observe_response.actual_pose)
            ),
            "capture": (
                self._capture_debug_dict(capture_response)
                if capture_response is not None
                else None
            ),
            "label_match": label,
            "validation": validation,
            "execution": json.loads(execute_response.execution_json),
        }
        if post_grasp_movement is not None:
            placement_payload["post_grasp_base_advance"] = post_grasp_movement
        placement_payload["post_place_base_advance"] = (
            self._advance_base_after_place(run_id=run_id, item_id=item_id)
        )
        return placement_payload

    def _post_grasp_base_advance(self, *, run_id: str, target_item_id: str) -> dict[str, object]:
        distance_m = float(self.get_parameter("post_grasp_base_advance_m").value)
        speed_mps = float(
            self.get_parameter("post_grasp_base_advance_speed_mps").value
        )
        timeout_s = float(
            self.get_parameter("post_grasp_base_advance_timeout_s").value
        )
        if distance_m <= 0.01:
            raise RuntimeError("post_grasp_base_advance_m must be greater than 0.01")
        self._publish_status(
            f"advancing_base_after_grasp: run_id={run_id} distance={distance_m:.3f}m"
        )
        movement = self._move_base_for_scan(
            distance_m,
            timeout_s=timeout_s,
            speed_mps=speed_mps,
        )
        return {
            **movement,
            "phase": "during_placement_observation_after_grasp_retreat",
            "target_item_id": target_item_id,
        }

    def _advance_base_after_place(self, *, run_id: str, item_id: str) -> dict[str, object]:
        distance_m = float(self.get_parameter("post_place_base_advance_m").value)
        speed_mps = float(
            self.get_parameter("post_place_base_advance_speed_mps").value
        )
        timeout_s = float(
            self.get_parameter("post_place_base_advance_timeout_s").value
        )
        if distance_m <= 0.01:
            raise RuntimeError("post_place_base_advance_m must be greater than 0.01")
        self._publish_status(
            f"advancing_base_after_place: run_id={run_id} distance={distance_m:.3f}m"
        )
        movement = self._move_base_for_scan(
            distance_m,
            timeout_s=timeout_s,
            speed_mps=speed_mps,
        )
        return {
            **movement,
            "phase": "after_place_release_and_retreat",
            "target_item_id": item_id,
        }

    def _return_home_and_advance_base_after_place(
        self,
        *,
        run_id: str,
        item_id: str,
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Fold the arm and advance Scout concurrently after a safe retreat."""
        home_values = [
            float(value)
            for value in list(self.get_parameter("post_place_home_pose").value or [])
        ]
        if len(home_values) != 6:
            raise RuntimeError("post_place_home_pose must contain 6 mm/deg values")

        base_result: dict[str, object] | None = None
        base_errors: list[BaseException] = []

        def base_worker() -> None:
            nonlocal base_result
            try:
                base_result = self._advance_base_after_place(
                    run_id=run_id,
                    item_id=item_id,
                )
            except BaseException as exc:
                base_errors.append(exc)

        self._publish_status(
            f"returning_home_and_advancing_base_after_place: run_id={run_id}"
        )
        base_thread = threading.Thread(
            target=base_worker,
            daemon=True,
            name=f"post-place-base-advance-{run_id}",
        )
        base_thread.start()
        try:
            home_response = self._execute_named_pose(
                name="home_after_place",
                position_m=(
                    home_values[0] / 1000.0,
                    home_values[1] / 1000.0,
                    home_values[2] / 1000.0,
                ),
                rpy_deg=(home_values[3], home_values[4], home_values[5]),
                speed_percent=float(self.get_parameter("home_speed").value),
                open_gripper_first=False,
                timeout_s=60.0,
            )
        except BaseException:
            if base_thread.is_alive():
                self._request_base_scan_stop()
            base_thread.join()
            raise
        base_thread.join()
        if base_errors:
            self.get_logger().warning(
                "post-place base advance failed after a successful release: "
                f"{base_errors[0]}"
            )
            base_result = {
                "success": False,
                "error": str(base_errors[0]),
            }
        if base_result is None:
            base_result = {
                "success": False,
                "error": "post-place base advance returned no result",
            }
        home_result = {
            "success": True,
            "requested_pose_mm_deg": {
                "x_mm": home_values[0],
                "y_mm": home_values[1],
                "z_mm": home_values[2],
                "roll_deg": home_values[3],
                "pitch_deg": home_values[4],
                "yaw_deg": home_values[5],
            },
            "actual_pose_mm_deg": self._pose_debug_dict(
                self._pose6d_msg_to_dataclass(home_response.actual_pose)
            ),
        }
        return home_result, base_result

    def _move_to_placement_observation(
        self,
        *,
        run_id: str,
        observe: tuple[float, float, float, float, float, float],
        advance_base: bool,
        timeout_s: float,
        target_item_id: str = "",
    ):
        base_result: dict[str, object] | None = None
        base_error: list[BaseException] = []

        def advance_base_worker() -> None:
            nonlocal base_result
            try:
                base_result = self._post_grasp_base_advance(
                    run_id=run_id,
                    target_item_id=target_item_id,
                )
            except BaseException as exc:  # Preserve the worker failure for the caller.
                base_error.append(exc)

        base_thread = None
        if advance_base:
            base_thread = threading.Thread(
                target=advance_base_worker,
                daemon=True,
                name=f"post-grasp-base-advance-{run_id}",
            )
            base_thread.start()

        self._publish_status(f"moving_to_placement_observation: run_id={run_id}")
        try:
            observe_response = self._execute_named_pose(
                name="placement_observation",
                position_m=(observe[0] / 1000.0, observe[1] / 1000.0, observe[2] / 1000.0),
                rpy_deg=(observe[3], observe[4], observe[5]),
                speed_percent=float(self.get_parameter("observation_speed").value),
                open_gripper_first=False,
                timeout_s=timeout_s,
            )
        except BaseException:
            if base_thread is not None and base_thread.is_alive():
                self._request_base_scan_stop()
                base_thread.join()
            raise
        if base_thread is not None:
            base_thread.join()
        if base_error:
            raise RuntimeError(str(base_error[0])) from base_error[0]
        return observe_response, base_result

    def _execute_grasp_and_optional_place(
        self,
        *,
        run_id: str,
        plan,
        move_home_after: bool,
        target_item_id: str,
        hand_eye: np.ndarray,
        advance_base_after_grasp: bool = False,
    ) -> dict[str, object]:
        place_enabled = bool(self.get_parameter("place_after_grasp").value)
        # A dashboard request may still carry the previous item's UI strategy.
        # Reassert the catalog-derived strategy immediately before execution
        # (including confirmed plans), so a bottle can never be executed as a
        # top-down block grasp.
        if target_item_id:
            execution_item = self._item_catalog().resolve(target_item_id)
            if execution_item is None:
                raise RuntimeError(f"unknown execution target_item_id: {target_item_id}")
            selected_strategy = self._set_executor_strategy_for_item(execution_item)
            self.get_logger().info(
                f"execution strategy locked: item={target_item_id} strategy={selected_strategy}"
            )
        execute_req = ExecuteGraspPlan.Request()
        execute_req.run_id = run_id
        execute_req.execute = True
        execute_req.move_home_after = bool(move_home_after and not place_enabled)
        execute_req.plan = grasp_plan_to_msg(plan)
        execute_response = self._call_client(self._execute_plan_client, execute_req, timeout_s=180.0)
        if not execute_response.success:
            raise RuntimeError(execute_response.message)
        grasp_payload = json.loads(execute_response.execution_json)
        if not place_enabled:
            move_to_observation = bool(
                self.get_parameter(
                    "move_to_placement_observation_after_grasp"
                ).value
            )
            if move_to_observation and not bool(move_home_after):
                observe = self._placement_observe_pose_values()
                observe_response, movement = self._move_to_placement_observation(
                    run_id=run_id,
                    observe=observe,
                    advance_base=advance_base_after_grasp,
                    timeout_s=45.0,
                    target_item_id=target_item_id,
                )
                if movement is not None:
                    grasp_payload["post_grasp_base_advance"] = movement
                grasp_payload["placement_observation_after_grasp"] = {
                    "requested_pose_mm_deg": {
                        "x_mm": observe[0],
                        "y_mm": observe[1],
                        "z_mm": observe[2],
                        "roll_deg": observe[3],
                        "pitch_deg": observe[4],
                        "yaw_deg": observe[5],
                    },
                    "actual_pose_mm_deg": self._pose_debug_dict(
                        self._pose6d_msg_to_dataclass(
                            observe_response.actual_pose
                        )
                    ),
                    "gripper_opened": False,
                }
            elif advance_base_after_grasp:
                grasp_payload["post_grasp_base_advance"] = (
                    self._post_grasp_base_advance(
                        run_id=run_id,
                        target_item_id=target_item_id,
                    )
                )
            return grasp_payload
        if not target_item_id:
            raise RuntimeError("place_after_grasp requires a resolved target_item_id")
        placement_payload = self._run_placement_stage(
            run_id=run_id,
            item_id=target_item_id,
            move_home_after=move_home_after,
            hand_eye=hand_eye,
            advance_base_during_observation=advance_base_after_grasp,
        )
        return {
            "status": "ok",
            "run_id": run_id,
            "target_item_id": target_item_id,
            "grasp": grasp_payload,
            "placement": placement_payload,
            "release_performed": True,
            "move_home_after": bool(move_home_after),
        }

    def _analyze_scene_once(
        self,
        *,
        run_id: str,
        phase_label: str,
        prompt: str,
        options: dict[str, object],
        capture_response,
        tcp_pose,
        base_to_camera: np.ndarray,
    ):
        self._publish_status(f"analyzing_scene: run_id={run_id} phase={phase_label}")
        analyze_req = AnalyzeScene.Request()
        analyze_req.run_id = run_id
        analyze_req.scene_id = capture_response.scene_id
        analyze_req.prompt = prompt
        analyze_req.camera_frame = capture_response.camera_frame
        analyze_req.color_image = capture_response.color_image
        analyze_req.depth_image = capture_response.depth_image
        analyze_req.camera_info = capture_response.camera_info
        analyze_req.tcp_pose = tcp_pose
        analyze_req.base_to_camera = matrix_to_transform_msg(
            base_to_camera,
            parent_frame="base_link",
            child_frame=str(capture_response.camera_frame or "camera_color_optical_frame"),
            stamp=self.get_clock().now().to_msg(),
        )
        analyze_options = dict(options)
        analyze_options["prompt"] = prompt
        analyze_req.options_json = json_dumps(analyze_options)
        analyze_response = self._call_client(self._analyze_client, analyze_req, timeout_s=120.0)
        if not analyze_response.success:
            raise RuntimeError(analyze_response.message)
        return analyze_response

    def _capture_analyze_cycle(
        self,
        *,
        run_id: str,
        prompt: str,
        options: dict[str, object],
        hand_eye: np.ndarray,
        phase_label: str,
    ) -> dict[str, object]:
        self._publish_status(f"reading_robot_state: run_id={run_id} phase={phase_label}")
        state_snapshot = self._read_robot_state_snapshot(hand_eye=hand_eye)
        capture_response = self._capture_scene_once(run_id=run_id, phase_label=phase_label)
        analyze_response = self._analyze_scene_once(
            run_id=run_id,
            phase_label=phase_label,
            prompt=prompt,
            options=options,
            capture_response=capture_response,
            tcp_pose=state_snapshot["tcp_pose"],
            base_to_camera=state_snapshot["base_to_camera"],
        )
        candidate = (
            grasp_candidate_from_msg(analyze_response.selected_candidate)
            if analyze_response.has_selected_candidate
            else None
        )
        candidate_pool = [grasp_candidate_from_msg(item) for item in list(analyze_response.candidate_pool)]
        diagnostics = json.loads(analyze_response.diagnostics_json).get("diagnostics", [])
        return {
            "capture_response": capture_response,
            "analyze_response": analyze_response,
            "current_pose": state_snapshot["current_pose"],
            "robot_state": state_snapshot["robot_state"],
            "base_to_camera": state_snapshot["base_to_camera"],
            "candidate": candidate,
            "candidate_pool": candidate_pool,
            "diagnostics": diagnostics,
            "debug": {
                "phase": phase_label,
                "robot_state": {
                    "success": bool(state_snapshot["robot_state"].success),
                    "message": str(state_snapshot["robot_state"].message),
                    "current_pose": self._pose_debug_dict(state_snapshot["current_pose"]),
                },
                "capture": self._capture_debug_dict(capture_response),
                "analyze": self._analyze_debug_dict(analyze_response),
                "base_to_camera": np.asarray(state_snapshot["base_to_camera"], dtype=np.float64).tolist(),
            },
        }

    def _capture_detect_target_2d(
        self,
        *,
        run_id: str,
        prompt: str,
        hand_eye: np.ndarray,
        phase_label: str,
        depth_fusion_frames: int | None = None,
    ) -> dict[str, object]:
        """Capture one scan view and run only inexpensive 2-D target detection."""
        self._publish_status(
            f"reading_robot_state: run_id={run_id} phase={phase_label}"
        )
        # During base search, use the current preview frame first.  It is
        # intentionally RGB-only: a centered preview is always re-captured
        # with full fused depth before GraspNet is called.
        if depth_fusion_frames is None:
            preview_detection = self._detect_target_2d_from_preview(
                run_id=run_id,
                prompt=prompt,
            )
            if preview_detection is not None:
                response, preview_sequence = preview_detection
                return {
                    "state_snapshot": None,
                    "capture_response": None,
                    "detection_response": response,
                    "preview_sequence": int(preview_sequence),
                }
        state_snapshot = self._read_robot_state_snapshot(hand_eye=hand_eye)
        capture_response = self._capture_scene_once(
            run_id=run_id,
            phase_label=phase_label,
            depth_fusion_frames=int(
                self.get_parameter("search_depth_fusion_frames").value
                if depth_fusion_frames is None
                else depth_fusion_frames
            ),
        )
        request = DetectTarget2D.Request()
        request.run_id = run_id
        request.prompt = prompt
        request.color_image = capture_response.color_image
        if hasattr(request, "options_json"):
            request.options_json = self._detect_target_2d_options_json()
        response = self._call_client(
            self._detect_target_2d_client,
            request,
            timeout_s=30.0,
        )
        if not response.success:
            raise RuntimeError(response.message)
        return {
            "state_snapshot": state_snapshot,
            "capture_response": capture_response,
            "detection_response": response,
        }

    def _analyze_captured_cycle(
        self,
        *,
        run_id: str,
        prompt: str,
        options: dict[str, object],
        phase_label: str,
        captured: dict[str, object],
    ) -> dict[str, object]:
        """Run full GraspNet once on an already captured, centered scan view."""
        state_snapshot = dict(captured["state_snapshot"])
        capture_response = captured["capture_response"]
        analyze_response = self._analyze_scene_once(
            run_id=run_id,
            phase_label=phase_label,
            prompt=prompt,
            options=options,
            capture_response=capture_response,
            tcp_pose=state_snapshot["tcp_pose"],
            base_to_camera=state_snapshot["base_to_camera"],
        )
        candidate = (
            grasp_candidate_from_msg(analyze_response.selected_candidate)
            if analyze_response.has_selected_candidate
            else None
        )
        candidate_pool = [
            grasp_candidate_from_msg(item)
            for item in list(analyze_response.candidate_pool)
        ]
        diagnostics = json.loads(analyze_response.diagnostics_json).get(
            "diagnostics", []
        )
        return {
            "capture_response": capture_response,
            "analyze_response": analyze_response,
            "current_pose": state_snapshot["current_pose"],
            "robot_state": state_snapshot["robot_state"],
            "base_to_camera": state_snapshot["base_to_camera"],
            "candidate": candidate,
            "candidate_pool": candidate_pool,
            "diagnostics": diagnostics,
            "debug": {
                "phase": phase_label,
                "robot_state": {
                    "success": bool(state_snapshot["robot_state"].success),
                    "message": str(state_snapshot["robot_state"].message),
                    "current_pose": self._pose_debug_dict(
                        state_snapshot["current_pose"]
                    ),
                },
                "capture": self._capture_debug_dict(capture_response),
                "analyze": self._analyze_debug_dict(analyze_response),
                "base_to_camera": np.asarray(
                    state_snapshot["base_to_camera"], dtype=np.float64
                ).tolist(),
            },
        }

    @staticmethod
    def _candidate_center_camera_m(candidate) -> tuple[float, float, float] | None:
        if candidate is None:
            return None
        center = candidate.object_center_camera_m or candidate.translation_camera_m
        if center is None:
            return None
        return (float(center[0]), float(center[1]), float(center[2]))

    def _retarget_plan_to_object_center(
        self,
        *,
        plan,
        candidate,
        base_to_camera: np.ndarray,
        prompt: str = "",
    ):
        if not bool(self.get_parameter("use_object_center_contact").value):
            return plan
        if candidate.object_center_camera_m is None or plan.target_contact_point_base_m is None:
            return plan

        selected_camera_xyz = np.asarray(candidate.translation_camera_m, dtype=np.float64).reshape(3)
        raw_center_camera_xyz = np.asarray(candidate.object_center_camera_m, dtype=np.float64).reshape(3)
        if float(raw_center_camera_xyz[2]) <= 1e-6 or float(selected_camera_xyz[2]) <= 1e-6:
            return replace(
                plan,
                within_workspace=False,
                workspace_violations=["object center or grasp depth is invalid"],
            )

        # Transparent bottles often return the table/background depth through the object.
        # Keep the segmented center ray (x/z, y/z), but project it onto the reliable
        # GraspNet contact depth instead of trusting the raw instance mean depth.
        grasp_depth_m = float(selected_camera_xyz[2])
        center_camera_xyz = np.array(
            [
                float(raw_center_camera_xyz[0] / raw_center_camera_xyz[2]) * grasp_depth_m,
                float(raw_center_camera_xyz[1] / raw_center_camera_xyz[2]) * grasp_depth_m,
                grasp_depth_m,
            ],
            dtype=np.float64,
        )
        center_offset_m = float(np.linalg.norm(center_camera_xyz - selected_camera_xyz))
        max_center_offset_m = float(self.get_parameter("object_center_contact_max_offset_m").value)
        if center_offset_m > max_center_offset_m:
            return replace(
                plan,
                within_workspace=False,
                workspace_violations=[
                    f"object center offset {center_offset_m:.3f}m exceeds "
                    f"object_center_contact_max_offset_m={max_center_offset_m:.3f}m"
                ],
            )

        transform = np.asarray(base_to_camera, dtype=np.float64).reshape(4, 4)
        selected_camera = np.array([*selected_camera_xyz, 1.0], dtype=np.float64)
        center_camera = np.array([*center_camera_xyz, 1.0], dtype=np.float64)
        selected_base = (transform @ selected_camera)[:3]
        center_base = (transform @ center_camera)[:3]
        original_contact = np.asarray(plan.target_contact_point_base_m, dtype=np.float64).reshape(3)

        # Preserve planner-side Z compensation and manual base-frame bias while replacing
        # only the GraspNet contact sample with the segmented object's geometric center.
        planner_contact_offset = original_contact - selected_base
        center_contact = center_base + planner_contact_offset
        normalized_prompt = str(prompt or "").strip().lower()
        if any(token in normalized_prompt for token in ("block", "物块", "方块", "立方体")):
            table_z_m = float(self.get_parameter("table_z_m").value)
            center_height_m = float(self.get_parameter("color_block_center_height_m").value)
            manual_bias_z_m = float(self.get_parameter("manual_target_bias_z_mm").value) / 1000.0
            # The competition blocks are fixed 60 mm cubes. Their RGB-D depth
            # can drift near the image edge, so use the known table-relative
            # grasp center instead of lowering the tool based on that drift.
            center_contact[2] = table_z_m + center_height_m + manual_bias_z_m
        minimum_contact_z_m = float(self.get_parameter("table_z_m").value) + float(
            self.get_parameter("min_gripper_table_clearance_m").value
        )
        violations: list[str] = []
        if float(center_contact[2]) < minimum_contact_z_m:
            violations.append(
                f"object-center contact z={float(center_contact[2]):.3f} below "
                f"table safety height {minimum_contact_z_m:.3f}"
            )

        return replace(
            plan,
            target_contact_point_base_m=tuple(float(value) for value in center_contact),
            within_workspace=not violations,
            workspace_violations=violations,
        )

    @staticmethod
    def _project_camera_point_to_uv(
        point_camera_m: tuple[float, float, float],
        intrinsics,
    ) -> tuple[int, int] | None:
        x_m, y_m, z_m = point_camera_m
        if z_m <= 1e-6:
            return None
        u = int(round((x_m * float(intrinsics.fx) / z_m) + float(intrinsics.ppx)))
        v = int(round((y_m * float(intrinsics.fy) / z_m) + float(intrinsics.ppy)))
        return (u, v)

    @staticmethod
    def _plan_centering_move(
        *,
        current_pose,
        base_to_camera: np.ndarray,
        intrinsics,
        config,
        target_uv: tuple[int, int],
        target_depth_m: float,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], dict[str, float]]:
        current_pos_m = np.array(
            [current_pose.x_mm / 1000.0, current_pose.y_mm / 1000.0, current_pose.z_mm / 1000.0],
            dtype=np.float64,
        )
        current_rpy = (current_pose.roll_deg, current_pose.pitch_deg, current_pose.yaw_deg)

        image_center_u = float(intrinsics.width) / 2.0
        image_center_v = float(intrinsics.height) / 2.0
        error_u = float(target_uv[0] - image_center_u)
        error_v = float(target_uv[1] - image_center_v)

        delta_camera = np.array(
            [
                (error_u / float(intrinsics.fx)) * float(target_depth_m),
                (error_v / float(intrinsics.fy)) * float(target_depth_m),
                0.0,
            ],
            dtype=np.float64,
        )
        step_norm = float(np.linalg.norm(delta_camera[:2]))
        if step_norm > float(config.center_max_step_m) and step_norm > 1e-9:
            delta_camera *= float(config.center_max_step_m) / step_norm

        delta_base = np.asarray(base_to_camera, dtype=np.float64)[:3, :3] @ delta_camera
        target_pos = current_pos_m + delta_base
        return (
            (float(target_pos[0]), float(target_pos[1]), float(target_pos[2])),
            current_rpy,
            {
                "error_u_px": error_u,
                "error_v_px": error_v,
                "delta_base_x_m": float(delta_base[0]),
                "delta_base_y_m": float(delta_base[1]),
            },
        )

    def _run_precenter_loop(
        self,
        *,
        run_id: str,
        prompt: str,
        options: dict[str, object],
        config,
        hand_eye: np.ndarray,
        planner: PureGraspPlanner,
    ) -> tuple[dict[str, object], list[str], list[dict[str, object]]]:
        logs: list[str] = []
        cycle_records: list[dict[str, object]] = []
        max_iterations = max(1, int(config.center_max_iterations))
        for iteration in range(max_iterations):
            phase_label = f"precenter_{iteration + 1}"
            cycle = self._capture_analyze_cycle(
                run_id=run_id,
                prompt=prompt,
                options=options,
                hand_eye=hand_eye,
                phase_label=phase_label,
            )
            cycle_records.append(dict(cycle["debug"]))
            candidate = cycle["candidate"]
            center_camera_m = self._candidate_center_camera_m(candidate)
            if candidate is None or center_camera_m is None:
                logs.append(f"centering iter {iteration + 1}: no valid candidate to center on")
                return cycle, logs, cycle_records

            intrinsics = camera_info_to_intrinsics(cycle["capture_response"].camera_info)
            center_uv = self._project_camera_point_to_uv(center_camera_m, intrinsics)
            if center_uv is None:
                logs.append(f"centering iter {iteration + 1}: invalid target depth for centering")
                return cycle, logs, cycle_records

            image_center = (int(intrinsics.width // 2), int(intrinsics.height // 2))
            pixel_error = (center_uv[0] - image_center[0], center_uv[1] - image_center[1])
            logs.append(
                f"centering iter {iteration + 1}: instance={candidate.instance_index} "
                f"pixel_error=({pixel_error[0]:+d}, {pixel_error[1]:+d}) "
                f"cam_center=({center_camera_m[0]:.3f}, {center_camera_m[1]:.3f}, {center_camera_m[2]:.3f})"
            )
            if (
                abs(pixel_error[0]) <= int(config.center_pixel_tolerance)
                and abs(pixel_error[1]) <= int(config.center_pixel_tolerance)
            ):
                logs.append("centering complete: target already near image center")
                return cycle, logs, cycle_records

            target_pose_m, target_rpy_deg, debug = self._plan_centering_move(
                current_pose=cycle["current_pose"],
                base_to_camera=cycle["base_to_camera"],
                intrinsics=intrinsics,
                config=config,
                target_uv=center_uv,
                target_depth_m=center_camera_m[2],
            )
            ok, violations = planner.check_workspace(target_pose_m)
            if not ok:
                logs.append("centering blocked by workspace: " + " | ".join(violations))
                return cycle, logs, cycle_records
            if bool(config.dry_run):
                logs.append(
                    "centering dry-run: "
                    f"move_to=({target_pose_m[0]:.3f}, {target_pose_m[1]:.3f}, {target_pose_m[2]:.3f}) "
                    f"delta=({debug['delta_base_x_m']:+.3f}, {debug['delta_base_y_m']:+.3f})"
                )
                return cycle, logs, cycle_records

            self._publish_status(f"precentering_move: run_id={run_id} iter={iteration + 1}")
            self._execute_named_pose(
                name=f"precenter_{iteration + 1}",
                position_m=target_pose_m,
                rpy_deg=target_rpy_deg,
                speed_percent=float(self.get_parameter("speed").value),
                open_gripper_first=False,
                timeout_s=20.0,
            )
            time.sleep(max(0.0, float(config.center_settle_time_s)))

        logs.append(f"centering reached max_iterations={max_iterations}, using latest observation")
        final_cycle = self._capture_analyze_cycle(
            run_id=run_id,
            prompt=prompt,
            options=options,
            hand_eye=hand_eye,
            phase_label="post_precenter",
        )
        cycle_records.append(dict(final_cycle["debug"]))
        return final_cycle, logs, cycle_records

    def _validate_candidate_plan(
        self,
        *,
        run_id: str,
        plan,
        move_home_after: bool,
    ) -> dict[str, object]:
        execute_req = ExecuteGraspPlan.Request()
        execute_req.run_id = run_id
        execute_req.execute = False
        execute_req.move_home_after = bool(move_home_after)
        execute_req.plan = grasp_plan_to_msg(plan)
        execute_response = self._call_client(self._execute_plan_client, execute_req, timeout_s=30.0)
        payload: dict[str, object] = {}
        raw_json = str(execute_response.execution_json or "").strip()
        if raw_json:
            try:
                loaded = json.loads(raw_json)
                if isinstance(loaded, dict):
                    payload = loaded
            except json.JSONDecodeError:
                payload = {}
        if execute_response.success:
            payload.setdefault("robot_validation_result", "accepted")
            payload.setdefault("robot_validation_stage", None)
            payload.setdefault("ik_error_type", None)
            payload.setdefault("ik_error_message", None)
            payload.setdefault("waypoint_results", [])
            return payload
        payload.setdefault("robot_validation_result", "rejected_by_robot_validation")
        payload.setdefault("robot_validation_stage", None)
        payload.setdefault("ik_error_type", "ik_error")
        payload.setdefault("ik_error_message", str(execute_response.message))
        payload.setdefault("waypoint_results", [])
        return payload

    def _maybe_auto_start(self) -> None:
        if not self._auto_start_armed:
            return
        self._auto_start_armed = False
        prompt = str(self.get_parameter("prompt").value or "").strip()
        if not prompt and not self._auto_target_from_card_enabled():
            self._publish_status("auto_start skipped: empty prompt")
            return
        accepted, message = self._start_background_run(prompt_override=prompt or None)
        if not accepted:
            self._publish_status(f"auto_start rejected: {message}")

    def _handle_run_prompt(self, msg: String) -> None:
        accepted, message = self._start_background_run(prompt_override=msg.data.strip())
        if not accepted:
            self._publish_status(f"run_prompt rejected: {message}")

    def _handle_run_service(self, _request, response):
        prompt = str(self.get_parameter("prompt").value or "").strip()
        accepted, message = self._start_background_run(prompt_override=prompt or None)
        response.success = accepted
        response.message = message
        return response

    def _handle_probe_service(self, _request, response):
        try:
            probe_lines = []
            for client in (
                self._capture_client,
                self._analyze_client,
                self._detect_target_2d_client,
                self._match_label_client,
                self._executor_set_parameters_client,
                self._get_state_client,
                self._named_pose_client,
                self._execute_plan_client,
                self._execute_place_client,
                self._stop_robot_client,
            ):
                self._wait_for_client(client, timeout_s=2.0)
                probe_lines.append(f"service ok: {client.srv_name}")
            robot_state = self._call_client(self._get_state_client, GetRobotState.Request(), timeout_s=5.0)
            probe_lines.append(f"robot state: {robot_state.message}")
            response.success = True
            response.message = " | ".join(probe_lines)
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    @staticmethod
    def _depth_preview(depth_meters: np.ndarray) -> np.ndarray:
        depth = np.asarray(depth_meters, dtype=np.float32)
        valid = np.isfinite(depth) & (depth > 0.10) & (depth < 3.0)
        preview = np.zeros(depth.shape, dtype=np.uint8)
        if np.count_nonzero(valid) > 20:
            low, high = np.percentile(depth[valid], [2.0, 98.0])
            if float(high) > float(low):
                scaled = np.clip((depth - low) / (high - low), 0.0, 1.0)
                preview[valid] = np.asarray(scaled[valid] * 255.0, dtype=np.uint8)
        colored = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
        colored[~valid] = 0
        return colored

    def _move_base_for_scan(
        self,
        distance_m: float,
        *,
        timeout_s: float | None = None,
        speed_mps: float | None = None,
    ) -> dict[str, object]:
        request = MoveBaseRelative.Request()
        request.distance_m = float(distance_m)
        request.speed_mps = float(
            speed_mps
            if speed_mps is not None
            else self.get_parameter("base_multiview_speed_mps").value
        )
        request.timeout_s = float(
            timeout_s
            if timeout_s is not None
            else self.get_parameter("base_multiview_move_timeout_s").value
        )
        response = self._call_client(
            self._move_base_client,
            request,
            timeout_s=float(request.timeout_s) + 5.0,
        )
        payload = {
            "success": bool(response.success),
            "message": str(response.message),
            "requested_distance_m": float(distance_m),
            "traveled_m": float(response.traveled_m),
            "lateral_error_m": float(response.lateral_error_m),
            "yaw_error_deg": float(response.yaw_error_deg),
        }
        if not response.success:
            raise RuntimeError(f"Scout scan move failed: {response.message}")
        return payload

    def _move_base_for_scan_until_preview_trigger(
        self,
        distance_m: float,
        *,
        run_id: str,
        preview_probe,
        should_stop,
        timeout_s: float | None = None,
        speed_mps: float | None = None,
    ) -> tuple[dict[str, object], object | None]:
        """Move one bounded scan segment and stop early on a live RGB trigger.

        The Scout controller remains responsible for odometry, drift, yaw and
        emergency-stop checks.  This method only adds a visual stop request;
        the returned trigger is always re-checked from a stopped full RGB-D
        capture before any grasp is executed.
        """
        request = MoveBaseRelative.Request()
        request.distance_m = float(distance_m)
        request.speed_mps = float(
            speed_mps
            if speed_mps is not None
            else self.get_parameter("base_multiview_speed_mps").value
        )
        request.timeout_s = float(
            timeout_s
            if timeout_s is not None
            else self.get_parameter("base_multiview_move_timeout_s").value
        )
        self._wait_for_client(self._move_base_client, timeout_s=5.0)
        future = self._move_base_client.call_async(request)
        deadline = time.monotonic() + float(request.timeout_s) + 3.0
        preview_sequence = -1
        trigger = None
        while rclpy.ok() and not future.done() and time.monotonic() < deadline:
            if self._stop_requested:
                self._request_base_scan_stop()
                raise RuntimeError("orchestrator stop requested")
            preview_detection = preview_probe(
                after_sequence=preview_sequence,
                wait_s=min(
                    0.10,
                    max(
                        0.02,
                        float(
                            self.get_parameter(
                                "continuous_search_poll_s"
                            ).value
                        ),
                    ),
                ),
            )
            if preview_detection is None:
                continue
            detection, preview_sequence = preview_detection
            if bool(should_stop(detection)):
                trigger = detection
                self._publish_status(
                    f"continuous_search_trigger: run_id={run_id} "
                    f"distance={float(distance_m):.3f}m"
                )
                self._request_base_scan_stop()
                break

        if trigger is not None:
            stop_deadline = time.monotonic() + 4.0
            while not future.done() and time.monotonic() < stop_deadline:
                time.sleep(0.02)
        if not future.done():
            self._request_base_scan_stop()
            raise TimeoutError("Scout continuous scan move did not complete")
        if future.cancelled():
            raise RuntimeError("Scout continuous scan move was cancelled")
        exception = future.exception()
        if exception is not None:
            raise exception
        response = future.result()
        payload = {
            "success": bool(response.success) or trigger is not None,
            "message": str(response.message),
            "requested_distance_m": float(distance_m),
            "traveled_m": float(response.traveled_m),
            "lateral_error_m": float(response.lateral_error_m),
            "yaw_error_deg": float(response.yaw_error_deg),
            "stopped_by_continuous_preview": trigger is not None,
        }
        if not response.success and trigger is None:
            raise RuntimeError(f"Scout scan move failed: {response.message}")
        return payload, trigger

    def _request_base_scan_stop(self) -> None:
        try:
            self._call_client(
                self._stop_base_client,
                Trigger.Request(),
                timeout_s=3.0,
            )
        except Exception:
            return

    @staticmethod
    def _grasp_scan_target_center_norm(
        cycle: dict[str, object],
    ) -> tuple[float, float] | None:
        """Return the segmented target center in normalized image coordinates.

        A catalog-target scan must use an actual segmented object center.  The
        grasp translation is deliberately not accepted as a substitute: scene
        fallback grasps do not prove that the requested object was detected.
        """
        candidate = cycle.get("candidate")
        if candidate is None:
            pool = list(cycle.get("candidate_pool") or [])
            candidate = pool[0] if pool else None
        center = getattr(candidate, "object_center_camera_m", None)
        if center is None or len(center) != 3:
            return None
        x_m, y_m, z_m = (float(value) for value in center)
        if not math.isfinite(z_m) or z_m <= 1e-6:
            return None

        capture = cycle.get("capture_response")
        camera_info = getattr(capture, "camera_info", None)
        width = float(getattr(camera_info, "width", 0.0) or 0.0)
        height = float(getattr(camera_info, "height", 0.0) or 0.0)
        raw_k = getattr(camera_info, "k", None)
        k = list(raw_k) if raw_k is not None else []
        if width <= 1.0 or height <= 1.0 or len(k) != 9:
            return None
        fx = float(k[0])
        cx = float(k[2])
        fy = float(k[4])
        cy = float(k[5])
        if (
            not math.isfinite(fx)
            or not math.isfinite(fy)
            or abs(fx) <= 1e-9
            or abs(fy) <= 1e-9
        ):
            return None
        center_u = (x_m * fx / z_m) + cx
        center_v = (y_m * fy / z_m) + cy
        return (center_u / width, center_v / height)

    def _run_base_grasp_target_scan(
        self,
        *,
        run_id: str,
        prompt: str,
        target_item_id: str,
        options: dict[str, object],
        hand_eye: np.ndarray,
    ) -> tuple[dict[str, object], dict[str, object]]:
        """Scan forward until the requested item has a valid grasp candidate.

        The Scout deliberately remains at the successful view: that is the
        base pose in which the following grasp plan is expressed.
        """
        if not target_item_id:
            raise RuntimeError(
                "base grasp scan requires one selected catalog target item"
            )
        self._base_odom_snapshot()
        step_m = abs(float(self.get_parameter("base_multiview_offset_m").value))
        max_travel_m = abs(
            float(self.get_parameter("base_multiview_max_travel_m").value)
        )
        max_views = max(
            1, int(self.get_parameter("base_multiview_max_views").value)
        )
        target_item = self._item_catalog().resolve(target_item_id)
        if target_item is None:
            raise RuntimeError(f"unknown grasp scan target: {target_item_id}")
        reference_parameter = (
            "base_grasp_bottle_center_norm"
            if str(target_item.kind) == "bottle"
            else "base_grasp_block_center_norm"
        )
        center_reference = [
            float(value)
            for value in list(self.get_parameter(reference_parameter).value or [])
        ]
        if len(center_reference) != 2:
            raise RuntimeError(f"{reference_parameter} must contain [u_norm, v_norm]")
        center_tolerance_u = max(
            0.02,
            min(
                0.30,
                float(
                    self.get_parameter(
                        "base_grasp_center_tolerance_u_norm"
                    ).value
                ),
            ),
        )
        center_tolerance_v = max(
            0.02,
            min(
                0.30,
                float(
                    self.get_parameter(
                        "base_grasp_center_tolerance_v_norm"
                    ).value
                ),
            ),
        )
        fine_step_m = abs(
            float(self.get_parameter("base_target_fine_step_m").value)
        )
        if step_m <= 0.01:
            raise RuntimeError("base_multiview_offset_m must be greater than 0.01")
        if fine_step_m <= 0.01:
            raise RuntimeError("base_target_fine_step_m must be greater than 0.01")

        def detection_is_usable(detection) -> bool:
            if not bool(getattr(detection, "found", False)):
                return False
            u_norm = float(getattr(detection, "center_u_norm", 0.0))
            v_norm = float(getattr(detection, "center_v_norm", 0.0))
            exclude_roi = getattr(self, "_grasp_exclude_roi_norm", None)
            search_roi = getattr(self, "_grasp_search_roi_norm", None)
            if exclude_roi and self._norm_point_in_roi(u_norm, v_norm, exclude_roi):
                return False
            if search_roi and not self._norm_point_in_roi(u_norm, v_norm, search_roi):
                return False
            return True

        def detection_metrics(detection):
            usable = detection_is_usable(detection)
            center_norm = (
                (float(detection.center_u_norm), float(detection.center_v_norm))
                if usable
                else None
            )
            center_error_u = (
                abs(center_norm[0] - center_reference[0])
                if center_norm is not None
                else None
            )
            center_error_v = (
                abs(center_norm[1] - center_reference[1])
                if center_norm is not None
                else None
            )
            centered = bool(
                usable
                and center_error_u is not None
                and center_error_v is not None
                and math.isfinite(center_error_u)
                and math.isfinite(center_error_v)
                and center_error_u <= center_tolerance_u
                and center_error_v <= center_tolerance_v
            )
            return center_norm, center_error_u, center_error_v, centered

        scan_id = f"grasp-single-pass-{int(time.time() * 1000)}"
        views: list[dict[str, object]] = []
        movements: list[dict[str, object]] = []
        current_offset = 0.0
        selected_cycle: dict[str, object] | None = None
        last_cycle: dict[str, object] | None = None
        last_captured: dict[str, object] | None = None
        last_phase_label = "grasp_scan_start"
        force_full_confirmation = False
        had_target = False
        scan_sign = 1.0
        consecutive_target_misses = 0
        self._publish_status(
            f"scanning_grasp_target: run_id={run_id} item={target_item_id}"
        )
        try:
            while len(views) < max_views:
                view_name = "start" if not views else f"forward_{len(views):02d}"
                phase_label = f"grasp_scan_{view_name}"
                captured = self._capture_detect_target_2d(
                    run_id=run_id,
                    prompt=prompt,
                    hand_eye=hand_eye,
                    phase_label=phase_label,
                    depth_fusion_frames=(
                        int(self.get_parameter("depth_fusion_frames").value)
                        if force_full_confirmation
                        else None
                    ),
                )
                force_full_confirmation = False
                last_captured = captured
                last_phase_label = phase_label
                detection = captured["detection_response"]
                (
                    center_norm,
                    center_error_u,
                    center_error_v,
                    detection_centered,
                ) = detection_metrics(detection)
                cycle = None
                candidates: list[object] = []
                target_centered = False
                if detection_centered:
                    # Search used one fast depth frame.  Before committing to
                    # a grasp, take a fresh full-fusion frame while stopped
                    # and verify the object is still in the calibrated window.
                    captured = self._capture_detect_target_2d(
                        run_id=run_id,
                        prompt=prompt,
                        hand_eye=hand_eye,
                        phase_label=f"{phase_label}_final",
                        depth_fusion_frames=int(
                            self.get_parameter("depth_fusion_frames").value
                        ),
                    )
                    last_captured = captured
                    last_phase_label = f"{phase_label}_final"
                    detection = captured["detection_response"]
                    (
                        center_norm,
                        center_error_u,
                        center_error_v,
                        detection_centered,
                    ) = detection_metrics(detection)
                if detection_centered:
                    cycle = self._analyze_captured_cycle(
                        run_id=run_id,
                        prompt=prompt,
                        options=options,
                        phase_label=phase_label,
                        captured=captured,
                    )
                    last_cycle = cycle
                    candidates = list(cycle.get("candidate_pool") or [])
                    if not candidates and cycle.get("candidate") is not None:
                        candidates = [cycle["candidate"]]
                    # A scene-level GraspNet fallback is not evidence that the
                    # requested item is graspable.  Keep only candidates tied
                    # to the detected instance; otherwise a centered label can
                    # accidentally authorize a grasp on a neighbouring object.
                    candidates = [
                        candidate
                        for candidate in candidates
                        if getattr(candidate, "object_center_camera_m", None)
                        is not None
                    ]
                    exclude_roi = getattr(self, "_grasp_exclude_roi_norm", None)
                    if exclude_roi and candidates:
                        kept: list[object] = []
                        for candidate in candidates:
                            center = self._grasp_scan_target_center_norm(
                                {"candidate": candidate, "capture_response": captured.get("capture_response")}
                            )
                            if center is None or not self._norm_point_in_roi(
                                center[0], center[1], exclude_roi
                            ):
                                kept.append(candidate)
                        candidates = kept
                    if candidates:
                        cycle = dict(cycle)
                        cycle["candidate_pool"] = candidates
                        if getattr(
                            cycle.get("candidate"),
                            "object_center_camera_m",
                            None,
                        ) is None:
                            cycle["candidate"] = candidates[0]
                        last_cycle = cycle
                    target_centered = bool(candidates)
                views.append(
                    {
                        "view_name": view_name,
                        "offset_from_start_m": float(current_offset),
                        "scene_id": (
                            str(captured["capture_response"].scene_id)
                            if captured.get("capture_response") is not None
                            else f"continuous-preview-{captured.get('preview_sequence', -1)}"
                        ),
                        "capture": (
                            self._capture_debug_dict(captured["capture_response"])
                            if captured.get("capture_response") is not None
                            else {"source": "continuous_preview"}
                        ),
                        "target_detection_2d": {
                            "found": bool(detection.found),
                            "center_norm": list(center_norm) if center_norm else None,
                            "confidence": float(detection.confidence),
                            "backend": str(detection.backend),
                            "centered_for_full_analysis": detection_centered,
                        },
                        "vision": (
                            self._analyze_debug_dict(cycle["analyze_response"])
                            if cycle is not None
                            else None
                        ),
                        "candidate_count": len(candidates),
                        "target_center_norm": list(center_norm) if center_norm else None,
                        "target_center_reference_norm": list(center_reference),
                        "target_center_error_u_norm": center_error_u,
                        "target_center_error_v_norm": center_error_v,
                        "target_centered": target_centered,
                    }
                )
                if target_centered:
                    selected_cycle = cycle
                    break

                target_seen = bool(
                    detection_is_usable(detection)
                    and center_norm is not None
                    and center_error_v is not None
                    # The calibrated block/bottle heights are deliberately
                    # different. A red bottle cap high in the image must not
                    # steer a red-block search merely because its hue matches.
                    and center_error_v <= center_tolerance_v
                )
                if target_seen:
                    had_target = True
                    consecutive_target_misses = 0
                    # Keep moving in the established direction while the
                    # target remains visible. Color-block proposals can jump
                    # between two red regions for one frame; reversing on a
                    # larger center error made Scout oscillate at the origin.
                    next_step_m = scan_sign * fine_step_m
                elif had_target:
                    consecutive_target_misses += 1
                    reverse_after = max(
                        1,
                        int(
                            self.get_parameter(
                                "grasp_scan_lost_frames_before_reverse"
                            ).value
                        ),
                    )
                    if consecutive_target_misses >= reverse_after:
                        # Do not abandon a briefly visible dark bottle. After
                        # a confirmed loss, reverse in fine steps toward the
                        # last visible view.
                        scan_sign = -1.0
                    next_step_m = scan_sign * fine_step_m
                else:
                    next_step_m = step_m
                if next_step_m > 0.0:
                    remaining = max_travel_m - current_offset
                    if remaining <= 0.01:
                        break
                    move_m = min(next_step_m, remaining)
                else:
                    if current_offset <= 0.02:
                        # The target was seen near the scan origin; continue
                        # forward rather than driving behind the cleared lane.
                        scan_sign = 1.0
                        move_m = min(fine_step_m, max_travel_m - current_offset)
                    else:
                        move_m = -min(abs(next_step_m), current_offset)
                if bool(
                    self.get_parameter("continuous_search_stop_on_center").value
                ) and bool(self.get_parameter("continuous_search_enabled").value):
                    movement, trigger = self._move_base_for_scan_until_preview_trigger(
                        move_m,
                        run_id=run_id,
                        preview_probe=lambda **kwargs: self._detect_target_2d_from_preview(
                            run_id=run_id,
                            prompt=prompt,
                            **kwargs,
                        ),
                        should_stop=lambda preview_detection: detection_metrics(
                            preview_detection
                        )[3],
                    )
                    force_full_confirmation = trigger is not None
                else:
                    movement = self._move_base_for_scan(move_m)
                current_offset += float(movement["traveled_m"])
                current_offset = max(0.0, float(current_offset))
                movement["offset_after_move_m"] = float(current_offset)
                movements.append(movement)
                time.sleep(
                    max(
                        0.0,
                        float(
                            self.get_parameter("base_multiview_settle_s").value
                        ),
                    )
                )
        finally:
            self._request_base_scan_stop()

        scan_payload = {
            "scan_id": scan_id,
            "scan_mode": "base_single_pass_grasp",
            "target_item_id": target_item_id,
            "success": selected_cycle is not None,
            "center_reference_norm": list(center_reference),
            "center_tolerance_u_norm": center_tolerance_u,
            "center_tolerance_v_norm": center_tolerance_v,
            "final_offset_from_start_m": float(current_offset),
            "views": views,
            "movements": movements,
            "base_returned_to_start": False,
            "motion_command_sent": bool(movements),
            "gripper_command_sent": False,
        }
        if selected_cycle is not None:
            return selected_cycle, scan_payload
        if last_cycle is None and last_captured is not None:
            # Preserve the old failure diagnostics while avoiding GraspNet on
            # every search frame: analyze only the final view if no centered
            # target ever triggered a full analysis.
            if last_captured.get("capture_response") is None:
                last_captured = self._capture_detect_target_2d(
                    run_id=run_id,
                    prompt=prompt,
                    hand_eye=hand_eye,
                    phase_label=f"{last_phase_label}_final_diagnostic",
                    depth_fusion_frames=int(
                        self.get_parameter("depth_fusion_frames").value
                    ),
                )
            last_cycle = self._analyze_captured_cycle(
                run_id=run_id,
                prompt=prompt,
                options=options,
                phase_label=last_phase_label,
                captured=last_captured,
            )
        if last_cycle is None:
            raise RuntimeError("grasp scan ended before capturing a camera view")
        return last_cycle, scan_payload

    @staticmethod
    def _label_overlay(
        color_bgr: np.ndarray,
        label: dict[str, object],
    ) -> np.ndarray:
        overlay = np.asarray(color_bgr).copy()
        detections = list(
            dict(label.get("diagnostics") or {}).get("detections") or []
        )
        for detection in detections:
            bbox = list(dict(detection).get("bbox_xywh") or [])
            if len(bbox) != 4:
                continue
            x, y, width, height = (int(value) for value in bbox)
            cv2.rectangle(
                overlay,
                (x, y),
                (x + width, y + height),
                (0, 255, 0),
                2,
            )
            method = str(dict(detection).get("method") or "")
            cv2.putText(
                overlay,
                f"{dict(detection).get('item_id') or ''} [{method}]",
                (x, max(18, y - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 80, 255),
                2,
                cv2.LINE_AA,
            )
        return overlay

    def _capture_placement_scan_view(
        self,
        *,
        scan_id: str,
        view_name: str,
        offset_from_start_m: float,
        item_id: str,
        base_to_camera: np.ndarray,
        output_dir: Path,
    ) -> dict[str, object]:
        preview_match = self._match_box_label_from_preview(
            run_id=f"{scan_id}-{view_name}",
            item_id=item_id,
        )
        if preview_match is not None:
            label, color, preview_sequence = preview_match
            image_names = {
                "color": f"{view_name}_color.png",
                "overlay": f"{view_name}_overlay.png",
            }
            for name, image in (
                ("color", color),
                ("overlay", self._label_overlay(color, label)),
            ):
                path = output_dir / image_names[name]
                if not cv2.imwrite(str(path), image):
                    raise RuntimeError(f"failed to write placement scan image: {path}")
            return {
                "view_name": view_name,
                "offset_from_start_m": float(offset_from_start_m),
                "capture": {
                    "source": "continuous_preview",
                    "scene_id": f"continuous-preview-{preview_sequence}",
                    "color_width": int(color.shape[1]),
                    "color_height": int(color.shape[0]),
                },
                "label_match": label,
                "images": image_names,
            }
        capture = self._capture_scene_once(
            run_id=f"{scan_id}-{view_name}",
            phase_label=f"placement_scan_{view_name}",
            depth_fusion_frames=int(
                self.get_parameter("search_depth_fusion_frames").value
            ),
        )
        label = self._match_box_label_once(
            run_id=f"{scan_id}-{view_name}",
            item_id=item_id,
            capture_response=capture,
            base_to_camera=base_to_camera,
            require_complete=False,
        )
        color = color_msg_to_bgr(capture.color_image)
        depth = depth_msg_to_meters(capture.depth_image)
        image_names = {
            "color": f"{view_name}_color.png",
            "overlay": f"{view_name}_overlay.png",
            "depth": f"{view_name}_depth.png",
        }
        for name, image in (
            ("color", color),
            ("overlay", self._label_overlay(color, label)),
            ("depth", self._depth_preview(depth)),
        ):
            path = output_dir / image_names[name]
            if not cv2.imwrite(str(path), image):
                raise RuntimeError(f"failed to write placement scan image: {path}")
        return {
            "view_name": view_name,
            "offset_from_start_m": float(offset_from_start_m),
            "capture": self._capture_debug_dict(capture),
            "label_match": label,
            "images": image_names,
        }

    def _fuse_single_pass_label_map(
        self,
        *,
        views: list[dict[str, object]],
        item_id: str,
    ) -> dict[str, object]:
        """Fuse overlapping 2-D label orders without using transparent-box depth."""
        expected_ids = set(self._item_catalog().items)
        observations: dict[str, list[dict[str, object]]] = {
            key: [] for key in expected_ids
        }
        precedence: dict[str, set[str]] = {key: set() for key in expected_ids}
        directional_weights: dict[tuple[str, str], float] = {}
        view_orders: list[dict[str, object]] = []
        rejected_observations: list[dict[str, object]] = []

        for view in views:
            diagnostics = dict(
                dict(view.get("label_match") or {}).get("diagnostics") or {}
            )
            best_by_item: dict[str, dict[str, object]] = {}
            for raw in list(diagnostics.get("detections") or []):
                detection = dict(raw)
                detected_id = str(detection.get("item_id") or "")
                bbox = list(detection.get("bbox_xywh") or [])
                if detected_id not in expected_ids or len(bbox) != 4:
                    continue
                method = str(detection.get("method") or "")
                capture = dict(view.get("capture") or {})
                image_width = int(capture.get("color_width") or 0)
                x, _y, width, _height = (float(value) for value in bbox)
                rejection_reason = ""
                if "partial" in method:
                    rejection_reason = "partial label detection"
                elif x <= 2.0 or (
                    image_width > 0 and x + width >= image_width - 2.0
                ):
                    rejection_reason = "label detection touches image boundary"
                if rejection_reason:
                    rejected_observations.append(
                        {
                            **detection,
                            "view_name": str(view["view_name"]),
                            "reason": rejection_reason,
                        }
                    )
                    continue
                previous = best_by_item.get(detected_id)
                if previous is None or float(
                    detection.get("confidence") or 0.0
                ) > float(previous.get("confidence") or 0.0):
                    best_by_item[detected_id] = detection

            ordered = sorted(
                best_by_item.items(),
                key=lambda value: float(value[1]["bbox_xywh"][0])
                + 0.5 * float(value[1]["bbox_xywh"][2]),
            )
            ordered_ids = [value[0] for value in ordered]
            view_orders.append(
                {
                    "view_name": str(view["view_name"]),
                    "offset_from_start_m": float(view["offset_from_start_m"]),
                    "detected_item_ids_left_to_right": ordered_ids,
                }
            )
            for detected_id, detection in ordered:
                bbox = [float(value) for value in detection["bbox_xywh"]]
                observations[detected_id].append(
                    {
                        "view_name": str(view["view_name"]),
                        "offset_from_start_m": float(view["offset_from_start_m"]),
                        "confidence": float(detection.get("confidence") or 0.0),
                        "bbox_xywh": bbox,
                        "bbox_center_x_px": bbox[0] + 0.5 * bbox[2],
                        "method": str(detection.get("method") or ""),
                    }
                )
            for left_index, (left_id, left_detection) in enumerate(ordered):
                for right_id, right_detection in ordered[left_index + 1 :]:
                    # A pairwise relation is only as reliable as its weaker
                    # detection. Accumulate repeated overlapping observations
                    # instead of allowing one weak false positive to create a
                    # hard graph cycle.
                    weight = min(
                        float(left_detection.get("confidence") or 0.0),
                        float(right_detection.get("confidence") or 0.0),
                    )
                    key = (left_id, right_id)
                    directional_weights[key] = (
                        directional_weights.get(key, 0.0) + weight
                    )

        missing = sorted(
            detected_id
            for detected_id, item_observations in observations.items()
            if not item_observations
        )
        if missing:
            raise RuntimeError(
                "single-pass scan still missing labels: " + ", ".join(missing)
            )

        pairwise_evidence: list[dict[str, object]] = []
        sorted_ids = sorted(expected_ids)
        for left_index, first_id in enumerate(sorted_ids):
            for second_id in sorted_ids[left_index + 1 :]:
                first_before = directional_weights.get(
                    (first_id, second_id), 0.0
                )
                second_before = directional_weights.get(
                    (second_id, first_id), 0.0
                )
                total = first_before + second_before
                margin = abs(first_before - second_before)
                normalized_margin = margin / total if total > 1e-9 else 0.0
                winner = ""
                loser = ""
                # Keep fail-closed semantics when evidence is weak or nearly
                # tied. One clear high-confidence observation is sufficient;
                # contradictory observations need a meaningful weighted lead.
                if margin >= 0.15 and normalized_margin >= 0.20:
                    if first_before > second_before:
                        winner, loser = first_id, second_id
                    else:
                        winner, loser = second_id, first_id
                    precedence[winner].add(loser)
                pairwise_evidence.append(
                    {
                        "first_item_id": first_id,
                        "second_item_id": second_id,
                        "first_before_weight": first_before,
                        "second_before_weight": second_before,
                        "normalized_margin": normalized_margin,
                        "accepted_order": (
                            [winner, loser] if winner and loser else []
                        ),
                    }
                )

        remaining = set(expected_ids)
        ordered_item_ids: list[str] = []
        while remaining:
            candidates = sorted(
                node
                for node in remaining
                if not any(node in precedence[source] for source in remaining)
            )
            if len(candidates) != 1:
                detail = ", ".join(candidates) if candidates else "cycle"
                raise RuntimeError(
                    "label order is ambiguous; increase scan overlap "
                    f"(next candidates: {detail})"
                )
            selected = candidates[0]
            ordered_item_ids.append(selected)
            remaining.remove(selected)

        item_to_slot = {
            detected_id: index
            for index, detected_id in enumerate(ordered_item_ids)
        }
        item_to_confidence = {
            detected_id: max(
                float(value.get("confidence") or 0.0)
                for value in item_observations
            )
            for detected_id, item_observations in observations.items()
        }
        return {
            "detected_item_ids_left_to_right": ordered_item_ids,
            "item_to_slot_index": item_to_slot,
            "item_to_confidence": item_to_confidence,
            "item_observations": observations,
            "view_orders": view_orders,
            "pairwise_evidence": pairwise_evidence,
            "rejected_observations": rejected_observations,
            "target_item_id": item_id,
            "target_slot_index": int(item_to_slot[item_id]),
            "localization_source": "2d_weighted_label_order",
            "transparent_depth_used": False,
        }

    def _fuse_multiview_box_map(
        self,
        *,
        views: list[dict[str, object]],
        item_id: str,
        base_to_camera_at_start: np.ndarray,
    ) -> dict[str, object]:
        catalog = self._item_catalog()
        observations_by_item: dict[str, list[dict[str, object]]] = {
            key: [] for key in catalog.items
        }
        rejected_observations: list[dict[str, object]] = []
        table_z_m = float(self.get_parameter("table_z_m").value)
        minimum_label_z_m = table_z_m - 0.015
        maximum_label_z_m = table_z_m + 0.35
        for view in views:
            offset = float(view["offset_from_start_m"])
            diagnostics = dict(
                dict(view.get("label_match") or {}).get("diagnostics") or {}
            )
            for raw in list(
                diagnostics.get("partial_label_observations_base_m") or []
            ):
                observation = dict(raw)
                detected_id = str(observation.get("item_id") or "")
                point = list(observation.get("point_base_m") or [])
                if detected_id not in observations_by_item or len(point) != 3:
                    continue
                method = str(observation.get("method") or "")
                depth_source = str(
                    observation.get("depth_source") or "measured"
                )
                if "partial" in method or depth_source != "measured":
                    rejected_observations.append(
                        {
                            **observation,
                            "view_name": str(view["view_name"]),
                            "reason": (
                                "partial label detection"
                                if "partial" in method
                                else f"non-measured depth source: {depth_source}"
                            ),
                        }
                    )
                    continue
                # The arm base axes are fixed to the Scout chassis and the scan
                # is straight along chassis +X/-X. Convert each current-base
                # point into the base frame at the scan start.
                point_at_start = [
                    float(point[0]) + offset,
                    float(point[1]),
                    float(point[2]),
                ]
                if (
                    not all(math.isfinite(value) for value in point_at_start)
                    or not minimum_label_z_m
                    <= point_at_start[2]
                    <= maximum_label_z_m
                ):
                    rejected_observations.append(
                        {
                            **observation,
                            "view_name": str(view["view_name"]),
                            "point_start_base_m": point_at_start,
                            "reason": (
                                "projected label height outside placement row "
                                f"[{minimum_label_z_m:.3f},"
                                f"{maximum_label_z_m:.3f}]m"
                            ),
                        }
                    )
                    continue
                observations_by_item[detected_id].append(
                    {
                        **observation,
                        "view_name": str(view["view_name"]),
                        "point_start_base_m": point_at_start,
                    }
                )

        missing = [
            key for key, observations in observations_by_item.items()
            if not observations
        ]
        if missing:
            raise RuntimeError(
                "multi-view scan still missing labels: " + ", ".join(missing)
            )

        fused_points: list[tuple[float, float, float]] = []
        fused_detections: list[LabelDetection] = []
        fusion_diagnostics: dict[str, object] = {}
        for detected_id, observations in observations_by_item.items():
            ranked = sorted(
                observations,
                key=lambda value: float(value.get("confidence") or 0.0),
                reverse=True,
            )
            anchor = np.asarray(
                ranked[0]["point_start_base_m"],
                dtype=np.float64,
            )
            compatible = [
                value
                for value in ranked
                if float(
                    np.linalg.norm(
                        np.asarray(value["point_start_base_m"], dtype=np.float64)
                        - anchor
                    )
                )
                <= 0.09
            ]
            weights = np.asarray(
                [
                    max(0.05, float(value.get("confidence") or 0.0))
                    for value in compatible
                ],
                dtype=np.float64,
            )
            points = np.asarray(
                [value["point_start_base_m"] for value in compatible],
                dtype=np.float64,
            )
            fused = np.average(points, axis=0, weights=weights)
            confidence = float(max(weights))
            fused_points.append(tuple(float(value) for value in fused))
            fused_detections.append(
                LabelDetection(
                    item_id=detected_id,
                    confidence=confidence,
                    bbox_xywh=(0, 0, 1, 1),
                    method="multi_view_fusion",
                )
            )
            fusion_diagnostics[detected_id] = {
                "observation_count": len(observations),
                "compatible_count": len(compatible),
                "observations": observations,
                "fused_point_start_base_m": list(fused),
            }

        transform = np.asarray(
            base_to_camera_at_start,
            dtype=np.float64,
        ).reshape(4, 4)
        image_right_xy = tuple(float(value) for value in transform[:2, 0])
        matcher = ReferenceLabelMatcher(catalog)
        localization = matcher.localize_box_row_from_points(
            detections=tuple(fused_detections),
            label_centers_base_m=tuple(fused_points),
            target_item_id=item_id,
            table_z_m=table_z_m,
            camera_xy_base=tuple(float(value) for value in transform[:2, 3]),
            image_right_direction_base_xy=image_right_xy,
        )

        row_direction = np.asarray(image_right_xy, dtype=np.float64)
        row_direction /= max(1e-9, float(np.linalg.norm(row_direction)))
        order = np.argsort(
            np.asarray(fused_points, dtype=np.float64)[:, :2] @ row_direction
        )
        ordered_item_ids = [
            fused_detections[int(index)].item_id for index in order
        ]
        item_to_slot = {
            detected_id: index
            for index, detected_id in enumerate(ordered_item_ids)
        }
        item_to_center = {
            detected_id: list(localization.box_centers_base_m[index])
            for index, detected_id in enumerate(ordered_item_ids)
        }
        item_to_confidence = {
            detection.item_id: float(detection.confidence)
            for detection in fused_detections
        }
        return {
            "detected_item_ids_left_to_right": ordered_item_ids,
            "item_to_slot_index": item_to_slot,
            "item_to_box_center_base_m": item_to_center,
            "item_to_confidence": item_to_confidence,
            "box_centers_base_m": [
                list(value) for value in localization.box_centers_base_m
            ],
            "label_centers_base_m": [
                list(value) for value in localization.label_centers_base_m
            ],
            "adjacent_pitch_mm": list(localization.adjacent_pitch_mm),
            "raw_adjacent_pitch_mm": list(
                localization.raw_adjacent_pitch_mm
            ),
            "row_fit_residual_mm": list(localization.fit_residual_mm),
            "interior_direction_base": list(
                localization.interior_direction_base
            ),
            "fusion_diagnostics": fusion_diagnostics,
            "rejected_observations": rejected_observations,
        }

    def _handle_scan_placement_service(self, _request, response):
        with self._run_lock:
            if self._scan_active:
                response.success = False
                response.message = "placement scan is already running"
                return response
            if self._run_thread is not None and self._run_thread.is_alive():
                response.success = False
                response.message = "cannot scan placement area while pipeline is running"
                return response
            if self._pending_confirmation is not None:
                response.success = False
                response.message = "cannot scan while grasp confirmation is pending"
                return response
            self._scan_active = True

        scan_id = f"placement-scan-{int(time.time() * 1000)}"
        try:
            self._publish_status(f"scanning_placement_area: scan_id={scan_id}")
            _, _, hand_eye, _ = self._build_runtime()
            state = self._read_robot_state_snapshot(hand_eye=hand_eye)
            capture = self._capture_scene_once(run_id=scan_id, phase_label="placement_scan")
            requested = str(self.get_parameter("target_item_id").value or "").strip()
            item = self._item_catalog().resolve(requested)
            if item is None:
                item = self._item_catalog().items["yellow_block"]
            label = self._match_box_label_once(
                run_id=scan_id,
                item_id=item.item_id,
                capture_response=capture,
                base_to_camera=state["base_to_camera"],
                require_complete=False,
            )

            output_dir = self._placement_scan_viz_dir()
            output_dir.mkdir(parents=True, exist_ok=True)
            color = color_msg_to_bgr(capture.color_image)
            overlay = color.copy()
            detections = list(label.get("diagnostics", {}).get("detections") or [])
            for detection in detections:
                bbox = list(detection.get("bbox_xywh") or [])
                if len(bbox) != 4:
                    continue
                x, y, width, height = (int(value) for value in bbox)
                cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(
                    overlay,
                    str(detection.get("item_id") or ""),
                    (x, max(18, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 80, 255),
                    2,
                    cv2.LINE_AA,
                )
            depth = depth_msg_to_meters(capture.depth_image)
            color_path = output_dir / "color.png"
            overlay_path = output_dir / "overlay.png"
            depth_path = output_dir / "depth.png"
            if not cv2.imwrite(str(color_path), color):
                raise RuntimeError(f"failed to write placement scan image: {color_path}")
            if not cv2.imwrite(str(overlay_path), overlay):
                raise RuntimeError(f"failed to write placement scan overlay: {overlay_path}")
            if not cv2.imwrite(str(depth_path), self._depth_preview(depth)):
                raise RuntimeError(f"failed to write placement depth preview: {depth_path}")

            payload = {
                "success": bool(label.get("complete")),
                "validation_message": str(label.get("message") or ""),
                "scan_id": scan_id,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "robot_pose_mm_deg": self._pose_debug_dict(state["current_pose"]),
                "capture": self._capture_debug_dict(capture),
                "label_match": label,
                "images": {
                    "color": "color.png",
                    "overlay": "overlay.png",
                    "depth": "depth.png",
                },
                "motion_command_sent": False,
                "gripper_command_sent": False,
            }
            self._write_json_file(output_dir / "latest.json", payload)
            self._result_pub.publish(String(data=json_dumps(payload)))
            response.success = True
            response.message = (
                (
                    f"placement scan ok: scan_id={scan_id} "
                    f"labels={label['detected_label_count']} slot={label['slot_index']}"
                )
                if bool(label.get("complete"))
                else (
                    f"placement scan captured but validation failed: scan_id={scan_id} "
                    f"labels={label['detected_label_count']}/6; {label['message']}"
                )
            )
            self._publish_status(
                f"idle: placement scan captured scan_id={scan_id} "
                f"complete={bool(label.get('complete'))}"
            )
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            self._publish_status(f"placement scan failed: {exc}")
        finally:
            with self._run_lock:
                self._scan_active = False
        return response

    def _handle_scan_placement_multi_view_service(self, _request, response):
        if not bool(self.get_parameter("base_multiview_enabled").value):
            response.success = False
            response.message = (
                "base single-pass scan is disabled; clear the forward scan lane "
                "and enable base_multiview_enabled"
            )
            return response
        with self._run_lock:
            if self._scan_active:
                response.success = False
                response.message = "placement scan is already running"
                return response
            if self._run_thread is not None and self._run_thread.is_alive():
                response.success = False
                response.message = "cannot scan placement area while pipeline is running"
                return response
            if self._pending_confirmation is not None:
                response.success = False
                response.message = "cannot scan while grasp confirmation is pending"
                return response
            self._scan_active = True
            self._stop_requested = False

        scan_id = f"placement-multiview-{int(time.time() * 1000)}"
        output_dir = self._placement_scan_viz_dir()
        views: list[dict[str, object]] = []
        movements: list[dict[str, object]] = []
        current_offset = 0.0
        start_odom: tuple[float, float, float] | None = None
        base_returned = False
        try:
            self._publish_status(f"scanning_placement_multiview: scan_id={scan_id}")
            start_odom = self._base_odom_snapshot()
            _, _, hand_eye, _ = self._build_runtime()
            state = self._read_robot_state_snapshot(hand_eye=hand_eye)
            requested = str(self.get_parameter("target_item_id").value or "").strip()
            item = self._item_catalog().resolve(requested)
            if item is None:
                item = self._item_catalog().items["yellow_block"]
            output_dir.mkdir(parents=True, exist_ok=True)

            def capture(view_name: str) -> None:
                views.append(
                    self._capture_placement_scan_view(
                        scan_id=scan_id,
                        view_name=view_name,
                        offset_from_start_m=current_offset,
                        item_id=item.item_id,
                        base_to_camera=state["base_to_camera"],
                        output_dir=output_dir,
                    )
                )

            def detected_ids() -> set[str]:
                found: set[str] = set()
                for view in views:
                    diagnostics = dict(
                        dict(view.get("label_match") or {}).get("diagnostics") or {}
                    )
                    found.update(
                        str(value)
                        for value in list(
                            diagnostics.get("detected_item_ids_left_to_right") or []
                        )
                    )
                return found

            def order_is_complete() -> bool:
                try:
                    self._fuse_single_pass_label_map(
                        views=views,
                        item_id=item.item_id,
                    )
                    return True
                except RuntimeError:
                    return False

            def move(distance_m: float) -> None:
                nonlocal current_offset
                movement = self._move_base_for_scan(distance_m)
                current_offset += float(movement["traveled_m"])
                movement["offset_after_move_m"] = float(current_offset)
                movements.append(movement)
                time.sleep(
                    max(
                        0.0,
                        float(self.get_parameter("base_multiview_settle_s").value),
                    )
                )

            capture("start")
            step_m = abs(
                float(self.get_parameter("base_multiview_offset_m").value)
            )
            max_travel_m = abs(
                float(self.get_parameter("base_multiview_max_travel_m").value)
            )
            max_views = max(
                1, int(self.get_parameter("base_multiview_max_views").value)
            )
            if step_m <= 0.01:
                raise RuntimeError("base_multiview_offset_m must be greater than 0.01")
            while not order_is_complete() and len(views) < max_views:
                remaining_travel = max_travel_m - current_offset
                if remaining_travel <= 0.01:
                    break
                move(min(step_m, remaining_travel))
                capture(f"forward_{len(views):02d}")

            end_odom = self._base_odom_snapshot()
            base_returned = False

            validation_message = ""
            fused_map: dict[str, object] = {}
            try:
                fused_map = self._fuse_single_pass_label_map(
                    views=views,
                    item_id=item.item_id,
                )
                validation_ok = True
            except Exception as exc:
                validation_ok = False
                validation_message = str(exc)

            target_slot = (
                int(dict(fused_map.get("item_to_slot_index") or {})[item.item_id])
                if validation_ok
                else -1
            )
            target_confidence = (
                float(
                    dict(fused_map.get("item_to_confidence") or {})[
                        item.item_id
                    ]
                )
                if validation_ok
                else 0.0
            )
            payload = {
                "success": validation_ok,
                "validation_message": validation_message,
                "scan_mode": "base_single_pass",
                "scan_id": scan_id,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "created_at_unix_s": time.time(),
                "robot_pose_mm_deg": self._pose_debug_dict(
                    state["current_pose"]
                ),
                "base_odom_origin": {
                    "x_m": end_odom[0],
                    "y_m": end_odom[1],
                    "yaw_rad": end_odom[2],
                },
                "base_odom_scan_start": {
                    "x_m": start_odom[0],
                    "y_m": start_odom[1],
                    "yaw_rad": start_odom[2],
                },
                "base_returned_to_start": base_returned,
                "views": views,
                "movements": movements,
                "fused_map": fused_map,
                "label_match": {
                    "success": validation_ok,
                    "complete": validation_ok,
                    "message": (
                        "multi-view six-label map verified"
                        if validation_ok
                        else validation_message
                    ),
                    "expected_item_id": item.item_id,
                    "matched_item_id": item.item_id if validation_ok else "",
                    "confidence": target_confidence,
                    "slot_index": target_slot,
                    "detected_label_count": (
                        len(
                            list(
                                fused_map.get(
                                    "detected_item_ids_left_to_right"
                                )
                                or []
                            )
                        )
                        if validation_ok
                        else len(detected_ids())
                    ),
                    "has_box_center": False,
                    "box_center_base_m": [0.0, 0.0, 0.0],
                    "diagnostics": fused_map,
                },
                "images": (
                    dict(views[0]["images"])
                    if views
                    else {}
                ),
                "motion_command_sent": bool(movements),
                "gripper_command_sent": False,
            }
            self._write_json_file(output_dir / "latest.json", payload)
            self._result_pub.publish(String(data=json_dumps(payload)))
            response.success = True
            response.message = (
                (
                    f"multi-view placement scan ok: scan_id={scan_id}; "
                    "six labels ordered without transparent-box depth"
                )
                if validation_ok
                else (
                    f"multi-view images captured but validation failed: "
                    f"{validation_message}"
                )
            )
            self._publish_status(
                f"idle: multi-view placement scan captured scan_id={scan_id} "
                f"complete={validation_ok}"
            )
        except Exception as exc:
            self._request_base_scan_stop()
            response.success = False
            response.message = str(exc)
            self._publish_status(
                f"single-pass placement scan stopped at current position: {exc}"
            )
        finally:
            self._request_base_scan_stop()
            with self._run_lock:
                self._scan_active = False
        return response

    def _handle_scan_and_align_placement_target_service(self, _request, response):
        """Scan for the selected box label, reversing after a miss instead of walking on."""
        if not bool(self.get_parameter("base_target_alignment_enabled").value):
            response.success = False
            response.message = (
                "target-box scan is disabled; clear the forward scan lane and "
                "enable base_target_alignment_enabled"
            )
            return response
        with self._run_lock:
            if self._scan_active:
                response.success = False
                response.message = "placement scan or alignment is already running"
                return response
            if self._run_thread is not None and self._run_thread.is_alive():
                response.success = False
                response.message = "cannot align target box while pipeline is running"
                return response
            self._scan_active = True
            self._stop_requested = False

        scan_id = f"placement-target-{int(time.time() * 1000)}"
        output_dir = self._placement_scan_viz_dir()
        views: list[dict[str, object]] = []
        movements: list[dict[str, object]] = []
        current_offset = 0.0
        try:
            requested = str(self.get_parameter("target_item_id").value or "").strip()
            item = self._item_catalog().resolve(requested)
            if item is None:
                raise RuntimeError("select one of the six target items first")
            start_odom = self._base_odom_snapshot()
            _, _, hand_eye, _ = self._build_runtime()
            state = self._read_robot_state_snapshot(hand_eye=hand_eye)
            output_dir.mkdir(parents=True, exist_ok=True)
            step_m = abs(float(self.get_parameter("base_multiview_offset_m").value))
            fine_step_m = abs(
                float(self.get_parameter("base_target_fine_step_m").value)
            )
            max_travel_m = abs(
                float(self.get_parameter("base_multiview_max_travel_m").value)
            )
            max_views = max(
                1, int(self.get_parameter("base_multiview_max_views").value)
            )
            center_limit = max(
                0.02,
                min(
                    0.45,
                    float(
                        self.get_parameter(
                            "base_target_center_tolerance_norm"
                        ).value
                    ),
                ),
            )
            taught_map = load_mapping_for_item(item.item_id)
            taught_align_u_px = (
                None if taught_map is None else taught_map.align_u_px
            )
            if taught_align_u_px is not None:
                self.get_logger().info(
                    f"placement scan align target is taught center "
                    f"u={float(taught_align_u_px):.1f}px, not image center"
                )
            if step_m <= 0.01 or fine_step_m <= 0.01:
                raise RuntimeError("base_multiview_offset_m must be greater than 0.01")

            def label_align_u_norm(width: float) -> float:
                if taught_map is not None:
                    taught_norm = taught_map.alignment_u_norm(width)
                    if taught_norm is not None:
                        return taught_norm
                return 0.5

            def label_center_error(label: dict[str, object]) -> float:
                bbox = list(label.get("bbox_xywh") or [])
                width = float(label.get("color_width") or 0.0)
                if len(bbox) != 4 or width <= 1.0:
                    return float("inf")
                u_px = float(bbox[0]) + (0.5 * float(bbox[2]))
                return abs(u_px / width - label_align_u_norm(width))

            def label_is_centered(label: dict[str, object]) -> bool:
                center_error = label_center_error(label)
                return bool(
                    str(label.get("matched_item_id") or "") == item.item_id
                    and float(label.get("confidence") or 0.0)
                    >= float(self.get_parameter("label_match_threshold").value)
                    and math.isfinite(center_error)
                    and center_error <= center_limit
                )

            def preview_label_probe(**kwargs):
                matched = self._match_box_label_from_preview(
                    run_id=scan_id,
                    item_id=item.item_id,
                    **kwargs,
                )
                if matched is None:
                    return None
                label, _color, sequence = matched
                return label, sequence

            try:
                max_retries = self._recognition_max_retries("placement_scan_max_retries")
            except Exception:
                max_retries = 2

            def reverse_to_start(offset: float, *, reason: str) -> float:
                if abs(float(offset)) <= 0.02:
                    return 0.0
                self._publish_status(
                    f"placement_scan_return: reason={reason} reverse_m={float(offset):.3f}"
                )
                movement = self._move_base_for_scan(-abs(float(offset)))
                movement["offset_after_move_m"] = 0.0
                movements.append(movement)
                return 0.0

            def restore_placement_observe() -> None:
                try:
                    observe = self._placement_observe_pose_values()
                except Exception:
                    return
                try:
                    self._execute_named_pose(
                        name="placement_observation",
                        position_m=(
                            observe[0] / 1000.0,
                            observe[1] / 1000.0,
                            observe[2] / 1000.0,
                        ),
                        rpy_deg=(observe[3], observe[4], observe[5]),
                        speed_percent=float(self.get_parameter("observation_speed").value),
                        open_gripper_first=False,
                        timeout_s=45.0,
                    )
                except Exception as exc:
                    self.get_logger().warning(
                        f"could not restore placement observation pose: {exc}"
                    )

            selected_view: dict[str, object] | None = None
            best_target_view: dict[str, object] | None = None
            best_target_error = float("inf")
            attempt_records: list[dict[str, object]] = []
            for attempt in range(1 + max_retries):
                scan_sign = 1.0
                last_center_error: float | None = None
                had_target = False
                worsening_streak = 0
                target_miss_streak = 0
                selected_view = None
                for view_index in range(max_views):
                    view_name = (
                        "start"
                        if not views
                        else f"{'rev' if scan_sign < 0 else 'forward'}_{len(views):02d}"
                    )
                    view = self._capture_placement_scan_view(
                        scan_id=scan_id,
                        view_name=view_name,
                        offset_from_start_m=current_offset,
                        item_id=item.item_id,
                        base_to_camera=state["base_to_camera"],
                        output_dir=output_dir,
                    )
                    label = dict(view.get("label_match") or {})
                    capture = dict(view.get("capture") or {})
                    if not label.get("color_width"):
                        label["color_width"] = float(capture.get("color_width") or 0.0)
                    center_error_norm = label_center_error(label)
                    label["center_error_norm"] = center_error_norm
                    label["centered"] = label_is_centered(label)
                    view["label_match"] = label
                    view["attempt"] = attempt + 1
                    views.append(view)
                    if bool(label["centered"]):
                        selected_view = view
                        break
                    target_seen = bool(
                        str(label.get("matched_item_id") or "") == item.item_id
                        and float(label.get("confidence") or 0.0)
                        >= float(self.get_parameter("label_match_threshold").value)
                        and math.isfinite(center_error_norm)
                    )
                    if target_seen:
                        had_target = True
                        target_miss_streak = 0
                        if center_error_norm < best_target_error:
                            best_target_error = float(center_error_norm)
                            best_target_view = view
                        if (
                            scan_sign > 0.0
                            and
                            last_center_error is not None
                            and center_error_norm > last_center_error + 0.02
                        ):
                            worsening_streak += 1
                        else:
                            worsening_streak = 0
                        # A single label-box jump is common with glossy box
                        # covers.  Only reverse after two consecutive views
                        # confirm that forward motion is making alignment worse.
                        if worsening_streak >= 2:
                            scan_sign = -1.0
                        last_center_error = center_error_norm
                        next_step_m = scan_sign * fine_step_m
                    elif had_target:
                        target_miss_streak += 1
                        # Do not react to one missed frame.  If the target is
                        # genuinely lost after being seen, make one controlled
                        # reverse search rather than oscillating every frame.
                        if scan_sign > 0.0 and target_miss_streak >= 2:
                            scan_sign = -1.0
                        next_step_m = scan_sign * fine_step_m
                    else:
                        next_step_m = step_m
                    if next_step_m > 0.0:
                        remaining = max_travel_m - current_offset
                        if remaining <= 0.01:
                            break
                        move_m = min(next_step_m, remaining)
                    else:
                        if current_offset <= 0.02:
                            break
                        move_m = -min(abs(next_step_m), current_offset)
                    if bool(
                        self.get_parameter("continuous_search_stop_on_center").value
                    ) and bool(self.get_parameter("continuous_search_enabled").value):
                        movement, _trigger = self._move_base_for_scan_until_preview_trigger(
                            move_m,
                            run_id=scan_id,
                            preview_probe=preview_label_probe,
                            should_stop=label_is_centered,
                        )
                    else:
                        movement = self._move_base_for_scan(move_m)
                    current_offset += float(movement["traveled_m"])
                    current_offset = max(0.0, float(current_offset))
                    movement["offset_after_move_m"] = float(current_offset)
                    movements.append(movement)
                    time.sleep(
                        max(
                            0.0,
                            float(self.get_parameter("base_multiview_settle_s").value),
                        )
                    )
                if selected_view is not None:
                    break
                attempt_records.append(
                    {
                        "attempt": attempt + 1,
                        "offset_m": float(current_offset),
                        "had_target": had_target,
                    }
                )
                # If the target was seen, its best view is more useful than
                # restarting the entire lane from zero.  Stop there below and
                # fail closed if it still is not inside the alignment window.
                if attempt < max_retries and not had_target:
                    current_offset = reverse_to_start(
                        current_offset,
                        reason=f"placement_retry_{attempt + 1}",
                    )
                    restore_placement_observe()
            if selected_view is None and best_target_view is not None:
                best_offset = float(best_target_view.get("offset_from_start_m") or 0.0)
                return_delta = best_offset - current_offset
                if abs(return_delta) > 0.02:
                    self._publish_status(
                        "placement_scan_best_view_return: "
                        f"delta_m={return_delta:.3f} error={best_target_error:.3f}"
                    )
                    movement = self._move_base_for_scan(return_delta)
                    current_offset += float(movement["traveled_m"])
                    current_offset = max(0.0, float(current_offset))
                    movement["offset_after_move_m"] = float(current_offset)
                    movement["target_alignment_best_view_return"] = True
                    movements.append(movement)
            self._last_placement_scan_travel_m = float(current_offset)
            if selected_view is None and best_target_view is None and current_offset > 0.02:
                current_offset = reverse_to_start(
                    current_offset,
                    reason="placement_scan_failed",
                )

            aligned_odom = self._base_odom_snapshot()
            selected_label = dict(selected_view.get("label_match") or {}) if selected_view else {}
            validation_ok = selected_view is not None
            selected_width = float(
                dict(selected_label).get("color_width")
                or dict((selected_view or {}).get("capture") or {}).get("color_width")
                or 0.0
            )
            alignment = {
                "success": validation_ok,
                "item_id": item.item_id,
                # Slot order is no longer used for the common fixed TCP path;
                # keep a valid index for the executor's plan contract.
                "slot_index": 0,
                "selected_view_name": str(selected_view.get("view_name") or "") if selected_view else "",
                "selected_view_offset_m": float(selected_view.get("offset_from_start_m") or 0.0) if selected_view else 0.0,
                "selected_confidence": float(selected_label.get("confidence") or 0.0),
                "selected_center_error_norm": float(selected_label.get("center_error_norm") or 0.0),
                "center_tolerance_norm": center_limit,
                "align_u_px": (
                    None if taught_align_u_px is None else float(taught_align_u_px)
                ),
                "align_u_norm": (
                    label_align_u_norm(selected_width)
                    if selected_width > 1.0
                    else None
                ),
                "align_source": (
                    "taught_center_sample"
                    if taught_align_u_px is not None
                    else "image_center"
                ),
                "aligned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            payload = {
                "success": validation_ok,
                "validation_message": (
                    "target label centered and base stopped"
                    if validation_ok
                    else (
                        f"target label {item.item_id} was not centered within "
                        f"{len(views)} views / {current_offset:.3f}m"
                    )
                ),
                "scan_mode": "base_target_single_pass",
                "scan_id": scan_id,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "created_at_unix_s": time.time(),
                "robot_pose_mm_deg": self._pose_debug_dict(state["current_pose"]),
                "base_odom_scan_start": {
                    "x_m": start_odom[0], "y_m": start_odom[1], "yaw_rad": start_odom[2],
                },
                "base_odom_origin": {
                    "x_m": aligned_odom[0], "y_m": aligned_odom[1], "yaw_rad": aligned_odom[2],
                },
                "base_returned_to_start": current_offset <= 0.02,
                "placement_scan_attempts": attempt_records,
                "views": views,
                "movements": movements,
                "target_alignment": alignment,
                "label_match": selected_label,
                "images": dict(selected_view.get("images") or {}) if selected_view else (dict(views[-1].get("images") or {}) if views else {}),
                "motion_command_sent": bool(movements),
                "gripper_command_sent": False,
            }
            self._write_json_file(output_dir / "latest.json", payload)
            self._result_pub.publish(String(data=json_dumps(payload)))
            response.success = validation_ok
            response.message = (
                f"Scout stopped aligned to {item.item_id}; "
                f"center_error={float(selected_label.get('center_error_norm') or 0.0):.3f}"
                if validation_ok
                else payload["validation_message"]
            )
        except Exception as exc:
            try:
                if current_offset > 0.02:
                    self._move_base_for_scan(-abs(current_offset))
                    current_offset = 0.0
            except Exception:
                pass
            response.success = False
            response.message = str(exc)
        finally:
            self._request_base_scan_stop()
            with self._run_lock:
                self._scan_active = False
        return response

    def _handle_align_placement_target_service(self, _request, response):
        if not bool(self.get_parameter("base_alignment_enabled").value):
            response.success = False
            response.message = (
                "base target alignment is disabled; confirm the reverse path "
                "is clear and enable base_alignment_enabled"
            )
            return response
        with self._run_lock:
            if self._scan_active:
                response.success = False
                response.message = "placement scan or alignment is already running"
                return response
            if self._run_thread is not None and self._run_thread.is_alive():
                response.success = False
                response.message = "cannot align base while pipeline is running"
                return response
            self._scan_active = True
            self._stop_requested = False

        try:
            path = self._placement_scan_viz_dir() / "latest.json"
            if not path.is_file():
                raise RuntimeError("no placement scan exists")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if (
                not bool(payload.get("success"))
                or str(payload.get("scan_mode") or "") != "base_single_pass"
            ):
                raise RuntimeError("latest single-pass placement scan is not valid")
            created_at = float(payload.get("created_at_unix_s") or 0.0)
            max_age = float(self.get_parameter("cached_box_map_max_age_s").value)
            if created_at <= 0.0 or time.time() - created_at > max_age:
                raise RuntimeError("latest placement scan is stale; scan again")

            expected_odom = dict(payload.get("base_odom_origin") or {})
            current_odom = self._base_odom_snapshot()
            position_error = math.hypot(
                current_odom[0] - float(expected_odom["x_m"]),
                current_odom[1] - float(expected_odom["y_m"]),
            )
            yaw_error_deg = abs(
                math.degrees(
                    math.atan2(
                        math.sin(
                            current_odom[2] - float(expected_odom["yaw_rad"])
                        ),
                        math.cos(
                            current_odom[2] - float(expected_odom["yaw_rad"])
                        ),
                    )
                )
            )
            if (
                position_error
                > float(
                    self.get_parameter(
                        "cached_box_map_position_tolerance_m"
                    ).value
                )
                or yaw_error_deg
                > float(
                    self.get_parameter(
                        "cached_box_map_yaw_tolerance_deg"
                    ).value
                )
            ):
                raise RuntimeError(
                    "Scout moved since placement scan: "
                    f"position_error={position_error:.3f}m "
                    f"yaw_error={yaw_error_deg:.2f}deg; scan again"
                )

            requested = str(
                self.get_parameter("target_item_id").value or ""
            ).strip()
            item = self._item_catalog().resolve(requested)
            if item is None:
                raise RuntimeError("select one of the six target items first")
            fused = dict(payload.get("fused_map") or {})
            item_observations = dict(
                fused.get("item_observations") or {}
            )
            observations = [
                dict(value)
                for value in list(item_observations.get(item.item_id) or [])
            ]
            if not observations:
                raise RuntimeError(
                    f"placement scan has no observation for {item.item_id}"
                )

            views_by_name = {
                str(dict(view).get("view_name") or ""): dict(view)
                for view in list(payload.get("views") or [])
            }
            ranked: list[tuple[float, float, dict[str, object]]] = []
            for observation in observations:
                view = views_by_name.get(
                    str(observation.get("view_name") or ""), {}
                )
                width = float(
                    dict(view.get("capture") or {}).get("color_width") or 0.0
                )
                if width <= 0.0:
                    continue
                center_error = abs(
                    float(observation.get("bbox_center_x_px") or 0.0)
                    - (0.5 * width)
                )
                confidence = float(observation.get("confidence") or 0.0)
                ranked.append((center_error / width, -confidence, observation))
            if not ranked:
                raise RuntimeError("target observations have no valid image width")
            selected = min(ranked, key=lambda value: (value[0], value[1]))[2]
            target_offset = float(selected["offset_from_start_m"])
            view_offsets = [
                float(dict(view).get("offset_from_start_m") or 0.0)
                for view in list(payload.get("views") or [])
            ]
            current_offset = max(view_offsets) if view_offsets else 0.0
            requested_delta = target_offset - current_offset
            movements: list[dict[str, object]] = []
            while abs(target_offset - current_offset) > 0.012:
                remaining = target_offset - current_offset
                segment = math.copysign(min(0.40, abs(remaining)), remaining)
                movement = self._move_base_for_scan(segment)
                current_offset += float(movement["traveled_m"])
                movement["offset_after_move_m"] = float(current_offset)
                movement["target_alignment_segment"] = True
                movements.append(movement)
                time.sleep(
                    max(
                        0.0,
                        float(
                            self.get_parameter(
                                "base_multiview_settle_s"
                            ).value
                        ),
                    )
                )

            aligned_odom = self._base_odom_snapshot()
            alignment = {
                "success": True,
                "item_id": item.item_id,
                "slot_index": int(
                    dict(fused.get("item_to_slot_index") or {})[item.item_id]
                ),
                "selected_view_name": str(selected.get("view_name") or ""),
                "selected_view_offset_m": target_offset,
                "requested_delta_m": requested_delta,
                "final_offset_from_scan_start_m": current_offset,
                "selected_bbox_center_x_px": float(
                    selected.get("bbox_center_x_px") or 0.0
                ),
                "selected_confidence": float(
                    selected.get("confidence") or 0.0
                ),
                "movements": movements,
                "aligned_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            payload["target_alignment"] = alignment
            # Future operations validate against the new, aligned base pose.
            payload["base_odom_origin"] = {
                "x_m": aligned_odom[0],
                "y_m": aligned_odom[1],
                "yaw_rad": aligned_odom[2],
            }
            self._write_json_file(path, payload)
            self._result_pub.publish(String(data=json_dumps(payload)))
            response.success = True
            response.message = (
                f"Scout aligned to {item.item_id} slot "
                f"{alignment['slot_index']} using "
                f"{alignment['selected_view_name']}; fixed placement requires "
                "a separate one-shot release confirmation"
            )
        except Exception as exc:
            self._request_base_scan_stop()
            response.success = False
            response.message = str(exc)
        finally:
            self._request_base_scan_stop()
            with self._run_lock:
                self._scan_active = False
        return response

    def _validated_aligned_place_context(
        self,
        item_id: str,
    ) -> tuple[dict[str, object], int, float]:
        path = self._placement_scan_viz_dir() / "latest.json"
        if not path.is_file():
            raise RuntimeError("no placement scan exists")
        payload = json.loads(path.read_text(encoding="utf-8"))
        scan_mode = str(payload.get("scan_mode") or "")
        if not bool(payload.get("success")) or scan_mode not in {
            "base_single_pass",
            "base_target_single_pass",
        }:
            raise RuntimeError("latest target-aligned placement scan is not valid")
        created_at = float(payload.get("created_at_unix_s") or 0.0)
        max_age = float(self.get_parameter("cached_box_map_max_age_s").value)
        if created_at <= 0.0 or time.time() - created_at > max_age:
            raise RuntimeError("latest placement scan is stale; scan and align again")

        alignment = dict(payload.get("target_alignment") or {})
        if not bool(alignment.get("success")):
            raise RuntimeError("target box has not been aligned")
        aligned_item_id = str(alignment.get("item_id") or "")
        if aligned_item_id != item_id:
            raise RuntimeError(
                f"base is aligned to {aligned_item_id or 'unknown'}, not {item_id}"
            )

        if scan_mode == "base_single_pass":
            fused = dict(payload.get("fused_map") or {})
            slots = dict(fused.get("item_to_slot_index") or {})
            if item_id not in slots:
                raise RuntimeError(f"placement scan has no slot for {item_id}")
            slot_index = int(slots[item_id])
            if slot_index != int(alignment.get("slot_index", -1)):
                raise RuntimeError("aligned slot does not match the verified label map")
        else:
            slot_index = int(alignment.get("slot_index", -1))
            if not 0 <= slot_index < 6:
                raise RuntimeError("target alignment has an invalid place-plan slot")

        expected_odom = dict(payload.get("base_odom_origin") or {})
        current_odom = self._base_odom_snapshot()
        position_error = math.hypot(
            current_odom[0] - float(expected_odom["x_m"]),
            current_odom[1] - float(expected_odom["y_m"]),
        )
        yaw_error_deg = abs(
            math.degrees(
                math.atan2(
                    math.sin(
                        current_odom[2] - float(expected_odom["yaw_rad"])
                    ),
                    math.cos(
                        current_odom[2] - float(expected_odom["yaw_rad"])
                    ),
                )
            )
        )
        position_limit = float(
            self.get_parameter("cached_box_map_position_tolerance_m").value
        )
        yaw_limit = float(
            self.get_parameter("cached_box_map_yaw_tolerance_deg").value
        )
        if position_error > position_limit or yaw_error_deg > yaw_limit:
            raise RuntimeError(
                "Scout moved after target alignment: "
                f"position_error={position_error:.3f}m "
                f"yaw_error={yaw_error_deg:.2f}deg; align again"
            )
        confidence = float(alignment.get("selected_confidence") or 0.0)
        return payload, slot_index, confidence

    def _place_plan_from_poses(
        self,
        *,
        item_id: str,
        slot_index: int,
        label_confidence: float,
        poses: dict[str, tuple[float, ...]],
    ) -> PlacePlan:
        catalog = self._item_catalog()
        message = PlacePlan()
        message.item_id = item_id
        message.slot_index = int(slot_index)
        message.approach_pose = self._pose6d_from_mm_deg(poses["approach"])
        message.release_pose = self._pose6d_from_mm_deg(poses["release"])
        message.retreat_pose = self._pose6d_from_mm_deg(poses["retreat"])
        message.box_outer_size_m = [float(value) for value in catalog.box.outer_size_m]
        message.label_verified = True
        message.label_confidence = float(label_confidence)
        return message

    def _build_live_or_fixed_place_plan(
        self,
        *,
        item_id: str,
        label_confidence: float,
        slot_index: int,
    ) -> tuple[PlacePlan, str, dict[str, object]]:
        """Prefer taught (u, v)->XY; then label depth; then calibrated TCP."""
        try:
            _, _, hand_eye, _ = self._build_runtime()
            state = self._read_robot_state_snapshot(hand_eye=hand_eye)
            capture = self._capture_scene_once(
                run_id=f"aligned-place-{item_id}",
                phase_label="aligned_place_label",
            )
            label = self._match_box_label_once(
                run_id=f"aligned-place-{item_id}",
                item_id=item_id,
                capture_response=capture,
                base_to_camera=state["base_to_camera"],
                require_complete=False,
            )
            bbox = [int(value) for value in list(label.get("bbox_xywh") or [])]
            if (
                str(label.get("matched_item_id") or "") != item_id
                or len(bbox) != 4
                or min(bbox[2], bbox[3]) <= 1
            ):
                raise RuntimeError(str(label.get("message") or "target label not confirmed"))
            u_px = float(bbox[0]) + (0.5 * float(bbox[2]))
            v_px = float(bbox[1]) + (0.5 * float(bbox[3]))
            confidence = float(label.get("confidence") or label_confidence)
            mapping = load_mapping_for_item(item_id)
            if mapping is not None:
                if not mapping.in_domain(u_px, v_px):
                    raise RuntimeError(
                        f"label center ({u_px:.1f},{v_px:.1f}) is outside the taught "
                        f"u{list(mapping.u_px_range)} v{list(mapping.v_px_range)} window"
                    )
                poses = mapping.poses_mm_deg(u_px, v_px)
                self.get_logger().info(
                    f"place XY from taught map: u={u_px:.1f} v={v_px:.1f} -> "
                    f"X={poses['release'][0]:.1f} Y={poses['release'][1]:.1f} "
                    f"Z={poses['release'][2]:.1f} "
                    f"rpy={list(poses['release'][3:])} "
                    f"rms={mapping.fit_rms_xy_mm:.1f}mm"
                )
                return (
                    self._place_plan_from_poses(
                        item_id=item_id,
                        slot_index=slot_index,
                        label_confidence=confidence,
                        poses=poses,
                    ),
                    "taught_uv_xy_map",
                    {
                        "source": "taught_uv_xy_map",
                        "u_px": u_px,
                        "v_px": v_px,
                        "release_xy_mm": [poses["release"][0], poses["release"][1]],
                        "label_bbox_xywh": bbox,
                        "label_confidence": confidence,
                        "fit_rms_xy_mm": mapping.fit_rms_xy_mm,
                    },
                )
            detection = LabelDetection(
                item_id=item_id,
                confidence=confidence,
                bbox_xywh=(bbox[0], bbox[1], bbox[2], bbox[3]),
                method=str(dict(label.get("diagnostics") or {}).get("method") or "label"),
            )
            box_center = self._item_catalog().localize_single_box_from_label(
                depth_meters=depth_msg_to_meters(capture.depth_image),
                camera_k=list(capture.camera_info.k),
                base_to_camera=state["base_to_camera"],
                detection=detection,
            )
            poses = self._item_catalog().build_vision_place_poses_mm_deg(
                item_id, box_center
            )
            return (
                self._place_plan_from_poses(
                    item_id=item_id,
                    slot_index=slot_index,
                    label_confidence=confidence,
                    poses=poses,
                ),
                "label_depth_xy",
                {
                    "source": "label_depth_xy",
                    "box_center_base_m": [float(value) for value in box_center],
                    "label_bbox_xywh": bbox,
                    "label_confidence": confidence,
                },
            )
        except Exception as exc:
            self.get_logger().warning(
                f"live place pose unavailable ({exc}); using calibrated TCP"
            )
            return (
                self._build_base_aligned_place_plan_message(
                    item_id=item_id,
                    label_confidence=label_confidence,
                    slot_index=slot_index,
                ),
                "base_aligned_fixed_tcp_calibration",
                {"source": "calibrated_fallback", "error": str(exc)},
            )

    def _handle_execute_aligned_place_service(self, _request, response):
        if not bool(self.get_parameter("base_aligned_place_enabled").value):
            response.success = False
            response.message = (
                "fixed base-aligned placement is disabled; acknowledge release "
                "safety and enable base_aligned_place_enabled"
            )
            return response
        with self._run_lock:
            if self._scan_active:
                response.success = False
                response.message = "placement scan/alignment is already running"
                return response
            if self._run_thread is not None and self._run_thread.is_alive():
                response.success = False
                response.message = "cannot place while pipeline is running"
                return response
            self._scan_active = True
            self._stop_requested = False

        try:
            requested = str(
                self.get_parameter("target_item_id").value or ""
            ).strip()
            item = self._item_catalog().resolve(requested)
            if item is None:
                raise RuntimeError("select one of the six target items first")
            scan_payload, slot_index, confidence = (
                self._validated_aligned_place_context(item.item_id)
            )
            plan, localization_source, live_diagnostics = (
                self._build_live_or_fixed_place_plan(
                    item_id=item.item_id,
                    label_confidence=confidence,
                    slot_index=slot_index,
                )
            )
            run_id = new_run_id("aligned-place")

            validate_request = ExecutePlacePlan.Request()
            validate_request.run_id = run_id
            validate_request.execute = False
            validate_request.move_home_after = False
            validate_request.plan = plan
            validation_response = self._call_client(
                self._execute_place_client,
                validate_request,
                timeout_s=45.0,
            )
            if not validation_response.success:
                raise RuntimeError(
                    "fixed place validation failed: "
                    f"{validation_response.message}"
                )

            execute_request = ExecutePlacePlan.Request()
            execute_request.run_id = run_id
            execute_request.execute = True
            execute_request.move_home_after = False
            execute_request.plan = plan
            execute_response = self._call_client(
                self._execute_place_client,
                execute_request,
                timeout_s=180.0,
            )
            if not execute_response.success:
                raise RuntimeError(
                    f"fixed place execution failed: {execute_response.message}"
                )
            execution = json.loads(execute_response.execution_json)
            post_place_home, post_place_base_advance = (
                self._return_home_and_advance_base_after_place(
                    run_id=run_id,
                    item_id=item.item_id,
                )
            )
            record = {
                "success": True,
                "run_id": run_id,
                "item_id": item.item_id,
                "slot_index": slot_index,
                "label_confidence": confidence,
                "executed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "localization_source": localization_source,
                "live_place_pose": live_diagnostics,
                "execution": execution,
                "post_place_home": post_place_home,
                "post_place_base_advance": post_place_base_advance,
            }
            scan_payload["aligned_place_execution"] = record
            path = self._placement_scan_viz_dir() / "latest.json"
            self._write_json_file(path, scan_payload)
            self._result_pub.publish(String(data=json_dumps(scan_payload)))
            response.success = True
            response.message = (
                f"aligned placement completed: {item.item_id} "
                f"slot={slot_index} source={localization_source} run_id={run_id}"
            )
        except Exception as exc:
            self.get_logger().error(f"aligned place failed: {exc}")
            response.success = False
            response.message = str(exc)
        finally:
            with self._run_lock:
                self._scan_active = False
        return response

    def _handle_stop_service(self, _request, response):
        # Stop both actuators. The base adapter continuously publishes several
        # zero-velocity commands, so this also covers an active multi-view scan.
        self._request_base_scan_stop()
        pending = self._peek_pending_confirmation()
        if pending is not None:
            self._set_pending_confirmation(None)
            summary = self._append_summary_line(
                str(pending.result_payload.get("summary") or ""),
                "pending confirmation cleared by stop request",
            )
            diagnostics = list(pending.result_payload.get("diagnostics") or [])
            diagnostics.append("stop requested while awaiting confirmation")
            payload = dict(pending.result_payload)
            payload["confirmed"] = False
            payload["execution"] = None
            self._finalize_pending_confirmation(
                pending=pending,
                result_payload=payload,
                status="stopped",
                summary=summary,
                diagnostics=diagnostics,
            )
            response.success = True
            response.message = f"pending confirmation cleared: run_id={pending.run_id}"
            return response

        self._stop_requested = True
        try:
            stop_response = self._call_client(self._stop_robot_client, StopRobot.Request(), timeout_s=5.0)
            response.success = bool(stop_response.success)
            response.message = str(stop_response.message)
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _start_background_run(self, prompt_override: str | None) -> tuple[bool, str]:
        with self._run_lock:
            if self._scan_active:
                return False, "placement scan is already running"
            if self._run_thread is not None and self._run_thread.is_alive():
                return False, "pipeline is already running"
            if self._pending_confirmation is not None:
                return (
                    False,
                    f"confirmation pending for run_id={self._pending_confirmation.run_id}; "
                    "call /grasp_pipeline/confirm or /grasp_pipeline/reject first",
                )
            prompt = prompt_override if prompt_override is not None else str(self.get_parameter("prompt").value or "").strip()
            auto_target = self._auto_target_from_card_enabled()
            if not prompt and not auto_target:
                return False, "prompt is empty; set parameter 'prompt' or publish to ~/run_prompt"
            if auto_target:
                prompt = "auto target card"
            worker = threading.Thread(
                target=self._run_pipeline_thread,
                args=(prompt,),
                daemon=True,
                name="distributed-grasp-orchestrator",
            )
            self._run_thread = worker
            self._run_id = new_run_id("grasp")
            self._stop_requested = False
            worker.start()
            return True, (
                "run accepted for automatic target-card identification"
                if auto_target
                else f"run accepted for prompt={prompt}"
            )

    def _handle_confirm_service(self, _request, response):
        with self._run_lock:
            if self._run_thread is not None and self._run_thread.is_alive():
                response.success = False
                response.message = "pipeline is already running"
                return response
            pending = self._pending_confirmation
            if pending is None:
                response.success = False
                response.message = "no pending confirmation"
                return response
            self._pending_confirmation = None
            self._stop_requested = False
            self._run_id = pending.run_id
            worker = threading.Thread(
                target=self._execute_confirmed_plan_thread,
                args=(pending,),
                daemon=True,
                name="distributed-grasp-confirm-executor",
            )
            self._run_thread = worker
            worker.start()
        response.success = True
        response.message = f"confirmation accepted: run_id={pending.run_id}"
        return response

    def _handle_reject_service(self, _request, response):
        pending = self._peek_pending_confirmation()
        if pending is None:
            response.success = False
            response.message = "no pending confirmation"
            return response
        self._set_pending_confirmation(None)
        summary = self._append_summary_line(
            str(pending.result_payload.get("summary") or ""),
            "execution cancelled by user",
        )
        diagnostics = list(pending.result_payload.get("diagnostics") or [])
        diagnostics.append("confirmation rejected by user")
        payload = dict(pending.result_payload)
        payload["confirmed"] = False
        payload["execution"] = None
        self._finalize_pending_confirmation(
            pending=pending,
            result_payload=payload,
            status="cancelled",
            summary=summary,
            diagnostics=diagnostics,
        )
        response.success = True
        response.message = f"confirmation rejected: run_id={pending.run_id}"
        return response

    def _execute_confirmed_plan_thread(self, pending: PendingConfirmation) -> None:
        status = "ok"
        summary = str(pending.result_payload.get("summary") or "")
        diagnostics = list(pending.result_payload.get("diagnostics") or [])
        result_payload = dict(pending.result_payload)
        try:
            self._publish_status(f"executing_plan: run_id={pending.run_id} confirmed=true")
            execution_payload = self._execute_grasp_and_optional_place(
                run_id=pending.run_id,
                plan=pending.plan,
                move_home_after=bool(pending.move_home_after),
                target_item_id=pending.target_item_id,
                hand_eye=pending.hand_eye,
                advance_base_after_grasp=bool(
                    dict(pending.request_payload.get("options") or {}).get(
                        "base_grasp_scan_enabled"
                    )
                ),
            )
            result_payload["execution"] = execution_payload
            result_payload["confirmed"] = True
            result_payload["status"] = "ok"
            result_payload["execution_message"] = "confirmed task execution completed"
            summary = self._append_summary_line(summary, "execution confirmed and completed")
        except Exception as exc:
            with self._run_lock:
                stop_requested = self._stop_requested
            if stop_requested:
                status = "stopped"
                summary = self._append_summary_line(summary, "execution stopped after confirmation")
                diagnostics.append("stop requested during confirmed execution")
            else:
                status = "failed"
                summary = self._append_summary_line(summary, str(exc))
            diagnostics.append(traceback.format_exc())
            result_payload["execution"] = result_payload.get("execution")
            result_payload["confirmed"] = True
        finally:
            with self._run_lock:
                self._run_thread = None
                self._run_id = None
            self._finalize_pending_confirmation(
                pending=pending,
                result_payload=result_payload,
                status=status,
                summary=summary,
                diagnostics=diagnostics,
            )

    def _run_pipeline_thread(self, prompt: str) -> None:
        run_id = self._run_id or new_run_id("grasp")
        status = "completed"
        summary = ""
        diagnostics: list[str] = []
        centering_logs: list[str] = []
        cycle_records: list[dict[str, object]] = []
        request_payload: dict[str, object] = {
            "run_id": run_id,
            "prompt": prompt,
            "started_at_unix_s": time.time(),
        }
        result_payload: dict[str, object] = {"run_id": run_id, "prompt": prompt}
        grasp_scan_payload: dict[str, object] | None = None
        auto_target_from_card = self._auto_target_from_card_enabled()
        base_grasp_scan_requested = auto_target_from_card or bool(
            self.get_parameter("base_grasp_scan_enabled").value
        )
        try:
            self._clear_grasp_card_exclusion()
            self._publish_candidate_validation_markers(
                validation_records=[],
                camera_frame="camera_color_optical_frame",
            )
            self._publish_status(f"preflight: run_id={run_id}")
            self._publish_status(f"moving_to_observation: run_id={run_id}")
            observe_pose = self._observe_pose_values()
            if bool(self.get_parameter("skip_observation_move").value):
                self._publish_status(f"observation_move_skipped: run_id={run_id}")
            else:
                self._execute_named_pose(
                    name="observation",
                    position_m=(observe_pose[0] / 1000.0, observe_pose[1] / 1000.0, observe_pose[2] / 1000.0),
                    rpy_deg=(observe_pose[3], observe_pose[4], observe_pose[5]),
                    speed_percent=float(self.get_parameter("observation_speed").value),
                    open_gripper_first=True,
                    timeout_s=45.0,
                )

            target_card_payload: dict[str, object] | None = None
            if auto_target_from_card:
                target_item, target_card_payload = self._identify_target_card_with_retries(
                    run_id=run_id
                )
                if target_item is None:
                    self._clear_grasp_card_exclusion()
                    status = "skipped_no_target_card"
                    summary = (
                        "target card not recognized after "
                        f"{int(target_card_payload.get('max_attempts') or 0)} "
                        "attempts; grasp skipped"
                    )
                    result_payload.update(
                        {
                            "status": status,
                            "summary": summary,
                            "target_card_identification": target_card_payload,
                            "diagnostics": diagnostics,
                        }
                    )
                    return
                # Publish the resolved identity for dashboards and external
                # observers, but every automatic run still performs a fresh
                # card capture and never trusts these retained values.
                self._remember_target_card_exclusion(target_card_payload)
                self.set_parameters(
                    [
                        Parameter("target_item_id", value=target_item.item_id),
                        Parameter("prompt", value=target_item.grasp_prompt),
                    ]
                )
            else:
                target_item = self._resolve_target_item(prompt)

            target_item_id = target_item.item_id if target_item is not None else ""
            if target_item is not None:
                prompt = target_item.grasp_prompt
                grasp_strategy = self._set_executor_strategy_for_item(target_item)
            else:
                grasp_strategy = ""

            options, config, hand_eye, planner = self._build_runtime()
            if target_item is not None:
                options["target_item_id"] = target_item.item_id
                options["prompt"] = prompt
                result_payload["prompt"] = prompt
                result_payload["target_item"] = {
                    "item_id": target_item.item_id,
                    "display_name": target_item.display_name,
                    "kind": target_item.kind,
                    "reference_image": str(target_item.reference_image),
                    "execution_strategy": grasp_strategy,
                    "source": (
                        "target_card" if auto_target_from_card else "manual"
                    ),
                }
            if target_card_payload is not None:
                result_payload["target_card_identification"] = target_card_payload
            if bool(self.get_parameter("place_after_grasp").value):
                if target_item is None:
                    raise RuntimeError(
                        "place_after_grasp requires target_item_id from the six-item catalog"
                    )
                self._placement_observe_pose_values()
                self._item_catalog().ensure_place_calibrated(
                    target_item.item_id,
                    require_slot_centers=not bool(
                        self.get_parameter("dynamic_box_localization").value
                    ),
                )
            if base_grasp_scan_requested and target_item is None:
                raise RuntimeError(
                    "base grasp scan requires target_item_id from the six-item catalog"
                )
            request_payload.update(
                {
                    "prompt": prompt,
                    "target_item_id": target_item_id,
                    "target_source": (
                        "target_card" if auto_target_from_card else "manual"
                    ),
                    "target_card_identification": target_card_payload,
                    "options": options,
                    "observe_pose": observe_pose,
                    "artifact_root": str(self._artifact_root_dir()),
                }
            )
            request_payload["confirm"] = bool(self.get_parameter("confirm").value)
            request_payload["execute"] = bool(self.get_parameter("execute").value)

            if base_grasp_scan_requested:
                scan_retries = self._recognition_max_retries("grasp_scan_max_retries")
                cycle = None
                grasp_scan_payload = None
                for scan_attempt in range(1 + scan_retries):
                    if scan_attempt > 0:
                        reverse_m = float(
                            (grasp_scan_payload or {}).get("final_offset_from_start_m") or 0.0
                        )
                        self._return_to_observation_for_retry(
                            run_id=run_id,
                            reverse_m=reverse_m,
                            reason=f"grasp_item_retry_{scan_attempt + 1}",
                        )
                    try:
                        cycle, grasp_scan_payload = self._run_base_grasp_target_scan(
                            run_id=run_id,
                            prompt=prompt,
                            target_item_id=target_item_id,
                            options=options,
                            hand_eye=hand_eye,
                        )
                    except RuntimeError as error:
                        if scan_attempt >= scan_retries:
                            raise
                        self.get_logger().warning(
                            f"grasp item scan attempt {scan_attempt + 1}/"
                            f"{1 + scan_retries} failed: {error}"
                        )
                        grasp_scan_payload = {
                            "final_offset_from_start_m": 0.0,
                            "views": [],
                            "success": False,
                        }
                        continue
                    grasp_scan_payload = dict(grasp_scan_payload)
                    grasp_scan_payload["attempt"] = scan_attempt + 1
                    grasp_scan_payload["max_attempts"] = 1 + scan_retries
                    if bool(grasp_scan_payload.get("success")):
                        break
                    self.get_logger().warning(
                        f"grasp item scan attempt {scan_attempt + 1}/"
                        f"{1 + scan_retries} did not find {target_item_id}"
                    )
                cycle_records = [dict(view) for view in grasp_scan_payload["views"]]
                diagnostics.append(
                    "grasp single-pass scan: "
                    f"views={len(grasp_scan_payload['views'])} "
                    f"offset={float(grasp_scan_payload['final_offset_from_start_m']):.3f}m "
                    f"target_found={bool(grasp_scan_payload['success'])} "
                    f"attempt={int(grasp_scan_payload.get('attempt') or 1)}/"
                    f"{int(grasp_scan_payload.get('max_attempts') or 1)}"
                )
                if bool(self.get_parameter("precenter").value):
                    diagnostics.append(
                        "precenter skipped because base grasp scan selected the final view"
                    )
            elif bool(self.get_parameter("precenter").value):
                self._publish_status(f"precentering: run_id={run_id}")
                cycle, centering_logs, cycle_records = self._run_precenter_loop(
                    run_id=run_id,
                    prompt=prompt,
                    options=options,
                    config=config,
                    hand_eye=hand_eye,
                    planner=planner,
                )
            else:
                cycle = self._capture_analyze_cycle(
                    run_id=run_id,
                    prompt=prompt,
                    options=options,
                    hand_eye=hand_eye,
                    phase_label="main",
                )
                cycle_records = [dict(cycle["debug"])]

            capture_response = cycle["capture_response"]
            analyze_response = cycle["analyze_response"]
            current_pose = cycle["current_pose"]
            base_to_camera = cycle["base_to_camera"]
            candidate_pool = list(cycle.get("candidate_pool") or [])
            if not candidate_pool and cycle["candidate"] is not None:
                candidate_pool = [cycle["candidate"]]

            if centering_logs:
                diagnostics.extend(centering_logs)
            diagnostics.extend(cycle["diagnostics"])
            if not candidate_pool:
                self._publish_candidate_validation_markers(
                    validation_records=[],
                    camera_frame=str(capture_response.camera_frame or "camera_color_optical_frame"),
                )
                status = "no_candidate"
                summary_lines = [line for line in centering_logs if line]
                summary_lines.append(analyze_response.summary or "no valid grasp candidate found")
                summary = "\n".join(summary_lines)
                result_payload.update(
                    {
                        "status": status,
                        "scene_id": capture_response.scene_id,
                        "summary": summary,
                        "diagnostics": diagnostics,
                        "centering_logs": centering_logs,
                        "vision": self._analyze_debug_dict(analyze_response),
                        "capture": self._capture_debug_dict(capture_response),
                        "grasp_target_scan": grasp_scan_payload,
                    }
                )
                return

            summary_lines = [line for line in centering_logs if line]
            if analyze_response.summary:
                summary_lines.append(analyze_response.summary)
            summary = "\n".join(summary_lines) if summary_lines else ""

            execution_payload = None
            execute_enabled = bool(self.get_parameter("execute").value)
            confirm_required = bool(self.get_parameter("confirm").value) and execute_enabled
            move_home_after = bool(self.get_parameter("move_home_after").value)
            validation_records: list[dict[str, object]] = []

            if execute_enabled:
                # Lock it again at the validation boundary.  This prevents a
                # stale dashboard parameter from turning a bottle validation
                # into safe_top_down before the execution-time lock below.
                if target_item is not None:
                    grasp_strategy = self._set_executor_strategy_for_item(target_item)
                validation_candidate_limit = max(1, int(options.get("robot_validation_candidate_limit", 6)))
                validation_variant_limit = max(1, int(options.get("robot_validation_variant_limit", 4)))
                validation_candidates = candidate_pool[:validation_candidate_limit]

                def build_candidate_plans(item):
                    return [
                        self._retarget_plan_to_object_center(
                            plan=built_plan,
                            candidate=item,
                            base_to_camera=base_to_camera,
                            prompt=prompt,
                        )
                        for built_plan in planner.plan_grasp_variants(
                            item,
                            current_pose,
                            base_to_camera,
                        )[:validation_variant_limit]
                    ]

                candidate, plan, validation_logs, validation_records = select_first_reachable_candidate(
                    validation_candidates,
                    build_plan=build_candidate_plans,
                    validate_plan=lambda _item, built_plan: self._validate_candidate_plan(
                        run_id=run_id,
                        plan=built_plan,
                        move_home_after=move_home_after,
                    ),
                )
                diagnostics.append(
                    "robot validation search: "
                    f"candidates_tried={len(validation_candidates)}/{len(candidate_pool)} "
                    f"variant_limit_per_candidate={validation_variant_limit}"
                )
                diagnostics.extend(validation_logs)
                self._publish_candidate_validation_markers(
                    validation_records=validation_records,
                    camera_frame=str(capture_response.camera_frame or "camera_color_optical_frame"),
                )
                if candidate is None or plan is None:
                    status = "failed"
                    summary = self._append_summary_line(summary, "no robot-reachable grasp candidate found")
                    result_payload.update(
                        {
                            "status": status,
                            "scene_id": capture_response.scene_id,
                            "summary": summary,
                            "diagnostics": diagnostics,
                            "centering_logs": centering_logs,
                            "vision": self._analyze_debug_dict(analyze_response),
                            "capture": self._capture_debug_dict(capture_response),
                            "grasp_target_scan": grasp_scan_payload,
                            "candidate_validation": validation_records,
                            "execution": None,
                        }
                    )
                    return
                selected_index = next(
                    index for index, item in enumerate(candidate_pool) if item is candidate
                )
                if selected_index > 0:
                    summary = self._append_summary_line(
                        summary,
                        f"selected fallback candidate[{selected_index}] score={candidate.score:.4f} after robot validation",
                    )
            else:
                self._publish_candidate_validation_markers(
                    validation_records=[],
                    camera_frame=str(capture_response.camera_frame or "camera_color_optical_frame"),
                )
                candidate = candidate_pool[0]
                plan = planner.plan_grasp(candidate, current_pose, base_to_camera)
                plan = self._retarget_plan_to_object_center(
                    plan=plan,
                    candidate=candidate,
                    base_to_camera=base_to_camera,
                    prompt=prompt,
                )

            result_payload.update(
                {
                    "vision": self._analyze_debug_dict(analyze_response),
                    "capture": self._capture_debug_dict(capture_response),
                    "grasp_target_scan": grasp_scan_payload,
                    "candidate_validation": validation_records,
                    "candidate": {
                        "score": candidate.score,
                        "instance_index": candidate.instance_index,
                        "translation_camera_m": list(candidate.translation_camera_m),
                        "object_center_camera_m": (
                            list(candidate.object_center_camera_m)
                            if candidate.object_center_camera_m is not None
                            else None
                        ),
                    },
                    "plan": plan_debug_dict(plan),
                    "confirmed": None if confirm_required else (True if execute_enabled else None),
                }
            )

            if confirm_required:
                status = "awaiting_confirmation"
                summary = self._append_summary_line(
                    summary,
                    f"awaiting confirmation for run_id={run_id}; call /grasp_pipeline/confirm to execute or /grasp_pipeline/reject to cancel",
                )
                result_payload.update(
                    {
                        "status": status,
                        "scene_id": capture_response.scene_id,
                        "summary": summary,
                        "diagnostics": diagnostics,
                        "centering_logs": centering_logs,
                        "execution": None,
                    }
                )
                self._set_pending_confirmation(
                    PendingConfirmation(
                        run_id=run_id,
                        prompt=prompt,
                        scene_id=str(capture_response.scene_id),
                        plan=plan,
                        move_home_after=move_home_after,
                        target_item_id=target_item_id,
                        hand_eye=np.asarray(hand_eye, dtype=np.float64).copy(),
                        request_payload=dict(request_payload),
                        cycle_records=[dict(record) for record in cycle_records],
                        result_payload=dict(result_payload),
                    )
                )
                return

            if execute_enabled:
                self._publish_status(f"executing_plan: run_id={run_id}")
                execution_payload = self._execute_grasp_and_optional_place(
                    run_id=run_id,
                    plan=plan,
                    move_home_after=move_home_after,
                    target_item_id=target_item_id,
                    hand_eye=hand_eye,
                    advance_base_after_grasp=base_grasp_scan_requested,
                )
                summary = self._append_summary_line(
                    summary,
                    "grasp and placement completed"
                    if bool(self.get_parameter("place_after_grasp").value)
                    else "execution completed",
                )

            result_payload.update(
                {
                    "status": "ok",
                    "scene_id": capture_response.scene_id,
                    "summary": summary,
                    "diagnostics": diagnostics,
                    "centering_logs": centering_logs,
                    "execution": execution_payload,
                }
            )
        except Exception as exc:
            status = "failed"
            summary = str(exc)
            diagnostics.append(traceback.format_exc())
            self.get_logger().error(f"distributed pipeline failed: {exc}")
            result_payload.update({"status": status, "summary": summary, "diagnostics": diagnostics})
        finally:
            if base_grasp_scan_requested:
                # One-shot base-motion permission: the dashboard must request
                # it again for every new grasp scan.
                try:
                    self.set_parameters(
                        [Parameter("base_grasp_scan_enabled", value=False)]
                    )
                except Exception as exc:
                    self.get_logger().warning(
                        f"could not clear base_grasp_scan_enabled: {exc}"
                    )
            result_payload.setdefault("artifacts", {})
            result_payload["artifacts"]["artifact_root"] = str(self._artifact_root_dir())
            artifact_dir = self._write_run_artifacts(
                run_id=run_id,
                request_payload=request_payload,
                cycle_records=cycle_records,
                result_payload=result_payload,
            )
            result_payload["artifacts"]["run_dir"] = str(artifact_dir)
            self._write_json_file(artifact_dir / "final_result.json", result_payload)
            final_status = f"{status}: {summary}" if summary else status
            self._publish_status(final_status)
            if summary:
                self._publish_summary(summary)
            if diagnostics:
                self._publish_diagnostics(diagnostics)
            self._publish_result(result_payload)
            self._run_id = None


def main(args=None) -> None:
    rclpy.init(args=args)
    node = PipelineOrchestratorNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
