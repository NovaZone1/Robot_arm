from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
import threading
import time
import traceback

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.exceptions import ParameterUninitializedException
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger
from visualization_msgs.msg import MarkerArray

from robot_grasp_msgs.srv import AnalyzeScene, CaptureScene, ExecuteGraspPlan, ExecuteNamedPose, GetRobotState, StopRobot
from robot_grasp_ros2.distributed_utils import (
    base_to_camera_from_tcp_and_hand_eye,
    build_runtime_config,
    camera_info_to_intrinsics,
    candidate_debug_dict,
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
from src.robot.plan_validation import select_first_reachable_candidate


@dataclass(slots=True)
class PendingConfirmation:
    run_id: str
    prompt: str
    scene_id: str
    plan: object
    move_home_after: bool
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
        self.create_service(Trigger, "~/stop", self._handle_stop_service)
        self.create_service(Trigger, "~/confirm", self._handle_confirm_service)
        self.create_service(Trigger, "~/reject", self._handle_reject_service)

        self._rpc_callback_group = ReentrantCallbackGroup()
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
        self._stop_robot_client = self.create_client(
            StopRobot,
            self._service_name("robot_stop_service"),
            callback_group=self._rpc_callback_group,
        )

        self._run_lock = threading.Lock()
        self._run_thread: threading.Thread | None = None
        self._run_id: str | None = None
        self._stop_requested = False
        self._pending_confirmation: PendingConfirmation | None = None

        self._auto_start_armed = bool(self.get_parameter("auto_start").value)
        self._auto_start_timer = self.create_timer(0.5, self._maybe_auto_start)
        self._publish_status("idle")

    def _declare_parameters(self) -> None:
        self.declare_parameter("prompt", "")
        self.declare_parameter("execute", False)
        self.declare_parameter("move_home_after", True)
        self.declare_parameter("enable_pregrasp", False)
        self.declare_parameter("show_pointcloud", False)
        self.declare_parameter("precenter", False)
        self.declare_parameter("confirm", False)
        self.declare_parameter("pointcloud_filter_mode", "bilateral")
        self.declare_parameter("pointcloud_backend", "sdk")
        self.declare_parameter("depth_fusion_frames", 8)
        self.declare_parameter("speed", 40)
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
        self.declare_parameter("observe_pose", [30.0, 0.0, 400.0, 0.0, 120.0, 0.0])
        self.declare_parameter("camera_capture_service", "/camera_server/capture")
        self.declare_parameter("vision_analyze_service", "/vision_worker/analyze")
        self.declare_parameter("robot_state_service", "/robot_executor/get_state")
        self.declare_parameter("robot_named_pose_service", "/robot_executor/execute_named_pose")
        self.declare_parameter("robot_execute_plan_service", "/robot_executor/execute_grasp_plan")
        self.declare_parameter("robot_stop_service", "/robot_executor/stop_robot")
        self.declare_parameter("artifact_root", "")

    def _publish_status(self, text: str) -> None:
        self.get_logger().info(text)
        self._status_pub.publish(String(data=text))

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
            "execute": bool(self.get_parameter("execute").value),
            "move_home_after": bool(self.get_parameter("move_home_after").value),
            "enable_pregrasp": bool(self.get_parameter("enable_pregrasp").value),
            "show_pointcloud": bool(self.get_parameter("show_pointcloud").value),
            "precenter": bool(self.get_parameter("precenter").value),
            "confirm": bool(self.get_parameter("confirm").value),
            "pointcloud_filter_mode": str(self.get_parameter("pointcloud_filter_mode").value or "bilateral"),
            "pointcloud_backend": str(self.get_parameter("pointcloud_backend").value or "sdk"),
            "depth_fusion_frames": int(self.get_parameter("depth_fusion_frames").value),
            "speed": int(self.get_parameter("speed").value),
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
        timeout_s: float = 20.0,
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

    def _capture_scene_once(self, *, run_id: str, phase_label: str):
        self._publish_status(f"capturing_scene: run_id={run_id} phase={phase_label}")
        capture_req = CaptureScene.Request()
        capture_req.run_id = run_id
        capture_req.depth_fusion_frames = int(self.get_parameter("depth_fusion_frames").value)
        capture_req.pointcloud_filter_mode = str(self.get_parameter("pointcloud_filter_mode").value)
        capture_req.pointcloud_backend = str(self.get_parameter("pointcloud_backend").value)
        capture_response = self._call_client(self._capture_client, capture_req, timeout_s=30.0)
        if not capture_response.success:
            raise RuntimeError(capture_response.message)
        return capture_response

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
        if not prompt:
            self._publish_status("auto_start skipped: empty prompt")
            return
        accepted, message = self._start_background_run(prompt_override=prompt)
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
                self._get_state_client,
                self._named_pose_client,
                self._execute_plan_client,
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

    def _handle_stop_service(self, _request, response):
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
            if self._run_thread is not None and self._run_thread.is_alive():
                return False, "pipeline is already running"
            if self._pending_confirmation is not None:
                return (
                    False,
                    f"confirmation pending for run_id={self._pending_confirmation.run_id}; "
                    "call /grasp_pipeline/confirm or /grasp_pipeline/reject first",
                )
            prompt = prompt_override if prompt_override is not None else str(self.get_parameter("prompt").value or "").strip()
            if not prompt:
                return False, "prompt is empty; set parameter 'prompt' or publish to ~/run_prompt"
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
            return True, f"run accepted for prompt={prompt}"

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
            execute_req = ExecuteGraspPlan.Request()
            execute_req.run_id = pending.run_id
            execute_req.execute = True
            execute_req.move_home_after = bool(pending.move_home_after)
            execute_req.plan = grasp_plan_to_msg(pending.plan)
            execute_response = self._call_client(self._execute_plan_client, execute_req, timeout_s=120.0)
            if not execute_response.success:
                raise RuntimeError(execute_response.message)
            execution_payload = json.loads(execute_response.execution_json)
            result_payload["execution"] = execution_payload
            result_payload["confirmed"] = True
            result_payload["status"] = "ok"
            result_payload["execution_message"] = str(execute_response.message)
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
        try:
            self._publish_candidate_validation_markers(
                validation_records=[],
                camera_frame="camera_color_optical_frame",
            )
            self._publish_status(f"preflight: run_id={run_id}")
            options, config, hand_eye, planner = self._build_runtime()
            request_payload.update(
                {
                    "options": options,
                    "observe_pose": self._observe_pose_values(),
                    "artifact_root": str(self._artifact_root_dir()),
                }
            )
            request_payload["confirm"] = bool(self.get_parameter("confirm").value)
            request_payload["execute"] = bool(self.get_parameter("execute").value)

            self._publish_status(f"moving_to_observation: run_id={run_id}")
            observe_pose = self._observe_pose_values()
            if bool(self.get_parameter("skip_observation_move").value):
                self._publish_status(f"observation_move_skipped: run_id={run_id}")
            else:
                self._execute_named_pose(
                    name="observation",
                    position_m=(observe_pose[0] / 1000.0, observe_pose[1] / 1000.0, observe_pose[2] / 1000.0),
                    rpy_deg=(observe_pose[3], observe_pose[4], observe_pose[5]),
                    speed_percent=float(self.get_parameter("speed").value),
                    open_gripper_first=True,
                    timeout_s=20.0,
                )

            if bool(self.get_parameter("precenter").value):
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
                validation_candidate_limit = max(1, int(options.get("robot_validation_candidate_limit", 6)))
                validation_variant_limit = max(1, int(options.get("robot_validation_variant_limit", 4)))
                validation_candidates = candidate_pool[:validation_candidate_limit]

                def build_candidate_plans(item):
                    return [
                        self._retarget_plan_to_object_center(
                            plan=built_plan,
                            candidate=item,
                            base_to_camera=base_to_camera,
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
                )

            result_payload.update(
                {
                    "vision": self._analyze_debug_dict(analyze_response),
                    "capture": self._capture_debug_dict(capture_response),
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
                        request_payload=dict(request_payload),
                        cycle_records=[dict(record) for record in cycle_records],
                        result_payload=dict(result_payload),
                    )
                )
                return

            if execute_enabled:
                self._publish_status(f"executing_plan: run_id={run_id}")
                execute_req = ExecuteGraspPlan.Request()
                execute_req.run_id = run_id
                execute_req.execute = True
                execute_req.move_home_after = move_home_after
                execute_req.plan = grasp_plan_to_msg(plan)
                execute_response = self._call_client(self._execute_plan_client, execute_req, timeout_s=120.0)
                if not execute_response.success:
                    raise RuntimeError(execute_response.message)
                execution_payload = json.loads(execute_response.execution_json)
                summary = self._append_summary_line(summary, "execution completed")

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
