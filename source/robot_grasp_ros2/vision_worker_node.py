from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from robot_grasp_msgs.srv import AnalyzeScene, DetectTarget2D, MatchItemLabel
from robot_grasp_ros2.distributed_utils import (
    camera_info_to_intrinsics,
    color_msg_to_bgr,
    depth_msg_to_meters,
    grasp_candidate_to_msg,
    json_dumps,
    make_latched_qos,
    make_perception_summary_msg,
    parse_options_json,
    pose6d_to_end_pose,
    transform_msg_to_matrix,
)
from robot_grasp_ros2.rviz_visualization import PipelineRvizPublisher
from src.grasping.models import GraspCandidate, GraspPlan, PerceptionResult
from src.perception.item_catalog import (
    ItemCatalog,
    ReferenceLabelMatcher,
    default_item_catalog_path,
)


class _ArrayPointCloud:
    def __init__(self, points: np.ndarray | None = None) -> None:
        array = np.asarray(points if points is not None else np.empty((0, 3), dtype=np.float32), dtype=np.float32)
        self.points = array.reshape(-1, 3)


class ExternalVisionWorkerBridge:
    def __init__(self, node: Node) -> None:
        self._node = node
        self._daemon_proc: subprocess.Popen | None = None
        self._daemon_lock = threading.Lock()

    def _worker_python(self) -> str:
        configured = str(self._node.get_parameter("worker_python_executable").value or "").strip()
        if configured:
            return configured
        worker_python = str(os.environ.get("ROBOT_GRASP_CONDA_PYTHON") or "").strip()
        if worker_python and Path(worker_python).exists():
            return worker_python
        return sys.executable

    def _worker_script(self) -> str:
        configured = str(self._node.get_parameter("worker_script").value or "").strip()
        if configured:
            return configured
        return str(Path(__file__).resolve().parents[1] / "src" / "perception" / "external_inference_worker.py")

    def _worker_timeout(self) -> float:
        return max(1.0, float(self._node.get_parameter("worker_timeout_s").value))

    def _worker_mode(self) -> str:
        try:
            return str(self._node.get_parameter("worker_mode").value or "daemon").strip().lower()
        except Exception:
            return "daemon"

    def _ensure_daemon(self) -> subprocess.Popen:
        """Start the daemon process if not already running."""
        if self._daemon_proc is not None and self._daemon_proc.poll() is None:
            return self._daemon_proc
        worker_python = self._worker_python()
        worker_script = self._worker_script()
        self._node.get_logger().info(
            f"[vision_bridge] starting inference daemon: {worker_python} {worker_script} --daemon"
        )
        proc = subprocess.Popen(
            [worker_python, worker_script, "--daemon"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,  # inherit node stderr so logs appear in session log
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        # Wait for the "daemon ready" signal (first stderr line) by doing a ping.
        deadline = time.monotonic() + 120.0  # allow up to 2 min for model load
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"inference daemon exited unexpectedly (rc={proc.returncode}) during startup"
                )
            try:
                proc.stdin.write(b'{"type":"ping"}\n')
                proc.stdin.flush()
                proc.stdout.readline()  # wait for pong
                break
            except Exception:
                time.sleep(0.5)
        else:
            proc.kill()
            raise RuntimeError("inference daemon did not respond to ping within 120 s")
        self._daemon_proc = proc
        self._node.get_logger().info("[vision_bridge] inference daemon is ready")
        return proc

    def _call_daemon(self, request: dict[str, object]) -> dict[str, object]:
        proc = self._ensure_daemon()
        line = json.dumps(request, ensure_ascii=False) + "\n"
        try:
            proc.stdin.write(line.encode("utf-8"))
            proc.stdin.flush()
        except BrokenPipeError:
            self._daemon_proc = None
            raise RuntimeError("inference daemon pipe broken; will restart on next call")
        timeout = self._worker_timeout()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                self._daemon_proc = None
                raise RuntimeError(f"inference daemon exited unexpectedly (rc={proc.returncode})")
            # Non-blocking readline via select to respect timeout.
            import select
            ready, _, _ = select.select([proc.stdout], [], [], 0.1)
            if ready:
                raw = proc.stdout.readline()
                if not raw:
                    self._daemon_proc = None
                    raise RuntimeError("inference daemon closed stdout unexpectedly")
                return json.loads(raw.decode("utf-8", errors="replace"))
        raise TimeoutError(f"inference daemon did not respond within {timeout:.0f} s")

    def shutdown(self) -> None:
        """Send shutdown to daemon and wait for it to exit."""
        with self._daemon_lock:
            proc = self._daemon_proc
            self._daemon_proc = None
        if proc is None or proc.poll() is not None:
            return
        try:
            proc.stdin.write(b'{"type":"shutdown"}\n')
            proc.stdin.flush()
            proc.wait(timeout=5.0)
        except Exception:
            proc.kill()

    def warmup(self, options_json: str = "") -> None:
        with self._daemon_lock:
            payload = self._call_daemon(
                {"type": "warmup", "options_json": str(options_json or "")}
            )
        if not bool(payload.get("success")):
            raise RuntimeError(str(payload.get("message") or "vision warmup failed"))

    def detect_target_2d(
        self,
        color_bgr: np.ndarray,
        prompt: str,
    ) -> dict[str, object]:
        with tempfile.TemporaryDirectory(prefix="target_2d_worker_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            color_path = tmp_dir / "color.npy"
            np.save(color_path, np.asarray(color_bgr))
            request_payload = {
                "type": "detect_target_2d",
                "work_dir": str(tmp_dir),
                "color_npy": color_path.name,
                "prompt": str(prompt),
            }
            with self._daemon_lock:
                payload = self._call_daemon(request_payload)
        if not bool(payload.get("success")):
            raise RuntimeError(str(payload.get("message") or "target detection failed"))
        return dict(payload.get("result") or {})

    def detect_label_bottles(
        self,
        color_bgr: np.ndarray,
    ) -> tuple[dict[str, object], ...]:
        """Run generic bottle-shape detection in the external ML runtime."""
        with tempfile.TemporaryDirectory(prefix="label_bottle_worker_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            color_path = tmp_dir / "color.npy"
            np.save(color_path, np.asarray(color_bgr))
            request_payload = {
                "type": "detect_label_bottles",
                "work_dir": str(tmp_dir),
                "color_npy": color_path.name,
            }
            with self._daemon_lock:
                payload = self._call_daemon(request_payload)
            if not bool(payload.get("success")):
                raise RuntimeError(
                    str(payload.get("message") or "label bottle detection failed")
                )
            result = dict(payload.get("result") or {})
            return tuple(
                dict(item)
                for item in list(result.get("bottle_proposals") or [])
            )

    @staticmethod
    def _candidate_from_dict(payload: dict[str, object]) -> GraspCandidate:
        object_center = payload.get("object_center_camera_m")
        center_offset = payload.get("center_offset_m")
        return GraspCandidate(
            instance_index=int(payload["instance_index"]),
            score=float(payload["score"]),
            width_m=float(payload["width_m"]),
            depth_m=float(payload["depth_m"]),
            translation_camera_m=tuple(float(value) for value in list(payload["translation_camera_m"])),
            rotation_camera=np.asarray(payload["rotation_camera"], dtype=np.float64).reshape(3, 3),
            object_center_camera_m=(
                tuple(float(value) for value in list(object_center))
                if object_center is not None
                else None
            ),
            center_offset_m=(float(center_offset) if center_offset is not None else None),
            raw_grasp=None,
        )

    @classmethod
    def _plan_from_dict(cls, payload: dict[str, object]) -> GraspPlan:
        return GraspPlan(
            candidate=cls._candidate_from_dict(dict(payload["candidate"])),
            target_base_m=tuple(float(value) for value in list(payload["target_base_m"])),
            target_rpy_deg=tuple(float(value) for value in list(payload["target_rpy_deg"])),
            pregrasp_base_m=tuple(float(value) for value in list(payload["pregrasp_base_m"])),
            grasp_base_m=tuple(float(value) for value in list(payload["grasp_base_m"])),
            retreat_base_m=tuple(float(value) for value in list(payload["retreat_base_m"])),
            within_workspace=bool(payload["within_workspace"]),
            workspace_violations=[str(item) for item in list(payload["workspace_violations"])],
            target_contact_point_base_m=(
                tuple(float(value) for value in list(payload["target_contact_point_base_m"]))
                if payload.get("target_contact_point_base_m") is not None
                else None
            ),
            tool_contact_offset_tool_m=(
                tuple(float(value) for value in list(payload["tool_contact_offset_tool_m"]))
                if payload.get("tool_contact_offset_tool_m") is not None
                else None
            ),
        )

    @staticmethod
    def _load_optional_npy(base_dir: Path, relative_path: str | None) -> np.ndarray | None:
        if not relative_path:
            return None
        path = base_dir / relative_path
        if not path.is_file():
            return None
        return np.load(path)

    def _perception_from_result(self, base_dir: Path, request, payload: dict[str, object]) -> PerceptionResult:
        scene_points = self._load_optional_npy(base_dir, payload.get("scene_points_path"))
        object_clouds: list[_ArrayPointCloud] = []
        for relative_path in list(payload.get("object_cloud_paths") or []):
            object_clouds.append(_ArrayPointCloud(self._load_optional_npy(base_dir, relative_path)))
        return PerceptionResult(
            color_bgr=color_msg_to_bgr(request.color_image),
            depth_meters=depth_msg_to_meters(request.depth_image),
            segmentation={},
            scene_points=scene_points,
            pointclouds=object_clouds,
            grasp_groups=[],
            scene_grasp_count=int(payload["scene_grasp_count"]),
            scene_point_count=int(payload["scene_point_count"]),
            object_point_counts=[int(value) for value in list(payload["object_point_counts"])],
            object_centers_camera_m=[None] * int(payload["instance_count"]),
            object_centers_uv=[None] * int(payload["instance_count"]),
        )

    def run(self, request) -> dict[str, object]:
        worker_python = self._worker_python()
        worker_script = self._worker_script()
        if shutil.which(worker_python) is None and not Path(worker_python).exists():
            raise RuntimeError(f"worker python executable not found: {worker_python}")
        if not Path(worker_script).is_file():
            raise RuntimeError(f"worker script not found: {worker_script}")

        options = parse_options_json(request.options_json)

        with tempfile.TemporaryDirectory(prefix="vision_worker_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            color_path = tmp_dir / "color.npy"
            depth_path = tmp_dir / "depth.npy"

            color_bgr = color_msg_to_bgr(request.color_image)
            depth_meters = depth_msg_to_meters(request.depth_image)
            np.save(color_path, color_bgr)
            np.save(depth_path, depth_meters)

            request_payload = {
                "run_id": str(request.run_id),
                "scene_id": str(request.scene_id),
                "prompt": str(request.prompt),
                "camera_frame": str(request.camera_frame),
                "work_dir": str(tmp_dir),
                "color_npy": color_path.name,
                "depth_npy": depth_path.name,
                "camera_info": {
                    "width": int(request.camera_info.width),
                    "height": int(request.camera_info.height),
                    "k": [float(value) for value in list(request.camera_info.k)],
                },
                "tcp_pose": [
                    float(request.tcp_pose.x_mm),
                    float(request.tcp_pose.y_mm),
                    float(request.tcp_pose.z_mm),
                    float(request.tcp_pose.roll_deg),
                    float(request.tcp_pose.pitch_deg),
                    float(request.tcp_pose.yaw_deg),
                ],
                "base_to_camera": transform_msg_to_matrix(request.base_to_camera).tolist(),
                "options_json": json.dumps(options, ensure_ascii=False),
            }

            use_daemon = self._worker_mode() == "daemon"
            if use_daemon:
                with self._daemon_lock:
                    payload = self._call_daemon(request_payload)
                worker_stdout = ""
                worker_stderr = ""
            else:
                # Legacy single-shot subprocess mode.
                request_path = tmp_dir / "request.json"
                response_path = tmp_dir / "response.json"
                request_path.write_text(
                    json.dumps(request_payload, ensure_ascii=False, indent=2), encoding="utf-8"
                )
                completed = subprocess.run(
                    [worker_python, worker_script, "--request-json", str(request_path),
                     "--response-json", str(response_path)],
                    cwd=str(Path(__file__).resolve().parents[1]),
                    capture_output=True,
                    text=True,
                    timeout=self._worker_timeout(),
                    check=False,
                )
                if not response_path.is_file():
                    raise RuntimeError(
                        "external vision worker did not produce a response. "
                        f"exit_code={completed.returncode} "
                        f"stdout={completed.stdout.strip()!r} stderr={completed.stderr.strip()!r}"
                    )
                payload = json.loads(response_path.read_text(encoding="utf-8"))
                worker_stdout = completed.stdout
                worker_stderr = completed.stderr

            if not bool(payload.get("success")):
                message = str(payload.get("message", "external worker failed"))
                trace = str(payload.get("traceback", "")).strip()
                details = [message]
                if trace:
                    details.append(trace)
                raise RuntimeError("\n".join(details))

            result = dict(payload["result"])
            perception_payload = dict(result["perception"])
            candidate_pool_payload = list(result.get("candidate_pool") or [])
            selected_candidate_payload = result.get("candidate")
            plan_payload = result.get("plan")

            perception = self._perception_from_result(tmp_dir, request, perception_payload)
            candidate_pool = [
                (
                    self._candidate_from_dict(dict(item["candidate"])),
                    float(item.get("approach_angle_deg", 0.0)),
                    float(item.get("rotation_delta_deg", 0.0)),
                )
                for item in candidate_pool_payload
            ]
            candidate = (
                self._candidate_from_dict(dict(selected_candidate_payload))
                if selected_candidate_payload is not None
                else None
            )
            plan = self._plan_from_dict(dict(plan_payload)) if plan_payload is not None else None

            return {
                "perception": perception,
                "candidate_pool": candidate_pool,
                "candidate": candidate,
                "plan": plan,
                "diagnostics": [str(line) for line in list(result.get("diagnostics") or [])],
                "summary": str(result.get("summary", "")),
                "use_pregrasp": bool(result.get("use_pregrasp", False)),
                "worker_stdout": worker_stdout,
                "worker_stderr": worker_stderr,
            }


class VisionWorkerNode(Node):
    """Distributed perception worker that delegates heavy inference to an external Python runtime."""

    def __init__(self) -> None:
        super().__init__("vision_worker")
        self.declare_parameter("worker_python_executable", "")
        self.declare_parameter("worker_script", "")
        self.declare_parameter("worker_timeout_s", 300.0)
        self.declare_parameter("worker_mode", "daemon")
        self.declare_parameter("item_catalog_path", "")

        text_qos = make_latched_qos(depth=10)
        self._status_pub = self.create_publisher(String, "~/status", text_qos)
        self._summary_pub = self.create_publisher(String, "~/summary", text_qos)
        self._result_pub = self.create_publisher(String, "~/result_json", text_qos)
        self._rviz_pub = PipelineRvizPublisher(self)
        self.create_service(AnalyzeScene, "~/analyze", self._handle_analyze)
        self.create_service(DetectTarget2D, "~/detect_target_2d", self._handle_detect_target_2d)
        self.create_service(MatchItemLabel, "~/match_item_label", self._handle_match_item_label)
        self.create_service(Trigger, "~/warmup", self._handle_warmup)
        self._bridge = ExternalVisionWorkerBridge(self)
        self._label_matcher: ReferenceLabelMatcher | None = None
        self._publish_status("idle")

    def destroy_node(self) -> None:
        self._bridge.shutdown()
        super().destroy_node()

    def _publish_status(self, text: str) -> None:
        self.get_logger().info(text)
        self._status_pub.publish(String(data=text))

    def _get_label_matcher(self) -> ReferenceLabelMatcher:
        if self._label_matcher is None:
            configured = str(self.get_parameter("item_catalog_path").value or "").strip()
            catalog_path = Path(configured).expanduser().resolve() if configured else default_item_catalog_path()
            self._label_matcher = ReferenceLabelMatcher(ItemCatalog.load(catalog_path))
        return self._label_matcher

    def _handle_warmup(self, _request, response):
        try:
            self._publish_status("warming_up")
            self._bridge.warmup(
                json.dumps(
                    {
                        "graspnet_checkpoint": "checkpoint.tar",
                        "hand_eye_config": str(
                            Path(__file__).resolve().parents[1]
                            / "config/hand_eye/verify_config_eyeinhand_cam2tcp.yaml"
                        ),
                    }
                )
            )
            response.success = True
            response.message = "vision models warmed up"
            self._publish_status("idle")
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            self._publish_status(f"warmup failed: {exc}")
        return response

    def _handle_detect_target_2d(self, request, response):
        run_id = request.run_id.strip() or "unknown"
        try:
            self._publish_status(f"detecting_target_2d: run_id={run_id}")
            result = self._bridge.detect_target_2d(
                color_msg_to_bgr(request.color_image),
                request.prompt,
            )
            response.success = True
            response.found = bool(result.get("found"))
            response.center_u_norm = float(result.get("center_u_norm") or 0.0)
            response.center_v_norm = float(result.get("center_v_norm") or 0.0)
            response.confidence = float(result.get("confidence") or 0.0)
            response.backend = str(result.get("backend") or "unknown")
            response.message = (
                f"target found at ({response.center_u_norm:.3f}, "
                f"{response.center_v_norm:.3f})"
                if response.found
                else "target not found"
            )
            self._publish_status(
                f"target_2d completed: run_id={run_id} found={response.found}"
            )
        except Exception as exc:
            response.success = False
            response.found = False
            response.message = str(exc)
            self._publish_status(f"target_2d failed: run_id={run_id} error={exc}")
        return response

    def _handle_match_item_label(
        self,
        request: MatchItemLabel.Request,
        response: MatchItemLabel.Response,
    ):
        run_id = request.run_id.strip() or "unknown"
        response.slot_index = -1
        response.has_box_center = False
        try:
            options = parse_options_json(request.options_json)
            raw_roi = options.get("label_search_roi_norm")
            roi = (
                tuple(float(value) for value in list(raw_roi))
                if isinstance(raw_roi, list) and len(raw_roi) == 4
                else None
            )
            threshold = options.get("label_match_threshold")
            localize_box_row = bool(options.get("localize_box_row", True))
            marker_detection_enabled = bool(
                options.get("label_marker_detection_enabled", False)
            )
            matcher = self._get_label_matcher()
            color_bgr = color_msg_to_bgr(request.color_image)
            try:
                bottle_proposals = self._bridge.detect_label_bottles(color_bgr)
            except Exception as exc:
                bottle_proposals = None
                self.get_logger().warning(
                    f"YOLO label bottle proposals unavailable; using local fallback: {exc}"
                )
            match = matcher.match_expected(
                color_bgr,
                request.expected_item_id,
                roi_norm=roi,
                threshold=(float(threshold) if threshold is not None else None),
                bottle_proposals=bottle_proposals,
                marker_detection_enabled=marker_detection_enabled,
            )
            partial_label_observations: list[dict[str, object]] = []
            partial_projection_errors: list[str] = []
            if match.detections and localize_box_row:
                depth_meters = depth_msg_to_meters(request.depth_image)
                camera_k = tuple(float(value) for value in request.camera_info.k)
                base_to_camera = transform_msg_to_matrix(request.base_to_camera)
                table_z_m = float(options.get("table_z_m", 0.0))
                minimum_label_z_m = table_z_m - 0.015
                maximum_label_z_m = table_z_m + 0.35
                raw_projections: list[
                    tuple[object, tuple[float, float, float] | None]
                ] = []
                for detection in match.detections:
                    try:
                        point = matcher.project_label_centers(
                            depth_meters=depth_meters,
                            camera_k=camera_k,
                            base_to_camera=base_to_camera,
                            detections=(detection,),
                        )[0]
                        raw_projections.append((detection, point))
                    except Exception as exc:
                        raw_projections.append((detection, None))
                        partial_projection_errors.append(
                            f"{detection.item_id}: {exc}"
                        )
                camera_from_base = np.linalg.inv(base_to_camera)
                valid_plane_depths: list[float] = []
                for _, point in raw_projections:
                    if (
                        point is not None
                        and minimum_label_z_m
                        <= float(point[2])
                        <= maximum_label_z_m
                    ):
                        camera_point = camera_from_base @ np.asarray(
                            [point[0], point[1], point[2], 1.0],
                            dtype=np.float64,
                        )
                        if 0.10 < float(camera_point[2]) < 3.0:
                            valid_plane_depths.append(float(camera_point[2]))
                row_plane_depth_m = (
                    float(np.median(valid_plane_depths))
                    if valid_plane_depths
                    else None
                )
                for detection, raw_point in raw_projections:
                    point = raw_point
                    depth_source = "measured"
                    if (
                        point is None
                        or not minimum_label_z_m
                        <= float(point[2])
                        <= maximum_label_z_m
                    ):
                        if row_plane_depth_m is None:
                            partial_projection_errors.append(
                                f"{detection.item_id}: no valid row-plane depth "
                                "anchor in this view"
                            )
                            continue
                        try:
                            point = matcher.project_label_centers(
                                depth_meters=depth_meters,
                                camera_k=camera_k,
                                base_to_camera=base_to_camera,
                                detections=(detection,),
                                depth_override_m=row_plane_depth_m,
                            )[0]
                            depth_source = "row_plane_fallback"
                        except Exception as exc:
                            partial_projection_errors.append(
                                f"{detection.item_id}: row-plane fallback: {exc}"
                            )
                            continue
                    partial_label_observations.append(
                        {
                            "item_id": detection.item_id,
                            "confidence": detection.confidence,
                            "method": detection.method,
                            "depth_source": depth_source,
                            "point_base_m": list(point),
                        }
                    )
            localization = None
            if match.accepted and localize_box_row:
                localization = matcher.localize_box_row(
                    depth_meters=depth_msg_to_meters(request.depth_image),
                    camera_k=tuple(float(value) for value in request.camera_info.k),
                    base_to_camera=transform_msg_to_matrix(request.base_to_camera),
                    detections=match.detections,
                    target_item_id=match.expected_item_id,
                    table_z_m=float(options.get("table_z_m", 0.0)),
                )
            target_present = bool(
                str(match.matched_item_id or "")
                == str(request.expected_item_id or "")
            )
            response.success = bool(
                (match.accepted and localization is not None)
                if localize_box_row
                else target_present
            )
            response.message = (
                f"label matched: {match.matched_item_id} slot={match.slot_index} "
                f"confidence={match.confidence:.3f}"
                if (match.accepted if localize_box_row else target_present)
                else f"complete six-label row not verified for {match.expected_item_id}: "
                f"detected={len(match.detected_item_ids)} confidence={match.confidence:.3f}"
            )
            response.matched_item_id = str(match.matched_item_id or "")
            response.confidence = float(match.confidence)
            response.slot_index = int(match.slot_index if match.slot_index is not None else -1)
            response.detected_label_count = len(match.detected_item_ids)
            if localization is not None:
                response.has_box_center = True
                response.box_center_base_m.x = float(localization.box_center_base_m[0])
                response.box_center_base_m.y = float(localization.box_center_base_m[1])
                response.box_center_base_m.z = float(localization.box_center_base_m[2])
            if match.bbox_xywh is not None:
                response.bbox_x, response.bbox_y, response.bbox_width, response.bbox_height = match.bbox_xywh
            response.diagnostics_json = json_dumps(
                {
                    "run_id": run_id,
                    "scene_id": request.scene_id,
                    "expected_item_id": match.expected_item_id,
                    "matched_item_id": match.matched_item_id,
                    "confidence": match.confidence,
                    "accepted": match.accepted,
                    "slot_index": match.slot_index,
                    "detected_item_ids_left_to_right": list(match.detected_item_ids),
                    "detections": [
                        {
                            "item_id": detection.item_id,
                            "confidence": detection.confidence,
                            "bbox_xywh": list(detection.bbox_xywh),
                            "method": detection.method,
                        }
                        for detection in match.detections
                    ],
                    "bottle_proposal_count": (
                        len(bottle_proposals)
                        if bottle_proposals is not None
                        else None
                    ),
                    "marker_detection_enabled": marker_detection_enabled,
                    "partial_label_observations_base_m": partial_label_observations,
                    "partial_projection_errors": partial_projection_errors,
                    "box_center_base_m": (
                        list(localization.box_center_base_m)
                        if localization is not None
                        else None
                    ),
                    "box_centers_base_m": (
                        [list(value) for value in localization.box_centers_base_m]
                        if localization is not None
                        else None
                    ),
                    "label_centers_base_m": (
                        [list(value) for value in localization.label_centers_base_m]
                        if localization is not None
                        else None
                    ),
                    "adjacent_pitch_mm": (
                        list(localization.adjacent_pitch_mm)
                        if localization is not None
                        else None
                    ),
                    "interior_direction_base": (
                        list(localization.interior_direction_base)
                        if localization is not None
                        else None
                    ),
                    "bbox_xywh": list(match.bbox_xywh) if match.bbox_xywh else None,
                    "search_roi_xywh": list(match.search_roi_xywh),
                }
            )
            self._publish_status(
                f"label_match {'ok' if match.accepted else 'failed'}: "
                f"run_id={run_id} item={match.expected_item_id} slot={match.slot_index} "
                f"detected={len(match.detected_item_ids)} confidence={match.confidence:.3f}"
            )
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            response.diagnostics_json = json_dumps(
                {"run_id": run_id, "error": str(exc), "traceback": traceback.format_exc()}
            )
        return response

    def _handle_analyze(self, request: AnalyzeScene.Request, response: AnalyzeScene.Response):
        run_id = request.run_id.strip() or "unknown"
        try:
            self._publish_status(f"analyzing: run_id={run_id}")
            result = self._bridge.run(request)

            perception = result["perception"]
            candidate_pool = result["candidate_pool"]
            candidate = result["candidate"]
            plan = result["plan"]
            diagnostics = list(result["diagnostics"])
            summary = str(result["summary"])

            self._rviz_pub.publish_result(
                {
                    "perception": perception,
                    "candidate_pool": candidate_pool,
                    "candidate": candidate,
                    "plan": plan,
                    "use_pregrasp": bool(result.get("use_pregrasp", False)),
                    "base_to_camera": transform_msg_to_matrix(request.base_to_camera),
                },
                camera_frame=str(request.camera_frame or request.color_image.header.frame_id or "camera_color_optical_frame"),
                base_frame=str(request.base_to_camera.header.frame_id or "base_link"),
                candidate_topk=5,
            )

            response.success = True
            response.message = "analysis completed"
            response.perception = make_perception_summary_msg(
                scene_id=request.scene_id,
                prompt=request.prompt,
                camera_frame=request.camera_frame,
                instance_count=len(perception.object_point_counts),
                scene_grasp_count=perception.scene_grasp_count,
                scene_point_count=perception.scene_point_count,
                object_point_counts=perception.object_point_counts,
                debug_lines=diagnostics,
            )
            response.candidate_pool = [grasp_candidate_to_msg(item[0]) for item in candidate_pool]
            response.has_selected_candidate = candidate is not None
            if candidate is not None:
                response.selected_candidate = grasp_candidate_to_msg(candidate)
            response.diagnostics_json = json_dumps({"diagnostics": diagnostics})
            response.summary = summary

            self._summary_pub.publish(String(data=summary))
            self._result_pub.publish(
                String(
                    data=json_dumps(
                        {
                            "status": "ok" if candidate is not None else "no_candidate",
                            "run_id": run_id,
                            "scene_id": request.scene_id,
                            "prompt": request.prompt,
                            "summary": summary,
                            "candidate_count": len(candidate_pool),
                            "has_plan": plan is not None,
                            "plan": (
                                {
                                    "target_base_m": list(plan.target_base_m),
                                    "target_rpy_deg": list(plan.target_rpy_deg),
                                    "pregrasp_base_m": list(plan.pregrasp_base_m),
                                    "grasp_base_m": list(plan.grasp_base_m),
                                    "retreat_base_m": list(plan.retreat_base_m),
                                    "within_workspace": plan.within_workspace,
                                    "workspace_violations": list(plan.workspace_violations),
                                }
                                if plan is not None
                                else None
                            ),
                            "diagnostics": diagnostics,
                        }
                    )
                )
            )
            self._publish_status(
                f"analysis completed: run_id={run_id} candidates={len(candidate_pool)} scene_points={perception.scene_point_count}"
            )
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            self._result_pub.publish(
                String(
                    data=json_dumps(
                        {
                            "status": "failed",
                            "run_id": run_id,
                            "message": str(exc),
                            "traceback": traceback.format_exc(),
                        }
                    )
                )
            )
            self._publish_status(f"analysis failed: {exc}")
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = VisionWorkerNode()
    executor = MultiThreadedExecutor(num_threads=2)
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
