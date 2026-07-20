from __future__ import annotations

import json
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

from robot_grasp_msgs.srv import AnalyzeScene
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
        default_conda = Path("/home/ybw/miniforge3/envs/piper/bin/python")
        if default_conda.exists():
            return str(default_conda)
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

        text_qos = make_latched_qos(depth=10)
        self._status_pub = self.create_publisher(String, "~/status", text_qos)
        self._summary_pub = self.create_publisher(String, "~/summary", text_qos)
        self._result_pub = self.create_publisher(String, "~/result_json", text_qos)
        self._rviz_pub = PipelineRvizPublisher(self)
        self.create_service(AnalyzeScene, "~/analyze", self._handle_analyze)
        self._bridge = ExternalVisionWorkerBridge(self)
        self._publish_status("idle")

    def destroy_node(self) -> None:
        self._bridge.shutdown()
        super().destroy_node()

    def _publish_status(self, text: str) -> None:
        self.get_logger().info(text)
        self._status_pub.publish(String(data=text))

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
