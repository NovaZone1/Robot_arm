from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import rclpy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from robot_grasp_msgs.srv import CaptureScene
from robot_grasp_ros2.distributed_utils import (
    color_image_to_msg,
    depth_image_to_msg,
    intrinsics_to_camera_info,
    make_latched_qos,
)


@dataclass(slots=True)
class _IntrinsicsPayload:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float


class ExternalCameraBridge:
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
        return str(Path(__file__).resolve().parents[1] / "src" / "perception" / "external_camera_capture_worker.py")

    def _worker_timeout(self) -> float:
        return max(1.0, float(self._node.get_parameter("worker_timeout_s").value))

    def _worker_mode(self) -> str:
        try:
            return str(self._node.get_parameter("worker_mode").value or "daemon").strip().lower()
        except Exception:
            return "daemon"

    def _ensure_daemon(self) -> subprocess.Popen:
        if self._daemon_proc is not None and self._daemon_proc.poll() is None:
            return self._daemon_proc
        worker_python = self._worker_python()
        worker_script = self._worker_script()
        self._node.get_logger().info(
            f"[camera_bridge] starting camera daemon: {worker_python} {worker_script} --daemon"
        )
        proc = subprocess.Popen(
            [worker_python, worker_script, "--daemon"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            cwd=str(Path(__file__).resolve().parents[1]),
        )
        # Ping to confirm the daemon is alive (camera opens on first capture request).
        deadline = time.monotonic() + 30.0
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(
                    f"camera daemon exited unexpectedly (rc={proc.returncode}) during startup"
                )
            try:
                proc.stdin.write(b'{"type":"ping"}\n')
                proc.stdin.flush()
                proc.stdout.readline()
                break
            except Exception:
                time.sleep(0.2)
        else:
            proc.kill()
            raise RuntimeError("camera daemon did not respond to ping within 30 s")
        self._daemon_proc = proc
        self._node.get_logger().info("[camera_bridge] camera daemon is ready")
        return proc

    def _call_daemon(self, request: dict[str, object]) -> dict[str, object]:
        import select
        proc = self._ensure_daemon()
        line = json.dumps(request, ensure_ascii=False) + "\n"
        try:
            proc.stdin.write(line.encode("utf-8"))
            proc.stdin.flush()
        except BrokenPipeError:
            self._daemon_proc = None
            raise RuntimeError("camera daemon pipe broken; will restart on next call")
        timeout = self._worker_timeout()
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                self._daemon_proc = None
                raise RuntimeError(f"camera daemon exited unexpectedly (rc={proc.returncode})")
            ready, _, _ = select.select([proc.stdout], [], [], 0.1)
            if ready:
                raw = proc.stdout.readline()
                if not raw:
                    self._daemon_proc = None
                    raise RuntimeError("camera daemon closed stdout unexpectedly")
                return json.loads(raw.decode("utf-8", errors="replace"))
        raise TimeoutError(f"camera daemon did not respond within {timeout:.0f} s")

    def shutdown(self) -> None:
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

    def capture(
        self,
        *,
        depth_fusion_frames: int,
        pointcloud_filter_mode: str,
        width: int,
        height: int,
        fps: int,
        clip_max_m: float,
        bilateral_diameter: int,
        bilateral_sigma_color: float,
        bilateral_sigma_space: float,
        median_kernel_size: int,
    ) -> tuple[np.ndarray, np.ndarray, _IntrinsicsPayload]:
        worker_python = self._worker_python()
        worker_script = self._worker_script()
        if shutil.which(worker_python) is None and not Path(worker_python).exists():
            raise RuntimeError(f"camera worker python executable not found: {worker_python}")
        if not Path(worker_script).is_file():
            raise RuntimeError(f"camera worker script not found: {worker_script}")

        with tempfile.TemporaryDirectory(prefix="camera_capture_") as tmp_dir_str:
            tmp_dir = Path(tmp_dir_str)
            request_payload = {
                "camera_width": int(width),
                "camera_height": int(height),
                "camera_fps": int(fps),
                "clip_max_m": float(clip_max_m),
                "depth_fusion_frames": int(depth_fusion_frames),
                "pointcloud_filter_mode": str(pointcloud_filter_mode),
                "bilateral_diameter": int(bilateral_diameter),
                "bilateral_sigma_color": float(bilateral_sigma_color),
                "bilateral_sigma_space": float(bilateral_sigma_space),
                "median_kernel_size": int(median_kernel_size),
                "work_dir": str(tmp_dir),
            }

            use_daemon = self._worker_mode() == "daemon"
            if use_daemon:
                with self._daemon_lock:
                    payload = self._call_daemon(request_payload)
            else:
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
                        "external camera worker did not produce a response. "
                        f"exit_code={completed.returncode} "
                        f"stdout={completed.stdout.strip()!r} stderr={completed.stderr.strip()!r}"
                    )
                payload = json.loads(response_path.read_text(encoding="utf-8"))

            if not bool(payload.get("success")):
                message = str(payload.get("message", "external camera worker failed"))
                trace = str(payload.get("traceback", "")).strip()
                details = [message]
                if trace:
                    details.append(trace)
                raise RuntimeError("\n".join(details))

            color_bgr = np.load(tmp_dir / str(payload["color_npy"]))
            depth_meters = np.load(tmp_dir / str(payload["depth_npy"]))
            intrinsics_payload = dict(payload["intrinsics"])
            intrinsics = _IntrinsicsPayload(
                width=int(intrinsics_payload["width"]),
                height=int(intrinsics_payload["height"]),
                fx=float(intrinsics_payload["fx"]),
                fy=float(intrinsics_payload["fy"]),
                ppx=float(intrinsics_payload["ppx"]),
                ppy=float(intrinsics_payload["ppy"]),
            )
            return color_bgr, depth_meters, intrinsics


class CameraServerNode(Node):
    """Capture RGBD frames and expose them through a typed ROS2 service."""

    def __init__(self) -> None:
        super().__init__("camera_server")
        self.declare_parameter("worker_python_executable", "")
        self.declare_parameter("worker_script", "")
        self.declare_parameter("worker_timeout_s", 60.0)
        self.declare_parameter("worker_mode", "daemon")
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("camera_fps", 30)
        self.declare_parameter("clip_max_m", 3.0)
        self.declare_parameter("camera_frame", "camera_color_optical_frame")
        self.declare_parameter("bilateral_diameter", 5)
        self.declare_parameter("bilateral_sigma_color", 0.02)
        self.declare_parameter("bilateral_sigma_space", 5.0)
        self.declare_parameter("median_kernel_size", 5)
        # A lightweight RGB-D preview stream lets search logic inspect the
        # scene while Scout is moving.  It shares the persistent camera daemon
        # with the explicit capture service, so final RGB-D captures retain
        # their existing semantics.
        self.declare_parameter("continuous_preview_enabled", True)
        self.declare_parameter("continuous_preview_rate_hz", 4.0)
        self.declare_parameter("continuous_preview_depth_fusion_frames", 1)

        text_qos = make_latched_qos(depth=10)
        image_qos = make_latched_qos(depth=1)
        self._status_pub = self.create_publisher(String, "~/status", text_qos)
        self._color_pub = self.create_publisher(Image, "~/latest/color", image_qos)
        self._depth_pub = self.create_publisher(Image, "~/latest/depth", image_qos)
        self._camera_info_pub = self.create_publisher(CameraInfo, "~/latest/camera_info", image_qos)
        self.create_service(CaptureScene, "~/capture", self._handle_capture)

        self._bridge = ExternalCameraBridge(self)
        self._preview_lock = threading.Lock()
        self._last_preview_error_at = 0.0
        self._preview_callback_group = ReentrantCallbackGroup()
        preview_rate_hz = max(
            0.2, float(self.get_parameter("continuous_preview_rate_hz").value)
        )
        self._preview_timer = self.create_timer(
            1.0 / preview_rate_hz,
            self._publish_continuous_preview,
            callback_group=self._preview_callback_group,
        )
        self._publish_status("idle")

    def destroy_node(self) -> None:
        self._bridge.shutdown()
        super().destroy_node()

    def _publish_status(self, text: str) -> None:
        self.get_logger().info(text)
        self._status_pub.publish(String(data=text))

    def _camera_frame(self) -> str:
        return str(self.get_parameter("camera_frame").value or "camera_color_optical_frame")

    def _capture_arrays(
        self,
        *,
        depth_fusion_frames: int,
        pointcloud_filter_mode: str = "bilateral",
    ):
        return self._bridge.capture(
            depth_fusion_frames=max(1, int(depth_fusion_frames)),
            pointcloud_filter_mode=str(pointcloud_filter_mode or "bilateral"),
            width=int(self.get_parameter("camera_width").value),
            height=int(self.get_parameter("camera_height").value),
            fps=int(self.get_parameter("camera_fps").value),
            clip_max_m=float(self.get_parameter("clip_max_m").value),
            bilateral_diameter=int(self.get_parameter("bilateral_diameter").value),
            bilateral_sigma_color=float(self.get_parameter("bilateral_sigma_color").value),
            bilateral_sigma_space=float(self.get_parameter("bilateral_sigma_space").value),
            median_kernel_size=int(self.get_parameter("median_kernel_size").value),
        )

    def _publish_continuous_preview(self) -> None:
        if not bool(self.get_parameter("continuous_preview_enabled").value):
            return
        if not self._preview_lock.acquire(blocking=False):
            return
        try:
            color_bgr, depth_meters, intrinsics = self._capture_arrays(
                depth_fusion_frames=int(
                    self.get_parameter("continuous_preview_depth_fusion_frames").value
                )
            )
            stamp = self.get_clock().now().to_msg()
            frame_id = self._camera_frame()
            self._color_pub.publish(
                color_image_to_msg(color_bgr, frame_id=frame_id, stamp=stamp)
            )
            # Publish depth/info as well so RViz and diagnostics stay in sync;
            # the target-search pipeline consumes RGB only during motion.
            self._depth_pub.publish(
                depth_image_to_msg(depth_meters, frame_id=frame_id, stamp=stamp)
            )
            self._camera_info_pub.publish(
                intrinsics_to_camera_info(intrinsics, frame_id=frame_id, stamp=stamp)
            )
        except Exception as exc:
            now = time.monotonic()
            if now - self._last_preview_error_at >= 5.0:
                self.get_logger().warning(f"continuous preview capture failed: {exc}")
                self._last_preview_error_at = now
        finally:
            self._preview_lock.release()

    def _handle_capture(self, request: CaptureScene.Request, response: CaptureScene.Response):
        run_id = request.run_id.strip() or f"capture-{int(time.time() * 1000)}"
        try:
            self._publish_status(f"capturing: run_id={run_id}")
            color_bgr, depth_meters, intrinsics = self._capture_arrays(
                depth_fusion_frames=int(request.depth_fusion_frames),
                pointcloud_filter_mode=str(
                    request.pointcloud_filter_mode or "bilateral"
                ),
            )

            stamp = self.get_clock().now().to_msg()
            frame_id = self._camera_frame()
            color_msg = color_image_to_msg(color_bgr, frame_id=frame_id, stamp=stamp)
            depth_msg = depth_image_to_msg(depth_meters, frame_id=frame_id, stamp=stamp)
            camera_info_msg = intrinsics_to_camera_info(intrinsics, frame_id=frame_id, stamp=stamp)

            self._color_pub.publish(color_msg)
            self._depth_pub.publish(depth_msg)
            self._camera_info_pub.publish(camera_info_msg)

            response.success = True
            response.message = "capture completed"
            response.scene_id = f"{run_id}-scene"
            response.camera_frame = frame_id
            response.color_image = color_msg
            response.depth_image = depth_msg
            response.camera_info = camera_info_msg
            self._publish_status(
                f"capture completed: run_id={run_id} size={color_bgr.shape[1]}x{color_bgr.shape[0]}"
            )
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            self._publish_status(f"capture failed: {exc}")
        return response


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CameraServerNode()
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
