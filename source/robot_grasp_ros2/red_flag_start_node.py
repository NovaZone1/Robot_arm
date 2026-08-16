from __future__ import annotations

import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

from robot_grasp_msgs.srv import CaptureScene
from src.perception.red_flag_signal import (
    RedFlagSignalConfig,
    RedFlagWaveTracker,
    detect_red_flag,
)


class RedFlagStartGate(Node):
    def __init__(self) -> None:
        super().__init__("red_flag_start_gate")
        self.declare_parameter("camera_capture_service", "/camera_server/capture")
        self.declare_parameter("use_continuous_preview", True)
        self.declare_parameter("continuous_preview_image_topic", "/camera_server/latest/color")
        self.declare_parameter("continuous_preview_max_age_s", 0.8)
        self.declare_parameter("continuous_preview_wait_s", 1.0)
        self.declare_parameter("timeout_s", 180.0)
        self.declare_parameter("capture_timeout_s", 8.0)
        self.declare_parameter("sample_interval_s", 0.08)
        self.declare_parameter("roi_norm", [0.10, 0.0, 1.0, 0.95])
        self.declare_parameter("saturation_min", 110)
        self.declare_parameter("value_min", 90)
        self.declare_parameter("component_min_area_ratio", 0.002)
        self.declare_parameter("peak_area_ratio", 0.03)
        self.declare_parameter("motion_window_s", 3.0)
        self.declare_parameter("min_valid_detections", 4)
        self.declare_parameter("min_axis_range_norm", 0.18)
        self.declare_parameter("min_path_length_norm", 0.30)
        self.declare_parameter("min_direction_step_norm", 0.025)
        self.declare_parameter("min_direction_reversals", 1)
        self.declare_parameter("artifact_dir", "")
        self._capture_client = self.create_client(
            CaptureScene,
            str(self.get_parameter("camera_capture_service").value),
        )
        self._preview_lock = threading.Lock()
        self._preview_image: np.ndarray | None = None
        self._preview_received_monotonic = 0.0
        self._preview_sequence = 0
        self._preview_subscription = self.create_subscription(
            Image,
            str(self.get_parameter("continuous_preview_image_topic").value),
            self._on_preview_image,
            qos_profile_sensor_data,
        )

    def _on_preview_image(self, message: Image) -> None:
        try:
            image = self._image_from_message(message).copy()
        except Exception as exc:
            self.get_logger().warning(f"ignoring invalid continuous preview frame: {exc}")
            return
        with self._preview_lock:
            self._preview_image = image
            self._preview_received_monotonic = time.monotonic()
            self._preview_sequence += 1

    def _config(self) -> RedFlagSignalConfig:
        roi = tuple(float(value) for value in self.get_parameter("roi_norm").value)
        if len(roi) != 4:
            raise RuntimeError("roi_norm must contain [x0, y0, x1, y1]")
        return RedFlagSignalConfig(
            roi_norm=roi,
            saturation_min=int(self.get_parameter("saturation_min").value),
            value_min=int(self.get_parameter("value_min").value),
            component_min_area_ratio=float(
                self.get_parameter("component_min_area_ratio").value
            ),
            peak_area_ratio=float(self.get_parameter("peak_area_ratio").value),
            motion_window_s=float(self.get_parameter("motion_window_s").value),
            min_valid_detections=int(
                self.get_parameter("min_valid_detections").value
            ),
            min_axis_range_norm=float(
                self.get_parameter("min_axis_range_norm").value
            ),
            min_path_length_norm=float(
                self.get_parameter("min_path_length_norm").value
            ),
            min_direction_step_norm=float(
                self.get_parameter("min_direction_step_norm").value
            ),
            min_direction_reversals=int(
                self.get_parameter("min_direction_reversals").value
            ),
        )

    @staticmethod
    def _image_from_message(message) -> np.ndarray:
        image = np.frombuffer(bytes(message.data), dtype=np.uint8).reshape(
            message.height,
            message.step,
        )
        image = image[:, : message.width * 3].reshape(
            message.height,
            message.width,
            3,
        )
        if message.encoding == "rgb8":
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif message.encoding != "bgr8":
            raise RuntimeError(f"unsupported color encoding: {message.encoding}")
        return image

    def _capture(self, index: int) -> np.ndarray:
        request = CaptureScene.Request()
        request.run_id = f"red-flag-gate-{index:04d}"
        request.depth_fusion_frames = 1
        request.pointcloud_filter_mode = "bilateral"
        request.pointcloud_backend = "sdk"
        future = self._capture_client.call_async(request)
        rclpy.spin_until_future_complete(
            self,
            future,
            timeout_sec=float(self.get_parameter("capture_timeout_s").value),
        )
        response = future.result()
        if response is None:
            raise RuntimeError("camera capture timed out")
        if not response.success:
            raise RuntimeError(response.message)
        return self._image_from_message(response.color_image)

    def _next_preview_image(self, after_sequence: int) -> tuple[np.ndarray | None, int]:
        """Return one fresh camera-server preview frame, without competing for RealSense."""
        if not bool(self.get_parameter("use_continuous_preview").value):
            return None, after_sequence
        deadline = time.monotonic() + max(
            0.0, float(self.get_parameter("continuous_preview_wait_s").value)
        )
        max_age_s = max(0.05, float(self.get_parameter("continuous_preview_max_age_s").value))
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            with self._preview_lock:
                image = self._preview_image
                received = self._preview_received_monotonic
                sequence = self._preview_sequence
                if (
                    image is not None
                    and sequence > after_sequence
                    and time.monotonic() - received <= max_age_s
                ):
                    return image.copy(), sequence
        return None, after_sequence

    def wait(self) -> dict[str, object]:
        use_preview = bool(self.get_parameter("use_continuous_preview").value)
        if not use_preview and not self._capture_client.wait_for_service(timeout_sec=10.0):
            raise RuntimeError("/camera_server/capture is unavailable")
        config = self._config()
        tracker = RedFlagWaveTracker(config)
        timeout_s = max(1.0, float(self.get_parameter("timeout_s").value))
        deadline = time.monotonic() + timeout_s
        index = 0
        last_image = None
        last_detection = None
        preview_sequence = 0
        started = time.monotonic()
        self.get_logger().info(
            "armed: waiting for a waved red flag; navigation remains stopped"
        )
        while rclpy.ok() and time.monotonic() < deadline:
            image, preview_sequence = self._next_preview_image(preview_sequence)
            if image is None:
                # A preview can be absent during camera start-up.  Preserve the old
                # service path as a safe fallback rather than failing the start gate.
                if not self._capture_client.wait_for_service(timeout_sec=1.0):
                    raise RuntimeError(
                        "no fresh continuous preview and /camera_server/capture is unavailable"
                    )
                image = self._capture(index)
            detection, _mask = detect_red_flag(image, config)
            triggered = tracker.update(time.monotonic(), detection)
            last_image = image
            last_detection = detection
            metrics = tracker.last_metrics
            self.get_logger().info(
                "frame=%d red=%s area=%.3f center=(%.3f,%.3f) "
                "range=(%.3f,%.3f) path=%.3f reversals=%d"
                % (
                    index,
                    detection.found,
                    detection.area_ratio,
                    detection.center_u_norm,
                    detection.center_v_norm,
                    float(metrics.get("x_range_norm", 0.0)),
                    float(metrics.get("y_range_norm", 0.0)),
                    float(metrics.get("path_length_norm", 0.0)),
                    int(metrics.get("direction_reversals", 0)),
                )
            )
            if triggered:
                payload = {
                    "success": True,
                    "message": "waved red flag detected",
                    "elapsed_s": time.monotonic() - started,
                    "frame_index": index,
                    "detection": detection.__dict__,
                    "metrics": dict(metrics),
                }
                self._save_artifact(payload, last_image, last_detection)
                return payload
            index += 1
            time.sleep(max(0.0, float(self.get_parameter("sample_interval_s").value)))
        payload = {
            "success": False,
            "message": f"red flag wait timed out after {timeout_s:.1f}s",
            "elapsed_s": time.monotonic() - started,
            "frame_index": index,
            "metrics": dict(tracker.last_metrics),
        }
        self._save_artifact(payload, last_image, last_detection)
        return payload

    def _artifact_directory(self) -> Path:
        configured = str(self.get_parameter("artifact_dir").value or "").strip()
        if configured:
            return Path(configured).expanduser().resolve()
        return (
            Path(__file__).resolve().parents[1]
            / "config/calibration/red_flag/runtime"
        )

    def _save_artifact(self, payload, image, detection) -> None:
        output_dir = self._artifact_directory()
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "latest.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if image is None:
            return
        overlay = np.asarray(image).copy()
        if detection is not None and detection.bbox_xywh is not None:
            x, y, width, height = detection.bbox_xywh
            cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 0), 3)
        cv2.putText(
            overlay,
            "START" if bool(payload.get("success")) else "WAITING",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 0) if bool(payload.get("success")) else (0, 200, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imwrite(str(output_dir / "latest.png"), overlay)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RedFlagStartGate()
    exit_code = 1
    try:
        result = node.wait()
        print(json.dumps(result, ensure_ascii=False))
        exit_code = 0 if bool(result.get("success")) else 2
    except KeyboardInterrupt:
        node.get_logger().warning("red flag wait cancelled")
        exit_code = 130
    except Exception as exc:
        node.get_logger().error(str(exc))
        exit_code = 3
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
