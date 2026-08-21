#!/usr/bin/env python3
"""Record label (u, v) and taught release TCP poses for placement mapping.

Keep the arm at the same observation pose for every capture. After each
capture, jog the arm to the box opening and record the TCP. Six scattered
samples are enough to fit (u, v) -> (X, Y).

Usage:
  scripts/record_placement_uv_xy.sh --item orange_bottle
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node

from robot_grasp_msgs.msg import Pose6D
from robot_grasp_msgs.srv import (
    CaptureScene,
    ExecuteNamedPose,
    GetRobotState,
    MatchItemLabel,
)
from robot_grasp_ros2.distributed_utils import color_msg_to_bgr, json_dumps

try:
    from piper_msgs.srv import Enable
except Exception:  # pragma: no cover
    Enable = None

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None

PROJECT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = PROJECT / "config" / "calibration" / "placement_uv_xy"
DEFAULT_OBSERVE_POSE = (0.0, -35.5, 491.1, 180.0, 67.77, 89.97)
SUGGESTED_TAGS = ("left", "center", "right", "near", "far", "diagonal")
DEFAULT_ORIENTATION_TOLERANCE_DEG = 15.0
VALID_ITEMS = (
    "red_block",
    "yellow_block",
    "blue_block",
    "orange_bottle",
    "dark_bottle",
    "green_bottle",
)


def _say(text: str) -> None:
    print(text, flush=True)


def _prompt(text: str) -> str:
    try:
        return input(text)
    except EOFError:
        return "q"


def _pose_to_dict(pose: Pose6D) -> dict[str, float]:
    return {
        "x_mm": float(pose.x_mm),
        "y_mm": float(pose.y_mm),
        "z_mm": float(pose.z_mm),
        "roll_deg": float(pose.roll_deg),
        "pitch_deg": float(pose.pitch_deg),
        "yaw_deg": float(pose.yaw_deg),
    }


def _angle_delta_deg(first: float, second: float) -> float:
    """Smallest signed angular difference, robust at the +/-180 boundary."""
    return ((float(first) - float(second) + 180.0) % 360.0) - 180.0


class PlacementUvRecorder(Node):
    def __init__(self, args: argparse.Namespace) -> None:
        super().__init__("placement_uv_xy_recorder")
        self._item_id = str(args.item)
        self._output_dir = Path(args.output_dir).expanduser().resolve() / self._item_id
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self._observe = tuple(float(value) for value in args.observe_pose)
        self._speed = float(args.speed_percent)
        self._orientation_tolerance_deg = float(args.orientation_tolerance_deg)
        self._samples_path = self._output_dir / "samples.json"
        self._samples: list[dict[str, object]] = []
        if self._samples_path.is_file() and not bool(args.reset):
            payload = json.loads(self._samples_path.read_text(encoding="utf-8"))
            self._samples = list(payload.get("samples") or [])
        self._pending_view: dict[str, object] | None = None

        self._capture = self.create_client(CaptureScene, "/camera_server/capture")
        self._match = self.create_client(MatchItemLabel, "/vision_worker/match_item_label")
        self._state = self.create_client(GetRobotState, "/robot_executor/get_state")
        self._named = self.create_client(
            ExecuteNamedPose, "/robot_executor/execute_named_pose"
        )
        self._enable = (
            self.create_client(Enable, "/enable_srv") if Enable is not None else None
        )

    def _wait(self, client, name: str, timeout_s: float = 10.0) -> None:
        if not client.wait_for_service(timeout_sec=timeout_s):
            raise RuntimeError(f"{name} is unavailable")

    def _call(self, client, request, timeout_s: float):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        if not future.done() or future.result() is None:
            raise RuntimeError(f"service call timed out after {timeout_s:.0f}s")
        return future.result()

    def connect(self) -> None:
        self._wait(self._capture, "/camera_server/capture")
        self._wait(self._match, "/vision_worker/match_item_label")
        self._wait(self._state, "/robot_executor/get_state")
        self._wait(self._named, "/robot_executor/execute_named_pose")
        if self._enable is not None:
            self._wait(self._enable, "/enable_srv")

    def enable_arm(self) -> None:
        if self._enable is None:
            _say("当前环境没有 /enable_srv，跳过使能")
            return
        request = Enable.Request()
        request.enable_request = True
        response = self._call(self._enable, request, timeout_s=8.0)
        if not bool(getattr(response, "enable_response", False)):
            raise RuntimeError("Piper enable failed")

    def get_tcp(self) -> dict[str, float]:
        response = self._call(self._state, GetRobotState.Request(), timeout_s=8.0)
        if not response.success:
            raise RuntimeError(response.message)
        return _pose_to_dict(response.current_pose)

    def _send_observation(self):
        request = ExecuteNamedPose.Request()
        request.name = "placement_observation"
        request.speed_percent = self._speed
        request.open_gripper_first = False
        request.pose.x_mm, request.pose.y_mm, request.pose.z_mm = self._observe[:3]
        request.pose.roll_deg, request.pose.pitch_deg, request.pose.yaw_deg = (
            self._observe[3:]
        )
        return self._call(self._named, request, timeout_s=60.0)

    def _observation_should_retry(self, message: str) -> bool:
        text = str(message or "").lower()
        return any(
            token in text
            for token in ("stalled", "timeout", "arrived", "not arrived", "enable")
        )

    def move_to_observation(self) -> dict[str, float]:
        current = self.get_tcp()
        _say(
            "当前 TCP "
            f"X={current['x_mm']:.1f} Y={current['y_mm']:.1f} Z={current['z_mm']:.1f} "
            f"rpy=({current['roll_deg']:.1f},{current['pitch_deg']:.1f},{current['yaw_deg']:.1f})"
        )
        last_error = "execute_named_pose failed"
        for attempt in range(1, 6):
            _say(f"正在使能机械臂（第 {attempt}/5 次）…")
            try:
                self.enable_arm()
            except Exception as exc:
                last_error = str(exc)
                _say(f"使能未成功: {exc}")
                time.sleep(0.6)
                continue
            time.sleep(0.4)
            if attempt == 1:
                _say(
                    "正在去观察位 "
                    f"[{self._observe[0]:.1f}, {self._observe[1]:.1f}, {self._observe[2]:.1f}, "
                    f"{self._observe[3]:.1f}, {self._observe[4]:.1f}, {self._observe[5]:.1f}] "
                    f"speed={self._speed:.0f}%"
                )
            else:
                _say("示教残留到位信号，重新下发观察位…")
            response = self._send_observation()
            if response.success:
                return _pose_to_dict(response.actual_pose)
            last_error = str(response.message or "execute_named_pose failed")
            if not self._observation_should_retry(last_error):
                raise RuntimeError(last_error)
            _say(f"观察位未吃进（{last_error}）；0.6s 后重试")
            time.sleep(0.6)
        raise RuntimeError(last_error)

    def capture_label(self) -> dict[str, object]:
        capture_req = CaptureScene.Request()
        capture_req.run_id = f"uv-xy-{int(time.time() * 1000)}"
        capture_req.depth_fusion_frames = 1
        capture_req.pointcloud_filter_mode = "bilateral"
        capture_req.pointcloud_backend = "sdk"
        capture = self._call(self._capture, capture_req, timeout_s=30.0)
        if not capture.success:
            raise RuntimeError(capture.message)
        color = color_msg_to_bgr(capture.color_image)
        match_req = MatchItemLabel.Request()
        match_req.run_id = capture_req.run_id
        match_req.scene_id = str(capture.scene_id)
        match_req.expected_item_id = self._item_id
        match_req.color_image = capture.color_image
        match_req.depth_image = capture.depth_image
        match_req.camera_info = capture.camera_info
        match_req.options_json = json_dumps(
            {
                "label_search_roi_norm": [0.0, 0.0, 1.0, 1.0],
                "label_match_threshold": 0.42,
                "label_marker_detection_enabled": False,
                "localize_box_row": False,
            }
        )
        match = self._call(self._match, match_req, timeout_s=30.0)
        bbox = [
            int(match.bbox_x),
            int(match.bbox_y),
            int(match.bbox_width),
            int(match.bbox_height),
        ]
        if str(match.matched_item_id or "") != self._item_id or min(bbox[2], bbox[3]) <= 1:
            debug_name = f"capture_failed_{int(time.time() * 1000)}"
            debug_image = self._output_dir / f"{debug_name}_color.png"
            debug_json = self._output_dir / f"{debug_name}_match.json"
            if cv2 is not None:
                cv2.imwrite(str(debug_image), color)
            debug_json.write_text(
                str(getattr(match, "diagnostics_json", "") or match.message or ""),
                encoding="utf-8",
            )
            raise RuntimeError(
                f"{match.message or f'did not detect {self._item_id} in this view'} "
                f"(saved diagnostic: {debug_image.name}, {debug_json.name})"
            )
        u_px = float(bbox[0]) + (0.5 * float(bbox[2]))
        v_px = float(bbox[1]) + (0.5 * float(bbox[3]))
        width = int(capture.camera_info.width or capture.color_image.width)
        height = int(capture.camera_info.height or capture.color_image.height)
        sample_index = len(self._samples) + 1
        color_name = f"sample_{sample_index:02d}_color.png"
        overlay_name = f"sample_{sample_index:02d}_overlay.png"
        if cv2 is not None:
            overlay = color.copy()
            x, y, box_w, box_h = bbox
            cv2.rectangle(overlay, (x, y), (x + box_w, y + box_h), (0, 220, 0), 2)
            cv2.circle(overlay, (int(round(u_px)), int(round(v_px))), 5, (0, 0, 255), -1)
            cv2.imwrite(str(self._output_dir / color_name), color)
            cv2.imwrite(str(self._output_dir / overlay_name), overlay)
        view = {
            "item_id": self._item_id,
            "u_px": u_px,
            "v_px": v_px,
            "u_norm": u_px / max(1.0, float(width)),
            "v_norm": v_px / max(1.0, float(height)),
            "bbox_xywh": bbox,
            "confidence": float(match.confidence),
            "image_width": width,
            "image_height": height,
            "color_image": color_name,
            "overlay_image": overlay_name,
            "observe_pose_mm_deg": self.get_tcp(),
            "captured_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self._pending_view = view
        return view

    def record_release(self, tag: str) -> dict[str, object]:
        if self._pending_view is None:
            raise RuntimeError("capture a label view first")
        sample = dict(self._pending_view)
        sample["tag"] = str(tag or f"sample_{len(self._samples) + 1:02d}")
        release_pose = self.get_tcp()
        # A UV->XY fit assumes one repeatable tool attitude.  A manually
        # rotated wrist may still look like a valid release point, but mixing
        # it into the fit produces an approach that sweeps into the label
        # baffle.  Use the first taught sample as the reference and reject
        # accidental changes of orientation before they poison the map.
        if self._samples and self._orientation_tolerance_deg > 0.0:
            reference = dict(self._samples[0].get("release_pose_mm_deg") or {})
            deltas = {
                axis: abs(_angle_delta_deg(release_pose[axis], float(reference.get(axis) or 0.0)))
                for axis in ("roll_deg", "pitch_deg", "yaw_deg")
            }
            worst_axis, worst_delta = max(deltas.items(), key=lambda pair: pair[1])
            if worst_delta > self._orientation_tolerance_deg:
                raise RuntimeError(
                    "放下姿态与首个样本不一致 "
                    f"({worst_axis} 相差 {worst_delta:.1f}°, "
                    f"限值 {self._orientation_tolerance_deg:.1f}°)。"
                    "请只平移末端到盒内安全释放点，不要扭转腕部；"
                    "若首样本本身不对，请 reset 后从正确中心样本重新采集。"
                )
        sample["release_pose_mm_deg"] = release_pose
        sample["recorded_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self._samples.append(sample)
        self._pending_view = None
        self._save()
        return sample

    def undo(self) -> None:
        if not self._samples:
            raise RuntimeError("no samples to undo")
        self._samples.pop()
        self._pending_view = None
        self._save()

    def _save(self) -> None:
        # A calibration set may be cleared while an interactive recorder is
        # still open.  Keep the pending capture usable instead of losing it on
        # the first `record` command.
        self._output_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "item_id": self._item_id,
            "observe_pose_mm_deg": list(self._observe),
            "sample_count": len(self._samples),
            "samples": self._samples,
        }
        self._samples_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    def print_samples(self) -> None:
        if not self._samples:
            print("还没有样本。")
            return
        print(f"{'n':>2}  {'tag':<10}  {'u':>7}  {'v':>7}  {'X':>8}  {'Y':>8}  {'Z':>8}")
        for index, sample in enumerate(self._samples, start=1):
            pose = dict(sample.get("release_pose_mm_deg") or {})
            print(
                f"{index:2d}  {str(sample.get('tag') or ''):<10}  "
                f"{float(sample['u_px']):7.1f}  {float(sample['v_px']):7.1f}  "
                f"{float(pose.get('x_mm') or 0.0):8.1f}  "
                f"{float(pose.get('y_mm') or 0.0):8.1f}  "
                f"{float(pose.get('z_mm') or 0.0):8.1f}"
            )
        print(f"已保存 {self._samples_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record placement (u, v) and taught release TCP samples."
    )
    parser.add_argument("--item", required=True, choices=VALID_ITEMS)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--observe-pose",
        nargs=6,
        type=float,
        default=list(DEFAULT_OBSERVE_POSE),
        metavar=("X", "Y", "Z", "R", "P", "YAW"),
        help="Observation pose in mm/deg. Default is the current right-side view.",
    )
    parser.add_argument("--speed-percent", type=float, default=25.0)
    parser.add_argument(
        "--orientation-tolerance-deg",
        type=float,
        default=DEFAULT_ORIENTATION_TOLERANCE_DEG,
        help="Maximum roll/pitch/yaw deviation from the first release sample; 0 disables it.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Ignore any previously saved samples for this item.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    rclpy.init()
    node = PlacementUvRecorder(args)
    try:
        node.connect()
        _say(f"采集目标: {args.item}")
        _say(f"保存目录: {node._output_dir}")
        _say("建议 6 组标签: " + ", ".join(SUGGESTED_TAGS))
        _say(
            "命令: observe | capture | record [tag] | list | undo | quit\n"
            "流程: observe → 确认画面 → capture → 手掰到放下点 → record left"
        )
        node.print_samples()
        while True:
            raw = _prompt("uv-xy> ").strip()
            if not raw:
                continue
            parts = raw.split()
            command = parts[0].lower()
            if command in {"q", "quit", "exit"}:
                node._save()
                _say(f"已保存 {len(node._samples)} 组到 {node._samples_path}")
                return 0
            try:
                if command in {"o", "observe"}:
                    pose = node.move_to_observation()
                    _say(
                        "已到观察位 "
                        f"X={pose['x_mm']:.1f} Y={pose['y_mm']:.1f} Z={pose['z_mm']:.1f}"
                    )
                elif command in {"c", "capture"}:
                    view = node.capture_label()
                    _say(
                        f"检测到 {args.item} 中心 u={view['u_px']:.1f} v={view['v_px']:.1f} "
                        f"conf={float(view['confidence']):.3f}"
                    )
                    _say("请把末端掰到这个盒子的放下点，然后输入 record <tag>")
                elif command in {"r", "record"}:
                    tag = parts[1] if len(parts) > 1 else (
                        SUGGESTED_TAGS[len(node._samples)]
                        if len(node._samples) < len(SUGGESTED_TAGS)
                        else f"sample_{len(node._samples) + 1:02d}"
                    )
                    sample = node.record_release(tag)
                    pose = sample["release_pose_mm_deg"]
                    _say(
                        f"已记录 {tag}: u={sample['u_px']:.1f} v={sample['v_px']:.1f} "
                        f"-> X={pose['x_mm']:.1f} Y={pose['y_mm']:.1f} Z={pose['z_mm']:.1f}"
                    )
                    node.print_samples()
                    if len(node._samples) >= 6:
                        _say("已有 6 组，可以 quit。还想补点就继续 observe。")
                elif command in {"l", "list"}:
                    node.print_samples()
                elif command in {"u", "undo"}:
                    node.undo()
                    _say("已删除上一组。")
                    node.print_samples()
                elif command in {"h", "help"}:
                    _say("observe  使能并回到观察位")
                    _say("capture  拍照并识别标签中心")
                    _say("record [tag]  记录当前 TCP 为放下点")
                    _say("list / undo / quit")
                else:
                    _say(f"未知命令: {command}  (help 查看)")
            except Exception as exc:
                _say(f"失败: {exc}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
