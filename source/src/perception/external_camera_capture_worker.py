from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
import traceback

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.realsense_rgbd import RealSenseRGBDCamera


def _json_dumps(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _bilateral_filter_depth(
    depth_meters: np.ndarray,
    diameter: int,
    sigma_color: float,
    sigma_space: float,
) -> np.ndarray:
    depth = np.asarray(depth_meters, dtype=np.float32)
    valid_mask = depth > 0
    if not np.any(valid_mask):
        return depth.copy()

    filtered = cv2.bilateralFilter(
        depth,
        d=max(1, int(diameter)),
        sigmaColor=float(sigma_color),
        sigmaSpace=float(sigma_space),
    )
    filtered[~valid_mask] = 0.0
    return filtered.astype(np.float32, copy=False)


def _median_filter_depth(depth_meters: np.ndarray, kernel_size: int) -> np.ndarray:
    depth = np.asarray(depth_meters, dtype=np.float32)
    valid_mask = depth > 0
    if not np.any(valid_mask):
        return depth.copy()

    kernel = max(1, int(kernel_size))
    if kernel % 2 == 0:
        kernel += 1
    filtered = cv2.medianBlur(depth, kernel)
    filtered[~valid_mask] = 0.0
    return filtered.astype(np.float32, copy=False)


def _capture_fused_rgbd(camera: RealSenseRGBDCamera, *, depth_fusion_frames: int) -> tuple[np.ndarray, np.ndarray]:
    frame_count = max(1, int(depth_fusion_frames))
    color_bgr = None
    depth_stack: list[np.ndarray] = []
    for _ in range(frame_count):
        ok, color_frame, _depth_raw, depth_meters = camera.get_frames()
        if not ok or color_frame is None or depth_meters is None:
            continue
        color_bgr = color_frame
        depth_stack.append(depth_meters.astype(np.float32, copy=False))

    if not depth_stack or color_bgr is None:
        raise RuntimeError("failed to capture RGBD frame")

    if len(depth_stack) == 1:
        depth_meters = depth_stack[0]
    else:
        depth_volume = np.stack(depth_stack, axis=0)
        valid_mask = depth_volume > 0
        depth_for_median = np.where(valid_mask, depth_volume, np.nan)
        depth_meters = np.nanmedian(depth_for_median, axis=0).astype(np.float32)
        depth_meters = np.nan_to_num(depth_meters, nan=0.0, posinf=0.0, neginf=0.0)

    return color_bgr, depth_meters


def _start_camera_with_retries(request: dict[str, object], *, retries: int = 3, retry_delay_s: float = 1.0) -> RealSenseRGBDCamera:
    last_error: Exception | None = None
    for attempt in range(1, max(1, retries) + 1):
        camera = RealSenseRGBDCamera(
            width=int(request["camera_width"]),
            height=int(request["camera_height"]),
            fps=int(request["camera_fps"]),
            clip_max=float(request["clip_max_m"]),
        )
        try:
            camera.start()
            return camera
        except Exception as exc:
            last_error = exc
            camera.release()
            if attempt >= retries:
                break
            time.sleep(float(retry_delay_s))
    if last_error is None:
        raise RuntimeError("failed to start RealSense camera")
    raise last_error


def _capture(request_json: Path, response_json: Path) -> None:
    request = json.loads(request_json.read_text(encoding="utf-8"))
    camera = None
    try:
        camera = _start_camera_with_retries(request)
        color_bgr, depth_meters = _capture_fused_rgbd(
            camera,
            depth_fusion_frames=int(request["depth_fusion_frames"]),
        )

        filter_mode = str(request.get("pointcloud_filter_mode") or "bilateral")
        if filter_mode == "bilateral":
            depth_meters = _bilateral_filter_depth(
                depth_meters,
                diameter=int(request.get("bilateral_diameter", 5)),
                sigma_color=float(request.get("bilateral_sigma_color", 0.02)),
                sigma_space=float(request.get("bilateral_sigma_space", 5.0)),
            )
        elif filter_mode == "median":
            depth_meters = _median_filter_depth(
                depth_meters,
                kernel_size=int(request.get("median_kernel_size", 5)),
            )

        color_path = response_json.parent / "color.npy"
        depth_path = response_json.parent / "depth.npy"
        np.save(color_path, color_bgr)
        np.save(depth_path, depth_meters)

        intrinsics = camera.intrinsics
        response = {
            "success": True,
            "color_npy": color_path.name,
            "depth_npy": depth_path.name,
            "intrinsics": {
                "width": int(intrinsics.width),
                "height": int(intrinsics.height),
                "fx": float(intrinsics.fx),
                "fy": float(intrinsics.fy),
                "ppx": float(intrinsics.ppx),
                "ppy": float(intrinsics.ppy),
            },
        }
    except Exception as exc:
        response = {
            "success": False,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        if camera is not None:
            camera.release()

    response_json.write_text(_json_dumps(response), encoding="utf-8")


def _run_daemon() -> int:
    """Daemon mode: keep the RealSense pipeline open and serve capture requests over stdin/stdout.

    Protocol: same JSON-line protocol as the inference worker.
    Supported request types:
      - {"type": "ping"}  → {"type": "pong"}
      - {"type": "shutdown"}  → clean exit
      - capture request dict  → capture response dict
    """
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    print("[camera_worker] daemon ready (camera not yet opened)", file=sys.stderr, flush=True)

    camera: RealSenseRGBDCamera | None = None
    current_camera_params: dict[str, object] = {}

    def _ensure_camera(request: dict[str, object]) -> RealSenseRGBDCamera:
        nonlocal camera, current_camera_params
        needed = {
            "camera_width": int(request["camera_width"]),
            "camera_height": int(request["camera_height"]),
            "camera_fps": int(request["camera_fps"]),
            "clip_max_m": float(request["clip_max_m"]),
        }
        if camera is not None and current_camera_params == needed:
            return camera
        # Params changed or first open — (re)start.
        if camera is not None:
            try:
                camera.release()
            except Exception:
                pass
            camera = None
        camera = _start_camera_with_retries(request)
        current_camera_params = needed
        print(f"[camera_worker] camera opened: {needed}", file=sys.stderr, flush=True)
        return camera

    while True:
        try:
            raw = stdin.readline()
        except Exception as exc:
            print(f"[camera_worker] stdin read error: {exc}", file=sys.stderr, flush=True)
            break
        if not raw:
            break
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            response: dict[str, object] = {"success": False, "message": f"invalid JSON: {exc}"}
            stdout.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
            stdout.flush()
            continue

        msg_type = str(request.get("type", ""))
        if msg_type == "ping":
            stdout.write(b'{"type":"pong"}\n')
            stdout.flush()
            continue
        if msg_type == "shutdown":
            print("[camera_worker] shutdown requested", file=sys.stderr, flush=True)
            break

        try:
            cam = _ensure_camera(request)
            color_bgr, depth_meters = _capture_fused_rgbd(
                cam,
                depth_fusion_frames=int(request["depth_fusion_frames"]),
            )
            filter_mode = str(request.get("pointcloud_filter_mode") or "bilateral")
            if filter_mode == "bilateral":
                depth_meters = _bilateral_filter_depth(
                    depth_meters,
                    diameter=int(request.get("bilateral_diameter", 5)),
                    sigma_color=float(request.get("bilateral_sigma_color", 0.02)),
                    sigma_space=float(request.get("bilateral_sigma_space", 5.0)),
                )
            elif filter_mode == "median":
                depth_meters = _median_filter_depth(
                    depth_meters,
                    kernel_size=int(request.get("median_kernel_size", 5)),
                )

            work_dir = Path(str(request["work_dir"])).expanduser().resolve()
            work_dir.mkdir(parents=True, exist_ok=True)
            color_path = work_dir / "color.npy"
            depth_path = work_dir / "depth.npy"
            np.save(color_path, color_bgr)
            np.save(depth_path, depth_meters)

            intrinsics = cam.intrinsics
            response = {
                "success": True,
                "color_npy": color_path.name,
                "depth_npy": depth_path.name,
                "intrinsics": {
                    "width": int(intrinsics.width),
                    "height": int(intrinsics.height),
                    "fx": float(intrinsics.fx),
                    "fy": float(intrinsics.fy),
                    "ppx": float(intrinsics.ppx),
                    "ppy": float(intrinsics.ppy),
                },
            }
        except Exception as exc:
            # On error, release the camera so next request triggers a fresh open.
            if camera is not None:
                try:
                    camera.release()
                except Exception:
                    pass
                camera = None
                current_camera_params = {}
            response = {"success": False, "message": str(exc), "traceback": traceback.format_exc()}
            print(f"[camera_worker] capture failed: {exc}", file=sys.stderr, flush=True)

        stdout.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
        stdout.flush()

    if camera is not None:
        try:
            camera.release()
        except Exception:
            pass
    print("[camera_worker] exiting", file=sys.stderr, flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="External RGBD capture worker.")
    parser.add_argument("--request-json", default="")
    parser.add_argument("--response-json", default="")
    parser.add_argument("--daemon", action="store_true",
                        help="Run as a persistent daemon keeping the camera open.")
    args = parser.parse_args()

    if args.daemon:
        return _run_daemon()

    # Legacy single-shot mode.
    _capture(Path(args.request_json), Path(args.response_json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
