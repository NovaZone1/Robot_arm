#!/usr/bin/env python3
"""Interactively collect labelled RGB images from the RealSense D435.

Move the arm to an observation pose first, then run this script.  It does not
move the arm or base.  Images are saved as a small, traceable training set for
the current objects and paper box labels.

Examples:
  .venv/bin/python source/scripts/collect_vision_dataset.py \
      --mode object --class-id orange_bottle
  .venv/bin/python source/scripts/collect_vision_dataset.py \
      --mode box_label --class-id red_block
  .venv/bin/python source/scripts/collect_vision_dataset.py \
      --mode negative --class-id red_cap
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.realsense_rgbd import RealSenseRGBDCamera  # noqa: E402


VALID_MODES = ("object", "box_label", "negative")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=VALID_MODES, required=True,
                        help="object=桌面物体；box_label=盒子纸质标识；negative=干扰/空桌面")
    parser.add_argument("--class-id", required=True,
                        help="类别或场景名，例如 red_block、orange_bottle、red_cap")
    parser.add_argument("--output-root", type=Path,
                        default=PROJECT_ROOT / "data" / "vision_dataset",
                        help="数据集根目录（默认 source/data/vision_dataset）")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--interval-s", type=float, default=0.40,
                        help="连续采集时两张图之间的最短间隔")
    parser.add_argument(
        "--lock-color",
        action="store_true",
        help="锁定白平衡/曝光；默认自动，与本项目已有训练照片及真机抓取一致。",
    )
    parser.add_argument(
        "--white-balance",
        type=int,
        default=int(os.environ.get("D435_WHITE_BALANCE", "4700")),
        help="固定白平衡（K），默认与真机栈一致：4700。",
    )
    parser.add_argument(
        "--exposure",
        type=int,
        default=int(os.environ.get("D435_EXPOSURE", "80")),
        help="固定曝光，默认与真机栈一致：80。",
    )
    return parser.parse_args()


def make_output_dir(root: Path, mode: str, class_id: str) -> Path:
    safe_class = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in class_id)
    output_dir = root / f"{mode}s" / safe_class / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_frame(image, output_dir: Path, manifest: Path, mode: str, class_id: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = output_dir / f"{class_id}_{stamp}.jpg"
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 96]):
        raise RuntimeError(f"failed to write {path}")

    record = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "class_id": class_id,
        "image": str(path.relative_to(manifest.parent)),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
    }
    with manifest.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def overlay(frame, *, mode: str, class_id: str, saved: int, continuous: bool) -> None:
    state = "ON" if continuous else "OFF"
    lines = [
        f"mode: {mode}    class: {class_id}",
        f"saved this session: {saved}    continuous: {state}",
        "SPACE: save one   R: toggle continuous   Q / ESC: quit",
    ]
    y = 28
    for line in lines:
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(frame, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (30, 255, 30), 1, cv2.LINE_AA)
        y += 27


def main() -> int:
    args = parse_args()
    # The existing training set was collected with D435 auto colour.  Keep it
    # as the default and make the live stack use the same profile.  Locking is
    # retained only when a future, newly-collected dataset deliberately uses a
    # calibrated fixed profile.
    os.environ["D435_LOCK_COLOR"] = "1" if args.lock_color else "0"
    os.environ["D435_WHITE_BALANCE"] = str(args.white_balance)
    os.environ["D435_EXPOSURE"] = str(args.exposure)
    output_dir = make_output_dir(args.output_root.resolve(), args.mode, args.class_id)
    manifest = args.output_root.resolve() / "manifest.jsonl"
    manifest.parent.mkdir(parents=True, exist_ok=True)

    print("打开 D435 实时画面。请确认 dashboard / camera_server 已停止，避免相机被占用。")
    color_mode = f"locked WB={args.white_balance}K exposure={args.exposure}" if args.lock_color else "auto"
    print(f"保存目录: {output_dir}")
    print(f"相机色彩配置: {color_mode}")
    camera = RealSenseRGBDCamera(args.width, args.height, args.fps)
    continuous = False
    saved = 0
    last_save = 0.0

    try:
        camera.start()
        cv2.namedWindow("Vision dataset collector", cv2.WINDOW_NORMAL)
        while True:
            ok, color, _, _ = camera.get_frames()
            if not ok or color is None:
                continue

            preview = color.copy()
            overlay(preview, mode=args.mode, class_id=args.class_id,
                    saved=saved, continuous=continuous)
            cv2.imshow("Vision dataset collector", preview)

            now = time.monotonic()
            if continuous and now - last_save >= args.interval_s:
                path = save_frame(color, output_dir, manifest, args.mode, args.class_id)
                saved += 1
                last_save = now
                print(f"[{saved}] {path.name}")

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("r"):
                continuous = not continuous
                print(f"continuous capture: {'on' if continuous else 'off'}")
            elif key == ord(" "):
                path = save_frame(color, output_dir, manifest, args.mode, args.class_id)
                saved += 1
                last_save = time.monotonic()
                print(f"[{saved}] {path.name}")
    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        cv2.destroyAllWindows()
    print(f"采集结束，本次保存 {saved} 张。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
