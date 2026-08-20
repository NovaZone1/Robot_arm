#!/usr/bin/env python3
"""Train or resume the six-class box-label YOLO detector."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = PROJECT_ROOT / "data" / "box_label_yolo6" / "data.yaml"
RUN_DIR = PROJECT_ROOT / "models" / "box_label_yolo6" / "train"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="从 train/weights/last.pt 继续")
    parser.add_argument("--epochs", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not DATA_YAML.is_file():
        raise SystemExit(f"dataset is missing: {DATA_YAML}; run prepare_box_label_yolo_dataset.py first")
    last = RUN_DIR / "weights" / "last.pt"
    if args.resume:
        if not last.is_file():
            raise SystemExit(f"cannot resume; missing {last}")
        YOLO(str(last)).train(resume=True)
        return 0

    YOLO(str(PROJECT_ROOT / "yolov8n.pt")).train(
        data=str(DATA_YAML), epochs=args.epochs, imgsz=640, batch=8, device=0,
        workers=4, patience=25, project=str(RUN_DIR.parent), name=RUN_DIR.name,
        exist_ok=True, pretrained=True, seed=20260820, deterministic=True,
        hsv_h=0.015, hsv_s=0.35, hsv_v=0.25, degrees=5.0, translate=0.08,
        scale=0.20, fliplr=0.0, mosaic=0.30, close_mosaic=15,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
