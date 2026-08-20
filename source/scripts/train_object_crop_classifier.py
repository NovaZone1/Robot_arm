#!/usr/bin/env python3
"""Train a six-class classifier for crops from the existing perception boxes."""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    data = PROJECT_ROOT / "data" / "object_crop_cls6"
    if not (data / "train").is_dir() or not (data / "val").is_dir():
        raise SystemExit("dataset is missing; run prepare_object_crop_classifier_dataset.py first")
    YOLO(str(PROJECT_ROOT / "yolo11n-cls.pt")).train(
        data=str(data), epochs=60, imgsz=224, batch=32, device=0, workers=4,
        patience=18, project=str(PROJECT_ROOT / "models" / "object_crop_cls6"),
        name="train", exist_ok=True, pretrained=True, seed=20260820,
        deterministic=True, hsv_h=0.015, hsv_s=0.30, hsv_v=0.20,
        degrees=5.0, translate=0.06, scale=0.15, fliplr=0.0,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
