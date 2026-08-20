#!/usr/bin/env python3
"""Build a six-class crop-classification dataset from reviewed object boxes."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_ORDER = (
    "dark_bottle", "green_bottle", "yellow_block",
    "red_block", "orange_bottle", "blue_block",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=PROJECT_ROOT / "data" / "vision_dataset" / "objects")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "data" / "object_crop_cls6")
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--padding", type=float, default=0.12, help="框四周保留的相对边长")
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def read_yolo_box(label: Path, image_width: int, image_height: int) -> tuple[int, int, int, int] | None:
    lines = [line.strip() for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) != 1:
        return None
    fields = lines[0].split()
    if len(fields) != 5:
        return None
    _, center_x, center_y, width, height = (float(value) for value in fields)
    crop_w, crop_h = int(round(width * image_width)), int(round(height * image_height))
    return int(round(center_x * image_width - crop_w / 2)), int(round(center_y * image_height - crop_h / 2)), crop_w, crop_h


def main() -> int:
    args = parse_args()
    source, output = args.source_root.resolve(), args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise SystemExit(f"output already exists and is not empty: {output}")
    if not 0.05 <= args.val_fraction <= 0.45:
        raise SystemExit("--val-fraction must be between 0.05 and 0.45")

    grouped: dict[str, list[tuple[Path, tuple[int, int, int, int]]]] = {}
    for item_id in CLASS_ORDER:
        samples = []
        for image_path in sorted((source / item_id / "images").glob("*.jpg")):
            label_path = source / "labels" / item_id / f"{image_path.stem}.txt"
            if not label_path.exists():
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            box = read_yolo_box(label_path, image.shape[1], image.shape[0])
            if box is not None:
                samples.append((image_path, box))
        if len(samples) < 8:
            raise SystemExit(f"{item_id}: only {len(samples)} usable labelled objects")
        grouped[item_id] = samples

    rng = random.Random(args.seed)
    total = {"train": 0, "val": 0}
    for item_id, samples in grouped.items():
        rng.shuffle(samples)
        val_count = max(1, round(len(samples) * args.val_fraction))
        for split, selected in (("val", samples[:val_count]), ("train", samples[val_count:])):
            destination = output / split / item_id
            destination.mkdir(parents=True, exist_ok=True)
            for image_path, (x, y, width, height) in selected:
                image = cv2.imread(str(image_path))
                pad = int(round(max(width, height) * args.padding))
                x0, y0 = max(0, x - pad), max(0, y - pad)
                x1, y1 = min(image.shape[1], x + width + pad), min(image.shape[0], y + height + pad)
                crop = image[y0:y1, x0:x1]
                target = destination / image_path.name
                if crop.size == 0 or not cv2.imwrite(str(target), crop):
                    raise RuntimeError(f"failed to save crop: {target}")
                total[split] += 1
    print(f"prepared {output}: train={total['train']} val={total['val']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
