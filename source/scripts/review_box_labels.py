#!/usr/bin/env python3
"""Read-only review of saved YOLO box-label annotations.

This tool never writes labels. Use A/D or left/right arrows to browse, and
Q/Esc to quit.  To correct a frame, note its filename then use
``annotate_box_labels.py --redo`` only for the editing pass.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-root", type=Path,
        default=PROJECT_ROOT / "data" / "vision_dataset" / "box_labels",
    )
    return parser.parse_args()


def label_path(images_root: Path, image_path: Path) -> Path:
    return images_root / "labels" / image_path.parent.parent.name / f"{image_path.stem}.txt"


def read_box(path: Path, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not path.exists():
        return None
    fields = path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 5:
        return None
    _, cx, cy, box_w, box_h = (float(value) for value in fields)
    w, h = int(round(box_w * width)), int(round(box_h * height))
    return int(round((cx * width) - (w / 2))), int(round((cy * height) - (h / 2))), w, h


def main() -> int:
    args = parse_args()
    root = args.images_root.resolve()
    images = sorted(root.glob("*/images/*.jpg"))
    if not images:
        raise SystemExit(f"no images found under {root}")
    index = 0
    cv2.namedWindow("Box label review (read-only)", cv2.WINDOW_NORMAL)
    while True:
        image_path = images[index]
        image = cv2.imread(str(image_path))
        if image is None:
            index = (index + 1) % len(images)
            continue
        box = read_box(label_path(root, image_path), image.shape[1], image.shape[0])
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        status = "label saved" if box is not None else "NO LABEL / skipped"
        lines = (
            f"{index + 1}/{len(images)}  class: {image_path.parent.parent.name}",
            image_path.name,
            f"{status}   A/Left: previous  D/Right/Space: next  Q/Esc: quit",
        )
        y = 28
        for line in lines:
            cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(image, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (30, 255, 30), 1, cv2.LINE_AA)
            y += 27
        cv2.imshow("Box label review (read-only)", image)
        key = cv2.waitKey(0) & 0xFF
        if key in (27, ord("q")):
            break
        if key in (ord("a"), 81):
            index = (index - 1) % len(images)
        elif key in (ord("d"), ord(" "), 83):
            index = (index + 1) % len(images)
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
