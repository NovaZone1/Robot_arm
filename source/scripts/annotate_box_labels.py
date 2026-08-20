#!/usr/bin/env python3
"""Review and create YOLO bounding boxes for collected labels or objects.

The tool proposes the white paper label region automatically.  Press Space to
accept it, or drag with the mouse to draw a corrected rectangle.  Labels are
written next to the image tree under ``box_labels/labels/<class>/``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_ORDER = (
    "dark_bottle", "green_bottle", "yellow_block",
    "red_block", "orange_bottle", "blue_block",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--images-root", type=Path,
        default=PROJECT_ROOT / "data" / "vision_dataset" / "box_labels",
    )
    parser.add_argument(
        "--mode", choices=("box_label", "object"), default="box_label",
        help="box_label=纸质标签框；object=抓取物体框",
    )
    parser.add_argument("--redo", action="store_true", help="也审核已保存的标注")
    return parser.parse_args()


def find_paper_candidate(image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Propose a paper label using its rectangular edge, not white colour.

    The white card often merges with the white table in a brightness mask, so
    its Canny outline is considerably more stable than its absolute colour.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 90)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    height, width = image.shape[:2]
    image_area = height * width
    best: tuple[float, tuple[int, int, int, int]] | None = None
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < image_area * 0.012 or area > image_area * 0.28:
            continue
        aspect = w / max(float(h), 1.0)
        if not 0.55 <= aspect <= 1.90:
            continue
        rectangularity = cv2.contourArea(contour) / max(float(area), 1.0)
        if rectangularity < 0.35:
            continue
        # Paper labels are normally mounted in the upper/central portion;
        # prefer them over a large white tabletop when candidates tie.
        upper_bias = max(0.0, 1.0 - (y / max(float(height), 1.0)))
        score = area * (0.65 + 0.35 * rectangularity) * (0.75 + 0.25 * upper_bias)
        if best is None or score > best[0]:
            best = (score, (x, y, w, h))
    return None if best is None else best[1]


OBJECT_HSV_RANGES = {
    "red_block": (((0, 65, 35), (12, 255, 255)), ((165, 65, 35), (179, 255, 255))),
    "yellow_block": (((20, 65, 45), (42, 255, 255)),),
    "blue_block": (((90, 45, 25), (140, 255, 255)),),
    "orange_bottle": (((8, 70, 35), (25, 255, 255)),),
    "green_bottle": (((35, 45, 25), (90, 255, 255)),),
    # Dark bottles are manually verified particularly carefully because a
    # chair or shadow can have a similar value range.
    "dark_bottle": (((0, 0, 15), (179, 150, 105)),),
}


def find_object_candidate(image: np.ndarray, item_id: str) -> tuple[int, int, int, int] | None:
    """Propose the desired object from its dominant liquid/block colour."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in OBJECT_HSV_RANGES[item_id]:
        mask = cv2.bitwise_or(mask, cv2.inRange(hsv, np.asarray(lower, np.uint8), np.asarray(upper, np.uint8)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8), iterations=2)
    height, width = image.shape[:2]
    best: tuple[float, tuple[int, int, int, int]] | None = None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if area < width * height * 0.002 or area > width * height * 0.35:
            continue
        aspect = w / max(float(h), 1.0)
        if not 0.20 <= aspect <= 2.5:
            continue
        score = area * (1.0 - min(0.5, y / max(float(height), 1.0)) * 0.20)
        if best is None or score > best[0]:
            padding = max(8, int(round(max(w, h) * 0.10)))
            best = (score, (max(0, x - padding), max(0, y - padding),
                            min(width - max(0, x - padding), w + (2 * padding)),
                            min(height - max(0, y - padding), h + (2 * padding))))
    return None if best is None else best[1]


class Annotator:
    def __init__(
        self,
        image: np.ndarray,
        initial_box: tuple[int, int, int, int] | None,
        *,
        progress: str,
        item_id: str,
        filename: str,
    ):
        self.image = image
        self.box = initial_box
        self.progress = progress
        self.item_id = item_id
        self.filename = filename
        self.drag_start: tuple[int, int] | None = None
        self.drag_box: tuple[int, int, int, int] | None = None

    def mouse(self, event: int, x: int, y: int, _flags: int, _userdata: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag_start = (x, y)
            self.drag_box = None
        elif event == cv2.EVENT_MOUSEMOVE and self.drag_start is not None:
            x0, y0 = self.drag_start
            self.drag_box = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
        elif event == cv2.EVENT_LBUTTONUP and self.drag_start is not None:
            x0, y0 = self.drag_start
            candidate = (min(x0, x), min(y0, y), abs(x - x0), abs(y - y0))
            self.box = candidate if candidate[2] >= 8 and candidate[3] >= 8 else self.box
            self.drag_start = None
            self.drag_box = None

    def render(self) -> np.ndarray:
        shown = self.image.copy()
        box = self.drag_box or self.box
        if box is not None:
            x, y, w, h = box
            cv2.rectangle(shown, (x, y), (x + w, y + h), (0, 255, 0), 2)
        for text, y in (
            (f"{self.progress}  class: {self.item_id}", 28),
            (self.filename, 55),
            ("SPACE: accept/save   mouse drag: redraw", 82),
            ("D/Right: next without saving   A/Left: previous", 109),
            ("N: skip/no label   Q/ESC: stop", 136),
        ):
            cv2.putText(shown, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(shown, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (40, 255, 40), 1, cv2.LINE_AA)
        return shown


def label_path(images_root: Path, image_path: Path) -> Path:
    relative = image_path.relative_to(images_root)
    return images_root / "labels" / relative.parent.parent.name / f"{image_path.stem}.txt"


def read_label_box(path: Path, image: np.ndarray) -> tuple[int, int, int, int] | None:
    """Read one existing YOLO label as a pixel rectangle for edit/review."""
    if not path.exists():
        return None
    fields = path.read_text(encoding="utf-8").strip().split()
    if len(fields) != 5:
        return None
    try:
        _, center_x, center_y, width_norm, height_norm = (float(value) for value in fields)
    except ValueError:
        return None
    height, width = image.shape[:2]
    box_w = int(round(width_norm * width))
    box_h = int(round(height_norm * height))
    return (
        int(round((center_x * width) - (box_w / 2))),
        int(round((center_y * height) - (box_h / 2))),
        box_w,
        box_h,
    )


def write_label(path: Path, item_id: str, box: tuple[int, int, int, int], image: np.ndarray) -> None:
    x, y, w, h = box
    height, width = image.shape[:2]
    class_index = CLASS_ORDER.index(item_id)
    values = (
        class_index,
        (x + (w / 2.0)) / width,
        (y + (h / 2.0)) / height,
        w / width,
        h / height,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{} {:.6f} {:.6f} {:.6f} {:.6f}\n".format(*values), encoding="utf-8")


def main() -> int:
    args = parse_args()
    images_root = args.images_root.resolve()
    images = sorted(images_root.glob("*/images/*.jpg"))
    if not images:
        raise SystemExit(f"no images found under {images_root}")
    unknown = sorted({path.parent.parent.name for path in images} - set(CLASS_ORDER))
    if unknown:
        raise SystemExit(f"unknown class directories: {unknown}; expected {CLASS_ORDER}")

    index = 0
    accepted = skipped = 0
    # Normal first-pass annotation skips existing files. Once the operator
    # goes backwards, this stays true so past annotations are displayed and
    # can be inspected without being overwritten.
    review_existing = bool(args.redo)
    cv2.namedWindow("Box label annotator", cv2.WINDOW_NORMAL)
    while 0 <= index < len(images):
        image_path = images[index]
        output = label_path(images_root, image_path)
        if output.exists() and not review_existing:
            index += 1
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"cannot read: {image_path}")
            index += 1
            continue
        # In edit mode, returning with A must show the saved box, not a fresh
        # automatic proposal. No existing file is changed until Space or N.
        initial_box = read_label_box(output, image) if output.exists() else None
        proposal = (
            find_object_candidate(image, image_path.parent.parent.name)
            if args.mode == "object"
            else find_paper_candidate(image)
        )
        annotator = Annotator(
            image,
            initial_box or proposal,
            progress=f"{index + 1}/{len(images)}",
            item_id=image_path.parent.parent.name,
            filename=image_path.name,
        )
        cv2.setMouseCallback("Box label annotator", annotator.mouse)
        while True:
            shown = annotator.render()
            cv2.imshow("Box label annotator", shown)
            key = cv2.waitKey(20) & 0xFF
            if key in (27, ord("q")):
                cv2.destroyAllWindows()
                print(f"saved={accepted}, skipped={skipped}; resume with the same command")
                return 0
            if key in (ord("a"), 81):
                index = max(-1, index - 1)
                review_existing = True
                break
            if key in (ord("d"), 83):
                index += 1
                review_existing = True
                break
            if key == ord("n"):
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("", encoding="utf-8")
                skipped += 1
                index += 1
                break
            if key == ord(" "):
                if annotator.box is None:
                    print(f"no box selected: {image_path.name}")
                    continue
                write_label(output, image_path.parent.parent.name, annotator.box, image)
                accepted += 1
                index += 1
                break
    cv2.destroyAllWindows()
    print(f"done: saved={accepted}, skipped={skipped}, total={len(images)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
