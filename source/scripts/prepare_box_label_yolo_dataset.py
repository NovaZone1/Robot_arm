#!/usr/bin/env python3
"""Create a deterministic YOLO train/validation dataset from reviewed labels."""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLASS_ORDER = (
    "dark_bottle", "green_bottle", "yellow_block",
    "red_block", "orange_bottle", "blue_block",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root", type=Path,
        default=PROJECT_ROOT / "data" / "vision_dataset" / "box_labels",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=PROJECT_ROOT / "data" / "box_label_yolo6",
    )
    parser.add_argument("--val-fraction", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"output already exists and is not empty: {output_root}")
    if not 0.05 <= args.val_fraction <= 0.45:
        raise SystemExit("--val-fraction must be between 0.05 and 0.45")

    pairs_by_class: dict[str, list[tuple[Path, Path]]] = {}
    for item_id in CLASS_ORDER:
        pairs = []
        for image in sorted((source_root / item_id / "images").glob("*.jpg")):
            label = source_root / "labels" / item_id / f"{image.stem}.txt"
            if not label.exists() or not label.read_text(encoding="utf-8").strip():
                continue
            pairs.append((image, label))
        if len(pairs) < 8:
            raise SystemExit(f"{item_id}: only {len(pairs)} usable labels; need at least 8")
        pairs_by_class[item_id] = pairs

    randomizer = random.Random(args.seed)
    for split in ("train", "val"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)

    totals = {"train": 0, "val": 0}
    for item_id, pairs in pairs_by_class.items():
        randomizer.shuffle(pairs)
        val_count = max(1, round(len(pairs) * args.val_fraction))
        for split, split_pairs in (("val", pairs[:val_count]), ("train", pairs[val_count:])):
            for image, label in split_pairs:
                # Prefix prevents future class folders from ever colliding on
                # filenames, while labels keep their original YOLO class id.
                name = f"{item_id}_{image.name}"
                shutil.copy2(image, output_root / "images" / split / name)
                shutil.copy2(label, output_root / "labels" / split / f"{Path(name).stem}.txt")
                totals[split] += 1

    (output_root / "data.yaml").write_text(
        "path: " + str(output_root) + "\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASS_ORDER)}\n"
        "names: [" + ", ".join(repr(name) for name in CLASS_ORDER) + "]\n",
        encoding="utf-8",
    )
    print(f"prepared {output_root}: train={totals['train']} val={totals['val']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
