#!/usr/bin/env python3
"""Fit a taught (u, v) -> (X, Y) placement map and write mapping.yaml."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.perception.placement_uv_map import (
    default_placement_uv_root,
    fit_placement_uv_map,
    load_samples,
    write_mapping,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--item", required=True)
    parser.add_argument("--root", default=str(default_placement_uv_root()))
    args = parser.parse_args()
    item_dir = Path(args.root) / args.item
    samples_path = item_dir / "samples.json"
    if not samples_path.is_file():
        raise SystemExit(f"missing samples: {samples_path}")
    mapping = fit_placement_uv_map(load_samples(samples_path))
    output = item_dir / "mapping.yaml"
    write_mapping(output, mapping)
    print(
        f"wrote {output}  samples={mapping.sample_count}  "
        f"rms_xy={mapping.fit_rms_xy_mm:.1f}mm  "
        f"u={mapping.u_px_range}  v={mapping.v_px_range}  "
        f"align_uv=({mapping.align_u_px},{mapping.align_v_px})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
