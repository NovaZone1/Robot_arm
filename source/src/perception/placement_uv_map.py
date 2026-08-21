"""Fit and apply (u, v) -> release XY maps taught at a fixed observation pose."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
from typing import Iterable

import numpy as np

try:
    import yaml
except Exception:  # pragma: no cover
    yaml = None


def default_placement_uv_root() -> Path:
    return Path(__file__).resolve().parents[2] / "config" / "calibration" / "placement_uv_xy"


def _design(u_px: np.ndarray, v_px: np.ndarray) -> np.ndarray:
    u = np.asarray(u_px, dtype=np.float64).reshape(-1)
    v = np.asarray(v_px, dtype=np.float64).reshape(-1)
    return np.column_stack([np.ones_like(u), u, v, u * v, v * v])


@dataclass(frozen=True, slots=True)
class PlacementUvMap:
    item_id: str
    x_mm_coeffs: tuple[float, ...]
    y_mm_coeffs: tuple[float, ...]
    release_z_mm: float
    approach_z_mm: float
    rpy_deg: tuple[float, float, float]
    u_px_range: tuple[float, float]
    v_px_range: tuple[float, float]
    fit_rms_xy_mm: float
    sample_count: int
    # Pixel of the taught "center" sample: arm facing the box, not image center.
    align_u_px: float | None = None
    align_v_px: float | None = None
    # (u, v, release_z, roll, pitch, yaw) from each taught sample.
    taught_releases: tuple[tuple[float, float, float, float, float, float], ...] = ()

    def predict_xy_mm(self, u_px: float, v_px: float) -> tuple[float, float]:
        features = _design([u_px], [v_px])[0]
        x_mm = float(features @ np.asarray(self.x_mm_coeffs, dtype=np.float64))
        y_mm = float(features @ np.asarray(self.y_mm_coeffs, dtype=np.float64))
        return x_mm, y_mm

    def alignment_u_norm(self, width_px: float) -> float | None:
        if self.align_u_px is None or float(width_px) <= 1.0:
            return None
        return float(self.align_u_px) / float(width_px)

    def in_domain(self, u_px: float, v_px: float, *, margin_px: float = 20.0) -> bool:
        return (
            self.u_px_range[0] - margin_px <= float(u_px) <= self.u_px_range[1] + margin_px
            and self.v_px_range[0] - margin_px <= float(v_px) <= self.v_px_range[1] + margin_px
        )

    def nearest_taught_release(
        self, u_px: float, v_px: float, *, skip_singular_rpy: bool = False
    ) -> tuple[float, float, float, float, float, float] | None:
        candidates = list(self.taught_releases)
        if skip_singular_rpy:
            candidates = [
                item
                for item in candidates
                if abs(abs(float(item[4])) - 90.0) > 1.0
            ]
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda item: (item[0] - float(u_px)) ** 2 + (item[1] - float(v_px)) ** 2,
        )

    def poses_mm_deg(self, u_px: float, v_px: float) -> dict[str, tuple[float, ...]]:
        x_mm, y_mm = self.predict_xy_mm(u_px, v_px)
        # Wrist is always the taught center pose. Near-90° pitch samples are
        # Euler singularities and must not be replayed as a target.
        rpy = tuple(float(value) for value in self.rpy_deg)
        nearest = self.nearest_taught_release(u_px, v_px, skip_singular_rpy=True)
        z_mm = float(nearest[2]) if nearest is not None else float(self.release_z_mm)
        release = (x_mm, y_mm, z_mm, *rpy)
        approach = (x_mm, y_mm, float(self.approach_z_mm), *rpy)
        retreat = approach
        return {"approach": approach, "release": release, "retreat": retreat}

    def to_dict(self) -> dict[str, object]:
        return {
            "item_id": self.item_id,
            "model": "1,u,v,u*v,v*v",
            "x_mm_coeffs": list(self.x_mm_coeffs),
            "y_mm_coeffs": list(self.y_mm_coeffs),
            "release_z_mm": float(self.release_z_mm),
            "approach_z_mm": float(self.approach_z_mm),
            "rpy_deg": list(self.rpy_deg),
            "u_px_range": list(self.u_px_range),
            "v_px_range": list(self.v_px_range),
            "fit_rms_xy_mm": float(self.fit_rms_xy_mm),
            "sample_count": int(self.sample_count),
            "align_u_px": self.align_u_px,
            "align_v_px": self.align_v_px,
            "taught_releases": [list(item) for item in self.taught_releases],
        }


def load_samples(path: str | Path) -> dict[str, object]:
    return json.loads(Path(path).expanduser().resolve().read_text(encoding="utf-8"))


def taught_align_uv_px(
    samples: Iterable[dict[str, object]],
) -> tuple[float | None, float | None]:
    """Use the sample tagged center: that is the arm facing the box."""
    sample = _center_sample(samples)
    if sample is None:
        return None, None
    return float(sample["u_px"]), float(sample["v_px"])


def _center_sample(
    samples: Iterable[dict[str, object]],
) -> dict[str, object] | None:
    tagged = {str(sample.get("tag") or "").strip().lower(): sample for sample in samples}
    for name in (
        "center",
        "normal_center",
        "near_center",
        "near_centerer",
        # new_ samples use a prefixed tag (new_near_center / new_normal_center).
        "new_normal_center",
        "new_near_center",
    ):
        if name in tagged:
            return tagged[name]
    return None


def _release_waypoint(
    sample: dict[str, object],
) -> tuple[float, float, float, float, float, float]:
    pose = dict(sample.get("release_pose_mm_deg") or {})
    return (
        float(sample["u_px"]),
        float(sample["v_px"]),
        float(pose["z_mm"]),
        float(pose["roll_deg"]),
        float(pose["pitch_deg"]),
        float(pose["yaw_deg"]),
    )


def _angle_delta_deg(first: float, second: float) -> float:
    return ((float(first) - float(second) + 180.0) % 360.0) - 180.0


def _validate_consistent_release_orientation(
    samples: Iterable[dict[str, object]], *, tolerance_deg: float = 15.0
) -> None:
    """Reject a map containing manually taught wrist orientations that differ.

    The fitted model replays one wrist orientation.  Combining poses with a
    rotated wrist creates an XY fit that looks numerically good but can sweep
    the gripper through a box's label baffle during the real approach.
    """
    items = list(samples)
    if not items:
        return
    reference = dict(items[0].get("release_pose_mm_deg") or {})
    for index, sample in enumerate(items[1:], start=2):
        pose = dict(sample.get("release_pose_mm_deg") or {})
        for axis in ("roll_deg", "pitch_deg", "yaw_deg"):
            delta = abs(
                _angle_delta_deg(
                    float(pose.get(axis) or 0.0), float(reference.get(axis) or 0.0)
                )
            )
            if delta > float(tolerance_deg):
                raise RuntimeError(
                    "inconsistent taught release orientation: "
                    f"sample {index} {axis} differs by {delta:.1f}° "
                    f"(limit {tolerance_deg:.1f}°). Re-record this item with a fixed wrist orientation."
                )


def _rpy_from_center_sample(
    samples: Iterable[dict[str, object]],
) -> tuple[float, float, float] | None:
    sample = _center_sample(samples)
    if sample is None:
        return None
    waypoint = _release_waypoint(sample)
    return (waypoint[3], waypoint[4], waypoint[5])


def fit_placement_uv_map(
    samples_payload: dict[str, object],
    *,
    approach_z_mm: float | None = None,
    rpy_deg: Iterable[float] | None = None,
) -> PlacementUvMap:
    samples = list(samples_payload.get("samples") or [])
    if len(samples) < 5:
        raise RuntimeError("need at least 5 taught (u, v, X, Y) samples")
    _validate_consistent_release_orientation(samples)
    u = np.array([float(sample["u_px"]) for sample in samples], dtype=np.float64)
    v = np.array([float(sample["v_px"]) for sample in samples], dtype=np.float64)
    x = np.array(
        [float(sample["release_pose_mm_deg"]["x_mm"]) for sample in samples],
        dtype=np.float64,
    )
    y = np.array(
        [float(sample["release_pose_mm_deg"]["y_mm"]) for sample in samples],
        dtype=np.float64,
    )
    z = np.array(
        [float(sample["release_pose_mm_deg"]["z_mm"]) for sample in samples],
        dtype=np.float64,
    )
    design = _design(u, v)
    x_coeffs, _, _, _ = np.linalg.lstsq(design, x, rcond=None)
    y_coeffs, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    residual = np.hypot(x - design @ x_coeffs, y - design @ y_coeffs)
    observe = list(samples_payload.get("observe_pose_mm_deg") or [])
    if approach_z_mm is None:
        approach_z_mm = float(observe[2]) if len(observe) >= 3 else 491.1
    if rpy_deg is None:
        rpy_deg = _rpy_from_center_sample(samples)
    if rpy_deg is None:
        raise RuntimeError("need a taught center sample for release RPY")
    rpy = tuple(float(value) for value in rpy_deg)
    if len(rpy) != 3:
        raise ValueError("rpy_deg must contain 3 values")
    align_u_px, align_v_px = taught_align_uv_px(samples)
    taught_releases = tuple(_release_waypoint(sample) for sample in samples)
    return PlacementUvMap(
        item_id=str(samples_payload.get("item_id") or "unknown"),
        x_mm_coeffs=tuple(float(value) for value in x_coeffs),
        y_mm_coeffs=tuple(float(value) for value in y_coeffs),
        release_z_mm=float(np.median(z)),
        approach_z_mm=float(approach_z_mm),
        rpy_deg=rpy,
        u_px_range=(float(np.min(u)), float(np.max(u))),
        v_px_range=(float(np.min(v)), float(np.max(v))),
        fit_rms_xy_mm=float(np.sqrt(np.mean(residual ** 2))),
        sample_count=len(samples),
        align_u_px=align_u_px,
        align_v_px=align_v_px,
        taught_releases=taught_releases,
    )


def write_mapping(path: str | Path, mapping: PlacementUvMap) -> None:
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = mapping.to_dict()
    if yaml is None:
        destination.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return
    destination.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def load_mapping(path: str | Path) -> PlacementUvMap:
    destination = Path(path).expanduser().resolve()
    text = destination.read_text(encoding="utf-8")
    if destination.suffix == ".json" or yaml is None:
        payload = json.loads(text)
    else:
        payload = yaml.safe_load(text)
    return PlacementUvMap(
        item_id=str(payload["item_id"]),
        x_mm_coeffs=tuple(float(value) for value in payload["x_mm_coeffs"]),
        y_mm_coeffs=tuple(float(value) for value in payload["y_mm_coeffs"]),
        release_z_mm=float(payload["release_z_mm"]),
        approach_z_mm=float(payload["approach_z_mm"]),
        rpy_deg=tuple(float(value) for value in payload["rpy_deg"]),
        u_px_range=tuple(float(value) for value in payload["u_px_range"]),
        v_px_range=tuple(float(value) for value in payload["v_px_range"]),
        fit_rms_xy_mm=float(payload.get("fit_rms_xy_mm") or 0.0),
        sample_count=int(payload.get("sample_count") or 0),
        align_u_px=(
            float(payload["align_u_px"])
            if payload.get("align_u_px") is not None
            else None
        ),
        align_v_px=(
            float(payload["align_v_px"])
            if payload.get("align_v_px") is not None
            else None
        ),
        taught_releases=tuple(
            (
                float(item[0]),
                float(item[1]),
                float(item[2]),
                float(item[3]),
                float(item[4]),
                float(item[5]),
            )
            for item in list(payload.get("taught_releases") or [])
            if isinstance(item, (list, tuple)) and len(item) == 6
        ),
    )


def mapping_path_for_item(
    item_id: str,
    *,
    root: str | Path | None = None,
) -> Path:
    base = Path(root) if root is not None else default_placement_uv_root()
    return base / str(item_id) / "mapping.yaml"


_SHARED_MAP_BY_KIND = {
    "orange_bottle": "orange_bottle",
    "dark_bottle": "orange_bottle",
    "green_bottle": "orange_bottle",
    "red_block": "red_block",
    "yellow_block": "red_block",
    "blue_block": "red_block",
}


def load_mapping_for_item(
    item_id: str,
    *,
    root: str | Path | None = None,
) -> PlacementUvMap | None:
    path = mapping_path_for_item(item_id, root=root)
    if path.is_file():
        return load_mapping(path)
    shared_item = _SHARED_MAP_BY_KIND.get(str(item_id))
    if shared_item and shared_item != str(item_id):
        shared = mapping_path_for_item(shared_item, root=root)
        if shared.is_file():
            return load_mapping(shared)
    return None
