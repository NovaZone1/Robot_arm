"""Competition item catalog, bottle identity filtering, and box-label matching."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


@dataclass(frozen=True, slots=True)
class PlacementSpec:
    enabled: bool
    base_aligned_enabled: bool
    base_aligned_approach_pose_mm_deg: tuple[float, float, float, float, float, float] | None
    base_aligned_release_pose_mm_deg: tuple[float, float, float, float, float, float] | None
    base_aligned_retreat_pose_mm_deg: tuple[float, float, float, float, float, float] | None
    release_offset_mm: tuple[float, float, float]
    release_rpy_deg: tuple[float, float, float] | None
    approach_height_mm: float
    retreat_height_mm: float


@dataclass(frozen=True, slots=True)
class ItemSpec:
    item_id: str
    display_name: str
    aliases: tuple[str, ...]
    grasp_prompt: str
    kind: str
    reference_image: Path
    object_size_m: tuple[float, float, float]
    placement: PlacementSpec


@dataclass(frozen=True, slots=True)
class BoxSpec:
    outer_size_m: tuple[float, float, float]
    wall_clearance_m: float
    label_surface: str
    label_match_threshold: float
    label_search_roi_norm: tuple[float, float, float, float]
    slot_centers_mm: tuple[tuple[float, float, float] | None, ...]
    row_first_slot_center_mm: tuple[float, float, float] | None
    row_last_slot_center_mm: tuple[float, float, float] | None
    row_slot_pitch_mm: float
    row_slot_pitch_tolerance_mm: float


@dataclass(frozen=True, slots=True)
class LabelDetection:
    item_id: str
    confidence: float
    bbox_xywh: tuple[int, int, int, int]
    method: str = "unknown"


@dataclass(frozen=True, slots=True)
class LabelMatch:
    expected_item_id: str
    matched_item_id: str | None
    confidence: float
    bbox_xywh: tuple[int, int, int, int] | None
    search_roi_xywh: tuple[int, int, int, int]
    accepted: bool
    slot_index: int | None
    detected_item_ids: tuple[str, ...]
    detections: tuple[LabelDetection, ...]


@dataclass(frozen=True, slots=True)
class BoxRowLocalization:
    target_item_id: str
    slot_index: int
    box_center_base_m: tuple[float, float, float]
    box_centers_base_m: tuple[tuple[float, float, float], ...]
    label_centers_base_m: tuple[tuple[float, float, float], ...]
    adjacent_pitch_mm: tuple[float, ...]
    raw_adjacent_pitch_mm: tuple[float, ...]
    fit_residual_mm: tuple[float, ...]
    interior_direction_base: tuple[float, float, float]


def default_item_catalog_path() -> Path:
    source_path = Path(__file__).resolve().parents[2] / "config" / "item_catalog.yaml"
    if source_path.is_file():
        return source_path
    try:
        from ament_index_python.packages import get_package_share_directory

        installed = Path(get_package_share_directory("robot_grasp_ros2")) / "config" / "item_catalog.yaml"
        if installed.is_file():
            return installed
    except Exception:
        pass
    return source_path


class ItemCatalog:
    def __init__(self, *, path: Path, box: BoxSpec, items: dict[str, ItemSpec]) -> None:
        self.path = path
        self.box = box
        self.items = items
        aliases: dict[str, str] = {}
        for item_id, item in items.items():
            for value in (item_id, item.display_name, item.grasp_prompt, *item.aliases):
                aliases[self._normalize(value)] = item_id
        self._aliases = aliases

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(str(value or "").strip().lower().replace("_", " ").split())

    @classmethod
    def load(cls, path: str | Path) -> "ItemCatalog":
        resolved = Path(path).expanduser().resolve()
        with resolved.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        box_payload = dict(payload.get("box") or {})
        slot_centers: list[tuple[float, float, float] | None] = []
        for index, raw_center in enumerate(list(box_payload.get("slot_centers_mm") or [])):
            values = list(raw_center or [])
            if not values:
                slot_centers.append(None)
                continue
            if len(values) != 3:
                raise ValueError(f"box slot_centers_mm[{index}] must contain 3 values")
            slot_centers.append(tuple(float(value) for value in values))

        def optional_point(name: str) -> tuple[float, float, float] | None:
            values = list(box_payload.get(name) or [])
            if not values:
                return None
            if len(values) != 3:
                raise ValueError(f"box {name} must contain 3 values")
            return tuple(float(value) for value in values)

        box = BoxSpec(
            outer_size_m=tuple(float(v) for v in box_payload["outer_size_m"]),
            wall_clearance_m=float(box_payload.get("wall_clearance_m", 0.0)),
            label_surface=str(box_payload.get("label_surface", "uncalibrated")),
            label_match_threshold=float(box_payload.get("label_match_threshold", 0.46)),
            label_search_roi_norm=tuple(
                float(v) for v in box_payload.get("label_search_roi_norm", [0.0, 0.0, 1.0, 1.0])
            ),
            slot_centers_mm=tuple(slot_centers),
            row_first_slot_center_mm=optional_point("row_first_slot_center_mm"),
            row_last_slot_center_mm=optional_point("row_last_slot_center_mm"),
            row_slot_pitch_mm=float(box_payload.get("row_slot_pitch_mm", 180.0)),
            row_slot_pitch_tolerance_mm=float(
                box_payload.get("row_slot_pitch_tolerance_mm", 20.0)
            ),
        )
        if len(box.outer_size_m) != 3 or len(box.label_search_roi_norm) != 4:
            raise ValueError("item catalog box dimensions/ROI have invalid length")
        if len(box.slot_centers_mm) != 6:
            raise ValueError("item catalog must contain 6 left-to-right box slot centers")

        items: dict[str, ItemSpec] = {}
        for item_id, raw in dict(payload.get("items") or {}).items():
            item_payload = dict(raw or {})
            placement_payload = dict(item_payload.get("placement") or {})
            raw_release_rpy = list(placement_payload.get("release_rpy_deg") or [])
            release_rpy = (
                tuple(float(v) for v in raw_release_rpy)
                if raw_release_rpy
                else None
            )
            if release_rpy is not None and len(release_rpy) != 3:
                raise ValueError(f"{item_id}: release_rpy_deg must contain 3 values")
            release_offset = tuple(
                float(v) for v in placement_payload.get("release_offset_mm", [0.0, 0.0, 0.0])
            )
            if len(release_offset) != 3:
                raise ValueError(f"{item_id}: release_offset_mm must contain 3 values")

            def optional_pose(name: str):
                values = list(placement_payload.get(name) or [])
                if not values:
                    return None
                if len(values) != 6:
                    raise ValueError(f"{item_id}: {name} must contain 6 values")
                return tuple(float(value) for value in values)

            reference_image = (resolved.parent / str(item_payload["reference_image"])).resolve()
            if not reference_image.is_file():
                raise FileNotFoundError(f"{item_id}: reference image not found: {reference_image}")
            items[str(item_id)] = ItemSpec(
                item_id=str(item_id),
                display_name=str(item_payload["display_name"]),
                aliases=tuple(str(v) for v in item_payload.get("aliases", [])),
                grasp_prompt=str(item_payload["grasp_prompt"]),
                kind=str(item_payload["kind"]).strip().lower(),
                reference_image=reference_image,
                object_size_m=tuple(float(v) for v in item_payload["object_size_m"]),
                placement=PlacementSpec(
                    enabled=bool(placement_payload.get("enabled", False)),
                    base_aligned_enabled=bool(
                        placement_payload.get("base_aligned_enabled", False)
                    ),
                    base_aligned_approach_pose_mm_deg=optional_pose(
                        "base_aligned_approach_pose_mm_deg"
                    ),
                    base_aligned_release_pose_mm_deg=optional_pose(
                        "base_aligned_release_pose_mm_deg"
                    ),
                    base_aligned_retreat_pose_mm_deg=optional_pose(
                        "base_aligned_retreat_pose_mm_deg"
                    ),
                    release_offset_mm=release_offset,
                    release_rpy_deg=release_rpy,
                    approach_height_mm=float(placement_payload.get("approach_height_mm", 120.0)),
                    retreat_height_mm=float(placement_payload.get("retreat_height_mm", 140.0)),
                ),
            )
        if not items:
            raise ValueError("item catalog contains no items")
        return cls(path=resolved, box=box, items=items)

    def resolve(self, value: str) -> ItemSpec | None:
        item_id = self._aliases.get(self._normalize(value))
        return self.items.get(item_id) if item_id else None

    def require(self, value: str) -> ItemSpec:
        item = self.resolve(value)
        if item is None:
            raise KeyError(f"unknown competition item: {value!r}")
        return item

    def build_place_poses_mm_deg(
        self,
        value: str,
        slot_index: int,
        *,
        slot_center_mm: tuple[float, float, float] | None = None,
    ) -> dict[str, tuple[float, float, float, float, float, float]]:
        item = self.require(value)
        placement = item.placement
        if not placement.enabled or placement.release_rpy_deg is None:
            raise RuntimeError(
                f"{item.item_id}: placement is not calibrated; set placement.enabled=true "
                "and release_rpy_deg in item_catalog.yaml"
            )
        if not 0 <= int(slot_index) < len(self.box.slot_centers_mm):
            raise RuntimeError(f"invalid box slot_index: {slot_index}")
        if slot_center_mm is None:
            slot_centers = self.resolved_slot_centers_mm()
            slot_center = slot_centers[int(slot_index)]
        else:
            if len(slot_center_mm) != 3:
                raise RuntimeError("dynamic slot_center_mm must contain 3 values")
            slot_center = tuple(float(value) for value in slot_center_mm)
        inner_x = self.box.outer_size_m[0] - (2.0 * self.box.wall_clearance_m)
        inner_y = self.box.outer_size_m[1] - (2.0 * self.box.wall_clearance_m)
        if min(item.object_size_m[0], item.object_size_m[1]) > min(inner_x, inner_y):
            raise RuntimeError(
                f"{item.item_id}: object footprint does not fit the configured box clearance"
            )
        release_position = tuple(
            float(slot_center[index]) + float(placement.release_offset_mm[index])
            for index in range(3)
        )
        release = (*release_position, *placement.release_rpy_deg)
        approach = (release[0], release[1], release[2] + placement.approach_height_mm, *release[3:])
        retreat = (release[0], release[1], release[2] + placement.retreat_height_mm, *release[3:])
        return {
            "approach": approach,
            "release": release,
            "retreat": retreat,
        }

    def build_base_aligned_place_poses_mm_deg(
        self,
        value: str,
    ) -> dict[str, tuple[float, float, float, float, float, float]]:
        """Return fixed TCP poses valid only after Scout target alignment."""
        item = self.require(value)
        placement = item.placement
        poses = {
            "approach": placement.base_aligned_approach_pose_mm_deg,
            "release": placement.base_aligned_release_pose_mm_deg,
            "retreat": placement.base_aligned_retreat_pose_mm_deg,
        }
        if not placement.base_aligned_enabled or any(
            pose is None for pose in poses.values()
        ):
            raise RuntimeError(
                f"{item.item_id}: fixed base-aligned placement is not calibrated"
            )
        typed = {
            name: tuple(float(value) for value in pose)
            for name, pose in poses.items()
            if pose is not None
        }
        release = typed["release"]
        for name in ("approach", "retreat"):
            pose = typed[name]
            if pose[2] - release[2] < 50.0:
                raise RuntimeError(
                    f"{item.item_id}: base-aligned {name} needs at least 50 mm clearance"
                )
            if np.hypot(pose[0] - release[0], pose[1] - release[1]) > 15.0:
                raise RuntimeError(
                    f"{item.item_id}: base-aligned {name} must stay vertical over release"
                )
        return typed

    def ensure_place_calibrated(
        self,
        value: str,
        *,
        require_slot_centers: bool = True,
    ) -> None:
        item = self.require(value)
        if not item.placement.enabled or item.placement.release_rpy_deg is None:
            raise RuntimeError(
                f"{item.item_id}: placement is not calibrated; set placement.enabled=true "
                "and release_rpy_deg in item_catalog.yaml"
            )
        if require_slot_centers:
            self.resolved_slot_centers_mm()

    def resolved_slot_centers_mm(
        self,
    ) -> tuple[tuple[float, float, float], ...]:
        if all(center is not None for center in self.box.slot_centers_mm):
            return tuple(center for center in self.box.slot_centers_mm if center is not None)
        first = self.box.row_first_slot_center_mm
        last = self.box.row_last_slot_center_mm
        if first is None or last is None:
            raise RuntimeError(
                "box row is not calibrated; set row_first_slot_center_mm and "
                "row_last_slot_center_mm, or all 6 slot_centers_mm"
            )
        slot_count = len(self.box.slot_centers_mm)
        delta = np.asarray(last, dtype=np.float64) - np.asarray(first, dtype=np.float64)
        actual_pitch = float(np.linalg.norm(delta) / max(1, slot_count - 1))
        expected_pitch = float(self.box.row_slot_pitch_mm)
        tolerance = max(0.0, float(self.box.row_slot_pitch_tolerance_mm))
        if abs(actual_pitch - expected_pitch) > tolerance:
            raise RuntimeError(
                f"calibrated box pitch {actual_pitch:.1f}mm does not match "
                f"expected {expected_pitch:.1f}±{tolerance:.1f}mm"
            )
        return tuple(
            tuple(
                float(first[axis]) + (float(delta[axis]) * index / float(slot_count - 1))
                for axis in range(3)
            )
            for index in range(slot_count)
        )


_BOTTLE_COLOR_RANGES: dict[str, tuple[tuple[int, int, int], tuple[int, int, int]]] = {
    "orange_bottle": ((4, 70, 60), (30, 255, 255)),
    "green_bottle": ((32, 50, 40), (90, 255, 255)),
    # Dark liquid is classified by value rather than hue.
    "dark_bottle": ((0, 0, 0), (179, 255, 130)),
}

_BOTTLE_IDENTITY_ALIASES: dict[str, tuple[str, ...]] = {
    "orange_bottle": ("orange bottle", "orange drink", "橙色饮料瓶", "橙色瓶"),
    "dark_bottle": ("dark bottle", "black bottle", "dark drink", "深色饮料瓶", "黑色瓶"),
    "green_bottle": ("green bottle", "green drink", "绿色饮料瓶", "绿色瓶"),
}


def bottle_item_id_from_prompt(prompt: str) -> str | None:
    normalized = ItemCatalog._normalize(prompt)
    for item_id, aliases in _BOTTLE_IDENTITY_ALIASES.items():
        if normalized == ItemCatalog._normalize(item_id):
            return item_id
        if any(ItemCatalog._normalize(alias) in normalized for alias in aliases):
            return item_id
    return None


def bottle_identity_score(
    color_bgr: np.ndarray,
    mask: np.ndarray,
    item_id: str,
) -> float:
    """Score liquid color inside the upper-middle body of one bottle mask."""
    if item_id not in _BOTTLE_COLOR_RANGES:
        return 0.0
    image = np.asarray(color_bgr)
    mask_bool = np.asarray(mask, dtype=bool)
    ys, xs = np.where(mask_bool)
    if ys.size < 20:
        return 0.0
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    body_y0 = y0 + int(round((y1 - y0) * 0.18))
    body_y1 = y0 + int(round((y1 - y0) * 0.58))
    body_mask = mask_bool[body_y0:body_y1, x0:x1]
    if int(body_mask.sum()) < 20:
        return 0.0
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[body_y0:body_y1, x0:x1]
    lower, upper = _BOTTLE_COLOR_RANGES[item_id]
    selected = cv2.inRange(
        hsv,
        np.asarray(lower, dtype=np.uint8),
        np.asarray(upper, dtype=np.uint8),
    ).astype(bool)
    return float(np.count_nonzero(selected & body_mask) / max(1, np.count_nonzero(body_mask)))


def classify_bottle_identity(
    color_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    minimum_score: float = 0.18,
    minimum_color_margin: float = 0.045,
) -> tuple[str | None, float, dict[str, float]]:
    """Classify one bottle without treating colored-liquid shadows as dark liquid."""
    scores = {
        item_id: bottle_identity_score(color_bgr, mask, item_id)
        for item_id in ("orange_bottle", "dark_bottle", "green_bottle")
    }
    chromatic = sorted(
        (
            ("orange_bottle", scores["orange_bottle"]),
            ("green_bottle", scores["green_bottle"]),
        ),
        key=lambda value: value[1],
        reverse=True,
    )
    color_id, color_score = chromatic[0]
    color_margin = float(color_score - chromatic[1][1])
    # The broad low-value range also covers shaded orange/green liquid. Strong
    # hue evidence is therefore more specific and must take precedence.
    if color_score >= minimum_score and color_margin >= minimum_color_margin:
        return color_id, float(color_score), scores

    dark_score = float(scores["dark_bottle"])
    if dark_score >= minimum_score and color_score < minimum_score:
        return "dark_bottle", dark_score, scores
    return None, 0.0, scores


def filter_bottle_instances(
    segmentation: dict[str, Any],
    color_bgr: np.ndarray,
    item_id: str,
    *,
    minimum_score: float = 0.18,
) -> dict[str, Any]:
    """Keep only bottle masks whose liquid color matches the requested item."""
    if item_id not in _BOTTLE_COLOR_RANGES:
        return segmentation
    masks = segmentation.get("masks")
    count = int(masks.shape[0]) if hasattr(masks, "shape") and len(masks.shape) >= 3 else 0
    identity_scores: list[float] = []
    predicted_item_ids: list[str | None] = []
    for index in range(count):
        raw_mask = masks[index]
        mask_np = raw_mask.detach().cpu().numpy() if hasattr(raw_mask, "detach") else np.asarray(raw_mask)
        predicted_id, predicted_score, _ = classify_bottle_identity(
            color_bgr,
            mask_np,
            minimum_score=minimum_score,
        )
        predicted_item_ids.append(predicted_id)
        identity_scores.append(
            predicted_score if predicted_id == item_id else 0.0
        )
    keep = [
        index
        for index, predicted_id in enumerate(predicted_item_ids)
        if predicted_id == item_id and identity_scores[index] >= minimum_score
    ]
    result = dict(segmentation)
    for key in ("masks", "scores", "boxes"):
        value = result.get(key)
        if value is not None:
            result[key] = value[keep]
    result["labels"] = [item_id] * len(keep)
    result["identity_scores"] = [identity_scores[index] for index in keep]
    result["requested_item_id"] = item_id
    result["allow_scene_fallback"] = False
    result["backend"] = "yolo+bottle_identity"
    return result


class ReferenceLabelMatcher:
    """Multi-scale image matcher for printed reference labels on transparent boxes."""

    _BLOCK_HSV_RANGES = {
        "red_block": (
            ((0, 65, 45), (10, 255, 255)),
            ((168, 65, 45), (179, 255, 255)),
        ),
        "yellow_block": (((8, 55, 45), (42, 255, 255)),),
        "blue_block": (((90, 50, 35), (138, 255, 255)),),
    }

    def __init__(self, catalog: ItemCatalog) -> None:
        self.catalog = catalog
        self._references: dict[str, np.ndarray] = {}

    @staticmethod
    def _crop_nonwhite(image: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        foreground = (hsv[:, :, 1] > 24) | (hsv[:, :, 2] < 238)
        ys, xs = np.where(foreground)
        if ys.size < 20:
            return image
        pad = 4
        y0, y1 = max(0, int(ys.min()) - pad), min(image.shape[0], int(ys.max()) + pad + 1)
        x0, x1 = max(0, int(xs.min()) - pad), min(image.shape[1], int(xs.max()) + pad + 1)
        return image[y0:y1, x0:x1]

    def _reference(self, item: ItemSpec) -> np.ndarray:
        cached = self._references.get(item.item_id)
        if cached is not None:
            return cached
        image = cv2.imread(str(item.reference_image), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"failed to load reference image: {item.reference_image}")
        cropped = self._crop_nonwhite(image)
        self._references[item.item_id] = cropped
        return cropped

    @staticmethod
    def _channels(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 45, 135)
        # Lab keeps luminance and opponent-color information, so geometrically
        # similar red/yellow/blue block labels remain distinguishable.
        appearance = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        return edges, appearance

    @staticmethod
    def _marker_components(
        mask: np.ndarray,
        *,
        shape: str,
    ) -> list[tuple[int, int, int, int, int]]:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 9))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        count, _, stats, _ = cv2.connectedComponentsWithStats(closed)
        height, width = mask.shape[:2]
        minimum_area = max(40, int(round(height * width * 0.00012)))
        components: list[tuple[int, int, int, int, int]] = []
        for x, y, w, h, area in stats[1:count]:
            aspect = float(h) / max(1.0, float(w))
            if int(area) < minimum_area or w < 8 or h < 12:
                continue
            maximum_width_fraction = 0.26 if shape == "block" else 0.18
            if w > width * maximum_width_fraction or h > height * 0.40:
                continue
            # Full block symbols are close to square. The held object can hide
            # the lower part of a real symbol, so a bounded squat component is
            # retained for the stronger card-support check below. Tall
            # dark/blue box hardware (observed aspect ~= 1.50) stays rejected.
            if shape == "block" and not 0.40 <= aspect <= 1.30:
                continue
            if shape == "bottle" and not 1.55 <= aspect <= 5.0:
                continue
            if shape == "dark_bottle" and not 1.20 <= aspect <= 5.0:
                continue
            components.append((int(area), int(x), int(y), int(w), int(h)))
        return sorted(components, reverse=True)

    def _match_color_markers(
        self,
        search: np.ndarray,
        search_xywh: tuple[int, int, int, int],
        *,
        threshold: float,
        include_bottles: bool = True,
    ) -> tuple[LabelDetection, ...]:
        """Detect printed symbols with color and coarse contour geometry."""
        hsv = cv2.cvtColor(search, cv2.COLOR_BGR2HSV)
        ranges = {
            "red_block": [((0, 65, 45), (10, 255, 255)), ((168, 65, 45), (179, 255, 255))],
            # Printed yellow often shifts toward orange under the D435 auto
            # exposure. Shape separates the square marker from tall bottles.
            "yellow_block": [((8, 55, 45), (42, 255, 255))],
            "blue_block": [((90, 50, 35), (138, 255, 255))],
            "orange_bottle": [((8, 55, 45), (25, 255, 255))],
            "green_bottle": [((35, 40, 30), (90, 255, 255))],
        }
        if not include_bottles:
            ranges = {
                item_id: item_ranges
                for item_id, item_ranges in ranges.items()
                if item_id.endswith("_block")
            }
        detections: list[LabelDetection] = []
        colored_marker_centers: list[float] = []
        for item_id, item_ranges in ranges.items():
            mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
            for lower, upper in item_ranges:
                mask = cv2.bitwise_or(
                    mask,
                    cv2.inRange(
                        hsv,
                        np.asarray(lower, dtype=np.uint8),
                        np.asarray(upper, dtype=np.uint8),
                    ),
                )
            shape = "bottle" if item_id.endswith("_bottle") else "block"
            components = []
            for component in self._marker_components(mask, shape=shape):
                _, x, y, w, h = component
                # A symbol clipped by the image/ROI boundary has unreliable
                # shape and position. Overlapping scan views will observe the
                # same card fully, so fail closed on the partial observation.
                if (
                    x <= 2
                    or y <= 2
                    or x + w >= search.shape[1] - 2
                    or y + h >= search.shape[0] - 2
                ):
                    continue
                if shape == "block":
                    minimum_width = max(
                        32,
                        int(round(search.shape[1] * 0.07)),
                    )
                    minimum_height = max(
                        30,
                        int(round(search.shape[0] * 0.06)),
                    )
                    if w < minimum_width or h < minimum_height:
                        continue
                    card_support = self._white_card_support(
                        search,
                        (x, y, w, h),
                    )
                    if card_support < 0.55:
                        continue
                    # A squat component represents only the visible upper part
                    # of an occluded cube. Require it to be large and backed by
                    # a particularly clear white label card.
                    aspect = float(h) / max(1.0, float(w))
                    if (
                        aspect < 0.70
                        and (
                            w < max(80, int(round(search.shape[1] * 0.12)))
                            or card_support < 0.65
                        )
                    ):
                        continue
                components.append(component)
            if not components:
                continue
            area, x, y, w, h = components[0]
            fill = float(area) / max(1.0, float(w * h))
            confidence = min(0.99, 0.55 + (0.40 * fill))
            if confidence < threshold:
                continue
            bbox = (search_xywh[0] + x, search_xywh[1] + y, w, h)
            detections.append(
                LabelDetection(
                    item_id=item_id,
                    confidence=confidence,
                    bbox_xywh=bbox,
                    method=(
                        "color_shape_partial"
                        if shape == "block"
                        and (float(h) / max(1.0, float(w))) < 0.70
                        else "color_shape"
                    ),
                )
            )
            colored_marker_centers.append(float(x) + (float(w) / 2.0))

        # The dark bottle has no stable hue. Detect a tall low-value component,
        # excluding the already located orange/green bottle columns.
        if include_bottles:
            dark_mask = cv2.inRange(
                hsv,
                np.asarray((0, 10, 0), dtype=np.uint8),
                np.asarray((179, 255, 150), dtype=np.uint8),
            )
            for area, x, y, w, h in self._marker_components(dark_mask, shape="dark_bottle"):
                center_x = float(x) + (float(w) / 2.0)
                if any(abs(center_x - known) < max(24.0, float(w) * 1.5) for known in colored_marker_centers):
                    continue
                fill = float(area) / max(1.0, float(w * h))
                confidence = min(0.99, 0.55 + (0.40 * fill))
                if confidence >= threshold:
                    detections.append(
                        LabelDetection(
                            item_id="dark_bottle",
                            confidence=confidence,
                            bbox_xywh=(search_xywh[0] + x, search_xywh[1] + y, w, h),
                            method="color_shape",
                        )
                    )
                break
        return tuple(detections)

    @classmethod
    def _block_color_support(
        cls,
        color_bgr: np.ndarray,
        bbox_xywh: tuple[int, int, int, int],
        item_id: str,
    ) -> tuple[float, float]:
        """Return expected-hue support and colored-pixel fraction for a block."""
        ranges = cls._BLOCK_HSV_RANGES.get(item_id)
        if not ranges:
            return (1.0, 1.0)
        image = np.asarray(color_bgr)
        height, width = image.shape[:2]
        x, y, box_w, box_h = bbox_xywh
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(width, x + box_w), min(height, y + box_h)
        if x1 <= x0 or y1 <= y0:
            return (0.0, 0.0)
        hsv = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        colored = (hsv[:, :, 1] >= 45) & (hsv[:, :, 2] >= 35)
        colored_count = int(np.count_nonzero(colored))
        if colored_count == 0:
            return (0.0, 0.0)
        expected = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lower, upper in ranges:
            expected = cv2.bitwise_or(
                expected,
                cv2.inRange(
                    hsv,
                    np.asarray(lower, dtype=np.uint8),
                    np.asarray(upper, dtype=np.uint8),
                ),
            )
        expected_count = int(np.count_nonzero((expected > 0) & colored))
        return (
            float(expected_count / colored_count),
            float(colored_count / max(1, colored.size)),
        )

    @staticmethod
    def _bbox_intersection_over_smaller(
        first: tuple[int, int, int, int],
        second: tuple[int, int, int, int],
    ) -> float:
        ax, ay, aw, ah = first
        bx, by, bw, bh = second
        x0, y0 = max(ax, bx), max(ay, by)
        x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        intersection = max(0, x1 - x0) * max(0, y1 - y0)
        return float(intersection) / max(1.0, float(min(aw * ah, bw * bh)))

    @staticmethod
    def _white_card_support(
        color_bgr: np.ndarray,
        bbox_xywh: tuple[int, int, int, int],
    ) -> float:
        """Measure neutral bright paper surrounding a candidate label symbol."""
        image = np.asarray(color_bgr)
        height, width = image.shape[:2]
        x, y, box_w, box_h = bbox_xywh
        pad_x = max(8, int(round(box_w * 0.30)))
        pad_y = max(8, int(round(box_h * 0.30)))
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1 = min(width, x + box_w + pad_x)
        y1 = min(height, y + box_h + pad_y)
        if x1 <= x0 or y1 <= y0:
            return 0.0
        ring = np.ones((y1 - y0, x1 - x0), dtype=bool)
        inner_x0, inner_y0 = max(0, x - x0), max(0, y - y0)
        inner_x1 = min(x1 - x0, inner_x0 + box_w)
        inner_y1 = min(y1 - y0, inner_y0 + box_h)
        ring[inner_y0:inner_y1, inner_x0:inner_x1] = False
        if int(ring.sum()) < 80:
            return 0.0
        hsv = cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        paper = (hsv[:, :, 1] < 70) & (hsv[:, :, 2] > 110)
        return float(np.count_nonzero(paper & ring) / np.count_nonzero(ring))

    def _match_yolo_bottles(
        self,
        color_bgr: np.ndarray,
        proposals: tuple[dict[str, object], ...],
        *,
        roi_xywh: tuple[int, int, int, int],
        threshold: float,
    ) -> tuple[LabelDetection, ...]:
        """Classify generic YOLO bottle boxes into the three printed bottle IDs."""
        image = np.asarray(color_bgr)
        height, width = image.shape[:2]
        roi_x, roi_y, roi_w, roi_h = roi_xywh
        detections: list[LabelDetection] = []
        for proposal in proposals:
            raw_bbox = list(proposal.get("bbox_xywh") or [])
            if len(raw_bbox) != 4:
                continue
            x, y, box_w, box_h = (int(round(float(value))) for value in raw_bbox)
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(width, x + box_w), min(height, y + box_h)
            if x1 <= x0 or y1 <= y0:
                continue
            center_x = (x0 + x1) / 2.0
            center_y = (y0 + y1) / 2.0
            if not (
                roi_x <= center_x <= roi_x + roi_w
                and roi_y <= center_y <= roi_y + roi_h
            ):
                continue
            mask = np.zeros((height, width), dtype=bool)
            mask[y0:y1, x0:x1] = True
            item_id, identity_score, _ = classify_bottle_identity(
                image,
                mask,
                minimum_score=0.18,
                minimum_color_margin=0.045,
            )
            if item_id is None:
                continue
            yolo_score = float(proposal.get("confidence") or 0.0)
            confidence = (0.60 * yolo_score) + (0.40 * identity_score)
            if confidence < threshold:
                continue
            detections.append(
                LabelDetection(
                    item_id=item_id,
                    confidence=min(0.99, confidence),
                    bbox_xywh=(x0, y0, x1 - x0, y1 - y0),
                    method="yolo_bottle+liquid_color",
                )
            )
        return tuple(detections)

    @staticmethod
    def _roi(image: np.ndarray, roi_norm: tuple[float, float, float, float]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
        height, width = image.shape[:2]
        x0 = int(round(max(0.0, min(1.0, roi_norm[0])) * width))
        y0 = int(round(max(0.0, min(1.0, roi_norm[1])) * height))
        x1 = int(round(max(0.0, min(1.0, roi_norm[2])) * width))
        y1 = int(round(max(0.0, min(1.0, roi_norm[3])) * height))
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"invalid normalized label ROI: {roi_norm}")
        return image[y0:y1, x0:x1], (x0, y0, x1 - x0, y1 - y0)

    def match_all(
        self,
        color_bgr: np.ndarray,
        *,
        roi_norm: tuple[float, float, float, float] | None = None,
        threshold: float | None = None,
        bottle_proposals: tuple[dict[str, object], ...] | None = None,
        marker_detection_enabled: bool = True,
    ) -> tuple[tuple[LabelDetection, ...], tuple[int, int, int, int]]:
        roi_values = roi_norm or self.catalog.box.label_search_roi_norm
        search, search_xywh = self._roi(np.asarray(color_bgr), roi_values)
        required = float(
            threshold if threshold is not None else self.catalog.box.label_match_threshold
        )
        marker_detections = (
            self._match_color_markers(
                search,
                search_xywh,
                threshold=required,
                include_bottles=bottle_proposals is None,
            )
            if marker_detection_enabled
            else ()
        )
        yolo_detections = (
            self._match_yolo_bottles(
                np.asarray(color_bgr),
                bottle_proposals,
                roi_xywh=search_xywh,
                threshold=required,
            )
            if bottle_proposals is not None
            else ()
        )
        if yolo_detections:
            # A liquid patch inside a bottle may look square after thresholding
            # (especially orange liquid under warm exposure). A generic YOLO
            # bottle outline is stronger shape evidence than that inner patch.
            marker_detections = tuple(
                detection
                for detection in marker_detections
                if not any(
                    self._bbox_intersection_over_smaller(
                        detection.bbox_xywh,
                        bottle.bbox_xywh,
                    )
                    >= 0.35
                    for bottle in yolo_detections
                )
            )
        # Keep multi-template verification responsive on the live camera stream.
        # The printed labels are large enough that a 320 px search image preserves
        # their discriminating shape/color while reducing template-match work.
        original_h, original_w = search.shape[:2]
        search_scale = min(1.0, 320.0 / max(1.0, float(original_w)))
        if search_scale < 1.0:
            search = cv2.resize(
                search,
                (
                    max(1, int(round(original_w * search_scale))),
                    max(1, int(round(original_h * search_scale))),
                ),
                interpolation=cv2.INTER_AREA,
            )
        search_edges, search_appearance = self._channels(search)

        candidates: list[LabelDetection] = []
        search_h, search_w = search.shape[:2]
        for candidate in self.catalog.items.values():
            best_score = -1.0
            best_bbox: tuple[int, int, int, int] | None = None
            reference = self._reference(candidate)
            ref_h, ref_w = reference.shape[:2]
            aspect = ref_h / max(1.0, float(ref_w))
            min_width = max(20, int(round(search_w * 0.08)))
            max_width = min(search_w, int(round(search_w * 0.55)))
            for template_w in np.unique(np.geomspace(min_width, max_width, num=18, dtype=int)):
                template_h = int(round(float(template_w) * aspect))
                if template_h < 24 or template_h > search_h or template_w > search_w:
                    continue
                resized = cv2.resize(
                    reference,
                    (int(template_w), template_h),
                    interpolation=cv2.INTER_AREA,
                )
                template_edges, template_appearance = self._channels(resized)
                edge_map = cv2.matchTemplate(
                    search_edges,
                    template_edges,
                    cv2.TM_CCOEFF_NORMED,
                )
                appearance_map = cv2.matchTemplate(
                    search_appearance,
                    template_appearance,
                    cv2.TM_CCOEFF_NORMED,
                )
                combined = (0.45 * edge_map) + (0.55 * appearance_map)
                _, score, _, location = cv2.minMaxLoc(combined)
                if float(score) > best_score:
                    best_score = float(score)
                    best_bbox = (
                        int(round(search_xywh[0] + (location[0] / search_scale))),
                        int(round(search_xywh[1] + (location[1] / search_scale))),
                        int(round(template_w / search_scale)),
                        int(round(template_h / search_scale)),
                    )
            template_required = max(required, 0.50)
            card_support = (
                self._white_card_support(np.asarray(color_bgr), best_bbox)
                if best_bbox is not None
                else 0.0
            )
            block_hue_support, block_colored_fraction = (
                self._block_color_support(
                    np.asarray(color_bgr),
                    best_bbox,
                    candidate.item_id,
                )
                if best_bbox is not None
                and candidate.item_id in self._BLOCK_HSV_RANGES
                else (1.0, 1.0)
            )
            if (
                best_bbox is not None
                and best_score >= template_required
                and card_support >= 0.55
                and block_hue_support >= 0.70
                and block_colored_fraction >= 0.20
            ):
                candidates.append(
                    LabelDetection(
                        item_id=candidate.item_id,
                        confidence=max(0.0, best_score),
                        bbox_xywh=best_bbox,
                        method="edge_lab_template",
                    )
                )

        # Each physical box may contribute only one label. If two templates peak
        # on the same paper card, retain the stronger one and fail closed later
        # because fewer than six unique labels remain.
        unique: list[LabelDetection] = []
        seen_item_ids: set[str] = set()
        fused_candidates = [*marker_detections, *yolo_detections, *candidates]
        for candidate in sorted(
            fused_candidates,
            key=lambda value: value.confidence,
            reverse=True,
        ):
            if candidate.item_id in seen_item_ids:
                continue
            cx = candidate.bbox_xywh[0] + (candidate.bbox_xywh[2] / 2.0)
            cy = candidate.bbox_xywh[1] + (candidate.bbox_xywh[3] / 2.0)
            overlaps_existing = False
            for kept in unique:
                kx = kept.bbox_xywh[0] + (kept.bbox_xywh[2] / 2.0)
                ky = kept.bbox_xywh[1] + (kept.bbox_xywh[3] / 2.0)
                separation = max(
                    12.0,
                    0.30 * min(candidate.bbox_xywh[2], kept.bbox_xywh[2]),
                )
                if (
                    (
                        abs(cx - kx) <= separation
                        and abs(cy - ky) <= separation
                    )
                    or self._bbox_intersection_over_smaller(
                        candidate.bbox_xywh,
                        kept.bbox_xywh,
                    )
                    >= 0.40
                ):
                    overlaps_existing = True
                    break
            if not overlaps_existing:
                unique.append(candidate)
                seen_item_ids.add(candidate.item_id)
        ordered = tuple(
            sorted(
                unique,
                key=lambda value: value.bbox_xywh[0] + (value.bbox_xywh[2] / 2.0),
            )
        )
        return ordered, search_xywh

    def match_expected(
        self,
        color_bgr: np.ndarray,
        expected_item_id: str,
        *,
        roi_norm: tuple[float, float, float, float] | None = None,
        threshold: float | None = None,
        bottle_proposals: tuple[dict[str, object], ...] | None = None,
    ) -> LabelMatch:
        item = self.catalog.require(expected_item_id)
        detections, search_xywh = self.match_all(
            color_bgr,
            roi_norm=roi_norm,
            threshold=threshold,
            bottle_proposals=bottle_proposals,
        )
        target = next(
            (detection for detection in detections if detection.item_id == item.item_id),
            None,
        )
        detected_item_ids = tuple(detection.item_id for detection in detections)
        slot_index = detected_item_ids.index(item.item_id) if target is not None else None
        accepted = (
            target is not None
            and len(detections) == len(self.catalog.items)
            and len(set(detected_item_ids)) == len(self.catalog.items)
        )
        return LabelMatch(
            expected_item_id=item.item_id,
            matched_item_id=item.item_id if target is not None else None,
            confidence=float(target.confidence) if target is not None else 0.0,
            bbox_xywh=target.bbox_xywh if target is not None else None,
            search_roi_xywh=search_xywh,
            accepted=accepted,
            slot_index=slot_index,
            detected_item_ids=detected_item_ids,
            detections=detections,
        )

    def localize_box_row(
        self,
        *,
        depth_meters: np.ndarray,
        camera_k: tuple[float, ...] | list[float],
        base_to_camera: np.ndarray,
        detections: tuple[LabelDetection, ...],
        target_item_id: str,
        table_z_m: float,
    ) -> BoxRowLocalization:
        """Locate a tightly packed box row from opaque rear-wall labels."""
        if len(detections) != len(self.catalog.items):
            raise RuntimeError(
                f"dynamic box localization requires 6 labels, got {len(detections)}"
            )
        label_centers = self.project_label_centers(
            depth_meters=depth_meters,
            camera_k=camera_k,
            base_to_camera=base_to_camera,
            detections=detections,
        )
        transform = np.asarray(base_to_camera, dtype=np.float64).reshape(4, 4)
        return self.localize_box_row_from_points(
            detections=detections,
            label_centers_base_m=label_centers,
            target_item_id=target_item_id,
            table_z_m=table_z_m,
            camera_xy_base=tuple(float(value) for value in transform[:2, 3]),
        )

    def project_label_centers(
        self,
        *,
        depth_meters: np.ndarray,
        camera_k: tuple[float, ...] | list[float],
        base_to_camera: np.ndarray,
        detections: tuple[LabelDetection, ...],
        depth_override_m: float | None = None,
    ) -> tuple[tuple[float, float, float], ...]:
        """Project label centers, optionally using a shared row-plane depth."""
        depth = np.asarray(depth_meters, dtype=np.float64)
        if depth.ndim != 2:
            raise ValueError("depth_meters must be a 2D array")
        intrinsics = tuple(float(value) for value in camera_k)
        if len(intrinsics) != 9:
            raise ValueError("camera_k must contain 9 values")
        fx, fy = intrinsics[0], intrinsics[4]
        cx, cy = intrinsics[2], intrinsics[5]
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("camera intrinsics have invalid focal length")
        transform = np.asarray(base_to_camera, dtype=np.float64).reshape(4, 4)

        label_centers: list[tuple[float, float, float]] = []
        for detection in detections:
            x, y, width, height = detection.bbox_xywh
            x0 = max(0, x + int(round(width * 0.15)))
            x1 = min(depth.shape[1], x + width - int(round(width * 0.15)))
            y0 = max(0, y + int(round(height * 0.15)))
            y1 = min(depth.shape[0], y + height - int(round(height * 0.15)))
            if depth_override_m is None:
                patch = depth[y0:y1, x0:x1]
                valid = patch[
                    np.isfinite(patch) & (patch > 0.10) & (patch < 3.0)
                ]
                if valid.size < 12:
                    raise RuntimeError(
                        f"{detection.item_id}: insufficient valid label depth "
                        f"samples ({valid.size})"
                    )
                z_m = float(np.median(valid))
            else:
                z_m = float(depth_override_m)
                if not np.isfinite(z_m) or not 0.10 < z_m < 3.0:
                    raise ValueError(
                        f"depth_override_m is invalid: {depth_override_m}"
                    )
            u = float(x) + (float(width) / 2.0)
            v = float(y) + (float(height) / 2.0)
            camera_point = np.array(
                [
                    ((u - cx) / fx) * z_m,
                    ((v - cy) / fy) * z_m,
                    z_m,
                    1.0,
                ],
                dtype=np.float64,
            )
            base_point = (transform @ camera_point)[:3]
            label_centers.append(tuple(float(value) for value in base_point))
        return tuple(label_centers)

    def localize_box_row_from_points(
        self,
        *,
        detections: tuple[LabelDetection, ...],
        label_centers_base_m: tuple[tuple[float, float, float], ...],
        target_item_id: str,
        table_z_m: float,
        camera_xy_base: tuple[float, float],
        image_right_direction_base_xy: tuple[float, float] | None = None,
    ) -> BoxRowLocalization:
        """Fit the six-box row from fused 3D label observations."""
        if len(detections) != len(self.catalog.items):
            raise RuntimeError(
                f"dynamic box localization requires 6 labels, got {len(detections)}"
            )
        if len(label_centers_base_m) != len(detections):
            raise ValueError("label center count must equal detection count")
        detected_item_ids = tuple(detection.item_id for detection in detections)
        if len(set(detected_item_ids)) != len(self.catalog.items):
            raise RuntimeError("dynamic box localization requires 6 unique labels")

        base_points = np.asarray(label_centers_base_m, dtype=np.float64).reshape(-1, 3)
        planar_mean = np.mean(base_points[:, :2], axis=0)
        _, _, principal_axes = np.linalg.svd(
            base_points[:, :2] - planar_mean,
            full_matrices=False,
        )
        row_direction = principal_axes[0]
        if image_right_direction_base_xy is not None:
            order_direction = np.asarray(
                image_right_direction_base_xy,
                dtype=np.float64,
            ).reshape(2)
        else:
            order_direction = base_points[-1, :2] - base_points[0, :2]
        if float(np.dot(row_direction, order_direction)) < 0.0:
            row_direction = -row_direction
        row_coordinates = (base_points[:, :2] - planar_mean) @ row_direction
        order = np.argsort(row_coordinates)
        base_points = base_points[order]
        row_coordinates = row_coordinates[order]
        ordered_detections = tuple(detections[int(index)] for index in order)
        planar_steps = np.diff(row_coordinates)
        raw_adjacent_pitch_mm = tuple(
            float(step * 1000.0) for step in planar_steps
        )
        expected_pitch = float(self.catalog.box.row_slot_pitch_mm)
        pitch_tolerance = max(
            1.0,
            float(self.catalog.box.row_slot_pitch_tolerance_mm),
        )
        expected_pitch_m = expected_pitch / 1000.0
        slot_indices = np.arange(len(row_coordinates), dtype=np.float64)
        fitted_origin = float(
            np.median(row_coordinates - (slot_indices * expected_pitch_m))
        )
        fitted_coordinates = fitted_origin + (slot_indices * expected_pitch_m)
        fit_residual_mm = tuple(
            float(value * 1000.0)
            for value in (row_coordinates - fitted_coordinates)
        )
        residual_inliers = sum(
            abs(value) <= pitch_tolerance for value in fit_residual_mm
        )
        max_residual = max(abs(value) for value in fit_residual_mm)
        hard_pitch_tolerance = pitch_tolerance * 2.0
        hard_pitch_ok = all(
            abs(value - expected_pitch) <= hard_pitch_tolerance
            for value in raw_adjacent_pitch_mm
        )
        if (
            residual_inliers < len(row_coordinates) - 1
            or max_residual > hard_pitch_tolerance
            or not hard_pitch_ok
        ):
            raise RuntimeError(
                "detected box-row pitch is inconsistent with tightly packed boxes: "
                f"actual={[round(value, 1) for value in raw_adjacent_pitch_mm]}mm "
                f"fit_residual={[round(value, 1) for value in fit_residual_mm]}mm "
                f"expected={expected_pitch:.1f}mm"
            )
        adjacent_pitch_mm = (expected_pitch,) * (len(row_coordinates) - 1)

        if float(row_coordinates[-1] - row_coordinates[0]) < 0.50:
            raise RuntimeError("detected box row direction is degenerate")
        target_index = next(
            (
                index
                for index, detection in enumerate(ordered_detections)
                if detection.item_id == target_item_id
            ),
            None,
        )
        if target_index is None:
            raise RuntimeError(f"target label not found in box row: {target_item_id}")

        camera_xy = np.asarray(camera_xy_base, dtype=np.float64).reshape(2)
        # Project the colored-symbol center onto the fitted rear-wall row. This
        # removes vertical offsets caused by tall bottle art versus square art.
        target_xy = planar_mean + (row_direction * fitted_coordinates[target_index])
        toward_camera = camera_xy - target_xy
        interior_xy = toward_camera - (row_direction * float(np.dot(toward_camera, row_direction)))
        interior_norm = float(np.linalg.norm(interior_xy))
        if interior_norm < 0.05:
            raise RuntimeError("cannot determine box interior direction from camera pose")
        interior_xy /= interior_norm
        # The long side (180 mm) follows the row; the short side (132 mm)
        # runs from the rear label wall toward the camera/open box interior.
        box_depth_m = float(self.catalog.box.outer_size_m[1])
        center_xy = target_xy + (interior_xy * (box_depth_m / 2.0))
        center_z = float(table_z_m) + (float(self.catalog.box.outer_size_m[2]) / 2.0)
        center = (float(center_xy[0]), float(center_xy[1]), center_z)
        box_centers = tuple(
            (
                float(
                    planar_mean[0]
                    + (row_direction[0] * coordinate)
                    + (interior_xy[0] * (box_depth_m / 2.0))
                ),
                float(
                    planar_mean[1]
                    + (row_direction[1] * coordinate)
                    + (interior_xy[1] * (box_depth_m / 2.0))
                ),
                center_z,
            )
            for coordinate in fitted_coordinates
        )
        return BoxRowLocalization(
            target_item_id=target_item_id,
            slot_index=int(target_index),
            box_center_base_m=center,
            box_centers_base_m=box_centers,
            label_centers_base_m=tuple(
                tuple(float(value) for value in point)
                for point in base_points
            ),
            adjacent_pitch_mm=adjacent_pitch_mm,
            raw_adjacent_pitch_mm=raw_adjacent_pitch_mm,
            fit_residual_mm=fit_residual_mm,
            interior_direction_base=(
                float(interior_xy[0]),
                float(interior_xy[1]),
                0.0,
            ),
        )
