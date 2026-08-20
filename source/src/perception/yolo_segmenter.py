from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
import torch

from src.perception.item_catalog import bottle_item_id_from_prompt, filter_bottle_instances
from src.perception.object_crop_classifier import (
    CLASS_NAMES as CROP_CLASS_NAMES,
    ObjectCropClassifier,
    filter_instances_by_crop_classifier,
)

_log = logging.getLogger(__name__)

COCO_CLASS_ID: dict[str, int] = {
    "person": 0,
    "bicycle": 1,
    "car": 2,
    "motorcycle": 3,
    "airplane": 4,
    "bus": 5,
    "train": 6,
    "truck": 7,
    "boat": 8,
    "traffic light": 9,
    "fire hydrant": 10,
    "stop sign": 11,
    "parking meter": 12,
    "bench": 13,
    "bird": 14,
    "cat": 15,
    "dog": 16,
    "horse": 17,
    "sheep": 18,
    "cow": 19,
    "elephant": 20,
    "bear": 21,
    "zebra": 22,
    "giraffe": 23,
    "backpack": 24,
    "umbrella": 25,
    "handbag": 26,
    "tie": 27,
    "suitcase": 28,
    "frisbee": 29,
    "skis": 30,
    "snowboard": 31,
    "sports ball": 32,
    "kite": 33,
    "baseball bat": 34,
    "baseball glove": 35,
    "skateboard": 36,
    "surfboard": 37,
    "tennis racket": 38,
    "bottle": 39,
    "wine glass": 40,
    "cup": 41,
    "fork": 42,
    "knife": 43,
    "spoon": 44,
    "bowl": 45,
    "banana": 46,
    "apple": 47,
    "sandwich": 48,
    "orange": 49,
    "broccoli": 50,
    "carrot": 51,
    "hot dog": 52,
    "pizza": 53,
    "donut": 54,
    "cake": 55,
    "chair": 56,
    "couch": 57,
    "potted plant": 58,
    "bed": 59,
    "dining table": 60,
    "toilet": 61,
    "tv": 62,
    "laptop": 63,
    "mouse": 64,
    "remote": 65,
    "keyboard": 66,
    "cell phone": 67,
    "microwave": 68,
    "oven": 69,
    "toaster": 70,
    "sink": 71,
    "refrigerator": 72,
    "book": 73,
    "clock": 74,
    "vase": 75,
    "scissors": 76,
    "teddy bear": 77,
    "hair drier": 78,
    "toothbrush": 79,
}


_BLOCK_KEYWORDS = ("block", "cube", "物块", "方块", "立方体")
_BLOCK_COLOR_ALIASES: dict[str, tuple[str, ...]] = {
    "red": ("red", "红", "紅"),
    "yellow": ("yellow", "黄", "黃"),
    "blue": ("blue", "蓝", "藍"),
}
_BLOCK_HSV_RANGES: dict[str, tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...]] = {
    "red": (
        ((0, 70, 45), (12, 255, 255)),
        ((168, 70, 45), (179, 255, 255)),
    ),
    # The yellow cube in the current D435 auto-colour profile is centred near
    # H=23.  The former upper bound of 40 admitted the lime-green drink (and
    # its reflections), which then looked like a valid yellow grasp target.
    "yellow": (((17, 80, 65), (32, 255, 255)),),
    "blue": (((88, 65, 40), (138, 255, 255)),),
}


def _catalog_item_id_from_prompt(text_prompt: str) -> str | None:
    """Resolve only the six trained-object prompts without loading YAML."""
    prompt = text_prompt.lower().replace("_", " ")
    bottle = bottle_item_id_from_prompt(prompt)
    if bottle is not None:
        return bottle
    for color in ("red", "yellow", "blue"):
        if color in prompt and any(keyword in prompt for keyword in _BLOCK_KEYWORDS):
            return f"{color}_block"
    return None


def _build_empty_result(
    device: torch.device,
    *,
    image_shape: tuple[int, int] = (1, 1),
    backend: str = "yolo",
    allow_scene_fallback: bool = True,
) -> dict[str, Any]:
    height, width = image_shape
    return {
        "masks": torch.empty((0, height, width), dtype=torch.bool, device=device),
        "scores": torch.empty((0,), dtype=torch.float32, device=device),
        "boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
        "labels": [],
        "backend": backend,
        "allow_scene_fallback": allow_scene_fallback,
    }


def _match_block_colors(text_prompt: str) -> list[str] | None:
    """Return requested block colors, or ``None`` for a non-block prompt."""
    prompt_lower = text_prompt.lower().strip()
    if not any(keyword in prompt_lower for keyword in _BLOCK_KEYWORDS):
        return None
    matched = [
        color
        for color, aliases in _BLOCK_COLOR_ALIASES.items()
        if any(alias in prompt_lower for alias in aliases)
    ]
    return matched or list(_BLOCK_COLOR_ALIASES)


def _segment_color_blocks(
    color_bgr: np.ndarray,
    colors: list[str],
    *,
    device: torch.device,
) -> dict[str, Any]:
    """Segment fixed red/yellow/blue printed blocks with HSV connected components."""
    image = np.asarray(color_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("color_bgr must be an HxWx3 BGR image")

    height, width = image.shape[:2]
    image_area = height * width
    # The competition blocks are 60 mm cubes. At the verified observation
    # distance they occupy thousands of pixels; small colored bottle caps and
    # label fragments must not become graspable instances.
    min_area = max(900, int(round(image_area * 0.003)))
    max_area = int(round(image_area * 0.35))
    min_box_side = max(28, int(round(min(height, width) * 0.08)))
    kernel_size = 5 if min(height, width) >= 240 else 3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    instances: list[tuple[int, np.ndarray, list[float], float, str]] = []
    for color in colors:
        color_instances: list[tuple[int, np.ndarray, list[float], float, str]] = []
        color_mask = np.zeros((height, width), dtype=np.uint8)
        for lower, upper in _BLOCK_HSV_RANGES[color]:
            color_mask = cv2.bitwise_or(
                color_mask,
                cv2.inRange(
                    hsv,
                    np.asarray(lower, dtype=np.uint8),
                    np.asarray(upper, dtype=np.uint8),
                ),
            )
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)

        component_count, component_map, stats, _ = cv2.connectedComponentsWithStats(
            color_mask,
            connectivity=8,
        )
        for component_index in range(1, component_count):
            x, y, box_width, box_height, area = (
                int(value) for value in stats[component_index]
            )
            if area < min_area or area > max_area or box_width <= 0 or box_height <= 0:
                continue
            if min(box_width, box_height) < min_box_side:
                continue
            aspect_ratio = box_width / float(box_height)
            # A bottle body can be tall but still pass the old 0.55 limit.
            # The three competition cubes remain close to square even with
            # perspective, so reject vertical bottle-shaped colour patches.
            if not 0.72 <= aspect_ratio <= 1.65:
                continue

            component = component_map == component_index
            contours, _ = cv2.findContours(
                component.astype(np.uint8),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if not contours:
                continue
            contour = max(contours, key=cv2.contourArea)
            hull_area = float(cv2.contourArea(cv2.convexHull(contour)))
            solidity = area / hull_area if hull_area > 0.0 else 0.0
            if solidity < 0.55:
                continue

            score = min(0.99, 0.75 + 0.20 * solidity)
            color_instances.append(
                (
                    area,
                    component,
                    [float(x), float(y), float(x + box_width - 1), float(y + box_height - 1)],
                    score,
                    f"{color} block",
                )
            )
        if color_instances:
            # There is one fixed competition block per color. Keeping only the
            # largest valid region prevents smaller same-hue objects (for
            # example a purple/blue bottle cap) from entering GraspNet.
            instances.append(max(color_instances, key=lambda item: item[0]))

    instances.sort(key=lambda item: item[0], reverse=True)
    if not instances:
        return _build_empty_result(
            device,
            image_shape=(height, width),
            backend="color_block",
            allow_scene_fallback=False,
        )

    masks_np = np.stack([item[1] for item in instances], axis=0)
    boxes_np = np.asarray([item[2] for item in instances], dtype=np.float32)
    scores_np = np.asarray([item[3] for item in instances], dtype=np.float32)
    return {
        "masks": torch.from_numpy(masks_np).to(device=device, dtype=torch.bool),
        "scores": torch.from_numpy(scores_np).to(device=device, dtype=torch.float32),
        "boxes": torch.from_numpy(boxes_np).to(device=device, dtype=torch.float32),
        "labels": [item[4] for item in instances],
        "backend": "color_block",
        "allow_scene_fallback": False,
    }


def _match_class_ids(text_prompt: str) -> list[int]:
    """Match a text prompt to COCO class IDs. Returns empty list if no match."""
    prompt_lower = text_prompt.lower().strip()
    # Exact match
    if prompt_lower in COCO_CLASS_ID:
        return [COCO_CLASS_ID[prompt_lower]]
    # Substring match: prompt contains class name or vice versa
    matches: list[int] = []
    for name, cid in COCO_CLASS_ID.items():
        if prompt_lower in name or name in prompt_lower:
            matches.append(cid)
    # Remove exact duplicates while preserving order
    seen: set[int] = set()
    result = [cid for cid in matches if not (cid in seen or seen.add(cid))]
    if not result:
        _log.warning("YOLOSegmenter: no COCO class match for prompt %r", text_prompt)
    return result


class YOLOSegmenter:
    """YOLOv8 instance-segmentation wrapper used by the grasp pipeline.

    Uses the Ultralytics YOLOv8-seg model on the 80 COCO classes. Text prompts
    are mapped to COCO class names (exact or substring match). The returned
    dict exposes the pipeline's standard ``masks`` / ``scores`` / ``boxes`` keys.
    """

    def __init__(
        self,
        device: str = "cuda",
        checkpoint_path: str = "",
        model_name: str = "yolov8n-seg.pt",
        conf_threshold: float = 0.25,
    ):
        if not torch.cuda.is_available() and device == "cuda":
            device = "cpu"
        self.device_str = device
        self._model_name = model_name
        self._conf_threshold = float(conf_threshold)
        self._model = None  # lazy load
        self._crop_classifier: ObjectCropClassifier | None = None

    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self._model_name, verbose=False)
        return self._model

    @property
    def crop_classifier(self) -> ObjectCropClassifier:
        if self._crop_classifier is None:
            self._crop_classifier = ObjectCropClassifier()
        return self._crop_classifier

    def _prefer_crop_classifier(
        self,
        segmentation: dict[str, Any],
        color_bgr: np.ndarray,
        requested_item_id: str | None,
    ) -> dict[str, Any]:
        if requested_item_id not in CROP_CLASS_NAMES:
            return segmentation
        try:
            classified = filter_instances_by_crop_classifier(
                segmentation,
                np.asarray(color_bgr),
                requested_item_id,
                classifier=self.crop_classifier,
            )
        except Exception as exc:
            _log.warning("object crop classifier unavailable; using legacy filter: %s", exc)
            return segmentation
        return classified if classified is not None else segmentation

    def segment_text(self, color_bgr, text_prompt: str) -> dict[str, Any]:
        """Segment a target with color blocks first, otherwise YOLOv8 COCO."""
        device = torch.device(self.device_str)
        block_colors = _match_block_colors(text_prompt)
        if block_colors is not None:
            segmented = _segment_color_blocks(color_bgr, block_colors, device=device)
            return self._prefer_crop_classifier(
                segmented, color_bgr, _catalog_item_id_from_prompt(text_prompt)
            )

        bottle_item_id = bottle_item_id_from_prompt(text_prompt)

        def empty_result() -> dict[str, Any]:
            result = _build_empty_result(
                device,
                image_shape=tuple(color_bgr.shape[:2]),
                backend=("yolo+bottle_identity" if bottle_item_id else "yolo"),
                allow_scene_fallback=bottle_item_id is None,
            )
            if bottle_item_id is not None:
                result["requested_item_id"] = bottle_item_id
            return result

        target_ids = _match_class_ids(text_prompt)
        if not target_ids:
            return empty_result()

        results = self.model(color_bgr, conf=self._conf_threshold, verbose=False)
        r = results[0]

        if r.boxes is None or r.masks is None:
            return empty_result()

        boxes_xyxy = r.boxes.xyxy  # (M, 4) or None
        cls_ids = r.boxes.cls  # (M,)
        scores = r.boxes.conf  # (M,)

        if boxes_xyxy is None or cls_ids is None or scores is None:
            return empty_result()

        # Filter by matched class IDs
        keep_mask = torch.isin(cls_ids.to(torch.int), torch.tensor(target_ids, device=cls_ids.device))
        indices = keep_mask.nonzero(as_tuple=False).squeeze(-1)
        if len(indices) == 0:
            return empty_result()

        kept_boxes = boxes_xyxy[indices]
        kept_scores = scores[indices]

        # Resize masks to the original image resolution
        masks_tensor = r.masks.data[indices]  # (N, H_mask, W_mask) float [0,1]
        h_img, w_img = r.orig_shape[:2]
        if masks_tensor.shape[1:3] != (h_img, w_img):
            masks_tensor = torch.nn.functional.interpolate(
                masks_tensor.unsqueeze(0),
                size=(h_img, w_img),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)

        binary_masks = masks_tensor > 0.5  # (N, H, W) bool
        segmentation = {
            "masks": binary_masks.to(torch.bool),
            "scores": kept_scores.to(torch.float32),
            "boxes": kept_boxes.to(torch.float32),
            "labels": [],
            "backend": "yolo",
            "allow_scene_fallback": True,
        }
        if bottle_item_id is not None:
            classified = self._prefer_crop_classifier(segmentation, color_bgr, bottle_item_id)
            # The old liquid-colour rule is retained when the learned classifier
            # has no confident target, e.g. under an unseen illumination.
            segmentation = (
                classified
                if classified.get("backend", "").endswith("+crop_classifier")
                else filter_bottle_instances(
                    segmentation,
                    np.asarray(color_bgr),
                    bottle_item_id,
                )
            )
        return segmentation
