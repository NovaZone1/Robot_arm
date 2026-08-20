"""Classify already-localized competition objects with the trained crop model.

This is deliberately a *second-stage* classifier: the existing segmenter still
finds a physical object and supplies a mask/box, then this model identifies the
object inside that box.  It avoids treating every other object visible in the
training photographs as a negative detection example.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import numpy as np


CLASS_NAMES = (
    "blue_block",
    "dark_bottle",
    "green_bottle",
    "orange_bottle",
    "red_block",
    "yellow_block",
)


def default_model_path() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "object_crop_cls6" / "train" / "weights" / "best.pt"


class ObjectCropClassifier:
    """Lazy YOLO classification wrapper for the six competition objects."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        configured = os.environ.get("ROBOT_GRASP_OBJECT_CLASSIFIER", "").strip()
        self.model_path = Path(configured or model_path or default_model_path()).expanduser()
        self._model = None

    @property
    def available(self) -> bool:
        return self.model_path.is_file()

    @property
    def model(self):
        if self._model is None:
            if not self.available:
                raise FileNotFoundError(f"object crop classifier not found: {self.model_path}")
            from ultralytics import YOLO

            self._model = YOLO(str(self.model_path))
        return self._model

    def predict(self, crop_bgr: np.ndarray) -> tuple[str | None, float, dict[str, float]]:
        """Return item id, confidence and all class confidences for one crop."""
        crop = np.asarray(crop_bgr)
        if crop.ndim != 3 or crop.shape[0] < 8 or crop.shape[1] < 8:
            return None, 0.0, {}
        result = self.model.predict(crop, imgsz=224, verbose=False)[0]
        probs = getattr(result, "probs", None)
        if probs is None or probs.top1 is None:
            return None, 0.0, {}
        names = result.names or {index: name for index, name in enumerate(CLASS_NAMES)}
        raw = probs.data.detach().cpu().numpy()
        scores = {
            str(names.get(index, index)): float(score)
            for index, score in enumerate(raw)
        }
        index = int(probs.top1)
        item_id = str(names.get(index, index))
        return item_id, float(probs.top1conf.item()), scores


def filter_instances_by_crop_classifier(
    segmentation: dict[str, Any],
    color_bgr: np.ndarray,
    requested_item_id: str,
    *,
    classifier: ObjectCropClassifier,
    minimum_confidence: float = 0.58,
    crop_padding_ratio: float = 0.12,
) -> dict[str, Any] | None:
    """Keep candidate instances classified as ``requested_item_id``.

    ``None`` means there was no sufficiently confident new-model match and the
    caller should use its established fallback.  This makes deployment safe on
    scenes outside the first small training set.
    """
    if requested_item_id not in CLASS_NAMES or not classifier.available:
        return None
    boxes = segmentation.get("boxes")
    masks = segmentation.get("masks")
    count = int(boxes.shape[0]) if hasattr(boxes, "shape") else 0
    if count == 0 or masks is None:
        return None
    image = np.asarray(color_bgr)
    height, width = image.shape[:2]
    keep: list[int] = []
    confidences: list[float] = []
    for index in range(count):
        raw_box = boxes[index]
        values = raw_box.detach().cpu().tolist() if hasattr(raw_box, "detach") else list(raw_box)
        x0, y0, x1, y1 = (float(value) for value in values)
        pad_x = (x1 - x0) * crop_padding_ratio
        pad_y = (y1 - y0) * crop_padding_ratio
        xa = max(0, int(np.floor(x0 - pad_x)))
        ya = max(0, int(np.floor(y0 - pad_y)))
        xb = min(width, int(np.ceil(x1 + pad_x)))
        yb = min(height, int(np.ceil(y1 + pad_y)))
        predicted, confidence, _ = classifier.predict(image[ya:yb, xa:xb])
        if predicted == requested_item_id and confidence >= minimum_confidence:
            keep.append(index)
            confidences.append(confidence)
    if not keep:
        return None
    result = dict(segmentation)
    for key in ("masks", "scores", "boxes"):
        value = result.get(key)
        if value is not None:
            result[key] = value[keep]
    result["labels"] = [requested_item_id] * len(keep)
    result["identity_scores"] = confidences
    result["requested_item_id"] = requested_item_id
    result["allow_scene_fallback"] = False
    result["backend"] = f"{segmentation.get('backend', 'segmentation')}+crop_classifier"
    return result
