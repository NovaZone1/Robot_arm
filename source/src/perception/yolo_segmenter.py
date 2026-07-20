from __future__ import annotations

import logging
from typing import Any

import cv2
import torch

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


def _build_empty_result(device: torch.device) -> dict[str, Any]:
    return {
        "masks": torch.empty((0, 1, 1), dtype=torch.bool, device=device),
        "scores": torch.empty((0,), dtype=torch.float32, device=device),
        "boxes": torch.empty((0, 4), dtype=torch.float32, device=device),
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

    @property
    def model(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(self._model_name, verbose=False)
        return self._model

    def segment_text(self, color_bgr, text_prompt: str) -> dict[str, Any]:
        """Run YOLOv8-seg and filter detections by a COCO class prompt."""
        target_ids = _match_class_ids(text_prompt)
        if not target_ids:
            return _build_empty_result(torch.device(self.device_str))

        results = self.model(color_bgr, conf=self._conf_threshold, verbose=False)
        r = results[0]

        if r.boxes is None or r.masks is None:
            return _build_empty_result(torch.device(self.device_str))

        boxes_xyxy = r.boxes.xyxy  # (M, 4) or None
        cls_ids = r.boxes.cls  # (M,)
        scores = r.boxes.conf  # (M,)

        if boxes_xyxy is None or cls_ids is None or scores is None:
            return _build_empty_result(torch.device(self.device_str))

        # Filter by matched class IDs
        keep_mask = torch.isin(cls_ids.to(torch.int), torch.tensor(target_ids, device=cls_ids.device))
        indices = keep_mask.nonzero(as_tuple=False).squeeze(-1)
        if len(indices) == 0:
            return _build_empty_result(torch.device(self.device_str))

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
        return {
            "masks": binary_masks.to(torch.bool),
            "scores": kept_scores.to(torch.float32),
            "boxes": kept_boxes.to(torch.float32),
        }
