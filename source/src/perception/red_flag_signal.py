from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class RedFlagDetection:
    found: bool
    area_ratio: float = 0.0
    center_u_norm: float = 0.0
    center_v_norm: float = 0.0
    bbox_xywh: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class RedFlagSignalConfig:
    roi_norm: tuple[float, float, float, float] = (0.10, 0.0, 1.0, 0.95)
    saturation_min: int = 110
    value_min: int = 90
    component_min_area_ratio: float = 0.002
    peak_area_ratio: float = 0.03
    motion_window_s: float = 3.0
    min_valid_detections: int = 4
    min_axis_range_norm: float = 0.18
    min_path_length_norm: float = 0.30
    min_direction_step_norm: float = 0.025
    min_direction_reversals: int = 1


def detect_red_flag(
    color_bgr: np.ndarray,
    config: RedFlagSignalConfig,
) -> tuple[RedFlagDetection, np.ndarray]:
    image = np.asarray(color_bgr)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("color_bgr must be an HxWx3 image")
    height, width = image.shape[:2]
    x0n, y0n, x1n, y1n = config.roi_norm
    x0 = int(round(max(0.0, min(1.0, x0n)) * width))
    y0 = int(round(max(0.0, min(1.0, y0n)) * height))
    x1 = int(round(max(0.0, min(1.0, x1n)) * width))
    y1 = int(round(max(0.0, min(1.0, y1n)) * height))
    if x1 <= x0 or y1 <= y0:
        raise ValueError("roi_norm must describe a non-empty image region")

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_a = np.asarray((0, config.saturation_min, config.value_min), dtype=np.uint8)
    upper_a = np.asarray((12, 255, 255), dtype=np.uint8)
    lower_b = np.asarray((168, config.saturation_min, config.value_min), dtype=np.uint8)
    upper_b = np.asarray((179, 255, 255), dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_a, upper_a)
    mask = cv2.bitwise_or(mask, cv2.inRange(hsv, lower_b, upper_b))
    roi_mask = np.zeros_like(mask)
    roi_mask[y0:y1, x0:x1] = 255
    mask = cv2.bitwise_and(mask, roi_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8,
    )
    if count <= 1:
        return RedFlagDetection(found=False), mask
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = int(stats[component, cv2.CC_STAT_AREA])
    area_ratio = area / float(width * height)
    if area_ratio < config.component_min_area_ratio:
        return RedFlagDetection(found=False, area_ratio=area_ratio), mask
    x = int(stats[component, cv2.CC_STAT_LEFT])
    y = int(stats[component, cv2.CC_STAT_TOP])
    box_width = int(stats[component, cv2.CC_STAT_WIDTH])
    box_height = int(stats[component, cv2.CC_STAT_HEIGHT])
    center_x, center_y = centroids[component]
    return (
        RedFlagDetection(
            found=True,
            area_ratio=float(area_ratio),
            center_u_norm=float(center_x) / float(width),
            center_v_norm=float(center_y) / float(height),
            bbox_xywh=(x, y, box_width, box_height),
        ),
        mask,
    )


class RedFlagWaveTracker:
    def __init__(self, config: RedFlagSignalConfig) -> None:
        self.config = config
        self._history: deque[tuple[float, RedFlagDetection]] = deque()
        self.triggered = False
        self.last_metrics: dict[str, float | int | bool] = {}

    @staticmethod
    def _direction_reversals(values: list[float], minimum_step: float) -> int:
        signs: list[int] = []
        for first, second in zip(values, values[1:]):
            delta = second - first
            if abs(delta) < minimum_step:
                continue
            sign = 1 if delta > 0.0 else -1
            if not signs or signs[-1] != sign:
                signs.append(sign)
        return max(0, len(signs) - 1)

    def update(self, timestamp_s: float, detection: RedFlagDetection) -> bool:
        if self.triggered:
            return True
        self._history.append((float(timestamp_s), detection))
        cutoff = float(timestamp_s) - self.config.motion_window_s
        while self._history and self._history[0][0] < cutoff:
            self._history.popleft()

        valid = [item for _stamp, item in self._history if item.found]
        if valid:
            xs = [item.center_u_norm for item in valid]
            ys = [item.center_v_norm for item in valid]
            x_range = max(xs) - min(xs)
            y_range = max(ys) - min(ys)
            path_length = sum(
                math.hypot(x1 - x0, y1 - y0)
                for x0, y0, x1, y1 in zip(xs, ys, xs[1:], ys[1:])
            )
            reversals = max(
                self._direction_reversals(
                    xs,
                    self.config.min_direction_step_norm,
                ),
                self._direction_reversals(
                    ys,
                    self.config.min_direction_step_norm,
                ),
            )
            peak_area = max(item.area_ratio for item in valid)
        else:
            x_range = y_range = path_length = peak_area = 0.0
            reversals = 0

        self.last_metrics = {
            "valid_detections": len(valid),
            "x_range_norm": float(x_range),
            "y_range_norm": float(y_range),
            "path_length_norm": float(path_length),
            "direction_reversals": int(reversals),
            "peak_area_ratio": float(peak_area),
        }
        self.triggered = bool(
            len(valid) >= self.config.min_valid_detections
            and peak_area >= self.config.peak_area_ratio
            and max(x_range, y_range) >= self.config.min_axis_range_norm
            and path_length >= self.config.min_path_length_norm
            and reversals >= self.config.min_direction_reversals
        )
        self.last_metrics["triggered"] = self.triggered
        return self.triggered

