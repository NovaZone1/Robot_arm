"""Perception utilities for YOLOv8-seg, RealSense RGBD, and GraspNet."""

from __future__ import annotations

from importlib import import_module
from typing import Any


__all__ = [
    "GraspNetRunner",
    "RealSenseRGBDCamera",
    "YOLOSegmenter",
    "resolve_graspnet_checkpoint",
]


def __getattr__(name: str) -> Any:
    if name in {"GraspNetRunner", "resolve_graspnet_checkpoint"}:
        module = import_module(".graspnet_runner", __name__)
        return getattr(module, name)
    if name == "RealSenseRGBDCamera":
        module = import_module(".realsense_rgbd", __name__)
        return getattr(module, name)
    if name == "YOLOSegmenter":
        module = import_module(".yolo_segmenter", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
