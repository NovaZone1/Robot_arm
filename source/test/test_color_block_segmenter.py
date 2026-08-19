from pathlib import Path
import sys
from types import SimpleNamespace

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.perception.yolo_segmenter import YOLOSegmenter, _match_block_colors


def _synthetic_blocks() -> np.ndarray:
    image = np.full((240, 320, 3), 25, dtype=np.uint8)
    cv2.rectangle(image, (20, 30), (85, 100), (0, 0, 230), thickness=-1)
    cv2.rectangle(image, (115, 45), (185, 115), (0, 220, 220), thickness=-1)
    cv2.rectangle(image, (220, 60), (290, 130), (220, 30, 10), thickness=-1)
    return image


def test_match_block_colors_supports_english_and_chinese_prompts():
    assert _match_block_colors("red block") == ["red"]
    assert _match_block_colors("黄色方块") == ["yellow"]
    assert _match_block_colors("蓝色立方体") == ["blue"]
    assert _match_block_colors("color blocks") == ["red", "yellow", "blue"]
    assert _match_block_colors("bottle") is None


def test_color_block_prompt_segments_only_requested_color_without_loading_yolo():
    segmenter = YOLOSegmenter(device="cpu")

    result = segmenter.segment_text(_synthetic_blocks(), "red block")

    assert segmenter._model is None
    assert result["backend"] == "color_block"
    assert result["allow_scene_fallback"] is False
    assert result["labels"] == ["red block"]
    assert tuple(result["masks"].shape) == (1, 240, 320)
    assert int(result["masks"][0].sum()) > 3500


def test_generic_block_prompt_returns_separate_red_yellow_blue_instances():
    result = YOLOSegmenter(device="cpu").segment_text(_synthetic_blocks(), "物块")

    assert set(result["labels"]) == {"red block", "yellow block", "blue block"}
    assert tuple(result["masks"].shape) == (3, 240, 320)
    assert tuple(result["boxes"].shape) == (3, 4)


def test_missing_requested_color_disables_scene_fallback():
    result = YOLOSegmenter(device="cpu").segment_text(_synthetic_blocks(), "蓝色物块")
    empty_image = np.zeros((240, 320, 3), dtype=np.uint8)
    missing = YOLOSegmenter(device="cpu").segment_text(empty_image, "蓝色物块")

    assert result["labels"] == ["blue block"]
    assert tuple(missing["masks"].shape) == (0, 240, 320)
    assert missing["allow_scene_fallback"] is False


def test_blue_block_ignores_smaller_same_hue_bottle_cap():
    image = _synthetic_blocks()
    cv2.rectangle(image, (130, 170), (170, 200), (180, 0, 60), thickness=-1)

    result = YOLOSegmenter(device="cpu").segment_text(image, "blue block")

    assert result["labels"] == ["blue block"]
    assert tuple(result["masks"].shape) == (1, 240, 320)
    x1, y1, x2, y2 = result["boxes"][0].tolist()
    assert x1 >= 210
    assert y1 < 80
    assert x2 > 280
    assert y2 > 120


def test_red_block_rejects_small_wide_bottle_cap():
    image = np.full((240, 320, 3), 25, dtype=np.uint8)
    cv2.rectangle(image, (120, 80), (165, 103), (0, 0, 230), thickness=-1)

    result = YOLOSegmenter(device="cpu").segment_text(image, "red block")

    assert result["labels"] == []
    assert tuple(result["masks"].shape) == (0, 240, 320)
    assert result["allow_scene_fallback"] is False


def test_missing_catalog_bottle_disables_scene_fallback():
    segmenter = YOLOSegmenter(device="cpu")
    segmenter._model = lambda *_args, **_kwargs: [
        SimpleNamespace(boxes=None, masks=None)
    ]

    result = segmenter.segment_text(
        np.zeros((240, 320, 3), dtype=np.uint8),
        "green bottle",
    )

    assert tuple(result["masks"].shape) == (0, 240, 320)
    assert result["backend"] == "yolo+bottle_identity"
    assert result["requested_item_id"] == "green_bottle"
    assert result["allow_scene_fallback"] is False
