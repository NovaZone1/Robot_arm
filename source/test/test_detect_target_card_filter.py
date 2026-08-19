import numpy as np

from src.perception.external_inference_worker import ExternalInferenceEngine


class _FakeSegmenter:
    def __init__(self, masks, scores):
        self._masks = masks
        self._scores = scores

    def segment_text(self, _image, _prompt):
        return {
            "masks": self._masks,
            "scores": self._scores,
            "backend": "fake",
        }


def test_detect_target_2d_skips_the_printed_card_and_keeps_the_real_object():
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    card_mask = np.zeros((480, 640), dtype=bool)
    object_mask = np.zeros((480, 640), dtype=bool)
    card_mask[80:180, 280:380] = True
    object_mask[300:400, 280:380] = True
    engine = ExternalInferenceEngine()
    engine._segmenter = _FakeSegmenter(
        [card_mask, object_mask],
        [0.95, 0.70],
    )

    result = engine.detect_target_2d(
        image,
        "red block",
        exclude_roi_norm=(0.0, 0.0, 1.0, 0.50),
        search_roi_norm=(0.0, 0.50, 1.0, 1.0),
    )

    assert result["found"] is True
    assert result["center_v_norm"] > 0.50
    assert result["confidence"] == 0.70
