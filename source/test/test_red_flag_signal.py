import numpy as np

from src.perception.red_flag_signal import (
    RedFlagDetection,
    RedFlagSignalConfig,
    RedFlagWaveTracker,
    detect_red_flag,
)


def test_red_flag_color_detection_rejects_empty_background():
    image = np.full((240, 320, 3), 160, dtype=np.uint8)

    detection, _mask = detect_red_flag(image, RedFlagSignalConfig())

    assert detection.found is False


def test_static_red_flag_does_not_trigger_start():
    tracker = RedFlagWaveTracker(RedFlagSignalConfig())

    for index in range(12):
        triggered = tracker.update(
            index * 0.25,
            RedFlagDetection(
                found=True,
                area_ratio=0.11,
                center_u_norm=0.54 + (0.002 if index % 2 else 0.0),
                center_v_norm=0.46,
            ),
        )

    assert triggered is False


def test_waving_red_flag_triggers_after_large_reversing_motion():
    tracker = RedFlagWaveTracker(RedFlagSignalConfig())
    centers = [(0.36, 0.10), (0.50, 0.22), (0.42, 0.25), (0.80, 0.72)]

    results = [
        tracker.update(
            index * 0.35,
            RedFlagDetection(
                found=True,
                area_ratio=0.10,
                center_u_norm=center_u,
                center_v_norm=center_v,
            ),
        )
        for index, (center_u, center_v) in enumerate(centers)
    ]

    assert results == [False, False, False, True]
    assert tracker.last_metrics["direction_reversals"] >= 1

