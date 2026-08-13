from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.perception.item_catalog import (
    ItemCatalog,
    LabelDetection,
    ReferenceLabelMatcher,
    bottle_identity_score,
    bottle_item_id_from_prompt,
    classify_bottle_identity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = PROJECT_ROOT / "config" / "item_catalog.yaml"


def _foreground_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return (hsv[:, :, 1] > 24) | (hsv[:, :, 2] < 238)


def test_catalog_resolves_all_six_items():
    catalog = ItemCatalog.load(CATALOG_PATH)
    assert len(catalog.items) == 6
    assert catalog.require("红色物块").item_id == "red_block"
    assert catalog.require("green bottle").item_id == "green_bottle"
    assert catalog.box.outer_size_m == (0.18, 0.132, 0.087)


def test_bottle_prompt_identity_mapping():
    assert bottle_item_id_from_prompt("orange bottle") == "orange_bottle"
    assert bottle_item_id_from_prompt("深色饮料瓶") == "dark_bottle"
    assert bottle_item_id_from_prompt("green bottle") == "green_bottle"
    assert bottle_item_id_from_prompt("bottle") is None


def test_place_pose_builder_is_locked_until_calibrated():
    catalog = ItemCatalog.load(CATALOG_PATH)

    with pytest.raises(RuntimeError, match="placement is not calibrated"):
        catalog.build_place_poses_mm_deg("red_block", 0)


def test_all_items_share_the_base_aligned_place_calibration():
    catalog = ItemCatalog.load(CATALOG_PATH)

    for item_id in (
        "yellow_block",
        "red_block",
        "blue_block",
        "orange_bottle",
        "dark_bottle",
        "green_bottle",
    ):
        poses = catalog.build_base_aligned_place_poses_mm_deg(item_id)
        assert poses["approach"] == pytest.approx(
            (-22.262, 283.822, 491.100, 85.074, 87.787, 179.697)
        )
        assert poses["release"] == pytest.approx(
            (-22.262, 283.822, 307.657, 85.074, 87.787, 179.697)
        )
        assert poses["retreat"] == pytest.approx(poses["approach"])


def test_place_pose_builder_adds_vertical_approach_and_retreat():
    catalog = ItemCatalog.load(CATALOG_PATH)
    item = catalog.items["red_block"]
    catalog.items["red_block"] = replace(
        item,
        placement=replace(
            item.placement,
            enabled=True,
            release_offset_mm=(1.0, 2.0, 3.0),
            release_rpy_deg=(180.0, 0.0, 0.0),
        ),
    )
    catalog.box = replace(
        catalog.box,
        slot_centers_mm=(
            (300.0, 50.0, 277.0),
            (300.0, 100.0, 277.0),
            (300.0, 150.0, 277.0),
            (300.0, 200.0, 277.0),
            (300.0, 250.0, 277.0),
            (300.0, 300.0, 277.0),
        ),
    )

    poses = catalog.build_place_poses_mm_deg("red_block", 0)

    assert poses["release"] == (301.0, 52.0, 280.0, 180.0, 0.0, 0.0)
    assert poses["approach"][2] == 400.0
    assert poses["retreat"][2] == 420.0


def test_tightly_packed_row_interpolates_180mm_slots():
    catalog = ItemCatalog.load(CATALOG_PATH)
    catalog.box = replace(
        catalog.box,
        row_first_slot_center_mm=(100.0, 200.0, 250.0),
        row_last_slot_center_mm=(1000.0, 200.0, 250.0),
    )

    centers = catalog.resolved_slot_centers_mm()

    assert len(centers) == 6
    assert centers[0] == (100.0, 200.0, 250.0)
    assert centers[3] == (640.0, 200.0, 250.0)
    assert centers[5] == (1000.0, 200.0, 250.0)


def test_reference_bottle_colors_are_separable():
    catalog = ItemCatalog.load(CATALOG_PATH)
    for expected in ("orange_bottle", "dark_bottle", "green_bottle"):
        image = cv2.imread(str(catalog.items[expected].reference_image), cv2.IMREAD_COLOR)
        mask = _foreground_mask(image)
        scores = {
            item_id: bottle_identity_score(image, mask, item_id)
            for item_id in ("orange_bottle", "dark_bottle", "green_bottle")
        }
        assert max(scores, key=scores.get) == expected
        assert scores[expected] >= 0.45


def test_chromatic_bottle_identity_wins_over_dark_shadow_score():
    scene = np.full((180, 240, 3), 220, dtype=np.uint8)
    scene[30:150, 20:70] = (0, 80, 150)
    scene[30:150, 95:145] = (25, 25, 25)
    scene[30:150, 170:220] = (45, 105, 45)

    expected_by_bounds = (
        ("orange_bottle", (20, 70)),
        ("dark_bottle", (95, 145)),
        ("green_bottle", (170, 220)),
    )
    for expected, (x0, x1) in expected_by_bounds:
        mask = np.zeros(scene.shape[:2], dtype=bool)
        mask[30:150, x0:x1] = True
        predicted, score, _ = classify_bottle_identity(scene, mask)
        assert predicted == expected
        assert score >= 0.18


def test_template_candidates_require_bright_neutral_label_card():
    bbox = (70, 70, 60, 60)
    on_white_card = np.full((200, 200, 3), 210, dtype=np.uint8)
    on_white_card[70:130, 70:130] = (150, 60, 20)
    on_dark_panel = np.full((200, 200, 3), 55, dtype=np.uint8)
    on_dark_panel[70:130, 70:130] = (150, 60, 20)

    assert ReferenceLabelMatcher._white_card_support(on_white_card, bbox) > 0.90
    assert ReferenceLabelMatcher._white_card_support(on_dark_panel, bbox) < 0.10


def test_color_block_rejects_label_clipped_by_image_edge():
    catalog = ItemCatalog.load(CATALOG_PATH)
    matcher = ReferenceLabelMatcher(catalog)
    scene = np.full((480, 640, 3), 210, dtype=np.uint8)
    scene[190:272, 550:640] = (15, 15, 185)
    scene[217:257, 304:342] = (15, 15, 185)

    detections = matcher._match_color_markers(
        scene,
        (0, 0, 640, 480),
        threshold=0.42,
        include_bottles=False,
    )
    assert all(
        detection.bbox_xywh[0] < 540 for detection in detections
    )


def test_block_template_requires_matching_dominant_hue():
    blue = np.full((100, 100, 3), (150, 55, 25), dtype=np.uint8)

    blue_support, colored_fraction = ReferenceLabelMatcher._block_color_support(
        blue,
        (0, 0, 100, 100),
        "blue_block",
    )
    red_support, _ = ReferenceLabelMatcher._block_color_support(
        blue,
        (0, 0, 100, 100),
        "red_block",
    )

    assert blue_support > 0.95
    assert colored_fraction > 0.95
    assert red_support == 0.0


def test_color_block_rejects_tall_blue_box_hardware_region():
    catalog = ItemCatalog.load(CATALOG_PATH)
    matcher = ReferenceLabelMatcher(catalog)
    scene = np.full((480, 640, 3), 210, dtype=np.uint8)
    # Mirrors the latest false positive: a 97x146 dark-blue vertical region
    # backed by bright transparent-box reflections.
    scene[178:324, 314:411] = (160, 55, 25)

    detections = matcher._match_color_markers(
        scene,
        (0, 0, 640, 480),
        threshold=0.42,
        include_bottles=False,
    )

    assert all(
        detection.item_id != "blue_block" for detection in detections
    )


def test_color_block_accepts_large_partially_occluded_red_label():
    catalog = ItemCatalog.load(CATALOG_PATH)
    matcher = ReferenceLabelMatcher(catalog)
    scene = np.full((480, 640, 3), 210, dtype=np.uint8)
    scene[205:246, 496:593] = (15, 15, 185)

    detections = matcher._match_color_markers(
        scene,
        (0, 0, 640, 480),
        threshold=0.42,
        include_bottles=False,
    )
    red = next(
        detection
        for detection in detections
        if detection.item_id == "red_block"
    )

    assert red.method == "color_shape_partial"
    assert red.bbox_xywh == (496, 205, 97, 41)


def test_label_projection_can_use_shared_row_plane_depth():
    catalog = ItemCatalog.load(CATALOG_PATH)
    matcher = ReferenceLabelMatcher(catalog)
    detection = LabelDetection(
        item_id="yellow_block",
        confidence=0.9,
        bbox_xywh=(300, 200, 80, 80),
    )
    depth = np.full((480, 640), 1.5, dtype=np.float32)

    point = matcher.project_label_centers(
        depth_meters=depth,
        camera_k=(
            600.0,
            0.0,
            320.0,
            0.0,
            600.0,
            240.0,
            0.0,
            0.0,
            1.0,
        ),
        base_to_camera=np.eye(4),
        detections=(detection,),
        depth_override_m=0.60,
    )[0]

    assert point == pytest.approx((0.02, 0.0, 0.60))


def _six_label_scene(matcher, order):
    scene = np.full((360, 760, 3), 255, dtype=np.uint8)
    for slot_index, item_id in enumerate(order):
        reference = matcher._reference(matcher.catalog.items[item_id])
        target_width = 64
        target_height = int(round(target_width * reference.shape[0] / reference.shape[1]))
        scaled = cv2.resize(reference, (target_width, target_height), interpolation=cv2.INTER_AREA)
        x0 = 38 + (slot_index * 118)
        y0 = 72
        scene[y0 : y0 + target_height, x0 : x0 + target_width] = scaled
    return scene


def test_reference_label_matcher_maps_shuffled_row_to_left_to_right_slots():
    catalog = ItemCatalog.load(CATALOG_PATH)
    matcher = ReferenceLabelMatcher(catalog)
    order = (
        "blue_block",
        "orange_bottle",
        "red_block",
        "green_bottle",
        "yellow_block",
        "dark_bottle",
    )
    scene = _six_label_scene(matcher, order)

    result = matcher.match_expected(scene, "green_bottle", threshold=0.34)

    assert result.accepted
    assert result.matched_item_id == "green_bottle"
    assert result.slot_index == 3
    assert result.detected_item_ids == order
    assert result.bbox_xywh is not None


def test_reference_label_matcher_rejects_incomplete_label_row():
    catalog = ItemCatalog.load(CATALOG_PATH)
    matcher = ReferenceLabelMatcher(catalog)
    reference = matcher._reference(catalog.items["red_block"])
    scaled = cv2.resize(reference, (120, 120), interpolation=cv2.INTER_AREA)
    scene = np.full((480, 640, 3), 255, dtype=np.uint8)
    scene[150:270, 260:380] = scaled

    result = matcher.match_expected(scene, "red_block", threshold=0.40)

    assert not result.accepted
    assert result.matched_item_id == "red_block"
    assert len(result.detected_item_ids) < 6


def test_yolo_bottle_shape_proposals_are_classified_by_liquid_color():
    catalog = ItemCatalog.load(CATALOG_PATH)
    matcher = ReferenceLabelMatcher(catalog)
    scene = np.full((240, 600, 3), 220, dtype=np.uint8)
    scene[55:205, 40:90] = (25, 25, 25)
    scene[55:205, 180:230] = (0, 125, 245)
    scene[85:155, 340:410] = (180, 50, 20)
    scene[55:205, 490:540] = (45, 105, 45)
    proposals = (
        {"confidence": 0.90, "bbox_xywh": [40, 55, 50, 150]},
        {"confidence": 0.85, "bbox_xywh": [180, 55, 50, 150]},
        {"confidence": 0.88, "bbox_xywh": [490, 55, 50, 150]},
    )

    detections, _ = matcher.match_all(
        scene,
        threshold=0.40,
        bottle_proposals=proposals,
    )
    by_id = {detection.item_id: detection for detection in detections}

    assert by_id["dark_bottle"].method == "yolo_bottle+liquid_color"
    assert by_id["orange_bottle"].method == "yolo_bottle+liquid_color"
    assert by_id["green_bottle"].method == "yolo_bottle+liquid_color"
    assert by_id["blue_block"].method == "color_shape"


def test_depth_localization_uses_label_row_and_known_box_depth():
    catalog = ItemCatalog.load(CATALOG_PATH)
    matcher = ReferenceLabelMatcher(catalog)
    order = (
        "blue_block",
        "red_block",
        "yellow_block",
        "orange_bottle",
        "dark_bottle",
        "green_bottle",
    )
    detections = tuple(
        LabelDetection(
            item_id=item_id,
            confidence=0.9,
            bbox_xywh=(350 + (index * 108), 290, 20, 20),
        )
        for index, item_id in enumerate(order)
    )
    depth = np.ones((720, 1280), dtype=np.float32)

    result = matcher.localize_box_row(
        depth_meters=depth,
        camera_k=(600.0, 0.0, 640.0, 0.0, 600.0, 360.0, 0.0, 0.0, 1.0),
        base_to_camera=np.eye(4, dtype=np.float64),
        detections=detections,
        target_item_id="yellow_block",
        table_z_m=0.161,
    )

    assert result.slot_index == 2
    assert result.adjacent_pitch_mm == pytest.approx((180.0,) * 5)
    assert result.box_center_base_m[2] == pytest.approx(0.2045)
    assert result.interior_direction_base == pytest.approx((0.0, 1.0, 0.0))


def test_box_row_fit_corrects_one_bounded_depth_outlier():
    catalog = ItemCatalog.load(CATALOG_PATH)
    matcher = ReferenceLabelMatcher(catalog)
    order = tuple(catalog.items)
    detections = tuple(
        LabelDetection(
            item_id=item_id,
            confidence=0.9,
            bbox_xywh=(0, 0, 1, 1),
        )
        for item_id in order
    )
    raw_x = (0.0, 0.18, 0.36, 0.54, 0.72, 0.93)

    result = matcher.localize_box_row_from_points(
        detections=detections,
        label_centers_base_m=tuple((x, 0.60, 0.20) for x in raw_x),
        target_item_id=order[-1],
        table_z_m=0.161,
        camera_xy_base=(0.0, 0.0),
        image_right_direction_base_xy=(1.0, 0.0),
    )

    assert result.adjacent_pitch_mm == pytest.approx((180.0,) * 5)
    assert result.raw_adjacent_pitch_mm[-1] == pytest.approx(210.0)
    assert sum(abs(value) <= 20.0 for value in result.fit_residual_mm) == 5
    fitted_x = [point[0] for point in result.box_centers_base_m]
    assert np.diff(fitted_x) == pytest.approx((0.18,) * 5)


def test_box_row_fit_rejects_large_depth_outlier():
    catalog = ItemCatalog.load(CATALOG_PATH)
    matcher = ReferenceLabelMatcher(catalog)
    order = tuple(catalog.items)
    detections = tuple(
        LabelDetection(
            item_id=item_id,
            confidence=0.9,
            bbox_xywh=(0, 0, 1, 1),
        )
        for item_id in order
    )

    with pytest.raises(RuntimeError, match="fit_residual"):
        matcher.localize_box_row_from_points(
            detections=detections,
            label_centers_base_m=(
                (0.0, 0.60, 0.20),
                (0.18, 0.60, 0.20),
                (0.36, 0.60, 0.20),
                (0.54, 0.60, 0.20),
                (0.72, 0.60, 0.20),
                (1.00, 0.60, 0.20),
            ),
            target_item_id=order[-1],
            table_z_m=0.161,
            camera_xy_base=(0.0, 0.0),
            image_right_direction_base_xy=(1.0, 0.0),
        )
