from pathlib import Path

import numpy as np
import pytest

from src.perception.placement_uv_map import (
    fit_placement_uv_map,
    load_samples,
    write_mapping,
    load_mapping,
    load_mapping_for_item,
)


SAMPLES = (
    Path(__file__).resolve().parents[1]
    / "config"
    / "calibration"
    / "placement_uv_xy"
    / "orange_bottle"
    / "samples.json"
)


def test_orange_bottle_uv_map_fits_within_box_opening(tmp_path):
    mapping = load_mapping(SAMPLES.parent / "mapping.yaml")
    assert mapping.sample_count == 10
    assert mapping.fit_rms_xy_mm < 10.0
    # Taught near-center from the 2026-08-19 recapture.
    x_mm, y_mm = mapping.predict_xy_mm(373.0, 212.0)
    assert abs(x_mm - 0.1) < 10.0
    assert abs(y_mm - (-283.3)) < 12.0
    path = tmp_path / "mapping.yaml"
    write_mapping(path, mapping)
    loaded = load_mapping(path)
    again = loaded.predict_xy_mm(373.0, 212.0)
    assert again == (x_mm, y_mm)
    assert loaded.in_domain(373.0, 212.0)
    assert not loaded.in_domain(20.0, 20.0)
    assert loaded.align_u_px == 373.0
    assert loaded.align_v_px == 212.0
    assert loaded.alignment_u_norm(640.0) == pytest.approx(373.0 / 640.0)
    # Wrist must come from the taught near-center sample, not the old catalog TCP.
    assert loaded.rpy_deg == pytest.approx((-177.392, 81.991, 92.767), abs=0.05)
    poses = loaded.poses_mm_deg(374.5, 143.5)
    assert poses["release"][3:] == pytest.approx((-177.392, 81.991, 92.767), abs=0.05)
    assert poses["release"][3:] != pytest.approx((85.074, 87.787, -179.697), abs=1.0)
    far = loaded.poses_mm_deg(351.0, 89.0)
    assert far["release"][3:] == pytest.approx((-177.392, 81.991, 92.767), abs=0.05)
    assert far["release"][3:] != pytest.approx((90.0, 90.0, 26.565), abs=1.0)


def test_green_bottle_uses_its_own_taught_map():
    orange = load_mapping_for_item("orange_bottle")
    green = load_mapping_for_item("green_bottle")
    assert orange is not None
    assert green is not None
    assert green.item_id == "green_bottle"
    assert green.sample_count == 10
    assert green.fit_rms_xy_mm < 14.0
    assert green.align_u_px == 385.0
    assert green.align_v_px == 210.0
    assert green.align_u_px != orange.align_u_px


def test_dark_bottle_uses_its_own_taught_map():
    orange = load_mapping_for_item("orange_bottle")
    dark = load_mapping_for_item("dark_bottle")
    assert orange is not None
    assert dark is not None
    assert dark.item_id == "dark_bottle"
    assert dark.sample_count == 10
    assert dark.fit_rms_xy_mm < 12.0
    assert dark.align_u_px == 379.0
    assert dark.align_v_px == 63.0
    assert dark.rpy_deg == pytest.approx((-106.032, 87.262, 163.925), abs=0.05)
    assert dark.align_u_px != orange.align_u_px
    x_mm, y_mm = dark.predict_xy_mm(379.0, 63.0)
    assert abs(x_mm - (-5.87)) < 8.0
    assert abs(y_mm - (-345.492)) < 12.0


def test_other_blocks_reuse_the_red_taught_map():
    red = load_mapping_for_item("red_block")
    blue = load_mapping_for_item("blue_block")
    assert red is not None
    assert blue is not None
    assert blue.align_u_px == red.align_u_px
    assert blue.rpy_deg == red.rpy_deg


def test_yellow_block_uses_its_own_taught_map():
    red = load_mapping_for_item("red_block")
    yellow = load_mapping_for_item("yellow_block")
    assert red is not None
    assert yellow is not None
    assert yellow.item_id == "yellow_block"
    assert yellow.sample_count == 10
    assert yellow.fit_rms_xy_mm < 6.0
    assert yellow.align_u_px == 351.0
    assert yellow.align_v_px == 210.0
    assert yellow.align_u_px != red.align_u_px
    x_mm, y_mm = yellow.predict_xy_mm(351.0, 210.0)
    assert abs(x_mm - 3.8) < 8.0
    assert abs(y_mm - (-275.6)) < 12.0


def test_uv_map_poses_keep_vertical_descent():
    mapping = fit_placement_uv_map(load_samples(SAMPLES))
    poses = mapping.poses_mm_deg(400.0, 130.0)
    assert poses["approach"][0] == poses["release"][0]
    assert poses["approach"][1] == poses["release"][1]
    assert poses["approach"][2] > poses["release"][2] + 50.0
    assert poses["approach"][3:] == poses["release"][3:]
