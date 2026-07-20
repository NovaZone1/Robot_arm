from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.models import GraspCandidate, GraspExecutionConfig, PerceptionResult
from src.grasping.planning import PureGraspPlanner
from src.robot.types import EndPoseMMDeg
from src.utils.transforms import rotation_matrix_from_rpy_deg


def _make_perception(grasp_group) -> PerceptionResult:
    return PerceptionResult(
        color_bgr=np.zeros((1, 1, 3), dtype=np.uint8),
        depth_meters=np.zeros((1, 1), dtype=np.float32),
        segmentation={},
        scene_points=np.zeros((1, 3), dtype=np.float32),
        pointclouds=[],
        grasp_groups=[grasp_group],
        scene_grasp_count=len(grasp_group),
        scene_point_count=1,
        object_point_counts=[1],
        object_centers_camera_m=[(0.0, 0.0, 0.5)],
        object_centers_uv=[(0, 0)],
    )


def _make_grasp(*, score: float = 0.9):
    return SimpleNamespace(
        score=score,
        width=0.08,
        depth=0.04,
        translation=(0.0, 0.0, 0.5),
        rotation_matrix=np.eye(3, dtype=np.float64),
    )


def test_collect_grasp_candidates_rejects_plan_points_below_hard_pose_floor(monkeypatch):
    config = GraspExecutionConfig(
        hand_eye_config_path="dummy.yaml",
        max_approach_angle_deg=120.0,
        workspace_z_limits_m=(-0.50, 0.60),
        table_z_m=-0.17,
        min_gripper_table_clearance_m=0.03,
    )
    planner = PureGraspPlanner(config, np.eye(4, dtype=np.float64))
    tcp_pose = EndPoseMMDeg(30.0, 0.0, 400.0, 180.0, 60.0, 180.0)
    perception = _make_perception([_make_grasp()])

    monkeypatch.setattr(planner, "approach_angle_to_vertical_deg", lambda *_args: 10.0)
    monkeypatch.setattr(
        planner,
        "build_plan_data",
        lambda *_args, **_kwargs: {
            "within_workspace": True,
            "workspace_violations": [],
            "chosen_rot_error_deg": 15.0,
            "target_base_m": (0.30, 0.02, 0.060),
            "pregrasp_base_m": (0.31, 0.02, -0.101),
            "grasp_base_m": (0.30, 0.02, 0.055),
            "retreat_base_m": (0.30, 0.02, 0.160),
        },
    )

    candidates, diagnostics, _max_angle = planner.collect_grasp_candidates(
        perception,
        tcp_pose,
        np.eye(4, dtype=np.float64),
    )

    assert candidates == []
    assert len(diagnostics) == 1
    assert "filtered_by_workspace=1" in diagnostics[0]
    assert "pose_floor_examples=" in diagnostics[0]
    assert "pregrasp z=-0.101 below minimum pose z -0.100" in diagnostics[0]


def test_build_plan_data_retreat_lifts_from_grasp_pose():
    config = GraspExecutionConfig(
        hand_eye_config_path="dummy.yaml",
        tool_contact_offset_tool_m=(0.0, 0.0, 0.10),
        grasp_y_bias_mm=0.0,
        descend_offset_m=0.01,
        retreat_offset_m=0.10,
    )
    planner = PureGraspPlanner(config, np.eye(4, dtype=np.float64))
    candidate = GraspCandidate(
        instance_index=0,
        score=0.9,
        width_m=0.08,
        depth_m=0.04,
        translation_camera_m=(0.2, 0.0, 0.3),
        rotation_camera=np.eye(3, dtype=np.float64),
        object_center_camera_m=None,
        center_offset_m=None,
        raw_grasp=None,
    )
    tcp_pose = EndPoseMMDeg(30.0, 0.0, 400.0, 180.0, 60.0, 180.0)

    data = planner.build_plan_data(
        candidate,
        tcp_pose,
        np.eye(4, dtype=np.float64),
    )

    assert np.allclose(data["grasp_base_m"], (0.09, 0.0, 0.3))
    assert np.allclose(data["retreat_base_m"], (0.09, 0.0, 0.4))


def test_collect_grasp_candidates_rejects_rotation_delta_above_limit(monkeypatch):
    config = GraspExecutionConfig(
        hand_eye_config_path="dummy.yaml",
        max_approach_angle_deg=120.0,
        max_reachable_rotation_delta_deg=120.0,
        workspace_z_limits_m=(-0.50, 0.60),
        table_z_m=-0.17,
        min_gripper_table_clearance_m=0.03,
    )
    planner = PureGraspPlanner(config, np.eye(4, dtype=np.float64))
    tcp_pose = EndPoseMMDeg(30.0, 0.0, 400.0, 180.0, 60.0, 180.0)
    perception = _make_perception([_make_grasp()])

    monkeypatch.setattr(planner, "approach_angle_to_vertical_deg", lambda *_args: 10.0)
    monkeypatch.setattr(
        planner,
        "build_plan_data",
        lambda *_args, **_kwargs: {
            "within_workspace": True,
            "workspace_violations": [],
            "chosen_rot_error_deg": 153.8,
            "target_base_m": (0.30, 0.02, 0.20),
            "pregrasp_base_m": (0.31, 0.02, 0.28),
            "grasp_base_m": (0.30, 0.02, 0.22),
            "retreat_base_m": (0.30, 0.02, 0.32),
        },
    )

    candidates, diagnostics, _max_angle = planner.collect_grasp_candidates(
        perception,
        tcp_pose,
        np.eye(4, dtype=np.float64),
    )

    assert candidates == []
    assert len(diagnostics) == 1
    assert "filtered_by_rotation_count=1" in diagnostics[0]
    assert "rotation_threshold_deg=120.0" in diagnostics[0]
    assert "rejected_rotation_examples_deg=[153.8]" in diagnostics[0]


def test_collect_grasp_candidates_accepts_equivalent_wrist_roll_variant():
    config = GraspExecutionConfig(
        hand_eye_config_path="dummy.yaml",
        max_approach_angle_deg=180.0,
        max_reachable_rotation_delta_deg=45.0,
        workspace_z_limits_m=(-0.50, 0.60),
        table_z_m=-0.17,
        min_gripper_table_clearance_m=0.03,
        grasp_y_bias_mm=0.0,
        max_grasp_center_offset_m=0.50,
    )
    planner = PureGraspPlanner(config, np.eye(4, dtype=np.float64))
    tcp_pose = EndPoseMMDeg(300.0, 0.0, 300.0, 0.0, 0.0, 0.0)

    adjusted_rotation = rotation_matrix_from_rpy_deg(0.0, 0.0, 90.0)
    raw_rotation = adjusted_rotation @ PureGraspPlanner._R_ADJUST.T
    grasp = SimpleNamespace(
        score=0.9,
        width=0.08,
        depth=0.04,
        translation=(0.30, 0.0, 0.30),
        rotation_matrix=raw_rotation,
    )

    candidates, diagnostics, _max_angle = planner.collect_grasp_candidates(
        _make_perception([grasp]),
        tcp_pose,
        np.eye(4, dtype=np.float64),
    )

    assert len(candidates) == 1
    _candidate, _approach_angle, rotation_delta = candidates[0]
    assert rotation_delta < 1.0
    assert "filtered_by_rotation_count=0" in diagnostics[0]
