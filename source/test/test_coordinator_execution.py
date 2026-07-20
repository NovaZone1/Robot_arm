from types import SimpleNamespace
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.coordinator import GraspPipelineCoordinator
from src.grasping.models import GraspCandidate, GraspPlan


def test_execute_grasp_plan_moves_to_target_before_closing(monkeypatch):
    coordinator = GraspPipelineCoordinator.__new__(GraspPipelineCoordinator)
    coordinator.config = SimpleNamespace(
        enable_pregrasp=True,
        handoff_pose_mm_deg=None,
        home_pose_mm_deg=None,
    )

    events: list[str] = []

    monkeypatch.setattr(coordinator, "_ensure_not_stopped", lambda: None)
    monkeypatch.setattr(coordinator, "_set_gripper_open", lambda: events.append("open_gripper"))
    monkeypatch.setattr(coordinator, "_set_gripper_closed", lambda: events.append("close_gripper"))

    def fake_move_to_pose(translation_m, _rpy_deg, **_kwargs):
        labels = {
            (0.1, 0.2, 0.35): "pregrasp",
            (0.1, 0.2, 0.3): "grasp",
            (0.1, 0.2, 0.28): "target",
            (0.1, 0.2, 0.4): "retreat",
        }
        events.append(labels[translation_m])
        return translation_m

    monkeypatch.setattr(coordinator, "_move_to_pose", fake_move_to_pose)

    candidate = GraspCandidate(
        instance_index=0,
        score=0.95,
        width_m=0.04,
        depth_m=0.02,
        translation_camera_m=(0.1, 0.2, 0.3),
        rotation_camera=np.eye(3),
        object_center_camera_m=None,
        center_offset_m=None,
        raw_grasp=None,
    )
    plan = GraspPlan(
        candidate=candidate,
        target_base_m=(0.1, 0.2, 0.28),
        target_rpy_deg=(1.0, 2.0, 3.0),
        pregrasp_base_m=(0.1, 0.2, 0.35),
        grasp_base_m=(0.1, 0.2, 0.3),
        retreat_base_m=(0.1, 0.2, 0.4),
        within_workspace=True,
        workspace_violations=[],
    )

    result = GraspPipelineCoordinator.execute_grasp_plan(coordinator, plan)

    assert events == [
        "open_gripper",
        "pregrasp",
        "grasp",
        "target",
        "close_gripper",
        "retreat",
        "open_gripper",
    ]
    assert result["target_pose"] == (0.1, 0.2, 0.28)
