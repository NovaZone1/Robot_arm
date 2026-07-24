import json

import numpy as np
import pytest

from robot_grasp_ros2.pipeline_orchestrator_node import PipelineOrchestratorNode
from src.grasping.models import GraspCandidate, GraspPlan


def test_write_run_artifacts_writes_execution_trace_file(tmp_path):
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    artifact_root = tmp_path / "distributed_runs"
    run_id = "run-456"

    node._artifact_root_dir = lambda: artifact_root
    node._run_artifact_dir = lambda incoming_run_id: artifact_root / incoming_run_id

    result_payload = {
        "status": "ok",
        "prompt": "cup",
        "scene_id": "scene-1",
        "execution": {
            "status": "ok",
            "execution_trace": [
                {
                    "step_name": "grasp",
                    "command_type": "move_pose",
                    "success": True,
                }
            ],
        },
    }

    artifact_dir = PipelineOrchestratorNode._write_run_artifacts(
        node,
        run_id=run_id,
        request_payload={"run_id": run_id},
        cycle_records=[],
        result_payload=result_payload,
    )

    trace_payload = json.loads((artifact_dir / "execution_trace.json").read_text(encoding="utf-8"))
    final_payload = json.loads((artifact_dir / "final_result.json").read_text(encoding="utf-8"))

    assert trace_payload["run_id"] == run_id
    assert trace_payload["execution_trace"][0]["step_name"] == "grasp"
    assert "execution_trace" not in final_payload["execution"]


def test_write_run_artifacts_writes_candidate_validation_file(tmp_path):
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    artifact_root = tmp_path / "distributed_runs"
    run_id = "run-789"

    node._artifact_root_dir = lambda: artifact_root
    node._run_artifact_dir = lambda incoming_run_id: artifact_root / incoming_run_id

    result_payload = {
        "status": "ok",
        "prompt": "cup",
        "scene_id": "scene-2",
        "candidate_validation": [
            {
                "candidate_index": 0,
                "selection_result": "rejected_by_robot_validation",
                "robot_validation_stage": "grasp",
                "ik_error_type": "timeout",
            },
            {
                "candidate_index": 1,
                "selection_result": "selected_for_execution",
                "robot_validation_result": "accepted",
            },
        ],
    }

    artifact_dir = PipelineOrchestratorNode._write_run_artifacts(
        node,
        run_id=run_id,
        request_payload={"run_id": run_id},
        cycle_records=[],
        result_payload=result_payload,
    )

    validation_payload = json.loads((artifact_dir / "candidate_validation.json").read_text(encoding="utf-8"))
    final_payload = json.loads((artifact_dir / "final_result.json").read_text(encoding="utf-8"))

    assert validation_payload["run_id"] == run_id
    assert validation_payload["candidate_validation"][0]["robot_validation_stage"] == "grasp"
    assert validation_payload["candidate_validation"][1]["selection_result"] == "selected_for_execution"
    assert "candidate_validation" not in final_payload


def test_retarget_plan_uses_object_center_and_preserves_planner_contact_offset(monkeypatch):
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    parameters = {
        "use_object_center_contact": True,
        "object_center_contact_max_offset_m": 0.08,
        "table_z_m": 0.161,
        "min_gripper_table_clearance_m": 0.03,
    }
    monkeypatch.setattr(node, "get_parameter", lambda name: type("P", (), {"value": parameters[name]})())

    candidate = GraspCandidate(
        instance_index=0,
        score=0.9,
        width_m=0.04,
        depth_m=0.02,
        translation_camera_m=(0.10, 0.20, 0.30),
        rotation_camera=np.eye(3),
        object_center_camera_m=(0.14, 0.24, 0.34),
        center_offset_m=0.069,
        raw_grasp=None,
    )
    plan = GraspPlan(
        candidate=candidate,
        target_base_m=(0.1, 0.2, 0.3),
        target_rpy_deg=(0.0, 0.0, 0.0),
        pregrasp_base_m=(0.1, 0.2, 0.4),
        grasp_base_m=(0.1, 0.2, 0.35),
        retreat_base_m=(0.1, 0.2, 0.45),
        within_workspace=False,
        workspace_violations=["legacy grasp orientation below table"],
        target_contact_point_base_m=(0.10, 0.20, 0.35),
        tool_contact_offset_tool_m=(0.0, 0.0, 0.105),
    )

    retargeted = PipelineOrchestratorNode._retarget_plan_to_object_center(
        node,
        plan=plan,
        candidate=candidate,
        base_to_camera=np.eye(4),
    )

    assert np.allclose(
        retargeted.target_contact_point_base_m,
        (0.1235294118, 0.2117647059, 0.35),
    )
    assert retargeted.within_workspace is True
    assert retargeted.workspace_violations == []


def test_retarget_plan_rejects_unreliable_object_center(monkeypatch):
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    parameters = {
        "use_object_center_contact": True,
        "object_center_contact_max_offset_m": 0.08,
    }
    monkeypatch.setattr(node, "get_parameter", lambda name: type("P", (), {"value": parameters[name]})())
    candidate = GraspCandidate(
        instance_index=0,
        score=0.9,
        width_m=0.04,
        depth_m=0.02,
        translation_camera_m=(0.0, 0.0, 0.45),
        rotation_camera=np.eye(3),
        object_center_camera_m=(0.20, 0.0, 0.62),
        center_offset_m=0.17,
        raw_grasp=None,
    )
    plan = GraspPlan(
        candidate=candidate,
        target_base_m=(0.0, 0.4, 0.2),
        target_rpy_deg=(0.0, 0.0, 0.0),
        pregrasp_base_m=(0.0, 0.4, 0.3),
        grasp_base_m=(0.0, 0.4, 0.25),
        retreat_base_m=(0.0, 0.4, 0.35),
        within_workspace=True,
        workspace_violations=[],
        target_contact_point_base_m=(0.0, 0.4, 0.22),
        tool_contact_offset_tool_m=(0.0, 0.0, 0.105),
    )

    retargeted = PipelineOrchestratorNode._retarget_plan_to_object_center(
        node,
        plan=plan,
        candidate=candidate,
        base_to_camera=np.eye(4),
    )

    assert retargeted.within_workspace is False
    assert "object center offset" in retargeted.workspace_violations[0]
    assert "exceeds" in retargeted.workspace_violations[0]


def test_retarget_color_block_uses_known_table_relative_center_height(monkeypatch):
    node = PipelineOrchestratorNode.__new__(PipelineOrchestratorNode)
    parameters = {
        "use_object_center_contact": True,
        "object_center_contact_max_offset_m": 0.08,
        "table_z_m": 0.161,
        "min_gripper_table_clearance_m": 0.03,
        "color_block_center_height_m": 0.045,
        "manual_target_bias_z_mm": 0.0,
    }
    monkeypatch.setattr(node, "get_parameter", lambda name: type("P", (), {"value": parameters[name]})())
    candidate = GraspCandidate(
        instance_index=0,
        score=0.9,
        width_m=0.04,
        depth_m=0.02,
        translation_camera_m=(-0.06, -0.06, 0.536),
        rotation_camera=np.eye(3),
        object_center_camera_m=(-0.046, -0.016, 0.532),
        center_offset_m=0.044,
        raw_grasp=None,
    )
    plan = GraspPlan(
        candidate=candidate,
        target_base_m=(-0.1, 0.4, 0.2),
        target_rpy_deg=(0.0, 0.0, 0.0),
        pregrasp_base_m=(-0.1, 0.4, 0.3),
        grasp_base_m=(-0.1, 0.4, 0.25),
        retreat_base_m=(-0.1, 0.4, 0.35),
        within_workspace=False,
        workspace_violations=["depth drift put contact below table clearance"],
        target_contact_point_base_m=(-0.1, 0.4, 0.17),
        tool_contact_offset_tool_m=(0.0, 0.0, 0.105),
    )

    retargeted = PipelineOrchestratorNode._retarget_plan_to_object_center(
        node,
        plan=plan,
        candidate=candidate,
        base_to_camera=np.eye(4),
        prompt="red block",
    )

    assert retargeted.target_contact_point_base_m[2] == pytest.approx(0.206)
    assert retargeted.within_workspace is True
    assert retargeted.workspace_violations == []
