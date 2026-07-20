from src.grasping.models import GraspCandidate, GraspPlan
from src.robot.plan_validation import (
    select_first_reachable_candidate,
    validate_grasp_plan_waypoints,
    validate_grasp_plan_waypoints_detailed,
)
from src.robot.types import EndPoseMMDeg


def _make_candidate(instance_index: int, score: float) -> GraspCandidate:
    return GraspCandidate(
        instance_index=instance_index,
        score=score,
        width_m=0.08,
        depth_m=0.04,
        translation_camera_m=(0.0, 0.0, 0.5),
        rotation_camera=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        object_center_camera_m=(0.0, 0.0, 0.5),
        center_offset_m=0.01,
        raw_grasp=None,
    )


def _make_plan(candidate: GraspCandidate, *, x_offset_mm: float = 0.0) -> GraspPlan:
    return GraspPlan(
        candidate=candidate,
        target_base_m=(0.50, 0.02, 0.20),
        target_rpy_deg=(180.0, 30.0, 180.0),
        pregrasp_base_m=((500.0 + x_offset_mm) / 1000.0, 0.02, 0.24),
        grasp_base_m=((490.0 + x_offset_mm) / 1000.0, 0.02, 0.18),
        retreat_base_m=((480.0 + x_offset_mm) / 1000.0, 0.02, 0.28),
        within_workspace=True,
        workspace_violations=[],
    )


def test_validate_grasp_plan_waypoints_checks_expected_order():
    candidate = _make_candidate(instance_index=0, score=0.9)
    plan = _make_plan(candidate)
    calls: list[EndPoseMMDeg] = []

    validated = validate_grasp_plan_waypoints(
        plan,
        include_pregrasp=True,
        compute_ik=lambda pose: calls.append(pose),
    )

    assert validated == ["pregrasp", "grasp", "target", "retreat"]
    assert [(round(pose.x_mm), round(pose.z_mm)) for pose in calls] == [
        (500, 240),
        (490, 180),
        (500, 200),
        (480, 280),
    ]


def test_select_first_reachable_candidate_skips_unreachable_top_candidate():
    first = _make_candidate(instance_index=0, score=0.95)
    second = _make_candidate(instance_index=1, score=0.80)
    plans = {
        0: _make_plan(first, x_offset_mm=0.0),
        1: _make_plan(second, x_offset_mm=20.0),
    }

    candidate, plan, diagnostics, validation_records = select_first_reachable_candidate(
        [first, second],
        build_plan=lambda item: plans[item.instance_index],
        validate_plan=lambda item, built_plan: (
            {
                "robot_validation_result": "rejected_by_robot_validation",
                "robot_validation_stage": "grasp",
                "ik_error_type": "timeout",
                "ik_error_message": "MoveIt IK request timed out: /compute_ik",
                "waypoint_results": [
                    {"stage": "pregrasp", "status": "ok"},
                    {
                        "stage": "grasp",
                        "status": "failed",
                        "ik_error_type": "timeout",
                        "ik_error_message": "MoveIt IK request timed out: /compute_ik",
                    },
                ],
            }
            if item.instance_index == 0
            else {
                "robot_validation_result": "accepted",
                "robot_validation_stage": None,
                "ik_error_type": None,
                "ik_error_message": None,
                "waypoint_results": [
                    {"stage": "pregrasp", "status": "ok"},
                    {"stage": "grasp", "status": "ok"},
                    {"stage": "target", "status": "ok"},
                    {"stage": "retreat", "status": "ok"},
                ],
            }
        ),
    )

    assert candidate is second
    assert plan is plans[1]
    assert diagnostics == ["candidate[0] score=0.9500 rejected by robot validation: MoveIt IK request timed out: /compute_ik"]
    assert [record["selection_result"] for record in validation_records] == [
        "rejected_by_robot_validation",
        "selected_for_execution",
    ]
    assert validation_records[0]["robot_validation_stage"] == "grasp"
    assert validation_records[0]["ik_error_type"] == "timeout"
    assert validation_records[1]["waypoint_results"][-1]["stage"] == "retreat"


def test_validate_grasp_plan_waypoints_detailed_records_timeout_stage():
    candidate = _make_candidate(instance_index=0, score=0.9)
    plan = _make_plan(candidate)
    calls: list[str] = []

    def fake_compute_ik(pose: EndPoseMMDeg):
        if round(pose.z_mm) == 240:
            call_name = "pregrasp"
        elif round(pose.z_mm) == 180:
            call_name = "grasp"
        else:
            call_name = "target"
        calls.append(call_name)
        if call_name == "target":
            raise TimeoutError("MoveIt IK request timed out: /compute_ik")

    results = validate_grasp_plan_waypoints_detailed(
        plan,
        include_pregrasp=True,
        compute_ik=fake_compute_ik,
    )

    assert calls == ["pregrasp", "grasp", "target"]
    assert results == [
        {"stage": "pregrasp", "status": "ok"},
        {"stage": "grasp", "status": "ok"},
        {
            "stage": "target",
            "status": "failed",
            "ik_error_type": "timeout",
            "ik_error_message": "MoveIt IK request timed out: /compute_ik",
        },
    ]


def test_validate_grasp_plan_waypoints_detailed_records_no_ik_solution_stage():
    candidate = _make_candidate(instance_index=0, score=0.9)
    plan = _make_plan(candidate)
    calls: list[tuple[int, int]] = []

    def fake_compute_ik(pose: EndPoseMMDeg):
        calls.append((round(pose.x_mm), round(pose.z_mm)))
        if round(pose.x_mm) == 500 and round(pose.z_mm) == 200:
            raise RuntimeError("MoveIt IK failed: code=-31")

    results = validate_grasp_plan_waypoints_detailed(
        plan,
        include_pregrasp=False,
        compute_ik=fake_compute_ik,
    )

    assert calls == [
        (490, 180),
        (500, 200),
    ]
    assert results == [
        {"stage": "grasp", "status": "ok"},
        {
            "stage": "target",
            "status": "failed",
            "ik_error_type": "no_ik_solution",
            "ik_error_message": "MoveIt IK failed: code=-31",
        },
    ]
