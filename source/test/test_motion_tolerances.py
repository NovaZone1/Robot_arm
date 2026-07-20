from src.robot.motion_tolerances import (
    effective_position_tolerance_mm,
    pose_goal_reached,
    wait_until_pose_goal,
)
from src.robot.types import EndPoseMMDeg


def test_small_translation_tightens_position_tolerance():
    start = EndPoseMMDeg(50.67, -0.086, 171.186, -165.533, 72.558, -166.143)
    target = EndPoseMMDeg(50.67, -0.086, 174.186, -165.533, 72.558, -166.143)

    tolerance_mm = effective_position_tolerance_mm(
        start=start,
        target=target,
        default_tolerance_mm=8.0,
    )

    assert tolerance_mm < 3.0


def test_large_translation_keeps_default_position_tolerance():
    start = EndPoseMMDeg(50.67, -0.086, 171.186, -165.533, 72.558, -166.143)
    target = EndPoseMMDeg(50.67, -0.086, 221.186, -165.533, 72.558, -166.143)

    tolerance_mm = effective_position_tolerance_mm(
        start=start,
        target=target,
        default_tolerance_mm=8.0,
    )

    assert tolerance_mm == 8.0


def test_small_translation_can_keep_default_tolerance_when_tightening_disabled():
    start = EndPoseMMDeg(30.0, 0.0, 400.0, 180.0, 60.0, 180.0)
    target = EndPoseMMDeg(32.0, 0.0, 400.0, 180.0, 60.0, 180.0)

    tolerance_mm = effective_position_tolerance_mm(
        start=start,
        target=target,
        default_tolerance_mm=8.0,
        enable_tightening=False,
    )

    assert tolerance_mm == 8.0


def test_pose_goal_reached_requires_arrived_status_when_requested():
    assert not pose_goal_reached(
        dpos_mm=1.0,
        drot_deg=1.0,
        pos_tolerance_mm=5.0,
        rot_tolerance_deg=6.0,
        motion_status="NOT_ARRIVED",
        require_motion_arrived=True,
    )

    assert pose_goal_reached(
        dpos_mm=1.0,
        drot_deg=1.0,
        pos_tolerance_mm=5.0,
        rot_tolerance_deg=6.0,
        motion_status="ARRIVED",
        require_motion_arrived=True,
    )


def test_wait_until_pose_goal_republishes_until_arrived():
    target = EndPoseMMDeg(0.0, 0.0, 103.0, 0.0, 0.0, 0.0)
    poses = iter(
        [
            EndPoseMMDeg(0.0, 0.0, 100.0, 0.0, 0.0, 0.0),
            EndPoseMMDeg(0.0, 0.0, 101.5, 0.0, 0.0, 0.0),
            EndPoseMMDeg(0.0, 0.0, 103.0, 0.0, 0.0, 0.0),
        ]
    )
    motion_states = iter(["NOT_ARRIVED", "NOT_ARRIVED", "ARRIVED"])
    refresh_calls: list[str] = []
    interrupt_checks: list[str] = []

    wait_until_pose_goal(
        target=target,
        timeout_s=1.0,
        poll_interval_s=0.0,
        pos_tolerance_mm=0.5,
        rot_tolerance_deg=1.0,
        check_interrupt=lambda: interrupt_checks.append("checked"),
        refresh_command=lambda: refresh_calls.append("refresh"),
        read_pose=lambda: next(poses),
        pose_error=lambda expected, actual: {
            "dpos_mm": abs(actual.z_mm - expected.z_mm),
            "drot_deg": 0.0,
        },
        get_motion_status=lambda: next(motion_states),
        require_motion_arrived=True,
        sleep=lambda _seconds: None,
    )

    assert len(refresh_calls) == 3
    assert len(interrupt_checks) == 3


def test_wait_until_pose_goal_requires_stable_hold_after_arrival():
    target = EndPoseMMDeg(0.0, 0.0, 103.0, 0.0, 0.0, 0.0)
    poses = iter(
        [
            EndPoseMMDeg(0.0, 0.0, 103.0, 0.0, 0.0, 0.0),
            EndPoseMMDeg(0.0, 0.0, 112.0, 0.0, 0.0, 0.0),
            EndPoseMMDeg(0.0, 0.0, 103.0, 0.0, 0.0, 0.0),
            EndPoseMMDeg(0.0, 0.0, 103.0, 0.0, 0.0, 0.0),
        ]
    )
    refresh_calls: list[str] = []

    timeline = iter([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

    wait_until_pose_goal(
        target=target,
        timeout_s=1.0,
        poll_interval_s=0.0,
        pos_tolerance_mm=0.5,
        rot_tolerance_deg=1.0,
        post_goal_hold_s=0.2,
        check_interrupt=lambda: None,
        refresh_command=lambda: refresh_calls.append("refresh"),
        read_pose=lambda: next(poses),
        pose_error=lambda expected, actual: {
            "dpos_mm": abs(actual.z_mm - expected.z_mm),
            "drot_deg": 0.0,
        },
        get_motion_status=lambda: "ARRIVED",
        require_motion_arrived=True,
        sleep=lambda _seconds: None,
        monotonic=lambda: next(timeline),
    )

    assert len(refresh_calls) == 4
