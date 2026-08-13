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


def test_wait_until_pose_goal_accepts_healthy_stable_pose_when_arrived_is_stuck():
    target = EndPoseMMDeg(0.0, 0.0, 103.0, 0.0, 0.0, 0.0)
    timeline = iter([0.0, 0.1, 0.2, 0.5, 0.8])
    health_checks: list[str] = []

    wait_until_pose_goal(
        target=target,
        timeout_s=2.0,
        poll_interval_s=0.0,
        pos_tolerance_mm=20.0,
        rot_tolerance_deg=10.0,
        check_interrupt=lambda: None,
        refresh_command=None,
        read_pose=lambda: EndPoseMMDeg(0.5, 0.0, 103.0, 0.0, 0.2, 0.0),
        pose_error=lambda expected, actual: {
            "dpos_mm": abs(actual.x_mm - expected.x_mm),
            "drot_deg": abs(actual.pitch_deg - expected.pitch_deg),
        },
        get_motion_status=lambda: "NOT_ARRIVED",
        require_motion_arrived=True,
        pose_only_fallback_enabled=True,
        pose_only_fallback_pos_tolerance_mm=2.0,
        pose_only_fallback_rot_tolerance_deg=2.0,
        pose_only_fallback_hold_s=0.6,
        can_accept_pose_only=lambda: health_checks.append("checked") or True,
        sleep=lambda _seconds: None,
        monotonic=lambda: next(timeline),
    )

    assert len(health_checks) == 2


def test_wait_until_pose_goal_does_not_fallback_when_controller_is_unhealthy():
    target = EndPoseMMDeg(0.0, 0.0, 103.0, 0.0, 0.0, 0.0)
    timeline = iter([0.0, 0.1, 0.2, 1.1])

    try:
        wait_until_pose_goal(
            target=target,
            timeout_s=1.0,
            poll_interval_s=0.0,
            pos_tolerance_mm=20.0,
            rot_tolerance_deg=10.0,
            check_interrupt=lambda: None,
            refresh_command=None,
            read_pose=lambda: target,
            pose_error=lambda _expected, _actual: {"dpos_mm": 0.0, "drot_deg": 0.0},
            get_motion_status=lambda: "NOT_ARRIVED",
            require_motion_arrived=True,
            pose_only_fallback_enabled=True,
            can_accept_pose_only=lambda: False,
            sleep=lambda _seconds: None,
            monotonic=lambda: next(timeline),
        )
    except TimeoutError as exc:
        assert "motion_status=NOT_ARRIVED" in str(exc)
    else:
        raise AssertionError("unhealthy controller must not use pose-only fallback")


def test_wait_until_pose_goal_progress_removes_fixed_total_deadline():
    target = EndPoseMMDeg(0.0, 0.0, 100.0, 0.0, 0.0, 0.0)
    poses = iter(
        [
            EndPoseMMDeg(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            EndPoseMMDeg(0.0, 0.0, 40.0, 0.0, 0.0, 0.0),
            EndPoseMMDeg(0.0, 0.0, 80.0, 0.0, 0.0, 0.0),
            target,
        ]
    )
    # Completion occurs at t=2.0, beyond the original one-second total limit.
    # New-best feedback at t=0.9 and t=1.2 refreshes the stall watchdog.
    timeline = iter([0.0, 0.1, 0.9, 0.9, 1.2, 1.2, 2.0, 2.0])

    wait_until_pose_goal(
        target=target,
        timeout_s=1.0,
        poll_interval_s=0.0,
        pos_tolerance_mm=0.5,
        rot_tolerance_deg=1.0,
        check_interrupt=lambda: None,
        refresh_command=None,
        read_pose=lambda: next(poses),
        pose_error=lambda expected, actual: {
            "dpos_mm": abs(actual.z_mm - expected.z_mm),
            "drot_deg": 0.0,
        },
        get_motion_status=lambda: "ARRIVED",
        require_motion_arrived=True,
        progress_extends_timeout=True,
        sleep=lambda _seconds: None,
        monotonic=lambda: next(timeline),
    )
