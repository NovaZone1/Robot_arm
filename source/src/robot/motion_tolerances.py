"""Helpers for selecting motion completion tolerances."""

from __future__ import annotations

import math
import time

from .types import EndPoseMMDeg


def translation_distance_mm(start: EndPoseMMDeg, target: EndPoseMMDeg) -> float:
    dx = float(target.x_mm) - float(start.x_mm)
    dy = float(target.y_mm) - float(start.y_mm)
    dz = float(target.z_mm) - float(start.z_mm)
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def effective_position_tolerance_mm(
    *,
    start: EndPoseMMDeg,
    target: EndPoseMMDeg,
    default_tolerance_mm: float,
    enable_tightening: bool = True,
    minimum_tolerance_mm: float = 0.5,
    tightening_ratio: float = 0.5,
    zero_motion_threshold_mm: float = 0.25,
) -> float:
    if not bool(enable_tightening):
        return float(default_tolerance_mm)
    requested_translation_mm = translation_distance_mm(start, target)
    if requested_translation_mm <= max(0.0, float(zero_motion_threshold_mm)):
        return float(default_tolerance_mm)
    if requested_translation_mm >= float(default_tolerance_mm):
        return float(default_tolerance_mm)
    return max(float(minimum_tolerance_mm), requested_translation_mm * float(tightening_ratio))


def pose_goal_reached(
    *,
    dpos_mm: float,
    drot_deg: float,
    pos_tolerance_mm: float,
    rot_tolerance_deg: float,
    motion_status: str | None = None,
    require_motion_arrived: bool = False,
) -> bool:
    if abs(float(dpos_mm)) > float(pos_tolerance_mm):
        return False
    if abs(float(drot_deg)) > float(rot_tolerance_deg):
        return False
    if require_motion_arrived and str(motion_status or "").strip().upper() != "ARRIVED":
        return False
    return True


def wait_until_pose_goal(
    *,
    target: EndPoseMMDeg,
    timeout_s: float,
    poll_interval_s: float,
    pos_tolerance_mm: float,
    rot_tolerance_deg: float,
    post_goal_hold_s: float = 0.0,
    check_interrupt,
    refresh_command,
    read_pose,
    pose_error,
    get_motion_status=None,
    require_motion_arrived: bool = False,
    sleep=time.sleep,
    monotonic=time.monotonic,
) -> None:
    deadline = monotonic() + max(0.2, float(timeout_s))
    hold_duration_s = max(0.0, float(post_goal_hold_s))
    while monotonic() < deadline:
        check_interrupt()
        if refresh_command is not None:
            refresh_command()
        actual = read_pose()
        err = pose_error(target, actual)
        motion_status = None
        if require_motion_arrived and get_motion_status is not None:
            motion_status = get_motion_status()
        if pose_goal_reached(
            dpos_mm=err["dpos_mm"],
            drot_deg=err["drot_deg"],
            pos_tolerance_mm=pos_tolerance_mm,
            rot_tolerance_deg=rot_tolerance_deg,
            motion_status=motion_status,
            require_motion_arrived=require_motion_arrived,
        ):
            if hold_duration_s <= 1e-9:
                return
            hold_deadline = monotonic() + hold_duration_s
            stable = True
            while monotonic() < hold_deadline:
                check_interrupt()
                if refresh_command is not None:
                    refresh_command()
                actual = read_pose()
                err = pose_error(target, actual)
                motion_status = None
                if require_motion_arrived and get_motion_status is not None:
                    motion_status = get_motion_status()
                if not pose_goal_reached(
                    dpos_mm=err["dpos_mm"],
                    drot_deg=err["drot_deg"],
                    pos_tolerance_mm=pos_tolerance_mm,
                    rot_tolerance_deg=rot_tolerance_deg,
                    motion_status=motion_status,
                    require_motion_arrived=require_motion_arrived,
                ):
                    stable = False
                    break
                sleep(max(0.0, float(poll_interval_s)))
            if stable:
                return
        sleep(max(0.0, float(poll_interval_s)))
    raise TimeoutError(
        f"move timeout: pos_tol={float(pos_tolerance_mm):.2f} rot_tol={float(rot_tolerance_deg):.2f}"
    )
