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
    pose_only_fallback_enabled: bool = False,
    pose_only_fallback_pos_tolerance_mm: float = 2.0,
    pose_only_fallback_rot_tolerance_deg: float = 2.0,
    pose_only_fallback_hold_s: float = 0.6,
    can_accept_pose_only=None,
    progress_extends_timeout: bool = False,
    progress_position_epsilon_mm: float = 0.5,
    progress_rotation_epsilon_deg: float = 0.25,
    sleep=time.sleep,
    monotonic=time.monotonic,
) -> None:
    # ``timeout_s`` is normally a wall-clock deadline.  On slow real Piper
    # moves that can reject an otherwise healthy command while the endpoint is
    # still converging.  When progress extension is enabled it instead acts as
    # a *stall* timeout: every meaningful new best pose refreshes the deadline.
    # Interrupt, controller-health and ARRIVED checks remain unchanged.
    stall_timeout_s = max(0.2, float(timeout_s))
    deadline = monotonic() + stall_timeout_s
    hold_duration_s = max(0.0, float(post_goal_hold_s))
    pose_only_since: float | None = None
    last_error: dict[str, float] | None = None
    last_motion_status: str | None = None
    best_dpos_mm: float | None = None
    best_drot_deg: float | None = None
    while monotonic() < deadline:
        check_interrupt()
        if refresh_command is not None:
            refresh_command()
        actual = read_pose()
        err = pose_error(target, actual)
        last_error = err
        dpos_mm = abs(float(err["dpos_mm"]))
        drot_deg = abs(float(err["drot_deg"]))
        if best_dpos_mm is None or best_drot_deg is None:
            best_dpos_mm = dpos_mm
            best_drot_deg = drot_deg
        elif bool(progress_extends_timeout):
            made_progress = (
                dpos_mm
                <= best_dpos_mm - max(0.0, float(progress_position_epsilon_mm))
                or drot_deg
                <= best_drot_deg - max(0.0, float(progress_rotation_epsilon_deg))
            )
            if made_progress:
                best_dpos_mm = min(best_dpos_mm, dpos_mm)
                best_drot_deg = min(best_drot_deg, drot_deg)
                deadline = monotonic() + stall_timeout_s
        motion_status = None
        if require_motion_arrived and get_motion_status is not None:
            motion_status = get_motion_status()
            last_motion_status = str(motion_status)
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

        # Piper occasionally leaves motion_status at NOT_ARRIVED even after the
        # endpoint has settled precisely on the commanded pose.  Keep ARRIVED
        # as the primary completion signal, but allow a tightly bounded,
        # continuously stable pose to complete the move when the caller also
        # confirms that the controller is healthy.  The strict fallback
        # tolerance prevents the normal (coarser) waypoint tolerance from
        # closing a gripper while the arm is merely passing near the target.
        fallback_allowed = (
            bool(pose_only_fallback_enabled)
            and bool(require_motion_arrived)
            and str(motion_status or "").strip().upper() == "NOT_ARRIVED"
            and (can_accept_pose_only is None or bool(can_accept_pose_only()))
            and pose_goal_reached(
                dpos_mm=err["dpos_mm"],
                drot_deg=err["drot_deg"],
                pos_tolerance_mm=pose_only_fallback_pos_tolerance_mm,
                rot_tolerance_deg=pose_only_fallback_rot_tolerance_deg,
                require_motion_arrived=False,
            )
        )
        if fallback_allowed:
            now = monotonic()
            if pose_only_since is None:
                pose_only_since = now
            elif now - pose_only_since >= max(0.0, float(pose_only_fallback_hold_s)):
                return
        else:
            pose_only_since = None
        sleep(max(0.0, float(poll_interval_s)))
    detail = ""
    if last_error is not None:
        detail = (
            f" last_dpos={float(last_error['dpos_mm']):.2f}mm"
            f" last_drot={float(last_error['drot_deg']):.2f}deg"
        )
    if last_motion_status is not None:
        detail += f" motion_status={last_motion_status}"
    failure_kind = "move stalled" if bool(progress_extends_timeout) else "move timeout"
    raise TimeoutError(
        f"{failure_kind}: pos_tol={float(pos_tolerance_mm):.2f} "
        f"rot_tol={float(rot_tolerance_deg):.2f}{detail}"
    )
