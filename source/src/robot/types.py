"""Robot-agnostic data structures shared by planning and control layers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class EndPoseMMDeg:
    x_mm: float
    y_mm: float
    z_mm: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float


@dataclass(slots=True)
class ArmStatusSnapshot:
    control_mode: str = "unknown"
    arm_status: str = "unknown"
    move_mode: str = "unknown"
    motion_status: str = "unknown"
    teach_status: str = "unknown"
    control_mode_code: int = -1
    arm_status_code: int = -1
    move_mode_code: int = -1
    motion_status_code: int = -1
    teach_status_code: int = -1
    trajectory_index: int = 0
    err_code: int = 0
    raw_summary: str = "unavailable"


@dataclass(slots=True)
class GripperStatus:
    angle_mm: float = 0.0
    effort_nm: float = 0.0
    enabled: bool = False
