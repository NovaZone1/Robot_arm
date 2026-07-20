"""Data models for distributed robot execution requests and results."""

from __future__ import annotations

from dataclasses import dataclass
import json
import time
from typing import Any

from .types import EndPoseMMDeg


@dataclass(slots=True)
class MovePoseCommand:
    name: str
    pose: EndPoseMMDeg
    speed_percent: float = 15.0
    timeout_s: float = 8.0
    pos_tolerance_mm: float = 8.0
    rot_tolerance_deg: float = 6.0


@dataclass(slots=True)
class OpenGripperCommand:
    name: str
    open_mm: float
    effort_nm: float | None = None
    wait_target_mm: float | None = None
    wait_tol_mm: float = 1.5
    wait_timeout_s: float = 3.0


@dataclass(slots=True)
class CloseGripperCommand:
    name: str
    effort_nm: float | None = None
    wait_effort_nm: float | None = None
    wait_timeout_s: float = 3.0


@dataclass(slots=True)
class SleepCommand:
    name: str
    duration_s: float


ExecutionCommand = MovePoseCommand | OpenGripperCommand | CloseGripperCommand | SleepCommand


@dataclass(slots=True)
class RobotExecutionPlan:
    plan_id: str
    commands: list[ExecutionCommand]
    metadata: dict[str, Any]


@dataclass(slots=True)
class RobotExecutionResult:
    status: str
    plan_id: str
    message: str
    backend: str
    started_at_s: float
    finished_at_s: float
    executed_commands: list[str]
    stop_requested: bool = False
    cancel_requested: bool = False
    failure_command: str | None = None
    error: str | None = None
    arm_status: str | None = None
    final_pose: EndPoseMMDeg | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "status": self.status,
            "plan_id": self.plan_id,
            "message": self.message,
            "backend": self.backend,
            "started_at_s": self.started_at_s,
            "finished_at_s": self.finished_at_s,
            "elapsed_s": max(0.0, self.finished_at_s - self.started_at_s),
            "executed_commands": self.executed_commands,
            "stop_requested": self.stop_requested,
            "cancel_requested": self.cancel_requested,
            "failure_command": self.failure_command,
            "error": self.error,
            "arm_status": self.arm_status,
        }
        if self.final_pose is not None:
            payload["final_pose_mm_deg"] = {
                "x_mm": self.final_pose.x_mm,
                "y_mm": self.final_pose.y_mm,
                "z_mm": self.final_pose.z_mm,
                "roll_deg": self.final_pose.roll_deg,
                "pitch_deg": self.final_pose.pitch_deg,
                "yaw_deg": self.final_pose.yaw_deg,
            }
        return payload

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


def _require_keys(payload: dict[str, Any], keys: list[str], *, path: str) -> None:
    missing = [k for k in keys if k not in payload]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")


def _parse_pose(payload: dict[str, Any], *, path: str) -> EndPoseMMDeg:
    _require_keys(payload, ["x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"], path=path)
    return EndPoseMMDeg(
        x_mm=float(payload["x_mm"]),
        y_mm=float(payload["y_mm"]),
        z_mm=float(payload["z_mm"]),
        roll_deg=float(payload["roll_deg"]),
        pitch_deg=float(payload["pitch_deg"]),
        yaw_deg=float(payload["yaw_deg"]),
    )


def _parse_command(payload: dict[str, Any], *, index: int) -> ExecutionCommand:
    cmd_type = str(payload.get("type", "")).strip().lower()
    if not cmd_type:
        raise ValueError(f"commands[{index}]: missing type")
    name = str(payload.get("name") or f"{cmd_type}_{index}")

    if cmd_type == "move_pose":
        pose_payload = payload.get("pose_mm_deg")
        if not isinstance(pose_payload, dict):
            raise ValueError(f"commands[{index}]: move_pose requires object pose_mm_deg")
        return MovePoseCommand(
            name=name,
            pose=_parse_pose(pose_payload, path=f"commands[{index}].pose_mm_deg"),
            speed_percent=float(payload.get("speed_percent", 15.0)),
            timeout_s=float(payload.get("timeout_s", 8.0)),
            pos_tolerance_mm=float(payload.get("pos_tolerance_mm", 8.0)),
            rot_tolerance_deg=float(payload.get("rot_tolerance_deg", 6.0)),
        )

    if cmd_type == "open_gripper":
        if "open_mm" not in payload:
            raise ValueError(f"commands[{index}]: open_gripper requires open_mm")
        wait_target = payload.get("wait_target_mm")
        return OpenGripperCommand(
            name=name,
            open_mm=float(payload["open_mm"]),
            effort_nm=float(payload["effort_nm"]) if payload.get("effort_nm") is not None else None,
            wait_target_mm=float(wait_target) if wait_target is not None else None,
            wait_tol_mm=float(payload.get("wait_tol_mm", 1.5)),
            wait_timeout_s=float(payload.get("wait_timeout_s", 3.0)),
        )

    if cmd_type == "close_gripper":
        wait_effort = payload.get("wait_effort_nm")
        return CloseGripperCommand(
            name=name,
            effort_nm=float(payload["effort_nm"]) if payload.get("effort_nm") is not None else None,
            wait_effort_nm=float(wait_effort) if wait_effort is not None else None,
            wait_timeout_s=float(payload.get("wait_timeout_s", 3.0)),
        )

    if cmd_type == "sleep":
        if "duration_s" not in payload:
            raise ValueError(f"commands[{index}]: sleep requires duration_s")
        return SleepCommand(
            name=name,
            duration_s=float(payload["duration_s"]),
        )

    raise ValueError(f"commands[{index}]: unknown type '{cmd_type}'")


def parse_execution_plan_json(text: str) -> RobotExecutionPlan:
    if not text or not text.strip():
        raise ValueError("empty plan payload")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("plan payload must be a JSON object")

    plan_id = str(payload.get("plan_id") or f"plan-{int(time.time() * 1000)}")
    commands_payload = payload.get("commands")
    if not isinstance(commands_payload, list) or len(commands_payload) == 0:
        raise ValueError("plan payload requires non-empty array field 'commands'")

    commands: list[ExecutionCommand] = []
    for index, item in enumerate(commands_payload):
        if not isinstance(item, dict):
            raise ValueError(f"commands[{index}] must be an object")
        commands.append(_parse_command(item, index=index))

    metadata = payload.get("metadata")
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object when provided")

    return RobotExecutionPlan(plan_id=plan_id, commands=commands, metadata=metadata)
