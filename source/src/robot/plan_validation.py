from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from src.grasping.models import GraspCandidate, GraspPlan

from .types import EndPoseMMDeg


def _pose_from_plan_point(
    position_m: tuple[float, float, float],
    rpy_deg: tuple[float, float, float],
) -> EndPoseMMDeg:
    return EndPoseMMDeg(
        x_mm=float(position_m[0]) * 1000.0,
        y_mm=float(position_m[1]) * 1000.0,
        z_mm=float(position_m[2]) * 1000.0,
        roll_deg=float(rpy_deg[0]),
        pitch_deg=float(rpy_deg[1]),
        yaw_deg=float(rpy_deg[2]),
    )


def validate_grasp_plan_waypoints(
    plan: GraspPlan,
    *,
    include_pregrasp: bool,
    compute_ik: Callable[[EndPoseMMDeg], object],
) -> list[str]:
    results = validate_grasp_plan_waypoints_detailed(
        plan,
        include_pregrasp=include_pregrasp,
        compute_ik=compute_ik,
    )
    return [str(item["stage"]) for item in results if item.get("status") == "ok"]


def _classify_ik_error(exc: Exception) -> tuple[str, str]:
    message = str(exc).strip()
    lowered = message.lower()
    if isinstance(exc, TimeoutError) or "timed out" in lowered or "timeout" in lowered:
        return "timeout", message
    if "code=-31" in lowered or "no_ik_solution" in lowered:
        return "no_ik_solution", message
    return "ik_error", message


def validate_grasp_plan_waypoints_detailed(
    plan: GraspPlan,
    *,
    include_pregrasp: bool,
    compute_ik: Callable[[EndPoseMMDeg], object],
) -> list[dict[str, Any]]:
    waypoints: list[tuple[str, tuple[float, float, float]]] = []
    if include_pregrasp:
        waypoints.append(("pregrasp", plan.pregrasp_base_m))
    waypoints.extend(
        [
            ("grasp", plan.grasp_base_m),
            ("target", plan.target_base_m),
            ("retreat", plan.retreat_base_m),
        ]
    )

    validated: list[dict[str, Any]] = []
    for name, position_m in waypoints:
        try:
            compute_ik(_pose_from_plan_point(position_m, plan.target_rpy_deg))
            validated.append({"stage": name, "status": "ok"})
        except Exception as exc:
            error_type, error_message = _classify_ik_error(exc)
            validated.append(
                {
                    "stage": name,
                    "status": "failed",
                    "ik_error_type": error_type,
                    "ik_error_message": error_message,
                }
            )
            break
    return validated


def select_first_reachable_candidate(
    candidates: Sequence[GraspCandidate],
    *,
    build_plan: Callable[[GraspCandidate], GraspPlan | Sequence[GraspPlan]],
    validate_plan: Callable[[GraspCandidate, GraspPlan], str | None | dict[str, Any]],
) -> tuple[GraspCandidate | None, GraspPlan | None, list[str], list[dict[str, Any]]]:
    diagnostics: list[str] = []
    validation_records: list[dict[str, Any]] = []

    for index, candidate in enumerate(candidates):
        try:
            built_plan = build_plan(candidate)
        except Exception as exc:
            diagnostics.append(f"candidate[{index}] score={candidate.score:.4f} rejected during plan build: {exc}")
            validation_records.append(
                {
                    "candidate_index": index,
                    "candidate_score": candidate.score,
                    "instance_index": candidate.instance_index,
                    "translation_camera_m": list(candidate.translation_camera_m),
                    "object_center_camera_m": (
                        list(candidate.object_center_camera_m)
                        if candidate.object_center_camera_m is not None
                        else None
                    ),
                    "center_offset_m": candidate.center_offset_m,
                    "candidate_width_m": candidate.width_m,
                    "candidate_depth_m": candidate.depth_m,
                    "target_base_m": None,
                    "target_rpy_deg": None,
                    "pregrasp_base_m": None,
                    "grasp_base_m": None,
                    "retreat_base_m": None,
                    "within_workspace": None,
                    "workspace_violations": [],
                    "robot_validation_result": "rejected_during_plan_build",
                    "robot_validation_stage": "plan_build",
                    "ik_error_type": "plan_build_error",
                    "ik_error_message": str(exc),
                    "waypoint_results": [],
                    "selection_result": "rejected_during_plan_build",
                }
            )
            continue

        if isinstance(built_plan, GraspPlan):
            plan_variants = [built_plan]
        else:
            plan_variants = [item for item in built_plan if isinstance(item, GraspPlan)]
        if not plan_variants:
            diagnostics.append(f"candidate[{index}] score={candidate.score:.4f} rejected during plan build: no plan variants")
            validation_records.append(
                {
                    "candidate_index": index,
                    "candidate_plan_variant_index": None,
                    "candidate_score": candidate.score,
                    "instance_index": candidate.instance_index,
                    "translation_camera_m": list(candidate.translation_camera_m),
                    "object_center_camera_m": (
                        list(candidate.object_center_camera_m)
                        if candidate.object_center_camera_m is not None
                        else None
                    ),
                    "center_offset_m": candidate.center_offset_m,
                    "candidate_width_m": candidate.width_m,
                    "candidate_depth_m": candidate.depth_m,
                    "target_base_m": None,
                    "target_rpy_deg": None,
                    "pregrasp_base_m": None,
                    "grasp_base_m": None,
                    "retreat_base_m": None,
                    "within_workspace": None,
                    "workspace_violations": [],
                    "robot_validation_result": "rejected_during_plan_build",
                    "robot_validation_stage": "plan_build",
                    "ik_error_type": "plan_build_error",
                    "ik_error_message": "no plan variants",
                    "waypoint_results": [],
                    "selection_result": "rejected_during_plan_build",
                }
            )
            continue

        for variant_index, plan in enumerate(plan_variants):
            record = {
                "candidate_index": index,
                "candidate_plan_variant_index": variant_index,
                "candidate_score": candidate.score,
                "instance_index": candidate.instance_index,
                "translation_camera_m": list(candidate.translation_camera_m),
                "object_center_camera_m": (
                    list(candidate.object_center_camera_m)
                    if candidate.object_center_camera_m is not None
                    else None
                ),
                "center_offset_m": candidate.center_offset_m,
                "candidate_width_m": candidate.width_m,
                "candidate_depth_m": candidate.depth_m,
                "target_base_m": list(plan.target_base_m),
                "target_rpy_deg": list(plan.target_rpy_deg),
                "pregrasp_base_m": list(plan.pregrasp_base_m),
                "grasp_base_m": list(plan.grasp_base_m),
                "retreat_base_m": list(plan.retreat_base_m),
                "within_workspace": bool(plan.within_workspace),
                "workspace_violations": list(plan.workspace_violations),
                "robot_validation_result": "accepted",
                "robot_validation_stage": None,
                "ik_error_type": None,
                "ik_error_message": None,
                "waypoint_results": [],
                "selection_result": "accepted_not_selected",
            }
            try:
                validation_result = validate_plan(candidate, plan)
            except Exception as exc:
                validation_result = {
                    "robot_validation_result": "rejected_by_robot_validation",
                    "robot_validation_stage": None,
                    "ik_error_type": _classify_ik_error(exc)[0],
                    "ik_error_message": str(exc),
                    "waypoint_results": [],
                }

            if isinstance(validation_result, dict):
                record["robot_validation_result"] = str(validation_result.get("robot_validation_result") or "accepted")
                record["robot_validation_stage"] = validation_result.get("robot_validation_stage")
                record["ik_error_type"] = validation_result.get("ik_error_type")
                record["ik_error_message"] = validation_result.get("ik_error_message")
                record["waypoint_results"] = list(validation_result.get("waypoint_results") or [])
                error_message = str(validation_result.get("ik_error_message") or "").strip()
                accepted = record["robot_validation_result"] == "accepted"
            else:
                error_message = str(validation_result or "").strip()
                accepted = not error_message
                if not accepted:
                    record["robot_validation_result"] = "rejected_by_robot_validation"
                    record["ik_error_message"] = error_message

            if accepted:
                record["selection_result"] = "selected_for_execution"
                validation_records.append(record)
                return candidate, plan, diagnostics, validation_records

            diagnostics.append(
                f"candidate[{index}] variant[{variant_index}] score={candidate.score:.4f} "
                f"rejected by robot validation: {error_message}"
            )
            record["selection_result"] = "rejected_by_robot_validation"
            validation_records.append(record)

    return None, None, diagnostics, validation_records
