"""Robot-agnostic grasp planning utilities extracted from the legacy pipeline."""

from __future__ import annotations

import numpy as np

from src.grasping.models import GraspCandidate, GraspExecutionConfig, GraspPlan, PerceptionResult
from src.robot.types import EndPoseMMDeg
from src.utils.transforms import (
    make_transform_matrix,
    rotation_matrix_from_rpy_deg,
    rpy_deg_from_rotation_matrix,
)


class PureGraspPlanner:
    """Build grasp plans from perception and transforms without touching robot IO."""

    _MIN_CANDIDATE_POSE_Z_M = -0.10

    # GraspNet convention: x=approach, y=closing, z=baseline.
    # Tool convention:     x=lateral,  y=closing, z=tool outward (approach).
    _R_ADJUST = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    _R_FLIP_AROUND_TOOL_Z = np.diag([-1.0, -1.0, 1.0])

    def __init__(
        self,
        config: GraspExecutionConfig,
        hand_eye: np.ndarray,
        online_bias: dict[str, object] | None = None,
    ):
        self.config = config
        self.hand_eye = np.asarray(hand_eye, dtype=np.float64).reshape(4, 4)
        self.online_bias = online_bias

    @staticmethod
    def _normalize_columns(matrix: np.ndarray) -> np.ndarray:
        normalized = np.asarray(matrix, dtype=np.float64).copy()
        for column in range(normalized.shape[1]):
            norm = float(np.linalg.norm(normalized[:, column]))
            if norm > 1e-8:
                normalized[:, column] /= norm
        return normalized

    @staticmethod
    def _rotation_angle_deg(rotation_a: np.ndarray, rotation_b: np.ndarray) -> float:
        relative = rotation_a.T @ rotation_b
        trace_value = float(np.trace(relative))
        cos_theta = max(-1.0, min(1.0, 0.5 * (trace_value - 1.0)))
        return float(np.degrees(np.arccos(cos_theta)))

    def tool_contact_offset_tool_m(self) -> np.ndarray:
        if self.config.tool_contact_offset_tool_m is not None:
            return np.asarray(self.config.tool_contact_offset_tool_m, dtype=np.float64).reshape(3)
        return np.array([0.0, 0.0, float(self.config.gripper_length_m)], dtype=np.float64)

    @staticmethod
    def estimate_tool_lowest_z(
        flange_point_m: tuple[float, float, float],
        tool_contact_offset_base: np.ndarray | None = None,
    ) -> float:
        flange = np.asarray(flange_point_m, dtype=np.float64).reshape(3)
        if tool_contact_offset_base is None:
            return float(flange[2])
        offset = np.asarray(tool_contact_offset_base, dtype=np.float64).reshape(3)
        tip = flange + offset
        return float(min(flange[2], tip[2]))

    def check_workspace(
        self,
        point_m: tuple[float, float, float],
        tool_contact_offset_base: np.ndarray | None = None,
    ) -> tuple[bool, list[str]]:
        x, y, z = point_m
        violations: list[str] = []
        if not (self.config.workspace_x_limits_m[0] <= x <= self.config.workspace_x_limits_m[1]):
            violations.append(f"x={x:.3f} not in {self.config.workspace_x_limits_m}")
        if not (self.config.workspace_y_limits_m[0] <= y <= self.config.workspace_y_limits_m[1]):
            violations.append(f"y={y:.3f} not in {self.config.workspace_y_limits_m}")
        if not (self.config.workspace_z_limits_m[0] <= z <= self.config.workspace_z_limits_m[1]):
            violations.append(f"z={z:.3f} not in {self.config.workspace_z_limits_m}")
        lowest_tool_z = self.estimate_tool_lowest_z(point_m, tool_contact_offset_base)
        min_safe_z = self.config.table_z_m + self.config.min_gripper_table_clearance_m
        if lowest_tool_z < min_safe_z:
            violations.append(
                f"tool_lowest_z={lowest_tool_z:.3f} below table safety height {min_safe_z:.3f} "
                f"(table_z_m={self.config.table_z_m:.3f} + clearance={self.config.min_gripper_table_clearance_m:.3f})"
            )
        return len(violations) == 0, violations

    def approach_angle_to_vertical_deg(
        self,
        rotation_camera: np.ndarray,
        base_to_camera: np.ndarray,
    ) -> float:
        approach_camera = np.asarray(rotation_camera, dtype=np.float64)[:, 0]
        approach_base = np.asarray(base_to_camera, dtype=np.float64)[:3, :3] @ approach_camera
        vertical_down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
        cos_angle = float(np.dot(approach_base, vertical_down))
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return float(np.degrees(np.arccos(cos_angle)))

    def build_plan_data(
        self,
        candidate: GraspCandidate,
        tcp_pose: EndPoseMMDeg,
        base_to_camera: np.ndarray,
        forced_tool_roll_deg: float | None = None,
        force_180deg_equivalent: bool = False,
        forced_target_rpy_deg: tuple[float, float, float] | None = None,
    ) -> dict[str, object]:
        tcp_rotation = rotation_matrix_from_rpy_deg(
            tcp_pose.roll_deg,
            tcp_pose.pitch_deg,
            tcp_pose.yaw_deg,
        )

        raw_axes_camera = np.asarray(candidate.rotation_camera, dtype=np.float64).copy()
        adjusted_rotation = raw_axes_camera @ self._R_ADJUST
        camera_to_grasp = make_transform_matrix(adjusted_rotation, candidate.translation_camera_m)
        base_to_camera = np.asarray(base_to_camera, dtype=np.float64).reshape(4, 4)

        raw_axes_base = base_to_camera[:3, :3] @ raw_axes_camera
        adjusted_axes_camera = adjusted_rotation.copy()
        adjusted_axes_base = base_to_camera[:3, :3] @ adjusted_axes_camera

        raw_axes_camera = self._normalize_columns(raw_axes_camera)
        raw_axes_base = self._normalize_columns(raw_axes_base)
        adjusted_axes_camera = self._normalize_columns(adjusted_axes_camera)
        adjusted_axes_base = self._normalize_columns(adjusted_axes_base)

        base_to_grasp = base_to_camera @ camera_to_grasp

        def transform_with_tool_roll(tool_roll_deg: float) -> np.ndarray:
            rolled = base_to_grasp.copy()
            roll_rotation = rotation_matrix_from_rpy_deg(0.0, 0.0, tool_roll_deg)
            rolled[:3, :3] = base_to_grasp[:3, :3] @ roll_rotation
            return rolled

        direct_rot_error_deg = self._rotation_angle_deg(tcp_rotation, base_to_grasp[:3, :3])
        flipped_base_to_grasp = base_to_grasp.copy()
        flipped_base_to_grasp[:3, :3] = base_to_grasp[:3, :3] @ self._R_FLIP_AROUND_TOOL_Z
        flipped_rot_error_deg = self._rotation_angle_deg(tcp_rotation, flipped_base_to_grasp[:3, :3])

        candidate_variants: list[tuple[str, float, float, np.ndarray]] = []
        if forced_target_rpy_deg is not None:
            forced_rotation = rotation_matrix_from_rpy_deg(
                float(forced_target_rpy_deg[0]),
                float(forced_target_rpy_deg[1]),
                float(forced_target_rpy_deg[2]),
            )
            forced_transform = base_to_grasp.copy()
            forced_transform[:3, :3] = forced_rotation
            candidate_variants.append(
                (
                    f"forced_rpy_{float(forced_target_rpy_deg[0]):.1f}_"
                    f"{float(forced_target_rpy_deg[1]):.1f}_"
                    f"{float(forced_target_rpy_deg[2]):.1f}",
                    0.0,
                    self._rotation_angle_deg(tcp_rotation, forced_rotation),
                    forced_transform,
                )
            )
        elif force_180deg_equivalent:
            candidate_variants.append(
                (
                    "flipped_180deg",
                    180.0,
                    flipped_rot_error_deg,
                    flipped_base_to_grasp,
                )
            )
        elif forced_tool_roll_deg is not None:
            roll_deg = float(forced_tool_roll_deg)
            rolled_transform = transform_with_tool_roll(roll_deg)
            variant_name = "direct" if abs(roll_deg) < 1e-9 else f"tool_roll_{roll_deg:.1f}deg"
            candidate_variants.append(
                (
                    variant_name,
                    roll_deg,
                    self._rotation_angle_deg(tcp_rotation, rolled_transform[:3, :3]),
                    rolled_transform,
                )
            )
        else:
            seen_rolls: set[float] = set()
            for roll_deg in self.config.wrist_roll_search_deg:
                normalized_roll = round(float(roll_deg), 6)
                if normalized_roll in seen_rolls:
                    continue
                seen_rolls.add(normalized_roll)
                rolled_transform = transform_with_tool_roll(float(roll_deg))
                variant_name = "direct" if abs(float(roll_deg)) < 1e-9 else f"tool_roll_{float(roll_deg):.1f}deg"
                candidate_variants.append(
                    (
                        variant_name,
                        float(roll_deg),
                        self._rotation_angle_deg(tcp_rotation, rolled_transform[:3, :3]),
                        rolled_transform,
                    )
                )
            if 0.0 not in seen_rolls:
                candidate_variants.append(("direct", 0.0, direct_rot_error_deg, base_to_grasp))
            if self.config.allow_180deg_equivalent_grasp:
                candidate_variants.append(
                    (
                        "flipped_180deg",
                        180.0,
                        flipped_rot_error_deg,
                        flipped_base_to_grasp,
                    )
                )

        chosen_variant, chosen_wrist_roll_deg, chosen_rot_error_deg, base_to_grasp = min(
            candidate_variants,
            key=lambda item: (item[2], abs(item[1])),
        )

        approach_axis_base = base_to_grasp[:3, 2]
        if np.linalg.norm(approach_axis_base) < 1e-8:
            raise RuntimeError("Invalid grasp approach axis")
        approach_axis_base = approach_axis_base / np.linalg.norm(approach_axis_base)

        grasp_contact_point = base_to_grasp[:3, 3]
        tool_contact_offset_tool_m = self.tool_contact_offset_tool_m()
        gripper_compensation_vector = base_to_grasp[:3, :3] @ tool_contact_offset_tool_m

        grasp_y_bias_mode = "manual"
        if self.config.grasp_y_bias_mm is None:
            grasp_y_bias_m = -float(gripper_compensation_vector[1])
            grasp_y_bias_mode = "auto_from_gripper_compensation"
        else:
            grasp_y_bias_m = float(self.config.grasp_y_bias_mm) / 1000.0

        if abs(grasp_y_bias_m) > 1e-9:
            grasp_contact_point = grasp_contact_point + np.array([0.0, grasp_y_bias_m, 0.0], dtype=np.float64)

        target_translation = grasp_contact_point - gripper_compensation_vector
        pregrasp_translation = target_translation - approach_axis_base * self.config.pregrasp_offset_m
        grasp_translation = target_translation - approach_axis_base * self.config.descend_offset_m
        retreat_translation = grasp_translation + np.array(
            [0.0, 0.0, float(self.config.retreat_offset_m)],
            dtype=np.float64,
        )

        if self.config.grasp_z_offset_m != 0.0:
            z_offset = np.array([0.0, 0.0, float(self.config.grasp_z_offset_m)], dtype=np.float64)
            target_translation = target_translation + z_offset
            pregrasp_translation = pregrasp_translation + z_offset
            grasp_translation = grasp_translation + z_offset
            retreat_translation = retreat_translation + z_offset

        online_bias_m = np.zeros(3, dtype=np.float64)
        online_bias_type = "disabled"
        if self.online_bias is not None:
            bias_mm = self.online_bias.get("bias_mm", {})
            if not isinstance(bias_mm, dict):
                raise RuntimeError("Invalid online_bias payload: bias_mm must be a dict")
            online_bias_m = np.array(
                [
                    float(bias_mm.get("x_mm", 0.0)) / 1000.0,
                    float(bias_mm.get("y_mm", 0.0)) / 1000.0,
                    float(bias_mm.get("z_mm", 0.0)) / 1000.0,
                ],
                dtype=np.float64,
            )
            online_bias_type = str(self.online_bias.get("reference_point_type", "unknown"))
            grasp_contact_point = grasp_contact_point + online_bias_m
            target_translation = target_translation + online_bias_m
            pregrasp_translation = pregrasp_translation + online_bias_m
            grasp_translation = grasp_translation + online_bias_m
            retreat_translation = retreat_translation + online_bias_m

        target_rpy_deg = rpy_deg_from_rotation_matrix(base_to_grasp[:3, :3])
        base_to_camera_rpy_deg = rpy_deg_from_rotation_matrix(base_to_camera[:3, :3])
        hand_eye_rpy_deg = rpy_deg_from_rotation_matrix(self.hand_eye[:3, :3])
        camera_origin_base = base_to_camera[:3, 3]
        camera_translation = np.asarray(candidate.translation_camera_m, dtype=np.float64)
        base_rotation = base_to_camera[:3, :3]
        grasp_xyz_base_from_cam = base_rotation @ camera_translation + camera_origin_base
        x_base_components_mm = (
            float(base_rotation[0, 0] * camera_translation[0] * 1000.0),
            float(base_rotation[0, 1] * camera_translation[1] * 1000.0),
            float(base_rotation[0, 2] * camera_translation[2] * 1000.0),
            float(camera_origin_base[0] * 1000.0),
        )
        y_base_components_mm = (
            float(base_rotation[1, 0] * camera_translation[0] * 1000.0),
            float(base_rotation[1, 1] * camera_translation[1] * 1000.0),
            float(base_rotation[1, 2] * camera_translation[2] * 1000.0),
            float(camera_origin_base[1] * 1000.0),
        )
        z_base_components_mm = (
            float(base_rotation[2, 0] * camera_translation[0] * 1000.0),
            float(base_rotation[2, 1] * camera_translation[1] * 1000.0),
            float(base_rotation[2, 2] * camera_translation[2] * 1000.0),
            float(camera_origin_base[2] * 1000.0),
        )
        grasp_from_camera_base = grasp_contact_point - camera_origin_base
        target_from_camera_base = target_translation - camera_origin_base
        target_tool_lowest_z = self.estimate_tool_lowest_z(tuple(target_translation.tolist()), gripper_compensation_vector)
        pregrasp_tool_lowest_z = self.estimate_tool_lowest_z(tuple(pregrasp_translation.tolist()), gripper_compensation_vector)
        grasp_tool_lowest_z = self.estimate_tool_lowest_z(tuple(grasp_translation.tolist()), gripper_compensation_vector)
        retreat_tool_lowest_z = self.estimate_tool_lowest_z(tuple(retreat_translation.tolist()), gripper_compensation_vector)
        target_contact_point = target_translation + gripper_compensation_vector

        target_base_m = (
            float(target_translation[0]),
            float(target_translation[1]),
            float(target_translation[2]),
        )
        pregrasp_base_m = (
            float(pregrasp_translation[0]),
            float(pregrasp_translation[1]),
            float(pregrasp_translation[2]),
        )
        grasp_base_m = (
            float(grasp_translation[0]),
            float(grasp_translation[1]),
            float(grasp_translation[2]),
        )
        retreat_base_m = (
            float(retreat_translation[0]),
            float(retreat_translation[1]),
            float(retreat_translation[2]),
        )

        workspace_violations: list[str] = []

        def record_pose_checks(name: str, point: tuple[float, float, float]) -> bool:
            ok, violations = self.check_workspace(point, gripper_compensation_vector)
            if not ok:
                workspace_violations.extend([f"{name} {violation}" for violation in violations])
            return ok

        if self.config.safe_top_down_candidate_filter:
            record_pose_checks("target", target_base_m)
        else:
            if self.config.enable_pregrasp:
                record_pose_checks("pregrasp", pregrasp_base_m)
            record_pose_checks("target", target_base_m)
            record_pose_checks("grasp", grasp_base_m)
            record_pose_checks("retreat", retreat_base_m)

        return {
            "tcp": tcp_pose,
            "base_to_camera": base_to_camera,
            "raw_axes_camera": raw_axes_camera,
            "raw_axes_base": raw_axes_base,
            "adjusted_axes_camera": adjusted_axes_camera,
            "adjusted_axes_base": adjusted_axes_base,
            "base_to_grasp": base_to_grasp,
            "chosen_variant": chosen_variant,
            "chosen_wrist_roll_deg": chosen_wrist_roll_deg,
            "chosen_rot_error_deg": chosen_rot_error_deg,
            "direct_rot_error_deg": direct_rot_error_deg,
            "flipped_rot_error_deg": flipped_rot_error_deg,
            "approach_axis_base": approach_axis_base,
            "grasp_contact_point": grasp_contact_point,
            "grasp_y_bias_m": grasp_y_bias_m,
            "grasp_y_bias_mode": grasp_y_bias_mode,
            "tool_contact_offset_tool_m": tool_contact_offset_tool_m,
            "gripper_compensation_vector": gripper_compensation_vector,
            "target_translation": target_translation,
            "pregrasp_translation": pregrasp_translation,
            "grasp_translation": grasp_translation,
            "retreat_translation": retreat_translation,
            "online_bias_m": online_bias_m,
            "online_bias_type": online_bias_type,
            "target_rpy_deg": target_rpy_deg,
            "base_to_camera_rpy_deg": base_to_camera_rpy_deg,
            "hand_eye_rpy_deg": hand_eye_rpy_deg,
            "camera_origin_base": camera_origin_base,
            "grasp_xyz_base_from_cam": grasp_xyz_base_from_cam,
            "x_base_components_mm": x_base_components_mm,
            "y_base_components_mm": y_base_components_mm,
            "z_base_components_mm": z_base_components_mm,
            "grasp_from_camera_base": grasp_from_camera_base,
            "target_from_camera_base": target_from_camera_base,
            "target_tool_lowest_z": target_tool_lowest_z,
            "pregrasp_tool_lowest_z": pregrasp_tool_lowest_z,
            "grasp_tool_lowest_z": grasp_tool_lowest_z,
            "retreat_tool_lowest_z": retreat_tool_lowest_z,
            "target_contact_point": target_contact_point,
            "target_base_m": target_base_m,
            "pregrasp_base_m": pregrasp_base_m,
            "grasp_base_m": grasp_base_m,
            "retreat_base_m": retreat_base_m,
            "workspace_violations": workspace_violations,
            "within_workspace": len(workspace_violations) == 0,
        }

    def plan_grasp(
        self,
        candidate: GraspCandidate,
        tcp_pose: EndPoseMMDeg,
        base_to_camera: np.ndarray,
    ) -> GraspPlan:
        data = self.build_plan_data(candidate, tcp_pose, base_to_camera)
        return GraspPlan(
            candidate=candidate,
            target_base_m=data["target_base_m"],
            target_rpy_deg=data["target_rpy_deg"],
            pregrasp_base_m=data["pregrasp_base_m"],
            grasp_base_m=data["grasp_base_m"],
            retreat_base_m=data["retreat_base_m"],
            within_workspace=data["within_workspace"],
            workspace_violations=data["workspace_violations"],
            target_contact_point_base_m=tuple(float(value) for value in data["target_contact_point"]),
            tool_contact_offset_tool_m=tuple(float(value) for value in data["tool_contact_offset_tool_m"]),
        )

    def plan_grasp_variants(
        self,
        candidate: GraspCandidate,
        tcp_pose: EndPoseMMDeg,
        base_to_camera: np.ndarray,
    ) -> list[GraspPlan]:
        plans: list[GraspPlan] = []
        seen_rolls: set[float] = set()
        for roll_deg in self.config.wrist_roll_search_deg:
            normalized_roll = round(float(roll_deg), 6)
            if normalized_roll in seen_rolls:
                continue
            seen_rolls.add(normalized_roll)
            try:
                data = self.build_plan_data(
                    candidate,
                    tcp_pose,
                    base_to_camera,
                    forced_tool_roll_deg=float(roll_deg),
                )
            except Exception:
                continue
            plans.append(
                GraspPlan(
                    candidate=candidate,
                    target_base_m=data["target_base_m"],
                    target_rpy_deg=data["target_rpy_deg"],
                    pregrasp_base_m=data["pregrasp_base_m"],
                    grasp_base_m=data["grasp_base_m"],
                    retreat_base_m=data["retreat_base_m"],
                    within_workspace=data["within_workspace"],
                    workspace_violations=data["workspace_violations"],
                    target_contact_point_base_m=tuple(float(value) for value in data["target_contact_point"]),
                    tool_contact_offset_tool_m=tuple(float(value) for value in data["tool_contact_offset_tool_m"]),
                )
            )
        if self.config.allow_180deg_equivalent_grasp:
            try:
                data = self.build_plan_data(
                    candidate,
                    tcp_pose,
                    base_to_camera,
                    force_180deg_equivalent=True,
                )
                plans.append(
                    GraspPlan(
                        candidate=candidate,
                        target_base_m=data["target_base_m"],
                        target_rpy_deg=data["target_rpy_deg"],
                        pregrasp_base_m=data["pregrasp_base_m"],
                        grasp_base_m=data["grasp_base_m"],
                        retreat_base_m=data["retreat_base_m"],
                        within_workspace=data["within_workspace"],
                        workspace_violations=data["workspace_violations"],
                        target_contact_point_base_m=tuple(float(value) for value in data["target_contact_point"]),
                        tool_contact_offset_tool_m=tuple(float(value) for value in data["tool_contact_offset_tool_m"]),
                    )
                )
            except Exception:
                pass
        if not plans:
            plans.append(self.plan_grasp(candidate, tcp_pose, base_to_camera))
        fallback_plans: list[GraspPlan] = []
        fallback_rpys = [
            (float(tcp_pose.roll_deg), float(tcp_pose.pitch_deg), float(tcp_pose.yaw_deg)),
            (0.0, 120.0, 0.0),
            (180.0, 60.0, 180.0),
        ]
        seen_rpys = {tuple(round(float(value), 3) for value in plan.target_rpy_deg) for plan in plans}
        for fallback_rpy in fallback_rpys:
            normalized = tuple(round(float(value), 3) for value in fallback_rpy)
            if normalized in seen_rpys:
                continue
            seen_rpys.add(normalized)
            try:
                data = self.build_plan_data(
                    candidate,
                    tcp_pose,
                    base_to_camera,
                    forced_target_rpy_deg=fallback_rpy,
                )
            except Exception:
                continue
            fallback_plans.append(
                GraspPlan(
                    candidate=candidate,
                    target_base_m=data["target_base_m"],
                    target_rpy_deg=data["target_rpy_deg"],
                    pregrasp_base_m=data["pregrasp_base_m"],
                    grasp_base_m=data["grasp_base_m"],
                    retreat_base_m=data["retreat_base_m"],
                    within_workspace=data["within_workspace"],
                    workspace_violations=data["workspace_violations"],
                    target_contact_point_base_m=tuple(float(value) for value in data["target_contact_point"]),
                    tool_contact_offset_tool_m=tuple(float(value) for value in data["tool_contact_offset_tool_m"]),
                )
            )
        plans = fallback_plans + plans
        return plans

    def _candidate_rank(self, candidate: GraspCandidate, angle_deg: float) -> tuple[float, float, float]:
        center_offset = candidate.center_offset_m if candidate.center_offset_m is not None else float("inf")
        if self.config.flat_object_mode:
            edge_priority = -min(center_offset, self.config.max_grasp_center_offset_m)
            return (angle_deg, edge_priority, -candidate.score)
        return (center_offset, -candidate.score, angle_deg)

    def _pose_floor_violations(self, plan_data: dict[str, object]) -> list[str]:
        violations: list[str] = []
        pose_names = (
            ("target_base_m",)
            if self.config.safe_top_down_candidate_filter
            else ("pregrasp_base_m", "target_base_m", "grasp_base_m", "retreat_base_m")
        )
        for name in pose_names:
            point = plan_data.get(name)
            if point is None:
                continue
            z_value = float(tuple(point)[2])
            if z_value < self._MIN_CANDIDATE_POSE_Z_M:
                pose_name = name.removesuffix("_base_m")
                violations.append(
                    f"{pose_name} z={z_value:.3f} below minimum pose z {self._MIN_CANDIDATE_POSE_Z_M:.3f}"
                )
        return violations

    def _candidate_collection_plan_data_variants(
        self,
        candidate: GraspCandidate,
        tcp_pose: EndPoseMMDeg,
        base_to_camera: np.ndarray,
    ) -> list[dict[str, object]]:
        variants: list[dict[str, object]] = []
        seen_rpys: set[tuple[float, float, float]] = set()

        def add_data(data: dict[str, object]) -> None:
            normalized_rpy = tuple(round(float(value), 3) for value in data["target_rpy_deg"])
            if normalized_rpy in seen_rpys:
                return
            seen_rpys.add(normalized_rpy)
            variants.append(data)

        try:
            add_data(self.build_plan_data(candidate, tcp_pose, base_to_camera))
        except Exception:
            pass

        fallback_rpys = [
            (float(tcp_pose.roll_deg), float(tcp_pose.pitch_deg), float(tcp_pose.yaw_deg)),
            (0.0, 120.0, 0.0),
            (180.0, 60.0, 180.0),
        ]
        for fallback_rpy in fallback_rpys:
            try:
                add_data(
                    self.build_plan_data(
                        candidate,
                        tcp_pose,
                        base_to_camera,
                        forced_target_rpy_deg=fallback_rpy,
                    )
                )
            except Exception:
                continue
        return variants

    def collect_grasp_candidates(
        self,
        perception: PerceptionResult,
        tcp_pose: EndPoseMMDeg,
        base_to_camera: np.ndarray,
        initial_diagnostics: list[str] | None = None,
    ) -> tuple[list[tuple[GraspCandidate, float, float]], list[str], float]:
        def collect_with_max_angle(
            max_angle: float,
        ) -> tuple[list[tuple[GraspCandidate, float, float]], list[str], int]:
            candidate_pool: list[tuple[GraspCandidate, float, float]] = []
            diagnostics: list[str] = list(initial_diagnostics or [])
            total_filtered_by_angle = 0

            for instance_index, grasp_group in enumerate(perception.grasp_groups):
                if grasp_group is None or len(grasp_group) == 0:
                    diagnostics.append(
                        f"instance {instance_index}: no grasp after mask filtering | "
                        f"scene_grasps={perception.scene_grasp_count} "
                        f"object_points={perception.object_point_counts[instance_index]}"
                    )
                    continue

                object_center = perception.object_centers_camera_m[instance_index]
                instance_filtered_center = 0
                instance_filtered_score = 0
                instance_filtered_angle = 0
                instance_filtered_workspace = 0
                instance_filtered_rotation = 0
                rejected_angle_examples: list[float] = []
                rejected_score_examples: list[float] = []
                rejected_rotation_examples: list[float] = []
                workspace_examples: list[str] = []
                pose_floor_examples: list[str] = []
                instance_candidates: list[tuple[GraspCandidate, float, float]] = []

                for grasp in grasp_group[: min(len(grasp_group), self.config.grasp_topk)]:
                    try:
                        score = float(getattr(grasp, "score", 0.0))
                        width_m = float(getattr(grasp, "width", 0.0))
                        depth_m = float(getattr(grasp, "depth", 0.0))
                        translation = np.asarray(getattr(grasp, "translation"), dtype=np.float64).reshape(3)
                        rotation = np.asarray(getattr(grasp, "rotation_matrix"), dtype=np.float64).reshape(3, 3)
                    except Exception as exc:
                        raise RuntimeError(
                            f"Failed to parse GraspNet result for instance {instance_index}: {exc}"
                        ) from exc

                    if score < self.config.min_grasp_score:
                        instance_filtered_score += 1
                        if len(rejected_score_examples) < 3:
                            rejected_score_examples.append(round(score, 4))
                        continue

                    center_offset_m = None
                    if object_center is not None:
                        center_offset_m = float(
                            np.linalg.norm(translation - np.asarray(object_center, dtype=np.float64))
                        )
                        if center_offset_m > self.config.max_grasp_center_offset_m:
                            instance_filtered_center += 1
                            continue

                    angle_deg = self.approach_angle_to_vertical_deg(rotation, base_to_camera)
                    if angle_deg > max_angle:
                        instance_filtered_angle += 1
                        if len(rejected_angle_examples) < 3:
                            rejected_angle_examples.append(round(angle_deg, 1))
                        continue

                    candidate = GraspCandidate(
                        instance_index=instance_index,
                        score=score,
                        width_m=width_m,
                        depth_m=depth_m,
                        translation_camera_m=(float(translation[0]), float(translation[1]), float(translation[2])),
                        rotation_camera=rotation,
                        object_center_camera_m=object_center,
                        center_offset_m=center_offset_m,
                        raw_grasp=grasp,
                    )
                    plan_data_variants = self._candidate_collection_plan_data_variants(
                        candidate,
                        tcp_pose,
                        base_to_camera,
                    )
                    if not plan_data_variants:
                        instance_filtered_workspace += 1
                        if len(workspace_examples) < 2:
                            workspace_examples.append("plan_build_failed")
                        continue

                    viable_plan_data: list[dict[str, object]] = []
                    candidate_pose_floor_examples: list[str] = []
                    candidate_workspace_examples: list[str] = []
                    candidate_rotation_examples: list[float] = []
                    for plan_data in plan_data_variants:
                        pose_floor_violations = self._pose_floor_violations(plan_data)
                        if pose_floor_violations:
                            if len(candidate_pose_floor_examples) < 2:
                                candidate_pose_floor_examples.append(pose_floor_violations[0])
                            continue
                        if not bool(plan_data["within_workspace"]):
                            if len(candidate_workspace_examples) < 2:
                                candidate_workspace_examples.append(
                                    f"target={tuple(round(v, 3) for v in plan_data['target_base_m'])} "
                                    f"violations={'; '.join(plan_data['workspace_violations'])}"
                                )
                            continue
                        chosen_rot_error_deg = float(plan_data["chosen_rot_error_deg"])
                        if chosen_rot_error_deg > self.config.max_reachable_rotation_delta_deg:
                            if len(candidate_rotation_examples) < 3:
                                candidate_rotation_examples.append(round(chosen_rot_error_deg, 1))
                            continue
                        viable_plan_data.append(plan_data)

                    if not viable_plan_data:
                        if candidate_rotation_examples and not candidate_pose_floor_examples and not candidate_workspace_examples:
                            instance_filtered_rotation += 1
                            if len(rejected_rotation_examples) < 3:
                                rejected_rotation_examples.extend(candidate_rotation_examples[: 3 - len(rejected_rotation_examples)])
                        else:
                            instance_filtered_workspace += 1
                            if candidate_pose_floor_examples and len(pose_floor_examples) < 2:
                                pose_floor_examples.append(candidate_pose_floor_examples[0])
                            if candidate_workspace_examples and len(workspace_examples) < 2:
                                workspace_examples.append(candidate_workspace_examples[0])
                        continue

                    chosen_rot_error_deg = min(float(data["chosen_rot_error_deg"]) for data in viable_plan_data)
                    if chosen_rot_error_deg > self.config.max_reachable_rotation_delta_deg:
                        instance_filtered_rotation += 1
                        if len(rejected_rotation_examples) < 3:
                            rejected_rotation_examples.append(round(chosen_rot_error_deg, 1))
                        continue
                    instance_candidates.append((candidate, angle_deg, chosen_rot_error_deg))

                total_filtered_by_angle += instance_filtered_angle
                if not instance_candidates:
                    diagnostics.append(
                        f"instance {instance_index}: no filtered grasp | object_points={perception.object_point_counts[instance_index]} "
                        f"filtered_by_center_count={instance_filtered_center} filtered_by_score_count={instance_filtered_score} "
                        f"min_grasp_score={self.config.min_grasp_score:.4f} "
                        f"filtered_by_angle_count={instance_filtered_angle} "
                        f"angle_threshold_deg={max_angle:.1f} "
                        f"filtered_by_workspace={instance_filtered_workspace}"
                        f" filtered_by_rotation_count={instance_filtered_rotation} "
                        f"rotation_threshold_deg={self.config.max_reachable_rotation_delta_deg:.1f}"
                        + (f" rejected_score_examples={rejected_score_examples}" if rejected_score_examples else "")
                        + (
                            f" rejected_angle_examples_deg={rejected_angle_examples}"
                            if rejected_angle_examples
                            else ""
                        )
                        + (
                            f" rejected_rotation_examples_deg={rejected_rotation_examples}"
                            if rejected_rotation_examples
                            else ""
                        )
                        + (f" pose_floor_examples={pose_floor_examples}" if pose_floor_examples else "")
                        + (f" workspace_examples={workspace_examples}" if workspace_examples else "")
                    )
                    continue

                instance_candidates.sort(key=lambda item: self._candidate_rank(item[0], item[1]))
                instance_best, best_angle, best_rotation_delta = instance_candidates[0]
                candidate_pool.extend(instance_candidates)
                diagnostics.append(
                    f"instance {instance_index}: grasps={len(grasp_group)} score={instance_best.score:.4f} "
                    f"approach_angle={best_angle:.1f}deg "
                    f"rotation_delta={best_rotation_delta:.1f}deg "
                    f"object_points={perception.object_point_counts[instance_index]} "
                    f"center_offset_m={instance_best.center_offset_m if instance_best.center_offset_m is not None else float('nan'):.4f} "
                    f"filtered_by_center_count={instance_filtered_center} filtered_by_score_count={instance_filtered_score} "
                    f"min_grasp_score={self.config.min_grasp_score:.4f} "
                    f"filtered_by_angle_count={instance_filtered_angle} "
                    f"angle_threshold_deg={max_angle:.1f} "
                    f"filtered_by_workspace={instance_filtered_workspace}"
                    f" filtered_by_rotation_count={instance_filtered_rotation} "
                    f"rotation_threshold_deg={self.config.max_reachable_rotation_delta_deg:.1f}"
                    + (f" rejected_score_examples={rejected_score_examples}" if rejected_score_examples else "")
                    + (
                        f" rejected_angle_examples_deg={rejected_angle_examples}"
                        if rejected_angle_examples
                        else ""
                    )
                    + (
                        f" rejected_rotation_examples_deg={rejected_rotation_examples}"
                        if rejected_rotation_examples
                        else ""
                    )
                    + (f" pose_floor_examples={pose_floor_examples}" if pose_floor_examples else "")
                    + (f" workspace_examples={workspace_examples}" if workspace_examples else "")
                )

            candidate_pool.sort(key=lambda item: self._candidate_rank(item[0], item[1]))
            return candidate_pool, diagnostics, total_filtered_by_angle

        max_angle = self.config.max_approach_angle_deg
        candidate_pool, diagnostics, total_filtered_by_angle = collect_with_max_angle(max_angle)
        if candidate_pool or total_filtered_by_angle == 0:
            return candidate_pool, diagnostics, max_angle

        for relaxed_max_angle in (90.0, 120.0, 150.0, 180.0):
            if relaxed_max_angle <= max_angle:
                continue
            relaxed_pool, relaxed_diagnostics, _ = collect_with_max_angle(relaxed_max_angle)
            if relaxed_pool:
                relaxed_diagnostics.insert(
                    0,
                    "strict top-down filter found no candidate, "
                    f"fallback relaxes max_approach_angle from {max_angle:.1f}deg to {relaxed_max_angle:.1f}deg",
                )
                return relaxed_pool, relaxed_diagnostics, relaxed_max_angle
        return candidate_pool, diagnostics, max_angle

    def select_best_grasp(
        self,
        perception: PerceptionResult,
        tcp_pose: EndPoseMMDeg,
        base_to_camera: np.ndarray,
    ) -> tuple[GraspCandidate, list[str], float]:
        candidate_pool, diagnostics, max_angle = self.collect_grasp_candidates(
            perception,
            tcp_pose,
            base_to_camera,
        )
        if not candidate_pool:
            raise RuntimeError("No valid grasp candidate found. details: " + " | ".join(diagnostics))
        return candidate_pool[0][0], diagnostics, max_angle
