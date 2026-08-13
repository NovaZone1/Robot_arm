from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping.models import GraspCandidate, PerceptionResult
from src.grasping.planning import PureGraspPlanner
from src.perception.geometry import (
    depth_to_scene_points,
    keep_largest_point_cluster,
    median_filter_depth,
    remove_radius_outliers,
    save_segmentation_outputs,
)
from src.robot.types import EndPoseMMDeg
from src.run_grasp_pipeline_ros2 import build_config, build_parser
from src.utils.calibration import load_camera_to_tcp_transform


@dataclass(slots=True)
class SimpleIntrinsics:
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float


def _json_dumps(payload: dict[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _parse_options_json(text: str) -> dict[str, object]:
    if not text or not text.strip():
        return {}
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("options_json must decode to an object")
    return payload


def _build_args_from_options(options: dict[str, object]) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args([])
    extra_cli_args: list[str] = []
    for key, value in options.items():
        if key == "extra_cli_args":
            if isinstance(value, list):
                extra_cli_args = [str(item) for item in value]
            continue
        if hasattr(args, key):
            setattr(args, key, value)
    if extra_cli_args:
        args = parser.parse_args(extra_cli_args, namespace=args)
    return args


def _load_hand_eye_matrix(config_path: str | Path) -> np.ndarray:
    resolved = Path(config_path).expanduser().resolve()
    with open(resolved, "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    calib = payload.get("calibration", {})
    transform, _ = load_camera_to_tcp_transform(calib, allow_legacy=False)
    return np.asarray(transform, dtype=np.float64).reshape(4, 4)


def _rotation_matrix_to_list(rotation: np.ndarray) -> list[list[float]]:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return [[float(value) for value in row] for row in matrix]


def _candidate_to_dict(candidate: GraspCandidate) -> dict[str, object]:
    return {
        "instance_index": int(candidate.instance_index),
        "score": float(candidate.score),
        "width_m": float(candidate.width_m),
        "depth_m": float(candidate.depth_m),
        "translation_camera_m": [float(value) for value in candidate.translation_camera_m],
        "rotation_camera": _rotation_matrix_to_list(candidate.rotation_camera),
        "object_center_camera_m": (
            [float(value) for value in candidate.object_center_camera_m]
            if candidate.object_center_camera_m is not None
            else None
        ),
        "center_offset_m": (
            float(candidate.center_offset_m) if candidate.center_offset_m is not None else None
        ),
    }


def _plan_to_dict(plan) -> dict[str, object]:
    return {
        "candidate": _candidate_to_dict(plan.candidate),
        "target_base_m": [float(value) for value in plan.target_base_m],
        "target_rpy_deg": [float(value) for value in plan.target_rpy_deg],
        "pregrasp_base_m": [float(value) for value in plan.pregrasp_base_m],
        "grasp_base_m": [float(value) for value in plan.grasp_base_m],
        "retreat_base_m": [float(value) for value in plan.retreat_base_m],
        "target_contact_point_base_m": (
            [float(value) for value in plan.target_contact_point_base_m]
            if plan.target_contact_point_base_m is not None
            else None
        ),
        "tool_contact_offset_tool_m": (
            [float(value) for value in plan.tool_contact_offset_tool_m]
            if plan.tool_contact_offset_tool_m is not None
            else None
        ),
        "within_workspace": bool(plan.within_workspace),
        "workspace_violations": [str(item) for item in plan.workspace_violations],
    }


def _save_grasp_visualization(
    *,
    output_dir: Path,
    color_bgr: np.ndarray,
    scene_id: str,
    segmentation: dict,
    scene_grasp_group,
    grasp_groups: list,
    intrinsics: SimpleIntrinsics,
) -> None:
    """Save visualization images: overlay + grasp prediction overlay."""
    import cv2 as _cv2

    # Save to persistent location so viz survives temp dir cleanup
    project_root = Path(__file__).resolve().parents[2]
    workspace_root = Path(os.environ.get("ROBOT_GRASP_WORKSPACE_ROOT", project_root.parent / "ros_ws"))
    viz_dir = workspace_root / "viz" / scene_id
    viz_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Segmentation overlay ──────────────────────────────────
    masks = segmentation.get("masks")
    scores_list = segmentation.get("scores")
    boxes = segmentation.get("boxes")
    if hasattr(masks, "shape") and len(masks.shape) >= 3:
        n = int(masks.shape[0])
    else:
        n = int(len(masks)) if masks is not None else 0

    overlay = color_bgr.copy()
    COLORS = [
        (0, 255, 0), (255, 0, 0), (0, 0, 255),
        (255, 255, 0), (255, 0, 255), (0, 255, 255),
    ]
    for idx in range(n):
        mask = masks[idx].squeeze()
        mask_np = mask.detach().cpu().numpy().astype(bool) if hasattr(mask, "detach") else np.asarray(mask).astype(bool)
        if mask_np.shape[:2] != overlay.shape[:2]:
            mask_np = _cv2.resize(mask_np.astype(np.uint8), (overlay.shape[1], overlay.shape[0]), interpolation=_cv2.INTER_NEAREST).astype(bool)
        color = COLORS[idx % len(COLORS)]
        overlay[mask_np] = (overlay[mask_np] * 0.5 + np.array(color) * 0.5).astype(np.uint8)
        contours, _ = _cv2.findContours(mask_np.astype(np.uint8), _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_SIMPLE)
        _cv2.drawContours(overlay, contours, -1, color, 2)
        if scores_list is not None and idx < len(scores_list):
            score = float(scores_list[idx].item() if hasattr(scores_list[idx], "item") else scores_list[idx])
            _cv2.putText(overlay, f"#{idx} {score:.2f}", (10, 30 + idx * 25), _cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    _cv2.imwrite(str(viz_dir / "segmentation_overlay.png"), overlay)

    # ── 2. Grasp prediction overlay ──────────────────────────────
    grasp_overlay = color_bgr.copy()
    fx, fy = float(intrinsics.fx), float(intrinsics.fy)
    cx, cy = float(intrinsics.ppx), float(intrinsics.ppy)

    def _draw_grasps(img, gg, color, label_prefix=""):
        if gg is None or len(gg) == 0:
            return
        topk = min(len(gg), 30)
        for i, g in enumerate(gg[:topk]):
            try:
                t = np.asarray(getattr(g, "translation"), dtype=np.float64).reshape(3)
            except Exception:
                continue
            z = float(t[2])
            if z <= 1e-6:
                continue
            u = int(round((float(t[0]) * fx / z) + cx))
            v = int(round((float(t[1]) * fy / z) + cy))
            u = max(0, min(img.shape[1] - 1, u))
            v = max(0, min(img.shape[0] - 1, v))
            _cv2.circle(img, (u, v), 5, color, -1)
            _cv2.putText(img, f"{label_prefix}{i}", (u + 8, v), _cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    if scene_grasp_group is not None and len(scene_grasp_group) > 0:
        _draw_grasps(grasp_overlay, scene_grasp_group, (255, 0, 0), "S")  # blue = scene
    for idx, gg in enumerate(grasp_groups):
        if gg is not None and len(gg) > 0:
            _draw_grasps(grasp_overlay, gg, COLORS[idx % len(COLORS)], f"G{idx}")  # colored = instance

    _cv2.imwrite(str(viz_dir / "grasp_projection.png"), grasp_overlay)

    # ── 3. Summary text ──────────────────────────────────────────
    lines = [
        f"scene: {scene_id}",
        f"instances: {n}",
        f"scene_grasps: {len(scene_grasp_group) if scene_grasp_group is not None else 0}",
        f"instance_grasps: {[len(gg) if gg is not None else 0 for gg in grasp_groups]}",
    ]
    summary = "\n".join(lines)
    (viz_dir / "summary.txt").write_text(summary, encoding="utf-8")


class ExternalInferenceEngine:
    def __init__(self) -> None:
        self._segmenter = None
        self._graspnet = None
        self._runtime_key: tuple[str, str] | None = None
        self._planner: PureGraspPlanner | None = None
        self._config = None

    def _ensure_runtime(self, options: dict[str, object]) -> None:
        args = _build_args_from_options(options)
        config, _summary = build_config(args)
        hand_eye_path = str(options.get("hand_eye_config") or config.hand_eye_config_path)
        runtime_key = (
            str(config.graspnet_checkpoint),
            hand_eye_path,
        )
        if self._runtime_key == runtime_key and self._planner is not None:
            self._config = config
            self._planner.config = config
            return

        from src.perception.graspnet_runner import GraspNetRunner, resolve_graspnet_checkpoint
        from src.perception.yolo_segmenter import YOLOSegmenter

        checkpoint = resolve_graspnet_checkpoint(config.graspnet_checkpoint)
        if not checkpoint:
            raise RuntimeError("GraspNet checkpoint is not configured")

        hand_eye = _load_hand_eye_matrix(hand_eye_path)
        self._segmenter = YOLOSegmenter(
            device=config.grasp_device,
            model_name="yolov8n-seg.pt",
        )
        self._graspnet = GraspNetRunner(
            checkpoint_path=checkpoint,
            device=config.grasp_device,
            num_point=config.grasp_num_point,
            topk=config.grasp_topk,
            voxel_size=config.grasp_voxel_size,
            collision_thresh=config.grasp_collision_thresh,
            approach_dist=config.grasp_approach_dist,
        )
        self._planner = PureGraspPlanner(config, hand_eye)
        self._config = config
        self._runtime_key = runtime_key

    def warmup(self, options: dict[str, object]) -> None:
        """Load models and execute one synthetic pass before a real task."""
        self._ensure_runtime(options)
        if self._segmenter is None or self._graspnet is None:
            raise RuntimeError("inference runtime did not initialize")

        # Both Ultralytics and CUDA defer meaningful setup until the first
        # forward pass.  Paying that cost while the dashboard stack starts
        # keeps it out of the navigation-to-grasp handoff critical path.
        dummy_image = np.zeros((320, 320, 3), dtype=np.uint8)
        self._segmenter.segment_text(dummy_image, "bottle")
        rng = np.random.default_rng(0)
        dummy_points = rng.uniform(
            low=(-0.10, -0.10, 0.35),
            high=(0.10, 0.10, 0.65),
            size=(max(64, int(self._graspnet.num_point)), 3),
        ).astype(np.float32)
        self._graspnet.predict(dummy_points, dummy_points)

    def detect_target_2d(
        self,
        color_bgr: np.ndarray,
        prompt: str,
    ) -> dict[str, object]:
        """Locate a target cheaply, without point-cloud or GraspNet inference."""
        if self._segmenter is None:
            from src.perception.yolo_segmenter import YOLOSegmenter

            self._segmenter = YOLOSegmenter(
                device=str(os.environ.get("ROBOT_GRASP_LABEL_DEVICE") or "cuda"),
                model_name="yolov8n-seg.pt",
            )
        segmentation = self._segmenter.segment_text(np.asarray(color_bgr), prompt)
        masks = segmentation.get("masks")
        scores = segmentation.get("scores")
        count = self._segmentation_count(segmentation)
        if count <= 0:
            return {
                "found": False,
                "center_u_norm": 0.0,
                "center_v_norm": 0.0,
                "confidence": 0.0,
                "backend": str(segmentation.get("backend") or "unknown"),
            }
        score_values = [
            float(value.detach().cpu().item()) if hasattr(value, "detach") else float(value)
            for value in list(scores)[:count]
        ]
        selected_index = int(np.argmax(score_values))
        mask = masks[selected_index].squeeze()
        mask_np = (
            mask.detach().cpu().numpy().astype(bool)
            if hasattr(mask, "detach")
            else np.asarray(mask).astype(bool)
        )
        center_uv = self._mask_centroid_uv(mask_np)
        height, width = np.asarray(color_bgr).shape[:2]
        if center_uv is None or width <= 0 or height <= 0:
            return {
                "found": False,
                "center_u_norm": 0.0,
                "center_v_norm": 0.0,
                "confidence": 0.0,
                "backend": str(segmentation.get("backend") or "unknown"),
            }
        return {
            "found": True,
            "center_u_norm": float(center_uv[0]) / float(width),
            "center_v_norm": float(center_uv[1]) / float(height),
            "confidence": float(score_values[selected_index]),
            "backend": str(segmentation.get("backend") or "unknown"),
        }

    def detect_label_bottles(
        self,
        color_bgr: np.ndarray,
    ) -> list[dict[str, object]]:
        """Use the existing COCO YOLO model as a bottle-shape proposal stage."""
        if self._segmenter is None:
            from src.perception.yolo_segmenter import YOLOSegmenter

            self._segmenter = YOLOSegmenter(
                device=str(os.environ.get("ROBOT_GRASP_LABEL_DEVICE") or "cuda"),
                model_name="yolov8n-seg.pt",
                conf_threshold=0.20,
            )
        segmentation = self._segmenter.segment_text(
            np.asarray(color_bgr),
            "bottle",
        )
        boxes = segmentation.get("boxes")
        scores = segmentation.get("scores")
        if boxes is None or scores is None:
            return []
        count = min(len(boxes), len(scores))
        proposals: list[dict[str, object]] = []
        for index in range(count):
            raw_box = boxes[index]
            raw_score = scores[index]
            box = (
                raw_box.detach().cpu().numpy()
                if hasattr(raw_box, "detach")
                else np.asarray(raw_box)
            )
            score = (
                float(raw_score.detach().cpu().item())
                if hasattr(raw_score, "detach")
                else float(raw_score)
            )
            x0, y0, x1, y1 = (float(value) for value in box.reshape(4))
            proposals.append(
                {
                    "confidence": score,
                    "bbox_xywh": [
                        x0,
                        y0,
                        max(0.0, x1 - x0),
                        max(0.0, y1 - y0),
                    ],
                }
            )
        return proposals

    @property
    def config(self):
        if self._config is None:
            raise RuntimeError("runtime is not initialized")
        return self._config

    @property
    def planner(self) -> PureGraspPlanner:
        if self._planner is None:
            raise RuntimeError("planner is not initialized")
        return self._planner

    def _segmentation_count(self, segmentation: dict) -> int:
        masks = segmentation.get("masks", [])
        if hasattr(masks, "shape") and len(masks.shape) >= 3:
            return int(masks.shape[0])
        return int(len(masks))

    def _mask_centroid_uv(self, mask_np: np.ndarray) -> tuple[int, int] | None:
        import cv2

        if mask_np is None or not np.any(mask_np):
            return None
        mask_uint8 = mask_np.astype(np.uint8)
        moments = cv2.moments(mask_uint8)
        if moments["m00"] > 0:
            u = int(round(moments["m10"] / moments["m00"]))
            v = int(round(moments["m01"] / moments["m00"]))
            return u, v
        ys, xs = np.where(mask_uint8 > 0)
        if len(xs) == 0:
            return None
        return int(np.mean(xs)), int(np.mean(ys))

    def _filter_scene_grasps_by_mask(
        self,
        *,
        scene_grasp_group,
        mask_np: np.ndarray,
        depth_meters: np.ndarray,
        intrinsics: SimpleIntrinsics,
        projection_radius_px: int = 8,
        depth_tolerance_m: float = 0.10,
    ):
        if scene_grasp_group is None or len(scene_grasp_group) == 0:
            return None

        height, width = mask_np.shape[:2]
        fx = float(intrinsics.fx)
        fy = float(intrinsics.fy)
        cx = float(intrinsics.ppx)
        cy = float(intrinsics.ppy)

        keep_indices: list[int] = []
        topk = min(len(scene_grasp_group), self.config.grasp_topk)
        for grasp_index, grasp in enumerate(scene_grasp_group[:topk]):
            try:
                translation = np.asarray(getattr(grasp, "translation"), dtype=np.float64).reshape(3)
            except Exception:
                continue
            z = float(translation[2])
            if z <= 1e-6:
                continue
            u = int(round((float(translation[0]) * fx / z) + cx))
            v = int(round((float(translation[1]) * fy / z) + cy))
            if not (0 <= u < width and 0 <= v < height):
                continue

            u0 = max(0, u - projection_radius_px)
            u1 = min(width, u + projection_radius_px + 1)
            v0 = max(0, v - projection_radius_px)
            v1 = min(height, v + projection_radius_px + 1)
            if not np.any(mask_np[v0:v1, u0:u1]):
                continue

            depth_patch = np.asarray(depth_meters[v0:v1, u0:u1], dtype=np.float32)
            valid_depths = depth_patch[(depth_patch > 0.0) & np.isfinite(depth_patch)]
            local_depth = float(np.median(valid_depths)) if valid_depths.size else 0.0
            if local_depth > 0.0 and abs(local_depth - z) > depth_tolerance_m:
                continue
            keep_indices.append(grasp_index)

        if not keep_indices:
            return None
        return self._graspnet.subset_grasp_group(scene_grasp_group, keep_indices)

    def _perception_overview_lines(self, perception: PerceptionResult, text_prompt: str) -> list[str]:
        segmentation_count = self._segmentation_count(perception.segmentation)
        instance_count = max(segmentation_count, len(perception.object_point_counts))
        lines = [
            f"perception overview: prompt={text_prompt!r} "
            f"instances={instance_count} segmentation_instances={segmentation_count} "
            f"scene_grasps={perception.scene_grasp_count} "
            f"scene_points={perception.scene_point_count}"
        ]
        allow_scene_fallback = bool(perception.segmentation.get("allow_scene_fallback", True))
        backend = str(perception.segmentation.get("backend", "unknown"))
        lines.append(f"segmentation backend={backend}")
        if segmentation_count == 0 and instance_count == 0:
            if allow_scene_fallback:
                lines.append("segmentation produced 0 instances for the current prompt")
            else:
                lines.append(
                    "segmentation produced 0 instances; scene grasp fallback is disabled "
                    "for this strict prompt"
                )
        elif segmentation_count == 0:
            lines.append("segmentation produced 0 instances; using scene grasp fallback pseudo-instance")
        return lines

    def analyze(
        self,
        *,
        scene_id: str,
        prompt: str,
        color_bgr: np.ndarray,
        depth_meters: np.ndarray,
        intrinsics: SimpleIntrinsics,
        tcp_pose: EndPoseMMDeg,
        base_to_camera: np.ndarray,
        options: dict[str, object],
        output_dir: Path,
    ) -> dict[str, object]:
        self._ensure_runtime(options)
        config = self.config

        filter_mode = str(config.pointcloud_filter_mode)
        if filter_mode == "median":
            depth_meters = median_filter_depth(depth_meters, kernel_size=config.median_kernel_size)

        segmentation = self._segmenter.segment_text(color_bgr, prompt)
        pointclouds = save_segmentation_outputs(
            color_bgr=color_bgr,
            depth_meters=depth_meters,
            masks=segmentation["masks"],
            scores=segmentation["scores"],
            boxes=segmentation["boxes"],
            text_prompt=prompt,
            intrinsics=intrinsics,
            clip_max=config.clip_max_m,
            output_dir=str(output_dir),
            pointcloud_filter_mode=filter_mode,
            island_eps_m=config.island_eps_m,
            island_min_points=config.island_min_points,
            radius_nb_points=config.radius_nb_points,
            radius_m=config.radius_m,
            pointcloud_backend="manual",
            depth_frame=None,
            color_frame=None,
        )

        scene_points = depth_to_scene_points(
            depth_meters,
            intrinsics,
            clip_max=config.clip_max_m,
            mask=None,
        )
        scene_grasp_group = self._graspnet.predict(
            scene_points=scene_points,
            object_points=scene_points,
        )
        scene_grasp_count = int(len(scene_grasp_group)) if scene_grasp_group is not None else 0

        grasp_groups = []
        grasp_source_debug: list[dict[str, object]] = []
        object_point_counts: list[int] = []
        object_centers_camera_m: list[tuple[float, float, float] | None] = []
        object_centers_uv: list[tuple[int, int] | None] = []
        object_cloud_paths: list[str | None] = []
        masks = segmentation["masks"]
        count = self._segmentation_count(segmentation)

        for index in range(count):
            mask = masks[index].squeeze()
            mask_np = (
                mask.detach().cpu().numpy().astype(bool)
                if hasattr(mask, "detach")
                else np.asarray(mask).astype(bool)
            )
            object_points = depth_to_scene_points(
                depth_meters,
                intrinsics,
                clip_max=config.clip_max_m,
                mask=mask_np,
            )
            if filter_mode == "island" and object_points is not None and len(object_points) > 0:
                filtered_points, _ = keep_largest_point_cluster(
                    object_points,
                    eps_m=config.island_eps_m,
                    min_points=config.island_min_points,
                )
                object_points = filtered_points.astype(np.float32, copy=False)
            elif filter_mode == "radius" and object_points is not None and len(object_points) > 0:
                filtered_points, _ = remove_radius_outliers(
                    object_points,
                    nb_points=config.radius_nb_points,
                    radius_m=config.radius_m,
                )
                object_points = filtered_points.astype(np.float32, copy=False)

            object_point_counts.append(int(len(object_points)) if object_points is not None else 0)
            if pointclouds[index] is not None and len(pointclouds[index].points) > 0:
                center = np.asarray(pointclouds[index].get_center(), dtype=np.float64).reshape(3)
                object_centers_camera_m.append((float(center[0]), float(center[1]), float(center[2])))
            else:
                object_centers_camera_m.append(None)
            object_centers_uv.append(self._mask_centroid_uv(mask_np))

            # Predict grasps directly on this instance's object point cloud
            instance_grasps = self._graspnet.predict(
                scene_points=scene_points,
                object_points=object_points,
            )
            if instance_grasps is not None and len(instance_grasps) > 0:
                instance_grasps = instance_grasps[: self.config.grasp_topk]
            scene_mask_grasps = self._filter_scene_grasps_by_mask(
                scene_grasp_group=scene_grasp_group,
                mask_np=mask_np,
                depth_meters=depth_meters,
                intrinsics=intrinsics,
            )
            merged_grasps = self._graspnet.merge_grasp_groups(
                instance_grasps,
                scene_mask_grasps,
                topk=self.config.grasp_topk,
            )
            grasp_groups.append(merged_grasps)
            grasp_source_debug.append(
                {
                    "instance_index": index,
                    "object_points": int(len(object_points)) if object_points is not None else 0,
                    "instance_grasps": int(len(instance_grasps)) if instance_grasps is not None else 0,
                    "scene_mask_grasps": int(len(scene_mask_grasps)) if scene_mask_grasps is not None else 0,
                    "merged_grasps": int(len(merged_grasps)) if merged_grasps is not None else 0,
                }
            )

            if object_points is not None and len(object_points) > 0:
                cloud_path = output_dir / f"instance_cloud_{index:02d}.npy"
                np.save(cloud_path, np.asarray(object_points, dtype=np.float32))
                object_cloud_paths.append(cloud_path.name)
            else:
                object_cloud_paths.append(None)

        if (
            count == 0
            and scene_grasp_count > 0
            and bool(segmentation.get("allow_scene_fallback", True))
        ):
            fallback_grasps = scene_grasp_group[: self.config.grasp_topk]
            grasp_groups.append(fallback_grasps)
            object_point_counts.append(int(len(scene_points)) if scene_points is not None else 0)
            object_centers_camera_m.append(None)
            object_centers_uv.append(None)
            object_cloud_paths.append(None)
            grasp_source_debug.append(
                {
                    "instance_index": 0,
                    "object_points": int(len(scene_points)) if scene_points is not None else 0,
                    "instance_grasps": 0,
                    "scene_mask_grasps": int(len(fallback_grasps)) if fallback_grasps is not None else 0,
                    "merged_grasps": int(len(fallback_grasps)) if fallback_grasps is not None else 0,
                    "fallback": "scene_grasps_without_segmentation",
                }
            )

        if scene_points is not None:
            np.save(output_dir / "scene_points.npy", np.asarray(scene_points, dtype=np.float32))

        # ── Save visualizations ──────────────────────────────────────
        _save_grasp_visualization(
            output_dir=output_dir,
            color_bgr=color_bgr,
            scene_id=scene_id,
            segmentation=segmentation,
            scene_grasp_group=scene_grasp_group,
            grasp_groups=grasp_groups,
            intrinsics=intrinsics,
        )

        preview_notice = ""
        if config.show_pointcloud:
            preview_notice = (
                "show_pointcloud requested but Open3D popup preview is disabled in external worker mode; "
                "use RViz topics for preview instead"
            )

        perception = PerceptionResult(
            color_bgr=color_bgr,
            depth_meters=depth_meters,
            segmentation=segmentation,
            scene_points=scene_points,
            pointclouds=pointclouds,
            grasp_groups=grasp_groups,
            scene_grasp_count=scene_grasp_count,
            scene_point_count=int(len(scene_points)) if scene_points is not None else 0,
            object_point_counts=object_point_counts,
            object_centers_camera_m=object_centers_camera_m,
            object_centers_uv=object_centers_uv,
        )
        initial_diagnostics = self._perception_overview_lines(perception, prompt)
        initial_diagnostics.extend(
            "grasp source instance[{instance_index}]: object_points={object_points} "
            "instance_grasps={instance_grasps} scene_mask_grasps={scene_mask_grasps} "
            "merged_grasps={merged_grasps}".format(**item)
            for item in grasp_source_debug
        )
        candidate_pool, diagnostics, max_angle = self.planner.collect_grasp_candidates(
            perception,
            tcp_pose,
            base_to_camera,
            initial_diagnostics=initial_diagnostics,
        )
        if preview_notice:
            diagnostics.append(preview_notice)

        candidate = candidate_pool[0][0] if candidate_pool else None
        plan = self.planner.plan_grasp(candidate, tcp_pose, base_to_camera) if candidate is not None else None
        summary_lines: list[str] = []
        if candidate is None:
            summary_lines.append("no valid grasp candidate found")
            summary_lines.extend(diagnostics)
        else:
            summary_lines.append(f"selected grasp score={candidate.score:.4f}")
            summary_lines.extend(diagnostics[:3])

        return {
            "scene_id": scene_id,
            "prompt": prompt,
            "perception": {
                "scene_id": scene_id,
                "prompt": prompt,
                "camera_frame": str(options.get("camera_frame") or ""),
                "instance_count": int(len(object_point_counts)),
                "scene_grasp_count": int(perception.scene_grasp_count),
                "scene_point_count": int(perception.scene_point_count),
                "object_point_counts": [int(value) for value in object_point_counts],
                "debug_lines": [str(line) for line in diagnostics],
                "scene_points_path": "scene_points.npy" if scene_points is not None else None,
                "object_cloud_paths": object_cloud_paths,
                "grasp_source_debug": grasp_source_debug,
            },
            "candidate_pool": [
                {
                    "candidate": _candidate_to_dict(item[0]),
                    "approach_angle_deg": float(item[1]),
                    "rotation_delta_deg": float(item[2]),
                }
                for item in candidate_pool
            ],
            "candidate": _candidate_to_dict(candidate) if candidate is not None else None,
            "plan": _plan_to_dict(plan) if plan is not None else None,
            "use_pregrasp": bool(config.enable_pregrasp),
            "diagnostics": [str(line) for line in diagnostics],
            "summary": "\n".join(summary_lines),
            "max_approach_angle_deg": float(max_angle),
        }


def _load_request(path: Path) -> dict[str, object]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _save_response(path: Path, payload: dict[str, object]) -> None:
    path.write_text(_json_dumps(payload), encoding="utf-8")


def _handle_one_request(engine: ExternalInferenceEngine, request: dict[str, object]) -> dict[str, object]:
    """Process a single inference request dict and return a response dict."""
    if str(request.get("type") or "") == "warmup":
        options = _parse_options_json(str(request.get("options_json", "")))
        engine.warmup(options)
        return {"success": True, "message": "inference runtime warmed up"}
    work_dir = Path(str(request["work_dir"])).expanduser().resolve()
    color_bgr = np.load(work_dir / str(request["color_npy"]))
    if str(request.get("type") or "") == "detect_target_2d":
        result = engine.detect_target_2d(color_bgr, str(request.get("prompt") or ""))
        return {"success": True, "message": "target detection completed", "result": result}
    if str(request.get("type") or "") == "detect_label_bottles":
        proposals = engine.detect_label_bottles(color_bgr)
        return {
            "success": True,
            "message": f"detected {len(proposals)} bottle proposals",
            "result": {"bottle_proposals": proposals},
        }
    depth_meters = np.load(work_dir / str(request["depth_npy"]))
    camera_info = dict(request["camera_info"])
    intrinsics = SimpleIntrinsics(
        width=int(camera_info["width"]),
        height=int(camera_info["height"]),
        fx=float(camera_info["k"][0]),
        fy=float(camera_info["k"][4]),
        ppx=float(camera_info["k"][2]),
        ppy=float(camera_info["k"][5]),
    )
    tcp_values = list(request["tcp_pose"])
    tcp_pose = EndPoseMMDeg(
        x_mm=float(tcp_values[0]),
        y_mm=float(tcp_values[1]),
        z_mm=float(tcp_values[2]),
        roll_deg=float(tcp_values[3]),
        pitch_deg=float(tcp_values[4]),
        yaw_deg=float(tcp_values[5]),
    )
    base_to_camera = np.asarray(request["base_to_camera"], dtype=np.float64).reshape(4, 4)
    options = _parse_options_json(str(request.get("options_json", "")))
    options["camera_frame"] = str(request.get("camera_frame", ""))

    result = engine.analyze(
        scene_id=str(request["scene_id"]),
        prompt=str(request["prompt"]),
        color_bgr=color_bgr,
        depth_meters=depth_meters,
        intrinsics=intrinsics,
        tcp_pose=tcp_pose,
        base_to_camera=base_to_camera,
        options=options,
        output_dir=work_dir,
    )
    return {"success": True, "message": "analysis completed", "result": result}


def _run_daemon() -> int:
    """Daemon mode: read JSON-line requests from stdin, write JSON-line responses to stdout.

    Protocol:
      - Parent writes one JSON object per line to the process stdin.
      - Worker writes one JSON object per line to stdout for each request.
      - A request with {"type": "ping"} returns {"type": "pong"}.
      - A request with {"type": "shutdown"} causes a clean exit.
      - All other requests are treated as inference requests.
      - stderr is used for diagnostic logging only.
    """
    engine = ExternalInferenceEngine()
    # Use unbuffered binary I/O to avoid line-buffering issues across Python versions.
    stdin = sys.stdin.buffer
    stdout = sys.stdout.buffer

    print("[inference_worker] daemon ready", file=sys.stderr, flush=True)

    while True:
        try:
            raw = stdin.readline()
        except Exception as exc:
            print(f"[inference_worker] stdin read error: {exc}", file=sys.stderr, flush=True)
            break
        if not raw:
            # EOF — parent closed the pipe, exit cleanly.
            break
        line = raw.decode("utf-8", errors="replace").strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            response = {"success": False, "message": f"invalid JSON: {exc}"}
            stdout.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
            stdout.flush()
            continue

        msg_type = str(request.get("type", ""))
        if msg_type == "ping":
            stdout.write(b'{"type":"pong"}\n')
            stdout.flush()
            continue
        if msg_type == "shutdown":
            print("[inference_worker] shutdown requested", file=sys.stderr, flush=True)
            break

        try:
            response = _handle_one_request(engine, request)
        except Exception as exc:
            response = {"success": False, "message": str(exc), "traceback": traceback.format_exc()}
            print(f"[inference_worker] request failed: {exc}", file=sys.stderr, flush=True)

        stdout.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
        stdout.flush()

    print("[inference_worker] exiting", file=sys.stderr, flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-json", default="")
    parser.add_argument("--response-json", default="")
    parser.add_argument("--daemon", action="store_true",
                        help="Run as a persistent daemon reading requests from stdin.")
    args = parser.parse_args()

    if args.daemon:
        return _run_daemon()

    # Legacy single-shot mode (kept for backward compatibility).
    request_path = Path(args.request_json).expanduser().resolve()
    response_path = Path(args.response_json).expanduser().resolve()
    response_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        request = _load_request(request_path)
        work_dir = request_path.parent
        # Inject work_dir so _handle_one_request can find the npy files.
        request["work_dir"] = str(work_dir)
        response = _handle_one_request(ExternalInferenceEngine(), request)
        _save_response(response_path, response)
        return 0
    except Exception as exc:
        _save_response(
            response_path,
            {"success": False, "message": str(exc), "traceback": traceback.format_exc()},
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
