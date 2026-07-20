"""Coordinator for the ROS2 migration that stitches planner and robot hooks."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any, Iterable

import numpy as np
import yaml

from src.grasping.models import (
    EmergencyStopRequested,
    GraspCandidate,
    GraspExecutionConfig,
    GraspPlan,
    PerceptionResult,
)
from src.grasping.planning import PureGraspPlanner
from src.robot.client import RobotArmClient
from src.robot.types import EndPoseMMDeg
from src.utils.calibration import load_camera_to_tcp_transform
from src.utils.transforms import make_transform_xyz_rpy_mm_deg


class GraspPipelineCoordinator:
    """Pipeline shell that wires PureGraspPlanner with robot execution hooks."""

    def __init__(
        self,
        config: GraspExecutionConfig,
        hand_eye_config_path: Path,
        online_bias_path: Path | None = None,
    ):
        self.config = config
        self.hand_eye_config_path = hand_eye_config_path
        self.online_bias_path = online_bias_path
        self.hand_eye = self._load_hand_eye(hand_eye_config_path)
        self.online_bias = self._load_online_bias(online_bias_path)
        self.planner = PureGraspPlanner(config, self.hand_eye, self.online_bias)
        self.robot_client: RobotArmClient | None = None
        self._stop_requested = False
        self._camera: Any | None = None
        self._segmenter: Any | None = None
        self._graspnet: Any | None = None
        self._resolved_graspnet_checkpoint: str | None = None

    def _load_hand_eye(self, config_path: Path) -> np.ndarray:
        with open(config_path, "r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle)
        calib = payload.get("calibration", {})
        transform, _ = load_camera_to_tcp_transform(calib, allow_legacy=False)
        return transform

    def _load_online_bias(self, bias_path: Path | None) -> dict[str, object] | None:
        if bias_path is None:
            return None
        with open(bias_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return payload

    def attach_robot_client(self, robot_client: RobotArmClient) -> None:
        self.robot_client = robot_client

    def ensure_robot(self) -> RobotArmClient:
        if self.robot_client is None:
            raise RuntimeError("Robot client is not attached yet.")
        return self.robot_client

    def request_emergency_stop(self, reason: str = "") -> None:
        if self._stop_requested:
            return
        self._stop_requested = True
        if self.robot_client is None:
            return
        try:
            self.robot_client.emergency_stop()
        except Exception as exc:
            raise RuntimeError(f"Emergency stop failed: {exc}") from exc

    def _ensure_not_stopped(self) -> None:
        if self._stop_requested:
            raise EmergencyStopRequested("Emergency stop is active")

    def _get_camera(self):
        if self._camera is None:
            from src.perception.realsense_rgbd import RealSenseRGBDCamera

            camera = RealSenseRGBDCamera(
                width=self.config.camera_width,
                height=self.config.camera_height,
                fps=self.config.camera_fps,
                clip_max=self.config.clip_max_m,
            )
            camera.start()
            self._camera = camera
        return self._camera

    def _get_segmenter(self):
        if self._segmenter is None:
            from src.perception.yolo_segmenter import YOLOSegmenter

            self._segmenter = YOLOSegmenter(
                device=self.config.grasp_device,
                model_name="yolov8n-seg.pt",
            )
        return self._segmenter

    def _get_graspnet(self):
        if self._graspnet is None:
            from src.perception.graspnet_runner import GraspNetRunner, resolve_graspnet_checkpoint

            resolved_checkpoint = resolve_graspnet_checkpoint(self.config.graspnet_checkpoint)
            if not resolved_checkpoint:
                raise RuntimeError(
                    "GraspNet checkpoint is not configured. "
                    "Pass --graspnet-checkpoint or set GRASPNET_CHECKPOINT."
                )
            self._graspnet = GraspNetRunner(
                checkpoint_path=resolved_checkpoint,
                device=self.config.grasp_device,
                num_point=self.config.grasp_num_point,
                topk=self.config.grasp_topk,
                voxel_size=self.config.grasp_voxel_size,
                collision_thresh=self.config.grasp_collision_thresh,
                approach_dist=self.config.grasp_approach_dist,
            )
            self._resolved_graspnet_checkpoint = resolved_checkpoint
        return self._graspnet

    def _release_perception(self) -> None:
        if self._camera is not None:
            try:
                self._camera.release()
            except Exception:
                pass
            finally:
                self._camera = None
        self._segmenter = None
        self._graspnet = None

    def connect(self) -> None:
        self._ensure_not_stopped()
        robot = self.ensure_robot()
        robot.connect()
        if not robot.enable():
            raise RuntimeError("Failed to enable robot")
        robot.open_gripper(
            open_mm=self.config.gripper_open_mm,
            effort_nm=self.config.gripper_effort_nm,
        )
        try:
            self.move_to_home()
        except Exception as exc:
            current = robot.read_end_pose_mm_deg()
            print(
                "[WARN] connect 阶段自动回 home 失败，保留当前位置继续。"
                f" current=({current.x_mm:.1f}, {current.y_mm:.1f}, {current.z_mm:.1f}, "
                f"{current.roll_deg:.1f}, {current.pitch_deg:.1f}, {current.yaw_deg:.1f})"
                f" cause={exc}"
            )

    def disconnect(self) -> None:
        try:
            if self.robot_client is not None:
                if not self._stop_requested:
                    try:
                        self.move_to_home()
                    except Exception as exc:
                        print(f"[WARN] 回到 home 位失败: {exc}")
                self.robot_client.disconnect()
        finally:
            self._release_perception()

    def current_tcp_pose(self) -> EndPoseMMDeg:
        self._ensure_not_stopped()
        return self.ensure_robot().read_end_pose_mm_deg()

    def current_base_to_camera(self) -> np.ndarray:
        end_pose = self.current_tcp_pose()
        base_to_tcp = make_transform_xyz_rpy_mm_deg(
            xyz_mm=(end_pose.x_mm, end_pose.y_mm, end_pose.z_mm),
            rpy_deg=(end_pose.roll_deg, end_pose.pitch_deg, end_pose.yaw_deg),
        )
        return base_to_tcp @ self.hand_eye

    @staticmethod
    def _segmentation_count(segmentation: dict) -> int:
        masks = segmentation.get("masks", [])
        if hasattr(masks, "shape") and len(masks.shape) >= 3:
            return int(masks.shape[0])
        return int(len(masks))

    def _capture_fused_rgbd(self) -> tuple[bool, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
        camera = self._get_camera()
        frame_count = max(1, int(self.config.depth_fusion_frames))
        if frame_count == 1:
            print("[DEBUG] 深度融合: frames=1 mode=single-frame")
            return camera.get_frames()

        color_bgr = None
        depth_raw = None
        depth_stack: list[np.ndarray] = []
        valid_counts: list[int] = []
        for _ in range(frame_count):
            ok, color_frame, depth_raw_frame, depth_meters_frame = camera.get_frames()
            if not ok or color_frame is None or depth_meters_frame is None:
                continue
            color_bgr = color_frame
            depth_raw = depth_raw_frame
            depth_stack.append(depth_meters_frame.astype(np.float32, copy=False))
            valid_counts.append(int(np.count_nonzero(depth_meters_frame > 0)))

        if not depth_stack or color_bgr is None:
            return False, None, None, None

        if len(depth_stack) == 1:
            print(
                "[DEBUG] 深度融合: "
                f"requested_frames={frame_count} captured_frames=1 mode=single-frame-fallback "
                f"valid={valid_counts[0]}"
            )
            return True, color_bgr, depth_raw, depth_stack[0]

        depth_volume = np.stack(depth_stack, axis=0)
        valid_mask = depth_volume > 0
        depth_for_median = np.where(valid_mask, depth_volume, np.nan)
        fused_depth = np.nanmedian(depth_for_median, axis=0).astype(np.float32)
        fused_depth = np.nan_to_num(fused_depth, nan=0.0, posinf=0.0, neginf=0.0)
        print(
            "[DEBUG] 深度融合: "
            f"requested_frames={frame_count} captured_frames={len(depth_stack)} mode=valid-median "
            f"valid_per_frame={valid_counts} fused_valid={int(np.count_nonzero(fused_depth > 0))}"
        )
        return True, color_bgr, depth_raw, fused_depth

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
        scene_grasp_group,
        mask_np: np.ndarray,
        depth_meters: np.ndarray,
        projection_radius_px: int = 2,
        depth_tolerance_m: float = 0.04,
    ):
        if scene_grasp_group is None or len(scene_grasp_group) == 0:
            return None

        camera = self._get_camera()
        if camera.intrinsics is None:
            return scene_grasp_group

        height, width = mask_np.shape[:2]
        fx = float(camera.intrinsics.fx)
        fy = float(camera.intrinsics.fy)
        cx = float(camera.intrinsics.ppx)
        cy = float(camera.intrinsics.ppy)

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

            local_depth = depth_meters[v, u]
            if local_depth > 0 and abs(float(local_depth) - z) > depth_tolerance_m:
                continue
            keep_indices.append(grasp_index)

        if not keep_indices:
            return None
        return self._get_graspnet().subset_grasp_group(scene_grasp_group, keep_indices)

    def capture_and_perceive(self, text_prompt: str) -> PerceptionResult:
        self._ensure_not_stopped()

        from src.perception.geometry import (
            bilateral_filter_depth,
            depth_frame_to_scene_points_rs,
            depth_to_scene_points,
            keep_largest_point_cluster,
            median_filter_depth,
            remove_radius_outliers,
            save_segmentation_outputs,
            scene_points_to_pointcloud,
            visualize_3d,
        )

        camera = self._get_camera()
        ok, color_bgr, _depth_raw, depth_meters = self._capture_fused_rgbd()
        if not ok or color_bgr is None or depth_meters is None:
            raise RuntimeError("Failed to capture frame from camera")

        filter_mode = self.config.pointcloud_filter_mode
        requested_backend = self.config.pointcloud_backend
        depth_frame, color_frame = camera.get_last_aligned_frames()
        can_use_sdk_backend = (
            requested_backend == "sdk"
            and self.config.depth_fusion_frames == 1
            and filter_mode not in {"bilateral", "median"}
            and depth_frame is not None
        )
        pointcloud_backend = "sdk" if can_use_sdk_backend else "manual"
        if requested_backend == "sdk" and pointcloud_backend != "sdk":
            print(
                "[DEBUG] 点云重建后端: requested=sdk actual=manual "
                f"(depth_fusion_frames={self.config.depth_fusion_frames}, filter_mode={filter_mode})"
            )
        else:
            print(f"[DEBUG] 点云重建后端: requested={requested_backend} actual={pointcloud_backend}")

        if filter_mode == "bilateral":
            depth_before = depth_meters
            depth_meters = bilateral_filter_depth(
                depth_meters,
                diameter=self.config.bilateral_diameter,
                sigma_color=self.config.bilateral_sigma_color,
                sigma_space=self.config.bilateral_sigma_space,
            )
            valid_before = int(np.count_nonzero(depth_before > 0))
            valid_after = int(np.count_nonzero(depth_meters > 0))
            print(
                "[DEBUG] 点云过滤: "
                f"mode=bilateral d={self.config.bilateral_diameter} "
                f"sigma_color={self.config.bilateral_sigma_color:.4f} "
                f"sigma_space={self.config.bilateral_sigma_space:.2f} "
                f"valid_before={valid_before} valid_after={valid_after}"
            )
        elif filter_mode == "median":
            depth_before = depth_meters
            depth_meters = median_filter_depth(
                depth_meters,
                kernel_size=self.config.median_kernel_size,
            )
            valid_before = int(np.count_nonzero(depth_before > 0))
            valid_after = int(np.count_nonzero(depth_meters > 0))
            print(
                "[DEBUG] 点云过滤: "
                f"mode=median kernel={self.config.median_kernel_size} "
                f"valid_before={valid_before} valid_after={valid_after}"
            )
        elif filter_mode == "island":
            print(
                "[DEBUG] 点云过滤: "
                f"mode=island eps={self.config.island_eps_m:.3f} "
                f"min_points={self.config.island_min_points}"
            )
        elif filter_mode == "radius":
            print(
                "[DEBUG] 点云过滤: "
                f"mode=radius nb_points={self.config.radius_nb_points} "
                f"radius={self.config.radius_m:.3f}"
            )
        else:
            print("[DEBUG] 点云过滤: mode=none")

        segmentation = self._get_segmenter().segment_text(color_bgr, text_prompt)
        pointclouds = save_segmentation_outputs(
            color_bgr=color_bgr,
            depth_meters=depth_meters,
            masks=segmentation["masks"],
            scores=segmentation["scores"],
            boxes=segmentation["boxes"],
            text_prompt=text_prompt,
            intrinsics=camera.intrinsics,
            clip_max=self.config.clip_max_m,
            output_dir=self.config.output_dir,
            pointcloud_filter_mode=filter_mode,
            island_eps_m=self.config.island_eps_m,
            island_min_points=self.config.island_min_points,
            radius_nb_points=self.config.radius_nb_points,
            radius_m=self.config.radius_m,
            pointcloud_backend=pointcloud_backend,
            depth_frame=(depth_frame if pointcloud_backend == "sdk" else None),
            color_frame=(color_frame if pointcloud_backend == "sdk" else None),
        )

        if pointcloud_backend == "sdk":
            scene_points = depth_frame_to_scene_points_rs(
                depth_frame=depth_frame,
                color_frame=color_frame,
                clip_max=self.config.clip_max_m,
                mask=None,
            )
        else:
            scene_points = depth_to_scene_points(
                depth_meters,
                camera.intrinsics,
                clip_max=self.config.clip_max_m,
                mask=None,
            )

        scene_grasp_group = self._get_graspnet().predict(
            scene_points=scene_points,
            object_points=scene_points,
        )
        scene_grasp_count = int(len(scene_grasp_group)) if scene_grasp_group is not None else 0
        print(f"[DEBUG] 全场景 GraspNet 候选数: {scene_grasp_count}")

        grasp_groups = []
        object_point_counts: list[int] = []
        object_centers_camera_m: list[tuple[float, float, float] | None] = []
        object_centers_uv: list[tuple[int, int] | None] = []
        masks = segmentation["masks"]
        count = self._segmentation_count(segmentation)
        for index in range(count):
            mask = masks[index].squeeze()
            mask_np = (
                mask.detach().cpu().numpy().astype(bool)
                if hasattr(mask, "detach")
                else np.asarray(mask).astype(bool)
            )
            if pointcloud_backend == "sdk":
                object_points = depth_frame_to_scene_points_rs(
                    depth_frame=depth_frame,
                    color_frame=color_frame,
                    clip_max=self.config.clip_max_m,
                    mask=mask_np,
                )
            else:
                object_points = depth_to_scene_points(
                    depth_meters,
                    camera.intrinsics,
                    clip_max=self.config.clip_max_m,
                    mask=mask_np,
                )

            if filter_mode == "island" and object_points is not None and len(object_points) > 0:
                filtered_points, _ = keep_largest_point_cluster(
                    object_points,
                    eps_m=self.config.island_eps_m,
                    min_points=self.config.island_min_points,
                )
                object_points = filtered_points.astype(np.float32, copy=False)
            elif filter_mode == "radius" and object_points is not None and len(object_points) > 0:
                filtered_points, _ = remove_radius_outliers(
                    object_points,
                    nb_points=self.config.radius_nb_points,
                    radius_m=self.config.radius_m,
                )
                object_points = filtered_points.astype(np.float32, copy=False)

            object_point_counts.append(int(len(object_points)) if object_points is not None else 0)
            if pointclouds[index] is not None and len(pointclouds[index].points) > 0:
                center = np.asarray(pointclouds[index].get_center(), dtype=np.float64).reshape(3)
                object_centers_camera_m.append((float(center[0]), float(center[1]), float(center[2])))
            else:
                object_centers_camera_m.append(None)
            object_centers_uv.append(self._mask_centroid_uv(mask_np))
            grasp_groups.append(
                self._filter_scene_grasps_by_mask(
                    scene_grasp_group=scene_grasp_group,
                    mask_np=mask_np,
                    depth_meters=depth_meters,
                )
            )

        if self.config.show_pointcloud:
            try:
                scene_pcd = scene_points_to_pointcloud(scene_points)
                preview_pointclouds = ([scene_pcd] if scene_pcd is not None else []) + [
                    pcd for pcd in pointclouds if pcd is not None
                ]
                preview_grasp_groups = [
                    group for group in grasp_groups if group is not None and len(group) > 0
                ]
                if preview_grasp_groups:
                    print(
                        "[DEBUG] 点云可视化: "
                        f"showing filtered instance grasp groups={len(preview_grasp_groups)} "
                        f"topk={self.config.preview_grasp_topk}"
                    )
                elif scene_grasp_group is not None and len(scene_grasp_group) > 0:
                    preview_grasp_groups = [scene_grasp_group]
                    print(
                        "[DEBUG] 点云可视化: "
                        f"no filtered instance grasp, fallback to scene grasps count={len(scene_grasp_group)} "
                        f"topk={self.config.preview_grasp_topk}"
                    )
                else:
                    print("[DEBUG] 点云可视化: no grasp geometry available, showing pointcloud only")
                visualize_3d(
                    preview_pointclouds,
                    text_prompt=f"{text_prompt} reconstruction",
                    grasp_groups=preview_grasp_groups,
                    grasp_topk_vis=self.config.preview_grasp_topk,
                )
            except Exception as exc:
                print(f"[WARN] show_pointcloud failed: {exc}")

        return PerceptionResult(
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

    def _choose_best_center_target(
        self,
        perception: PerceptionResult,
    ) -> tuple[int, tuple[int, int], tuple[float, float, float]] | None:
        best_index = None
        best_score = None
        scores = perception.segmentation["scores"]
        for index, center_uv in enumerate(perception.object_centers_uv):
            center_cam = perception.object_centers_camera_m[index]
            if center_uv is None or center_cam is None:
                continue
            score = scores[index].item() if hasattr(scores[index], "item") else float(scores[index])
            if best_index is None or score > best_score:
                best_index = index
                best_score = score
        if best_index is None:
            return None
        return (
            best_index,
            perception.object_centers_uv[best_index],
            perception.object_centers_camera_m[best_index],
        )

    def _plan_centering_move(
        self,
        target_uv: tuple[int, int],
        target_depth_m: float,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float], dict[str, float]]:
        camera = self._get_camera()
        if camera.intrinsics is None:
            raise RuntimeError("Camera intrinsics are not ready for centering")

        tcp = self.current_tcp_pose()
        current_pos_m = np.array(
            [tcp.x_mm / 1000.0, tcp.y_mm / 1000.0, tcp.z_mm / 1000.0],
            dtype=np.float64,
        )
        current_rpy = (tcp.roll_deg, tcp.pitch_deg, tcp.yaw_deg)

        image_center_u = camera.width / 2.0
        image_center_v = camera.height / 2.0
        error_u = float(target_uv[0] - image_center_u)
        error_v = float(target_uv[1] - image_center_v)

        delta_camera = np.array(
            [
                (error_u / camera.intrinsics.fx) * target_depth_m,
                (error_v / camera.intrinsics.fy) * target_depth_m,
                0.0,
            ],
            dtype=np.float64,
        )
        step_norm = float(np.linalg.norm(delta_camera[:2]))
        if step_norm > self.config.center_max_step_m and step_norm > 1e-9:
            delta_camera *= self.config.center_max_step_m / step_norm

        base_to_camera = self.current_base_to_camera()
        delta_base = base_to_camera[:3, :3] @ delta_camera
        target_pos = current_pos_m + delta_base
        return (
            (float(target_pos[0]), float(target_pos[1]), float(target_pos[2])),
            current_rpy,
            {
                "error_u_px": error_u,
                "error_v_px": error_v,
                "delta_base_x_m": float(delta_base[0]),
                "delta_base_y_m": float(delta_base[1]),
            },
        )

    def _move_to_center_target(self, text_prompt: str) -> list[str]:
        logs: list[str] = []
        camera = self._get_camera()
        for iteration in range(self.config.center_max_iterations):
            self._ensure_not_stopped()
            perception = self.capture_and_perceive(text_prompt)
            chosen = self._choose_best_center_target(perception)
            if chosen is None:
                logs.append(f"centering iter {iteration + 1}: no valid target with depth")
                break
            index, center_uv, center_cam = chosen
            image_center = (int(camera.width // 2), int(camera.height // 2))
            pixel_error = (center_uv[0] - image_center[0], center_uv[1] - image_center[1])
            logs.append(
                f"centering iter {iteration + 1}: instance={index} pixel_error=({pixel_error[0]:+d}, {pixel_error[1]:+d}) "
                f"cam_center=({center_cam[0]:.3f}, {center_cam[1]:.3f}, {center_cam[2]:.3f})"
            )
            if (
                abs(pixel_error[0]) <= self.config.center_pixel_tolerance
                and abs(pixel_error[1]) <= self.config.center_pixel_tolerance
            ):
                logs.append("centering complete: target already near image center")
                break

            target_pose_m, target_rpy_deg, debug = self._plan_centering_move(center_uv, center_cam[2])
            ok, violations = self.planner.check_workspace(target_pose_m)
            if not ok:
                logs.append("centering blocked by workspace: " + " | ".join(violations))
                break
            if self.config.dry_run:
                logs.append(
                    "centering dry-run: "
                    f"move_to=({target_pose_m[0]:.3f}, {target_pose_m[1]:.3f}, {target_pose_m[2]:.3f}) "
                    f"delta=({debug['delta_base_x_m']:+.3f}, {debug['delta_base_y_m']:+.3f})"
                )
                break

            self._move_to_pose(target_pose_m, target_rpy_deg)
            time.sleep(self.config.center_settle_time_s)
        return logs

    def collect_candidates(
        self,
        perception: PerceptionResult,
        tcp_pose: EndPoseMMDeg,
        base_to_camera: np.ndarray,
    ) -> tuple[list[tuple[GraspCandidate, float, float]], list[str], float]:
        return self.planner.collect_grasp_candidates(perception, tcp_pose, base_to_camera)

    def _perception_overview_lines(
        self,
        perception: PerceptionResult,
        text_prompt: str,
    ) -> list[str]:
        instance_count = self._segmentation_count(perception.segmentation)
        lines = [
            f"perception overview: prompt={text_prompt!r} "
            f"instances={instance_count} scene_grasps={perception.scene_grasp_count} "
            f"scene_points={perception.scene_point_count}"
        ]
        if instance_count == 0:
            lines.append("segmentation produced 0 instances for the current prompt")
        return lines

    def select_best_grasp(
        self,
        perception: PerceptionResult,
    ) -> tuple[GraspCandidate, list[tuple[GraspCandidate, float, float]], list[str], float]:
        tcp_pose = self.current_tcp_pose()
        base_to_camera = self.current_base_to_camera()
        candidate_pool, diagnostics, max_angle = self.collect_candidates(
            perception,
            tcp_pose,
            base_to_camera,
        )
        if not candidate_pool:
            raise RuntimeError("No valid grasp candidate found. details: " + " | ".join(diagnostics))
        return candidate_pool[0][0], candidate_pool, diagnostics, max_angle

    def plan_grasp(
        self,
        candidate: GraspCandidate,
        tcp_pose: EndPoseMMDeg,
        base_to_camera: np.ndarray,
    ) -> GraspPlan:
        return self.planner.plan_grasp(candidate, tcp_pose, base_to_camera)

    def plan_grasp_for_current_state(self, candidate: GraspCandidate) -> GraspPlan:
        return self.plan_grasp(candidate, self.current_tcp_pose(), self.current_base_to_camera())

    def summarize_plan(self, plan: GraspPlan, candidate: GraspCandidate) -> Iterable[str]:
        lines = [
            f"candidate score: {candidate.score:.4f}",
            f"target (m): {tuple(round(v, 3) for v in plan.target_base_m)}",
            f"grasp rotation (deg): {tuple(round(v, 2) for v in plan.target_rpy_deg)}",
            f"workspace pass: {plan.within_workspace}",
        ]
        if plan.workspace_violations:
            lines.append("violations: " + " | ".join(plan.workspace_violations))
        return lines

    def summarize_plan_text(self, candidate: GraspCandidate, plan: GraspPlan) -> str:
        tcp = self.current_tcp_pose()
        lines = [
            f"best grasp score: {candidate.score:.4f}",
            f"camera grasp translation m: {candidate.translation_camera_m}",
            (
                "object center camera m: "
                + (
                    str(tuple(round(v, 4) for v in candidate.object_center_camera_m))
                    if candidate.object_center_camera_m is not None
                    else "None"
                )
            ),
            (
                f"grasp-object center offset m: {candidate.center_offset_m:.4f}"
                if candidate.center_offset_m is not None
                else "grasp-object center offset m: None"
            ),
            f"current tcp mm/deg: ({tcp.x_mm:.2f}, {tcp.y_mm:.2f}, {tcp.z_mm:.2f}, "
            f"{tcp.roll_deg:.2f}, {tcp.pitch_deg:.2f}, {tcp.yaw_deg:.2f})",
            f"target base m: {plan.target_base_m}",
            f"target rpy deg: {tuple(round(v, 2) for v in plan.target_rpy_deg)}",
            (
                f"pregrasp base m: {plan.pregrasp_base_m}"
                if self.config.enable_pregrasp
                else "pregrasp: disabled"
            ),
            f"grasp base m: {plan.grasp_base_m}",
            f"retreat base m: {plan.retreat_base_m}",
            f"within workspace: {plan.within_workspace}",
        ]
        if plan.workspace_violations:
            lines.append("workspace violations: " + " | ".join(plan.workspace_violations))
        return "\n".join(lines)

    def _format_candidate_pose_debug(
        self,
        candidate: GraspCandidate,
        angle_deg: float,
        rotation_delta_deg: float,
    ) -> str:
        plan = self.plan_grasp_for_current_state(candidate)
        target_rpy_deg = tuple(round(v, 2) for v in plan.target_rpy_deg)
        target_base_m = tuple(round(v, 4) for v in plan.target_base_m)
        pregrasp_base_m = tuple(round(v, 4) for v in plan.pregrasp_base_m)
        center_offset = None if candidate.center_offset_m is None else round(candidate.center_offset_m, 4)
        return (
            f"score={candidate.score:.4f} "
            f"instance={candidate.instance_index} "
            f"cam_t=({candidate.translation_camera_m[0]:.4f}, {candidate.translation_camera_m[1]:.4f}, {candidate.translation_camera_m[2]:.4f}) "
            f"target_rpy={target_rpy_deg} "
            f"target_base={target_base_m} "
            f"pregrasp_base={pregrasp_base_m} "
            f"approach_angle={angle_deg:.1f}deg "
            f"rotation_delta={rotation_delta_deg:.1f}deg "
            f"center_offset={center_offset}"
        )

    def _arm_status_debug_string(self) -> str:
        return self.ensure_robot().format_arm_status()

    def _move_to_pose(
        self,
        translation_m: tuple[float, float, float],
        rpy_deg: tuple[float, float, float],
        command_time_scale: float = 1.0,
        wait_timeout_scale: float = 1.0,
        min_command_time_s: float | None = None,
        min_wait_timeout_s: float | None = None,
    ) -> EndPoseMMDeg:
        self._ensure_not_stopped()
        robot = self.ensure_robot()
        target = EndPoseMMDeg(
            x_mm=translation_m[0] * 1000.0,
            y_mm=translation_m[1] * 1000.0,
            z_mm=translation_m[2] * 1000.0,
            roll_deg=rpy_deg[0],
            pitch_deg=rpy_deg[1],
            yaw_deg=rpy_deg[2],
        )
        start_pose = self.current_tcp_pose()
        command_time_s = self.config.command_time_s * command_time_scale
        if min_command_time_s is not None:
            command_time_s = max(command_time_s, min_command_time_s)
        wait_timeout_s = self.config.move_check_timeout_s * wait_timeout_scale
        if min_wait_timeout_s is not None:
            wait_timeout_s = max(wait_timeout_s, min_wait_timeout_s)

        robot.move_end_pose_mm_deg(
            x_mm=target.x_mm,
            y_mm=target.y_mm,
            z_mm=target.z_mm,
            roll_deg=target.roll_deg,
            pitch_deg=target.pitch_deg,
            yaw_deg=target.yaw_deg,
            speed_percent=self.config.robot_speed_percent,
        )
        reached, actual, target_error = robot.wait_until_pose_reached(
            target=target,
            timeout_s=wait_timeout_s,
            pos_tolerance_mm=self.config.move_pos_tolerance_mm,
            rot_tolerance_deg=self.config.move_rot_tolerance_deg,
        )
        if reached:
            time.sleep(self.config.settle_time_s)
        self._ensure_not_stopped()
        actual = self.current_tcp_pose()
        motion_delta = robot.pose_error(start_pose, actual)
        target_error = robot.pose_error(target, actual)
        if (
            motion_delta["dpos_mm"] < 5.0
            and motion_delta["drot_deg"] < 3.0
            and (target_error["dpos_mm"] > 30.0 or target_error["drot_deg"] > 15.0)
        ):
            raise RuntimeError(
                "Robot pose command did not take effect. "
                f"status={self._arm_status_debug_string()}"
            )
        if not reached:
            raise RuntimeError(
                "Robot did not reach target pose before timeout. "
                f"target_error_pos={target_error['dpos_mm']:.2f}mm "
                f"target_error_rot={target_error['drot_deg']:.2f}deg "
                f"status={self._arm_status_debug_string()}"
            )
        return actual

    def move_to_home(self) -> EndPoseMMDeg | None:
        pose = self.config.home_pose_mm_deg
        if pose is None:
            return None
        self._set_gripper_open()
        return self._move_to_pose(
            translation_m=(pose[0] / 1000.0, pose[1] / 1000.0, pose[2] / 1000.0),
            rpy_deg=(pose[3], pose[4], pose[5]),
            command_time_scale=2.0,
            wait_timeout_scale=3.0,
            min_command_time_s=5.0,
            min_wait_timeout_s=12.0,
        )

    def move_to_observation_pose(self) -> str | None:
        pose = self.config.observe_pose_mm_deg
        if pose is None:
            return None
        self._set_gripper_open()
        actual = self._move_to_pose(
            translation_m=(pose[0] / 1000.0, pose[1] / 1000.0, pose[2] / 1000.0),
            rpy_deg=(pose[3], pose[4], pose[5]),
        )
        return (
            "moved to observation pose mm/deg: "
            f"target=({pose[0]:.2f}, {pose[1]:.2f}, {pose[2]:.2f}, {pose[3]:.2f}, {pose[4]:.2f}, {pose[5]:.2f}) "
            f"actual=({actual.x_mm:.2f}, {actual.y_mm:.2f}, {actual.z_mm:.2f}, "
            f"{actual.roll_deg:.2f}, {actual.pitch_deg:.2f}, {actual.yaw_deg:.2f})"
        )

    def move_to_handoff_pose(self) -> EndPoseMMDeg | None:
        pose = self.config.handoff_pose_mm_deg
        if pose is None:
            return None
        return self._move_to_pose(
            translation_m=(pose[0] / 1000.0, pose[1] / 1000.0, pose[2] / 1000.0),
            rpy_deg=(pose[3], pose[4], pose[5]),
            command_time_scale=1.5,
        )

    def _set_gripper_open(self) -> None:
        self._ensure_not_stopped()
        robot = self.ensure_robot()
        target_mm = self.config.gripper_open_mm
        robot.open_gripper(open_mm=target_mm, effort_nm=self.config.gripper_effort_nm)
        success = robot.wait_for_gripper(target_mm=target_mm, tol_mm=5.0, timeout_s=4.0)
        if not success:
            raise RuntimeError(f"Unable to open gripper to {target_mm:.2f}mm")

    def _set_gripper_closed(self) -> None:
        self._ensure_not_stopped()
        robot = self.ensure_robot()
        target_effort_nm = self.config.grasp_close_effort_nm
        robot.close_gripper(effort_nm=target_effort_nm)
        success = robot.wait_for_gripper_effort(
            target_effort_nm=target_effort_nm,
            timeout_s=self.config.gripper_close_timeout_s,
        )
        status = robot.get_gripper_status()
        if not success:
            raise RuntimeError(
                f"Gripper did not reach target effort {target_effort_nm:.2f}Nm: "
                f"angle={status.angle_mm:.2f} effort={status.effort_nm:.2f}"
            )

    def execute_grasp_plan(self, plan: GraspPlan) -> dict[str, object]:
        self._ensure_not_stopped()
        if not plan.within_workspace:
            raise RuntimeError(
                "Planned grasp target is outside workspace: "
                + "; ".join(plan.workspace_violations)
            )

        self._set_gripper_open()
        pregrasp_pose = None
        if self.config.enable_pregrasp:
            pregrasp_pose = self._move_to_pose(
                plan.pregrasp_base_m,
                plan.target_rpy_deg,
                command_time_scale=1.5,
                wait_timeout_scale=2.0,
                min_command_time_s=2.5,
                min_wait_timeout_s=5.0,
            )

        grasp_pose = self._move_to_pose(
            plan.grasp_base_m,
            plan.target_rpy_deg,
            command_time_scale=1.5,
            wait_timeout_scale=4.0,
            min_command_time_s=2.5,
            min_wait_timeout_s=8.0,
        )
        target_pose = self._move_to_pose(
            plan.target_base_m,
            plan.target_rpy_deg,
            command_time_scale=1.5,
            wait_timeout_scale=4.0,
            min_command_time_s=2.5,
            min_wait_timeout_s=8.0,
        )
        self._set_gripper_closed()

        retreat_pose = self._move_to_pose(
            plan.retreat_base_m,
            plan.target_rpy_deg,
            command_time_scale=1.5,
            wait_timeout_scale=3.0,
            min_command_time_s=2.5,
            min_wait_timeout_s=6.0,
        )
        handoff_pose = self.move_to_handoff_pose() if self.config.handoff_pose_mm_deg is not None else None
        self._set_gripper_open()
        home_pose = self.move_to_home() if self.config.home_pose_mm_deg is not None else None

        return {
            "pregrasp_pose": pregrasp_pose,
            "grasp_pose": grasp_pose,
            "target_pose": target_pose,
            "retreat_pose": retreat_pose,
            "handoff_pose": handoff_pose,
            "home_pose": home_pose,
            "candidate_score": plan.candidate.score,
            "target_base_m": plan.target_base_m,
            "target_rpy_deg": plan.target_rpy_deg,
        }

    def request_execution_confirmation(self) -> bool:
        self._ensure_not_stopped()
        answer = input("Type 'grasp' to execute the grasp, or press Enter to cancel: ").strip().lower()
        return answer == "grasp"

    def preview_grasp(
        self,
        perception: PerceptionResult,
        candidate: GraspCandidate,
        text_prompt: str,
    ) -> None:
        from src.perception.geometry import visualize_3d

        preview_pointclouds = []
        preview_grasp_groups = []
        if 0 <= candidate.instance_index < len(perception.pointclouds):
            preview_pointclouds.append(perception.pointclouds[candidate.instance_index])
        if 0 <= candidate.instance_index < len(perception.grasp_groups):
            preview_grasp_groups.append(perception.grasp_groups[candidate.instance_index])
        if not preview_pointclouds and not preview_grasp_groups:
            return
        try:
            visualize_3d(
                preview_pointclouds,
                text_prompt=f"{text_prompt} grasp preview",
                grasp_groups=preview_grasp_groups,
                grasp_topk_vis=self.config.preview_grasp_topk,
            )
        except Exception as exc:
            print(f"[WARN] preview_grasp failed: {exc}")

    def run_once_from_perception(
        self,
        text_prompt: str,
        perception: PerceptionResult,
        observation_log: str | None = None,
        centering_logs: list[str] | None = None,
    ) -> dict[str, object]:
        self._ensure_not_stopped()
        centering_logs = list(centering_logs or [])
        tcp_pose = self.current_tcp_pose()
        base_to_camera = self.current_base_to_camera()
        candidate_pool, diagnostics, max_angle = self.collect_candidates(
            perception,
            tcp_pose,
            base_to_camera,
        )
        if not diagnostics:
            diagnostics = self._perception_overview_lines(perception, text_prompt)

        if not candidate_pool:
            summary_lines: list[str] = []
            if observation_log:
                summary_lines.append(observation_log)
            if centering_logs:
                summary_lines.extend(centering_logs)
            summary_lines.append("no valid grasp candidate found")
            summary_lines.extend(diagnostics)
            return {
                "status": "no_candidate",
                "text_prompt": text_prompt,
                "perception": perception,
                "base_to_camera": base_to_camera,
                "use_pregrasp": bool(self.config.enable_pregrasp),
                "candidate": None,
                "candidate_pool": [],
                "candidate_preview_lines": [],
                "diagnostics": diagnostics,
                "max_approach_angle_deg": max_angle,
                "plan": None,
                "execution": None,
                "confirmed": None,
                "observation_log": observation_log,
                "centering_logs": centering_logs,
                "summary": "\n".join(summary_lines),
            }

        candidate = candidate_pool[0][0]
        plan = self.plan_grasp(candidate, tcp_pose, base_to_camera)
        candidate_preview_lines = [
            self._format_candidate_pose_debug(candidate_preview, angle_deg, rotation_delta_deg)
            for candidate_preview, angle_deg, rotation_delta_deg in candidate_pool[:3]
        ]

        execution = None
        confirmed: bool | None = None
        if not self.config.dry_run:
            self.preview_grasp(perception, candidate, text_prompt)
            if not plan.within_workspace:
                confirmed = False
            else:
                confirmed = (
                    self.request_execution_confirmation()
                    if self.config.confirm_before_execute
                    else True
                )

            if confirmed:
                last_error: Exception | None = None
                for candidate_option, _angle_deg, _rot_delta in candidate_pool:
                    candidate = candidate_option
                    plan = self.plan_grasp_for_current_state(candidate)
                    try:
                        execution = self.execute_grasp_plan(plan)
                        break
                    except RuntimeError as exec_error:
                        last_error = exec_error
                        if "ANGLE_LIMIT" in str(exec_error):
                            continue
                        raise
                if execution is None and last_error is not None:
                    raise last_error

        summary_lines: list[str] = []
        if observation_log:
            summary_lines.append(observation_log)
        if centering_logs:
            summary_lines.extend(centering_logs)
        summary_lines.append(self.summarize_plan_text(candidate, plan))
        if not self.config.dry_run and confirmed is False:
            summary_lines.append("execution cancelled by user")

        return {
            "status": "ok",
            "text_prompt": text_prompt,
            "perception": perception,
            "base_to_camera": base_to_camera,
            "use_pregrasp": bool(self.config.enable_pregrasp),
            "candidate": candidate,
            "candidate_pool": candidate_pool,
            "candidate_preview_lines": candidate_preview_lines,
            "diagnostics": diagnostics,
            "max_approach_angle_deg": max_angle,
            "plan": plan,
            "execution": execution,
            "confirmed": confirmed,
            "observation_log": observation_log,
            "centering_logs": centering_logs,
            "summary": "\n".join(summary_lines),
        }

    def run_once_with_inputs(
        self,
        text_prompt: str,
        perception: PerceptionResult,
        observation_log: str | None = None,
        centering_logs: list[str] | None = None,
    ) -> dict[str, object]:
        """Compatibility alias for the current external-perception entrypoint."""
        return self.run_once_from_perception(
            text_prompt=text_prompt,
            perception=perception,
            observation_log=observation_log,
            centering_logs=centering_logs,
        )

    def run_once(self, text_prompt: str) -> dict[str, object]:
        self._ensure_not_stopped()

        observation_log = self.move_to_observation_pose()
        centering_logs: list[str] = []
        if self.config.precenter_before_grasp:
            centering_logs = self._move_to_center_target(text_prompt)
        perception = self.capture_and_perceive(text_prompt)
        return self.run_once_from_perception(
            text_prompt=text_prompt,
            perception=perception,
            observation_log=observation_log,
            centering_logs=centering_logs,
        )

    def describe_environment(self) -> Iterable[str]:
        yield f"hand-eye config: {self.hand_eye_config_path}"
        if self.online_bias_path:
            yield f"online bias: {self.online_bias_path}"
        yield f"GraspNet checkpoint: {self.config.graspnet_checkpoint}"
        if self.robot_client is None:
            yield "planner is ready, robot client awaiting attachment"
        else:
            yield f"planner is ready, robot client attached: {type(self.robot_client).__name__}"
