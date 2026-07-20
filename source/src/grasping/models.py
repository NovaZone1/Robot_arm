"""Shared grasping data models for the ROS2 migration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class EmergencyStopRequested(RuntimeError):
    """Raised when an emergency stop has been requested."""


@dataclass(slots=True)
class GraspExecutionConfig:
    hand_eye_config_path: str
    graspnet_checkpoint: str = ""
    camera_width: int = 640
    camera_height: int = 480
    camera_fps: int = 30
    clip_max_m: float = 3.0
    depth_fusion_frames: int = 1
    pointcloud_filter_mode: str = "bilateral"
    pointcloud_backend: str = "sdk"
    bilateral_diameter: int = 5
    bilateral_sigma_color: float = 0.02
    bilateral_sigma_space: float = 5.0
    median_kernel_size: int = 5
    island_eps_m: float = 0.02
    island_min_points: int = 30
    radius_nb_points: int = 12
    radius_m: float = 0.02
    grasp_device: str = "cuda"
    grasp_num_point: int = 20000
    grasp_topk: int = 30
    grasp_voxel_size: float = 0.01
    grasp_collision_thresh: float = 0.01
    grasp_approach_dist: float = 0.05
    robot_can_name: str = "can0"
    robot_speed_percent: int = 40
    home_pose_mm_deg: tuple[float, float, float, float, float, float] | None = (
        57.0,
        0.0,
        215.0,
        0.0,
        85.0,
        0.0,
    )
    handoff_pose_mm_deg: tuple[float, float, float, float, float, float] | None = (
        200.0,
        20.0,
        300.0,
        10.0,
        120.0,
        0.0,
    )
    observe_pose_mm_deg: tuple[float, float, float, float, float, float] | None = (
        30.0,
        0.0,
        400.0,
        0.0,
        120.0,
        0.0,
    )
    precenter_before_grasp: bool = False
    confirm_before_execute: bool = True
    preview_grasp_topk: int = 5
    center_pixel_tolerance: int = 20
    center_max_step_m: float = 0.03
    center_max_iterations: int = 5
    center_settle_time_s: float = 0.6
    max_grasp_center_offset_m: float = 0.35
    safe_top_down_candidate_filter: bool = False
    min_grasp_score: float = 0.01
    flat_object_mode: bool = False
    max_reachable_rotation_delta_deg: float = 180.0
    allow_180deg_equivalent_grasp: bool = True
    wrist_roll_search_deg: tuple[float, ...] = (-90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0)
    gripper_length_m: float = 0.105
    tool_contact_offset_tool_m: tuple[float, float, float] | None = None
    grasp_y_bias_mm: float | None = 0.0
    enable_pregrasp: bool = False
    pregrasp_offset_m: float = 0.0
    descend_offset_m: float = 0.0
    grasp_z_offset_m: float = 0.0
    retreat_offset_m: float = 0.0
    table_z_m: float = 0.0
    min_gripper_table_clearance_m: float = 0.03
    gripper_open_mm: float = 70.0
    gripper_effort_nm: float = 1.5
    grasp_close_effort_nm: float = 0.6
    gripper_close_timeout_s: float = 6.0
    online_bias_enabled: bool = False
    online_bias_path: str | None = None
    command_time_s: float = 2.0
    command_interval_s: float = 0.01
    settle_time_s: float = 1.0
    move_check_timeout_s: float = 3.0
    move_pos_tolerance_mm: float = 15.0
    move_rot_tolerance_deg: float = 8.0
    output_dir: str = "output"
    show_pointcloud: bool = False
    dry_run: bool = True
    max_approach_angle_deg: float = 180.0
    workspace_x_limits_m: tuple[float, float] = (0.10, 1.20)
    workspace_y_limits_m: tuple[float, float] = (-0.50, 0.50)
    workspace_z_limits_m: tuple[float, float] = (0.00, 0.60)


@dataclass(slots=True)
class PerceptionResult:
    color_bgr: np.ndarray
    depth_meters: np.ndarray
    segmentation: dict
    scene_points: np.ndarray | None
    pointclouds: list
    grasp_groups: list
    scene_grasp_count: int
    scene_point_count: int
    object_point_counts: list[int]
    object_centers_camera_m: list[tuple[float, float, float] | None]
    object_centers_uv: list[tuple[int, int] | None]


@dataclass(slots=True)
class GraspCandidate:
    instance_index: int
    score: float
    width_m: float
    depth_m: float
    translation_camera_m: tuple[float, float, float]
    rotation_camera: np.ndarray
    object_center_camera_m: tuple[float, float, float] | None
    center_offset_m: float | None
    raw_grasp: object


@dataclass(slots=True)
class GraspPlan:
    candidate: GraspCandidate
    target_base_m: tuple[float, float, float]
    target_rpy_deg: tuple[float, float, float]
    pregrasp_base_m: tuple[float, float, float]
    grasp_base_m: tuple[float, float, float]
    retreat_base_m: tuple[float, float, float]
    within_workspace: bool
    workspace_violations: list[str]
    target_contact_point_base_m: tuple[float, float, float] | None = None
    tool_contact_offset_tool_m: tuple[float, float, float] | None = None
