from __future__ import annotations

from datetime import datetime
import os

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
import torch

COLORS = [
    (0, 255, 0),
    (255, 100, 0),
    (0, 0, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (128, 255, 0),
    (255, 128, 0),
]


_RS_POINTCLOUD = rs.pointcloud()


def bilateral_filter_depth(
    depth_meters: np.ndarray,
    diameter: int = 5,
    sigma_color: float = 0.02,
    sigma_space: float = 5.0,
) -> np.ndarray:
    depth = depth_meters.astype(np.float32, copy=False)
    valid_mask = depth > 0
    if not np.any(valid_mask):
        return depth.copy()

    filled = depth.copy()
    if np.any(~valid_mask):
        filled[~valid_mask] = 0.0

    filtered = cv2.bilateralFilter(
        filled,
        d=max(1, int(diameter)),
        sigmaColor=float(sigma_color),
        sigmaSpace=float(sigma_space),
    )
    filtered[~valid_mask] = 0.0
    return filtered.astype(np.float32, copy=False)


def median_filter_depth(
    depth_meters: np.ndarray,
    kernel_size: int = 5,
) -> np.ndarray:
    depth = depth_meters.astype(np.float32, copy=False)
    valid_mask = depth > 0
    if not np.any(valid_mask):
        return depth.copy()

    kernel = max(1, int(kernel_size))
    if kernel % 2 == 0:
        kernel += 1
    filtered = cv2.medianBlur(depth, kernel)
    filtered[~valid_mask] = 0.0
    return filtered.astype(np.float32, copy=False)


def keep_largest_point_cluster(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    eps_m: float = 0.02,
    min_points: int = 30,
) -> tuple[np.ndarray, np.ndarray | None]:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return pts, colors
    if pts.shape[0] < max(10, min_points):
        return pts, colors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    labels = np.asarray(
        pcd.cluster_dbscan(eps=float(eps_m), min_points=int(min_points), print_progress=False),
        dtype=np.int32,
    )
    valid_labels = labels[labels >= 0]
    if valid_labels.size == 0:
        return pts, colors

    largest_label = int(np.bincount(valid_labels).argmax())
    keep_mask = labels == largest_label
    filtered_points = pts[keep_mask]
    filtered_colors = None if colors is None else np.asarray(colors)[keep_mask]
    return filtered_points, filtered_colors


def remove_radius_outliers(
    points: np.ndarray,
    colors: np.ndarray | None = None,
    nb_points: int = 12,
    radius_m: float = 0.02,
) -> tuple[np.ndarray, np.ndarray | None]:
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return pts, colors
    if pts.shape[0] < max(10, nb_points):
        return pts, colors

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    _, indices = pcd.remove_radius_outlier(nb_points=int(nb_points), radius=float(radius_m))
    if not indices:
        return pts, colors
    indices_np = np.asarray(indices, dtype=np.int32)
    filtered_points = pts[indices_np]
    filtered_colors = None if colors is None else np.asarray(colors)[indices_np]
    return filtered_points, filtered_colors


def depth_to_xyz_image(depth_meters: np.ndarray, intrinsics) -> np.ndarray:
    if intrinsics is None:
        raise ValueError("Camera intrinsics are required")
    height, width = depth_meters.shape[:2]
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    z = depth_meters.astype(np.float32)
    x = z * (cols - intrinsics.ppx) / intrinsics.fx
    y = z * (rows - intrinsics.ppy) / intrinsics.fy
    return np.stack([x, y, z], axis=-1).astype(np.float32)


def depth_to_scene_points(depth_meters: np.ndarray, intrinsics, clip_max: float = 3.0, mask=None) -> np.ndarray:
    xyz = depth_to_xyz_image(depth_meters, intrinsics)
    valid = (depth_meters > 0) & (depth_meters < clip_max)
    if mask is not None:
        if mask.shape[:2] != depth_meters.shape[:2]:
            mask = cv2.resize(mask.astype(np.uint8), (depth_meters.shape[1], depth_meters.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        else:
            mask = mask.astype(bool)
        valid = valid & mask
    return xyz[valid].astype(np.float32)


def _rs_frames_to_points_and_colors(
    depth_frame,
    color_frame=None,
    clip_max: float = 3.0,
    mask=None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if depth_frame is None:
        return np.empty((0, 3), dtype=np.float32), None

    mapped = color_frame if color_frame is not None else depth_frame
    _RS_POINTCLOUD.map_to(mapped)
    points = _RS_POINTCLOUD.calculate(depth_frame)

    vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    valid = (vertices[:, 2] > 0.0) & (vertices[:, 2] < float(clip_max)) & ~np.all(vertices == 0.0, axis=1)

    if mask is not None:
        mask_np = np.asarray(mask).astype(bool)
        h, w = mask_np.shape[:2]
        valid &= mask_np.reshape(h * w)

    filtered_points = vertices[valid].astype(np.float32, copy=False)
    if color_frame is None or filtered_points.shape[0] == 0:
        return filtered_points, None

    tex_coords = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
    color_image = np.asanyarray(color_frame.get_data())
    h, w = color_image.shape[:2]
    u = np.clip((tex_coords[valid, 0] * w).astype(np.int32), 0, w - 1)
    v = np.clip((tex_coords[valid, 1] * h).astype(np.int32), 0, h - 1)
    colors = color_image[v, u, ::-1].astype(np.float64) / 255.0
    return filtered_points, colors


def depth_frame_to_scene_points_rs(depth_frame, color_frame=None, clip_max: float = 3.0, mask=None) -> np.ndarray:
    points, _colors = _rs_frames_to_points_and_colors(
        depth_frame=depth_frame,
        color_frame=color_frame,
        clip_max=clip_max,
        mask=mask,
    )
    return points


def scene_points_to_pointcloud(
    points: np.ndarray,
    color_rgb: tuple[float, float, float] = (0.65, 0.65, 0.65),
):
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    color = np.asarray(color_rgb, dtype=np.float64).reshape(3)
    colors = np.tile(color, (pts.shape[0], 1))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def mask_to_pointcloud(
    color_bgr: np.ndarray,
    depth_meters: np.ndarray,
    mask,
    intrinsics,
    clip_max: float = 3.0,
    pointcloud_filter_mode: str = "none",
    island_eps_m: float = 0.02,
    island_min_points: int = 30,
    radius_nb_points: int = 12,
    radius_m: float = 0.02,
    pointcloud_backend: str = "manual",
    depth_frame=None,
    color_frame=None,
):
    if intrinsics is None:
        return None
    height, width = depth_meters.shape[:2]
    if mask.shape[:2] != (height, width):
        mask_resized = cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)
    else:
        mask_resized = mask.astype(bool)

    valid = mask_resized & (depth_meters > 0) & (depth_meters < clip_max)
    if np.sum(valid) == 0:
        return None

    if pointcloud_backend == "sdk" and depth_frame is not None:
        points, colors = _rs_frames_to_points_and_colors(
            depth_frame=depth_frame,
            color_frame=color_frame,
            clip_max=clip_max,
            mask=valid,
        )
        if points.shape[0] == 0:
            return None
    else:
        xyz = depth_to_xyz_image(depth_meters, intrinsics)
        points = xyz[valid]
        color_rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
        colors = color_rgb[valid].astype(np.float64) / 255.0

    if pointcloud_filter_mode == "island":
        points, colors = keep_largest_point_cluster(
            points,
            colors=colors,
            eps_m=island_eps_m,
            min_points=island_min_points,
        )
        if points.shape[0] == 0:
            return None
    elif pointcloud_filter_mode == "radius":
        points, colors = remove_radius_outliers(
            points,
            colors=colors,
            nb_points=radius_nb_points,
            radius_m=radius_m,
        )
        if points.shape[0] == 0:
            return None

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pcd.colors = o3d.utility.Vector3dVector(colors)
    if pointcloud_filter_mode not in {"island", "radius"} and len(pcd.points) > 20:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    return pcd


def compute_3d_bbox(pcd):
    if pcd is None or len(pcd.points) == 0:
        return None, None
    aabb = pcd.get_axis_aligned_bounding_box()
    aabb.color = (1, 0, 0)
    extent = aabb.get_extent()
    center = aabb.get_center()
    return aabb, {
        "center": center,
        "extent": extent,
        "width_m": extent[0],
        "height_m": extent[1],
        "depth_m": extent[2],
    }


def apply_masks_overlay(image: np.ndarray, masks, scores, boxes, alpha: float = 0.45) -> np.ndarray:
    overlay = image.copy()
    count = masks.shape[0] if hasattr(masks, "shape") and len(masks.shape) >= 3 else len(masks)
    for index in range(count):
        color = COLORS[index % len(COLORS)]
        mask = masks[index].squeeze()
        if isinstance(mask, torch.Tensor):
            mask_np = mask.cpu().numpy().astype(bool)
        else:
            mask_np = mask.astype(bool)

        if mask_np.shape[:2] != overlay.shape[:2]:
            mask_np = cv2.resize(mask_np.astype(np.uint8), (overlay.shape[1], overlay.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)

        overlay[mask_np] = (overlay[mask_np] * (1 - alpha) + np.array(color) * alpha).astype(np.uint8)
        contours, _ = cv2.findContours(mask_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color, 2)

        if boxes is not None:
            box = boxes[index].cpu().numpy().astype(int) if isinstance(boxes[index], torch.Tensor) else np.array(boxes[index]).astype(int)
            score = scores[index].item() if isinstance(scores[index], torch.Tensor) else float(scores[index])
            cv2.rectangle(overlay, (box[0], box[1]), (box[2], box[3]), color, 2)
            label = f"#{index} {score:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(overlay, (box[0], box[1] - th - 10), (box[0] + tw + 4, box[1]), color, -1)
            cv2.putText(overlay, label, (box[0] + 2, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def depth_to_colormap(depth_image: np.ndarray) -> np.ndarray:
    depth_norm = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    return cv2.applyColorMap(depth_norm, cv2.COLORMAP_JET)


def save_segmentation_outputs(
    color_bgr,
    depth_meters,
    masks,
    scores,
    boxes,
    text_prompt,
    intrinsics,
    clip_max: float = 3.0,
    output_dir: str = "output",
    pointcloud_filter_mode: str = "none",
    island_eps_m: float = 0.02,
    island_min_points: int = 30,
    radius_nb_points: int = 12,
    radius_m: float = 0.02,
    pointcloud_backend: str = "manual",
    depth_frame=None,
    color_frame=None,
):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = text_prompt.replace(" ", "_")[:20]

    overlay = apply_masks_overlay(color_bgr, masks, scores, boxes)
    cv2.imwrite(os.path.join(output_dir, f"overlay_{tag}_{timestamp}.png"), overlay)
    cv2.imwrite(os.path.join(output_dir, f"raw_{tag}_{timestamp}.png"), color_bgr)

    depth_color = depth_to_colormap((depth_meters * 1000).astype(np.uint16))
    cv2.imwrite(os.path.join(output_dir, f"depth_{tag}_{timestamp}.png"), depth_color)

    count = masks.shape[0] if hasattr(masks, "shape") and len(masks.shape) >= 3 else len(masks)
    pointclouds = []

    for index in range(count):
        mask = masks[index].squeeze()
        mask_np = mask.cpu().numpy() if isinstance(mask, torch.Tensor) else np.array(mask)
        mask_path = os.path.join(output_dir, f"mask_{tag}_{index}_{timestamp}.png")
        cv2.imwrite(mask_path, (mask_np * 255).astype(np.uint8))

        rgba = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2BGRA)
        mask_resized = mask_np
        if mask_np.shape[:2] != color_bgr.shape[:2]:
            mask_resized = cv2.resize(mask_np.astype(np.uint8), (color_bgr.shape[1], color_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        rgba[:, :, 3] = (mask_resized * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, f"cutout_{tag}_{index}_{timestamp}.png"), rgba)

        pcd = mask_to_pointcloud(
            color_bgr,
            depth_meters,
            mask_np,
            intrinsics,
            clip_max,
            pointcloud_filter_mode=pointcloud_filter_mode,
            island_eps_m=island_eps_m,
            island_min_points=island_min_points,
            radius_nb_points=radius_nb_points,
            radius_m=radius_m,
            pointcloud_backend=pointcloud_backend,
            depth_frame=depth_frame,
            color_frame=color_frame,
        )
        if pcd is not None and len(pcd.points) > 0:
            ply_path = os.path.join(output_dir, f"pointcloud_{tag}_{index}_{timestamp}.ply")
            o3d.io.write_point_cloud(ply_path, pcd)
        pointclouds.append(pcd)

    merged = o3d.geometry.PointCloud()
    for pcd in pointclouds:
        if pcd is not None:
            merged += pcd
    if len(merged.points) > 0:
        o3d.io.write_point_cloud(os.path.join(output_dir, f"merged_{tag}_{timestamp}.ply"), merged)

    return pointclouds


def visualize_3d(pointclouds, text_prompt: str = "", grasp_groups=None, grasp_topk_vis: int = 10) -> None:
    geometries = []
    for index, pcd in enumerate(pointclouds):
        if pcd is None:
            continue
        geometries.append(pcd)
        aabb, _ = compute_3d_bbox(pcd)
        if aabb is not None:
            aabb.color = np.array(COLORS[index % len(COLORS)]) / 255.0
            geometries.append(aabb)

    if grasp_groups is not None:
        for grasp_group in grasp_groups:
            if grasp_group is None or len(grasp_group) == 0:
                continue
            try:
                geometries.extend(grasp_group[: min(len(grasp_group), grasp_topk_vis)].to_open3d_geometry_list())
            except Exception:
                pass

    if not geometries:
        return
    geometries.append(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
    o3d.visualization.draw_geometries(geometries, window_name=f'YOLOv8-seg + GraspNet - "{text_prompt}"', width=1280, height=720)
