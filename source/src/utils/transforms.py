from __future__ import annotations

import math

import numpy as np


def rotation_matrix_from_rpy_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    rx = math.radians(roll_deg)
    ry = math.radians(pitch_deg)
    rz = math.radians(yaw_deg)

    rx_m = np.array(
        [[1.0, 0.0, 0.0], [0.0, math.cos(rx), -math.sin(rx)], [0.0, math.sin(rx), math.cos(rx)]],
        dtype=np.float64,
    )
    ry_m = np.array(
        [[math.cos(ry), 0.0, math.sin(ry)], [0.0, 1.0, 0.0], [-math.sin(ry), 0.0, math.cos(ry)]],
        dtype=np.float64,
    )
    rz_m = np.array(
        [[math.cos(rz), -math.sin(rz), 0.0], [math.sin(rz), math.cos(rz), 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return rz_m @ ry_m @ rx_m


def rpy_deg_from_rotation_matrix(rotation: np.ndarray) -> tuple[float, float, float]:
    pitch = math.asin(-float(rotation[2, 0]))
    roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return (math.degrees(roll), math.degrees(pitch), math.degrees(yaw))


def make_transform_xyz_rpy_mm_deg(xyz_mm: tuple[float, float, float], rpy_deg: tuple[float, float, float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_matrix_from_rpy_deg(*rpy_deg)
    transform[:3, 3] = np.array(xyz_mm, dtype=np.float64) / 1000.0
    return transform


def make_transform_matrix(rotation: np.ndarray, translation_m: tuple[float, float, float] | np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    transform[:3, 3] = np.asarray(translation_m, dtype=np.float64).reshape(3)
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inv = np.eye(4, dtype=np.float64)
    inv[:3, :3] = rotation.T
    inv[:3, 3] = -(rotation.T @ translation)
    return inv


def transform_point(transform: np.ndarray, point_m: tuple[float, float, float]) -> tuple[float, float, float]:
    point_h = np.array([point_m[0], point_m[1], point_m[2], 1.0], dtype=np.float64)
    transformed = transform @ point_h
    return (float(transformed[0]), float(transformed[1]), float(transformed[2]))
