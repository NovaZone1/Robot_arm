from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class NPointToolOffsetResult:
    tool_contact_offset_tool_m: np.ndarray
    fixed_point_base_m: np.ndarray
    residual_vectors_m: np.ndarray
    rmse_m: float
    max_error_m: float
    mean_error_m: float
    condition_number: float


def _as_positions(points_m: np.ndarray | list[list[float]] | list[tuple[float, float, float]]) -> np.ndarray:
    arr = np.asarray(points_m, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"Expected Nx3 positions, got shape {arr.shape}")
    if arr.shape[0] < 3:
        raise ValueError("Need at least 3 poses to estimate tool contact offset")
    return arr


def _as_rotations(rotations: np.ndarray | list[np.ndarray]) -> np.ndarray:
    arr = np.asarray(rotations, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (3, 3):
        raise ValueError(f"Expected Nx3x3 rotations, got shape {arr.shape}")
    return arr


def estimate_tool_contact_offset_from_fixed_point(
    tcp_positions_base_m: np.ndarray | list[list[float]] | list[tuple[float, float, float]],
    tcp_rotations_base: np.ndarray | list[np.ndarray],
) -> NPointToolOffsetResult:
    """Solve t_tool and fixed point P from p_i + R_i @ t_tool = P."""
    positions = _as_positions(tcp_positions_base_m)
    rotations = _as_rotations(tcp_rotations_base)
    if positions.shape[0] != rotations.shape[0]:
        raise ValueError(
            f"Pose count mismatch: positions {positions.shape[0]}, rotations {rotations.shape[0]}"
        )

    pose_count = positions.shape[0]
    a = np.zeros((pose_count * 3, 6), dtype=np.float64)
    b = np.zeros((pose_count * 3,), dtype=np.float64)
    identity = np.eye(3, dtype=np.float64)

    for i in range(pose_count):
        row = i * 3
        a[row : row + 3, 0:3] = rotations[i]
        a[row : row + 3, 3:6] = -identity
        b[row : row + 3] = -positions[i]

    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    tool_contact_offset_tool_m = solution[0:3]
    fixed_point_base_m = solution[3:6]

    predicted_points = positions + np.einsum("nij,j->ni", rotations, tool_contact_offset_tool_m)
    residual_vectors = predicted_points - fixed_point_base_m
    residual_norms = np.linalg.norm(residual_vectors, axis=1)

    rmse_m = float(np.sqrt(np.mean(np.sum(residual_vectors**2, axis=1))))
    max_error_m = float(np.max(residual_norms))
    mean_error_m = float(np.mean(residual_norms))
    singular_values = np.linalg.svd(a, compute_uv=False)
    min_sv = float(np.min(singular_values))
    condition_number = float(np.max(singular_values) / min_sv) if min_sv > 1e-12 else float("inf")

    return NPointToolOffsetResult(
        tool_contact_offset_tool_m=tool_contact_offset_tool_m,
        fixed_point_base_m=fixed_point_base_m,
        residual_vectors_m=residual_vectors,
        rmse_m=rmse_m,
        max_error_m=max_error_m,
        mean_error_m=mean_error_m,
        condition_number=condition_number,
    )


def tool_offset_result_to_dict(result: NPointToolOffsetResult) -> dict[str, object]:
    return {
        "tool_contact_offset_tool_mm": (result.tool_contact_offset_tool_m * 1000.0).tolist(),
        "fixed_point_base_mm": (result.fixed_point_base_m * 1000.0).tolist(),
        "rmse_mm": result.rmse_m * 1000.0,
        "max_error_mm": result.max_error_m * 1000.0,
        "mean_error_mm": result.mean_error_m * 1000.0,
        "condition_number": result.condition_number,
        "residual_vectors_mm": (result.residual_vectors_m * 1000.0).tolist(),
    }
