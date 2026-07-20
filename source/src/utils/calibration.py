from __future__ import annotations

import numpy as np

from .transforms import invert_transform, make_transform_xyz_rpy_mm_deg, rpy_deg_from_rotation_matrix


def load_camera_to_tcp_transform(calib: dict, *, allow_legacy: bool = True) -> tuple[np.ndarray, str]:
    """Load camera->tcp transform from calibration config.

    Preferred fields:
    - camera_to_tcp_xyz_mm
    - camera_to_tcp_rpy_deg

    Backward-compatible legacy fields:
    - tcp_to_camera_xyz_mm
    - tcp_to_camera_rpy_deg
      These were historically misnamed in this repo but actually stored camera->tcp.
    """

    has_primary = "camera_to_tcp_xyz_mm" in calib and "camera_to_tcp_rpy_deg" in calib
    has_legacy = "tcp_to_camera_xyz_mm" in calib and "tcp_to_camera_rpy_deg" in calib

    if has_primary:
        transform = make_transform_xyz_rpy_mm_deg(
            xyz_mm=tuple(calib["camera_to_tcp_xyz_mm"]),
            rpy_deg=tuple(calib["camera_to_tcp_rpy_deg"]),
        )
        return transform, "camera_to_tcp"

    if has_legacy and allow_legacy:
        semantics = calib.get("tcp_to_camera_semantics")
        if semantics == "tcp->camera":
            raise KeyError(
                "Calibration config only provides tcp_to_camera_* with true tcp->camera semantics. "
                "Please regenerate or migrate the config so camera_to_tcp_* is present."
            )
        transform = make_transform_xyz_rpy_mm_deg(
            xyz_mm=tuple(calib["tcp_to_camera_xyz_mm"]),
            rpy_deg=tuple(calib["tcp_to_camera_rpy_deg"]),
        )
        return transform, "legacy_tcp_to_camera_as_camera_to_tcp"

    if has_legacy and not allow_legacy:
        raise KeyError(
            "Calibration config missing canonical camera_to_tcp_* fields. "
            "Runtime path requires explicit camera_to_tcp_xyz_mm/camera_to_tcp_rpy_deg."
        )

    raise KeyError("Calibration config missing camera_to_tcp_xyz_mm/camera_to_tcp_rpy_deg")


def calibration_fields_from_camera_to_tcp(camera_to_tcp: np.ndarray) -> dict[str, list[float]]:
    tcp_to_camera = invert_transform(camera_to_tcp)
    camera_to_tcp_rpy = rpy_deg_from_rotation_matrix(camera_to_tcp[:3, :3])
    tcp_to_camera_rpy = rpy_deg_from_rotation_matrix(tcp_to_camera[:3, :3])
    return {
        "camera_to_tcp_xyz_mm": [float(v) for v in (camera_to_tcp[:3, 3] * 1000.0)],
        "camera_to_tcp_rpy_deg": [float(v) for v in camera_to_tcp_rpy],
        "tcp_to_camera_xyz_mm": [float(v) for v in (tcp_to_camera[:3, 3] * 1000.0)],
        "tcp_to_camera_rpy_deg": [float(v) for v in tcp_to_camera_rpy],
    }
