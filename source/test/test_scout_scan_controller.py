from types import SimpleNamespace
import threading

import robot_grasp_ros2.scout_scan_controller_node as controller_module
from robot_grasp_ros2.scout_scan_controller_node import ScoutScanControllerNode


def _node_with_parameters():
    node = ScoutScanControllerNode.__new__(ScoutScanControllerNode)
    node._motion_lock = threading.Lock()
    node._stop_event = threading.Event()
    parameters = {
        "max_distance_m": 1.50,
        "max_speed_mps": 0.06,
        "min_speed_mps": 0.02,
        "position_tolerance_m": 0.012,
        "command_rate_hz": 20.0,
        "max_lateral_error_m": 0.05,
        "max_yaw_error_deg": 3.0,
    }
    node.get_parameter = lambda name: SimpleNamespace(value=parameters[name])
    node._publish_status = lambda _text: None
    return node


def _response():
    return SimpleNamespace(
        success=False,
        message="",
        traveled_m=0.0,
        lateral_error_m=0.0,
        yaw_error_deg=0.0,
    )


def test_relative_base_move_rejects_distance_above_hard_limit(monkeypatch):
    node = _node_with_parameters()
    published = []
    node._publish_velocity = published.append
    monkeypatch.setattr(controller_module.time, "sleep", lambda _seconds: None)

    result = ScoutScanControllerNode._on_move_relative(
        node,
        SimpleNamespace(distance_m=1.51, speed_mps=0.04, timeout_s=45.0),
        _response(),
    )

    assert result.success is False
    assert "exceeds safety limit" in result.message
    assert published == [0.0] * 5


def test_relative_base_move_uses_odom_and_finishes_with_zero_velocity(monkeypatch):
    node = _node_with_parameters()
    published = []
    node._publish_velocity = published.append
    snapshots = iter(
        (
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.05, 0.0, 0.0),
            (0.095, 0.0, 0.0),
        )
    )
    node._odom_snapshot = lambda: next(snapshots)
    monkeypatch.setattr(controller_module.time, "sleep", lambda _seconds: None)

    result = ScoutScanControllerNode._on_move_relative(
        node,
        SimpleNamespace(distance_m=0.10, speed_mps=0.04, timeout_s=10.0),
        _response(),
    )

    assert result.success is True
    assert result.traveled_m == 0.095
    assert any(value > 0.0 for value in published)
    assert published[-5:] == [0.0] * 5
