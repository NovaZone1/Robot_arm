from src.robot.client import Ros2PiperClient
from src.robot.types import GripperStatus


def test_open_gripper_wait_accepts_feedback_beyond_requested_clearance(monkeypatch):
    client = Ros2PiperClient()
    monkeypatch.setattr(
        client,
        "get_gripper_status",
        lambda: GripperStatus(angle_mm=80.0, effort_nm=0.0, enabled=True),
    )

    assert client.wait_for_gripper(
        target_mm=70.0,
        tol_mm=5.0,
        timeout_s=0.1,
    )
