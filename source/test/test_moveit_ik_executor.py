import math
import sys
import types

import pytest

from src.robot.types import EndPoseMMDeg
from src.robot.moveit_ik import (
    MoveItIkExecutor,
    clamp_joint_command_speed,
    ensure_moveit_pose_mode_supported,
    extract_moveit_arm_seed_state,
    extract_arm_joint_positions,
    make_joint_command_payload,
    normalize_joint_state_for_moveit,
    pose_to_quaternion_xyzw,
)


def test_extract_arm_joint_positions_orders_solution_by_joint_name():
    names = ["joint3", "joint1", "joint6", "joint2", "joint5", "joint4", "joint7"]
    positions = [0.3, 0.1, 0.6, 0.2, 0.5, 0.4, 0.07]

    ordered = extract_arm_joint_positions(names, positions)

    assert ordered == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def test_make_joint_command_payload_preserves_gripper_and_speed():
    payload = make_joint_command_payload(
        arm_joint_positions=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        gripper_position=0.07,
        speed_percent=12.0,
        gripper_effort=1.2,
    )

    assert payload["name"] == ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"]
    assert payload["position"] == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.07]
    assert payload["velocity"] == [12.0, 12.0, 12.0, 12.0, 12.0, 12.0, 12.0]
    assert payload["effort"] == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.2]


def test_publish_joint_command_uses_locked_gripper_command_after_close():
    class FakeStamp:
        def to_msg(self):
            return "stamp"

    class FakeClock:
        def now(self):
            return FakeStamp()

    class FakeNode:
        def get_clock(self):
            return FakeClock()

    class FakeJointState:
        def __init__(self):
            self.header = types.SimpleNamespace(stamp=None)
            self.name = []
            self.position = []
            self.velocity = []
            self.effort = []

    class FakePublisher:
        def __init__(self):
            self.messages = []

        def publish(self, msg):
            self.messages.append(msg)

    executor = MoveItIkExecutor.__new__(MoveItIkExecutor)
    executor._node = FakeNode()
    executor.config = types.SimpleNamespace(default_gripper_effort=1.0)
    executor._JointState = FakeJointState
    executor._joint_cmd_pub = FakePublisher()
    executor._joint_state_snapshot = lambda: (
        ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper"],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.07],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    )

    executor.lock_gripper_command(position_m=0.0, effort_nm=0.6)
    executor.publish_joint_command([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], speed_percent=20.0)

    msg = executor._joint_cmd_pub.messages[-1]
    assert msg.position == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.0]
    assert msg.velocity == [20.0, 20.0, 20.0, 20.0, 20.0, 20.0, 20.0]
    assert msg.effort == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6]


def test_clamp_joint_command_speed_stays_within_driver_range():
    assert clamp_joint_command_speed(-5.0) == 1.0
    assert clamp_joint_command_speed(20.0) == 20.0
    assert clamp_joint_command_speed(150.0) == 100.0


def test_normalize_joint_state_for_moveit_maps_gripper_to_joint7():
    names, positions = normalize_joint_state_for_moveit(
        ["joint1", "joint2", "gripper"],
        [0.1, 0.2, 0.07],
    )

    assert names == ["joint1", "joint2", "joint7"]
    assert positions == [0.1, 0.2, 0.07]


def test_extract_moveit_arm_seed_state_drops_gripper_joint():
    names, positions = extract_moveit_arm_seed_state(
        ["joint3", "joint1", "joint6", "joint2", "joint5", "joint4", "gripper"],
        [0.3, 0.1, 0.6, 0.2, 0.5, 0.4, 0.07],
    )

    assert names == ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    assert positions == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]


def test_pose_to_quaternion_xyzw_returns_unit_quaternion():
    quat = pose_to_quaternion_xyzw(EndPoseMMDeg(1.0, 2.0, 3.0, 180.0, 0.0, 0.0))

    assert len(quat) == 4
    assert math.isclose(sum(v * v for v in quat), 1.0, rel_tol=0.0, abs_tol=1e-6)


def test_moveit_pose_mode_rejects_non_ros2_backend():
    with pytest.raises(RuntimeError, match="requires robot_backend=ros2"):
        ensure_moveit_pose_mode_supported("fake")


def test_moveit_executor_response_timeout_has_padding():
    executor = MoveItIkExecutor.__new__(MoveItIkExecutor)
    executor.config = types.SimpleNamespace(timeout_s=5.0, response_timeout_padding_s=2.0)

    assert executor._service_response_timeout_s() == 7.0


def test_moveit_executor_uses_reentrant_callback_group_for_joint_feedback(monkeypatch):
    moveit_srv_module = types.ModuleType("moveit_msgs.srv")
    sensor_msg_module = types.ModuleType("sensor_msgs.msg")
    callback_group_module = types.ModuleType("rclpy.callback_groups")

    class FakeGetPositionIK:
        class Request:
            pass

    class FakeJointState:
        pass

    class FakeReentrantCallbackGroup:
        pass

    moveit_srv_module.GetPositionIK = FakeGetPositionIK
    sensor_msg_module.JointState = FakeJointState
    callback_group_module.ReentrantCallbackGroup = FakeReentrantCallbackGroup

    monkeypatch.setitem(sys.modules, "moveit_msgs.srv", moveit_srv_module)
    monkeypatch.setitem(sys.modules, "sensor_msgs.msg", sensor_msg_module)
    monkeypatch.setitem(sys.modules, "rclpy.callback_groups", callback_group_module)

    class FakeNode:
        def __init__(self):
            self.subscription_kwargs = None
            self.client_kwargs = None

        def create_subscription(self, *args, **kwargs):
            self.subscription_kwargs = kwargs
            return object()

        def create_publisher(self, *args, **kwargs):
            return object()

        def create_client(self, *args, **kwargs):
            self.client_kwargs = kwargs
            return object()

    node = FakeNode()

    MoveItIkExecutor(node)

    assert isinstance(node.subscription_kwargs["callback_group"], FakeReentrantCallbackGroup)
    assert node.client_kwargs["callback_group"] is node.subscription_kwargs["callback_group"]
