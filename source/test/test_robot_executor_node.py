import json
from types import SimpleNamespace

import numpy as np
import pytest

from robot_grasp_ros2.distributed_utils import grasp_plan_from_msg, grasp_plan_to_msg
from robot_grasp_ros2.robot_executor_node import RobotExecutorNode
from src.grasping.models import GraspCandidate, GraspPlan
from src.robot.executor_models import MovePoseCommand
from src.robot.types import EndPoseMMDeg
from src.utils.transforms import rotation_matrix_from_rpy_deg


def test_named_pose_timeout_reads_configured_parameter(monkeypatch):
    node = RobotExecutorNode.__new__(RobotExecutorNode)
    monkeypatch.setattr(
        node,
        "get_parameter",
        lambda name: SimpleNamespace(value=22.5 if name == "named_pose_timeout_s" else None),
    )

    assert RobotExecutorNode._named_pose_timeout_s(node) == 22.5


def test_plan_pose_timeout_reads_configured_parameter(monkeypatch):
    node = RobotExecutorNode.__new__(RobotExecutorNode)
    monkeypatch.setattr(
        node,
        "get_parameter",
        lambda name: SimpleNamespace(value=27.5 if name == "plan_pose_timeout_s" else None),
    )

    assert RobotExecutorNode._plan_pose_timeout_s(node) == 27.5


def test_pose_only_completion_requires_healthy_can_control():
    healthy = SimpleNamespace(
        err_code=0,
        teach_status_code=0,
        arm_status="NORMAL",
        control_mode="CAN",
        motion_status="NOT_ARRIVED",
    )
    assert RobotExecutorNode._pose_only_completion_is_safe(healthy) is True

    unhealthy = SimpleNamespace(**{**healthy.__dict__, "err_code": 1})
    assert RobotExecutorNode._pose_only_completion_is_safe(unhealthy) is False


def _place_request(
    *,
    label_verified=True,
    label_confidence=0.80,
    slot_index=2,
    box_size=(0.180, 0.132, 0.087),
    approach=(300.0, 50.0, 380.0, 180.0, 0.0, 0.0),
    release=(300.0, 50.0, 300.0, 180.0, 0.0, 0.0),
    retreat=(300.0, 50.0, 400.0, 180.0, 0.0, 0.0),
):
    def pose(values):
        return SimpleNamespace(
            x_mm=values[0],
            y_mm=values[1],
            z_mm=values[2],
            roll_deg=values[3],
            pitch_deg=values[4],
            yaw_deg=values[5],
        )

    return SimpleNamespace(
        plan=SimpleNamespace(
            item_id="red_block",
            slot_index=slot_index,
            label_verified=label_verified,
            label_confidence=label_confidence,
            box_outer_size_m=list(box_size),
            approach_pose=pose(approach),
            release_pose=pose(release),
            retreat_pose=pose(retreat),
        )
    )


def _place_validation_node(monkeypatch):
    node = RobotExecutorNode.__new__(RobotExecutorNode)
    values = {
        "placement_box_outer_size_m": [0.180, 0.132, 0.087],
        "placement_slot_count": 6,
        "placement_box_size_tolerance_m": 0.005,
        "placement_min_label_confidence": 0.42,
        "placement_min_vertical_clearance_mm": 50.0,
    }
    monkeypatch.setattr(node, "get_parameter", lambda name: SimpleNamespace(value=values[name]))
    return node


def test_place_plan_requires_verified_matching_label(monkeypatch):
    node = _place_validation_node(monkeypatch)

    with pytest.raises(RuntimeError, match="label was not verified"):
        RobotExecutorNode._validate_place_request(
            node,
            _place_request(label_verified=False),
        )


def test_place_plan_rejects_wrong_box_size(monkeypatch):
    node = _place_validation_node(monkeypatch)

    with pytest.raises(RuntimeError, match="does not match configured size"):
        RobotExecutorNode._validate_place_request(
            node,
            _place_request(box_size=(0.300, 0.132, 0.087)),
        )


def test_place_plan_rejects_invalid_slot(monkeypatch):
    node = _place_validation_node(monkeypatch)

    with pytest.raises(RuntimeError, match="slot_index"):
        RobotExecutorNode._validate_place_request(
            node,
            _place_request(slot_index=6),
        )


def test_place_plan_rejects_low_label_confidence(monkeypatch):
    node = _place_validation_node(monkeypatch)

    with pytest.raises(RuntimeError, match="label confidence"):
        RobotExecutorNode._validate_place_request(
            node,
            _place_request(label_confidence=0.20),
        )


def test_place_plan_requires_vertical_clearance(monkeypatch):
    node = _place_validation_node(monkeypatch)

    with pytest.raises(RuntimeError, match="enough vertical clearance"):
        RobotExecutorNode._validate_place_request(
            node,
            _place_request(approach=(300.0, 50.0, 330.0, 180.0, 0.0, 0.0)),
        )


def test_safe_top_down_recomputes_flange_position_for_final_orientation(monkeypatch):
    node = RobotExecutorNode.__new__(RobotExecutorNode)
    monkeypatch.setattr(node, "_top_down_rpy_deg", lambda: (180.0, 0.0, 0.0))
    monkeypatch.setattr(node, "_top_down_min_target_z_mm", lambda: 300.0)
    candidate = GraspCandidate(
        instance_index=0,
        score=0.9,
        width_m=0.04,
        depth_m=0.02,
        translation_camera_m=(0.0, 0.0, 0.4),
        rotation_camera=np.eye(3),
        object_center_camera_m=None,
        center_offset_m=None,
        raw_grasp=None,
    )
    plan = GraspPlan(
        candidate=candidate,
        target_base_m=(0.1, 0.2, 0.13),
        target_rpy_deg=(0.0, 0.0, 0.0),
        pregrasp_base_m=(0.1, 0.2, 0.2),
        grasp_base_m=(0.1, 0.2, 0.15),
        retreat_base_m=(0.1, 0.2, 0.3),
        within_workspace=True,
        workspace_violations=[],
        target_contact_point_base_m=(0.1, 0.2, 0.235),
        tool_contact_offset_tool_m=(0.0, 0.0, 0.105),
    )

    pose = RobotExecutorNode._target_pose_from_plan_top_down(node, plan)

    assert pose.x_mm == pytest.approx(100.0)
    assert pose.y_mm == pytest.approx(200.0)
    assert pose.z_mm == pytest.approx(340.0)
    assert (pose.roll_deg, pose.pitch_deg, pose.yaw_deg) == (180.0, 0.0, 0.0)


def test_safe_top_down_rejects_plan_without_tool_contact_geometry(monkeypatch):
    node = RobotExecutorNode.__new__(RobotExecutorNode)
    monkeypatch.setattr(node, "_top_down_rpy_deg", lambda: (180.0, 60.0, 180.0))
    plan = SimpleNamespace(
        target_contact_point_base_m=None,
        tool_contact_offset_tool_m=None,
    )

    with pytest.raises(RuntimeError, match="requires target contact geometry"):
        RobotExecutorNode._target_pose_from_plan_top_down(node, plan)


def test_safe_top_down_rejects_recomputed_target_below_minimum(monkeypatch):
    node = RobotExecutorNode.__new__(RobotExecutorNode)
    monkeypatch.setattr(node, "_top_down_rpy_deg", lambda: (0.0, 0.0, 0.0))
    monkeypatch.setattr(node, "_top_down_min_target_z_mm", lambda: 300.0)
    plan = SimpleNamespace(
        target_contact_point_base_m=(0.1, 0.2, 0.25),
        tool_contact_offset_tool_m=(0.0, 0.0, 0.105),
    )

    with pytest.raises(RuntimeError, match="below top_down_min_target_z_mm"):
        RobotExecutorNode._target_pose_from_plan_top_down(node, plan)


def test_safe_top_down_variant_search_uses_first_fully_reachable_rpy(monkeypatch):
    node = RobotExecutorNode.__new__(RobotExecutorNode)
    variants = [(180.0, 60.0, 180.0), (180.0, 60.0, -90.0)]
    monkeypatch.setattr(node, "_safe_cartesian_rpy_variants", lambda _plan: variants)
    monkeypatch.setattr(
        node,
        "_build_safe_top_down_waypoints",
        lambda **kwargs: [
            (
                "topdown_lateral",
                EndPoseMMDeg(0.0, 0.0, 400.0, *kwargs["rpy_deg"]),
            ),
            (
                "topdown_descend",
                EndPoseMMDeg(0.0, 0.0, 300.0, *kwargs["rpy_deg"]),
            ),
        ],
    )

    def compute_ik(pose):
        if pose.yaw_deg == 180.0:
            raise RuntimeError("MoveIt IK failed: code=-31")
        return [0.0] * 6

    selected_rpy, waypoints, attempts = RobotExecutorNode._evaluate_safe_top_down_variants(
        node,
        plan=SimpleNamespace(),
        current_pose=EndPoseMMDeg(0.0, 0.0, 500.0, 0.0, 0.0, 0.0),
        compute_ik=compute_ik,
    )

    assert selected_rpy == (180.0, 60.0, -90.0)
    assert waypoints is not None
    assert [attempt["status"] for attempt in attempts] == ["rejected", "accepted"]
    assert attempts[0]["waypoint_results"][0]["stage"] == "topdown_lateral"


def test_center_horizontal_lifts_only_configured_distance(monkeypatch):
    node = RobotExecutorNode.__new__(RobotExecutorNode)
    rpy = (180.0, 85.0, -90.0)
    target = EndPoseMMDeg(10.0, 361.0, 229.0, *rpy)
    monkeypatch.setattr(node, "_top_down_rpy_deg", lambda: rpy)
    monkeypatch.setattr(node, "_target_pose_from_plan_top_down", lambda _plan, rpy_deg=None: target)
    monkeypatch.setattr(node, "_top_down_approach_height_mm", lambda: 110.0)
    monkeypatch.setattr(node, "_top_down_min_safe_z_mm", lambda: 350.0)
    monkeypatch.setattr(node, "_top_down_lift_height_mm", lambda: 80.0)
    monkeypatch.setattr(node, "_top_down_lift_to_safe_z", lambda: False)
    monkeypatch.setattr(node, "_top_down_vertical_step_mm", lambda: 1000.0)
    monkeypatch.setattr(node, "_top_down_lateral_step_mm", lambda: 1000.0)

    waypoints = RobotExecutorNode._build_safe_top_down_waypoints(
        node,
        plan=SimpleNamespace(),
        current_pose=EndPoseMMDeg(0.0, 35.5, 491.1, 180.0, 67.77, -89.97),
    )

    lift_pose = next(pose for name, pose in waypoints if name.startswith("topdown_lift_object"))
    assert lift_pose.z_mm == pytest.approx(309.0)


def test_top_down_lowers_high_observation_before_lateral_motion(monkeypatch):
    node = RobotExecutorNode.__new__(RobotExecutorNode)
    rpy = (180.0, 67.8, 90.0)
    target = EndPoseMMDeg(-15.0, -355.0, 254.0, *rpy)
    monkeypatch.setattr(node, "_top_down_rpy_deg", lambda: rpy)
    monkeypatch.setattr(
        node,
        "_target_pose_from_plan_top_down",
        lambda _plan, rpy_deg=None: target,
    )
    monkeypatch.setattr(node, "_top_down_approach_height_mm", lambda: 110.0)
    monkeypatch.setattr(node, "_top_down_min_safe_z_mm", lambda: 350.0)
    monkeypatch.setattr(node, "_top_down_lift_height_mm", lambda: 80.0)
    monkeypatch.setattr(node, "_top_down_lift_to_safe_z", lambda: False)
    monkeypatch.setattr(node, "_top_down_vertical_step_mm", lambda: 1000.0)
    monkeypatch.setattr(node, "_top_down_lateral_step_mm", lambda: 1000.0)

    current = EndPoseMMDeg(0.0, -35.5, 491.1, *rpy)
    waypoints = RobotExecutorNode._build_safe_top_down_waypoints(
        node,
        plan=SimpleNamespace(),
        current_pose=current,
    )

    transit = next(
        pose for name, pose in waypoints if name.startswith("topdown_lift_clear")
    )
    lateral = next(
        pose for name, pose in waypoints if name.startswith("topdown_lateral")
    )
    assert transit.x_mm == pytest.approx(current.x_mm)
    assert transit.y_mm == pytest.approx(current.y_mm)
    assert transit.z_mm == pytest.approx(364.0)
    assert transit.z_mm < current.z_mm
    assert lateral.z_mm == pytest.approx(364.0)


def test_center_horizontal_coarse_vertical_steps_reduce_near_object_replanning(monkeypatch):
    node = RobotExecutorNode.__new__(RobotExecutorNode)
    rpy = (180.0, 85.0, -90.0)
    target = EndPoseMMDeg(10.0, 361.0, 256.0, *rpy)
    monkeypatch.setattr(node, "_top_down_rpy_deg", lambda: rpy)
    monkeypatch.setattr(node, "_target_pose_from_plan_top_down", lambda _plan, rpy_deg=None: target)
    monkeypatch.setattr(node, "_top_down_approach_height_mm", lambda: 110.0)
    monkeypatch.setattr(node, "_top_down_min_safe_z_mm", lambda: 350.0)
    monkeypatch.setattr(node, "_top_down_lift_height_mm", lambda: 80.0)
    monkeypatch.setattr(node, "_top_down_lift_to_safe_z", lambda: False)
    monkeypatch.setattr(node, "_top_down_vertical_step_mm", lambda: 80.0)
    monkeypatch.setattr(node, "_top_down_lateral_step_mm", lambda: 1000.0)

    waypoints = RobotExecutorNode._build_safe_top_down_waypoints(
        node,
        plan=SimpleNamespace(),
        current_pose=EndPoseMMDeg(0.0, 35.5, 491.0, 180.0, 67.77, -89.97),
    )

    descend = [pose for name, pose in waypoints if name.startswith("topdown_descend")]
    lift = [pose for name, pose in waypoints if name.startswith("topdown_lift_object")]
    assert len(descend) == 2
    assert descend[-1] == target
    assert len(lift) == 1


def test_top_down_speed_uses_dashboard_speed_when_guard_is_100(monkeypatch):
    node = RobotExecutorNode.__new__(RobotExecutorNode)
    values = {
        "default_speed_percent": 18.0,
        "top_down_max_speed_percent": 100.0,
    }
    monkeypatch.setattr(node, "get_parameter", lambda name: SimpleNamespace(value=values[name]))

    assert RobotExecutorNode._top_down_speed_percent(node) == 18.0


def test_moveit_pose_command_is_published_once_while_waiting(monkeypatch):
    class DummyMoveIt:
        def __init__(self):
            self.compute_calls = []
            self.publish_calls = []

        def compute_ik(self, pose):
            self.compute_calls.append(pose)
            return [0.1] * 6

        def publish_joint_command(self, joints, speed):
            self.publish_calls.append((list(joints), float(speed)))

    node = RobotExecutorNode.__new__(RobotExecutorNode)
    moveit = DummyMoveIt()
    wait_refresh = []
    monkeypatch.setattr(node, "_robot_client", lambda: SimpleNamespace())
    monkeypatch.setattr(node, "_check_interrupt", lambda: None)
    monkeypatch.setattr(node, "_pose_execution_mode", lambda: "moveit_ik")
    monkeypatch.setattr(node, "_ensure_moveit_pose_mode_ready", lambda: None)
    monkeypatch.setattr(node, "_moveit_ik_executor", lambda: moveit)
    monkeypatch.setattr(
        node,
        "_wait_pose_with_interrupt",
        lambda _cmd, refresh_command=None: wait_refresh.append(refresh_command),
    )
    cmd = MovePoseCommand(
        name="descend",
        pose=EndPoseMMDeg(1.0, 2.0, 3.0, 180.0, 85.0, -90.0),
        speed_percent=8.0,
        timeout_s=5.0,
        pos_tolerance_mm=2.0,
        rot_tolerance_deg=2.0,
    )
    executed = []

    RobotExecutorNode._execute_command(node, cmd, executed)

    assert len(moveit.compute_calls) == 1
    assert moveit.publish_calls == [([0.1] * 6, 8.0)]
    assert wait_refresh == [None]
    assert executed == ["descend"]


def test_center_horizontal_uses_safe_cartesian_strategy(monkeypatch):
    node = RobotExecutorNode.__new__(RobotExecutorNode)
    monkeypatch.setattr(node, "_execution_strategy", lambda: "center_horizontal")

    assert RobotExecutorNode._uses_safe_cartesian_strategy(node) is True


def test_top_down_rpy_variants_keep_primary_first_and_deduplicate(monkeypatch):
    node = RobotExecutorNode.__new__(RobotExecutorNode)
    parameters = {
        "top_down_rpy_deg": [180.0, 60.0, 180.0],
        "top_down_rpy_variants_deg": [
            180.0,
            60.0,
            180.0,
            180.0,
            60.0,
            -90.0,
        ],
    }
    monkeypatch.setattr(node, "get_parameter", lambda name: SimpleNamespace(value=parameters[name]))

    assert RobotExecutorNode._top_down_rpy_variants_deg(node) == [
        (180.0, 60.0, 180.0),
        (180.0, 60.0, -90.0),
    ]


def test_center_horizontal_yaw_follows_offset_target_azimuth(monkeypatch):
    node = RobotExecutorNode.__new__(RobotExecutorNode)
    parameters = {
        "center_horizontal_follow_target_azimuth": True,
        "center_horizontal_reference_azimuth_deg": 90.0,
        "center_horizontal_max_yaw_adjust_deg": 45.0,
    }
    monkeypatch.setattr(node, "_execution_strategy", lambda: "center_horizontal")
    monkeypatch.setattr(node, "_top_down_rpy_variants_deg", lambda: [(180.0, 85.0, -90.0)])
    monkeypatch.setattr(node, "get_parameter", lambda name: SimpleNamespace(value=parameters[name]))
    plan = SimpleNamespace(target_contact_point_base_m=(-0.0695901, 0.4549121, 0.2359))

    variants = RobotExecutorNode._safe_cartesian_rpy_variants(node, plan)

    target_azimuth_deg = np.degrees(np.arctan2(0.4549121, -0.0695901))
    assert variants == pytest.approx([(180.0, 85.0, -90.0 + target_azimuth_deg - 90.0)])


def test_center_horizontal_adaptive_yaw_preserves_contact_point(monkeypatch):
    node = RobotExecutorNode.__new__(RobotExecutorNode)
    parameters = {
        "center_horizontal_follow_target_azimuth": True,
        "center_horizontal_reference_azimuth_deg": 90.0,
        "center_horizontal_max_yaw_adjust_deg": 45.0,
    }
    contact = np.array([-0.0695901, 0.4549121, 0.2359], dtype=np.float64)
    offset = np.array([0.0, 0.0, 0.105], dtype=np.float64)
    plan = SimpleNamespace(
        target_contact_point_base_m=tuple(contact),
        tool_contact_offset_tool_m=tuple(offset),
    )
    monkeypatch.setattr(node, "_execution_strategy", lambda: "center_horizontal")
    monkeypatch.setattr(node, "_top_down_rpy_variants_deg", lambda: [(180.0, 85.0, -90.0)])
    monkeypatch.setattr(node, "_top_down_min_target_z_mm", lambda: 200.0)
    monkeypatch.setattr(node, "get_parameter", lambda name: SimpleNamespace(value=parameters[name]))

    adaptive_rpy = RobotExecutorNode._safe_cartesian_rpy_variants(node, plan)[0]
    target = RobotExecutorNode._target_pose_from_plan_top_down(node, plan, rpy_deg=adaptive_rpy)
    target_translation = np.array([target.x_mm, target.y_mm, target.z_mm]) / 1000.0
    reconstructed_contact = target_translation + rotation_matrix_from_rpy_deg(*adaptive_rpy) @ offset

    assert reconstructed_contact == pytest.approx(contact)


def test_safe_top_down_yaw_follows_offset_target_azimuth(monkeypatch):
    node = RobotExecutorNode.__new__(RobotExecutorNode)
    parameters = {
        "safe_top_down_follow_target_azimuth": True,
        "safe_top_down_reference_azimuth_deg": 90.0,
        "safe_top_down_max_yaw_adjust_deg": 45.0,
    }
    monkeypatch.setattr(node, "_execution_strategy", lambda: "safe_top_down")
    monkeypatch.setattr(node, "_top_down_rpy_variants_deg", lambda: [(180.0, 85.0, -90.0)])
    monkeypatch.setattr(node, "get_parameter", lambda name: SimpleNamespace(value=parameters[name]))
    plan = SimpleNamespace(target_contact_point_base_m=(-0.1569, 0.3927, 0.196))

    variants = RobotExecutorNode._safe_cartesian_rpy_variants(node, plan)

    target_azimuth_deg = np.degrees(np.arctan2(0.3927, -0.1569))
    assert len(variants) == 1
    assert variants[0] == pytest.approx(
        (180.0, 85.0, -90.0 + target_azimuth_deg - 90.0)
    )


def test_safe_top_down_uses_safe_cube_descent_step(monkeypatch):
    node = RobotExecutorNode.__new__(RobotExecutorNode)
    monkeypatch.setattr(node, "_execution_strategy", lambda: "safe_top_down")
    monkeypatch.setattr(
        node,
        "get_parameter",
        lambda name: SimpleNamespace(
            value={
                "safe_top_down_vertical_step_mm": 80.0,
                "top_down_vertical_step_mm": 80.0,
            }[name]
        ),
    )

    assert RobotExecutorNode._top_down_vertical_step_mm(node) == 80.0


def test_safe_top_down_final_descent_uses_slow_contact_speed(monkeypatch):
    node = RobotExecutorNode.__new__(RobotExecutorNode)
    monkeypatch.setattr(
        node,
        "get_parameter",
        lambda name: SimpleNamespace(
            value={"safe_top_down_final_speed_percent": 2.0}[name]
        ),
    )

    assert RobotExecutorNode._safe_top_down_final_speed_percent(node) == 2.0


def test_grasp_plan_message_preserves_tool_contact_geometry():
    candidate = GraspCandidate(
        instance_index=0,
        score=0.9,
        width_m=0.04,
        depth_m=0.02,
        translation_camera_m=(0.0, 0.0, 0.4),
        rotation_camera=np.eye(3),
        object_center_camera_m=None,
        center_offset_m=None,
        raw_grasp=None,
    )
    plan = GraspPlan(
        candidate=candidate,
        target_base_m=(0.1, 0.2, 0.3),
        target_rpy_deg=(180.0, 60.0, 180.0),
        pregrasp_base_m=(0.1, 0.2, 0.4),
        grasp_base_m=(0.1, 0.2, 0.35),
        retreat_base_m=(0.1, 0.2, 0.45),
        within_workspace=True,
        workspace_violations=[],
        target_contact_point_base_m=(0.11, 0.22, 0.24),
        tool_contact_offset_tool_m=(0.001, -0.002, 0.105),
    )

    restored = grasp_plan_from_msg(grasp_plan_to_msg(plan))

    assert restored.target_contact_point_base_m == pytest.approx((0.11, 0.22, 0.24))
    assert restored.tool_contact_offset_tool_m == pytest.approx((0.001, -0.002, 0.105))


def test_ensure_robot_ready_does_not_reenable_connected_enabled_robot(monkeypatch):
    class DummyRobot:
        def __init__(self):
            self.connect_calls = 0
            self.enable_calls = 0

        def connect(self):
            self.connect_calls += 1

        def enable(self):
            self.enable_calls += 1
            return True

    node = RobotExecutorNode.__new__(RobotExecutorNode)
    robot = DummyRobot()

    node._robot_connected = True
    node._robot_enabled = True
    monkeypatch.setattr(node, "_robot_client", lambda: robot)
    monkeypatch.setattr(node, "_auto_enable", lambda: True)

    RobotExecutorNode._ensure_robot_ready(node)

    assert robot.connect_calls == 0
    assert robot.enable_calls == 0
    assert node._robot_connected is True
    assert node._robot_enabled is True


def test_ensure_robot_ready_enables_only_once_after_connect(monkeypatch):
    class DummyRobot:
        def __init__(self):
            self.connect_calls = 0
            self.enable_calls = 0

        def connect(self):
            self.connect_calls += 1

        def enable(self):
            self.enable_calls += 1
            return True

    node = RobotExecutorNode.__new__(RobotExecutorNode)
    robot = DummyRobot()

    node._robot_connected = False
    node._robot_enabled = False
    monkeypatch.setattr(node, "_robot_client", lambda: robot)
    monkeypatch.setattr(node, "_auto_enable", lambda: True)

    RobotExecutorNode._ensure_robot_ready(node)
    RobotExecutorNode._ensure_robot_ready(node)

    assert robot.connect_calls == 1
    assert robot.enable_calls == 1
    assert node._robot_connected is True
    assert node._robot_enabled is True


def test_set_gripper_closed_raises_when_effort_wait_times_out(monkeypatch):
    class DummyRobot:
        def __init__(self):
            self.close_effort = None

        def close_gripper(self, effort_nm=None):
            self.close_effort = effort_nm

        def wait_for_gripper_effort(self, target_effort_nm, timeout_s):
            return False

    node = RobotExecutorNode.__new__(RobotExecutorNode)
    robot = DummyRobot()

    node._moveit_ik = None
    monkeypatch.setattr(node, "_robot_client", lambda: robot)
    monkeypatch.setattr(node, "_default_gripper_close_effort_nm", lambda: 0.6)

    with pytest.raises(TimeoutError, match="close gripper wait timeout"):
        RobotExecutorNode._set_gripper_closed(node)

    assert robot.close_effort == 0.6


def test_lock_gripper_command_is_remembered_before_moveit_initializes():
    node = RobotExecutorNode.__new__(RobotExecutorNode)
    node._moveit_ik = None
    node._desired_gripper_command = None

    RobotExecutorNode._lock_gripper_command(node, position_m=0.0, effort_nm=0.6)

    assert node._desired_gripper_command == (0.0, 0.6)


def test_set_gripper_open_locks_moveit_gripper_after_position_reached(monkeypatch):
    class DummyRobot:
        def open_gripper(self, open_mm=None, effort_nm=None):
            self.open_mm = open_mm
            self.open_effort = effort_nm

        def wait_for_gripper(self, target_mm, tol_mm, timeout_s):
            self.wait_args = (target_mm, tol_mm, timeout_s)
            return True

    class DummyMoveIt:
        def __init__(self):
            self.locked = None

        def lock_gripper_command(self, *, position_m, effort_nm):
            self.locked = (position_m, effort_nm)

    node = RobotExecutorNode.__new__(RobotExecutorNode)
    robot = DummyRobot()
    moveit = DummyMoveIt()

    node._moveit_ik = moveit
    monkeypatch.setattr(node, "_robot_client", lambda: robot)
    monkeypatch.setattr(node, "_default_gripper_open_mm", lambda: 70.0)
    monkeypatch.setattr(node, "_default_gripper_close_effort_nm", lambda: 0.6)

    RobotExecutorNode._set_gripper_open(node)

    assert robot.open_mm == 70.0
    assert robot.open_effort is None
    assert robot.wait_args == (70.0, 5.0, 4.0)
    assert moveit.locked == (0.07, 0.6)


def test_set_gripper_closed_locks_moveit_gripper_after_effort_reached(monkeypatch):
    class DummyRobot:
        def close_gripper(self, effort_nm=None):
            self.close_effort = effort_nm

        def wait_for_gripper_effort(self, target_effort_nm, timeout_s):
            return True

    class DummyMoveIt:
        def __init__(self):
            self.locked = None

        def lock_gripper_command(self, *, position_m, effort_nm):
            self.locked = (position_m, effort_nm)

    node = RobotExecutorNode.__new__(RobotExecutorNode)
    robot = DummyRobot()
    moveit = DummyMoveIt()

    node._moveit_ik = moveit
    monkeypatch.setattr(node, "_robot_client", lambda: robot)
    monkeypatch.setattr(node, "_default_gripper_close_effort_nm", lambda: 0.6)

    RobotExecutorNode._set_gripper_closed(node)

    assert moveit.locked == (0.0, 0.6)


def test_execute_grasp_plan_returns_execution_trace(monkeypatch):
    node = RobotExecutorNode.__new__(RobotExecutorNode)

    monkeypatch.setattr(node, "_reset_interrupt_flags", lambda: None)
    monkeypatch.setattr(node, "_ensure_robot_ready", lambda: None)
    monkeypatch.setattr(node, "_execution_strategy", lambda: "planned_waypoints")
    monkeypatch.setattr(node, "_enable_pregrasp", lambda: True)
    monkeypatch.setattr(node, "_default_speed", lambda: 35.0)
    monkeypatch.setattr(node, "_default_gripper_open_mm", lambda: 70.0)
    monkeypatch.setattr(node, "_default_gripper_close_effort_nm", lambda: 0.6)
    monkeypatch.setattr(node, "_plan_pose_timeout_s", lambda: 12.0)
    monkeypatch.setattr(node, "_use_handoff_pose", lambda: False)
    monkeypatch.setattr(node, "_configured_pose", lambda _name: None)
    monkeypatch.setattr(node, "_publish_service_result", lambda _payload: None)
    monkeypatch.setattr(node, "_assert_execution_waypoints_safe", lambda _plan: None)

    feedback = {
        "feedback_pose_mm_deg": {
            "x_mm": 101.0,
            "y_mm": 202.0,
            "z_mm": 303.0,
            "roll_deg": 4.0,
            "pitch_deg": 5.0,
            "yaw_deg": 6.0,
        },
        "feedback_gripper": {
            "angle_mm": 70.0,
            "effort_nm": 0.2,
            "enabled": True,
        },
        "feedback_arm_status": "control_mode=CAN",
    }
    monkeypatch.setattr(node, "_capture_execution_feedback", lambda: dict(feedback))

    def fake_move_pose_sync(*, name, pose, speed_percent, timeout_s):
        return pose

    monkeypatch.setattr(node, "_move_pose_sync", fake_move_pose_sync)
    monkeypatch.setattr(node, "_set_gripper_open", lambda: None)
    monkeypatch.setattr(node, "_set_gripper_closed", lambda: None)

    pose = EndPoseMMDeg(100.0, 200.0, 300.0, 1.0, 2.0, 3.0)
    candidate = GraspCandidate(
        instance_index=0,
        score=0.95,
        width_m=0.04,
        depth_m=0.02,
        translation_camera_m=(0.1, 0.2, 0.3),
        rotation_camera=np.eye(3),
        object_center_camera_m=None,
        center_offset_m=None,
        raw_grasp=None,
    )
    plan = GraspPlan(
        candidate=candidate,
        target_base_m=(0.1, 0.2, 0.3),
        target_rpy_deg=(1.0, 2.0, 3.0),
        pregrasp_base_m=(0.1, 0.2, 0.35),
        grasp_base_m=(0.1, 0.2, 0.29),
        retreat_base_m=(0.1, 0.2, 0.4),
        within_workspace=True,
        workspace_violations=[],
    )
    request = SimpleNamespace(
        run_id="run-123",
        execute=True,
        move_home_after=False,
        plan=grasp_plan_to_msg(plan),
    )
    response = SimpleNamespace(success=False, message="", execution_json="")

    result = RobotExecutorNode._handle_execute_grasp_plan(node, request, response)
    payload = json.loads(result.execution_json)

    assert result.success is True
    assert payload["status"] == "ok"
    assert [step["step_name"] for step in payload["execution_trace"]] == [
        "open_gripper",
        "pregrasp",
        "grasp",
        "target",
        "close_gripper",
        "retreat",
    ]
    assert payload["execution_trace"][0]["feedback_gripper"]["angle_mm"] == 70.0
    assert payload["execution_trace"][3]["feedback_pose_mm_deg"]["z_mm"] == 303.0


def test_execute_grasp_plan_dry_run_returns_waypoint_validation_details(monkeypatch):
    node = RobotExecutorNode.__new__(RobotExecutorNode)

    monkeypatch.setattr(node, "_reset_interrupt_flags", lambda: None)
    monkeypatch.setattr(node, "_ensure_robot_ready", lambda: None)
    monkeypatch.setattr(node, "_publish_service_result", lambda _payload: None)
    monkeypatch.setattr(
        node,
        "_validate_grasp_plan_kinematics",
        lambda _plan: {
            "robot_validation_result": "rejected_by_robot_validation",
            "robot_validation_stage": "grasp",
            "ik_error_type": "timeout",
            "ik_error_message": "MoveIt IK request timed out: /compute_ik",
            "validated_waypoints": ["pregrasp"],
            "waypoint_results": [
                {"stage": "pregrasp", "status": "ok"},
                {
                    "stage": "grasp",
                    "status": "failed",
                    "ik_error_type": "timeout",
                    "ik_error_message": "MoveIt IK request timed out: /compute_ik",
                },
            ],
        },
    )

    candidate = GraspCandidate(
        instance_index=0,
        score=0.95,
        width_m=0.04,
        depth_m=0.02,
        translation_camera_m=(0.1, 0.2, 0.3),
        rotation_camera=np.eye(3),
        object_center_camera_m=None,
        center_offset_m=None,
        raw_grasp=None,
    )
    plan = GraspPlan(
        candidate=candidate,
        target_base_m=(0.1, 0.2, 0.3),
        target_rpy_deg=(1.0, 2.0, 3.0),
        pregrasp_base_m=(0.1, 0.2, 0.35),
        grasp_base_m=(0.1, 0.2, 0.3),
        retreat_base_m=(0.1, 0.2, 0.4),
        within_workspace=True,
        workspace_violations=[],
    )
    request = SimpleNamespace(
        run_id="run-dry",
        execute=False,
        move_home_after=False,
        plan=grasp_plan_to_msg(plan),
    )
    response = SimpleNamespace(success=False, message="", execution_json="")

    result = RobotExecutorNode._handle_execute_grasp_plan(node, request, response)
    payload = json.loads(result.execution_json)

    assert result.success is False
    assert payload["robot_validation_stage"] == "grasp"
    assert payload["ik_error_type"] == "timeout"
    assert payload["waypoint_results"][0]["stage"] == "pregrasp"


def test_execute_grasp_plan_rejects_degenerate_waypoints_before_motion(monkeypatch):
    node = RobotExecutorNode.__new__(RobotExecutorNode)

    monkeypatch.setattr(node, "_reset_interrupt_flags", lambda: None)
    monkeypatch.setattr(node, "_ensure_robot_ready", lambda: None)
    monkeypatch.setattr(node, "_execution_strategy", lambda: "planned_waypoints")
    monkeypatch.setattr(node, "_publish_service_result", lambda _payload: None)
    monkeypatch.setattr(node, "_reject_degenerate_grasp_waypoints", lambda: True)
    monkeypatch.setattr(node, "_enable_pregrasp", lambda: False)
    monkeypatch.setattr(node, "_min_grasp_approach_offset_m", lambda: 0.005)
    monkeypatch.setattr(node, "_min_retreat_lift_m", lambda: 0.03)

    move_calls = []
    monkeypatch.setattr(node, "_move_pose_sync", lambda **_kwargs: move_calls.append(_kwargs))

    candidate = GraspCandidate(
        instance_index=0,
        score=0.95,
        width_m=0.04,
        depth_m=0.02,
        translation_camera_m=(0.1, 0.2, 0.3),
        rotation_camera=np.eye(3),
        object_center_camera_m=None,
        center_offset_m=None,
        raw_grasp=None,
    )
    plan = GraspPlan(
        candidate=candidate,
        target_base_m=(0.1, 0.2, 0.3),
        target_rpy_deg=(1.0, 2.0, 3.0),
        pregrasp_base_m=(0.1, 0.2, 0.3),
        grasp_base_m=(0.1, 0.2, 0.3),
        retreat_base_m=(0.1, 0.2, 0.3),
        within_workspace=True,
        workspace_violations=[],
    )
    request = SimpleNamespace(
        run_id="run-unsafe",
        execute=True,
        move_home_after=False,
        plan=grasp_plan_to_msg(plan),
    )
    response = SimpleNamespace(success=False, message="", execution_json="")

    result = RobotExecutorNode._handle_execute_grasp_plan(node, request, response)
    payload = json.loads(result.execution_json)

    assert result.success is False
    assert "unsafe grasp plan waypoints" in result.message
    assert "grasp->target distance" in result.message
    assert payload["status"] == "failed"
    assert move_calls == []
