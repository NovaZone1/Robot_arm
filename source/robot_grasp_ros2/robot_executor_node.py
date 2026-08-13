from __future__ import annotations

import json
import math
import threading
import time
import traceback

import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from robot_grasp_msgs.srv import (
    ExecuteGraspPlan,
    ExecuteNamedPose,
    ExecutePlacePlan,
    GetRobotState,
    StopRobot,
)
from robot_grasp_ros2.distributed_utils import (
    grasp_plan_from_msg,
    json_dumps,
    make_latched_qos,
    pose6d_from_end_pose,
    pose6d_to_end_pose,
)
from src.robot import FakeRobotArmClient, MoveItIkConfig, MoveItIkExecutor, Ros2PiperClient
from src.robot.client import RobotArmClient
from src.robot.executor_models import (
    CloseGripperCommand,
    MovePoseCommand,
    OpenGripperCommand,
    RobotExecutionPlan,
    RobotExecutionResult,
    SleepCommand,
    parse_execution_plan_json,
)
from src.robot.motion_tolerances import effective_position_tolerance_mm, wait_until_pose_goal
from src.robot.moveit_ik import ensure_moveit_pose_mode_supported
from src.robot.plan_validation import validate_grasp_plan_waypoints_detailed
from src.robot.types import EndPoseMMDeg
from src.utils.transforms import rotation_matrix_from_rpy_deg


class RobotExecutorNode(Node):
    """Distributed-friendly robot executor skeleton with fake-first behavior."""

    def __init__(self) -> None:
        super().__init__("robot_executor")
        self._declare_parameters()

        text_qos = make_latched_qos(depth=20)
        self._status_pub = self.create_publisher(String, "~/status", text_qos)
        self._result_pub = self.create_publisher(String, "~/result_json", text_qos)

        self.create_subscription(String, "~/plan_request", self._handle_plan_request, 20)
        self.create_service(Trigger, "~/run", self._handle_run)
        self.create_service(Trigger, "~/stop", self._handle_stop)
        self.create_service(Trigger, "~/cancel", self._handle_cancel)
        self.create_service(Trigger, "~/probe", self._handle_probe)
        self.create_service(Trigger, "~/open_gripper", self._handle_open_gripper)
        self.create_service(GetRobotState, "~/get_state", self._handle_get_state)
        self.create_service(ExecuteNamedPose, "~/execute_named_pose", self._handle_execute_named_pose)
        self.create_service(ExecuteGraspPlan, "~/execute_grasp_plan", self._handle_execute_grasp_plan)
        self.create_service(ExecutePlacePlan, "~/execute_place_plan", self._handle_execute_place_plan)
        self.create_service(StopRobot, "~/stop_robot", self._handle_stop_robot)

        self._lock = threading.Lock()
        self._runner: threading.Thread | None = None
        self._active_plan: RobotExecutionPlan | None = None
        self._active_plan_json = ""
        self._cancel_requested = False
        self._stop_requested = False

        self._robot: RobotArmClient | None = None
        self._robot_connected = False
        self._robot_enabled = False
        self._moveit_ik: MoveItIkExecutor | None = None
        self._desired_gripper_command: tuple[float, float] | None = None

        self._publish_status("idle")

    def _declare_parameters(self) -> None:
        self.declare_parameter("robot_backend", "fake")
        self.declare_parameter("auto_enable", True)
        self.declare_parameter("disconnect_after_run", False)
        self.declare_parameter("poll_interval_s", 0.05)
        self.declare_parameter("default_speed_percent", 40.0)
        self.declare_parameter("default_gripper_open_mm", 70.0)
        self.declare_parameter("default_gripper_close_effort_nm", 0.6)
        self.declare_parameter("enable_pregrasp", False)
        self.declare_parameter("use_handoff_pose", True)
        self.declare_parameter("execution_strategy", "safe_top_down")
        self.declare_parameter("pose_execution_mode", "direct")
        self.declare_parameter("moveit_ik_service", "/compute_ik")
        self.declare_parameter("moveit_group_name", "arm")
        self.declare_parameter("moveit_ik_link_name", "link6")
        self.declare_parameter("moveit_base_frame", "base_link")
        self.declare_parameter("moveit_ik_timeout_s", 5.0)
        self.declare_parameter("pose_goal_hold_s", 0.8)
        self.declare_parameter("motion_arrived_pose_fallback_enabled", True)
        self.declare_parameter("motion_arrived_pose_fallback_pos_tolerance_mm", 2.0)
        self.declare_parameter("motion_arrived_pose_fallback_rot_tolerance_deg", 2.0)
        self.declare_parameter("motion_arrived_pose_fallback_hold_s", 0.6)
        self.declare_parameter("named_pose_timeout_s", 20.0)
        self.declare_parameter("plan_pose_timeout_s", 30.0)
        self.declare_parameter("motion_progress_extends_timeout", False)
        self.declare_parameter("motion_progress_position_epsilon_mm", 0.5)
        self.declare_parameter("motion_progress_rotation_epsilon_deg", 0.25)
        self.declare_parameter("move_pos_tolerance_mm", 20.0)
        self.declare_parameter("move_rot_tolerance_deg", 10.0)
        self.declare_parameter("reject_degenerate_grasp_waypoints", True)
        self.declare_parameter("min_grasp_approach_offset_m", 0.005)
        self.declare_parameter("min_retreat_lift_m", 0.03)
        self.declare_parameter("top_down_rpy_deg", [180.0, 60.0, 180.0])
        self.declare_parameter(
            "top_down_rpy_variants_deg",
            [180.0, 60.0, 90.0, 180.0, 60.0, 0.0, 180.0, 60.0, -90.0],
        )
        self.declare_parameter("center_horizontal_follow_target_azimuth", True)
        self.declare_parameter("center_horizontal_reference_azimuth_deg", 90.0)
        self.declare_parameter("center_horizontal_max_yaw_adjust_deg", 45.0)
        self.declare_parameter("safe_top_down_follow_target_azimuth", True)
        self.declare_parameter("safe_top_down_reference_azimuth_deg", 90.0)
        self.declare_parameter("safe_top_down_max_yaw_adjust_deg", 45.0)
        self.declare_parameter("top_down_min_safe_z_mm", 300.0)
        self.declare_parameter("top_down_min_target_z_mm", 300.0)
        self.declare_parameter("top_down_approach_height_mm", 110.0)
        self.declare_parameter("top_down_lift_height_mm", 120.0)
        self.declare_parameter("top_down_lift_to_safe_z", True)
        self.declare_parameter("top_down_lateral_step_mm", 35.0)
        self.declare_parameter("top_down_vertical_step_mm", 25.0)
        self.declare_parameter("safe_top_down_vertical_step_mm", 80.0)
        self.declare_parameter("safe_top_down_final_speed_percent", 2.0)
        self.declare_parameter("top_down_max_speed_percent", 10.0)
        self.declare_parameter("placement_box_outer_size_m", [0.180, 0.132, 0.087])
        self.declare_parameter("placement_slot_count", 6)
        self.declare_parameter("placement_box_size_tolerance_m", 0.005)
        self.declare_parameter("placement_min_label_confidence", 0.42)
        self.declare_parameter("placement_min_vertical_clearance_mm", 50.0)
        self.declare_parameter("placement_final_speed_percent", 2.0)
        self.declare_parameter("handoff_pose", [200.0, 20.0, 300.0, 10.0, 120.0, 0.0])
        self.declare_parameter("home_pose", [57.0, 0.0, 215.0, 0.0, 85.0, 0.0])
        self.declare_parameter("move_to_post_grasp_pose", True)
        self.declare_parameter("post_grasp_safe_lift_z_mm", 350.0)
        self.declare_parameter(
            "post_grasp_pose",
            [256.885, 0.0, 400.0, 0.0, 84.939, 0.0],
        )
        self.declare_parameter("plan_json", "")

    def _publish_status(self, text: str) -> None:
        self.get_logger().info(text)
        self._status_pub.publish(String(data=text))

    def _publish_result(self, result: RobotExecutionResult) -> None:
        self._result_pub.publish(String(data=result.to_json()))

    def _backend(self) -> str:
        return str(self.get_parameter("robot_backend").value or "fake").strip().lower()

    def _auto_enable(self) -> bool:
        return bool(self.get_parameter("auto_enable").value)

    def _disconnect_after_run(self) -> bool:
        return bool(self.get_parameter("disconnect_after_run").value)

    def _poll_interval(self) -> float:
        return max(0.01, float(self.get_parameter("poll_interval_s").value))

    def _default_speed(self) -> float:
        return float(self.get_parameter("default_speed_percent").value)

    def _default_gripper_open_mm(self) -> float:
        return float(self.get_parameter("default_gripper_open_mm").value)

    def _default_gripper_close_effort_nm(self) -> float:
        return float(self.get_parameter("default_gripper_close_effort_nm").value)

    def _enable_pregrasp(self) -> bool:
        return bool(self.get_parameter("enable_pregrasp").value)

    def _use_handoff_pose(self) -> bool:
        return bool(self.get_parameter("use_handoff_pose").value)

    def _move_to_post_grasp_pose_enabled(self) -> bool:
        try:
            return bool(self.get_parameter("move_to_post_grasp_pose").value)
        except Exception:
            # Some offline/unit-test harnesses construct the executor without
            # running Node.__init__. Real nodes always declare this parameter.
            return False

    def _post_grasp_safe_lift_z_mm(self) -> float:
        return max(
            self._top_down_min_target_z_mm(),
            float(self.get_parameter("post_grasp_safe_lift_z_mm").value),
        )

    def _execution_strategy(self) -> str:
        return str(self.get_parameter("execution_strategy").value or "planned_waypoints").strip().lower()

    def _uses_safe_cartesian_strategy(self) -> bool:
        return self._execution_strategy() in {"safe_top_down", "center_horizontal"}

    def _pose_execution_mode(self) -> str:
        return str(self.get_parameter("pose_execution_mode").value or "direct").strip().lower()

    def _named_pose_timeout_s(self) -> float:
        return max(0.1, float(self.get_parameter("named_pose_timeout_s").value or 20.0))

    def _plan_pose_timeout_s(self) -> float:
        return max(0.1, float(self.get_parameter("plan_pose_timeout_s").value or 30.0))

    def _move_pos_tolerance_mm(self) -> float:
        return float(self.get_parameter("move_pos_tolerance_mm").value or 20.0)

    def _move_rot_tolerance_deg(self) -> float:
        return float(self.get_parameter("move_rot_tolerance_deg").value or 10.0)

    def _reject_degenerate_grasp_waypoints(self) -> bool:
        return bool(self.get_parameter("reject_degenerate_grasp_waypoints").value)

    def _min_grasp_approach_offset_m(self) -> float:
        return max(0.0, float(self.get_parameter("min_grasp_approach_offset_m").value or 0.0))

    def _min_retreat_lift_m(self) -> float:
        return max(0.0, float(self.get_parameter("min_retreat_lift_m").value or 0.0))

    def _top_down_rpy_deg(self) -> tuple[float, float, float]:
        values = list(self.get_parameter("top_down_rpy_deg").value or [180.0, 60.0, 180.0])
        if len(values) != 3:
            raise RuntimeError(f"top_down_rpy_deg must contain 3 values, got {len(values)}")
        return (float(values[0]), float(values[1]), float(values[2]))

    def _top_down_rpy_variants_deg(self) -> list[tuple[float, float, float]]:
        primary = self._top_down_rpy_deg()
        values = list(self.get_parameter("top_down_rpy_variants_deg").value or [])
        if len(values) % 3 != 0:
            raise RuntimeError(
                "top_down_rpy_variants_deg must contain a multiple of 3 values, "
                f"got {len(values)}"
            )
        variants = [primary]
        variants.extend(
            (float(values[index]), float(values[index + 1]), float(values[index + 2]))
            for index in range(0, len(values), 3)
        )
        unique: list[tuple[float, float, float]] = []
        for rpy in variants:
            if not any(all(abs(a - b) < 1e-9 for a, b in zip(rpy, item)) for item in unique):
                unique.append(rpy)
        return unique

    @staticmethod
    def _normalize_angle_deg(value: float) -> float:
        return (float(value) + 180.0) % 360.0 - 180.0

    def _safe_cartesian_rpy_variants(self, plan) -> list[tuple[float, float, float]]:
        variants = self._top_down_rpy_variants_deg()
        strategy = self._execution_strategy()
        if strategy == "center_horizontal":
            follow_parameter = "center_horizontal_follow_target_azimuth"
            reference_parameter = "center_horizontal_reference_azimuth_deg"
            max_adjust_parameter = "center_horizontal_max_yaw_adjust_deg"
        elif strategy == "safe_top_down":
            follow_parameter = "safe_top_down_follow_target_azimuth"
            reference_parameter = "safe_top_down_reference_azimuth_deg"
            max_adjust_parameter = "safe_top_down_max_yaw_adjust_deg"
        else:
            return variants
        if not bool(self.get_parameter(follow_parameter).value):
            return variants
        contact = plan.target_contact_point_base_m
        if contact is None or len(contact) < 2:
            raise RuntimeError("center_horizontal target azimuth requires target contact point")
        x_m = float(contact[0])
        y_m = float(contact[1])
        if math.hypot(x_m, y_m) <= 1e-6:
            raise RuntimeError("center_horizontal target contact point is too close to base axis")
        target_azimuth_deg = math.degrees(math.atan2(y_m, x_m))
        reference_azimuth_deg = float(self.get_parameter(reference_parameter).value)
        yaw_delta_deg = self._normalize_angle_deg(target_azimuth_deg - reference_azimuth_deg)
        max_adjust_deg = max(
            0.0,
            float(self.get_parameter(max_adjust_parameter).value),
        )
        yaw_delta_deg = max(-max_adjust_deg, min(max_adjust_deg, yaw_delta_deg))
        return [
            (roll_deg, pitch_deg, self._normalize_angle_deg(yaw_deg + yaw_delta_deg))
            for roll_deg, pitch_deg, yaw_deg in variants
        ]

    def _top_down_min_safe_z_mm(self) -> float:
        return float(self.get_parameter("top_down_min_safe_z_mm").value or 300.0)

    def _top_down_min_target_z_mm(self) -> float:
        return float(self.get_parameter("top_down_min_target_z_mm").value or 300.0)

    def _top_down_approach_height_mm(self) -> float:
        return max(10.0, float(self.get_parameter("top_down_approach_height_mm").value or 110.0))

    def _top_down_lift_height_mm(self) -> float:
        return max(10.0, float(self.get_parameter("top_down_lift_height_mm").value or 120.0))

    def _top_down_lift_to_safe_z(self) -> bool:
        return bool(self.get_parameter("top_down_lift_to_safe_z").value)

    def _top_down_lateral_step_mm(self) -> float:
        return max(5.0, float(self.get_parameter("top_down_lateral_step_mm").value or 35.0))

    def _top_down_vertical_step_mm(self) -> float:
        if self._execution_strategy() == "safe_top_down":
            return max(
                5.0,
                float(self.get_parameter("safe_top_down_vertical_step_mm").value or 80.0),
            )
        return max(5.0, float(self.get_parameter("top_down_vertical_step_mm").value or 25.0))

    def _safe_top_down_final_speed_percent(self) -> float:
        return max(
            1.0,
            float(self.get_parameter("safe_top_down_final_speed_percent").value or 2.0),
        )

    def _top_down_speed_percent(self) -> float:
        return min(self._default_speed(), max(1.0, float(self.get_parameter("top_down_max_speed_percent").value or 10.0)))

    def _configured_pose(self, parameter_name: str) -> EndPoseMMDeg | None:
        values = list(self.get_parameter(parameter_name).value or [])
        if not values:
            return None
        if len(values) != 6:
            raise RuntimeError(f"{parameter_name} must contain 6 values, got {len(values)}")
        pose_values = [float(value) for value in values]
        return EndPoseMMDeg(
            x_mm=pose_values[0],
            y_mm=pose_values[1],
            z_mm=pose_values[2],
            roll_deg=pose_values[3],
            pitch_deg=pose_values[4],
            yaw_deg=pose_values[5],
        )

    @staticmethod
    def _pose_to_dict(pose: EndPoseMMDeg | None) -> dict[str, float] | None:
        if pose is None:
            return None
        return {
            "x_mm": float(pose.x_mm),
            "y_mm": float(pose.y_mm),
            "z_mm": float(pose.z_mm),
            "roll_deg": float(pose.roll_deg),
            "pitch_deg": float(pose.pitch_deg),
            "yaw_deg": float(pose.yaw_deg),
        }

    @staticmethod
    def _point_distance_m(
        first: tuple[float, float, float],
        second: tuple[float, float, float],
    ) -> float:
        return sum((float(a) - float(b)) ** 2 for a, b in zip(first, second)) ** 0.5

    def _assert_execution_waypoints_safe(self, plan) -> None:
        if not self._reject_degenerate_grasp_waypoints():
            return

        violations: list[str] = []
        min_approach = self._min_grasp_approach_offset_m()
        min_retreat_lift = self._min_retreat_lift_m()

        if self._enable_pregrasp():
            pregrasp_distance = self._point_distance_m(plan.pregrasp_base_m, plan.grasp_base_m)
            if pregrasp_distance < min_approach:
                violations.append(
                    f"pregrasp->grasp distance {pregrasp_distance:.3f}m below minimum {min_approach:.3f}m"
                )

        grasp_distance = self._point_distance_m(plan.grasp_base_m, plan.target_base_m)
        if grasp_distance < min_approach:
            violations.append(
                f"grasp->target distance {grasp_distance:.3f}m below minimum {min_approach:.3f}m"
            )

        retreat_lift = float(plan.retreat_base_m[2]) - float(plan.grasp_base_m[2])
        if retreat_lift < min_retreat_lift:
            violations.append(
                f"retreat lift {retreat_lift:.3f}m below minimum {min_retreat_lift:.3f}m"
            )

        if violations:
            raise RuntimeError("unsafe grasp plan waypoints: " + "; ".join(violations))

    @staticmethod
    def _gripper_to_dict(gripper) -> dict[str, object] | None:
        if gripper is None:
            return None
        return {
            "angle_mm": float(gripper.angle_mm),
            "effort_nm": float(gripper.effort_nm),
            "enabled": bool(gripper.enabled),
        }

    def _robot_client(self) -> RobotArmClient:
        if self._robot is not None:
            return self._robot

        backend = self._backend()
        if backend == "fake":
            robot: RobotArmClient = FakeRobotArmClient()
        elif backend == "ros2":
            client = Ros2PiperClient()
            client.attach_ros_node(self)
            robot = client
        else:
            raise RuntimeError(f"unsupported robot backend: {backend}")

        self._robot = robot
        return robot

    def _ensure_robot_ready(self) -> None:
        robot = self._robot_client()
        if not self._robot_connected:
            robot.connect()
            self._robot_connected = True
            self._robot_enabled = False
        if self._auto_enable() and not self._robot_enabled:
            if not robot.enable():
                raise RuntimeError("robot enable failed")
            self._robot_enabled = True

    def _read_pose_with_retry(
        self,
        *,
        attempts: int = 2,
        retry_delay_s: float = 0.2,
    ):
        last_error: Exception | None = None
        total_attempts = max(1, int(attempts))
        for attempt in range(total_attempts):
            try:
                return self._robot_client().read_end_pose_mm_deg()
            except Exception as exc:
                last_error = exc
                if attempt + 1 >= total_attempts:
                    break
                time.sleep(max(0.0, float(retry_delay_s)))
        if last_error is None:
            raise RuntimeError("failed to read robot pose")
        raise last_error

    def _release_robot(self) -> None:
        if self._robot is None:
            return
        try:
            self._robot.disconnect()
        finally:
            self._robot_connected = False
            self._robot_enabled = False

    def _moveit_ik_executor(self) -> MoveItIkExecutor:
        if self._moveit_ik is None:
            self._moveit_ik = MoveItIkExecutor(
                self,
                config=MoveItIkConfig(
                    ik_service=str(self.get_parameter("moveit_ik_service").value or "/compute_ik"),
                    joint_state_topic="/joint_states_feedback",
                    joint_command_topic="/joint_ctrl_single",
                    group_name=str(self.get_parameter("moveit_group_name").value or "arm"),
                    ik_link_name=str(self.get_parameter("moveit_ik_link_name").value or "link6"),
                    base_frame=str(self.get_parameter("moveit_base_frame").value or "base_link"),
                    timeout_s=float(self.get_parameter("moveit_ik_timeout_s").value),
                ),
            )
            desired_gripper_command = getattr(self, "_desired_gripper_command", None)
            if desired_gripper_command is not None:
                self._moveit_ik.lock_gripper_command(
                    position_m=desired_gripper_command[0],
                    effort_nm=desired_gripper_command[1],
                )
        return self._moveit_ik

    def _lock_gripper_command(self, *, position_m: float, effort_nm: float) -> None:
        command = (float(position_m), float(effort_nm))
        self._desired_gripper_command = command
        moveit_ik = getattr(self, "_moveit_ik", None)
        if moveit_ik is not None:
            moveit_ik.lock_gripper_command(
                position_m=command[0],
                effort_nm=command[1],
            )

    def _ensure_moveit_pose_mode_ready(self) -> None:
        ensure_moveit_pose_mode_supported(self._backend())

    def _handle_plan_request(self, msg: String) -> None:
        try:
            plan = parse_execution_plan_json(msg.data)
        except Exception as exc:
            self._publish_status(f"plan rejected: {exc}")
            return
        with self._lock:
            self._active_plan = plan
            self._active_plan_json = msg.data
        self._publish_status(f"plan accepted: plan_id={plan.plan_id} commands={len(plan.commands)}")

    def _handle_run(self, _request, response):
        with self._lock:
            active = self._runner is not None and self._runner.is_alive()
            plan = self._active_plan
            plan_json = self._active_plan_json
            self._cancel_requested = False
            self._stop_requested = False
        if active:
            response.success = False
            response.message = "executor is already running"
            return response

        if plan is None:
            plan_param = str(self.get_parameter("plan_json").value or "").strip() if self.has_parameter("plan_json") else ""
            if not plan_param:
                response.success = False
                response.message = "no plan available; publish to ~/plan_request first"
                return response
            try:
                plan = parse_execution_plan_json(plan_param)
                plan_json = plan_param
            except Exception as exc:
                response.success = False
                response.message = f"invalid plan_json parameter: {exc}"
                return response

        worker = threading.Thread(
            target=self._run_thread,
            args=(plan, plan_json),
            daemon=True,
            name="robot-executor-runner",
        )
        with self._lock:
            self._runner = worker
        worker.start()
        response.success = True
        response.message = f"run accepted: plan_id={plan.plan_id}"
        return response

    def _handle_stop(self, _request, response):
        with self._lock:
            running = self._runner is not None and self._runner.is_alive()
            self._stop_requested = True
            self._cancel_requested = True
        if not running:
            response.success = False
            response.message = "executor is not running"
            return response
        if self._robot is not None:
            try:
                self._robot.emergency_stop()
                self._robot_enabled = False
            except Exception as exc:
                response.success = False
                response.message = f"stop request failed: {exc}"
                return response
        response.success = True
        response.message = "stop requested"
        return response

    def _handle_cancel(self, _request, response):
        with self._lock:
            running = self._runner is not None and self._runner.is_alive()
            self._cancel_requested = True
        if not running:
            response.success = False
            response.message = "executor is not running"
            return response
        response.success = True
        response.message = "cancel requested"
        return response

    def _handle_probe(self, _request, response):
        try:
            self._ensure_robot_ready()
            pose = self._read_pose_with_retry()
            arm_status = self._robot_client().format_arm_status()
            response.success = True
            response.message = (
                "probe completed: "
                f"pose=({pose.x_mm:.2f},{pose.y_mm:.2f},{pose.z_mm:.2f},{pose.roll_deg:.2f},"
                f"{pose.pitch_deg:.2f},{pose.yaw_deg:.2f}) "
                f"status={arm_status}"
            )
        except Exception as exc:
            response.success = False
            response.message = f"probe failed: {exc}"
        return response

    def _check_interrupt(self) -> None:
        with self._lock:
            if self._stop_requested:
                raise RuntimeError("execution stopped")
            if self._cancel_requested:
                raise RuntimeError("execution cancelled")

    def _reset_interrupt_flags(self) -> None:
        with self._lock:
            self._stop_requested = False
            self._cancel_requested = False

    def _wait_pose_with_interrupt(self, cmd: MovePoseCommand, refresh_command=None) -> None:
        robot = self._robot_client()
        require_motion_arrived = self._backend() == "ros2"
        post_goal_hold_s = float(self.get_parameter("pose_goal_hold_s").value or 0.0)
        wait_until_pose_goal(
            target=cmd.pose,
            timeout_s=cmd.timeout_s,
            poll_interval_s=self._poll_interval(),
            pos_tolerance_mm=cmd.pos_tolerance_mm,
            rot_tolerance_deg=cmd.rot_tolerance_deg,
            post_goal_hold_s=post_goal_hold_s if require_motion_arrived else 0.0,
            check_interrupt=self._check_interrupt,
            refresh_command=refresh_command,
            read_pose=robot.read_end_pose_mm_deg,
            pose_error=robot.pose_error,
            get_motion_status=lambda: robot.get_arm_status_snapshot().motion_status,
            require_motion_arrived=require_motion_arrived,
            pose_only_fallback_enabled=(
                require_motion_arrived
                and bool(self.get_parameter("motion_arrived_pose_fallback_enabled").value)
            ),
            pose_only_fallback_pos_tolerance_mm=float(
                self.get_parameter("motion_arrived_pose_fallback_pos_tolerance_mm").value
            ),
            pose_only_fallback_rot_tolerance_deg=float(
                self.get_parameter("motion_arrived_pose_fallback_rot_tolerance_deg").value
            ),
            pose_only_fallback_hold_s=float(
                self.get_parameter("motion_arrived_pose_fallback_hold_s").value
            ),
            can_accept_pose_only=lambda: self._pose_only_completion_is_safe(
                robot.get_arm_status_snapshot()
            ),
            progress_extends_timeout=bool(
                self.get_parameter("motion_progress_extends_timeout").value
            ),
            progress_position_epsilon_mm=float(
                self.get_parameter("motion_progress_position_epsilon_mm").value
            ),
            progress_rotation_epsilon_deg=float(
                self.get_parameter("motion_progress_rotation_epsilon_deg").value
            ),
        )

    @staticmethod
    def _pose_only_completion_is_safe(snapshot) -> bool:
        """Permit strict pose fallback only for a healthy, controlled Piper."""
        return (
            int(snapshot.err_code) == 0
            and int(snapshot.teach_status_code) == 0
            and str(snapshot.arm_status).strip().upper() == "NORMAL"
            and str(snapshot.control_mode).strip().upper() == "CAN"
            and str(snapshot.motion_status).strip().upper() == "NOT_ARRIVED"
        )

    def _execute_command(self, cmd, executed_commands: list[str]) -> None:
        robot = self._robot_client()
        self._check_interrupt()

        if isinstance(cmd, MovePoseCommand):
            speed = cmd.speed_percent if cmd.speed_percent > 0 else self._default_speed()
            refresh_command = None
            if self._pose_execution_mode() == "moveit_ik":
                self._ensure_moveit_pose_mode_ready()
                moveit_executor = self._moveit_ik_executor()
                arm_joint_positions = moveit_executor.compute_ik(cmd.pose)
                moveit_executor.publish_joint_command(arm_joint_positions, speed)
                # Piper treats /joint_ctrl_single as a persistent joint target.
                # Re-publishing the same target every poll re-enters
                # MotionCtrl_2 + JointCtrl and produces visible stop/re-correct
                # motion near an offset bottle.  Publish once and only poll
                # feedback while the controller finishes the move.
                refresh_command = None
            else:
                refresh_command = lambda: robot.move_end_pose_mm_deg(
                    x_mm=cmd.pose.x_mm,
                    y_mm=cmd.pose.y_mm,
                    z_mm=cmd.pose.z_mm,
                    roll_deg=cmd.pose.roll_deg,
                    pitch_deg=cmd.pose.pitch_deg,
                    yaw_deg=cmd.pose.yaw_deg,
                    speed_percent=speed,
                )
                refresh_command()
            self._wait_pose_with_interrupt(cmd, refresh_command=refresh_command)
            executed_commands.append(cmd.name)
            return

        if isinstance(cmd, OpenGripperCommand):
            robot.open_gripper(open_mm=cmd.open_mm, effort_nm=cmd.effort_nm)
            if cmd.wait_target_mm is not None:
                ok = robot.wait_for_gripper(
                    target_mm=cmd.wait_target_mm,
                    tol_mm=cmd.wait_tol_mm,
                    timeout_s=cmd.wait_timeout_s,
                )
                if not ok:
                    raise TimeoutError(f"open gripper wait timeout: cmd={cmd.name}")
            self._lock_gripper_command(
                position_m=float(cmd.open_mm) / 1000.0,
                effort_nm=float(
                    self._default_gripper_close_effort_nm()
                    if cmd.effort_nm is None
                    else cmd.effort_nm
                ),
            )
            executed_commands.append(cmd.name)
            return

        if isinstance(cmd, CloseGripperCommand):
            robot.close_gripper(effort_nm=cmd.effort_nm)
            if cmd.wait_effort_nm is not None:
                ok = robot.wait_for_gripper_effort(
                    target_effort_nm=cmd.wait_effort_nm,
                    timeout_s=cmd.wait_timeout_s,
                )
                if not ok:
                    raise TimeoutError(f"close gripper wait timeout: cmd={cmd.name}")
            self._lock_gripper_command(
                position_m=0.0,
                effort_nm=float(
                    self._default_gripper_close_effort_nm()
                    if cmd.effort_nm is None
                    else cmd.effort_nm
                ),
            )
            executed_commands.append(cmd.name)
            return

        if isinstance(cmd, SleepCommand):
            deadline = time.monotonic() + max(0.0, cmd.duration_s)
            while time.monotonic() < deadline:
                self._check_interrupt()
                time.sleep(self._poll_interval())
            executed_commands.append(cmd.name)
            return

        raise RuntimeError(f"unsupported command instance: {type(cmd)}")

    def _set_gripper_open(self, open_mm: float | None = None) -> None:
        robot = self._robot_client()
        target_mm = float(self._default_gripper_open_mm() if open_mm is None else open_mm)
        robot.open_gripper(open_mm=target_mm, effort_nm=None)
        ok = robot.wait_for_gripper(target_mm=target_mm, tol_mm=5.0, timeout_s=4.0)
        if not ok:
            raise TimeoutError(f"open gripper wait timeout: target_mm={target_mm:.2f}")
        self._lock_gripper_command(
            position_m=target_mm / 1000.0,
            effort_nm=self._default_gripper_close_effort_nm(),
        )

    def _set_gripper_closed(self, effort_nm: float | None = None) -> None:
        robot = self._robot_client()
        target_effort = float(self._default_gripper_close_effort_nm() if effort_nm is None else effort_nm)
        robot.close_gripper(effort_nm=target_effort)
        ok = robot.wait_for_gripper_effort(target_effort_nm=target_effort, timeout_s=6.0)
        if not ok:
            raise TimeoutError(f"close gripper wait timeout: target_effort_nm={target_effort:.2f}")
        self._lock_gripper_command(position_m=0.0, effort_nm=target_effort)

    def _move_pose_sync(
        self,
        *,
        name: str,
        pose,
        speed_percent: float,
        pos_tolerance_mm: float | None = None,
        rot_tolerance_deg: float | None = None,
        timeout_s: float = 8.0,
        tighten_position_tolerance: bool = True,
    ):
        if pos_tolerance_mm is None:
            pos_tolerance_mm = self._move_pos_tolerance_mm()
        if rot_tolerance_deg is None:
            rot_tolerance_deg = self._move_rot_tolerance_deg()
        start_pose = self._read_pose_with_retry()
        effective_pos_tolerance_mm = effective_position_tolerance_mm(
            start=start_pose,
            target=pose,
            default_tolerance_mm=pos_tolerance_mm,
            enable_tightening=tighten_position_tolerance,
        )
        cmd = MovePoseCommand(
            name=name,
            pose=pose,
            speed_percent=speed_percent,
            timeout_s=timeout_s,
            pos_tolerance_mm=effective_pos_tolerance_mm,
            rot_tolerance_deg=rot_tolerance_deg,
        )
        executed_commands: list[str] = []
        self._execute_command(cmd, executed_commands)
        return self._robot_client().read_end_pose_mm_deg()

    def _move_configured_pose(
        self,
        *,
        name: str,
        pose: EndPoseMMDeg | None,
        speed_percent: float,
        timeout_s: float = 12.0,
    ) -> EndPoseMMDeg | None:
        if pose is None:
            return None
        return self._move_pose_sync(
            name=name,
            pose=pose,
            speed_percent=speed_percent,
            timeout_s=timeout_s,
            tighten_position_tolerance=False,
        )

    def _move_to_post_grasp_pose_safely(
        self,
        *,
        current_pose: EndPoseMMDeg,
        speed_percent: float,
        execution_trace: list[dict[str, object]],
        execution_strategy: str | None = None,
    ) -> tuple[EndPoseMMDeg | None, EndPoseMMDeg]:
        """Lift vertically before any post-grasp lateral return motion."""
        lift_pose = None
        safe_z_mm = max(
            float(current_pose.z_mm),
            self._post_grasp_safe_lift_z_mm(),
        )
        if safe_z_mm - float(current_pose.z_mm) > 1.0:
            lift_target = self._copy_pose_with(current_pose, z_mm=safe_z_mm)
            lift_pose = self._run_traced_action(
                execution_trace=execution_trace,
                step_name="post_grasp_vertical_safe_lift",
                command_type="move_pose",
                command_payload={
                    "pose_mm_deg": self._pose_to_dict(lift_target),
                    "speed_percent": float(speed_percent),
                    "timeout_s": float(self._plan_pose_timeout_s()),
                    "execution_strategy": execution_strategy,
                },
                action=lambda: self._move_pose_sync(
                    name="post_grasp_vertical_safe_lift",
                    pose=lift_target,
                    speed_percent=speed_percent,
                    timeout_s=self._plan_pose_timeout_s(),
                    tighten_position_tolerance=False,
                ),
            )

        post_grasp_target = self._configured_pose("post_grasp_pose")
        post_grasp_pose = self._run_traced_action(
            execution_trace=execution_trace,
            step_name="post_grasp_high_hold",
            command_type="move_pose",
            command_payload={
                "pose_mm_deg": self._pose_to_dict(post_grasp_target),
                "speed_percent": float(speed_percent),
                "timeout_s": float(self._plan_pose_timeout_s()),
                "execution_strategy": execution_strategy,
            },
            action=lambda: self._move_configured_pose(
                name="post_grasp_high_hold",
                pose=post_grasp_target,
                speed_percent=speed_percent,
                timeout_s=self._plan_pose_timeout_s(),
            ),
        )
        return lift_pose, post_grasp_pose

    @staticmethod
    def _copy_pose_with(
        pose: EndPoseMMDeg,
        *,
        x_mm: float | None = None,
        y_mm: float | None = None,
        z_mm: float | None = None,
        rpy_deg: tuple[float, float, float] | None = None,
    ) -> EndPoseMMDeg:
        roll_deg, pitch_deg, yaw_deg = (
            (pose.roll_deg, pose.pitch_deg, pose.yaw_deg)
            if rpy_deg is None
            else (float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2]))
        )
        return EndPoseMMDeg(
            x_mm=float(pose.x_mm if x_mm is None else x_mm),
            y_mm=float(pose.y_mm if y_mm is None else y_mm),
            z_mm=float(pose.z_mm if z_mm is None else z_mm),
            roll_deg=float(roll_deg),
            pitch_deg=float(pitch_deg),
            yaw_deg=float(yaw_deg),
        )

    def _interpolate_pose_segment(
        self,
        *,
        name_prefix: str,
        start: EndPoseMMDeg,
        end: EndPoseMMDeg,
        max_step_mm: float,
    ) -> list[tuple[str, EndPoseMMDeg]]:
        distance_mm = math.sqrt(
            (float(end.x_mm) - float(start.x_mm)) ** 2
            + (float(end.y_mm) - float(start.y_mm)) ** 2
            + (float(end.z_mm) - float(start.z_mm)) ** 2
        )
        steps = max(1, int(math.ceil(distance_mm / max(5.0, float(max_step_mm)))))
        out: list[tuple[str, EndPoseMMDeg]] = []
        for index in range(1, steps + 1):
            alpha = index / steps
            pose = EndPoseMMDeg(
                x_mm=float(start.x_mm) + (float(end.x_mm) - float(start.x_mm)) * alpha,
                y_mm=float(start.y_mm) + (float(end.y_mm) - float(start.y_mm)) * alpha,
                z_mm=float(start.z_mm) + (float(end.z_mm) - float(start.z_mm)) * alpha,
                roll_deg=float(end.roll_deg),
                pitch_deg=float(end.pitch_deg),
                yaw_deg=float(end.yaw_deg),
            )
            suffix = "" if steps == 1 else f"_{index:02d}"
            out.append((f"{name_prefix}{suffix}", pose))
        return out

    def _target_pose_from_plan_top_down(
        self,
        plan,
        *,
        rpy_deg: tuple[float, float, float] | None = None,
    ) -> EndPoseMMDeg:
        rpy = self._top_down_rpy_deg() if rpy_deg is None else tuple(float(value) for value in rpy_deg)
        if plan.target_contact_point_base_m is None or plan.tool_contact_offset_tool_m is None:
            raise RuntimeError(
                "safe_top_down requires target contact geometry; rebuild robot_grasp_msgs "
                "and generate a new grasp plan"
            )
        contact_point = np.asarray(plan.target_contact_point_base_m, dtype=np.float64).reshape(3)
        tool_offset = np.asarray(plan.tool_contact_offset_tool_m, dtype=np.float64).reshape(3)
        final_rotation = rotation_matrix_from_rpy_deg(*rpy)
        target_translation = contact_point - final_rotation @ tool_offset
        target_z_mm = float(target_translation[2]) * 1000.0
        min_target_z_mm = self._top_down_min_target_z_mm()
        if target_z_mm < min_target_z_mm:
            raise RuntimeError(
                f"recomputed safe_top_down link6 z={target_z_mm:.1f}mm below "
                f"top_down_min_target_z_mm={min_target_z_mm:.1f}mm"
            )
        return EndPoseMMDeg(
            x_mm=float(target_translation[0]) * 1000.0,
            y_mm=float(target_translation[1]) * 1000.0,
            z_mm=float(target_translation[2]) * 1000.0,
            roll_deg=rpy[0],
            pitch_deg=rpy[1],
            yaw_deg=rpy[2],
        )

    def _build_safe_top_down_waypoints(
        self,
        *,
        plan,
        current_pose: EndPoseMMDeg,
        rpy_deg: tuple[float, float, float] | None = None,
    ) -> list[tuple[str, EndPoseMMDeg]]:
        rpy = self._top_down_rpy_deg() if rpy_deg is None else tuple(float(value) for value in rpy_deg)
        target = self._target_pose_from_plan_top_down(plan, rpy_deg=rpy)
        # Use the lowest transit plane that still clears the target and the
        # configured global safety floor.  Keeping an unusually high camera
        # observation Z here can make the subsequent lateral pose unreachable
        # (notably after mirroring the work area to the arm's right side).
        # The first vertical segment therefore lifts *or lowers* in place to a
        # geometrically safe transit height before changing wrist orientation
        # or moving laterally.
        safe_z_mm = max(
            float(target.z_mm) + self._top_down_approach_height_mm(),
            self._top_down_min_safe_z_mm(),
        )
        configured_lift_z_mm = float(target.z_mm) + self._top_down_lift_height_mm()
        lift_z_mm = (
            max(safe_z_mm, configured_lift_z_mm)
            if self._top_down_lift_to_safe_z()
            else configured_lift_z_mm
        )
        current_safe_current_rpy = self._copy_pose_with(current_pose, z_mm=safe_z_mm)
        start_safe = self._copy_pose_with(current_pose, z_mm=safe_z_mm, rpy_deg=rpy)
        above_target = self._copy_pose_with(target, z_mm=safe_z_mm, rpy_deg=rpy)
        lifted_target = self._copy_pose_with(target, z_mm=lift_z_mm, rpy_deg=rpy)

        waypoints: list[tuple[str, EndPoseMMDeg]] = []
        waypoints.extend(
            self._interpolate_pose_segment(
                name_prefix="topdown_lift_clear",
                start=current_pose,
                end=current_safe_current_rpy,
                max_step_mm=self._top_down_vertical_step_mm(),
            )
        )
        waypoints.extend(
            self._interpolate_pose_segment(
                name_prefix="topdown_set_wrist",
                start=current_safe_current_rpy,
                end=start_safe,
                max_step_mm=self._top_down_lateral_step_mm(),
            )
        )
        waypoints.extend(
            self._interpolate_pose_segment(
                name_prefix="topdown_lateral",
                start=start_safe,
                end=above_target,
                max_step_mm=self._top_down_lateral_step_mm(),
            )
        )
        waypoints.extend(
            self._interpolate_pose_segment(
                name_prefix="topdown_descend",
                start=above_target,
                end=target,
                max_step_mm=self._top_down_vertical_step_mm(),
            )
        )
        waypoints.extend(
            self._interpolate_pose_segment(
                name_prefix="topdown_lift_object",
                start=target,
                end=lifted_target,
                max_step_mm=self._top_down_vertical_step_mm(),
            )
        )
        return waypoints

    def _evaluate_safe_top_down_variants(
        self,
        *,
        plan,
        current_pose: EndPoseMMDeg,
        compute_ik,
    ) -> tuple[
        tuple[float, float, float] | None,
        list[tuple[str, EndPoseMMDeg]] | None,
        list[dict[str, object]],
    ]:
        attempts: list[dict[str, object]] = []
        for variant_index, rpy in enumerate(self._safe_cartesian_rpy_variants(plan)):
            waypoint_results: list[dict[str, object]] = []
            try:
                waypoints = self._build_safe_top_down_waypoints(
                    plan=plan,
                    current_pose=current_pose,
                    rpy_deg=rpy,
                )
            except Exception as exc:
                waypoint_results.append(
                    {
                        "stage": "topdown_geometry",
                        "status": "failed",
                        "ik_error_type": type(exc).__name__,
                        "ik_error_message": str(exc),
                    }
                )
                attempts.append(
                    {
                        "variant_index": variant_index,
                        "rpy_deg": list(rpy),
                        "status": "rejected",
                        "waypoint_results": waypoint_results,
                    }
                )
                continue

            for stage, pose in waypoints:
                try:
                    compute_ik(pose)
                    waypoint_results.append({"stage": stage, "status": "ok"})
                except Exception as exc:
                    waypoint_results.append(
                        {
                            "stage": stage,
                            "status": "failed",
                            "ik_error_type": type(exc).__name__,
                            "ik_error_message": str(exc),
                        }
                    )
                    break
            failed = next((item for item in waypoint_results if item.get("status") == "failed"), None)
            attempts.append(
                {
                    "variant_index": variant_index,
                    "rpy_deg": list(rpy),
                    "status": "accepted" if failed is None else "rejected",
                    "waypoint_results": waypoint_results,
                }
            )
            if failed is None:
                return rpy, waypoints, attempts
        return None, None, attempts

    def _validate_grasp_plan_kinematics(self, plan) -> dict[str, object]:
        if self._pose_execution_mode() != "moveit_ik":
            return {
                "robot_validation_result": "accepted",
                "robot_validation_stage": None,
                "ik_error_type": None,
                "ik_error_message": None,
                "validated_waypoints": [],
                "waypoint_results": [],
            }
        self._ensure_moveit_pose_mode_ready()
        moveit_executor = self._moveit_ik_executor()
        if self._uses_safe_cartesian_strategy():
            strategy = self._execution_strategy()
            current_pose = self._read_pose_with_retry()
            selected_rpy, _waypoints, attempts = self._evaluate_safe_top_down_variants(
                plan=plan,
                current_pose=current_pose,
                compute_ik=moveit_executor.compute_ik,
            )
            selected_attempt = next((item for item in attempts if item.get("status") == "accepted"), None)
            report_attempt = selected_attempt or (attempts[0] if attempts else {"waypoint_results": []})
            waypoint_results = list(report_attempt.get("waypoint_results") or [])
            failed = next((item for item in waypoint_results if item.get("status") == "failed"), None)
            return {
                "robot_validation_result": "accepted" if selected_rpy is not None else "rejected_by_robot_validation",
                "robot_validation_stage": None if selected_rpy is not None else (failed or {}).get("stage"),
                "ik_error_type": None if selected_rpy is not None else (failed or {}).get("ik_error_type"),
                "ik_error_message": None if selected_rpy is not None else (failed or {}).get("ik_error_message"),
                "validated_waypoints": [str(item["stage"]) for item in waypoint_results if item.get("status") == "ok"],
                "waypoint_results": waypoint_results,
                "execution_strategy": strategy,
                "top_down_rpy_deg": None if selected_rpy is None else list(selected_rpy),
                "top_down_variant_attempts": attempts,
            }
        waypoint_results = validate_grasp_plan_waypoints_detailed(
            plan,
            include_pregrasp=self._enable_pregrasp(),
            compute_ik=moveit_executor.compute_ik,
        )
        failed = next((item for item in waypoint_results if item.get("status") == "failed"), None)
        return {
            "robot_validation_result": "accepted" if failed is None else "rejected_by_robot_validation",
            "robot_validation_stage": None if failed is None else failed.get("stage"),
            "ik_error_type": None if failed is None else failed.get("ik_error_type"),
            "ik_error_message": None if failed is None else failed.get("ik_error_message"),
            "validated_waypoints": [str(item["stage"]) for item in waypoint_results if item.get("status") == "ok"],
            "waypoint_results": waypoint_results,
        }

    def _publish_service_result(self, payload: dict[str, object]) -> None:
        self._result_pub.publish(String(data=json_dumps(payload)))

    def _capture_execution_feedback(self) -> dict[str, object]:
        feedback: dict[str, object] = {
            "feedback_pose_mm_deg": None,
            "feedback_gripper": None,
            "feedback_arm_status": None,
        }
        feedback_errors: list[str] = []
        robot = self._robot_client()

        try:
            feedback["feedback_pose_mm_deg"] = self._pose_to_dict(self._read_pose_with_retry())
        except Exception as exc:
            feedback_errors.append(f"pose: {exc}")

        try:
            feedback["feedback_gripper"] = self._gripper_to_dict(robot.get_gripper_status())
        except Exception as exc:
            feedback_errors.append(f"gripper: {exc}")

        try:
            feedback["feedback_arm_status"] = robot.format_arm_status()
        except Exception as exc:
            feedback_errors.append(f"arm_status: {exc}")

        if feedback_errors:
            feedback["feedback_errors"] = feedback_errors
        return feedback

    def _run_traced_action(
        self,
        *,
        execution_trace: list[dict[str, object]],
        step_name: str,
        command_type: str,
        command_payload: dict[str, object],
        action,
    ):
        commanded_at_unix_s = time.time()
        success = False
        error = None
        result = None
        try:
            result = action()
            success = True
            return result
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            step_payload = {
                "step_name": step_name,
                "command_type": command_type,
                "commanded_at_unix_s": commanded_at_unix_s,
                "feedback_at_unix_s": time.time(),
                "success": success,
                "command": command_payload,
                "error": error,
            }
            step_payload.update(self._capture_execution_feedback())
            execution_trace.append(step_payload)

    def _execute_safe_top_down_grasp_plan(
        self,
        *,
        request,
        plan,
        run_id: str,
    ) -> dict[str, object]:
        strategy = self._execution_strategy()
        speed_percent = self._top_down_speed_percent()
        execution_trace: list[dict[str, object]] = []
        pregrasp_actual = None
        target_actual = None
        retreat_pose = None
        post_grasp_safe_lift_pose = None
        post_grasp_pose = None
        handoff_pose = None
        home_pose = None

        current_pose = self._read_pose_with_retry()
        selected_rpy = self._safe_cartesian_rpy_variants(plan)[0]
        if self._pose_execution_mode() == "moveit_ik":
            self._ensure_moveit_pose_mode_ready()
            selected_rpy, top_down_waypoints, variant_attempts = self._evaluate_safe_top_down_variants(
                plan=plan,
                current_pose=current_pose,
                compute_ik=self._moveit_ik_executor().compute_ik,
            )
            if selected_rpy is None or top_down_waypoints is None:
                raise RuntimeError("no IK-reachable safe_top_down RPY variant")
        else:
            top_down_waypoints = self._build_safe_top_down_waypoints(
                plan=plan,
                current_pose=current_pose,
                rpy_deg=selected_rpy,
            )
            variant_attempts = []

        self._run_traced_action(
            execution_trace=execution_trace,
            step_name="open_gripper",
            command_type="open_gripper",
            command_payload={
                "open_mm": float(self._default_gripper_open_mm()),
                "effort_nm": None,
            },
            action=lambda: self._set_gripper_open(),
        )

        descend_steps = 0
        descend_step_total = sum(
            1 for name, _pose in top_down_waypoints if name.startswith("topdown_descend")
        )
        lift_steps = 0
        for step_name, waypoint in top_down_waypoints:
            command_speed_percent = speed_percent
            is_descend = step_name.startswith("topdown_descend")
            if (
                strategy == "safe_top_down"
                and is_descend
                and descend_steps + 1 == descend_step_total
            ):
                command_speed_percent = min(
                    speed_percent,
                    self._safe_top_down_final_speed_percent(),
                )
            actual = self._run_traced_action(
                execution_trace=execution_trace,
                step_name=step_name,
                command_type="move_pose",
                command_payload={
                    "pose_mm_deg": self._pose_to_dict(waypoint),
                    "speed_percent": float(command_speed_percent),
                    "timeout_s": float(self._plan_pose_timeout_s()),
                    "execution_strategy": strategy,
                },
                action=lambda pose=waypoint, name=step_name, move_speed=command_speed_percent: self._move_pose_sync(
                    name=name,
                    pose=pose,
                    speed_percent=move_speed,
                    timeout_s=self._plan_pose_timeout_s(),
                    tighten_position_tolerance=True,
                ),
            )
            if step_name.startswith("topdown_lift_clear") or step_name.startswith("topdown_lateral"):
                pregrasp_actual = actual
            elif is_descend:
                target_actual = actual
                descend_steps += 1
            elif step_name.startswith("topdown_lift_object"):
                retreat_pose = actual
                lift_steps += 1

            if is_descend and descend_steps == descend_step_total:
                self._run_traced_action(
                    execution_trace=execution_trace,
                    step_name="close_gripper",
                    command_type="close_gripper",
                    command_payload={
                        "effort_nm": float(self._default_gripper_close_effort_nm()),
                    },
                    action=lambda: self._set_gripper_closed(),
                )

        if lift_steps <= 0:
            raise RuntimeError(f"{strategy} produced no lift-object waypoint")

        if self._move_to_post_grasp_pose_enabled() and not bool(request.move_home_after):
            post_grasp_safe_lift_pose, post_grasp_pose = (
                self._move_to_post_grasp_pose_safely(
                    current_pose=retreat_pose,
                    speed_percent=speed_percent,
                    execution_trace=execution_trace,
                    execution_strategy=strategy,
                )
            )

        if bool(request.move_home_after):
            handoff_target = self._configured_pose("handoff_pose") if self._use_handoff_pose() else None
            handoff_pose = self._run_traced_action(
                execution_trace=execution_trace,
                step_name="handoff",
                command_type="move_pose",
                command_payload={
                    "pose_mm_deg": self._pose_to_dict(handoff_target),
                    "speed_percent": float(speed_percent),
                    "timeout_s": float(self._plan_pose_timeout_s()),
                    "execution_strategy": strategy,
                },
                action=lambda: self._move_configured_pose(
                    name="handoff",
                    pose=handoff_target,
                    speed_percent=speed_percent,
                    timeout_s=self._plan_pose_timeout_s(),
                ),
            )
            self._run_traced_action(
                execution_trace=execution_trace,
                step_name="release_gripper",
                command_type="open_gripper",
                command_payload={
                    "open_mm": float(self._default_gripper_open_mm()),
                    "effort_nm": None,
                },
                action=lambda: self._set_gripper_open(),
            )
            home_target = self._configured_pose("home_pose")
            home_pose = self._run_traced_action(
                execution_trace=execution_trace,
                step_name="home",
                command_type="move_pose",
                command_payload={
                    "pose_mm_deg": self._pose_to_dict(home_target),
                    "speed_percent": float(speed_percent),
                    "timeout_s": float(self._plan_pose_timeout_s()),
                    "execution_strategy": strategy,
                },
                action=lambda: self._move_configured_pose(
                    name="home",
                    pose=home_target,
                    speed_percent=speed_percent,
                    timeout_s=self._plan_pose_timeout_s(),
                ),
            )

        return {
            "status": "ok",
            "run_id": run_id,
            "candidate_score": plan.candidate.score,
            "execution_strategy": strategy,
            "top_down_rpy_deg": list(selected_rpy),
            "top_down_variant_attempts": variant_attempts,
            "top_down_speed_percent": float(speed_percent),
            "top_down_final_descent_speed_percent": float(
                min(speed_percent, self._safe_top_down_final_speed_percent())
                if strategy == "safe_top_down"
                else speed_percent
            ),
            "pregrasp_executed": True,
            "move_home_after": bool(request.move_home_after),
            "release_performed": bool(request.move_home_after),
            "pregrasp_pose_mm_deg": self._pose_to_dict(pregrasp_actual),
            "grasp_pose_mm_deg": self._pose_to_dict(target_actual),
            "target_pose_mm_deg": self._pose_to_dict(target_actual),
            "retreat_pose_mm_deg": self._pose_to_dict(retreat_pose),
            "post_grasp_safe_lift_pose_mm_deg": self._pose_to_dict(
                post_grasp_safe_lift_pose
            ),
            "post_grasp_pose_mm_deg": self._pose_to_dict(post_grasp_pose),
            "handoff_pose_mm_deg": self._pose_to_dict(handoff_pose),
            "home_pose_mm_deg": self._pose_to_dict(home_pose),
            "final_pose_mm_deg": self._pose_to_dict(
                home_pose or handoff_pose or post_grasp_pose or retreat_pose
            ),
            "execution_trace": execution_trace,
        }

    def _handle_get_state(self, _request, response):
        try:
            self._ensure_robot_ready()
            pose = self._read_pose_with_retry()
            response.success = True
            response.message = "state ok"
            response.current_pose = pose6d_from_end_pose(pose)
            response.arm_status = self._robot_client().format_arm_status()
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            response.arm_status = "unavailable"
        return response

    def _handle_open_gripper(self, _request, response):
        """Open the gripper without commanding any arm motion."""
        try:
            self._reset_interrupt_flags()
            self._ensure_robot_ready()
            self._set_gripper_open()
            response.success = True
            response.message = (
                f"gripper opened to {self._default_gripper_open_mm():.1f} mm; "
                "arm pose unchanged"
            )
            self._publish_service_result(
                {
                    "status": "ok",
                    "kind": "open_gripper",
                    "open_mm": float(self._default_gripper_open_mm()),
                }
            )
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            self._publish_service_result(
                {
                    "status": "failed",
                    "kind": "open_gripper",
                    "message": str(exc),
                }
            )
        return response

    def _handle_execute_named_pose(self, request, response):
        try:
            self._reset_interrupt_flags()
            self._ensure_robot_ready()
            if bool(request.open_gripper_first):
                self._set_gripper_open()
            actual_pose = self._move_pose_sync(
                name=str(request.name or "named_pose"),
                pose=pose6d_to_end_pose(request.pose),
                speed_percent=float(request.speed_percent or self._default_speed()),
                timeout_s=self._named_pose_timeout_s(),
                # Named poses are coarse transit/observation poses.  When the
                # arm is already very close to one, deriving the tolerance from
                # the tiny remaining translation can shrink it to 0.5 mm and
                # turn normal Piper endpoint jitter into a false timeout.
                # Precision grasp/place waypoints retain their explicit
                # tightening in the execution paths below.
                tighten_position_tolerance=False,
            )
            response.success = True
            response.message = f"move completed: {request.name}"
            response.actual_pose = pose6d_from_end_pose(actual_pose)
            self._publish_service_result(
                {
                    "status": "ok",
                    "kind": "execute_named_pose",
                    "name": request.name,
                    "actual_pose_mm_deg": {
                        "x_mm": actual_pose.x_mm,
                        "y_mm": actual_pose.y_mm,
                        "z_mm": actual_pose.z_mm,
                        "roll_deg": actual_pose.roll_deg,
                        "pitch_deg": actual_pose.pitch_deg,
                        "yaw_deg": actual_pose.yaw_deg,
                    },
                }
            )
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            self._publish_service_result(
                {"status": "failed", "kind": "execute_named_pose", "name": request.name, "message": str(exc)}
            )
        return response

    def _handle_execute_grasp_plan(self, request, response):
        run_id = str(request.run_id or "")
        try:
            self._reset_interrupt_flags()
            self._ensure_robot_ready()
            plan = grasp_plan_from_msg(request.plan)
            if not bool(request.execute):
                validation_payload = self._validate_grasp_plan_kinematics(plan)
                accepted = validation_payload["robot_validation_result"] == "accepted"
                response.success = accepted
                response.message = (
                    "dry-run only; plan IK validated"
                    if accepted
                    else str(validation_payload.get("ik_error_message") or "dry-run only; plan rejected")
                )
                payload = {
                    "status": "dry_run" if accepted else "rejected",
                    "run_id": run_id,
                    "candidate_score": plan.candidate.score,
                    "within_workspace": plan.within_workspace,
                    "workspace_violations": list(plan.workspace_violations),
                }
                payload.update(validation_payload)
                response.execution_json = json_dumps(payload)
                self._publish_service_result(json.loads(response.execution_json))
                return response

            if not plan.within_workspace:
                raise RuntimeError("plan is outside workspace: " + " | ".join(plan.workspace_violations))
            if not self._uses_safe_cartesian_strategy():
                self._assert_execution_waypoints_safe(plan)

            if self._uses_safe_cartesian_strategy():
                strategy = self._execution_strategy()
                execution_payload = self._execute_safe_top_down_grasp_plan(
                    request=request,
                    plan=plan,
                    run_id=run_id,
                )
                response.success = True
                response.message = f"grasp plan executed with {strategy}"
                response.execution_json = json_dumps(execution_payload)
                self._publish_service_result(execution_payload)
                return response

            pregrasp_actual = None
            grasp_actual = None
            target_actual = None
            retreat_pose = None
            post_grasp_safe_lift_pose = None
            post_grasp_pose = None
            handoff_pose = None
            home_pose = None
            speed_percent = self._default_speed()
            execution_trace: list[dict[str, object]] = []

            self._run_traced_action(
                execution_trace=execution_trace,
                step_name="open_gripper",
                command_type="open_gripper",
                command_payload={
                    "open_mm": float(self._default_gripper_open_mm()),
                    "effort_nm": None,
                },
                action=lambda: self._set_gripper_open(),
            )
            if self._enable_pregrasp():
                pregrasp_target = pose6d_to_end_pose(request.plan.pregrasp_pose)
                pregrasp_actual = self._run_traced_action(
                    execution_trace=execution_trace,
                    step_name="pregrasp",
                    command_type="move_pose",
                    command_payload={
                        "pose_mm_deg": self._pose_to_dict(pregrasp_target),
                        "speed_percent": float(speed_percent),
                        "timeout_s": float(self._plan_pose_timeout_s()),
                    },
                    action=lambda: self._move_pose_sync(
                        name="pregrasp",
                        pose=pregrasp_target,
                        speed_percent=speed_percent,
                        timeout_s=self._plan_pose_timeout_s(),
                    ),
                )
            grasp_target = pose6d_to_end_pose(request.plan.grasp_pose)
            grasp_actual = self._run_traced_action(
                execution_trace=execution_trace,
                step_name="grasp",
                command_type="move_pose",
                command_payload={
                    "pose_mm_deg": self._pose_to_dict(grasp_target),
                    "speed_percent": float(speed_percent),
                    "timeout_s": float(self._plan_pose_timeout_s()),
                },
                action=lambda: self._move_pose_sync(
                    name="grasp",
                    pose=grasp_target,
                    speed_percent=speed_percent,
                    timeout_s=self._plan_pose_timeout_s(),
                ),
            )
            target_target = pose6d_to_end_pose(request.plan.target_pose)
            target_actual = self._run_traced_action(
                execution_trace=execution_trace,
                step_name="target",
                command_type="move_pose",
                command_payload={
                    "pose_mm_deg": self._pose_to_dict(target_target),
                    "speed_percent": float(speed_percent),
                    "timeout_s": float(self._plan_pose_timeout_s()),
                },
                action=lambda: self._move_pose_sync(
                    name="target",
                    pose=target_target,
                    speed_percent=speed_percent,
                    timeout_s=self._plan_pose_timeout_s(),
                ),
            )
            self._run_traced_action(
                execution_trace=execution_trace,
                step_name="close_gripper",
                command_type="close_gripper",
                command_payload={
                    "effort_nm": float(self._default_gripper_close_effort_nm()),
                },
                action=lambda: self._set_gripper_closed(),
            )
            retreat_target = pose6d_to_end_pose(request.plan.retreat_pose)
            retreat_pose = self._run_traced_action(
                execution_trace=execution_trace,
                step_name="retreat",
                command_type="move_pose",
                command_payload={
                    "pose_mm_deg": self._pose_to_dict(retreat_target),
                    "speed_percent": float(speed_percent),
                    "timeout_s": float(self._plan_pose_timeout_s()),
                },
                action=lambda: self._move_pose_sync(
                    name="retreat",
                    pose=retreat_target,
                    speed_percent=speed_percent,
                    timeout_s=self._plan_pose_timeout_s(),
                ),
            )
            if self._move_to_post_grasp_pose_enabled() and not bool(request.move_home_after):
                post_grasp_safe_lift_pose, post_grasp_pose = (
                    self._move_to_post_grasp_pose_safely(
                        current_pose=retreat_pose,
                        speed_percent=speed_percent,
                        execution_trace=execution_trace,
                    )
                )
            if bool(request.move_home_after):
                handoff_target = self._configured_pose("handoff_pose") if self._use_handoff_pose() else None
                handoff_pose = self._run_traced_action(
                    execution_trace=execution_trace,
                    step_name="handoff",
                    command_type="move_pose",
                    command_payload={
                        "pose_mm_deg": self._pose_to_dict(handoff_target),
                        "speed_percent": float(speed_percent),
                        "timeout_s": float(self._plan_pose_timeout_s()),
                    },
                    action=lambda: self._move_configured_pose(
                        name="handoff",
                        pose=handoff_target,
                        speed_percent=speed_percent,
                        timeout_s=self._plan_pose_timeout_s(),
                    ),
                )
                self._run_traced_action(
                    execution_trace=execution_trace,
                    step_name="release_gripper",
                    command_type="open_gripper",
                    command_payload={
                        "open_mm": float(self._default_gripper_open_mm()),
                        "effort_nm": None,
                    },
                    action=lambda: self._set_gripper_open(),
                )
                home_target = self._configured_pose("home_pose")
                home_pose = self._run_traced_action(
                    execution_trace=execution_trace,
                    step_name="home",
                    command_type="move_pose",
                    command_payload={
                        "pose_mm_deg": self._pose_to_dict(home_target),
                        "speed_percent": float(speed_percent),
                        "timeout_s": float(self._plan_pose_timeout_s()),
                    },
                    action=lambda: self._move_configured_pose(
                        name="home",
                        pose=home_target,
                        speed_percent=speed_percent,
                        timeout_s=self._plan_pose_timeout_s(),
                    ),
                )

            execution_payload = {
                "status": "ok",
                "run_id": run_id,
                "candidate_score": plan.candidate.score,
                "pregrasp_executed": self._enable_pregrasp(),
                "move_home_after": bool(request.move_home_after),
                "release_performed": bool(request.move_home_after),
                "pregrasp_pose_mm_deg": self._pose_to_dict(pregrasp_actual),
                "grasp_pose_mm_deg": self._pose_to_dict(grasp_actual),
                "target_pose_mm_deg": self._pose_to_dict(target_actual),
                "retreat_pose_mm_deg": self._pose_to_dict(retreat_pose),
                "post_grasp_safe_lift_pose_mm_deg": self._pose_to_dict(
                    post_grasp_safe_lift_pose
                ),
                "post_grasp_pose_mm_deg": self._pose_to_dict(post_grasp_pose),
                "handoff_pose_mm_deg": self._pose_to_dict(handoff_pose),
                "home_pose_mm_deg": self._pose_to_dict(home_pose),
                "final_pose_mm_deg": self._pose_to_dict(
                    home_pose or handoff_pose or post_grasp_pose or retreat_pose
                ),
                "execution_trace": execution_trace,
            }
            response.success = True
            response.message = "grasp plan executed"
            response.execution_json = json_dumps(execution_payload)
            self._publish_service_result(execution_payload)
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            response.execution_json = json_dumps({"status": "failed", "run_id": run_id, "message": str(exc)})
            self._publish_service_result(json.loads(response.execution_json))
        return response

    def _validate_place_request(self, request) -> tuple[EndPoseMMDeg, EndPoseMMDeg, EndPoseMMDeg]:
        if not str(request.plan.item_id or "").strip():
            raise RuntimeError("place plan item_id is empty")
        slot_index = int(request.plan.slot_index)
        slot_count = max(1, int(self.get_parameter("placement_slot_count").value or 6))
        if not 0 <= slot_index < slot_count:
            raise RuntimeError(
                f"place plan slot_index {slot_index} is outside 0..{slot_count - 1}"
            )
        if not bool(request.plan.label_verified):
            raise RuntimeError("place plan rejected: target box label was not verified")
        minimum_label_confidence = max(
            0.0,
            float(self.get_parameter("placement_min_label_confidence").value or 0.42),
        )
        if float(request.plan.label_confidence) < minimum_label_confidence:
            raise RuntimeError(
                "place plan rejected: box label confidence "
                f"{float(request.plan.label_confidence):.3f} is below "
                f"{minimum_label_confidence:.3f}"
            )
        actual_size = [float(value) for value in list(request.plan.box_outer_size_m)]
        expected_size = [
            float(value)
            for value in list(self.get_parameter("placement_box_outer_size_m").value or [])
        ]
        if len(actual_size) != 3 or len(expected_size) != 3:
            raise RuntimeError("place plan box size must contain 3 values")
        tolerance = max(
            0.0,
            float(self.get_parameter("placement_box_size_tolerance_m").value or 0.005),
        )
        if any(abs(actual - expected) > tolerance for actual, expected in zip(actual_size, expected_size)):
            raise RuntimeError(
                f"place plan box size {actual_size} does not match configured size {expected_size}"
            )

        approach = pose6d_to_end_pose(request.plan.approach_pose)
        release = pose6d_to_end_pose(request.plan.release_pose)
        retreat = pose6d_to_end_pose(request.plan.retreat_pose)
        minimum_clearance = max(
            1.0,
            float(self.get_parameter("placement_min_vertical_clearance_mm").value or 50.0),
        )
        if float(approach.z_mm) - float(release.z_mm) < minimum_clearance:
            raise RuntimeError("place approach does not provide enough vertical clearance")
        if float(retreat.z_mm) - float(release.z_mm) < minimum_clearance:
            raise RuntimeError("place retreat does not provide enough vertical clearance")
        for name, pose in (("approach", approach), ("retreat", retreat)):
            lateral_delta = math.hypot(
                float(pose.x_mm) - float(release.x_mm),
                float(pose.y_mm) - float(release.y_mm),
            )
            if lateral_delta > 15.0:
                raise RuntimeError(
                    f"place {name} must remain vertical over the release point; "
                    f"lateral delta={lateral_delta:.1f}mm"
                )
        return approach, release, retreat

    def _validate_place_kinematics(
        self,
        approach: EndPoseMMDeg,
        release: EndPoseMMDeg,
        retreat: EndPoseMMDeg,
    ) -> list[dict[str, object]]:
        results: list[dict[str, object]] = []
        if self._pose_execution_mode() != "moveit_ik":
            return [{"stage": name, "status": "ok"} for name in ("place_approach", "place_release", "place_retreat")]
        self._ensure_moveit_pose_mode_ready()
        compute_ik = self._moveit_ik_executor().compute_ik
        for stage, pose in (
            ("place_approach", approach),
            ("place_release", release),
            ("place_retreat", retreat),
        ):
            try:
                compute_ik(pose)
                results.append({"stage": stage, "status": "ok"})
            except Exception as exc:
                results.append(
                    {
                        "stage": stage,
                        "status": "failed",
                        "ik_error_type": type(exc).__name__,
                        "ik_error_message": str(exc),
                    }
                )
                break
        return results

    def _handle_execute_place_plan(self, request, response):
        run_id = str(request.run_id or "")
        try:
            self._reset_interrupt_flags()
            self._ensure_robot_ready()
            approach, release, retreat = self._validate_place_request(request)
            waypoint_results = self._validate_place_kinematics(approach, release, retreat)
            failed = next((item for item in waypoint_results if item.get("status") == "failed"), None)
            if failed is not None:
                response.success = False
                response.message = str(failed.get("ik_error_message") or "place plan rejected")
                response.execution_json = json_dumps(
                    {
                        "status": "rejected",
                        "run_id": run_id,
                        "item_id": request.plan.item_id,
                        "slot_index": int(request.plan.slot_index),
                        "label_confidence": float(request.plan.label_confidence),
                        "waypoint_results": waypoint_results,
                    }
                )
                self._publish_service_result(json.loads(response.execution_json))
                return response

            if not bool(request.execute):
                response.success = True
                response.message = "dry-run only; place plan IK validated"
                response.execution_json = json_dumps(
                    {
                        "status": "dry_run",
                        "run_id": run_id,
                        "item_id": request.plan.item_id,
                        "slot_index": int(request.plan.slot_index),
                        "label_confidence": float(request.plan.label_confidence),
                        "waypoint_results": waypoint_results,
                    }
                )
                self._publish_service_result(json.loads(response.execution_json))
                return response

            trace: list[dict[str, object]] = []
            speed = self._default_speed()
            final_speed = min(
                speed,
                max(1.0, float(self.get_parameter("placement_final_speed_percent").value or 2.0)),
            )
            for stage, pose, move_speed in (
                ("place_approach", approach, speed),
                ("place_release", release, final_speed),
            ):
                self._run_traced_action(
                    execution_trace=trace,
                    step_name=stage,
                    command_type="move_pose",
                    command_payload={
                        "pose_mm_deg": self._pose_to_dict(pose),
                        "speed_percent": float(move_speed),
                        "timeout_s": float(self._plan_pose_timeout_s()),
                    },
                    action=lambda target=pose, name=stage, target_speed=move_speed: self._move_pose_sync(
                        name=name,
                        pose=target,
                        speed_percent=target_speed,
                        timeout_s=self._plan_pose_timeout_s(),
                        tighten_position_tolerance=True,
                    ),
                )
            self._run_traced_action(
                execution_trace=trace,
                step_name="place_release_gripper",
                command_type="open_gripper",
                command_payload={"open_mm": float(self._default_gripper_open_mm()), "effort_nm": None},
                action=lambda: self._set_gripper_open(),
            )
            self._run_traced_action(
                execution_trace=trace,
                step_name="place_retreat",
                command_type="move_pose",
                command_payload={
                    "pose_mm_deg": self._pose_to_dict(retreat),
                    "speed_percent": float(speed),
                    "timeout_s": float(self._plan_pose_timeout_s()),
                },
                action=lambda: self._move_pose_sync(
                    name="place_retreat",
                    pose=retreat,
                    speed_percent=speed,
                    timeout_s=self._plan_pose_timeout_s(),
                    tighten_position_tolerance=True,
                ),
            )
            home_pose = None
            if bool(request.move_home_after):
                home_target = self._configured_pose("home_pose")
                home_pose = self._run_traced_action(
                    execution_trace=trace,
                    step_name="home",
                    command_type="move_pose",
                    command_payload={
                        "pose_mm_deg": self._pose_to_dict(home_target),
                        "speed_percent": float(speed),
                        "timeout_s": float(self._plan_pose_timeout_s()),
                    },
                    action=lambda: self._move_configured_pose(
                        name="home",
                        pose=home_target,
                        speed_percent=speed,
                        timeout_s=self._plan_pose_timeout_s(),
                    ),
                )
            payload = {
                "status": "ok",
                "run_id": run_id,
                "item_id": request.plan.item_id,
                "slot_index": int(request.plan.slot_index),
                "label_confidence": float(request.plan.label_confidence),
                "box_outer_size_m": [float(value) for value in list(request.plan.box_outer_size_m)],
                "waypoint_results": waypoint_results,
                "release_performed": True,
                "move_home_after": bool(request.move_home_after),
                "final_pose_mm_deg": self._pose_to_dict(home_pose or retreat),
                "execution_trace": trace,
            }
            response.success = True
            response.message = "place plan executed"
            response.execution_json = json_dumps(payload)
            self._publish_service_result(payload)
        except Exception as exc:
            response.success = False
            response.message = str(exc)
            response.execution_json = json_dumps(
                {"status": "failed", "run_id": run_id, "message": str(exc)}
            )
            self._publish_service_result(json.loads(response.execution_json))
        return response

    def _handle_stop_robot(self, _request, response):
        try:
            with self._lock:
                self._stop_requested = True
                self._cancel_requested = True
            if self._robot is not None:
                self._robot.emergency_stop()
                self._robot_enabled = False
            response.success = True
            response.message = "stop requested"
            self._publish_service_result({"status": "stopped", "kind": "stop_robot"})
        except Exception as exc:
            response.success = False
            response.message = str(exc)
        return response

    def _run_thread(self, plan: RobotExecutionPlan, plan_json: str) -> None:
        backend = self._backend()
        started = time.time()
        executed_commands: list[str] = []
        self._publish_status(f"running: plan_id={plan.plan_id} backend={backend}")

        status = "succeeded"
        message = f"plan completed: {plan.plan_id}"
        failure_command: str | None = None
        error: str | None = None
        current_command: str | None = None

        try:
            self._ensure_robot_ready()
            # Preserve the submitted plan payload for replay/debug consistency.
            with self._lock:
                self._active_plan_json = plan_json
            for cmd in plan.commands:
                current_command = getattr(cmd, "name", type(cmd).__name__)
                self._execute_command(cmd, executed_commands)
                current_command = None
        except Exception as exc:
            with self._lock:
                stop_requested = self._stop_requested
                cancel_requested = self._cancel_requested
            failure_command = current_command
            error = str(exc)
            if stop_requested:
                status = "stopped"
                message = f"stop requested: {exc}"
            elif cancel_requested:
                status = "cancelled"
                message = f"cancel requested: {exc}"
            else:
                status = "failed"
                message = str(exc)
            self.get_logger().error("executor failed: %s\n%s", exc, traceback.format_exc())
        finally:
            if self._disconnect_after_run():
                try:
                    self._release_robot()
                except Exception as exc:
                    self.get_logger().warning("disconnect failed: %s", exc)

            with self._lock:
                stop_requested = self._stop_requested
                cancel_requested = self._cancel_requested
                self._runner = None

            finished = time.time()
            arm_status = None
            final_pose = None
            try:
                if self._robot is not None:
                    arm_status = self._robot.format_arm_status()
                    final_pose = self._robot.read_end_pose_mm_deg()
            except Exception:
                arm_status = "unavailable"

            result = RobotExecutionResult(
                status=status,
                plan_id=plan.plan_id,
                message=message,
                backend=backend,
                started_at_s=started,
                finished_at_s=finished,
                executed_commands=executed_commands,
                stop_requested=stop_requested,
                cancel_requested=cancel_requested,
                failure_command=failure_command,
                error=error,
                arm_status=arm_status,
                final_pose=final_pose,
            )
            self._publish_result(result)
            self._publish_status(f"{status}: {message}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotExecutorNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
