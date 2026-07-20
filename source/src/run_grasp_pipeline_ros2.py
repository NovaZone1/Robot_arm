#!/usr/bin/env python3
"""ROS2 entry point aligned with the old Piper grasp CLI."""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
from pathlib import Path
import socket
import shutil
import signal
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.grasping import EmergencyStopRequested, GraspExecutionConfig, GraspPipelineCoordinator
from src.robot.client import FakeRobotArmClient, Ros2PiperClient


logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

BUNDLE_ROOT = PROJECT_ROOT.parent
DEFAULT_HAND_EYE_CONFIG = PROJECT_ROOT / "config" / "hand_eye" / "verify_config.yaml"
TOP_DOWN_DEFAULT_MAX_ANGLE = 180.0
FLAT_MODE_MAX_ANGLE = 90.0
FLAT_MODE_PREGRASP_OFFSET = 0.10
FLAT_MODE_DESCEND_OFFSET = 0.008
FLAT_MODE_MIN_CLEARANCE = 0.05
SLENDER_OBJECT_MAX_ANGLE = 110.0


def is_slender_flat_prompt(prompt: str) -> bool:
    normalized = prompt.strip().lower()
    keywords = {
        "scissor",
        "scissors",
        "剪刀",
        "plier",
        "pliers",
        "钳子",
        "tweezer",
        "tweezers",
        "镊子",
    }
    return any(keyword in normalized for keyword in keywords)


def parse_grasp_y_bias_mm(value: str) -> float | None:
    normalized = value.strip().lower()
    if normalized == "auto":
        return None
    return float(value)


def load_tool_contact_offset_mm_from_json(json_path: str) -> tuple[float, float, float]:
    payload = json.loads(Path(json_path).read_text(encoding="utf-8"))
    values = payload.get("tool_contact_offset_tool_mm")
    if not isinstance(values, list) or len(values) != 3:
        raise RuntimeError(
            f"Invalid tool offset JSON: expected key 'tool_contact_offset_tool_mm' with 3 values in {json_path}"
        )
    return (float(values[0]), float(values[1]), float(values[2]))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ROS2 Piper grasp pipeline migration shell."
    )
    parser.add_argument("prompt", nargs="?", help="YOLOv8-seg COCO 类别名，如 cup")
    parser.add_argument("--hand-eye-config", default=str(DEFAULT_HAND_EYE_CONFIG))
    parser.add_argument("--graspnet-checkpoint", default="checkpoint.tar")
    parser.add_argument("--can", default="can0", help="CAN 接口名")
    parser.add_argument("--speed", type=int, default=40, help="运动速度百分比")
    parser.add_argument("--depth-fusion-frames", type=int, default=8)
    parser.add_argument(
        "--pointcloud-filter-mode",
        choices=("none", "bilateral", "median", "island", "radius"),
        default="bilateral",
    )
    parser.add_argument(
        "--pointcloud-backend",
        choices=("manual", "sdk"),
        default="sdk",
    )
    parser.add_argument("--bilateral-diameter", type=int, default=5)
    parser.add_argument("--bilateral-sigma-color", type=float, default=0.02)
    parser.add_argument("--bilateral-sigma-space", type=float, default=5.0)
    parser.add_argument("--median-kernel-size", type=int, default=5)
    parser.add_argument("--island-eps-m", type=float, default=0.02)
    parser.add_argument("--island-min-points", type=int, default=30)
    parser.add_argument("--radius-nb-points", type=int, default=12)
    parser.add_argument("--radius-m", type=float, default=0.02)
    parser.add_argument("--show-pointcloud", action="store_true", help="点云重建后弹出 Open3D 窗口")
    parser.add_argument("--min-grasp-score", type=float, default=0.01)
    parser.add_argument("--max-grasp-center-offset-m", type=float, default=0.35)
    parser.add_argument(
        "--safe-top-down-candidate-filter",
        action="store_true",
        help="候选筛选按 safe top-down 执行语义，只用最终 target 做 pose-floor 硬过滤",
    )
    parser.add_argument("--max-approach-angle", type=float, default=TOP_DOWN_DEFAULT_MAX_ANGLE)
    parser.add_argument("--max-reachable-rotation-delta-deg", type=float, default=180.0)
    parser.add_argument(
        "--allow-180deg-equivalent-grasp",
        dest="allow_180deg_equivalent_grasp",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-allow-180deg-equivalent-grasp",
        dest="allow_180deg_equivalent_grasp",
        action="store_false",
    )
    parser.add_argument("--gripper-open-mm", type=float, default=70.0)
    parser.add_argument("--gripper-length-m", type=float, default=0.105)
    parser.add_argument("--tool-contact-offset-scale", type=float, default=1.0)
    parser.add_argument("--apply-npoint-tool-offset", action="store_true")
    parser.add_argument(
        "--npoint-tool-offset-file",
        default="",
    )
    parser.add_argument(
        "--grasp-y-bias-mm",
        type=parse_grasp_y_bias_mm,
        default=0,
    )
    parser.add_argument("--grasp-close-effort-nm", type=float, default=0.6)
    parser.add_argument("--gripper-close-timeout-s", type=float, default=6.0)
    parser.add_argument("--apply-online-bias", action="store_true")
    parser.add_argument(
        "--online-bias-file",
        default="",
    )
    parser.add_argument("--flat-object-mode", action="store_true")
    parser.add_argument("--enable-pregrasp", action="store_true")
    parser.add_argument("--pregrasp-offset-m", type=float, default=0.0)
    parser.add_argument("--descend-offset-m", type=float, default=0.0)
    parser.add_argument("--grasp-z-offset-m", type=float, default=0.0)
    parser.add_argument("--retreat-offset-m", type=float, default=0.0)
    parser.add_argument("--table-z-m", type=float, default=0.0)
    parser.add_argument("--min-gripper-table-clearance-m", type=float, default=0.03)
    parser.add_argument(
        "--home-pose",
        type=float,
        nargs=6,
        default=(57, 0, 215, 0, 85, 0),
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
    )
    parser.add_argument(
        "--handoff-pose",
        type=float,
        nargs=6,
        default=(200, 20, 300, 10, 120, 0),
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
    )
    parser.add_argument(
        "--observe-pose",
        type=float,
        nargs=6,
        default=(30, 0, 400, 0, 120, 0),
        metavar=("X", "Y", "Z", "ROLL", "PITCH", "YAW"),
    )
    parser.add_argument("--command-time-s", type=float, default=1.5)
    parser.add_argument("--move-timeout-s", type=float, default=2.0)
    parser.add_argument("--settle-time-s", type=float, default=0.8)
    parser.add_argument("--center-settle-time-s", type=float, default=0.8)
    parser.add_argument("--center-max-step-m", type=float, default=0.02)
    parser.add_argument("--precenter", action="store_true")
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--workspace-x", type=float, nargs=2, default=(0.10, 1.20))
    parser.add_argument("--workspace-y", type=float, nargs=2, default=(-0.50, 0.50))
    parser.add_argument("--workspace-z", type=float, nargs=2, default=(0.00, 0.60))
    parser.add_argument("--dry-run", action="store_true", help="兼容旧参数；默认也是仅规划")
    parser.add_argument("--execute", action="store_true", help="执行真实机器人运动")
    parser.add_argument(
        "--robot-backend",
        choices=("none", "fake", "ros2"),
        default="none",
        help="机器人后端：none/fake/ros2",
    )
    parser.add_argument("--probe-robot", action="store_true")
    parser.add_argument("--enable-robot", action="store_true")
    return parser


def build_config(args: argparse.Namespace) -> tuple[GraspExecutionConfig, dict[str, object]]:
    home_pose = tuple(args.home_pose)
    handoff_pose = tuple(args.handoff_pose)
    observe_pose = tuple(args.observe_pose)
    dry_run = not args.execute
    max_approach_angle = args.max_approach_angle
    pregrasp_offset_m = args.pregrasp_offset_m
    descend_offset_m = args.descend_offset_m
    min_gripper_table_clearance_m = args.min_gripper_table_clearance_m

    if args.flat_object_mode:
        max_approach_angle = min(max_approach_angle, FLAT_MODE_MAX_ANGLE)
        pregrasp_offset_m = max(pregrasp_offset_m, FLAT_MODE_PREGRASP_OFFSET)
        descend_offset_m = min(descend_offset_m, FLAT_MODE_DESCEND_OFFSET)
        min_gripper_table_clearance_m = max(
            min_gripper_table_clearance_m,
            FLAT_MODE_MIN_CLEARANCE,
        )
    elif args.prompt and is_slender_flat_prompt(args.prompt):
        max_approach_angle = max(max_approach_angle, SLENDER_OBJECT_MAX_ANGLE)

    tool_contact_offset_mm = None
    if args.apply_npoint_tool_offset and args.npoint_tool_offset_file:
        tool_contact_offset_mm = load_tool_contact_offset_mm_from_json(args.npoint_tool_offset_file)
        tool_contact_offset_mm = tuple(float(value) * float(args.tool_contact_offset_scale) for value in tool_contact_offset_mm)

    config = GraspExecutionConfig(
        hand_eye_config_path=args.hand_eye_config,
        graspnet_checkpoint=args.graspnet_checkpoint,
        depth_fusion_frames=args.depth_fusion_frames,
        pointcloud_filter_mode=args.pointcloud_filter_mode,
        pointcloud_backend=args.pointcloud_backend,
        bilateral_diameter=args.bilateral_diameter,
        bilateral_sigma_color=args.bilateral_sigma_color,
        bilateral_sigma_space=args.bilateral_sigma_space,
        median_kernel_size=args.median_kernel_size,
        island_eps_m=args.island_eps_m,
        island_min_points=args.island_min_points,
        radius_nb_points=args.radius_nb_points,
        radius_m=args.radius_m,
        show_pointcloud=args.show_pointcloud,
        robot_can_name=args.can,
        robot_speed_percent=args.speed,
        dry_run=dry_run,
        flat_object_mode=args.flat_object_mode,
        enable_pregrasp=args.enable_pregrasp,
        precenter_before_grasp=args.precenter,
        confirm_before_execute=args.confirm,
        max_approach_angle_deg=max_approach_angle,
        min_grasp_score=args.min_grasp_score,
        max_grasp_center_offset_m=args.max_grasp_center_offset_m,
        safe_top_down_candidate_filter=args.safe_top_down_candidate_filter,
        max_reachable_rotation_delta_deg=args.max_reachable_rotation_delta_deg,
        allow_180deg_equivalent_grasp=args.allow_180deg_equivalent_grasp,
        gripper_open_mm=args.gripper_open_mm,
        grasp_close_effort_nm=args.grasp_close_effort_nm,
        gripper_close_timeout_s=args.gripper_close_timeout_s,
        online_bias_enabled=args.apply_online_bias,
        online_bias_path=args.online_bias_file if args.apply_online_bias else None,
        gripper_length_m=args.gripper_length_m,
        tool_contact_offset_tool_m=(
            tuple(v / 1000.0 for v in tool_contact_offset_mm)
            if tool_contact_offset_mm is not None
            else None
        ),
        grasp_y_bias_mm=args.grasp_y_bias_mm,
        pregrasp_offset_m=pregrasp_offset_m,
        descend_offset_m=descend_offset_m,
        grasp_z_offset_m=args.grasp_z_offset_m,
        retreat_offset_m=args.retreat_offset_m,
        table_z_m=args.table_z_m,
        min_gripper_table_clearance_m=min_gripper_table_clearance_m,
        home_pose_mm_deg=home_pose,
        handoff_pose_mm_deg=handoff_pose,
        observe_pose_mm_deg=observe_pose,
        command_time_s=args.command_time_s,
        move_check_timeout_s=args.move_timeout_s,
        settle_time_s=args.settle_time_s,
        center_settle_time_s=args.center_settle_time_s,
        center_max_step_m=args.center_max_step_m,
        workspace_x_limits_m=tuple(args.workspace_x),
        workspace_y_limits_m=tuple(args.workspace_y),
        workspace_z_limits_m=tuple(args.workspace_z),
    )

    summary = {
        "prompt": args.prompt,
        "robot_backend": args.robot_backend,
        "can": args.can,
        "speed": args.speed,
        "show_pointcloud": args.show_pointcloud,
        "execute": args.execute,
        "dry_run": dry_run,
        "max_approach_angle_deg": max_approach_angle,
        "safe_top_down_candidate_filter": args.safe_top_down_candidate_filter,
        "home_pose": home_pose,
        "handoff_pose": handoff_pose,
        "observe_pose": observe_pose,
        "apply_online_bias": args.apply_online_bias,
        "apply_npoint_tool_offset": args.apply_npoint_tool_offset,
        "tool_contact_offset_scale": args.tool_contact_offset_scale,
        "tool_contact_offset_mm": tool_contact_offset_mm,
    }
    return config, summary


def make_robot_client(backend: str):
    if backend == "fake":
        return FakeRobotArmClient()
    if backend == "ros2":
        return Ros2PiperClient()
    return None


def _module_check(name: str) -> tuple[bool, str]:
    try:
        importlib.import_module(name)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"


def _check_realsense_device() -> tuple[bool, str]:
    try:
        import pyrealsense2 as rs  # type: ignore
    except Exception as exc:
        return False, f"pyrealsense2 import failed: {type(exc).__name__}: {exc}"
    try:
        context = rs.context()
        devices = list(context.query_devices())
    except Exception as exc:
        return False, f"RealSense device query failed: {type(exc).__name__}: {exc}"
    if not devices:
        return False, "No RealSense device detected"
    names: list[str] = []
    for device in devices:
        try:
            names.append(device.get_info(rs.camera_info.name))
        except Exception:
            names.append("unknown")
    return True, ", ".join(names)


def _has_host_interface(interface_name: str) -> bool:
    try:
        return interface_name in {name for _, name in socket.if_nameindex()}
    except OSError:
        return False


def _unique_paths(paths: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for path in paths:
        if not path:
            continue
        normalized = str(Path(path).expanduser().resolve())
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _resolve_graspnet_checkpoint_path(configured_path: str) -> str:
    search_bases = [
        PROJECT_ROOT / "src" / "perception",
        PROJECT_ROOT / "src",
        PROJECT_ROOT,
        REPO_ROOT,
        Path.cwd(),
    ]
    env_root = os.environ.get("GRASPNET_BASELINE_ROOT", "")
    env_checkpoint = os.environ.get("GRASPNET_CHECKPOINT", "")

    candidates: list[str] = [configured_path, env_checkpoint]
    if env_root:
        candidates.extend(
            [
                str(Path(env_root) / "checkpoint-rs.tar"),
                str(Path(env_root) / "checkpoint.tar"),
            ]
        )

    for base in search_bases:
        candidates.append(str(base / "checkpoint-rs.tar"))
        candidates.append(str(base / "checkpoint.tar"))
        candidates.append(str(base / "graspnet-baseline" / "checkpoint-rs.tar"))
        candidates.append(str(base / "graspnet-baseline" / "checkpoint.tar"))
        candidates.append(str(base / "graspnet_baseline" / "checkpoint-rs.tar"))
        candidates.append(str(base / "graspnet_baseline" / "checkpoint.tar"))

    for candidate in _unique_paths(candidates):
        if Path(candidate).is_file():
            return candidate
    return configured_path


def run_startup_preflight(
    *,
    args: argparse.Namespace,
    config: GraspExecutionConfig,
) -> tuple[list[str], list[str], list[str]]:
    info: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []

    needs_robot = args.robot_backend == "ros2"
    needs_perception = not args.probe_robot

    if needs_robot:
        ros_setup = Path("/opt/ros/humble/setup.bash")
        overlay_setup = BUNDLE_ROOT / "piper_ros_ws" / "install" / "setup.bash"
        if ros_setup.exists():
            info.append(f"found ROS setup: {ros_setup}")
        else:
            warnings.append(f"ROS setup script not found: {ros_setup}")
        if overlay_setup.exists():
            info.append(f"found piper_ros overlay: {overlay_setup}")
        else:
            warnings.append(f"piper_ros overlay setup not found: {overlay_setup}")

        if shutil.which("ros2"):
            info.append("ros2 CLI is on PATH")
        else:
            warnings.append("ros2 CLI is not on PATH")

        if _has_host_interface(args.can):
            info.append(f"host interface present: {args.can}")
        else:
            warnings.append(
                f"host interface not found: {args.can}. "
                "If you will launch piper_single_ctrl on this machine, bring up the CAN device first "
                "or pass --can with the actual interface name."
            )

        for module_name in ("rclpy", "geometry_msgs.msg", "sensor_msgs.msg", "piper_msgs.msg", "piper_msgs.srv"):
            ok, detail = _module_check(module_name)
            if ok:
                info.append(f"import ok: {module_name}")
            else:
                errors.append(
                    f"missing ROS runtime module {module_name}: {detail}. "
                    "Source /opt/ros/humble/setup.bash and the piper_ws overlay."
                )

    if needs_perception:
        for module_name in (
            "cv2",
            "numpy",
            "open3d",
            "PIL",
            "pyrealsense2",
            "torch",
            "ultralytics",
        ):
            ok, detail = _module_check(module_name)
            if ok:
                info.append(f"import ok: {module_name}")
            else:
                errors.append(f"missing perception module {module_name}: {detail}")

        if "pyrealsense2" not in "\n".join(errors):
            ok, detail = _check_realsense_device()
            if ok:
                info.append(f"RealSense detected: {detail}")
            else:
                errors.append(detail)

        checkpoint_path = _resolve_graspnet_checkpoint_path(config.graspnet_checkpoint)
        if checkpoint_path and Path(checkpoint_path).is_file():
            info.append(f"GraspNet checkpoint: {checkpoint_path}")
        else:
            errors.append(
                "GraspNet checkpoint not found. "
                f"Configured value={config.graspnet_checkpoint!r}. "
                "Pass --graspnet-checkpoint with a real file path."
            )

        info.append("segmentation backend: YOLOv8-seg (yolov8n-seg.pt)")

    return info, warnings, errors


def probe_robot(robot_client, enable_robot: bool, expected_can: str | None = None) -> None:
    robot_client.connect()
    try:
        if enable_robot:
            print(f"enable_result: {robot_client.enable()}")
        print(f"arm_status: {robot_client.format_arm_status()}")
        try:
            pose = robot_client.read_end_pose_mm_deg()
        except TimeoutError as exc:
            can_hint = ""
            if expected_can and not _has_host_interface(expected_can):
                can_hint = (
                    f" Host interface {expected_can!r} is not present right now, "
                    "so a local piper_single_ctrl process cannot talk to the robot until CAN is brought up."
                )
            raise RuntimeError(
                f"No pose feedback received on {robot_client.config.end_pose_topic}. "
                "Make sure the piper_ros single-arm node is running and publishing live robot feedback."
                f"{can_hint}"
            ) from exc
        print(
            "tcp_pose_mm_deg: "
            f"({pose.x_mm:.2f}, {pose.y_mm:.2f}, {pose.z_mm:.2f}, "
            f"{pose.roll_deg:.2f}, {pose.pitch_deg:.2f}, {pose.yaw_deg:.2f})"
        )
        gripper = robot_client.get_gripper_status()
        print(
            "gripper_status: "
            f"(open_mm={gripper.angle_mm:.2f}, effort_nm={gripper.effort_nm:.2f}, "
            f"enabled={gripper.enabled})"
        )
    finally:
        robot_client.disconnect()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if not args.prompt and not args.probe_robot:
        parser.error("prompt is required unless --probe-robot is used")

    config, summary = build_config(args)
    hand_eye_config = Path(config.hand_eye_config_path).expanduser().resolve()
    if not hand_eye_config.exists():
        raise SystemExit(f"hand-eye config not found: {hand_eye_config}")
    online_bias = None
    if args.apply_online_bias:
        online_bias = Path(args.online_bias_file).expanduser().resolve()
        if not online_bias.exists():
            raise SystemExit(f"online bias file not found: {online_bias}")

    logger.debug("初始化 ROS2 迁移入口配置")
    for key, value in summary.items():
        logger.debug("  %s: %s", key, value)

    preflight_info, preflight_warnings, preflight_errors = run_startup_preflight(
        args=args,
        config=config,
    )
    for line in preflight_info:
        logger.debug("preflight ok: %s", line)
    for line in preflight_warnings:
        logger.warning("preflight warn: %s", line)
    if preflight_errors:
        print("Runtime preflight failed:")
        for line in preflight_errors:
            print("  -", line)
        raise SystemExit(
            "Fix the runtime dependencies above and rerun. "
            "Use --probe-robot with a chosen backend if you only want robot connectivity checks."
        )

    coordinator = GraspPipelineCoordinator(config, hand_eye_config, online_bias)
    robot_client = make_robot_client(args.robot_backend)
    if robot_client is not None:
        coordinator.attach_robot_client(robot_client)

    previous_sigint_handler = signal.getsignal(signal.SIGINT)

    def handle_sigint(signum, frame):
        try:
            coordinator.request_emergency_stop("Ctrl+C")
        except Exception as exc:  # pragma: no cover
            logger.warning("SIGINT emergency_stop failed: %s", exc)
        raise EmergencyStopRequested("Emergency stop requested by Ctrl+C")

    signal.signal(signal.SIGINT, handle_sigint)
    try:
        print("Migration shell ready:")
        for line in coordinator.describe_environment():
            print("  ", line)
        print(f"  robot_backend: {args.robot_backend}")
        if args.prompt:
            print(f"  prompt: {args.prompt}")

        if args.probe_robot:
            if robot_client is None:
                raise SystemExit("--probe-robot requires --robot-backend fake or ros2")
            try:
                probe_robot(robot_client, args.enable_robot, args.can)
            except RuntimeError as exc:
                raise SystemExit(str(exc)) from exc
            return

        if robot_client is None:
            raise SystemExit(
                "A robot backend is required for the migrated pipeline. "
                "Use --robot-backend fake for dry-run validation or --robot-backend ros2 for hardware."
            )

        coordinator.connect()
        result = coordinator.run_once(args.prompt)

        print("\nPipeline summary:")
        print(result["summary"])
        diagnostics = result.get("diagnostics") or []
        if diagnostics:
            print("\nDiagnostics:")
            for line in diagnostics:
                print("  ", line)
        candidate_preview_lines = result.get("candidate_preview_lines") or []
        if candidate_preview_lines:
            print("\nCandidate Preview:")
            for index, line in enumerate(candidate_preview_lines, start=1):
                print(f"  #{index} {line}")
        plan = result.get("plan")
        if plan is not None:
            print(
                "\nPlan target:"
                f" target_base_m={tuple(round(v, 4) for v in plan.target_base_m)}"
                f" target_rpy_deg={tuple(round(v, 2) for v in plan.target_rpy_deg)}"
                f" within_workspace={plan.within_workspace}"
            )
    except EmergencyStopRequested as exc:
        logger.warning("急停触发: %s", exc)
    finally:
        if not args.probe_robot:
            try:
                coordinator.disconnect()
            except Exception as exc:
                logger.warning("disconnect failed: %s", exc)
        signal.signal(signal.SIGINT, previous_sigint_handler)


if __name__ == "__main__":
    main()
