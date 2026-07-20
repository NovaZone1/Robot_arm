from __future__ import annotations

import json
from pathlib import Path
import threading
import traceback

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import String
from std_srvs.srv import Trigger

from robot_grasp_ros2.distributed_utils import make_latched_qos
from robot_grasp_ros2.rviz_visualization import PipelineRvizPublisher
from src.grasping import EmergencyStopRequested, GraspPipelineCoordinator
from src.robot.client import Ros2PiperClient
from src.run_grasp_pipeline_ros2 import (
    DEFAULT_HAND_EYE_CONFIG,
    build_config,
    build_parser,
    make_robot_client,
    probe_robot,
    run_startup_preflight,
)


class GraspPipelineNode(Node):
    """ROS2 wrapper node around the current migrated grasp pipeline."""

    def __init__(self) -> None:
        super().__init__("grasp_pipeline")
        self._declare_parameters()

        text_qos = make_latched_qos(depth=10)
        diagnostics_qos = make_latched_qos(depth=20)
        self._status_pub = self.create_publisher(String, "~/status", text_qos)
        self._summary_pub = self.create_publisher(String, "~/summary", text_qos)
        self._diagnostics_pub = self.create_publisher(String, "~/diagnostics", diagnostics_qos)
        self._result_pub = self.create_publisher(String, "~/result_json", text_qos)
        self._rviz_pub = PipelineRvizPublisher(self)

        self.create_subscription(String, "~/run_prompt", self._handle_run_prompt, 10)
        self.create_service(Trigger, "~/run", self._handle_run_service)
        self.create_service(Trigger, "~/probe", self._handle_probe_service)
        self.create_service(Trigger, "~/stop", self._handle_stop_service)

        self._run_lock = threading.Lock()
        self._run_thread: threading.Thread | None = None
        self._active_coordinator: GraspPipelineCoordinator | None = None

        self._auto_start_armed = bool(self.get_parameter("auto_start").value)
        self._auto_start_timer = self.create_timer(0.5, self._maybe_auto_start)
        self._publish_status("idle")

    def _declare_parameters(self) -> None:
        self.declare_parameter("prompt", "")
        self.declare_parameter("robot_backend", "fake")
        self.declare_parameter("execute", False)
        self.declare_parameter("probe_robot", False)
        self.declare_parameter("enable_robot", False)
        self.declare_parameter("show_pointcloud", False)
        self.declare_parameter("precenter", False)
        self.declare_parameter("confirm", False)
        self.declare_parameter("pointcloud_filter_mode", "bilateral")
        self.declare_parameter("pointcloud_backend", "sdk")
        self.declare_parameter("depth_fusion_frames", 8)
        self.declare_parameter("can", "can0")
        self.declare_parameter("speed", 40)
        self.declare_parameter("graspnet_checkpoint", "checkpoint.tar")
        self.declare_parameter("hand_eye_config", str(DEFAULT_HAND_EYE_CONFIG))
        self.declare_parameter("extra_cli_args", [])
        self.declare_parameter("auto_start", False)
        self.declare_parameter("rviz_camera_frame", "camera_color_optical_frame")
        self.declare_parameter("rviz_base_frame", "base_link")
        self.declare_parameter("rviz_candidate_topk", 5)

    def _publish_status(self, text: str) -> None:
        self.get_logger().info(text)
        self._status_pub.publish(String(data=text))

    def _publish_summary(self, text: str) -> None:
        self._summary_pub.publish(String(data=text))

    def _publish_diagnostics(self, lines: list[str]) -> None:
        for line in lines:
            self._diagnostics_pub.publish(String(data=line))

    def _publish_result(self, payload: dict[str, object]) -> None:
        self._result_pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))

    def _parser_defaults(self):
        return build_parser().parse_args([])

    def _rviz_camera_frame(self) -> str:
        return str(self.get_parameter("rviz_camera_frame").value or "camera_color_optical_frame")

    def _rviz_base_frame(self) -> str:
        return str(self.get_parameter("rviz_base_frame").value or "base_link")

    def _rviz_candidate_topk(self) -> int:
        return int(self.get_parameter("rviz_candidate_topk").value)

    def _clear_rviz(self) -> None:
        self._rviz_pub.clear(
            camera_frame=self._rviz_camera_frame(),
            base_frame=self._rviz_base_frame(),
        )

    def _publish_rviz_result(self, result: dict[str, object]) -> None:
        self._rviz_pub.publish_result(
            result,
            camera_frame=self._rviz_camera_frame(),
            base_frame=self._rviz_base_frame(),
            candidate_topk=self._rviz_candidate_topk(),
        )

    def _build_args(self, prompt_override: str | None = None, *, force_probe: bool = False):
        parser = build_parser()
        args = self._parser_defaults()
        args.prompt = str(self.get_parameter("prompt").value or "")
        args.robot_backend = str(self.get_parameter("robot_backend").value or "fake")
        args.execute = bool(self.get_parameter("execute").value)
        args.probe_robot = bool(self.get_parameter("probe_robot").value)
        args.enable_robot = bool(self.get_parameter("enable_robot").value)
        args.show_pointcloud = bool(self.get_parameter("show_pointcloud").value)
        args.precenter = bool(self.get_parameter("precenter").value)
        args.confirm = bool(self.get_parameter("confirm").value)
        args.pointcloud_filter_mode = str(self.get_parameter("pointcloud_filter_mode").value or "bilateral")
        args.pointcloud_backend = str(self.get_parameter("pointcloud_backend").value or "sdk")
        args.depth_fusion_frames = int(self.get_parameter("depth_fusion_frames").value)
        args.can = str(self.get_parameter("can").value or "can0")
        args.speed = int(self.get_parameter("speed").value)
        args.graspnet_checkpoint = str(self.get_parameter("graspnet_checkpoint").value or "checkpoint.tar")
        args.hand_eye_config = str(self.get_parameter("hand_eye_config").value or DEFAULT_HAND_EYE_CONFIG)

        extra_cli_args = list(self.get_parameter("extra_cli_args").value or [])
        if extra_cli_args:
            args = parser.parse_args(extra_cli_args, namespace=args)

        if prompt_override is not None:
            args.prompt = prompt_override
        if force_probe:
            args.probe_robot = True
        return args

    def _maybe_auto_start(self) -> None:
        if not self._auto_start_armed:
            return
        self._auto_start_armed = False
        prompt = str(self.get_parameter("prompt").value or "").strip()
        if not prompt and not bool(self.get_parameter("probe_robot").value):
            self._publish_status("auto_start skipped: empty prompt")
            return
        accepted, message = self._start_background_run(prompt_override=prompt or None)
        if not accepted:
            self._publish_status(f"auto_start rejected: {message}")

    def _handle_run_prompt(self, msg: String) -> None:
        prompt = msg.data.strip()
        accepted, message = self._start_background_run(prompt_override=prompt)
        if not accepted:
            self._publish_status(f"run_prompt rejected: {message}")

    def _handle_run_service(self, _request, response):
        prompt = str(self.get_parameter("prompt").value or "").strip()
        accepted, message = self._start_background_run(prompt_override=prompt or None)
        response.success = accepted
        response.message = message
        return response

    def _handle_probe_service(self, _request, response):
        accepted, message = self._start_background_run(prompt_override=None, force_probe=True)
        response.success = accepted
        response.message = message
        return response

    def _handle_stop_service(self, _request, response):
        coordinator = self._active_coordinator
        if coordinator is None:
            response.success = False
            response.message = "pipeline is not running"
            return response
        try:
            coordinator.request_emergency_stop("stop service")
        except Exception as exc:
            response.success = False
            response.message = f"failed to stop pipeline: {exc}"
            return response
        response.success = True
        response.message = "stop requested"
        return response

    def _start_background_run(
        self,
        prompt_override: str | None,
        *,
        force_probe: bool = False,
    ) -> tuple[bool, str]:
        with self._run_lock:
            if self._run_thread is not None and self._run_thread.is_alive():
                return False, "pipeline is already running"

            args = self._build_args(prompt_override=prompt_override, force_probe=force_probe)
            if not args.prompt and not args.probe_robot:
                return False, "prompt is empty; set parameter 'prompt' or publish to ~/run_prompt"

            worker = threading.Thread(
                target=self._run_pipeline_thread,
                args=(args,),
                daemon=True,
                name="grasp-pipeline-runner",
            )
            self._run_thread = worker
            worker.start()
            mode = "probe" if args.probe_robot else "run"
            prompt_text = args.prompt if args.prompt else "<none>"
            return True, f"{mode} accepted for prompt={prompt_text}"

    def _prepare_runtime(self, args):
        config, _summary = build_config(args)
        hand_eye_config = Path(config.hand_eye_config_path).expanduser().resolve()
        if not hand_eye_config.exists():
            raise RuntimeError(f"hand-eye config not found: {hand_eye_config}")

        online_bias = None
        if args.apply_online_bias:
            online_bias = Path(args.online_bias_file).expanduser().resolve()
            if not online_bias.exists():
                raise RuntimeError(f"online bias file not found: {online_bias}")

        preflight_info, preflight_warnings, preflight_errors = run_startup_preflight(
            args=args,
            config=config,
        )
        for line in preflight_info:
            self.get_logger().info(f"preflight ok: {line}")
        for line in preflight_warnings:
            self.get_logger().warning(f"preflight warn: {line}")
        if preflight_errors:
            raise RuntimeError("Runtime preflight failed: " + " | ".join(preflight_errors))

        coordinator = GraspPipelineCoordinator(config, hand_eye_config, online_bias)
        robot_client = make_robot_client(args.robot_backend)
        if isinstance(robot_client, Ros2PiperClient):
            robot_client.attach_ros_node(self)
        if robot_client is not None:
            coordinator.attach_robot_client(robot_client)
        return config, coordinator, robot_client

    def _run_pipeline_thread(self, args) -> None:
        status = "completed"
        message = ""
        diagnostics: list[str] = []
        result_payload: dict[str, object] = {}
        coordinator: GraspPipelineCoordinator | None = None

        try:
            self._clear_rviz()
            self._publish_status("preflight")
            _config, coordinator, robot_client = self._prepare_runtime(args)
            self._active_coordinator = coordinator

            if args.probe_robot:
                self._publish_status("probing_robot")
                if robot_client is None:
                    raise RuntimeError("probe requires robot backend fake or ros2")
                probe_robot(robot_client, args.enable_robot, args.can)
                message = "probe completed"
                result_payload = {
                    "status": "probe_completed",
                    "robot_backend": args.robot_backend,
                }
            else:
                self._publish_status("connecting")
                coordinator.connect()
                self._publish_status("running_pipeline")
                result = coordinator.run_once(args.prompt)
                self._publish_rviz_result(result)
                diagnostics = list(result.get("diagnostics") or [])
                message = str(result.get("summary") or "")
                self._publish_summary(message)
                self._publish_diagnostics(diagnostics)
                result_payload = {
                    "status": str(result.get("status") or "ok"),
                    "robot_backend": args.robot_backend,
                    "prompt": args.prompt,
                    "summary": message,
                    "diagnostics": diagnostics,
                }
        except EmergencyStopRequested as exc:
            status = "stopped"
            message = str(exc)
            self.get_logger().warning("pipeline stopped: %s", exc)
        except Exception as exc:
            status = "failed"
            message = str(exc)
            diagnostics = [str(exc), traceback.format_exc()]
            self.get_logger().error("pipeline failed: %s", exc)
            self._publish_diagnostics(diagnostics)
        finally:
            if coordinator is not None and not args.probe_robot:
                try:
                    coordinator.disconnect()
                except Exception as exc:
                    self.get_logger().warning("disconnect failed: %s", exc)
            self._active_coordinator = None
            final_status = f"{status}: {message}" if message else status
            self._publish_status(final_status)
            payload = {"status": status, "message": message}
            payload.update(result_payload)
            self._publish_result(payload)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = GraspPipelineNode()
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
