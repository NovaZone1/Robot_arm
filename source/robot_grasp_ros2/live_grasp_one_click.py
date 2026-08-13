from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import IO


@dataclass(frozen=True)
class LaunchSpec:
    name: str
    argv: list[str]


@dataclass(frozen=True)
class LaunchPlan:
    project_root: Path
    prompt: str
    robot_backend: str
    execute: bool
    once: bool
    launches: list[LaunchSpec]
    configure_param_argvs: list[list[str]]
    probe_argv: list[str]
    trigger_argv: list[str]
    compute_ik_service: str | None
    probe_timeout_s: float
    moveit_warmup_s: float
    result_timeout_s: float


@dataclass(frozen=True)
class RunningComponents:
    driver_running: bool
    moveit_running: bool
    distributed_running: bool
    distributed_partial_nodes: tuple[str, ...]
    rviz_running: bool
    details: tuple[str, ...]


class RunnerError(RuntimeError):
    pass


class ChildExitError(RunnerError):
    def __init__(self, name: str, returncode: int, log_path: Path):
        super().__init__(
            f"{name} exited unexpectedly with code {returncode}. Check log: {log_path}"
        )
        self.name = name
        self.returncode = returncode
        self.log_path = log_path


@dataclass
class ManagedProcess:
    spec: LaunchSpec
    process: subprocess.Popen[str]
    log_path: Path
    log_handle: IO[str]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Start the live Piper grasp stack, wait for readiness, open RViz, and "
            "trigger one grasp run while keeping the stack alive afterward."
        )
    )
    parser.add_argument(
        "prompt",
        nargs="?",
        default="cup",
        help="Prompt to send to /grasp_pipeline. Default: cup",
    )
    parser.add_argument(
        "--robot-backend",
        choices=("ros2", "fake"),
        default="ros2",
        help="Robot backend to start. Default: ros2.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Allow final robot execution. Without this flag the run is planning-only.",
    )
    parser.add_argument(
        "--plan-only",
        dest="execute",
        action="store_false",
        help="Run observation, perception, and planning without executing the final grasp.",
    )
    parser.set_defaults(execute=False)
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Require a later /grasp_pipeline/confirm before final grasp execution.",
    )
    parser.add_argument(
        "--precenter",
        action="store_true",
        help="Enable distributed precenter before final planning.",
    )
    parser.add_argument(
        "--enable-pregrasp",
        action="store_true",
        help="Enable the pregrasp waypoint in planner and executor.",
    )
    parser.add_argument(
        "--show-pointcloud",
        action="store_true",
        help="Ask vision_worker to open the Open3D pointcloud preview during analysis.",
    )
    parser.add_argument(
        "--no-rviz",
        dest="open_rviz",
        action="store_false",
        help="Skip launching RViz.",
    )
    parser.set_defaults(open_rviz=True)
    parser.add_argument(
        "--probe-timeout",
        type=float,
        default=60.0,
        help="Seconds to wait for /compute_ik and /grasp_pipeline/probe readiness. Default: 60.",
    )
    parser.add_argument(
        "--moveit-warmup",
        type=float,
        default=0.0,
        help="Minimum warm-up time to leave after /compute_ik first appears. Default: 0.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Trigger one run, wait for its result artifact, then stop only processes started by this wrapper.",
    )
    parser.add_argument(
        "--result-timeout",
        type=float,
        default=600.0,
        help="Seconds to wait for one-run result when --once is used. Default: 600.",
    )
    return parser


def build_launch_plan(project_root: Path, args: argparse.Namespace) -> LaunchPlan:
    robot_backend = str(args.robot_backend)
    execute = bool(args.execute)
    confirm = bool(args.confirm or (robot_backend == "ros2" and execute))
    launches: list[LaunchSpec] = []
    compute_ik_service: str | None = None

    if robot_backend == "ros2":
        launches.extend(
            [
                LaunchSpec("piper_driver", ["./scripts/run_piper_driver.sh"]),
                LaunchSpec("moveit_ik", ["./scripts/run_piper_moveit_ik.sh"]),
            ]
        )
        compute_ik_service = "/compute_ik"

    distributed_argv = [
        "./scripts/run_distributed_stack_graspnet.sh",
        "--robot-backend",
        robot_backend,
    ]
    if robot_backend == "ros2":
        distributed_argv.extend(["--pose-execution-mode", "moveit_ik"])
    if execute:
        distributed_argv.append("--execute")
    if confirm:
        distributed_argv.append("--confirm")
    if args.precenter:
        distributed_argv.append("--precenter")
    if args.enable_pregrasp:
        distributed_argv.append("--enable-pregrasp")
    if args.show_pointcloud:
        distributed_argv.append("--show-pointcloud")
    launches.append(LaunchSpec("distributed_stack", distributed_argv))

    if args.open_rviz:
        launches.append(LaunchSpec("rviz", ["./scripts/open_distributed_rviz.sh"]))

    def bool_text(value: bool) -> str:
        return "true" if value else "false"

    return LaunchPlan(
        project_root=project_root,
        prompt=args.prompt,
        robot_backend=robot_backend,
        execute=execute,
        once=bool(args.once),
        launches=launches,
        configure_param_argvs=[
            ["./scripts/ros2_system.sh", "param", "set", "/grasp_pipeline", "execute", bool_text(execute)],
            ["./scripts/ros2_system.sh", "param", "set", "/grasp_pipeline", "confirm", bool_text(confirm)],
            ["./scripts/ros2_system.sh", "param", "set", "/grasp_pipeline", "precenter", bool_text(bool(args.precenter))],
            [
                "./scripts/ros2_system.sh",
                "param",
                "set",
                "/grasp_pipeline",
                "show_pointcloud",
                bool_text(bool(args.show_pointcloud)),
            ],
            [
                "./scripts/ros2_system.sh",
                "param",
                "set",
                "/grasp_pipeline",
                "enable_pregrasp",
                bool_text(bool(args.enable_pregrasp)),
            ],
            [
                "./scripts/ros2_system.sh",
                "param",
                "set",
                "/robot_executor",
                "enable_pregrasp",
                bool_text(bool(args.enable_pregrasp)),
            ],
        ],
        probe_argv=[
            "./scripts/ros2_system.sh",
            "service",
            "call",
            "/grasp_pipeline/probe",
            "std_srvs/srv/Trigger",
            "{}",
        ],
        trigger_argv=["./scripts/run_pipeline_service.sh", args.prompt],
        compute_ik_service=compute_ik_service,
        probe_timeout_s=float(args.probe_timeout),
        moveit_warmup_s=float(args.moveit_warmup),
        result_timeout_s=float(args.result_timeout),
    )


def _session_root(project_root: Path) -> Path:
    workspace_root = project_root.parent
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return workspace_root / "log" / "one_click" / timestamp


def _process_lines(pattern: str) -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af", pattern],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def detect_running_components(project_root: Path) -> RunningComponents:
    rviz_config = project_root / "rviz" / "distributed_grasp_pipeline.rviz"
    driver_lines = _process_lines("piper_single_ctrl")
    moveit_lines = _process_lines("move_group")
    distributed_lines = _process_lines(
        "robot_grasp_ros2\\.(camera_server_node|vision_worker_node|robot_executor_node|pipeline_orchestrator_node|scout_scan_controller_node)"
    )
    rviz_lines = _process_lines(f"rviz2 .*{rviz_config}")

    distributed_node_names: list[str] = []
    for node_name in (
        "camera_server_node",
        "vision_worker_node",
        "robot_executor_node",
        "pipeline_orchestrator_node",
        "scout_scan_controller_node",
    ):
        dotted = f"robot_grasp_ros2.{node_name}"
        if any(dotted in line for line in distributed_lines):
            distributed_node_names.append(node_name)

    distributed_running = False
    distributed_partial_nodes: tuple[str, ...] = ()
    if distributed_node_names:
        if len(distributed_node_names) == 5:
            distributed_running = True
        else:
            distributed_partial_nodes = tuple(distributed_node_names)

    return RunningComponents(
        driver_running=bool(driver_lines),
        moveit_running=bool(moveit_lines),
        distributed_running=distributed_running,
        distributed_partial_nodes=distributed_partial_nodes,
        rviz_running=bool(rviz_lines),
        details=tuple(driver_lines + moveit_lines + distributed_lines + rviz_lines),
    )


def resolve_launches_for_running_components(
    plan: LaunchPlan,
    running: RunningComponents,
) -> list[LaunchSpec]:
    if running.distributed_partial_nodes:
        joined = ", ".join(running.distributed_partial_nodes)
        raise RunnerError(
            f"Partial distributed stack detected ({joined}). Stop it first to avoid mixing old and new nodes."
        )

    launches_to_start: list[LaunchSpec] = []
    for spec in plan.launches:
        if spec.name == "piper_driver" and running.driver_running:
            continue
        if spec.name == "moveit_ik" and running.moveit_running:
            continue
        if spec.name == "distributed_stack" and running.distributed_running:
            continue
        if spec.name == "rviz" and running.rviz_running:
            continue
        launches_to_start.append(spec)
    return launches_to_start


def _spawn_process(spec: LaunchSpec, project_root: Path, session_root: Path) -> ManagedProcess:
    log_path = session_root / f"{spec.name}.log"
    log_handle = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        spec.argv,
        cwd=project_root,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return ManagedProcess(spec=spec, process=process, log_path=log_path, log_handle=log_handle)


def _check_children(processes: list[ManagedProcess]) -> None:
    for managed in processes:
        returncode = managed.process.poll()
        if returncode is not None:
            raise ChildExitError(managed.spec.name, returncode, managed.log_path)


def _run_command(
    argv: list[str],
    project_root: Path,
    timeout_s: float = 15.0,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        argv,
        cwd=project_root,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def _wait_for_compute_ik(
    project_root: Path,
    processes: list[ManagedProcess],
    service_name: str,
    timeout_s: float,
) -> float:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        _check_children(processes)
        result = _run_command(
            ["./scripts/ros2_system.sh", "service", "type", service_name],
            project_root,
        )
        if result.returncode == 0 and result.stdout.strip():
            return time.monotonic()
        time.sleep(1.0)
    raise RunnerError(
        f"Timed out waiting for {service_name} to appear within {timeout_s:.1f}s."
    )


def _wait_for_probe(
    project_root: Path,
    processes: list[ManagedProcess],
    probe_argv: list[str],
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_output = ""
    while time.monotonic() < deadline:
        _check_children(processes)
        result = _run_command(probe_argv, project_root)
        combined = (result.stdout or "") + (result.stderr or "")
        last_output = combined.strip()
        if result.returncode == 0 and "success=True" in combined:
            return
        time.sleep(2.0)
    detail = f" Last probe output: {last_output}" if last_output else ""
    raise RunnerError(
        f"Timed out waiting for /grasp_pipeline/probe readiness within {timeout_s:.1f}s.{detail}"
    )


def _sleep_with_child_checks(
    processes: list[ManagedProcess],
    duration_s: float,
) -> None:
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        _check_children(processes)
        time.sleep(min(0.5, deadline - time.monotonic()))


def _trigger_initial_run(project_root: Path, trigger_argv: list[str]) -> None:
    result = _run_command(trigger_argv, project_root, timeout_s=30.0)
    combined = (result.stdout or "") + (result.stderr or "")
    if result.returncode == 0 and "success=True" in combined:
        return
    raise RunnerError(
        "Initial grasp trigger failed."
        f" Command: {' '.join(trigger_argv)}"
        f" Output: {combined.strip()}"
    )


def _configure_pipeline_params(project_root: Path, configure_param_argvs: list[list[str]]) -> None:
    for argv in configure_param_argvs:
        result = _run_command(argv, project_root, timeout_s=10.0)
        combined = (result.stdout or "") + (result.stderr or "")
        if result.returncode != 0:
            raise RunnerError(
                "Failed to configure pipeline parameter before triggering run."
                f" Command: {' '.join(argv)}"
                f" Output: {combined.strip()}"
            )


def _artifact_root(project_root: Path) -> Path:
    return project_root.parent / "log" / "distributed_runs"


def _latest_run_dir(project_root: Path) -> Path | None:
    latest_file = _artifact_root(project_root) / "latest_run.txt"
    if not latest_file.is_file():
        return None
    text = latest_file.read_text(encoding="utf-8").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    return path if path.is_dir() else None


def _read_final_result(run_dir: Path) -> dict[str, object] | None:
    result_path = run_dir / "final_result.json"
    if not result_path.is_file():
        return None
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _wait_for_one_run_result(
    *,
    project_root: Path,
    processes: list[ManagedProcess],
    prompt: str,
    previous_run_dir: Path | None,
    started_at_s: float,
    timeout_s: float,
) -> tuple[Path, dict[str, object]]:
    terminal_statuses = {
        "ok",
        "completed",
        "no_candidate",
        "failed",
        "cancelled",
        "stopped",
    }
    deadline = time.monotonic() + max(1.0, float(timeout_s))
    last_seen_status = ""
    while time.monotonic() < deadline:
        _check_children(processes)
        run_dir = _latest_run_dir(project_root)
        if run_dir is not None and run_dir != previous_run_dir:
            payload = _read_final_result(run_dir)
            if payload is not None:
                result_prompt = str(payload.get("prompt", ""))
                status = str(payload.get("status", ""))
                last_seen_status = status or last_seen_status
                result_path = run_dir / "final_result.json"
                fresh_enough = result_path.stat().st_mtime >= started_at_s - 1.0
                prompt_matches = not result_prompt or result_prompt == prompt
                if fresh_enough and prompt_matches and status in terminal_statuses:
                    return run_dir, payload
        time.sleep(1.0)
    detail = f" Last seen status: {last_seen_status}" if last_seen_status else ""
    raise RunnerError(
        f"Timed out waiting for one-run result within {float(timeout_s):.1f}s.{detail}"
    )


def _terminate_processes(processes: list[ManagedProcess]) -> None:
    for managed in reversed(processes):
        if managed.process.poll() is None:
            managed.process.terminate()
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        live = [managed for managed in processes if managed.process.poll() is None]
        if not live:
            break
        time.sleep(0.2)
    for managed in reversed(processes):
        if managed.process.poll() is None:
            managed.process.kill()
    for managed in reversed(processes):
        try:
            managed.process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
        managed.log_handle.close()


def _print_running_component_summary(running: RunningComponents) -> None:
    if not running.details:
        return
    print("[one-click] reusing existing processes:")
    for line in running.details:
        print(f"  {line}")


def _print_launch_summary(session_root: Path, processes: list[ManagedProcess]) -> None:
    print(f"[one-click] wrapper logs: {session_root}")
    for managed in processes:
        print(
            f"[one-click] started {managed.spec.name:<18} "
            f"pid={managed.process.pid} log={managed.log_path}"
        )


def _print_ready_message(plan: LaunchPlan, started_launch_names: set[str]) -> None:
    print("[one-click] initial live grasp request accepted")
    print(f"[one-click] prompt: {plan.prompt}")
    print(f"[one-click] backend: {plan.robot_backend} execute={plan.execute}")
    if plan.once:
        print("[one-click] waiting for this single run to finish")
        return
    print("[one-click] stack stays up for more runs")
    print("[one-click] next commands:")
    print("  ./scripts/show_last_run_artifact.sh")
    print(f"  ./scripts/run_pipeline_service.sh {plan.prompt}")
    if started_launch_names:
        started = ", ".join(sorted(started_launch_names))
        print(f"Press Ctrl+C here to stop only the components started by this wrapper: {started}.")
    else:
        print("Press Ctrl+C here to exit the wrapper. Reused external processes will keep running.")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    plan = build_launch_plan(project_root, args)
    running = detect_running_components(project_root)

    try:
        launches_to_start = resolve_launches_for_running_components(plan, running)
    except RunnerError as exc:
        print(f"[one-click] {exc}", file=sys.stderr)
        for line in running.details:
            print(f"  {line}", file=sys.stderr)
        return 1

    if any(spec.name == "rviz" for spec in launches_to_start) and not os.environ.get("DISPLAY"):
        print(
            "DISPLAY is not set, so RViz cannot be opened. Use --no-rviz or start from a desktop session.",
            file=sys.stderr,
        )
        return 1

    session_root = _session_root(project_root)
    session_root.mkdir(parents=True, exist_ok=True)
    processes: list[ManagedProcess] = []
    stop_requested = False

    def _handle_signal(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.signal(signal.SIGINT, _handle_signal)
    previous_sigterm = signal.signal(signal.SIGTERM, _handle_signal)

    try:
        if running.details:
            _print_running_component_summary(running)

        for spec in launches_to_start:
            managed = _spawn_process(spec, project_root, session_root)
            processes.append(managed)
            _check_children(processes)

        if processes:
            _print_launch_summary(session_root, processes)
        else:
            print(f"[one-click] wrapper logs: {session_root}")
            print("[one-click] no new processes were started; using the existing live stack")

        compute_ik_ready_at: float | None = None
        if plan.compute_ik_service:
            print(f"[one-click] waiting for {plan.compute_ik_service}")
            compute_ik_ready_at = _wait_for_compute_ik(
                project_root,
                processes,
                plan.compute_ik_service,
                plan.probe_timeout_s,
            )
        print("[one-click] waiting for /grasp_pipeline/probe")
        _wait_for_probe(project_root, processes, plan.probe_argv, plan.probe_timeout_s)

        started_launch_names = {managed.spec.name for managed in processes}
        remaining_warmup = 0.0
        if "moveit_ik" in started_launch_names and compute_ik_ready_at is not None:
            remaining_warmup = max(
                0.0,
                plan.moveit_warmup_s - (time.monotonic() - compute_ik_ready_at),
            )
        if remaining_warmup > 0.0:
            print(f"[one-click] moveit warm-up: {remaining_warmup:.1f}s")
            _sleep_with_child_checks(processes, remaining_warmup)

        print("[one-click] configuring /grasp_pipeline runtime parameters")
        _configure_pipeline_params(project_root, plan.configure_param_argvs)
        print(f"[one-click] triggering first run with prompt={plan.prompt}")
        previous_run_dir = _latest_run_dir(project_root)
        triggered_at_s = time.time()
        _trigger_initial_run(project_root, plan.trigger_argv)
        _print_ready_message(plan, started_launch_names)
        if plan.once:
            run_dir, result_payload = _wait_for_one_run_result(
                project_root=project_root,
                processes=processes,
                prompt=plan.prompt,
                previous_run_dir=previous_run_dir,
                started_at_s=triggered_at_s,
                timeout_s=plan.result_timeout_s,
            )
            print(f"[one-click] result: status={result_payload.get('status')} run_dir={run_dir}")
            summary = str(result_payload.get("summary") or "").strip()
            if summary:
                print("[one-click] summary:")
                print(summary)
            return 0

        while not stop_requested:
            _check_children(processes)
            time.sleep(1.0)
        return 0
    except RunnerError as exc:
        print(f"[one-click] {exc}", file=sys.stderr)
        if session_root.exists():
            print(f"[one-click] wrapper logs: {session_root}", file=sys.stderr)
        return 1
    finally:
        _terminate_processes(processes)
        signal.signal(signal.SIGINT, previous_sigint)
        signal.signal(signal.SIGTERM, previous_sigterm)


if __name__ == "__main__":
    raise SystemExit(main())
