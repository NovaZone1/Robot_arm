from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time


@dataclass(frozen=True)
class CleanupTarget:
    name: str
    pattern: str


@dataclass(frozen=True)
class MatchedProcess:
    pid: int
    command: str


def build_cleanup_targets(project_root: Path) -> list[CleanupTarget]:
    rviz_config = project_root / "rviz" / "distributed_grasp_pipeline.rviz"
    return [
        CleanupTarget(
            "distributed_stack",
            r"robot_grasp_ros2\.(camera_server_node|vision_worker_node|robot_executor_node|pipeline_orchestrator_node|scout_scan_controller_node)",
        ),
        CleanupTarget("moveit_ik", r"move_group"),
        CleanupTarget("piper_driver", r"piper_single_ctrl"),
        CleanupTarget("distributed_rviz", rf"rviz2 .*{re.escape(str(rviz_config))}"),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stop the live Piper grasp driver, MoveIt IK, distributed nodes, and distributed RViz."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print matching processes without sending signals.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=8.0,
        help="Seconds to wait after SIGTERM before SIGKILL. Default: 8.",
    )
    parser.add_argument(
        "--no-rviz",
        action="store_true",
        help="Leave the distributed RViz process running.",
    )
    return parser


def _pgrep_lines(pattern: str) -> list[str]:
    result = subprocess.run(
        ["pgrep", "-af", pattern],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode not in (0, 1):
        return []
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def parse_pgrep_lines(
    targets: list[CleanupTarget],
    pgrep_lines: list[str],
    *,
    current_pid: int,
) -> dict[str, list[MatchedProcess]]:
    grouped: dict[str, list[MatchedProcess]] = {target.name: [] for target in targets}
    seen_pids: set[int] = set()

    for line in pgrep_lines:
        pid_text, sep, command = line.partition(" ")
        if not sep:
            continue
        try:
            pid = int(pid_text)
        except ValueError:
            continue
        if pid == current_pid or pid in seen_pids:
            continue
        for target in targets:
            if re.search(target.pattern, command):
                grouped[target.name].append(MatchedProcess(pid=pid, command=command))
                seen_pids.add(pid)
                break
    return grouped


def find_matching_processes(targets: list[CleanupTarget]) -> dict[str, list[MatchedProcess]]:
    all_lines: list[str] = []
    for target in targets:
        all_lines.extend(_pgrep_lines(target.pattern))
    return parse_pgrep_lines(targets, all_lines, current_pid=os.getpid())


def _all_processes(grouped: dict[str, list[MatchedProcess]]) -> list[MatchedProcess]:
    processes: list[MatchedProcess] = []
    seen_pids: set[int] = set()
    for matches in grouped.values():
        for process in matches:
            if process.pid in seen_pids:
                continue
            seen_pids.add(process.pid)
            processes.append(process)
    return processes


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def terminate_processes(processes: list[MatchedProcess], *, timeout_s: float) -> tuple[list[int], list[int]]:
    terminated: list[int] = []
    killed: list[int] = []

    for process in processes:
        if not _is_running(process.pid):
            continue
        try:
            os.kill(process.pid, signal.SIGTERM)
            terminated.append(process.pid)
        except ProcessLookupError:
            continue

    deadline = time.monotonic() + max(0.0, timeout_s)
    while time.monotonic() < deadline:
        if not any(_is_running(process.pid) for process in processes):
            return terminated, killed
        time.sleep(0.2)

    for process in processes:
        if not _is_running(process.pid):
            continue
        try:
            os.kill(process.pid, signal.SIGKILL)
            killed.append(process.pid)
        except ProcessLookupError:
            continue
    return terminated, killed


def _print_matches(grouped: dict[str, list[MatchedProcess]]) -> None:
    any_match = False
    for name, matches in grouped.items():
        if not matches:
            continue
        any_match = True
        print(f"[clear-nodes] {name}:")
        for process in matches:
            print(f"  {process.pid} {process.command}")
    if not any_match:
        print("[clear-nodes] no live grasp processes matched")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    project_root = Path(__file__).resolve().parents[1]
    targets = build_cleanup_targets(project_root)
    if args.no_rviz:
        targets = [target for target in targets if target.name != "distributed_rviz"]

    grouped = find_matching_processes(targets)
    _print_matches(grouped)
    processes = _all_processes(grouped)
    if args.dry_run or not processes:
        return 0

    terminated, killed = terminate_processes(processes, timeout_s=float(args.timeout))
    print(f"[clear-nodes] sent SIGTERM to: {terminated or 'none'}")
    if killed:
        print(f"[clear-nodes] sent SIGKILL to: {killed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
