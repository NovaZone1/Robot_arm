#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import quote, unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = PROJECT_ROOT.parent
WORKSPACE_ROOT = BUNDLE_ROOT / "ros_ws"
ARTIFACT_ROOT = BUNDLE_ROOT / "log" / "distributed_runs"
VIZ_ROOT = WORKSPACE_ROOT / "viz"
PLACEMENT_SCAN_ROOT = VIZ_ROOT / "placement_scan"
LOG_ROOT = WORKSPACE_ROOT / "log" / "distributed"
DASHBOARD_STACK_LOG_ROOT = WORKSPACE_ROOT / "log" / "dashboard_stack"
DASHBOARD_STACK_STATE_FILE = DASHBOARD_STACK_LOG_ROOT / "current.json"
LIVE_STACK_COMMAND = [
    "./scripts/run_distributed_stack_graspnet.sh",
    "--robot-backend",
    "ros2",
    "--pose-execution-mode",
    "moveit_ik",
    "--with-piper-driver",
    "--with-moveit-ik",
    "--warmup",
]
CAN_PORT = str(os.environ.get("PIPER_CAN_PORT", "can0")).strip() or "can0"
CAN_BITRATE = "1000000"


def _read_json(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _safe_relative_path(root: Path, relative_text: str) -> Path | None:
    try:
        candidate = (root / unquote(relative_text)).resolve()
        candidate.relative_to(root.resolve())
    except Exception:
        return None
    return candidate if candidate.is_file() else None


def _run_dirs() -> list[Path]:
    if not ARTIFACT_ROOT.is_dir():
        return []
    return sorted(
        (path for path in ARTIFACT_ROOT.iterdir() if path.is_dir() and path.name.startswith("grasp-")),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def _scene_viz_files(scene_id: str) -> dict[str, str]:
    if not scene_id:
        return {}
    scene_dir = VIZ_ROOT / scene_id
    if not scene_dir.is_dir():
        return {}
    out: dict[str, str] = {}
    for filename in ("segmentation_overlay.png", "grasp_projection.png", "summary.txt"):
        path = scene_dir / filename
        if path.is_file():
            out[filename] = f"/viz/{quote(scene_id)}/{quote(filename)}"
    return out


def _latest_log_session() -> Path | None:
    if not LOG_ROOT.is_dir():
        return None
    sessions = sorted((p for p in LOG_ROOT.iterdir() if p.is_dir()), key=lambda p: p.stat().st_mtime, reverse=True)
    return sessions[0] if sessions else None


def _tail_text(path: Path, *, lines: int = 80) -> str:
    if not path.is_file():
        return ""
    try:
        data = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    return "\n".join(data[-max(1, int(lines)):])


def _process_lines() -> list[str]:
    pattern = (
        r"run_one_grasp_task|live_grasp_one_click|run_distributed_stack|"
        r"robot_grasp_ros2\.(camera_server_node|vision_worker_node|robot_executor_node|pipeline_orchestrator_node|scout_scan_controller_node)|"
        r"move_group|piper_single_ctrl"
    )
    result = subprocess.run(["pgrep", "-af", pattern], capture_output=True, text=True, check=False)
    if result.returncode not in (0, 1):
        return []
    return [line for line in result.stdout.splitlines() if "pgrep -af" not in line]


def _component_status() -> dict[str, object]:
    lines = _process_lines()
    expected_nodes = {
        "camera": "robot_grasp_ros2.camera_server_node",
        "vision": "robot_grasp_ros2.vision_worker_node",
        "executor": "robot_grasp_ros2.robot_executor_node",
        "pipeline": "robot_grasp_ros2.pipeline_orchestrator_node",
        "base_scan": "robot_grasp_ros2.scout_scan_controller_node",
        "moveit": "move_group",
        "driver": "piper_single_ctrl",
    }
    components = {
        name: any(pattern in line for line in lines)
        for name, pattern in expected_nodes.items()
    }
    # This dashboard is a real-robot entry point. Do not enable grasp controls
    # unless the motion planner and Piper driver are online as well.
    stack_ready = all(components.values())
    return {
        "components": components,
        "stack_ready": stack_ready,
        "process_count": len(lines),
        "processes": lines,
    }


def _run_command(args: list[str], *, timeout_s: float = 10.0) -> dict[str, object]:
    result = subprocess.run(
        args,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_s,
    )
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": " ".join(args),
    }


def _can_interface_is_up(can_port: str = CAN_PORT) -> bool:
    result = subprocess.run(["ip", "link", "show", can_port], capture_output=True, text=True, check=False)
    return result.returncode == 0 and "UP" in result.stdout.splitlines()[0]


def _ensure_can_interface_up(can_port: str = CAN_PORT, bitrate: str = CAN_BITRATE) -> dict[str, object]:
    check = _run_command(["ip", "link", "show", can_port], timeout_s=3.0)
    if not check["ok"]:
        return {
            "ok": False,
            "stage": "can",
            "error": f"CAN 口 {can_port} 不存在，请检查 USB-CAN 是否插好",
            "commands": [check],
        }
    if _can_interface_is_up(can_port):
        return {
            "ok": True,
            "stage": "can",
            "message": f"{can_port} 已经 UP",
            "commands": [check],
        }

    commands = [
        ["sudo", "-n", "ip", "link", "set", can_port, "down"],
        ["sudo", "-n", "ip", "link", "set", can_port, "type", "can", "bitrate", str(bitrate)],
        ["sudo", "-n", "ip", "link", "set", can_port, "up"],
    ]
    results: list[dict[str, object]] = []
    for command in commands:
        item = _run_command(command, timeout_s=8.0)
        results.append(item)
        if not item["ok"]:
            detail = str(item.get("stderr") or item.get("stdout") or "").strip()
            return {
                "ok": False,
                "stage": "can",
                "error": (
                    f"自动打开 {can_port} 失败。需要给当前用户配置免密 sudo，"
                    f"或手动执行：sudo ip link set {can_port} type can bitrate {bitrate}; "
                    f"sudo ip link set {can_port} up"
                ),
                "detail": detail,
                "commands": results,
            }
    verify = _run_command(["ip", "link", "show", can_port], timeout_s=3.0)
    results.append(verify)
    if not _can_interface_is_up(can_port):
        return {
            "ok": False,
            "stage": "can",
            "error": f"{can_port} 自动配置后仍未 UP",
            "commands": results,
        }
    return {
        "ok": True,
        "stage": "can",
        "message": f"{can_port} 已自动配置为 bitrate {bitrate} 并 UP",
        "commands": results,
    }


def _dashboard_stack_process() -> tuple[int, dict[str, object]] | None:
    state = _read_json(DASHBOARD_STACK_STATE_FILE)
    try:
        pid = int(state.get("pid") or 0)
    except (TypeError, ValueError):
        return None
    if pid <= 1:
        return None
    try:
        os.kill(pid, 0)
        cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8",
            errors="replace",
        )
    except (OSError, PermissionError):
        return None
    if "run_distributed_stack_graspnet.sh" not in cmdline:
        return None
    return pid, state


def _start_live_stack() -> dict[str, object]:
    active_stack = _dashboard_stack_process()
    if active_stack is not None:
        pid, state = active_stack
        return {
            "ok": True,
            "stage": "starting_stack",
            "message": "真机栈已在启动或运行中，请等待状态变绿，不要重复启动",
            "pid": pid,
            "log": str(state.get("log") or ""),
            "system": _component_status(),
        }

    status = _component_status()
    components = status.get("components", {})
    if bool(status.get("stack_ready")) and bool(components.get("driver")) and bool(components.get("moveit")):
        return {
            "ok": True,
            "message": "真机栈已经在运行，可以直接发起抓取任务",
            "system": status,
        }
    if any(bool(components.get(name)) for name in ("camera", "vision", "executor", "pipeline")):
        return {
            "ok": False,
            "stage": "partial_stack",
            "error": "检测到分布式节点残留，请先点击“停止真机栈”清理后再启动",
            "system": status,
        }

    can_result = _ensure_can_interface_up()
    if not can_result.get("ok"):
        return {
            "ok": False,
            "stage": "can",
            "error": str(can_result.get("error") or "CAN 口自动配置失败"),
            "can": can_result,
            "system": status,
        }

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    session_dir = DASHBOARD_STACK_LOG_ROOT / timestamp
    session_dir.mkdir(parents=True, exist_ok=True)
    log_path = session_dir / "live_stack.log"
    log_file = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            LIVE_STACK_COMMAND,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    finally:
        log_file.close()

    state = {
        "pid": process.pid,
        "command": LIVE_STACK_COMMAND,
        "log": str(log_path),
        "started_at": timestamp,
        "can": can_result,
    }
    DASHBOARD_STACK_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    DASHBOARD_STACK_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "ok": True,
        "stage": "starting_stack",
        "message": "真机栈启动中：等 pipeline/camera/vision/executor 都变绿后再抓取",
        "pid": process.pid,
        "command": " ".join(LIVE_STACK_COMMAND),
        "log": str(log_path),
        "can": can_result,
        "system": status,
    }


def _stop_live_stack() -> dict[str, object]:
    command = ["./scripts/clear_live_grasp_nodes.sh", "--timeout", "8"]
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=20.0,
    )
    if DASHBOARD_STACK_STATE_FILE.is_file():
        try:
            state = _read_json(DASHBOARD_STACK_STATE_FILE)
            state["stopped_at"] = time.strftime("%Y%m%d_%H%M%S")
            state["stop_returncode"] = result.returncode
            DASHBOARD_STACK_STATE_FILE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    return {
        "ok": result.returncode == 0,
        "stage": "stop_stack",
        "message": "真机栈停止完成" if result.returncode == 0 else "真机栈停止脚本返回失败",
        "command": " ".join(command),
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "system": _component_status(),
    }


def _run_ros2_command(args: list[str], *, timeout_s: float = 15.0) -> dict[str, object]:
    result = subprocess.run(
        [str(PROJECT_ROOT / "scripts" / "ros2_system.sh"), *args],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout_s,
    )
    return {
        "ok": result.returncode == 0,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": " ".join([str(PROJECT_ROOT / "scripts" / "ros2_system.sh"), *args]),
    }


def _param_set(
    node_name: str,
    param_name: str,
    value: object,
    *,
    timeout_s: float = 10.0,
) -> dict[str, object]:
    if isinstance(value, bool):
        value_text = "true" if value else "false"
    elif isinstance(value, str) and not value:
        # ros2 param set parses its final argument as YAML. Passing a bare
        # empty argv is inconsistent across ros2cli versions.
        value_text = "''"
    else:
        value_text = str(value)
    result = _run_ros2_command(
        ["param", "set", node_name, param_name, value_text],
        timeout_s=timeout_s,
    )
    output = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
    if "Setting parameter failed" in output:
        result["ok"] = False
    return result


def _param_get(
    node_name: str,
    param_name: str,
    *,
    timeout_s: float = 10.0,
) -> dict[str, object]:
    return _run_ros2_command(
        ["param", "get", node_name, param_name],
        timeout_s=timeout_s,
    )


def _set_boolean_param_verified(
    node_name: str,
    param_name: str,
    value: bool,
) -> dict[str, object]:
    """Set a safety-related boolean and confirm the target node received it."""
    set_result = _param_set(node_name, param_name, value)
    if not set_result.get("ok"):
        return set_result

    get_result = _param_get(node_name, param_name)
    output = f"{get_result.get('stdout', '')}\n{get_result.get('stderr', '')}"
    expected = "true" if value else "false"
    match = re.search(
        r"(?:Boolean value is:|value:)\s*(true|false)\b",
        output,
        flags=re.IGNORECASE,
    )
    actual = match.group(1).lower() if match else ""
    verified = bool(get_result.get("ok")) and actual == expected
    return {
        "ok": verified,
        "expected": expected,
        "actual": actual or "unreadable",
        "set": set_result,
        "get": get_result,
    }


def _trigger_service(service_name: str, *, timeout_s: float = 20.0) -> dict[str, object]:
    return _run_ros2_command(
        ["service", "call", service_name, "std_srvs/srv/Trigger", "{}"],
        timeout_s=timeout_s,
    )


def _placement_scan_payload() -> dict[str, object]:
    payload = _read_json(PLACEMENT_SCAN_ROOT / "latest.json")
    if not payload:
        return {"available": False}
    payload["available"] = True
    payload["viz"] = {
        name: f"/viz/placement_scan/{filename}"
        for name, filename in dict(payload.get("images") or {}).items()
    }
    views = []
    for raw_view in list(payload.get("views") or []):
        view = dict(raw_view)
        view["viz"] = {
            name: f"/viz/placement_scan/{filename}"
            for name, filename in dict(view.get("images") or {}).items()
        }
        views.append(view)
    if views:
        payload["views"] = views
    return payload


def _scan_placement_from_payload(payload: dict[str, object]) -> dict[str, object]:
    status = _component_status()
    if not bool(status.get("stack_ready")):
        return {
            "ok": False,
            "error": "真机栈未就绪，不能扫描放置区",
            "system": status,
        }
    _param_set("/grasp_pipeline", "use_cached_multiview_box_map", False)
    target_item_id = str(payload.get("target_item_id") or "").strip()
    if target_item_id:
        set_result = _param_set("/grasp_pipeline", "target_item_id", target_item_id)
        if not set_result.get("ok"):
            return {
                "ok": False,
                "error": "目标物品参数下发失败",
                "result": set_result,
            }
    latest_path = PLACEMENT_SCAN_ROOT / "latest.json"
    previous_mtime = latest_path.stat().st_mtime if latest_path.is_file() else 0.0
    trigger = _trigger_service("/grasp_pipeline/scan_placement", timeout_s=60.0)
    stdout = str(trigger.get("stdout") or "")
    service_success = bool(
        re.search(r"success\s*=\s*True|success:\s*true", stdout, flags=re.IGNORECASE)
    )
    scan = _placement_scan_payload()
    current_mtime = latest_path.stat().st_mtime if latest_path.is_file() else 0.0
    fresh = current_mtime > previous_mtime
    if not trigger.get("ok") or not service_success or not fresh:
        return {
            "ok": False,
            "error": "放置区扫描失败",
            "trigger": trigger,
            "scan": scan,
        }
    validation_ok = bool(scan.get("success"))
    detected_count = int(
        dict(scan.get("label_match") or {}).get("detected_label_count") or 0
    )
    return {
        "ok": True,
        "validation_ok": validation_ok,
        "message": (
            "放置区扫描及六盒标校验完成；未发送运动或夹爪命令"
            if validation_ok
            else (
                f"扫描图像已保存，但盒标校验未通过（{detected_count}/6）；"
                "请根据检测框调整相机或盒标，禁止执行放置"
            )
        ),
        "trigger": trigger,
        "scan": scan,
    }


def _scan_placement_multi_view_from_payload(
    payload: dict[str, object],
) -> dict[str, object]:
    if not bool(payload.get("base_motion_ack")):
        return {
            "ok": False,
            "error": "未确认底盘前方扫描通道与硬件急停，拒绝单向扫描",
        }
    status = _component_status()
    if not bool(status.get("stack_ready")):
        return {
            "ok": False,
            "error": "真机栈未就绪，不能执行底盘单向扫描",
            "system": status,
        }
    target_item_id = str(payload.get("target_item_id") or "").strip()
    if target_item_id:
        set_result = _param_set(
            "/grasp_pipeline",
            "target_item_id",
            target_item_id,
        )
        if not set_result.get("ok"):
            return {
                "ok": False,
                "error": "目标物品参数下发失败",
                "result": set_result,
            }
    latest_path = PLACEMENT_SCAN_ROOT / "latest.json"
    previous_mtime = latest_path.stat().st_mtime if latest_path.is_file() else 0.0
    enabled = _set_boolean_param_verified(
        "/grasp_pipeline",
        "base_multiview_enabled",
        True,
    )
    if not enabled.get("ok"):
        return {
            "ok": False,
            "error": "无法启用一次性底盘单向扫描",
            "result": enabled,
        }
    try:
        trigger = _trigger_service(
            "/grasp_pipeline/scan_placement_multi_view",
            timeout_s=120.0,
        )
    finally:
        _set_boolean_param_verified(
            "/grasp_pipeline",
            "base_multiview_enabled",
            False,
        )
    stdout = str(trigger.get("stdout") or "")
    service_success = bool(
        re.search(r"success\s*=\s*True|success:\s*true", stdout, flags=re.IGNORECASE)
    )
    scan = _placement_scan_payload()
    current_mtime = latest_path.stat().st_mtime if latest_path.is_file() else 0.0
    fresh = current_mtime > previous_mtime
    if not trigger.get("ok") or not service_success or not fresh:
        _param_set("/grasp_pipeline", "use_cached_multiview_box_map", False)
        return {
            "ok": False,
            "error": "底盘单向扫描失败；已发送停车并禁用缓存盒位",
            "trigger": trigger,
            "scan": scan,
        }
    validation_ok = bool(scan.get("success"))
    # The scan currently verifies label-to-fixed-slot order only. It must not
    # unlock placement until base alignment and the fixed arm release pose have
    # both been calibrated.
    _param_set("/grasp_pipeline", "use_cached_multiview_box_map", False)
    detected_count = int(
        dict(scan.get("label_match") or {}).get("detected_label_count") or 0
    )
    return {
        "ok": True,
        "validation_ok": validation_ok,
        "message": (
            "六个盒标顺序校验通过；底盘停在扫描终点，尚未解锁自动放置"
            if validation_ok
            else (
                f"单向扫描结束，但顺序校验未通过（{detected_count}/6）；"
                "禁止放置"
            )
        ),
        "trigger": trigger,
        "scan": scan,
    }


def _scan_and_align_placement_target_from_payload(
    payload: dict[str, object],
) -> dict[str, object]:
    if not bool(payload.get("base_motion_ack")):
        return {
            "ok": False,
            "error": "未确认底盘前方扫描通道与硬件急停，拒绝扫描并对准目标盒",
        }
    target_item_id = str(payload.get("target_item_id") or "").strip()
    if not target_item_id:
        return {"ok": False, "error": "请先选择要放置的指定物品"}
    status = _component_status()
    if not bool(status.get("stack_ready")):
        return {
            "ok": False,
            "error": "真机栈未就绪，不能扫描并对准目标盒",
            "system": status,
        }
    target_result = _param_set(
        "/grasp_pipeline", "target_item_id", target_item_id
    )
    if not target_result.get("ok"):
        return {"ok": False, "error": "目标物品参数下发失败", "result": target_result}
    latest_path = PLACEMENT_SCAN_ROOT / "latest.json"
    previous_mtime = latest_path.stat().st_mtime if latest_path.is_file() else 0.0
    enabled = _set_boolean_param_verified(
        "/grasp_pipeline", "base_target_alignment_enabled", True
    )
    if not enabled.get("ok"):
        return {"ok": False, "error": "无法启用目标盒单向扫描", "result": enabled}
    try:
        trigger = _trigger_service(
            "/grasp_pipeline/scan_and_align_placement_target", timeout_s=120.0
        )
    finally:
        _set_boolean_param_verified(
            "/grasp_pipeline", "base_target_alignment_enabled", False
        )
    scan = _placement_scan_payload()
    current_mtime = latest_path.stat().st_mtime if latest_path.is_file() else 0.0
    stdout = str(trigger.get("stdout") or "")
    service_success = bool(
        re.search(r"success\s*=\s*True|success:\s*true", stdout, flags=re.IGNORECASE)
    )
    if not trigger.get("ok") or not service_success or current_mtime <= previous_mtime:
        return {
            "ok": False,
            "error": "目标盒单向扫描未完成；底盘已停车",
            "trigger": trigger,
            "scan": scan,
        }
    return {
        "ok": True,
        "validation_ok": True,
        "message": "已找到并对准目标盒；底盘已停车，可执行标定放置",
        "trigger": trigger,
        "scan": scan,
    }


def _align_placement_target_from_payload(
    payload: dict[str, object],
) -> dict[str, object]:
    if not bool(payload.get("base_motion_ack")):
        return {
            "ok": False,
            "error": "未确认底盘返回路径与硬件急停，拒绝目标盒对准",
        }
    target_item_id = str(payload.get("target_item_id") or "").strip()
    if not target_item_id:
        return {"ok": False, "error": "请先选择要放置的指定物品"}
    status = _component_status()
    if not bool(status.get("stack_ready")):
        return {
            "ok": False,
            "error": "真机栈未就绪，不能执行目标盒对准",
            "system": status,
        }
    set_target = _param_set(
        "/grasp_pipeline", "target_item_id", target_item_id
    )
    if not set_target.get("ok"):
        return {
            "ok": False,
            "error": "目标物品参数下发失败",
            "result": set_target,
        }
    enabled = _param_set(
        "/grasp_pipeline", "base_alignment_enabled", True
    )
    if not enabled.get("ok"):
        return {
            "ok": False,
            "error": "无法启用一次性目标盒对准",
            "result": enabled,
        }
    try:
        trigger = _trigger_service(
            "/grasp_pipeline/align_placement_target",
            timeout_s=90.0,
        )
    finally:
        _param_set("/grasp_pipeline", "base_alignment_enabled", False)
    stdout = str(trigger.get("stdout") or "")
    service_success = bool(
        re.search(
            r"success\s*=\s*True|success:\s*true",
            stdout,
            flags=re.IGNORECASE,
        )
    )
    if not trigger.get("ok") or not service_success:
        return {
            "ok": False,
            "error": "目标盒对准失败；已发送停车",
            "trigger": trigger,
            "scan": _placement_scan_payload(),
        }
    return {
        "ok": True,
        "message": "底盘已对准目标标签；机械臂与夹爪未动作，请确认后点击“执行标定放置”",
        "trigger": trigger,
        "scan": _placement_scan_payload(),
    }


def _execute_aligned_place_from_payload(
    payload: dict[str, object],
) -> dict[str, object]:
    if not bool(payload.get("release_ack")):
        return {
            "ok": False,
            "error": "未确认夹爪将释放物体，拒绝执行标定放置",
        }
    target_item_id = str(payload.get("target_item_id") or "").strip()
    if not target_item_id:
        return {"ok": False, "error": "请先选择要放置的指定物品"}
    status = _component_status()
    if not bool(status.get("stack_ready")):
        return {
            "ok": False,
            "error": "真机栈未就绪，不能执行标定放置",
            "system": status,
        }
    set_target = _param_set(
        "/grasp_pipeline", "target_item_id", target_item_id
    )
    if not set_target.get("ok"):
        return {
            "ok": False,
            "error": "目标物品参数下发失败",
            "result": set_target,
        }
    enabled = _param_set(
        "/grasp_pipeline", "base_aligned_place_enabled", True
    )
    if not enabled.get("ok"):
        return {
            "ok": False,
            "error": "无法启用一次性标定放置",
            "result": enabled,
        }
    try:
        trigger = _trigger_service(
            "/grasp_pipeline/execute_aligned_place",
            timeout_s=210.0,
        )
    finally:
        _param_set(
            "/grasp_pipeline", "base_aligned_place_enabled", False
        )
    stdout = str(trigger.get("stdout") or "")
    service_success = bool(
        re.search(
            r"success\s*=\s*True|success:\s*true",
            stdout,
            flags=re.IGNORECASE,
        )
    )
    if not trigger.get("ok") or not service_success:
        return {
            "ok": False,
            "error": "标定放置失败；请检查执行器状态，物体可能仍在夹爪中",
            "trigger": trigger,
            "scan": _placement_scan_payload(),
        }
    return {
        "ok": True,
        "message": "标定放置完成；机械臂已撤离到盒口上方安全位",
        "trigger": trigger,
        "scan": _placement_scan_payload(),
    }


def _bool_text(value: object) -> str:
    return "true" if bool(value) else "false"


def _start_grasp_from_payload(payload: dict[str, object]) -> dict[str, object]:
    auto_target_from_card = bool(payload.get("auto_target_from_card", True))
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt and not auto_target_from_card:
        return {"ok": False, "error": "Prompt 不能为空"}
    if (
        bool(payload.get("place_after_grasp"))
        and not auto_target_from_card
        and not str(payload.get("target_item_id") or "").strip()
    ):
        return {"ok": False, "error": "自动放置必须先选择六类指定物品之一"}
    base_grasp_scan = auto_target_from_card or bool(payload.get("base_grasp_scan"))
    if base_grasp_scan:
        if not auto_target_from_card and not str(payload.get("target_item_id") or "").strip():
            return {"ok": False, "error": "抓取前底盘扫描必须选择指定物品"}
        if not bool(payload.get("base_motion_ack")):
            return {
                "ok": False,
                "error": "请先确认底盘前方扫描通道无障碍，并准备硬件急停",
            }
    status = _component_status()
    components = status.get("components", {})
    if not bool(status.get("stack_ready")):
        return {
            "ok": False,
            "stage": "stack_offline",
            "error": "真机抓取栈未就绪：请确认 pipeline / camera / vision / executor / MoveIt / driver 全部在线",
            "system": status,
        }

    speed = float(payload.get("speed") or 5.0)
    speed = max(1.0, min(100.0, speed))
    pipeline_speed_text = str(int(round(speed)))
    executor_speed_text = str(speed)
    requested_strategy = str(payload.get("execution_strategy") or "").strip()
    if requested_strategy not in {"center_horizontal", "safe_top_down"}:
        requested_strategy = (
            "center_horizontal" if payload.get("use_object_center_contact", True) else "safe_top_down"
        )
    commands = [
        _param_set("/grasp_pipeline", "auto_target_from_card", _bool_text(auto_target_from_card)),
        _param_set("/grasp_pipeline", "prompt", "" if auto_target_from_card else prompt),
        _param_set(
            "/grasp_pipeline",
            "target_item_id",
            "" if auto_target_from_card else str(payload.get("target_item_id") or ""),
        ),
        _param_set("/grasp_pipeline", "execute", _bool_text(payload.get("execute", True))),
        _param_set(
            "/grasp_pipeline",
            "place_after_grasp",
            _bool_text(payload.get("place_after_grasp", False)),
        ),
        _param_set(
            "/grasp_pipeline",
            "dynamic_box_localization",
            _bool_text(payload.get("place_after_grasp", False)),
        ),
        _param_set(
            "/grasp_pipeline",
            "base_grasp_scan_enabled",
            _bool_text(base_grasp_scan),
        ),
        _param_set("/grasp_pipeline", "confirm", _bool_text(payload.get("confirm", True))),
        _param_set("/grasp_pipeline", "move_home_after", _bool_text(payload.get("move_home_after", False))),
        _param_set("/grasp_pipeline", "precenter", _bool_text(payload.get("precenter", False))),
        _param_set("/grasp_pipeline", "show_pointcloud", _bool_text(payload.get("show_pointcloud", False))),
        _param_set("/grasp_pipeline", "enable_pregrasp", _bool_text(payload.get("enable_pregrasp", False))),
        _param_set(
            "/grasp_pipeline",
            "use_object_center_contact",
            _bool_text(payload.get("use_object_center_contact", True)),
        ),
        _param_set("/grasp_pipeline", "speed", pipeline_speed_text),
        _param_set("/grasp_pipeline", "manual_target_bias_x_mm", str(float(payload.get("bias_x_mm") or 0.0))),
        _param_set("/grasp_pipeline", "manual_target_bias_y_mm", str(float(payload.get("bias_y_mm") or 0.0))),
        _param_set("/grasp_pipeline", "manual_target_bias_z_mm", str(float(payload.get("bias_z_mm") or 0.0))),
        _param_set("/robot_executor", "default_speed_percent", executor_speed_text),
        _param_set("/robot_executor", "enable_pregrasp", _bool_text(payload.get("enable_pregrasp", False))),
    ]
    if not auto_target_from_card:
        commands.append(
            _param_set(
                "/robot_executor",
                "execution_strategy",
                requested_strategy,
            )
        )
    failed = [item for item in commands if not item.get("ok")]
    if failed:
        node_missing = any("Node not found" in str(item.get("stderr") or item.get("stdout") or "") for item in failed)
        return {
            "ok": False,
            "stage": "param_set",
            "error": "抓取栈未启动：找不到 ROS 节点" if node_missing else "参数下发失败",
            "commands": commands,
            "failed_commands": failed,
            "system": status,
        }

    trigger = _trigger_service("/grasp_pipeline/run", timeout_s=30.0)
    if not trigger.get("ok"):
        return {
            "ok": False,
            "stage": "run",
            "error": "抓取任务触发失败",
            "commands": commands,
            "trigger": trigger,
        }
    return {"ok": True, "stage": "run", "commands": commands, "trigger": trigger}


def _run_payload(run_dir: Path) -> dict:
    final_result = _read_json(run_dir / "final_result.json")
    request = _read_json(run_dir / "request.json")
    cycles = _read_json(run_dir / "cycles.json")
    candidate_validation = _read_json(run_dir / "candidate_validation.json").get("candidate_validation", [])
    if not isinstance(candidate_validation, list):
        candidate_validation = []

    scene_id = str(final_result.get("scene_id") or "")
    if not scene_id:
        vision = final_result.get("vision")
        if isinstance(vision, dict):
            perception = vision.get("perception")
            if isinstance(perception, dict):
                scene_id = str(perception.get("scene_id") or "")

    latest_log = _latest_log_session()
    logs = {}
    if latest_log is not None:
        for name in ("grasp_pipeline.log", "robot_executor.log", "vision_worker.log", "camera_server.log", "moveit_ik.log", "piper_driver.log"):
            text = _tail_text(latest_log / name)
            if text:
                logs[name] = text

    viz = _scene_viz_files(scene_id)
    target_card_dir = run_dir / "target_card"
    for filename, key in (
        ("color.png", "target_card_color.png"),
        ("overlay.png", "target_card_overlay.png"),
    ):
        if (target_card_dir / filename).is_file():
            viz[key] = (
                f"/run-artifact/{quote(run_dir.name)}/target_card/{quote(filename)}"
            )

    return {
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "final_result": final_result,
        "request": request,
        "cycles": cycles,
        "candidate_validation": candidate_validation,
        "viz": viz,
        "logs": logs,
        "processes": _process_lines(),
    }


def _runs_payload() -> dict:
    runs = []
    for run_dir in _run_dirs()[:50]:
        final_result = _read_json(run_dir / "final_result.json")
        runs.append(
            {
                "run_id": run_dir.name,
                "path": str(run_dir),
                "prompt": final_result.get("prompt"),
                "status": final_result.get("status"),
                "summary": final_result.get("summary"),
                "mtime": run_dir.stat().st_mtime,
            }
        )
    status = _component_status()
    return {"runs": runs, "processes": status["processes"], "system": status}


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>抓取操作台</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d8dde6;
      --text: #16202a;
      --muted: #667085;
      --ok: #14804a;
      --bad: #c2342d;
      --warn: #a15c00;
      --blue: #175cd3;
      --blue-dark: #0f3f99;
      --green: #116149;
      --soft: #fbfcfe;
      --ink: #111827;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: var(--bg);
    }
    header {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      padding: 16px 20px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    h1 { font-size: 20px; margin: 0; letter-spacing: 0; }
    input[type="text"], input[type="number"] {
      height: 34px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 6px;
      padding: 0 10px;
      font: inherit;
    }
    input[type="range"] { accent-color: var(--green); }
    main { padding: 16px 20px 28px; max-width: 1680px; margin: 0 auto; }
    select, button {
      height: 34px;
      border: 1px solid var(--line);
      background: #fff;
      color: var(--text);
      border-radius: 6px;
      padding: 0 10px;
      font: inherit;
    }
    button { cursor: pointer; }
    button:disabled { cursor: not-allowed; opacity: 0.55; }
    button.primary { background: #175cd3; border-color: #175cd3; color: #fff; }
    button.primary:hover { background: #0f3f99; border-color: #0f3f99; }
    button.go { height: 44px; padding: 0 18px; font-weight: 800; background: #116149; border-color: #116149; color: #fff; }
    button.go:hover { background: #0d4a38; border-color: #0d4a38; }
    button.danger { background: #c2342d; border-color: #c2342d; color: #fff; }
    .toolbar { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }
    .toolbar-label { color: var(--muted); font-size: 13px; }
    .operator {
      padding: 18px 20px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }
    .operator-inner {
      max-width: 1680px;
      margin: 0 auto;
      display: grid;
      grid-template-columns: minmax(420px, 1fr) auto;
      gap: 18px;
      align-items: end;
    }
    .prompt-row { display: grid; grid-template-columns: minmax(220px, 1fr) minmax(220px, 1fr) minmax(220px, 320px); gap: 14px; align-items: end; }
    .operator label { display: grid; gap: 5px; color: var(--muted); font-size: 13px; }
    .operator input[type="text"], .operator input[type="number"] { height: 44px; font-size: 16px; }
    .speed-control { display: grid; grid-template-columns: 1fr 72px; gap: 8px; align-items: center; }
    .toggles { display: flex; align-items: center; gap: 12px; flex-wrap: wrap; margin-top: 10px; }
    .toggles label { display: inline-flex; align-items: center; gap: 6px; }
    .bias-row { display: flex; align-items: end; gap: 8px; flex-wrap: wrap; margin-top: 10px; }
    .bias-row label { width: 112px; }
    .bias-row input[type="number"] { height: 34px; font-size: 14px; }
    .quick-prompts { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 8px; }
    .quick-prompts button { height: 28px; font-size: 12px; color: var(--muted); background: #f8fafc; }
    .actions { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; justify-content: flex-end; }
    .system-strip { display: flex; align-items: center; gap: 8px; flex-wrap: wrap; margin-top: 12px; }
    .chip { display: inline-flex; align-items: center; height: 26px; padding: 0 8px; border-radius: 6px; font-size: 12px; border: 1px solid var(--line); background: #fff; color: var(--muted); }
    .chip.on { color: var(--ok); background: #eef9f2; border-color: #bee8cc; }
    .chip.off { color: var(--bad); background: #fff2f0; border-color: #ffd0cc; }
    .live-hint {
      display: none;
      grid-column: 1 / -1;
      border: 1px solid #f6d99b;
      border-radius: 8px;
      padding: 10px 12px;
      background: #fff8e8;
      color: #7a4100;
      font-size: 13px;
    }
    .live-hint.show { display: block; }
    #actionStatus {
      grid-column: 1 / -1;
      min-height: 38px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 9px 10px;
      font-size: 13px;
      color: var(--muted);
      background: #fbfcfe;
      overflow-wrap: anywhere;
    }
    #actionStatus.ok { color: var(--ok); background: #eef9f2; border-color: #bee8cc; }
    #actionStatus.bad { color: var(--bad); background: #fff2f0; border-color: #ffd0cc; }
    #actionStatus.warn { color: var(--warn); background: #fff8e8; border-color: #f6d99b; }
    .grid { display: grid; grid-template-columns: 340px 1fr; gap: 16px; }
    .stack { display: grid; gap: 16px; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px;
      min-width: 0;
    }
    .panel h2 {
      margin: 0 0 10px;
      font-size: 14px;
      font-weight: 700;
      color: var(--ink);
    }
    .kpis { display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; }
    .kpi { border: 1px solid var(--line); border-radius: 8px; padding: 10px; background: #fbfcfe; }
    .label { color: var(--muted); font-size: 12px; }
    .value { margin-top: 4px; font-size: 18px; font-weight: 700; word-break: break-word; }
    .pill { display: inline-flex; align-items: center; height: 24px; padding: 0 8px; border-radius: 999px; font-size: 12px; font-weight: 700; border: 1px solid var(--line); }
    .ok { color: var(--ok); background: #eaf7ef; border-color: #bee8cc; }
    .bad { color: var(--bad); background: #fff0ef; border-color: #ffd0cc; }
    .warn { color: var(--warn); background: #fff7e8; border-color: #f6d99b; }
    .neutral { color: var(--blue); background: #eef4ff; border-color: #c7d7fe; }
    .images { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    figure { margin: 0; }
    figure img { width: 100%; display: block; border-radius: 6px; border: 1px solid var(--line); background: #f0f2f5; }
    figcaption { color: var(--muted); font-size: 12px; margin-top: 6px; }
    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th, td { border-bottom: 1px solid var(--line); padding: 8px 7px; text-align: left; vertical-align: top; }
    th { color: var(--muted); font-size: 12px; font-weight: 700; background: #fbfcfe; position: sticky; top: 67px; }
    tr.selected { background: #eef7ff; }
    .scroll { max-height: 440px; overflow: auto; border: 1px solid var(--line); border-radius: 8px; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 12px; white-space: pre-wrap; overflow-wrap: anywhere; }
    .two { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 16px; }
    .plot { width: 100%; height: 320px; border: 1px solid var(--line); border-radius: 8px; background: #fff; }
    .paths { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .metric { display: grid; grid-template-columns: 1fr auto; gap: 8px; padding: 6px 0; border-bottom: 1px solid #eef1f5; }
    .metric:last-child { border-bottom: 0; }
    .small { font-size: 12px; color: var(--muted); }
    .process { padding: 6px 0; border-bottom: 1px solid #eef1f5; }
    .process:last-child { border-bottom: 0; }
    @media (max-width: 1100px) {
      .operator-inner, .prompt-row, .grid, .two, .images, .paths { grid-template-columns: 1fr; }
      .kpis { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      th { position: static; }
      .actions { justify-content: flex-start; }
    }
  </style>
</head>
<body>
<header>
  <h1>抓取操作台</h1>
  <div class="toolbar">
    <span class="toolbar-label">历史记录</span>
    <select id="runSelect"></select>
    <button id="refreshBtn">刷新</button>
  </div>
</header>
<section class="operator">
  <form id="runForm" class="operator-inner">
    <div>
      <div class="prompt-row">
        <label>指定物品
          <select id="targetItemInput">
            <option value="">通用 Prompt（不做标签配对）</option>
            <option value="yellow_block">黄色物块</option>
            <option value="red_block">红色物块</option>
            <option value="blue_block">蓝色物块</option>
            <option value="orange_bottle">橙色饮料瓶</option>
            <option value="dark_bottle">深色饮料瓶</option>
            <option value="green_bottle">绿色饮料瓶</option>
          </select>
        </label>
        <label>Prompt
          <input id="promptInput" type="text" value="bottle" autocomplete="off" />
        </label>
        <label>速度
          <div class="speed-control">
            <input id="speedRangeInput" type="range" min="1" max="100" step="1" value="5" />
            <input id="speedInput" type="number" min="1" max="100" step="1" value="5" />
          </div>
        </label>
      </div>
      <div class="quick-prompts">
        <button type="button" data-prompt="bottle" data-center-contact="true" data-execution-strategy="center_horizontal">bottle</button>
        <button type="button" data-prompt="cup" data-center-contact="true" data-execution-strategy="center_horizontal">cup</button>
        <button type="button" data-prompt="bowl" data-center-contact="true" data-execution-strategy="center_horizontal">bowl</button>
        <button type="button" data-item-id="red_block" data-prompt="red block" data-center-contact="true" data-execution-strategy="safe_top_down">红色物块</button>
        <button type="button" data-item-id="yellow_block" data-prompt="yellow block" data-center-contact="true" data-execution-strategy="safe_top_down">黄色物块</button>
        <button type="button" data-item-id="blue_block" data-prompt="blue block" data-center-contact="true" data-execution-strategy="safe_top_down">蓝色物块</button>
        <button type="button" data-item-id="orange_bottle" data-prompt="orange bottle" data-center-contact="true" data-execution-strategy="center_horizontal">橙色瓶</button>
        <button type="button" data-item-id="dark_bottle" data-prompt="dark bottle" data-center-contact="true" data-execution-strategy="center_horizontal">深色瓶</button>
        <button type="button" data-item-id="green_bottle" data-prompt="green bottle" data-center-contact="true" data-execution-strategy="center_horizontal">绿色瓶</button>
      </div>
      <div class="toggles">
        <label><input id="autoTargetCardInput" type="checkbox" checked /> 从标识牌照片自动识别抓取目标</label>
        <label><input id="centerContactInput" type="checkbox" checked /> 使用分割物体中心作为落点</label>
        <label>抓取方式
          <select id="executionStrategyInput">
            <option value="center_horizontal">侧向水平抓取（瓶/杯）</option>
            <option value="safe_top_down">顶部抓取（方块）</option>
          </select>
        </label>
        <label><input id="precenterInput" type="checkbox" /> 预居中</label>
        <label><input id="pregraspInput" type="checkbox" /> 预抓取点</label>
        <label><input id="showPointcloudInput" type="checkbox" /> 点云窗口</label>
        <label><input id="placeAfterGraspInput" type="checkbox" /> 标签对应固定槽位并放置（标定后）</label>
        <label><input id="baseGraspScanInput" type="checkbox" checked /> 识别目标后底盘单向扫描</label>
        <label><input id="baseMotionAckInput" type="checkbox" /> 已清空底盘前方累计 4.5m 通道，并准备硬件急停</label>
        <label><input id="moveHomeInput" type="checkbox" /> 完成后回 Home</label>
      </div>
      <div class="bias-row">
        <label>X补偿 mm
          <input id="biasXInput" type="number" step="0.5" value="0" />
        </label>
        <label>Y补偿 mm
          <input id="biasYInput" type="number" step="0.5" value="0" />
        </label>
        <label>Z补偿 mm
          <input id="biasZInput" type="number" step="0.5" value="0" />
        </label>
        <button id="clearBiasBtn" type="button">补偿清零</button>
        <button id="flipYBiasBtn" type="button">反转 Y</button>
      </div>
      <div id="systemStrip" class="system-strip"></div>
    </div>
    <div class="actions">
      <button id="startStackBtn" class="primary" type="button">启动真机栈</button>
      <button id="stopStackBtn" class="danger" type="button">停止真机栈</button>
      <button id="planConfirmBtn" class="primary" type="button">规划后确认</button>
      <button id="directGraspBtn" class="go" type="submit">直接抓取</button>
      <button id="confirmBtn" type="button">确认执行</button>
      <button id="rejectBtn" type="button">拒绝</button>
      <button id="stopBtn" type="button">停止任务</button>
      <button id="openGripperBtn" class="danger" type="button">释放夹爪</button>
      <button id="probeBtn" type="button">Probe</button>
      <button id="scanPlacementBtn" type="button">扫描放置区</button>
      <button id="scanPlacementMultiBtn" class="danger" type="button">扫描并对准目标盒</button>
      <button id="executeAlignedPlaceBtn" class="danger" type="button">执行标定放置</button>
    </div>
    <div id="liveHint" class="live-hint"></div>
    <span id="actionStatus"></span>
  </form>
</section>
<main>
  <div class="grid">
    <aside class="stack">
      <section class="panel">
        <h2>任务状态</h2>
        <div id="runMetrics"></div>
      </section>
      <section class="panel">
        <h2>运行进程</h2>
        <div id="processes" class="mono"></div>
      </section>
      <section class="panel">
        <h2>本次参数</h2>
        <div id="options" class="mono"></div>
      </section>
    </aside>
    <section class="stack">
      <div class="kpis" id="kpis"></div>
      <section class="panel">
        <h2>视觉结果</h2>
        <div class="images" id="images"></div>
      </section>
      <section class="panel">
        <h2>放置区扫描</h2>
        <div id="placementScan"><div class="small">尚未扫描</div></div>
      </section>
      <section class="panel">
        <h2>抓取路径</h2>
        <div class="paths">
          <svg id="topPlot" class="plot" viewBox="0 0 640 320"></svg>
          <svg id="sidePlot" class="plot" viewBox="0 0 640 320"></svg>
        </div>
      </section>
      <section class="panel">
        <h2>候选验证</h2>
        <div class="scroll"><table id="candidateTable"></table></div>
      </section>
      <section class="two">
        <div class="panel">
          <h2>流程数据</h2>
          <div id="cycles" class="mono"></div>
        </div>
        <div class="panel">
          <h2>日志</h2>
          <div id="logs" class="mono"></div>
        </div>
      </section>
    </section>
  </div>
</main>
<script>
const state = { runs: [], selected: "", systemReady: false };
const stackButtons = ["startStackBtn", "stopStackBtn"];
const runButtons = ["planConfirmBtn", "directGraspBtn", "confirmBtn", "rejectBtn", "stopBtn", "openGripperBtn", "probeBtn", "scanPlacementBtn", "scanPlacementMultiBtn", "executeAlignedPlaceBtn"];

function statusClass(status) {
  if (status === "ok" || status === "completed") return "ok";
  if (status === "failed") return "bad";
  if (status === "awaiting_confirmation") return "warn";
  if (status === "cancelled" || status === "stopped") return "neutral";
  return "neutral";
}

function fmt(v, digits=3) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return "-";
  return Number(v).toFixed(digits);
}

function posText(v) {
  if (!Array.isArray(v)) return "-";
  return v.map(x => fmt(x, 3)).join(", ");
}

function mmText(v) {
  if (!Array.isArray(v)) return "-";
  return v.map(x => fmt(Number(x) * 1000, 1)).join(", ");
}

function setText(id, text) {
  document.getElementById(id).textContent = text || "";
}

function setActionStatus(text, cls="") {
  const node = document.getElementById("actionStatus");
  node.textContent = text || "";
  node.className = cls;
}

function setActionBusy(busy) {
  for (const id of stackButtons) {
    const node = document.getElementById(id);
    if (node) node.disabled = Boolean(busy);
  }
  for (const id of runButtons) {
    const node = document.getElementById(id);
    if (node) node.disabled = Boolean(busy) || !state.systemReady;
  }
}

function friendlyError(data) {
  if (!data || typeof data !== "object") return "请求失败";
  if (data.message && !data.error) return data.message;
  if (data.error) {
    const failed = data.failed_commands || [];
    if (Array.isArray(failed) && failed.length) {
      const first = failed[0] || {};
      const detail = (first.stderr || first.stdout || first.command || "").trim();
      return `${data.error}${detail ? `：${detail}` : ""}`;
    }
    const trigger = data.trigger || {};
    const detail = (trigger.stderr || trigger.stdout || trigger.command || "").trim();
    return `${data.error}${detail ? `：${detail}` : ""}`;
  }
  if (data.stage === "param_set") return "参数下发失败";
  if (data.stage === "run") return "抓取任务触发失败";
  return JSON.stringify(data);
}

async function postJson(path, payload={}) {
  const res = await fetch(path, {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok || data.ok === false) {
    throw new Error(friendlyError(data));
  }
  return data;
}

function metric(label, value) {
  return `<div class="metric"><span class="label">${label}</span><span>${value}</span></div>`;
}

function kpi(label, value, cls="") {
  return `<div class="kpi"><div class="label">${label}</div><div class="value ${cls}">${value}</div></div>`;
}

async function loadRuns() {
  const res = await fetch("/api/runs");
  const data = await res.json();
  state.runs = data.runs || [];
  renderSystem(data.system || {});
  const select = document.getElementById("runSelect");
  const previous = state.selected || select.value;
  select.innerHTML = state.runs.map(r => `<option value="${encodeURIComponent(r.run_id)}">${r.run_id} | ${r.prompt || "-"} | ${r.status || "-"}</option>`).join("");
  if (state.runs.some(r => r.run_id === previous)) select.value = encodeURIComponent(previous);
  state.selected = decodeURIComponent(select.value || (state.runs[0] ? state.runs[0].run_id : ""));
}

function renderSystem(system) {
  const components = system.components || {};
  state.systemReady = Boolean(system.stack_ready);
  const labels = [
    ["pipeline", "pipeline"],
    ["camera", "camera"],
    ["vision", "vision"],
    ["executor", "executor"],
    ["moveit", "MoveIt"],
    ["driver", "driver"],
  ];
  document.getElementById("systemStrip").innerHTML = labels.map(([key, label]) => {
    const on = Boolean(components[key]);
    return `<span class="chip ${on ? "on" : "off"}">${label}: ${on ? "运行" : "未运行"}</span>`;
  }).join("");
  const hint = document.getElementById("liveHint");
  if (state.systemReady) {
    hint.className = "live-hint";
    hint.textContent = "";
  } else {
    hint.className = "live-hint show";
    hint.textContent = "抓取栈未启动。先点“启动真机栈”，等 pipeline/camera/vision/executor 变绿，再点“直接抓取”。";
  }
  setActionBusy(false);
}

async function loadRun() {
  if (!state.selected) return;
  const res = await fetch(`/api/run/${encodeURIComponent(state.selected)}`);
  const data = await res.json();
  render(data);
}

function render(data) {
  const final = data.final_result || {};
  const request = data.request || {};
  const cycles = data.cycles || {};
  const vision = final.vision || {};
  const perception = vision.perception || {};
  const selected = vision.selected_candidate || final.candidate || {};
  const plan = final.plan || {};
  const status = final.status || "-";
  const statusPill = `<span class="pill ${statusClass(status)}">${status}</span>`;

  document.getElementById("kpis").innerHTML = [
    kpi("状态", statusPill),
    kpi("Prompt", final.prompt || request.prompt || "-"),
    kpi("候选数", vision.candidate_count ?? (data.candidate_validation || []).length ?? "-"),
    kpi("选中分数", fmt(selected.score, 4)),
  ].join("");

  document.getElementById("runMetrics").innerHTML = [
    metric("Run ID", data.run_id || "-"),
    metric("确认", final.confirmed === null || final.confirmed === undefined ? "-" : String(final.confirmed)),
    metric("场景", final.scene_id || perception.scene_id || "-"),
    metric("实例数", perception.instance_count ?? "-"),
    metric("场景点数", perception.scene_point_count ?? "-"),
    metric("物体点数", Array.isArray(perception.object_point_counts) ? perception.object_point_counts.join(", ") : "-"),
    metric("目录", `<span class="small">${data.run_dir || "-"}</span>`),
  ].join("");

  document.getElementById("options").textContent = JSON.stringify(request.options || {}, null, 2);
  document.getElementById("processes").innerHTML = (data.processes || []).length
    ? (data.processes || []).map(x => `<div class="process">${x}</div>`).join("")
    : '<span class="small">没有检测到抓取相关进程</span>';

  renderImages(data.viz || {});
  renderPlots(plan, data.candidate_validation || []);
  renderCandidates(data.candidate_validation || []);
  renderCycles(cycles);
  renderLogs(data.logs || {});
}

function renderImages(viz) {
  const entries = [
    ["target_card_color.png", "目标标识牌原图"],
    ["target_card_overlay.png", "目标标识牌识别"],
    ["segmentation_overlay.png", "分割结果"],
    ["grasp_projection.png", "抓取投影"],
  ].filter(([name]) => viz[name]);
  document.getElementById("images").innerHTML = entries.length
    ? entries.map(([name, label]) => `<figure><img src="${viz[name]}" alt="${label}"><figcaption>${label}</figcaption></figure>`).join("")
    : '<div class="small">本次任务还没有可视化图片</div>';
}

function renderPlacementScan(scan) {
  const root = document.getElementById("placementScan");
  if (!scan || !scan.available) {
    root.innerHTML = '<div class="small">尚未扫描</div>';
    return;
  }
  const match = scan.label_match || {};
  const diagnostics = match.diagnostics || {};
  const order = diagnostics.detected_item_ids_left_to_right || [];
  const pitches = diagnostics.adjacent_pitch_mm || [];
  const centers = diagnostics.box_centers_base_m || [];
  const pose = scan.robot_pose_mm_deg || {};
  const alignment = scan.target_alignment || {};
  const viz = scan.viz || {};
  const validationOk = scan.success === true;
  const detectedCount = Number(match.detected_label_count || order.length || 0);
  const validationMessage = scan.validation_message || match.message || "";
  const validationText = validationOk ? "通过（6/6）" : "未通过（" + detectedCount + "/6）";
  const imageEntries = [
    [viz.overlay, "六盒标识"],
    [viz.depth, "对齐深度"],
    [viz.color, "RGB 原图"],
  ].filter(([url]) => url);
  const viewEntries = (scan.views || []).flatMap(view => {
    const viewViz = view.viz || {};
    return [
      [viewViz.overlay, `${view.view_name || "view"} 检测框`],
      [viewViz.depth, `${view.view_name || "view"} 深度`],
    ].filter(([url]) => url);
  });
  const allImageEntries = viewEntries.length ? viewEntries : imageEntries;
  root.innerHTML = `
    <div class="small">扫描：${scan.scan_id || "-"}　时间：${scan.created_at || "-"}</div>
    <div class="small ${validationOk ? "ok" : "bad"}">盒标校验：${validationText} ${validationMessage}</div>
    <div class="small">当前位姿 mm/deg：${JSON.stringify(pose)}</div>
    <div class="small">从左到右：${order.join(" → ") || "-"}</div>
    <div class="small">相邻间距 mm：${pitches.map(v => Number(v).toFixed(1)).join(", ") || "-"}</div>
    <div class="small">盒中心 base(m)：${centers.length ? JSON.stringify(centers) : "-"}</div>
    <div class="small">扫描模式：${scan.scan_mode || "single_view"}　底盘回原点：${scan.base_returned_to_start === undefined ? "不适用" : String(scan.base_returned_to_start)}</div>
    <div class="small">目标对准：${alignment.success ? `${alignment.item_id} / 槽位 ${alignment.slot_index} / ${alignment.selected_view_name}` : "尚未执行"}</div>
    <div class="images">${allImageEntries.map(([url, label]) =>
      `<figure><img src="${url}?t=${encodeURIComponent(scan.scan_id || Date.now())}" alt="${label}"><figcaption>${label}</figcaption></figure>`
    ).join("")}</div>
  `;
}

async function loadPlacementScan() {
  try {
    const response = await fetch("/api/placement-scan", {cache: "no-store"});
    if (!response.ok) return;
    renderPlacementScan(await response.json());
  } catch (_) {
    return;
  }
}

function renderCandidates(records) {
  const rows = records.map(r => {
    const cls = r.selection_result === "selected_for_execution" ? "selected" : "";
    const result = r.robot_validation_result || r.selection_result || "-";
    const err = r.ik_error_message || "";
    const waypoints = (r.waypoint_results || []).map(w => `${w.stage}:${w.status}`).join(" ");
    return `<tr class="${cls}">
      <td>${r.candidate_index ?? "-"}</td>
      <td>${fmt(r.candidate_score, 4)}</td>
      <td>${result}</td>
      <td>${r.robot_validation_stage || "-"}</td>
      <td>${mmText(r.target_base_m)}</td>
      <td>${posText(r.target_rpy_deg)}</td>
      <td>${waypoints}</td>
      <td>${err}</td>
    </tr>`;
  }).join("");
  document.getElementById("candidateTable").innerHTML = `
    <thead><tr>
      <th>#</th><th>分数</th><th>结果</th><th>阶段</th><th>目标 mm</th><th>RPY deg</th><th>IK</th><th>消息</th>
    </tr></thead><tbody>${rows || '<tr><td colspan="8" class="small">没有候选验证记录</td></tr>'}</tbody>`;
}

function renderCycles(cycles) {
  const slim = (cycles.cycles || []).map(c => ({
    phase: c.phase,
    robot_pose: c.robot_state && c.robot_state.current_pose,
    capture: c.capture && {
      scene_id: c.capture.scene_id,
      size: `${c.capture.color_width}x${c.capture.color_height}`,
      intrinsics: c.capture.camera_intrinsics,
    },
    analyze: c.analyze && {
      candidate_count: c.analyze.candidate_count,
      selected: c.analyze.selected_candidate,
      diagnostics: c.analyze.diagnostics,
    },
    base_to_camera: c.base_to_camera,
  }));
  document.getElementById("cycles").textContent = JSON.stringify(slim, null, 2);
}

function renderLogs(logs) {
  const names = Object.keys(logs);
  document.getElementById("logs").textContent = names.length
    ? names.map(name => `# ${name}\n${logs[name]}`).join("\n\n")
    : "没有找到 session 日志";
}

function renderPlots(plan, records) {
  const points = [];
  const add = (name, arr, color) => { if (Array.isArray(arr)) points.push({name, x: Number(arr[0]), y: Number(arr[1]), z: Number(arr[2]), color}); };
  add("pregrasp", plan.pregrasp_base_m, "#2563eb");
  add("grasp", plan.grasp_base_m, "#d97706");
  add("target", plan.target_base_m, "#dc2626");
  add("retreat", plan.retreat_base_m, "#16a34a");
  for (const r of records) add(`cand${r.candidate_index}`, r.target_base_m, r.selection_result === "selected_for_execution" ? "#7c3aed" : "#94a3b8");
  drawPlot("topPlot", points, "x", "y", "俯视 X/Y");
  drawPlot("sidePlot", points, "x", "z", "侧视 X/Z");
}

function drawPlot(id, points, ax, ay, title) {
  const svg = document.getElementById(id);
  if (!points.length) { svg.innerHTML = ""; return; }
  const pad = 38, w = 640, h = 320;
  const xs = points.map(p => p[ax]), ys = points.map(p => p[ay]);
  const minX = Math.min(...xs) - 0.04, maxX = Math.max(...xs) + 0.04;
  const minY = Math.min(...ys) - 0.04, maxY = Math.max(...ys) + 0.04;
  const sx = x => pad + (x - minX) / Math.max(0.001, maxX - minX) * (w - pad * 2);
  const sy = y => h - pad - (y - minY) / Math.max(0.001, maxY - minY) * (h - pad * 2);
  const main = points.filter(p => ["pregrasp", "grasp", "target", "retreat"].includes(p.name));
  const line = main.map(p => `${sx(p[ax])},${sy(p[ay])}`).join(" ");
  svg.innerHTML = `
    <text x="14" y="22" font-size="13" fill="#334155">${title}</text>
    <line x1="${pad}" y1="${h-pad}" x2="${w-pad}" y2="${h-pad}" stroke="#cbd5e1"/>
    <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${h-pad}" stroke="#cbd5e1"/>
    <text x="${w-pad-35}" y="${h-12}" font-size="11" fill="#64748b">${ax} m</text>
    <text x="10" y="${pad-10}" font-size="11" fill="#64748b">${ay} m</text>
    ${line ? `<polyline points="${line}" fill="none" stroke="#111827" stroke-width="2" stroke-dasharray="5 4"/>` : ""}
    ${points.map(p => `<g><circle cx="${sx(p[ax])}" cy="${sy(p[ay])}" r="${["pregrasp","grasp","target","retreat"].includes(p.name) ? 6 : 3}" fill="${p.color}"/><text x="${sx(p[ax])+7}" y="${sy(p[ay])-7}" font-size="11" fill="#0f172a">${p.name}</text></g>`).join("")}
  `;
}

document.getElementById("runSelect").addEventListener("change", e => {
  state.selected = decodeURIComponent(e.target.value);
  loadRun();
});
document.getElementById("refreshBtn").addEventListener("click", async () => {
  await loadRuns();
  await loadRun();
});

document.querySelectorAll("[data-prompt]").forEach(button => {
  button.addEventListener("click", () => {
    document.getElementById("promptInput").value = button.dataset.prompt || "";
    document.getElementById("targetItemInput").value = button.dataset.itemId || "";
    if (button.dataset.centerContact) {
      document.getElementById("centerContactInput").checked =
        button.dataset.centerContact === "true";
    }
    if (button.dataset.executionStrategy) {
      document.getElementById("executionStrategyInput").value =
        button.dataset.executionStrategy;
    }
  });
});

document.getElementById("clearBiasBtn").addEventListener("click", () => {
  document.getElementById("biasXInput").value = "0";
  document.getElementById("biasYInput").value = "0";
  document.getElementById("biasZInput").value = "0";
});

document.getElementById("flipYBiasBtn").addEventListener("click", () => {
  const input = document.getElementById("biasYInput");
  input.value = String(-Number(input.value || 0));
});

const speedInput = document.getElementById("speedInput");
const speedRangeInput = document.getElementById("speedRangeInput");
speedInput.addEventListener("input", () => { speedRangeInput.value = speedInput.value || "5"; });
speedRangeInput.addEventListener("input", () => { speedInput.value = speedRangeInput.value || "5"; });

function buildRunPayload({confirm}) {
  return {
    auto_target_from_card: document.getElementById("autoTargetCardInput").checked,
    prompt: document.getElementById("promptInput").value.trim(),
    target_item_id: document.getElementById("targetItemInput").value,
    speed: Number(document.getElementById("speedInput").value || 5),
    execute: true,
    confirm,
    place_after_grasp: document.getElementById("placeAfterGraspInput").checked,
    base_grasp_scan: document.getElementById("baseGraspScanInput").checked,
    base_motion_ack: document.getElementById("baseMotionAckInput").checked,
    move_home_after: document.getElementById("moveHomeInput").checked,
    use_object_center_contact: document.getElementById("centerContactInput").checked,
    execution_strategy: document.getElementById("executionStrategyInput").value,
    precenter: document.getElementById("precenterInput").checked,
    enable_pregrasp: document.getElementById("pregraspInput").checked,
    show_pointcloud: document.getElementById("showPointcloudInput").checked,
    bias_x_mm: Number(document.getElementById("biasXInput").value || 0),
    bias_y_mm: Number(document.getElementById("biasYInput").value || 0),
    bias_z_mm: Number(document.getElementById("biasZInput").value || 0),
  };
}

function syncAutomaticTargetUi() {
  const automatic = document.getElementById("autoTargetCardInput").checked;
  document.getElementById("targetItemInput").disabled = automatic;
  document.getElementById("promptInput").disabled = automatic;
  document.getElementById("executionStrategyInput").disabled = automatic;
  document.getElementById("baseGraspScanInput").checked = automatic || document.getElementById("baseGraspScanInput").checked;
  document.getElementById("baseGraspScanInput").disabled = automatic;
}
document.getElementById("autoTargetCardInput").addEventListener("change", syncAutomaticTargetUi);
syncAutomaticTargetUi();

async function submitRun(runPayload, label) {
  if (runPayload.base_grasp_scan && !window.confirm(
    "底盘将向前扫描最多 1.5m；目标居中并完成抓取抬升后，无论瓶子或物块，底盘都会在机械臂移动到观察位的同时连续前进 1.5m；放置并退出盒体后还会再连续前进 1.5m。请确认前方累计 4.5m 通道无人且硬件急停可用。"
  )) {
    return;
  }
  try {
    setActionBusy(true);
    setActionStatus(`${label}：正在下发参数并触发任务...`, "warn");
    const data = await postJson("/api/run", runPayload);
    setActionStatus((data.trigger && data.trigger.stdout || "任务已触发").trim(), "ok");
    await loadRuns();
    await loadRun();
  } catch (err) {
    setActionStatus(String(err.message || err), "bad");
  } finally {
    setActionBusy(false);
  }
}

document.getElementById("runForm").addEventListener("submit", e => {
  e.preventDefault();
  submitRun(buildRunPayload({confirm: false}), "直接抓取");
});

document.getElementById("planConfirmBtn").addEventListener("click", () => {
  submitRun(buildRunPayload({confirm: true}), "规划");
});

async function triggerControl(path, label) {
  try {
    setActionBusy(true);
    setActionStatus(`${label}...`);
    const data = await postJson(path, {});
    const resultText = data.message || (data.result && data.result.stdout) || `${label} 已发送`;
    const logText = data.log ? ` | 日志：${data.log}` : "";
    setActionStatus(`${String(resultText).trim()}${logText}`, "ok");
    await loadRuns();
    await loadRun();
  } catch (err) {
    setActionStatus(String(err.message || err), "bad");
  } finally {
    setActionBusy(false);
  }
}

document.getElementById("startStackBtn").addEventListener("click", () => triggerControl("/api/start_stack", "启动真机栈"));
document.getElementById("stopStackBtn").addEventListener("click", () => triggerControl("/api/stop_stack", "停止真机栈"));
document.getElementById("confirmBtn").addEventListener("click", () => triggerControl("/api/confirm", "确认执行"));
document.getElementById("rejectBtn").addEventListener("click", () => triggerControl("/api/reject", "拒绝"));
document.getElementById("stopBtn").addEventListener("click", () => triggerControl("/api/stop", "停止任务"));
document.getElementById("openGripperBtn").addEventListener("click", () => {
  if (!window.confirm("夹爪将立即打开，机械臂不会移动。请先托住物体并确认下方安全。是否继续？")) {
    return;
  }
  triggerControl("/api/open-gripper", "释放夹爪");
});
document.getElementById("probeBtn").addEventListener("click", () => triggerControl("/api/probe", "probe"));
document.getElementById("scanPlacementBtn").addEventListener("click", async () => {
  try {
    setActionBusy(true);
    setActionStatus("正在读取位姿并采集放置区 RGB-D；不会移动机械臂...", "warn");
    const data = await postJson("/api/scan-placement", {
      target_item_id: document.getElementById("targetItemInput").value,
    });
    renderPlacementScan(data.scan);
    setActionStatus(
      data.message || "放置区扫描完成",
      data.validation_ok === false ? "warn" : "ok"
    );
  } catch (err) {
    setActionStatus(String(err.message || err), "bad");
  } finally {
    setActionBusy(false);
  }
});
document.getElementById("scanPlacementMultiBtn").addEventListener("click", async () => {
  try {
    const acknowledged = document.getElementById("baseMotionAckInput").checked;
    if (!acknowledged) {
      throw new Error("请先确认底盘前方累计 4.5m 通道无障碍，并准备好硬件急停");
    }
    const targetItem = document.getElementById("targetItemInput").value;
    if (!targetItem) {
      throw new Error("请先选择要放置的指定物品");
    }
    if (!window.confirm(`底盘将向前分段扫描，最多 1.5m；仅在 ${targetItem} 盒标位于画面中心附近时停车。机械臂与夹爪不会动作。确认开始？`)) {
      return;
    }
    setActionBusy(true);
    setActionStatus("正在单向扫描并对准目标盒：请勿进入运动区域，随时准备按硬件急停...", "warn");
    const data = await postJson("/api/scan-and-align-placement-target", {
      target_item_id: targetItem,
      base_motion_ack: true,
    });
    renderPlacementScan(data.scan);
    setActionStatus(
      data.message || "目标盒扫描并对准完成",
      data.validation_ok === false ? "warn" : "ok"
    );
  } catch (err) {
    setActionStatus(String(err.message || err), "bad");
  } finally {
    setActionBusy(false);
  }
});
document.getElementById("executeAlignedPlaceBtn").addEventListener("click", async () => {
  try {
    const targetItem = document.getElementById("targetItemInput").value;
    if (!targetItem) {
      throw new Error("请先选择要放置的指定物品");
    }
    if (!window.confirm(`机械臂将移动到 ${targetItem} 对应盒口、下降并张开夹爪释放物体，然后抬升撤离。请确认底盘已对准、物体夹持稳定、盒内和路径无人手。是否执行？`)) {
      return;
    }
    setActionBusy(true);
    setActionStatus("正在校验底盘对准状态并执行标定放置；夹爪将自动释放...", "warn");
    const data = await postJson("/api/execute-aligned-place", {
      target_item_id: targetItem,
      release_ack: true,
    });
    renderPlacementScan(data.scan);
    setActionStatus(data.message || "标定放置完成", "ok");
  } catch (err) {
    setActionStatus(String(err.message || err), "bad");
  } finally {
    setActionBusy(false);
  }
});

(async function init() {
  await loadRuns();
  await loadRun();
  await loadPlacementScan();
  setInterval(async () => { await loadRuns(); await loadRun(); await loadPlacementScan(); }, 5000);
})();
</script>
</body>
</html>
"""


class DashboardHandler(BaseHTTPRequestHandler):
    def _send_bytes(self, data: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            self.wfile.write(data)
        except BrokenPipeError:
            return

    def _send_json(self, payload: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send_bytes(json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8", status)

    def _read_json_body(self) -> dict[str, object]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw.decode("utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/" or path == "/index.html":
            self._send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/api/runs":
            self._send_json(_runs_payload())
            return
        if path == "/api/status":
            self._send_json(_component_status())
            return
        if path == "/api/placement-scan":
            self._send_json(_placement_scan_payload())
            return
        if path.startswith("/api/run/"):
            run_id = unquote(path.removeprefix("/api/run/")).strip("/")
            run_dir = _safe_relative_path(ARTIFACT_ROOT, f"{run_id}/final_result.json")
            if run_dir is None:
                self._send_json({"error": "run not found"}, HTTPStatus.NOT_FOUND)
                return
            self._send_json(_run_payload(run_dir.parent))
            return
        if path.startswith("/viz/"):
            parts = path.removeprefix("/viz/").split("/", 1)
            if len(parts) != 2:
                self._send_json({"error": "bad viz path"}, HTTPStatus.BAD_REQUEST)
                return
            file_path = _safe_relative_path(VIZ_ROOT, f"{parts[0]}/{parts[1]}")
            if file_path is None:
                self._send_json({"error": "file not found"}, HTTPStatus.NOT_FOUND)
                return
            content_type = "image/png" if file_path.suffix.lower() == ".png" else "text/plain; charset=utf-8"
            self._send_bytes(file_path.read_bytes(), content_type)
            return
        if path.startswith("/run-artifact/"):
            relative = unquote(path.removeprefix("/run-artifact/")).strip("/")
            file_path = _safe_relative_path(ARTIFACT_ROOT, relative)
            if file_path is None or file_path.suffix.lower() != ".png":
                self._send_json({"error": "file not found"}, HTTPStatus.NOT_FOUND)
                return
            self._send_bytes(file_path.read_bytes(), "image/png")
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        path = parsed.path
        payload = self._read_json_body()
        try:
            if path == "/api/start_stack":
                result = _start_live_stack()
                self._send_json(result, HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/stop_stack":
                result = _stop_live_stack()
                self._send_json(result, HTTPStatus.OK if result.get("ok") else HTTPStatus.INTERNAL_SERVER_ERROR)
                return
            if path == "/api/run":
                result = _start_grasp_from_payload(payload)
                self._send_json(result, HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/scan-placement":
                result = _scan_placement_from_payload(payload)
                self._send_json(result, HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/scan-placement-multiview":
                result = _scan_placement_multi_view_from_payload(payload)
                self._send_json(result, HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/scan-and-align-placement-target":
                result = _scan_and_align_placement_target_from_payload(payload)
                self._send_json(result, HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/align-placement-target":
                result = _align_placement_target_from_payload(payload)
                self._send_json(result, HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST)
                return
            if path == "/api/execute-aligned-place":
                result = _execute_aligned_place_from_payload(payload)
                self._send_json(result, HTTPStatus.OK if result.get("ok") else HTTPStatus.BAD_REQUEST)
                return
            if path in {"/api/confirm", "/api/reject", "/api/stop", "/api/open-gripper", "/api/probe"}:
                service = {
                    "/api/confirm": "/grasp_pipeline/confirm",
                    "/api/reject": "/grasp_pipeline/reject",
                    "/api/stop": "/grasp_pipeline/stop",
                    "/api/open-gripper": "/robot_executor/open_gripper",
                    "/api/probe": "/grasp_pipeline/probe",
                }[path]
                result = _trigger_service(service, timeout_s=30.0)
                self._send_json({"ok": bool(result.get("ok")), "result": result})
                return
        except subprocess.TimeoutExpired as exc:
            self._send_json({"ok": False, "error": f"command timed out: {exc}"}, HTTPStatus.REQUEST_TIMEOUT)
            return
        except Exception as exc:
            self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.INTERNAL_SERVER_ERROR)
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def log_message(self, fmt: str, *args) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description="Operator dashboard for distributed grasp runs.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), DashboardHandler)
    print(f"Grasp dashboard: http://{args.host}:{args.port}")
    print(f"artifact_root: {ARTIFACT_ROOT}")
    print(f"viz_root: {VIZ_ROOT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
