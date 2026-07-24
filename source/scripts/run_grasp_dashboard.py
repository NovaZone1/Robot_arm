#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import subprocess
import time
from urllib.parse import quote, unquote, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUNDLE_ROOT = PROJECT_ROOT.parent
WORKSPACE_ROOT = BUNDLE_ROOT / "ros_ws"
ARTIFACT_ROOT = BUNDLE_ROOT / "log" / "distributed_runs"
VIZ_ROOT = WORKSPACE_ROOT / "viz"
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
]
CAN_PORT = "can0"
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
        r"robot_grasp_ros2\.(camera_server_node|vision_worker_node|robot_executor_node|pipeline_orchestrator_node)|"
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


def _start_live_stack() -> dict[str, object]:
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


def _param_set(node_name: str, param_name: str, value: str, *, timeout_s: float = 10.0) -> dict[str, object]:
    result = _run_ros2_command(["param", "set", node_name, param_name, value], timeout_s=timeout_s)
    output = f"{result.get('stdout', '')}\n{result.get('stderr', '')}"
    if "Setting parameter failed" in output:
        result["ok"] = False
    return result


def _trigger_service(service_name: str, *, timeout_s: float = 20.0) -> dict[str, object]:
    return _run_ros2_command(
        ["service", "call", service_name, "std_srvs/srv/Trigger", "{}"],
        timeout_s=timeout_s,
    )


def _bool_text(value: object) -> str:
    return "true" if bool(value) else "false"


def _start_grasp_from_payload(payload: dict[str, object]) -> dict[str, object]:
    prompt = str(payload.get("prompt") or "").strip()
    if not prompt:
        return {"ok": False, "error": "Prompt 不能为空"}
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
        _param_set("/grasp_pipeline", "prompt", prompt),
        _param_set("/grasp_pipeline", "execute", _bool_text(payload.get("execute", True))),
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
        _param_set(
            "/robot_executor",
            "execution_strategy",
            requested_strategy,
        ),
    ]
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

    return {
        "run_dir": str(run_dir),
        "run_id": run_dir.name,
        "final_result": final_result,
        "request": request,
        "cycles": cycles,
        "candidate_validation": candidate_validation,
        "viz": _scene_viz_files(scene_id),
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
    .prompt-row { display: grid; grid-template-columns: minmax(260px, 560px) minmax(220px, 320px); gap: 14px; align-items: end; }
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
        <button type="button" data-prompt="red block" data-center-contact="true" data-execution-strategy="safe_top_down">红色物块</button>
        <button type="button" data-prompt="yellow block" data-center-contact="true" data-execution-strategy="safe_top_down">黄色物块</button>
        <button type="button" data-prompt="blue block" data-center-contact="true" data-execution-strategy="safe_top_down">蓝色物块</button>
      </div>
      <div class="toggles">
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
        <label><input id="moveHomeInput" type="checkbox" /> 抓后交接并回 Home</label>
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
      <button id="probeBtn" type="button">Probe</button>
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
const runButtons = ["planConfirmBtn", "directGraspBtn", "confirmBtn", "rejectBtn", "stopBtn", "probeBtn"];

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
    ["segmentation_overlay.png", "分割结果"],
    ["grasp_projection.png", "抓取投影"],
  ].filter(([name]) => viz[name]);
  document.getElementById("images").innerHTML = entries.length
    ? entries.map(([name, label]) => `<figure><img src="${viz[name]}" alt="${label}"><figcaption>${label}</figcaption></figure>`).join("")
    : '<div class="small">本次任务还没有可视化图片</div>';
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
    prompt: document.getElementById("promptInput").value.trim(),
    speed: Number(document.getElementById("speedInput").value || 5),
    execute: true,
    confirm,
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

async function submitRun(runPayload, label) {
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
document.getElementById("probeBtn").addEventListener("click", () => triggerControl("/api/probe", "probe"));

(async function init() {
  await loadRuns();
  await loadRun();
  setInterval(async () => { await loadRuns(); await loadRun(); }, 5000);
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
            if path in {"/api/confirm", "/api/reject", "/api/stop", "/api/probe"}:
                service = {
                    "/api/confirm": "/grasp_pipeline/confirm",
                    "/api/reject": "/grasp_pipeline/reject",
                    "/api/stop": "/grasp_pipeline/stop",
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
