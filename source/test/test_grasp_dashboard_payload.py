import importlib.util
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_dashboard_module():
    path = PROJECT_ROOT / "scripts" / "run_grasp_dashboard.py"
    spec = importlib.util.spec_from_file_location("run_grasp_dashboard", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_start_grasp_uses_explicit_safe_top_down_strategy(monkeypatch):
    dashboard = _load_dashboard_module()
    parameter_calls = []

    monkeypatch.setattr(
        dashboard,
        "_component_status",
        lambda: {"stack_ready": True, "components": {}},
    )
    monkeypatch.setattr(
        dashboard,
        "_param_set",
        lambda node, name, value: (
            parameter_calls.append((node, name, value))
            or {"ok": True}
        ),
    )
    monkeypatch.setattr(
        dashboard,
        "_trigger_service",
        lambda *_args, **_kwargs: {"ok": True, "stdout": "accepted"},
    )

    result = dashboard._start_grasp_from_payload(
        {
            "prompt": "red block",
            "execution_strategy": "safe_top_down",
            "use_object_center_contact": True,
            "execute": True,
            "confirm": True,
        }
    )

    assert result["ok"] is True
    assert (
        "/robot_executor",
        "execution_strategy",
        "safe_top_down",
    ) in parameter_calls
