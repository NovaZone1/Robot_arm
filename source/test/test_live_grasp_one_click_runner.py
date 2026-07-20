from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from robot_grasp_ros2.live_grasp_one_click import (
    RunningComponents,
    build_launch_plan,
    build_parser,
    resolve_launches_for_running_components,
)


def test_build_parser_defaults_for_live_one_click_runner():
    parser = build_parser()

    args = parser.parse_args([])

    assert args.prompt == "cup"
    assert args.robot_backend == "ros2"
    assert args.execute is False
    assert args.confirm is False
    assert args.precenter is False
    assert args.enable_pregrasp is False
    assert args.show_pointcloud is False
    assert args.open_rviz is True
    assert args.probe_timeout == 60.0
    assert args.moveit_warmup == 10.0
    assert args.once is False
    assert args.result_timeout == 600.0


def test_build_launch_plan_uses_explicit_trigger_after_probe():
    parser = build_parser()
    args = parser.parse_args(
        [
            "mug",
            "--execute",
            "--confirm",
            "--precenter",
            "--enable-pregrasp",
            "--show-pointcloud",
            "--no-rviz",
        ]
    )

    plan = build_launch_plan(PROJECT_ROOT, args)

    assert plan.robot_backend == "ros2"
    assert plan.execute is True
    assert plan.once is False
    assert [launch.name for launch in plan.launches] == [
        "piper_driver",
        "moveit_ik",
        "distributed_stack",
    ]

    distributed_cmd = plan.launches[2].argv
    assert distributed_cmd[:2] == ["./scripts/run_distributed_stack_graspnet.sh", "--robot-backend"]
    assert "--execute" in distributed_cmd
    assert "--confirm" in distributed_cmd
    assert "--precenter" in distributed_cmd
    assert "--enable-pregrasp" in distributed_cmd
    assert "--show-pointcloud" in distributed_cmd
    assert "--prompt" not in distributed_cmd
    assert "mug" not in distributed_cmd

    assert plan.compute_ik_service == "/compute_ik"
    assert plan.configure_param_argvs == [
        ["./scripts/ros2_system.sh", "param", "set", "/grasp_pipeline", "execute", "true"],
        ["./scripts/ros2_system.sh", "param", "set", "/grasp_pipeline", "confirm", "true"],
        ["./scripts/ros2_system.sh", "param", "set", "/grasp_pipeline", "precenter", "true"],
        ["./scripts/ros2_system.sh", "param", "set", "/grasp_pipeline", "show_pointcloud", "true"],
        ["./scripts/ros2_system.sh", "param", "set", "/grasp_pipeline", "enable_pregrasp", "true"],
        ["./scripts/ros2_system.sh", "param", "set", "/robot_executor", "enable_pregrasp", "true"],
    ]
    assert plan.probe_argv == [
        "./scripts/ros2_system.sh",
        "service",
        "call",
        "/grasp_pipeline/probe",
        "std_srvs/srv/Trigger",
        "{}",
    ]
    assert plan.trigger_argv == ["./scripts/run_pipeline_service.sh", "mug"]


def test_build_launch_plan_supports_fake_plan_only_once():
    parser = build_parser()
    args = parser.parse_args(["cup", "--robot-backend", "fake", "--plan-only", "--once", "--no-rviz"])

    plan = build_launch_plan(PROJECT_ROOT, args)

    assert plan.robot_backend == "fake"
    assert plan.execute is False
    assert plan.once is True
    assert plan.compute_ik_service is None
    assert [launch.name for launch in plan.launches] == ["distributed_stack"]
    assert plan.launches[0].argv == [
        "./scripts/run_distributed_stack_graspnet.sh",
        "--robot-backend",
        "fake",
    ]
    assert ["./scripts/ros2_system.sh", "param", "set", "/grasp_pipeline", "execute", "false"] in plan.configure_param_argvs


def test_build_launch_plan_resets_confirm_false_by_default_before_trigger():
    parser = build_parser()
    args = parser.parse_args([])

    plan = build_launch_plan(PROJECT_ROOT, args)

    assert ["./scripts/ros2_system.sh", "param", "set", "/grasp_pipeline", "execute", "false"] in plan.configure_param_argvs
    assert ["./scripts/ros2_system.sh", "param", "set", "/grasp_pipeline", "confirm", "false"] in plan.configure_param_argvs


def test_build_launch_plan_forces_confirmation_for_ros2_execution():
    parser = build_parser()
    args = parser.parse_args(["cup", "--execute", "--no-rviz"])

    plan = build_launch_plan(PROJECT_ROOT, args)

    assert plan.execute is True
    assert "--execute" in plan.launches[2].argv
    assert "--confirm" in plan.launches[2].argv
    assert ["./scripts/ros2_system.sh", "param", "set", "/grasp_pipeline", "confirm", "true"] in plan.configure_param_argvs


def test_resolve_launches_skips_already_running_stack_components():
    parser = build_parser()
    args = parser.parse_args([])
    plan = build_launch_plan(PROJECT_ROOT, args)

    running = RunningComponents(
        driver_running=True,
        moveit_running=True,
        distributed_running=True,
        distributed_partial_nodes=(),
        rviz_running=False,
        details=("move_group", "piper_single_ctrl"),
    )

    resolved = resolve_launches_for_running_components(plan, running)

    assert [launch.name for launch in resolved] == ["rviz"]


def test_resolve_launches_rejects_partial_distributed_stack():
    parser = build_parser()
    args = parser.parse_args([])
    plan = build_launch_plan(PROJECT_ROOT, args)

    running = RunningComponents(
        driver_running=False,
        moveit_running=False,
        distributed_running=False,
        distributed_partial_nodes=("pipeline_orchestrator_node",),
        rviz_running=False,
        details=("pipeline_orchestrator_node",),
    )

    try:
        resolve_launches_for_running_components(plan, running)
    except RuntimeError as exc:
        assert "Partial distributed stack" in str(exc)
    else:
        raise AssertionError("expected partial distributed stack to be rejected")
