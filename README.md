# Piper Grasp Project Layout

这个目录把比赛抓取任务的源码、ROS 工作区、Piper 依赖、GraspNet 和运行产物集中在一起。

## Directory Map

- `source/`：抓取项目主源码；日常开发入口。
- `ros_ws/`：抓取系统 ROS2 工作区；`src/` 指向 `source/` 中的包。
- `piper_ros_ws/`：AgileX `piper_ros` 源码和安装 overlay。
- `piper_sdk/`：Piper ROS 驱动运行时使用的底层 SDK。
- `graspnet/`：GraspNet baseline 和 `checkpoint-rs.tar`。
- `logs/`：分布式单次抓取产物和一键启动日志。
- `tmp/`：Piper 驱动的临时 ROS 日志。

## Main Entry Points

```bash
cd /home/ybw/piper_grasp_project/source
./scripts/run_grasp_dashboard.py
```

或者：

```bash
cd /home/ybw/piper_grasp_project/source
./scripts/run_live_grasp_one_click.sh
```

## Development Rules

- 只在 `source/` 中修改抓取业务代码。
- `ros_ws/src/` 是软链接，不是另一份源码。
- 修改 `robot_grasp_msgs/msg`、`srv` 或 `action` 后，需要在 `ros_ws/` 重新构建。
- 真机控制必须经过 `piper_ros`，业务代码不要直接依赖 `piper_sdk`。
- 正常失能机械臂前必须先回到 Home；急停除外。
