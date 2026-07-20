# robot_grasp_ros2 Bootstrap

开始任何任务前，先读本文件。

## Active Scope

当前主写入区限定为 `grasp_ros/robot_grasp_ros2`。

以下目录默认视为参考输入，不作为主写入区，除非任务明确要求：

- `old/robot_grasp/`
- `Agilex-College/`
- `robotic_arm_kinematics/`
- `piper_ros_humble/`

## Mandatory Read Order

每次开始新任务时，按下面顺序建立上下文：

1. `README.md`
2. `docs/CURRENT_STATUS.md`
3. `docs/DISTRIBUTED_RUNBOOK.md`
4. `docs/DISTRIBUTED_ARCHITECTURE.md`
5. `docs/MIGRATION_CONTRACT.md`
6. `docs/ENGINEERING_SPEC.md`
7. `docs/MIGRATION_TODO.md`
8. `docs/ROBOT_COUPLING_MAP.md`
9. 与当前任务直接相关的 `src/` 或 `robot_grasp_ros2/` 文件

## Source Of Truth Priority

当文档、旧实现和当前代码不一致时，按下面优先级判断：

1. 当前可执行代码：`robot_grasp_ros2/src/`、`robot_grasp_ros2/robot_grasp_ros2/`
2. 当前状态与运行文档：`docs/CURRENT_STATUS.md`、`docs/DISTRIBUTED_RUNBOOK.md`
3. 当前迁移契约与工程规范：`docs/MIGRATION_CONTRACT.md`、`docs/ENGINEERING_SPEC.md`
4. 当前迁移 backlog：`docs/MIGRATION_TODO.md`
5. 旧版行为基线：`old/robot_grasp/src/`
6. 外部参考：`piper_ros_humble/`、`Agilex-College/`、`robotic_arm_kinematics/`

## Working Agreement

- 当前推荐主线是分布式模式，单节点模式是兼容 / 调试路径。
- 规划层继续使用 `mm/deg` 语义，ROS2 适配层负责 `mm/deg <-> m/rad` 转换。
- 新工程业务逻辑禁止直接依赖 `piper_sdk`。
- 所有硬件访问必须经过 `src/robot/`。
- 优先做 `fake`、`probe`、单步验证，不直接做危险真机联调。
- 新增 CLI、topic、service、字段或单位语义时，必须同步更新 `docs/`。
