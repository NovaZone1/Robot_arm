# Code Status Map

更新时间：2026-04-28

本文档记录 `robot_grasp_ros2` 当前工作区里“哪些代码已经是相对稳定的基线、哪些文件仍在迁移/重构中、哪些引用已经过时或需要复核”。

定位说明：

- 这是“接手时的文件状态清单”，不是运行手册
- 结论基于 2026-04-27 的当前 worktree 检查结果
- 当代码、文档、旧迁移计划不一致时，优先相信当前可执行代码

## 1. 状态定义

- `stable-baseline`
  - 已经接入当前主线，后续开发默认可在此基础上继续扩展
- `active-refactor`
  - 当前 worktree 里仍有未提交修改，或者是最近新引入的迁移实现
- `stale-or-verify`
  - 仍有旧路径、旧假设、机器绑定配置，使用前必须先复核

## 2. 当前 worktree 快照

本次检查看到：

- 已跟踪修改文件：22 个
- 未跟踪新文件：4 个

当前最明显的活跃变更面有三块：

1. `camera_server` 外部采集 worker 化
2. `vision_worker` 外部推理 worker 化
3. 分布式运行脚本、参数文件和状态文档同步调整

## 3. Stable Baseline

下面这些文件/模块已经构成当前推荐主线的稳定骨架。

### 3.1 ROS2 包骨架与接口契约

- `robot_grasp_msgs/`
  - `msg/`
  - `srv/`
  - `action/RunGraspPipeline.action`
- `robot_grasp_ros2/setup.py`
- `robot_grasp_ros2/launch/distributed_grasp_pipeline.launch.py`

说明：

- `robot_grasp_msgs` 已经不是占位目录，而是当前分布式节点之间正在使用的 typed contract
- `distributed_grasp_pipeline.launch.py` 已明确四节点主线：`camera_server`、`vision_worker`、`robot_executor`、`grasp_pipeline`

### 3.2 分布式主链路骨架

- `robot_grasp_ros2/robot_grasp_ros2/pipeline_orchestrator_node.py`
- `robot_grasp_ros2/robot_grasp_ros2/grasp_pipeline_node.py`
- `robot_grasp_ros2/robot_grasp_ros2/distributed_utils.py`
- `robot_grasp_ros2/robot_grasp_ros2/rviz_visualization.py`

说明：

- `pipeline_orchestrator_node.py` 是当前推荐主入口，对外暴露 `/grasp_pipeline/run`、`/probe`、`/stop`、`/confirm`、`/reject`
- `grasp_pipeline_node.py` 仍然保留，但定位是单节点兼容/对照路径，不是主线

### 3.3 规划与纯逻辑层

- `robot_grasp_ros2/src/grasping/models.py`
- `robot_grasp_ros2/src/grasping/planning.py`
- `robot_grasp_ros2/src/grasping/coordinator.py`
- `robot_grasp_ros2/src/utils/calibration.py`
- `robot_grasp_ros2/src/utils/transforms.py`
- `robot_grasp_ros2/src/utils/npoint_tool_offset.py`
- `robot_grasp_ros2/src/perception/geometry.py`

说明：

- 这部分总体上已经完成“旧逻辑迁移 + 新工程落位”
- 后续更多是行为调优，不是架构层面的推倒重写

### 3.4 机器人抽象层骨架与调试链路

- `robot_grasp_ros2/src/robot/types.py`
- `robot_grasp_ros2/src/robot/executor_models.py`
- `robot_grasp_ros2/robot_grasp_ros2/piper_interactive_marker_node.py`
- `robot_grasp_ros2/robot_grasp_ros2/piper_pose_bridge_node.py`
- `robot_grasp_ros2/robot_grasp_ros2/joint_state_feedback_relay_node.py`

说明：

- “机器人必须经过抽象接口访问” 这条约束已经落实到代码结构里
- RViz 交互调试链路已经独立出来，不再需要上层流程直接碰底层 Piper 接口

## 4. Active Refactor

下面这些文件属于“当前仍在演进、改动密度较高”的区域。继续开发前，先看当前代码和本地 diff，不要只看旧文档。

### 4.1 相机采集外部 worker 化

- `robot_grasp_ros2/robot_grasp_ros2/camera_server_node.py`
- `robot_grasp_ros2/src/perception/external_camera_capture_worker.py`

当前判断：

- 这条链已经不再倾向于把 D435 访问直接留在 ROS 节点进程内
- 当前实现已经改成 `camera_server_node -> subprocess -> external_camera_capture_worker.py`
- 这是最近新增的迁移实现，后续还可能继续调参数、超时和异常处理

### 4.2 重型感知外部 worker 化

- `robot_grasp_ros2/robot_grasp_ros2/vision_worker_node.py`
- `robot_grasp_ros2/src/perception/external_inference_worker.py`

当前判断：

- 这是当前 worktree 里最活跃的一块
- 方向已经明确：ROS 节点负责 typed service / RViz 发布，YOLOv8-seg + GraspNet + 点云推理由外部 Python runtime 执行
- 后续如果出问题，默认要成对阅读 `vision_worker_node.py` 和 `external_inference_worker.py`

### 4.3 执行后端与 ROS2 机器人接入细节

- `robot_grasp_ros2/robot_grasp_ros2/robot_executor_node.py`
- `robot_grasp_ros2/src/robot/client.py`
- `robot_grasp_ros2/src/robot/motion_tolerances.py`
- `robot_grasp_ros2/src/run_grasp_pipeline_ros2.py`

当前判断：

- 机器人抽象接口本身已经成立
- 但 `ros2` 真机后端、CLI 兼容入口、状态读取和旧默认参数这几块仍在继续收口
- 当前已经补上一层“微小位移自动收紧容差”的判定逻辑，避免 `3mm` 级命令在实机未动时被误判成功
- `src/robot/client.py` 的接口可以当作稳定契约看待，但实现细节还不是完全冻结状态

### 4.4 运行脚本与参数文件

- `robot_grasp_ros2/scripts/ros_env_graspnet.sh`
- `robot_grasp_ros2/scripts/ros2_system.sh`
- `robot_grasp_ros2/scripts/run_distributed_stack_graspnet.sh`
- `robot_grasp_ros2/scripts/open_distributed_rviz.sh`
- `robot_grasp_ros2/scripts/show_last_distributed_snapshot.sh`
- `robot_grasp_ros2/scripts/run_piper_driver.sh`
- `robot_grasp_ros2/config/distributed/camera_server.params.yaml`
- `robot_grasp_ros2/config/distributed/vision_worker.params.yaml`
- `robot_grasp_ros2/config/distributed/pipeline_orchestrator.params.yaml`
- `robot_grasp_ros2/config/grasp_pipeline.params.yaml`

当前判断：

- 这些文件直接承接了“Jazzy system Python + conda worker + 本地 overlay”的运行形态
- 当前机器上它们是可用的，但仍在跟随架构调整同步变化

### 4.5 正在被更新的项目文档

- `robot_grasp_ros2/README.md`
- `robot_grasp_ros2/AGENTS.md`
- `robot_grasp_ros2/docs/CURRENT_STATUS.md`
- `robot_grasp_ros2/docs/DISTRIBUTED_ARCHITECTURE.md`
- `robot_grasp_ros2/docs/DISTRIBUTED_RUNBOOK.md`
- `robot_grasp_ros2/docs/ENGINEERING_SPEC.md`
- `robot_grasp_ros2/docs/MIGRATION_CONTRACT.md`
- `robot_grasp_ros2/docs/MIGRATION_TODO.md`

当前判断：

- 这些文档总体方向是对的
- 但当前 worktree 里它们本身也在被修改，引用时要优先看最新代码是否已经领先于文档

## 5. Stale Or Verify

下面这些内容不是不能看，而是不能再把它们当成“当前仓库真实状态”的直接事实。

### 5.1 根目录迁移启动文档已经过时

- `AGENTS.md`
- `TODO_ROS2_MIGRATION.md`

原因：

- 这两份文件仍然把仓库描述成“`new/` 准备开始迁移”的阶段
- 它们还引用了 `../old/robot_grasp/...`
- 但当前实际主线已经是 `robot_grasp_ros2/` 成型工程，不再是空骨架

建议：

- 把它们当历史背景或迁移初衷说明看
- 不要把它们当当前代码结构的 source of truth

### 5.2 当前代码里仍残留旧工程默认路径

- `robot_grasp_ros2/src/run_grasp_pipeline_ros2.py`
  - 默认 `hand_eye_config` 指向 `old/robot_grasp`
  - 默认 `npoint_tool_offset_file` 指向 `old/robot_grasp`
  - 默认 `online_bias_file` 指向 `old/robot_grasp`
- `robot_grasp_ros2/config/distributed/pipeline_orchestrator.params.yaml`
  - `hand_eye_config` 当前写死为本机绝对路径

原因：

- 这说明单节点 CLI 兼容路径和部分配置仍然带着迁移期遗留
- 不影响“分布式主线已经成型”这个结论
- 但会影响移植性、默认开箱可用性和后续清理成本

### 5.3 机器环境说明文档是本机事实，不是包级事实

- `WORKSPACE_STATUS.md`

原因：

- 这份文档对这台机器很有用
- 但它记录的是本机 overlay、Conda 干扰、系统 Python 选择等环境状态
- 它不能替代 `robot_grasp_ros2/docs/` 下的包级架构和运行说明

## 6. 当前最实用的接手建议

如果后续要继续开发，建议按下面方式切入：

1. 做分布式主线问题：
   - 先看 `robot_grasp_ros2/robot_grasp_ros2/pipeline_orchestrator_node.py`
   - 再看对应 `robot_grasp_msgs/srv/`
2. 做感知问题：
   - 成对看 `vision_worker_node.py` 和 `external_inference_worker.py`
3. 做相机问题：
   - 成对看 `camera_server_node.py` 和 `external_camera_capture_worker.py`
4. 做机器人执行问题：
   - 先看 `robot_executor_node.py`
   - 再看 `src/robot/client.py`
5. 做“当前仓库真实状态”判断时：
   - 先信 `robot_grasp_ros2/src/` 和 `robot_grasp_ros2/robot_grasp_ros2/`
   - 再信 `robot_grasp_ros2/docs/CURRENT_STATUS.md`
   - 最后再把根目录迁移文档当背景材料参考

## 7. 当前剩余风险

- 当前仓库的自动化测试骨架仍然很薄；目前只新增了 `test/test_motion_tolerances.py` 这类最小回归测试，很多“已验证”事实仍来自人工运行基线
- 分布式主线已经成型，但外部 worker 化这条链仍处在近期活跃调整区
- 部分默认路径和参数仍带有旧工程或本机环境绑定，需要后续专门清理
