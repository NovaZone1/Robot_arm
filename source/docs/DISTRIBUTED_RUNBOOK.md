# Distributed Runbook

本文档描述 `robot_grasp_ros2` 当前推荐的分布式运行方式。

适用范围：

- 同机多进程验证
- 后续拆分成 Robot Host / Vision Host / Operator Host 的多机部署
- 当前本机的 ROS2 system Python + conda external worker 运行时

## 1. 运行前必须知道的两件事

### 1.1 当前推荐主线是分布式模式

当前推荐主线由四个节点组成：

1. `/grasp_pipeline`
2. `/camera_server`
3. `/vision_worker`
4. `/robot_executor`

单节点模式仍保留，但现在只是兼容 / 调试路径。

### 1.2 这台机器必须区分两套 shell

1. 运行 ROS 节点的 shell
   - ROS 节点实际用系统 Python
   - 但运行环境统一通过 `scripts/ros_env_graspnet.sh`
2. 跑 `ros2` CLI 的 shell
   - 统一通过 `scripts/ros2_system.sh`
3. 跑 AgileX Piper 驱动
   - 统一通过 `scripts/run_piper_driver.sh`
4. 跑 ROS 侧 MoveIt IK 包装层
   - 统一通过 `scripts/run_piper_moveit_ik.sh`

原因：

- Jazzy 的 `rclpy` 只能跟系统 Python 一起工作
- `YOLOv8-seg + GraspNet + Open3D + pyrealsense2` 在 `piper` conda 环境里
- 当前做法是让 `camera_server` 和 `vision_worker` 调 conda worker 子进程
- 把两套运行时混在一个 shell 里，容易触发 `_rclpy_pybind11` 导入失败

### 1.3 实例分割使用 YOLOv8-seg

当前 distributed 与单节点路径均使用 `YOLOSegmenter`，默认模型为 `yolov8n-seg.pt`。

- prompt 必须能匹配 YOLOv8 COCO 类别名，例如 `cup`、`bowl` 或 `bottle`
- 首次运行时 Ultralytics 需要能找到 `yolov8n-seg.pt`；真机联调前建议先在本地准备好权重

## 2. 目录与产物约定

当前有两类重要目录：

1. 分布式 session 日志
   - `ros_ws/log/distributed/<timestamp>/`
   - 里面是每个节点的日志文件
2. 单次 run 结构化产物
   - `logs/distributed_runs/<run_id>/`
   - 当前至少包含：
     - `request.json`
     - `cycles.json`
     - `final_result.json`

辅助脚本：

- `./scripts/show_last_distributed_snapshot.sh`
- `./scripts/show_last_run_artifact.sh`

## 3. 构建要求

只有在以下情况才需要重新 `colcon build`：

- 修改了 `robot_grasp_msgs/msg`
- 修改了 `robot_grasp_msgs/srv`
- 修改了 `robot_grasp_msgs/action`
- 想让 `install/` 的安装产物同步更新

推荐构建命令：

```bash
env -u CONDA_DEFAULT_ENV -u CONDA_EXE -u CONDA_PREFIX -u CONDA_PROMPT_MODIFIER -u CONDA_PYTHON_EXE -u CONDA_SHLVL \
  PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin \
  bash -lc '
    source /opt/ros/jazzy/setup.bash
    source /home/ybw/piper_grasp_project/piper_ros_ws/install/setup.bash
    cd /home/ybw/piper_grasp_project/ros_ws
    colcon build --symlink-install --cmake-clean-cache \
      --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3 -DPython_EXECUTABLE=/usr/bin/python3
  '
```

原因：

- 如果直接在 conda 污染的 shell 里 `colcon build`，`rosidl_adapter` 可能会错误使用 conda Python
- 当前 workspace 已经验证需要用 `/usr/bin/python3` 重新生成消息代码

## 4. 同机分布式最短用法

### 4.1 启动整套分布式节点

终端 A：

```bash
source /home/ybw/piper_grasp_project/source/scripts/ros_env_graspnet.sh
cd /home/ybw/piper_grasp_project/source
./scripts/run_distributed_stack_graspnet.sh --robot-backend fake
```

默认行为：

- 四个节点同时启动
- 终端保持占用是正常行为
- 按 `Ctrl+C` 会一起停掉四个节点
- session 日志写到 `ros_ws/log/distributed/<timestamp>/`
- 如果旧 distributed 栈还活着，脚本会拒绝再次启动

自动带 prompt 的例子：

```bash
./scripts/run_distributed_stack_graspnet.sh --robot-backend fake --prompt cup
```

真机执行例子：

```bash
./scripts/run_piper_driver.sh
```

另一个终端：

```bash
./scripts/run_distributed_stack_graspnet.sh --robot-backend ros2 --prompt cup --execute --confirm --enable-pregrasp
```

如果要让 `robot_executor` 的位姿执行走 ROS 侧 MoveIt IK，再开一个终端：

```bash
./scripts/run_piper_moveit_ik.sh
```

然后用：

```bash
./scripts/run_distributed_stack_graspnet.sh --robot-backend ros2 --pose-execution-mode moveit_ik --prompt cup
```

当前比赛瓶子真机默认执行策略是 `execution_strategy=center_horizontal`：

- `use_object_center_contact=true` 时使用 YOLOv8 实例中心射线与 GraspNet 有效接触深度融合中心点，并保留原计划的 Z/人工补偿；透明瓶的实例平均深度不会直接用于目标 Z。
- 实例中心与候选接触点偏差超过 `object_center_contact_max_offset_m=0.08` 时拒绝执行，需先检查瓶子是否直立及分割深度是否正常。
- 执行器使用 `[180, 85, -90] deg` 水平夹爪姿态，根据接触点和工具偏移重新求解 `link6` 位置。
- 运动顺序为：先抬到安全高度，再高处横移/调平，按 `top_down_vertical_step_mm=80` 分段垂直下降，闭爪后垂直抬升 80 mm。典型下降只下发约 3 个目标，减少偏置瓶位附近的反复停顿和姿态微调，最终抓取落点不变。
- `top_down_max_speed_percent=100` 仅作为 Piper 驱动合法范围守卫；实际速度由 Dashboard Speed 同步设置，不再固定封顶为 `5%`。首次真机验证仍建议使用 `5%`。
- MoveIt IK 模式下每个路点只向 `/joint_ctrl_single` 发布一次；等待到达时不再以轮询周期重复下发同一目标，避免 Piper 驱动反复执行 `MotionCtrl_2 + JointCtrl` 造成瓶口附近的细碎修正。
- `center_horizontal_follow_target_azimuth=true` 时，夹爪 yaw 会随瓶子接触中心相对 base 的方位角变化。参考方位角为 `90 deg`、参考夹爪 yaw 为 `-90 deg`，调整量受 `center_horizontal_max_yaw_adjust_deg=45` 限制。该策略减少偏置瓶位下腕部为维持全局固定 yaw 产生的大幅反向补偿；TCP 目标会按自适应姿态重算，瓶子中心落点不变。
- 当前比赛配置不启用姿态回退，避免失败时切回已知会斜向下压瓶子的旧姿态。
- 兼容策略 `safe_top_down` 仍保留；取消 Dashboard 的“瓶子中心水平抓取”后可切回该路径。
- 修改抓取策略或 `robot_grasp_msgs/msg/GraspPlan.msg` 后必须重新构建并重启整套 distributed 栈。

相关参数在 `config/distributed/robot_executor.params.yaml`：

- `top_down_rpy_deg`
- `top_down_rpy_variants_deg`（扁平 RPY 列表，每 3 个数为一组，首选姿态失败后依次尝试）
- `top_down_min_safe_z_mm`
- `top_down_min_target_z_mm`（最终重算后的 `link6` 目标高度下限；低于该值直接拒绝）
- `top_down_approach_height_mm`
- `top_down_lift_height_mm`
- `top_down_lateral_step_mm`
- `top_down_vertical_step_mm`
- `top_down_max_speed_percent`
- `center_horizontal_follow_target_azimuth`
- `center_horizontal_reference_azimuth_deg`
- `center_horizontal_max_yaw_adjust_deg`

如果机械臂已经由人工分段操作并确认处于观察位，可临时设置
`grasp_pipeline.skip_observation_move=true`，让下一轮任务跳过冗余观察位运动、直接读取当前状态并采图。
该参数默认关闭，不能用于尚未确认姿态的自动启动流程。

启用 distributed precenter：

```bash
./scripts/run_distributed_stack_graspnet.sh --robot-backend ros2 --prompt cup --execute --confirm --precenter
```

注意：

- `robot_backend=ros2` 但不带 `--execute` 时，当前 orchestrator 仍会先走 observation pose
- 所以任何 `ros2` 真机联调前，都先单独启动 `run_piper_driver.sh`，确认 `/arm_status` 和 `/enable_srv`
- 如果使用 `moveit_ik`，还要先确认 `run_piper_moveit_ik.sh` 已启动并暴露 `/compute_ik`

### 4.2 做健康检查

终端 B：

```bash
cd /home/ybw/piper_grasp_project/source
./scripts/ros2_system.sh service call /grasp_pipeline/probe std_srvs/srv/Trigger "{}"
```

如果返回成功，说明至少已经验证：

- 四个节点存在
- orchestrator 能打通内部服务
- `robot_executor/get_state` 至少能返回当前后端状态

### 4.2.1 当前这台机器的桌面几何参数

2026-07-11 机械臂改装到车底盘后，已按新安装几何更新：

- `base_link` 离地 `339 mm`
- 操作平面离地 `500 mm`
- `table_z_m = (500 - 339) / 1000 = 0.161`

当前 `workspace_x/y/z` 仍是历史值，尚未按车体边界重新验证，所以不应将它们视为已验收的防碰工作空间。

2026-04-28 桌面安装时的历史回归记录（不适用于当前底盘安装）：

- `table_z_m=0.0`
- `workspace_z=[0.0, 0.60]` 或更高下界

会把真实桌面上的杯子候选稳定过滤成 `no_candidate`。

当时在旧桌面环境能放出有效候选的一组参数是：

- `table_z_m = -0.17`
- `min_gripper_table_clearance_m = 0.03`
- `workspace_z = [-0.15, 0.60]`

这组值已被当前底盘安装的 `table_z_m=0.161` 取代；其中 workspace 仍待重新测量。

如果后面更换桌面高度、底座安装高度或相机安装姿态，优先改这几个参数，而不是先怀疑 YOLOv8-seg 或 GraspNet。

2026-06-07 补充：当前默认候选策略已改成“宽松候选、严格执行前验证”：

- `max_approach_angle` 默认放宽到 `180deg`，不再把 top-down 当作强过滤条件。
- `min_grasp_score` 默认降到 `0.01`，`max_grasp_center_offset_m` 默认放宽到 `0.35m`。
- `max_reachable_rotation_delta_deg` 默认放宽到 `180deg`，并默认允许 `180deg` 等价抓取姿态。
- 若分割器返回 0 个实例但全场景 GraspNet 有结果，会使用全场景 grasp 作为 pseudo-instance 兜底，继续交给 planner / robot validation。
- `execute=true` 时，同一个 candidate 会尝试多个 wrist-roll / 180deg 姿态变体，通过 robot validation 的第一个计划才会进入确认或执行。
- 姿态变体会优先尝试机器人友好的 fallback RPY：当前 observation 姿态、`(0, 120, 0)`、`(180, 60, 180)`，再尝试 GraspNet 原始姿态变体。
- fallback RPY 会重新计算该姿态下的工具接触补偿和所有执行 waypoint，避免姿态通过 IK 但 TCP 目标仍沿用旧补偿导致横向偏抓。
- 候选收集阶段会同步用 fallback RPY 做 workspace / pose-floor 可行性检查；这样原始 GraspNet 姿态触地或越界时，候选仍有机会进入 robot validation。
- 非扁平物体候选排序优先靠近实例中心，其次 GraspNet score，最后才是 approach angle；扁平物体模式仍保留边缘优先。
- 为避免现场 IK validation 长时间阻塞，distributed 默认只对前 `6` 个 candidate、每个 candidate 前 `4` 个姿态变体做 robot validation；对应参数是 `robot_validation_candidate_limit` 和 `robot_validation_variant_limit`。
- `robot_executor` 的 `moveit_ik_timeout_s` 默认设为 `1.5s`，用于快速跳过明显不可达的 IK 姿态。

如果需要回到更保守的筛选，可以通过 `extra_cli_args` 或单节点 CLI 显式设置：

```bash
--max-approach-angle 90
--min-grasp-score 0.05
--max-grasp-center-offset-m 0.20
--max-reachable-rotation-delta-deg 120
--no-allow-180deg-equivalent-grasp
```

另外，当前 distributed `robot_executor` 默认还带一个观察位稳定窗口：

- `pose_goal_hold_s = 0.8`

作用：

- 在 `moveit_ik` 首次判定“到位”之后，继续短时间重发同一目标
- 避免机械臂刚到 observation pose 就立刻发生短暂漂移，导致下一步采图视角不稳定

### 4.2.2 当前机器最短真机联调步骤

如果你要“一键拉起真机栈并自动打第一枪”，现在最推荐先用：

```bash
cd /home/ybw/piper_grasp_project/source
./scripts/run_live_grasp_one_click.sh
```

默认行为：

- 默认 prompt 为 `cup`
- 自动启动 driver、MoveIt IK、distributed stack 和 RViz
- 自动等待 `/compute_ik` 和 `/grasp_pipeline/probe`
- 只在 readiness 通过后才调用 `run_pipeline_service.sh`
- 首次任务触发后整套栈保持常驻

常用参数：

```bash
./scripts/run_live_grasp_one_click.sh bowl
./scripts/run_live_grasp_one_click.sh cup --execute
./scripts/run_live_grasp_one_click.sh cup --enable-pregrasp --precenter
./scripts/run_live_grasp_one_click.sh cup --no-rviz
```

如果只想“一条命令打一轮任务，结果落盘后自动收掉本 wrapper 启动的进程”，用：

```bash
./scripts/run_one_grasp_task.sh cup --robot-backend fake --plan-only --no-rviz
./scripts/run_one_grasp_task.sh cup --robot-backend ros2 --no-rviz
./scripts/run_one_grasp_task.sh cup --robot-backend ros2 --execute --no-rviz
```

说明：

- 这个脚本等价于 `run_live_grasp_one_click.sh --once`
- `fake + --plan-only` 是推荐的无硬件安全验证入口
- `ros2` 默认只做观察、感知、规划和 robot validation
- 真机最终执行必须显式加 `--execute`；ros2 执行会自动进入 confirm 等待，需要另开终端调用 `./scripts/confirm_pipeline_service.sh`
- 脚本只停止自己启动的进程；如果复用了外部已运行的 driver / MoveIt / distributed 栈，不会主动杀掉它们

如果要一键清理当前真机联调进程：

```bash
./scripts/clear_live_grasp_nodes.sh --dry-run
./scripts/clear_live_grasp_nodes.sh
```

关键语义：

- 新脚本不会把 `prompt` 通过 distributed `--prompt` 做启动即执行
- 它会先等待 `/compute_ik` 与 `/grasp_pipeline/probe` 都 ready
- 如果整套 driver / MoveIt IK / distributed / RViz 已经在跑，它会直接复用现有进程
- 如果只残留半套 distributed 节点，它会继续拒绝，避免新旧节点混跑
- wrapper 日志落到 `logs/one_click/<timestamp>/`
- distributed 子节点日志仍按原路径落到 `ros_ws/log/distributed/<timestamp>/`

如果你要手动分终端控制每个组件，继续按下面 4 终端方式操作：

 推荐固定开 4 个终端：

1. 终端 A，启动驱动

```bash
cd /home/ybw/piper_grasp_project/source
./scripts/run_piper_driver.sh
```

2. 终端 B，启动 MoveIt IK

```bash
cd /home/ybw/piper_grasp_project/source
./scripts/run_piper_moveit_ik.sh
```

3. 终端 C，启动 distributed 主线

```bash
source /home/ybw/piper_grasp_project/source/scripts/ros_env_graspnet.sh
cd /home/ybw/piper_grasp_project/source
./scripts/run_distributed_stack_graspnet.sh --robot-backend ros2 --pose-execution-mode moveit_ik
```

4. 终端 D，健康检查和触发任务

```bash
cd /home/ybw/piper_grasp_project/source
./scripts/ros2_system.sh service call /grasp_pipeline/probe std_srvs/srv/Trigger "{}"
./scripts/run_pipeline_service.sh cup
./scripts/show_last_run_artifact.sh
```

### 4.2.3 当前结果怎么读

当前这台机器上，`show_last_run_artifact.sh` 的结果大致按下面理解：

- `status=ok` 或 `status=completed`
  - 当前帧已经有有效候选
  - 可以继续做执行链路验证
- `status=no_candidate` 且 diagnostics 里含 `no grasp after mask filtering`
  - 当前帧的问题在 mask 和 scene grasp 的重叠质量
  - 优先重拍一帧、调整 prompt、换物体位置或观察位
- `status=no_candidate` 且 diagnostics 里含 `workspace`、`table_z_m`、`tool_lowest_z`
  - 说明桌面几何参数没对上
  - 先确认 distributed 栈是否吃到了当前 YAML，再考虑改参数
- `status=failed`
  - 先看 `final_result.json.summary`
  - 如果是启动类错误，优先检查：
    - `/arm_status`
    - `/compute_ik`
    - `yolov8n-seg.pt`
    - `can0`
  - 如果 summary 里已经出现 `selected fallback candidate[...]`
    - 说明视觉 top-1 被 robot validation 拒绝了
    - 下一步不要只看最终 plan，要继续看 `candidate_validation.json`

补充说明：

- 当前 `show_last_run_artifact.sh` 默认只打印：
  - `request.json`
  - `cycles.json`
  - `final_result.json`
- 如果要排查 top-1 为什么被 IK 拒绝，还要手动打开：
  - `logs/distributed_runs/<run_id>/candidate_validation.json`
  - `logs/distributed_runs/<run_id>/execution_trace.json`

浏览器操作台 / 可视化 dashboard：

```bash
./scripts/run_grasp_dashboard.py
```

打开 `http://127.0.0.1:8765`，可以先点“启动真机栈”拉起 Piper driver、MoveIt IK 和分布式抓取节点；这个启动动作不会自动触发抓取。等 pipeline / camera / vision / executor 状态变绿后，可以直接输入 prompt、设置速度、触发 `run`、`confirm`、`reject`、`stop`、`probe`。页面里的 `X/Y/Z 补偿 mm` 会在每次抓取前作为 base 坐标下的目标位姿微调下发，适合先验证 2-5mm 级系统偏差。也可以同时回看分割图、GraspNet 投影、候选验证、规划路径、执行轨迹、当前进程和最近节点日志。

比赛瓶子使用默认勾选的“瓶子中心水平抓取”，并保持 X/Y/Z 补偿为 `0`。点“直接抓取”会自动下发 `use_object_center_contact=true`、`execution_strategy=center_horizontal` 并触发一次完整任务。默认不勾选“抓后交接并回 Home”，因此成功后机械臂会在抬升位保持瓶子；停止或失能前必须先放置物体并回 Home。

当前默认执行速度为 `5%`，对应参数是 `/grasp_pipeline speed` 和 `/robot_executor default_speed_percent`。Dashboard 顶部的 Speed 输入框会在触发 run 前同步设置这两个参数，MoveIt IK 会把该值写入 `/joint_ctrl_single.velocity[6]`，Piper 驱动最终用它设置 `MotionCtrl_2` 速度百分比。建议从 `5%` 开始逐步调高。

分布式栈启动默认不再做 camera / vision warmup，以便尽快进入可触发状态；如果要提前加载相机和模型，可以在启动栈时显式加：

```bash
./scripts/run_distributed_stack_graspnet.sh --robot-backend ros2 --pose-execution-mode moveit_ik --warmup
```

### 4.3 触发一次任务

最推荐方式：

```bash
./scripts/run_pipeline_service.sh cup
```

这条脚本会做两件事：

1. 设置 `/grasp_pipeline` 的 `prompt`
2. 调用 `/grasp_pipeline/run`

重要说明：

- `/grasp_pipeline/run` 目前仍然是 `std_srvs/srv/Trigger`
- 它不会从 request 里拿 prompt
- 不要把“service call 成功”理解成“已经抓取成功”

如果你一定要手动执行，顺序必须是：

```bash
./scripts/ros2_system.sh param set /grasp_pipeline prompt cup
./scripts/ros2_system.sh service call /grasp_pipeline/run std_srvs/srv/Trigger "{}"
```

### 4.4 先出 plan，再人工确认执行

先启动：

```bash
./scripts/run_distributed_stack_graspnet.sh --robot-backend ros2 --prompt cup --execute --confirm
```

语义如下：

1. orchestrator 先执行 `capture -> analyze -> plan`
2. 如果有有效候选，状态进入 `awaiting_confirmation`
3. 此时不会立刻调用执行器
4. 外部显式确认后才真正执行

确认执行：

```bash
./scripts/confirm_pipeline_service.sh
```

拒绝这次计划：

```bash
./scripts/reject_pipeline_service.sh
```

### 4.5 看运行结果

查看当前 topic：

```bash
./scripts/ros2_system.sh topic echo /grasp_pipeline/status
./scripts/ros2_system.sh topic echo /grasp_pipeline/result_json
./scripts/ros2_system.sh topic echo /grasp_pipeline/diagnostics
```

更推荐的回看方式：

```bash
./scripts/show_last_distributed_snapshot.sh
./scripts/show_last_run_artifact.sh
```

如果你要专门排查“为什么 top-1 没被执行”，建议固定按下面顺序看：

1. `final_result.json`
   - 看最终 `status`
   - 看 `summary` 是否出现 `selected fallback candidate[...]`
   - 看 `candidate` 和 `plan`，确认最终落到哪个 fallback
2. `candidate_validation.json`
   - 找 `candidate_index=0`
   - 看 `selection_result`
   - 看 `robot_validation_stage`
   - 看 `ik_error_type`
   - 看 `ik_error_message`
   - 看 `waypoint_results`
3. `execution_trace.json`
   - 只有最终真的执行了才需要看
   - 它表示执行层逐步下发后的真实回读，不是 top-k 的 IK 预校验

字段口径：

- `translation_camera_m`
  - 这是视觉候选在相机系下的位置，不等于真正失败的机械臂位姿
- `target_base_m / pregrasp_base_m / grasp_base_m / retreat_base_m`
  - 这是 candidate 经过规划后的 base 系执行位姿
  - 当前语义是：
    - `grasp`: 预接触位
    - `target`: 实际接触 / 闭爪位
    - `retreat`: 抓后抬升位
- `robot_validation_stage`
  - 表示当前 candidate 第一个失败的 waypoint
- `ik_error_type`
  - 当前至少会区分：
    - `timeout`
    - `no_ik_solution`
    - `ik_error`
- `selection_result`
  - 当前至少会看到：
    - `rejected_by_robot_validation`
    - `selected_for_execution`

当前 waypoint 顺序：

- `enable_pregrasp=false` 时，按 `grasp -> target -> retreat` 校验
- `enable_pregrasp=true` 时，按 `pregrasp -> grasp -> target -> retreat` 校验
- 真机执行时，闭爪发生在 `target` 到位之后，不再发生在 `grasp` 之后

当前关键状态和结果 topic 已使用 transient-local QoS：

- `/grasp_pipeline/status`
- `/grasp_pipeline/summary`
- `/grasp_pipeline/diagnostics`
- `/grasp_pipeline/result_json`
- `/vision_worker/status`
- `/vision_worker/summary`
- `/vision_worker/result_json`
- `/robot_executor/status`
- `/robot_executor/result_json`
- `/camera_server/status`
- `/camera_server/latest/color`
- `/camera_server/latest/depth`
- `/camera_server/latest/camera_info`

## 5. RViz 可视化

### 5.1 打开方式

```bash
cd /home/ybw/piper_grasp_project/source
./scripts/open_distributed_rviz.sh
```

这会加载：

- `rviz/distributed_grasp_pipeline.rviz`

### 5.2 当前应该订阅的话题

分布式模式下，RViz topic 由 `vision_worker_node` 发布，不是 `/grasp_pipeline/rviz/*`。

当前重点 topic：

- `/vision_worker/rviz/scene_pointcloud`
- `/vision_worker/rviz/instance_pointcloud`
- `/vision_worker/rviz/candidate_grasps`
- `/vision_worker/rviz/selected_grasp`
- `/vision_worker/rviz/plan_waypoints`
- `/vision_worker/rviz/candidate_markers`
- `/vision_worker/rviz/selected_grasp_markers`
- `/vision_worker/rviz/plan_markers`
- `/vision_worker/rviz/camera_transform`
- `/tf`

### 5.3 推荐的 RViz 设置

建议按这个顺序使用：

1. 默认 `Fixed Frame` 先设为 `camera_color_optical_frame`
2. 先启用：
   - `scene_pointcloud`
   - `instance_pointcloud`
   - `candidate_markers`
   - `selected_grasp_markers`
3. 跑过一次 pipeline 且 `/tf` 正常后，再切 `Fixed Frame=base_link`

补充说明：

- 如果结果是 `no_candidate`，`selected_grasp` 和 `plan_*` 为空是正常的
- 即使 `no_candidate`，点云 topic 仍应保留最后一次感知结果

## 6. 当前控制面语义

当前对外控制面：

- `/grasp_pipeline/run`
- `/grasp_pipeline/probe`
- `/grasp_pipeline/stop`
- `/grasp_pipeline/confirm`
- `/grasp_pipeline/reject`

当前系统语义：

- `execute=false`
  - 只做 `capture -> analyze -> plan/no_candidate`
- `execute=true`
  - executor 执行：
    - 可选 `pregrasp`
    - `grasp`
    - `retreat`
    - 可选 `handoff`
    - release
    - 可选 `home`
- `precenter=true`
  - orchestrator 会在最终规划前增加一个 centering loop
- `execute=true` 且 `confirm=true`
  - 先进入 `awaiting_confirmation`
  - 显式 confirm 后才执行

## 7. 推荐部署拓扑

### 7.1 同机验证

当前最稳：

- 一台机器起四个节点
- `robot_executor` 使用 `fake`
- 先验证 `capture -> analyze -> no_candidate/plan`

### 7.2 多机拆分

推荐角色：

1. Robot Host
   - `robot_executor_node`
   - `piper_ros`
   - 可选 `camera_server_node`
2. Vision Host
   - `vision_worker_node`
   - GPU
3. Operator Host
   - `pipeline_orchestrator_node`
   - RViz
   - 触发脚本

跨机前先保证：

- `ROS_DOMAIN_ID` 一致
- DDS 网卡配置固定
- 时间同步正常

## 8. 常见问题

### 8.1 `_rclpy_pybind11` 导入失败

原因通常是：

- 把 `graspnet` Python 3.10 overlay 和 system ROS CLI 混在同一个 shell

处理方式：

- 不要在 `source scripts/ros_env_graspnet.sh` 后直接执行 `ros2 ...`
- 统一改用：
  - `./scripts/ros2_system.sh ...`

### 8.2 `/grasp_pipeline/probe` 卡住或超时

优先检查：

- `ros_ws/log/distributed/<timestamp>/grasp_pipeline.log`
- `ros_ws/log/distributed/<timestamp>/camera_server.log`
- `ros_ws/log/distributed/<timestamp>/vision_worker.log`
- `ros_ws/log/distributed/<timestamp>/robot_executor.log`

### 8.3 任务被接受，但最后是 `no valid grasp candidate found`

这不是系统挂了，而是一次“正常完成但没有候选”的运行。

常见原因：

- prompt 不在视野里
- YOLOv8-seg 没有检测到 prompt 对应的 COCO 实例
- GraspNet 候选被筛光

### 8.4 `capture failed`

优先检查：

- RealSense 是否在线
- 相机是否被别的进程占用
- `camera_server.log`

### 8.5 真机执行没反应

优先检查：

- `piper_ros` 单臂节点是否已启动
- `can0` 是否存在
- `/end_pose` 是否持续更新
- `robot_executor` 是否切到了 `ros2` 后端

2026-04-08 本机已验证的最小真机链路：

- `./scripts/run_piper_driver.sh`
- 单独启动 `robot_executor_node --ros-args -p robot_backend:=ros2 -p auto_enable:=false`
- 第一次 `/robot_executor/get_state` 直接成功返回当前位姿和 `arm_status`

这说明当前如果“完全没反应”，应优先排查运行环境、节点未启动、CAN 或 topic 无数据，而不是继续假设 `get_state` 首次调用本身有已知缺陷。

### 8.6 `rviz2: command not found`

这是 RViz 没装，不是 pipeline 本身的问题。

## 9. 当前建议

如果你的目标是“先把系统用清楚”，建议固定顺序：

1. 跑 `./scripts/run_distributed_stack_graspnet.sh`
2. 跑 `/grasp_pipeline/probe`
3. 用 `./scripts/run_pipeline_service.sh cup` 触发任务
4. 用 `./scripts/show_last_distributed_snapshot.sh` 回看
5. 最后再开 RViz 看 `/vision_worker/rviz/*`

如果你的目标是“开始实机闭环”，建议顺序：

1. 保持分布式模式不变
2. 只把 `robot_executor` 从 `fake` 切到 `ros2`
3. 先把 `/robot_executor/get_state` 当作已通过基线，再单独验证 `/execute_named_pose`
4. 最后再打开 `--execute`

如果你切到了 `--pose-execution-mode moveit_ik`，还要补一条：

5. 先确认 `/compute_ik` 已存在，再验证 `/execute_named_pose`

当前 2026-04-28 的新证据：

- `run_piper_moveit_ik.sh` 已能稳定拉起最小 `move_group`
- `/compute_ik` 类型确认是 `moveit_msgs/srv/GetPositionIK`
- 本地 joint-limit override 对齐 live pose 后，对“当前回读位姿”和“z + 10 mm”做纯 IK 探针都已返回成功
- `robot_executor` 的 MoveIt IK 路径已补上独立 `ReentrantCallbackGroup`，避免 service 回调里收不到 `/joint_states_feedback`
- `robot_executor` 的 MoveIt IK 路径现在会在等待到位期间持续重发同一组 `joint_ctrl_single` 关节目标，避免“单发一帧但实机不动”
- 默认 `moveit_ik_timeout_s` 已提高到 `5.0`
- 真机上建议在 `run_piper_moveit_ik.sh` 启动后先预留约 `10s` 再发第一条 `moveit_ik` 命令
- 最新一次最小真机验证里，机械臂最终从 `z=169.502 mm` 抬到了 `z=178.684 mm`，距离原始 `+10 mm` 目标 `179.502 mm` 还差约 `0.818 mm`
- 隔离命名空间 `/verify_moveit` 的端到端验证里，`/execute_named_pose` 已让真实机械臂从 `z=182.751 mm` 抬到 `z=186.104 mm`

如果你现在看到的是 `MoveIt IK request timed out: /compute_ik`，优先检查：

- `run_piper_moveit_ik.sh` 起稳后是否已经预留了 warm-up 时间
- `robot_executor` 是否已使用带 `ReentrantCallbackGroup` 的最新安装产物
- `moveit_ik_timeout_s` 是否仍被外部参数覆盖成过小的值
