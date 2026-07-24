# Current Status

更新时间：2026-04-29

本文档记录 `robot_grasp_ros2` 当前已经完成的工作、已经验证的基线、仍未完成的部分，以及接下来继续推进时最应该依赖的事实。

这份文档的定位是“接手点”，不是完整使用手册。运行命令和排障步骤请看 `docs/DISTRIBUTED_RUNBOOK.md`。

## 1. 当前主线判断

截至 2026-04-08，这个仓库的主线已经明确为：

- 分布式模式是推荐主线
- 单节点模式保留为兼容 / 对照 / 调试路径
- 这台机器的稳定运行形态是：
  - system Python 跑 ROS 节点
  - `piper` conda 环境跑重型感知
  - `camera_server` 和 `vision_worker` 通过外部 worker 子进程访问 D435 / YOLOv8-seg / GraspNet
- 当前最稳定的验收目标不是“真机抓取成功”，而是：
  - `observation -> get_state -> capture -> analyze -> result_json -> rviz topics`
- `fake` 后端完整数据流已经证明是通的

一句话总结：

这个项目现在已经不是“从零开始迁移”，而是“基于已跑通的 distributed fake 基线，继续把感知命中率、结果结构和真机切换条件做扎实”。

2026-06-07 补充：

- 候选筛选默认改为宽松模式：top-down approach angle、分数、中心偏移、腕部旋转预筛都不再作为强收窄条件。
- `vision_worker` 现在合并两条候选来源：
  - 实例点云上的 GraspNet 预测
  - 全场景 GraspNet 预测再按 mask/depth 粗过滤
- 如果分割器返回 0 个实例但全场景 GraspNet 已经有结果，会临时使用全场景 grasp 作为 pseudo-instance 兜底，继续交给 workspace / robot validation 筛选。
- `execute=true` 的 robot validation 会对同一个 candidate 尝试多个 wrist-roll / 180deg 姿态变体，避免第一个姿态 IK 不通就直接报 `no robot-reachable grasp candidate found`。
- 非扁平物体的候选排序现在优先靠近实例中心，再看 GraspNet score，最后才看 approach angle；宽松 top-down 模式下不再让角度把明显偏离物体中心的候选排到最前。
- robot-friendly fallback RPY 会重新计算工具接触补偿和 `target/pregrasp/grasp/retreat`，不能只替换姿态复用旧位置。
- 候选收集阶段也会用 robot-friendly fallback RPY 做 workspace / pose-floor 可行性检查，避免 GraspNet 原始姿态不安全时过早过滤掉本可由 fallback 姿态执行的候选。

## 2. 当前范围与硬约束

2026-07-23 红黄蓝标准物块感知：

- 新增固定 3D 打印物块的 HSV 颜色实例分割，支持 `red block`、`yellow block`、
  `blue block` 以及中文颜色物块 prompt。
- `物块` / `color block` 会同时检测红、黄、蓝实例，后续点云、GraspNet、规划与
  robot validation 继续复用现有分布式链路。
- 指定颜色未检测到时禁止退化到全场景 grasp pseudo-instance，避免错误执行。
- Dashboard 新增三种物块快捷按钮，并自动关闭瓶子专用的中心水平抓取模式。
- 首轮真机验证必须使用“规划后确认”，不能直接执行。

2026-07-11 底盘安装后的操作平面高度已重新测量：

- `base_link` 到地面：`339 mm`
- 操作平面到地面：`500 mm`
- 按 `base_link` Z 轴向上计算，操作平面为 `+161 mm`
- distributed 配置已更新为 `table_z_m: 0.161`
- 底盘环境的 `workspace_x/y/z` 尚未重新测量，当前值不代表新环境安全边界

2026-07-13 工具接触点补偿修正：

- `safe_top_down` 不再只替换计划姿态并复用旧的 `link6` XYZ。
- `GraspPlan` 会携带目标接触点和 `link6 -> 接触点` 的工具系偏移；执行器按最终 `top_down_rpy_deg` 重新计算 `link6` 目标。
- 缺少上述工具接触几何的旧计划会被 `safe_top_down` 拒绝，必须重新生成计划，避免错误 TCP 补偿导致夹爪下探过深。
- 最终重算后的 `link6` Z 还必须通过 `top_down_min_target_z_mm`，当前默认 `300 mm`；该值是临时安全下限，不替代后续工具接触点实测。
- 该修正不改变手眼标定，也不等价于在 base Z 方向固定增加某个毫米数。

2026-07-13 左侧比赛工作区观察位：

- 工作区位于车体左侧，已通过真机低速运动确认基座关节转向正确。
- 当前实机确认的观察位已固化为 `[0.0, 35.5, 491.1, 180.0, 67.77, -89.97]`（`mm/deg`）。
- 机械臂失能前必须先回到 Home；观察位只用于采图，不作为失能位姿。

2026-07-13 分段运动夹爪保持修正：

- `robot_executor` 不再在每次 service / 分段位姿调用前重复调用 Piper 使能。
- 同一连接周期只在首次连接时自动使能；重复使能会导致真机夹爪短暂回零，表现为每段运动前先闭合再恢复目标开度。
- 明确急停、失能或断开连接后会清除内部使能状态，下一次恢复时才重新使能。
- MoveIt 关节命令会在张开或闭合确认后持续携带对应夹爪目标，避免后续位姿运动采用旧反馈值覆盖夹爪命令。
- 夹爪目标保存在 `robot_executor`，不依赖 MoveIt 执行器是否已经初始化；首次位姿运动延迟创建 MoveIt 时也必须先恢复该目标，再发布机械臂关节命令。

2026-07-13 比赛瓶子自动流程：

- 新增 `grasp_pipeline.use_object_center_contact`：使用 YOLOv8 实例中心射线与 GraspNet 有效接触深度融合出瓶子中心，同时保留规划器生成的 Z 补偿和 base 人工偏置；不直接采用透明瓶容易穿透到背景的实例平均深度。
- `object_center_contact_max_offset_m` 默认 `0.08 m`；实例中心与候选接触点偏差过大时自动拒绝，避免倾倒物体或异常深度生成危险中心目标。
- 真机分布式默认执行策略改为 `center_horizontal`，固定使用已验证的 `[180, 85, -90] deg` 水平夹爪姿态。
- 自动路径为“观察 -> 感知 -> 高位调平/横移 -> 25 mm 级分段下降 -> 闭爪 -> 抬升 80 mm”；默认抓后保持夹持，不自动交接或回 Home。
- Dashboard 默认勾选“瓶子中心水平抓取”，X/Y/Z 人工补偿默认清零；“直接抓取”会同步下发中心模式与执行策略后触发 `/grasp_pipeline/run`。
- 需要停止/失能时仍必须先显式回 Home，不能把“抓后保持”当作可直接失能状态。

2026-07-17 偏置瓶位运动与网页速度修正：

- `center_horizontal` 的垂直分段由 `25 mm` 调整为 `80 mm`。典型约 `235 mm` 下降由约 10 次目标下发减少为 3 次，减少瓶子偏离机械臂正前方时在瓶口附近反复停顿和修姿态；最终接触点、水平夹爪姿态、TCP 补偿和桌面安全检查保持不变。
- `top_down_max_speed_percent` 由固定 `5%` 上限改为 `100%` 配置守卫，Dashboard 的 Speed 现在会真实传递到 Piper 驱动的 `MotionCtrl_2` 速度百分比，不再出现网页数值变化但实际始终为 `5%` 的情况。
- Dashboard、pipeline 和 executor 的默认速度统一为 `5%`。真机初次复测建议使用 `5%`，确认路径后再逐步提高，避免直接使用高速度。
- 真机复测确认 `80 mm` 配置已把最新执行轨迹降为 4 个下降路点，但 MoveIt 等待到达期间仍以 `50 ms` 周期重复发布同一个 `/joint_ctrl_single` 目标，驱动会重复进入 `MotionCtrl_2 + JointCtrl`，肉眼仍可能看到多次细碎修正。现已改为每个 MoveIt 路点只发布一次，等待阶段仅轮询位姿与到达状态。
- 偏置瓶位不能继续固定使用全局 yaw `-90 deg`：底座朝偏置目标转动时，腕部会为维持绝对方向做反向补偿。`center_horizontal_follow_target_azimuth=true` 后，夹爪 yaw 按目标接触点在 base 平面的方位角同步调整；正对左侧工作区（方位角 `90 deg`）时仍保持 `-90 deg`，偏置目标则由底座和夹爪共同转向。姿态调整在高处横移前完成，下降阶段保持固定姿态，并按新姿态重新计算 link6，因此 TCP 接触中心不变。

2026-07-13 左侧工作区 top-down 可达性修正：

- `safe_top_down` 保留“安全抬升 -> 高处横移 -> 垂直下降 -> 垂直回升”的执行结构，但不再依赖唯一固定腕部方向。
- `robot_executor` 会先验证 `top_down_rpy_deg`，再验证 `top_down_rpy_variants_deg`；默认补充 yaw `90/0/-90 deg` 三个方向。
- 每个方向必须通过完整 waypoint IK 才能被选中；验证和执行结果会记录 `top_down_rpy_deg` 与 `top_down_variant_attempts`。
- 该策略不会放宽 TCP 接触几何、`top_down_min_target_z_mm` 或桌面安全约束。
- 使用既有瓶子计划做 `execute=false` 真机 IK 扫描时，yaw `180/90/0 deg` 分别在横移或转腕阶段无解，yaw `-90 deg` 的完整 waypoint 全部通过；左侧工作区配置因此将 `[180, 60, -90] deg` 设为首选，其他方向保留为回退。
- 人工已确认机械臂处于观察位时，可临时启用 `skip_observation_move` 跳过冗余命名位姿调用；默认关闭，任务仍会自动进入观察位。

当前主写入区：

- `grasp_ros/robot_grasp_ros2`

默认不改：

- `old/robot_grasp`
- `Agilex-College`
- `robotic_arm_kinematics`
- `piper_ros_humble`

当前硬约束：

- 新工程业务逻辑禁止直接依赖 `piper_sdk`
- 真机控制必须经过 `piper_ros` / ROS2
- 规划层继续使用 `mm/deg`
- ROS2 适配层负责 `mm/deg <-> m/rad`
- 新增 topic / service / 参数 / 运行方式时必须同步更新 `docs/`

## 3. 当前可用架构

### 3.1 分布式主线

当前推荐主线由四个节点组成：

1. `/grasp_pipeline`
   - 统一入口、状态机、`run_id`、最终抓取规划
2. `/camera_server`
   - 按需采集 RGBD
3. `/vision_worker`
   - YOLOv8-seg、点云重建、GraspNet、实例筛选、RViz 发布
4. `/robot_executor`
   - `fake` / `ros2` 两种执行后端

### 3.2 单节点兼容模式

- 节点：`grasp_pipeline_node`
- 作用：兼容旧的“一个 node 串行跑完整流程”方式
- 定位：迁移对照和局部调试

## 4. 已完成的工作

### 4.1 迁移与工程化

已经完成：

- `old/robot_grasp` 到 `robot_grasp_ros2` 的工程骨架迁移
- `perception/`、`grasping/`、`utils/` 的主逻辑迁移
- `RobotArmClient` 抽象层建立
- `FakeRobotArmClient` 与 `Ros2PiperClient` 双后端接入
- `run_grasp_pipeline_ros2.py` 与 `grasp_pipeline_node` 兼容入口保留

### 4.2 分布式主链路

已经完成：

- `pipeline_orchestrator_node`
- `camera_server_node`
- `vision_worker_node`
- `robot_executor_node`
- `external_inference_worker.py`
- `external_camera_capture_worker.py`
- 内部 typed service 串联
- 对外控制面：
  - `/grasp_pipeline/run`
  - `/grasp_pipeline/probe`
  - `/grasp_pipeline/stop`
  - `/grasp_pipeline/confirm`
  - `/grasp_pipeline/reject`

### 4.3 规划与执行语义

已经完成：

- `enable_pregrasp` 贯通 planner 和 executor
- `move_home_after` 逻辑接回 executor
- `handoff_pose` / `home_pose` 分布式执行语义接回
- distributed `precenter` loop 接回 orchestrator
- `execute=true` 且 `confirm=true` 时可进入 `awaiting_confirmation`

### 4.4 RViz 可视化

已经完成：

- 场景点云发布
- 实例点云发布
- candidate / selected grasp marker 发布
- plan waypoints / plan markers 发布
- camera transform / TF 发布
- `distributed_grasp_pipeline.rviz` 预配置修正

### 4.5 运行产物与回看能力

已经完成：

- 关键 topic 改成 transient-local QoS
- 新增 `show_last_distributed_snapshot.sh`
- 新增每次 run 的落盘产物：
  - `logs/distributed_runs/<run_id>/request.json`
  - `logs/distributed_runs/<run_id>/cycles.json`
  - `logs/distributed_runs/<run_id>/final_result.json`
  - `logs/distributed_runs/<run_id>/candidate_validation.json`
  - `logs/distributed_runs/<run_id>/execution_trace.json`
- 新增 `show_last_run_artifact.sh`
- 分布式栈重复启动拦截，避免同名 service/topic 混串
- session 内 ROS 日志重定向到工作区，避免默认写 `~/.ros/log`
- 新增 `run_piper_driver.sh`，为本机 Piper 驱动补充 `piper_sdk` 和 `python-can`

### 4.6 本机 Piper 非硬件调试链

2026-04-28 补充确认：

- `piper_description` 的本地交互模型链已经可以在这台机器上拉起
- 当前可正常启动：
  - `joint_state_publisher_gui`
  - `robot_state_publisher`
  - `rviz2`
- 这条链适合做：
  - 模型显示
  - 关节滑块交互
  - 本地 RViz 调试
- 这条链不等价于：
  - 真实 `piper_ros` 驱动
  - Gazebo 物理仿真
  - MuJoCo 仿真

详细步骤和本机限制请看：

- `docs/PIPER_LOCAL_SIM.md`

## 5. 已验证证据

### 5.1 已验证运行模式

验证环境：

- D435 已接入
- `robot_backend=fake`
- 分布式模式
- prompt=`cup`

已验证命令：

```bash
source /home/ybw/piper_grasp_project/source/scripts/ros_env_graspnet.sh
cd /home/ybw/piper_grasp_project/source
./scripts/run_distributed_stack_graspnet.sh --robot-backend fake --prompt cup
```

另一个终端：

```bash
./scripts/ros2_system.sh service call /grasp_pipeline/run std_srvs/srv/Trigger "{}"
```

### 5.2 已验证状态流

日志中已经看到完整状态流：

- `preflight`
- `moving_to_observation`
- `reading_robot_state`
- `capturing_scene`
- `analyzing_scene`
- 终态会落到 `completed` 或 `no_candidate`

这说明：

- orchestrator 能调通 robot state
- camera server 能返回 RGBD
- vision worker 能完成分析并回传结果
- orchestrator 能正常收尾并生成最终结果

### 5.3 已验证结果摘要

2026-04-08 最近一次已验证结果：

- `status=completed`
- `prompt=cup`
- `selected grasp score=0.1655`
- `candidates=7`
- `scene_points=304161`

这说明当前瓶颈已经不在 ROS2 数据流，也不在 Jazzy / conda 兼容性。

### 5.4 已验证回看能力

已经确认下面内容在任务结束后仍可回看：

- `/grasp_pipeline/result_json`
- `/vision_worker/result_json`
- `/camera_server/latest/camera_info`
- `/vision_worker/rviz/scene_pointcloud`
- `logs/distributed_runs/<run_id>/` 下的结构化产物

最近一次快照里，场景点云摘要为：

- `frame_id=camera_color_optical_frame`
- `width=304161`
- `approx_empty=false`

### 5.5 已验证 Piper 真机状态接入与最小运动判定

2026-04-28 已单独验证：

- 本机 `can0` 已经成功拉起到 `bitrate=1000000`
- USB-CAN 适配器已被系统识别为 `gs_usb`
- `./scripts/run_piper_driver.sh` 可在本机正常拉起 AgileX `piper_single_ctrl`
- `/arm_status` 可正常返回
- `/end_pose` 可正常持续发布
- `robot_executor_node` 以 `robot_backend:=ros2` 启动后，`/robot_executor/get_state` 调用可稳定返回成功
- `robot_executor -> piper_ros` 的 `PosCmd` 下发链路已确认打通
- `run_piper_moveit_ik.sh` 可在本机稳定拉起一个最小 `move_group` 包装层
- `/compute_ik` 已确认暴露为 `moveit_msgs/srv/GetPositionIK`

本次验证结论：

- `Ros2PiperClient` 当前的订阅 / service 并发问题已经落地修复
- `robot_executor -> piper_ros` 的只读状态链路已通过真实硬件验证
- `robot_executor` 已新增 `pose_execution_mode:=moveit_ik` 分支，`MovePoseCommand` 可以改走 ROS 侧 MoveIt IK，再下发到 `joint_ctrl_single`
- `MoveItIkExecutor` 现在会把 joint feedback 订阅和 `/compute_ik` client 绑定到独立 `ReentrantCallbackGroup`
- `robot_executor` 的 `moveit_ik` 等待路径现在会在到位前持续重发同一组 `joint_ctrl_single` 关节目标，不再只发单帧
- 本地最小 MoveIt 包装层仍然有效，不依赖 upstream `piper_gazebo`
- 本地 joint-limit override 已经覆盖 live pose 的关节范围差异后，纯 `/compute_ik` 探针已验证：
  - “当前回读位姿”返回 `code=1`
  - “当前位姿 z + 10 mm”返回 `code=1`
- `moveit_ik_timeout_s` 默认值已提高到 `5.0`
- 2026-04-28 真机最小抬升验证：
  - 起点：`z=169.502 mm`
  - 原始目标：`z=179.502 mm`
  - 实际稳定到：`z=178.684 mm`
  - 剩余误差：约 `0.818 mm`
- 2026-04-28 隔离命名空间 `/verify_moveit` 的端到端实机验证：
  - 起点：`z=182.751 mm`

### 5.6 分布式视觉后端

当前分布式与单节点路径统一使用 `YOLOSegmenter`，默认权重为 `yolov8n-seg.pt`。旧分割实现、checkpoint 参数和兼容导出已删除，视觉 worker 不再依赖该旧模型的下载与代理配置。

2026-04-28 已做最新回归验证：

- distributed run：`grasp-1777364633494`
- 结果：`status=no_candidate`
- 关键结论：
  - 已经越过原先的 `download_ckpt_from_hf()` / proxy 报错
  - `vision_worker` 已正常完成 `analysis completed`
  - 当前需要观察的是候选是否被 YOLOv8 mask / workspace / table clearance 规则过滤

2026-04-28 继续排查后已确认：

- distributed 主线之前缺少正式的几何调参入口：
  - `table_z_m`
  - `min_gripper_table_clearance_m`
  - `workspace_x/y/z`
- 同时单节点入口的 `workspace_z` 默认值已经比旧基线更紧：
  - 新入口此前默认：`(0.04, 0.60)`
  - 旧基线默认：`(0.00, 0.60)`

当前已落地修复：

- `PipelineOrchestratorNode` 现在正式声明并透传：
  - `table_z_m`
  - `min_gripper_table_clearance_m`
  - `workspace_x`
  - `workspace_y`
  - `workspace_z`
- `extra_cli_args` 现在改为可正常动态设置的 `string array`
- `run_grasp_pipeline_ros2.py` 的 `--workspace-z` 默认值已恢复到旧基线 `0.00 0.60`

2026-04-28 旧桌面安装时的真机分布式几何回归（历史记录，不适用于当前底盘安装）：

- 先前失败 run：`grasp-1777365047477`
  - 仍使用默认 `table_z_m=0.0`
  - 结果：`status=no_candidate`
- 运行时把 `/grasp_pipeline` 参数改成：
  - `table_z_m = -0.17`
  - `workspace_z = [-0.15, 0.60]`
  - `min_gripper_table_clearance_m = 0.03`
- 新 run：`grasp-1777365305801`
  - 结果：`status=ok`
  - `candidate_count=4`
  - `selected grasp score=0.5147`
  - `within_workspace=true`
- 把这组值写入 `pipeline_orchestrator.params.yaml` 后再次冷启动验证：
  - run：`grasp-1777365423861`
  - 结果：`status=no_candidate`
  - 但失败原因已经变成：
    - `instance 0: no grasp after mask filtering`
  - 不再出现：
    - `workspace_z` 过滤
    - `table_z_m + clearance` 过滤

结论：

- 当前这台机器上，桌面相对 `base_link` 明显低于 `z=0`
- 如果继续用 `table_z_m=0` 和偏高的 `workspace_z` 下界，distributed 真机会稳定退化成 `no_candidate`
- 当前已把验证通过的参数固化到：
  - `config/distributed/pipeline_orchestrator.params.yaml`
- 当前如果再出现 `no_candidate`，应优先排查：
  - mask 约束后的实例点云质量
  - scene grasp 与 mask 的重叠质量
  - 当前帧的感知随机性
  而不是再回退到 `table_z_m/workspace_z` 默认值问题

### 5.7 已确认 MoveIt 观察位存在“到位后短暂漂移”并已补稳定窗口

2026-04-28 在继续排查 `mask filtering` 时，又确认了一个执行链问题：

- 同样的 `cup` prompt 下，需要重点比较 YOLOv8 mask 与 GraspNet 候选的空间重叠
- 失败 run `grasp-1777365423861` 在 `moving_to_observation` 之后的真实回读位姿是：
  - `x=22.776 mm`
  - `y=-21.282 mm`
  - `z=419.647 mm`
- 这和目标 observation pose：
  - `(30, 0, 400, 0, 120, 0)`
  存在明显偏差
- 成功 run `grasp-1777365305801` 的观察位回读则接近目标：
  - `x=30.026 mm`
  - `y=0.0 mm`
  - `z=399.996 mm`

当前结论：

- `moveit_ik` 路径在首次判定“到位”后，真实机械臂还可能短时间漂移
- 这会让 orchestrator 在错误视角下立刻采图，进一步触发：
  - `no grasp after mask filtering`

当前已落地修复：

- `wait_until_pose_goal()` 新增 `post_goal_hold_s`
- `robot_executor_node` 新增参数：
  - `pose_goal_hold_s`
- distributed 默认值已设为：
  - `pose_goal_hold_s = 0.8`
- 在持位窗口内，执行器会继续重发相同目标并反复确认姿态没有漂出容差

验证证据：

- 新回归 run：`grasp-1777366247997`
- 结果：`status=ok`
- 观察位回读：
  - `x=30.132 mm`
  - `y=0.0 mm`
  - `z=399.953 mm`
- 候选结果：
  - `candidate_count=3`
  - `selected grasp score=0.3795`
  - 请求：`/verify_moveit/robot_executor/execute_named_pose`，目标 `z + 3 mm`
  - 实际稳定到：`z=186.104 mm`
  - 说明：真实机械臂已确认向上运动，`moveit_ik -> joint_ctrl_single -> piper_single_ctrl` 主链路已打通
- 当前剩余现象已经不再是 `NO_IK_SOLUTION(-31)`，而是：
  - `run_piper_moveit_ik.sh` 启动后前几秒存在 warm-up 窗口
  - 非常小的补偿动作在某些复现场景下仍可能遇到 `MoveIt IK request timed out: /compute_ik`

### 5.6 已验证 Piper 本地模型链

2026-04-28 已单独验证：

- 在没有 `can0` 的情况下，`piper_description display_urdf.launch.py` 可以在本机正常启动
- 已实际看到：
  - `joint_state_publisher_gui`
  - `robot_state_publisher`
  - `rviz2`
- `display_urdf_follow.launch.py` 这条更轻量的显示链也已验证可启动

本次验证结论：

- 当前已经有一条不依赖真实 CAN 的本地 Piper 可视化入口
- 它适合继续做模型、关节和 RViz 层面的前置调试
- 它不能替代真实 `piper_ros` 驱动链，也不能证明 Gazebo / MuJoCo 可用

### 5.9 2026-04-29 真机失败根因与低位姿候选前置过滤

2026-04-29 继续做 distributed 真机执行时，最新失败 run 为：

- `grasp-1777431413354`

这次链路实际已经走到：

- `moving_to_observation`
- `capturing_scene`
- `analyzing_scene`

但最终仍然失败，直接结果为：

- `status=failed`
- `no robot-reachable grasp candidate found`
- `candidate[0] ... rejected by robot validation: MoveIt IK request timed out: /compute_ik`

继续做最小对照探针后，已经把问题进一步收敛为“低位姿候选本身不可解”，不是 ROS 链路整体失效：

- 最新一轮 `plan_waypoints` 里，三段关键位姿为：
  - `pregrasp z=-0.0195 m`
  - `grasp z=-0.0109 m`
  - `retreat z=+0.0805 m`
- 在同一套 `joint_states_feedback` seed 下，直接对 `/compute_ik` 发请求：
  - `pregrasp` 返回 `NO_IK_SOLUTION(-31)`
  - `grasp` 返回 `NO_IK_SOLUTION(-31)`
  - `retreat` 返回 `code=1`

这说明：

- 当前失败并不是 `/compute_ik` 完全不可用
- 也不是 `joint_states_feedback` 丢失
- 而是当前规划出来的低位姿候选已经落到机械臂无解区 / 桌面危险区附近

基于这组证据，当前已落地一个更靠前的保护规则：

- 在 `collect_grasp_candidates()` 里新增 `0.05 m` 位姿地板
- 对 `pregrasp / target / grasp / retreat` 四个位姿统一检查
- 任一关键位姿 `z < 0.05 m`，就直接拒绝该候选
- diagnostics 现在会新增：
  - `pose_floor_examples=[...]`

验证结果：

- 新增回归测试 `test_grasp_candidate_pose_floor.py`
- 已确认：
  - 低于 `0.05 m` 的候选不会再进入执行候选池

当前结论：

- 这类“明显过低”的候选现在会在规划筛选阶段就被挡掉
- 不再继续拖到 `robot_executor -> MoveIt IK` 才暴露为 timeout / no solution
- 后续如果再出现 `no robot-reachable grasp candidate found`，要优先区分：
  - 是 `pose_floor` 先挡掉了
  - 还是候选本身在正常高度但仍然 `NO_IK_SOLUTION`

基于这次过滤继续重启整条真机链路后，2026-04-29 又补做了两轮真机重测：

- `grasp-1777432752076`
- `grasp-1777432900621`

这两轮都没有进入 `capture/analyze`，而是更早失败在：

- `service call timeout: /robot_executor/execute_named_pose`

这说明：

- `0.05 m` 低位姿前置过滤已经不是这两轮失败的直接原因
- 当前最新阻塞点变成了“冷启动后 observation named pose 下发 / 执行超时”
- 这属于比候选生成更早的一层问题，需要和前面的 `NO_IK_SOLUTION` 区分开记录

## 6. 当前未完成的部分

当前还没有证明：

- `ros2` 真机运动执行稳定
- 单次抓取成功率满足验收
- action 入口已替代当前 `Trigger + prompt` 控制面

当前仍需关注的真机侧风险：

- `robot_backend=ros2` 下 orchestrator 会先移动到 observation pose
- D435 偶发第一次打开返回 `Device or resource busy`，当前外部 worker 已加重试
- `run_piper_driver.sh` 当前通过 `PYTHONPATH` 注入 `piper_sdk` 和 `python-can`，不是系统级安装
- `moveit_ik` 真机首条命令前，建议在 `run_piper_moveit_ik.sh` 启动后额外预留约 `10s`
- `run_live_grasp_one_click.sh` 已把上述 warm-up 和 `/grasp_pipeline/probe` 等待封装进一键启动路径；如果仍有 readiness 失败，优先先看 `logs/one_click/<timestamp>/`
- 当前毫米级补偿动作仍需继续观察 `MoveIt IK request timed out: /compute_ik` 是否会复现
- 当前如果候选本身落到 `z < 0.05 m`，现在会在 planner 阶段被前置过滤；这类 run 更可能直接退化成 `no_candidate`

当前仍未完成的仿真侧事项：

- Gazebo 运行时当前不在这台 Jazzy 机器上
- `piper_mujoco` 当前缺 `mujoco_py`
- 因此“完整物理仿真”仍不能视为当前已可用项

## 7. 当前必须知道的坑

### 7.1 `no_candidate` 不等于系统坏了

当前最常见的“失败”其实是正常完成但没有候选：

- prompt 不在视野里
- YOLOv8-seg 没有检测到 prompt 对应的 COCO 实例
- 候选被筛光

如果日志能走到 `analyzing_scene -> no_candidate`，说明数据流本身大概率已经通了。

### 7.2 不要依赖 `ros2 topic echo --once` 回看历史结果

在这台工作站上，`ros2 topic echo --once` 对 transient-local 历史样本并不稳定。

当前推荐：

- 优先使用 `./scripts/show_last_distributed_snapshot.sh`
- 需要结构化结果时，再直接看 `logs/distributed_runs/<run_id>/`
  - `final_result.json` 看最终状态、最终选中的 candidate 和执行摘要
  - `candidate_validation.json` 看 top-k 候选的 robot validation 淘汰链路
  - `execution_trace.json` 看真正执行下发后的逐步回读

### 7.3 不能重复拉起两套 distributed stack

之前已经实际遇到过：

- 旧栈没退出
- 新栈又启动
- 同名 service/topic 被不同进程同时提供

现在 `run_distributed_stack_graspnet.sh` 已加入启动拦截。

### 7.4 没有真机时，不要把实机行为当成已验收

当前只证明了：

- `fake` 后端下前半段分布式链路正确
- `ros2` 真机后端下状态读取和命令下发链路正确

还没有证明：

- 真机执行与回零闭环稳定
- 机械臂实际能完成抓取、撤退、交接和回家

## 8. 当前建议的下一步

如果继续按“无真机优先”推进，建议顺序如下：

1. 继续稳定前半段
   - 固定几个已知场景
   - 提高 `prompt -> segmentation -> candidate` 命中率
2. 继续标准化结果结构
   - 尤其是 `vision_worker/result_json` 和产物目录内容
3. 真机侧继续排障
   - 先观察 `0.05 m` 低位姿前置过滤后，distributed 真机结果是转成更干净的 `no_candidate`，还是能稳定落到更高、更可达的 fallback candidate
   - 再把 `robot_executor + pose_execution_mode:=moveit_ik` 的 warm-up / timeout 行为继续收敛
   - 再验证 observation / home pose 等较大动作是否真实执行
   - 最后再打开 `execute=true`

### 8.1 Top-1 候选被 IK 拒绝时的可追溯诊断

当前已经确认一种常见现象：

- 视觉 top-1 候选可能被 `robot validation` 拒绝
- 最终系统会退到更可达的 fallback candidate
- 现在这条链已经补齐到 run artifact：
  - `candidate_validation.json` 会记录 top-k 候选进入 robot validation 之后的结构化结果
  - 每个候选至少会带：
    - `candidate_index`
    - `candidate_score`
    - `instance_index`
    - `translation_camera_m`
    - `target_base_m`
    - `target_rpy_deg`
    - `pregrasp_base_m`
    - `grasp_base_m`
    - `retreat_base_m`
    - `within_workspace`
    - `workspace_violations`
    - `robot_validation_result`
    - `robot_validation_stage`
    - `ik_error_type`
    - `ik_error_message`
    - `waypoint_results`
    - `selection_result`
  - `enable_pregrasp=false` 时，按 `grasp -> target -> retreat` 顺序校验
  - `enable_pregrasp=true` 时，按 `pregrasp -> grasp -> target -> retreat` 顺序校验
  - 哪个 waypoint 先失败，就把哪个 waypoint 记成该 candidate 的失败位姿
  - `timeout` 和 `MoveIt IK failed: code=-31` 已区分成不同 `ik_error_type`
  - fallback 被选中时，artifact 里会同时保留：
    - top-1 rejected
    - fallback selected

当前推荐的排查入口：

1. 看 `final_result.json`
   - 确认最终执行的是哪个 candidate
   - 看 summary 是否已经提示 `selected fallback candidate[...]`
2. 看 `candidate_validation.json`
   - 找 `candidate_index=0`
   - 看它的 `robot_validation_stage`
   - 看 `ik_error_type`
   - 看 `ik_error_message`
   - 看 `waypoint_results`
3. 如果最终真的执行了，再看 `execution_trace.json`
   - 这里是下发执行后的真实逐步回读，不是 IK 预校验结果

仍未完成的后续项：

1. 补齐 top-k 的完整淘汰链路
   - 当前已覆盖 `rejected_by_robot_validation` 和 `selected_for_execution`
   - 还没有统一落盘 `filtered_by_score / filtered_by_angle / filtered_by_workspace / filtered_by_pose_floor`
2. 让 RViz 可视化和拒绝原因对应起来
   - 还不能直接在 marker 上区分 top-1、fallback 和 rejected candidate
3. 增加一条“只做 validation、不做执行”的调试路径
   - 目前仍要走 distributed 主流程，尚无单独的 validation-only service
4. 补更宽的测试契约
   - 当前已覆盖 robot validation 失败、timeout 与 `NO_IK_SOLUTION` 区分、fallback 被选中
   - 还没覆盖 `workspace / pose_floor` 被挡掉时的统一 artifact 结构
5. 规划下一轮验证顺序
   - 建议先做 validation-only，再做 RViz 增强，最后再做真机执行扩展

这组 TODO 的最终验收标准：

- 能直接报出 `candidate[0]` 的完整无解位姿
- 能明确说出失败发生在 `pregrasp / grasp / target / retreat` 的哪一步
- 能明确区分 timeout 和 `NO_IK_SOLUTION`
- 能只靠 run artifact 完成排查，不依赖现场临时 `echo`

### 8.2 2026-04-29 姿势链路语义修正与三轮真机结果

当前已经明确并落地的姿势语义是：

- `grasp`: 沿工具轴先下探的预接触位
- `target`: 补偿后的实际接触 / 闭爪位
- `retreat`: 抓后抬升位

这次修正的核心不是删掉 `target`，而是把 distributed 与单节点执行链都统一成：

- `pregrasp -> grasp -> target -> close_gripper -> retreat`

同时，MoveIt IK 预校验也已经统一成：

- `enable_pregrasp=false` 时：`grasp -> target -> retreat`
- `enable_pregrasp=true` 时：`pregrasp -> grasp -> target -> retreat`

相关回归测试现已覆盖：

- `test_plan_validation.py`
- `test_robot_executor_node.py`
- `test_coordinator_execution.py`

2026-04-29 基于这套新语义补做了 3 轮真机复测：

1. `grasp-1777443044750`
   - `status=ok`
   - `candidate_validation` 里 `grasp -> target -> retreat` 全部 `ok`
   - `execution_trace` 已确认真实执行到了 `target`，然后才闭爪、后撤、handoff、home
2. `grasp-1777443244739`
   - `status=failed`
   - 仅剩 1 个候选进入 robot validation
   - artifact 表面记录为 `MoveIt IK request timed out`
   - 但对该轮 `grasp_base_m` 与 `target_base_m` 直接做 `/compute_ik` 探针后，两者都返回 `NO_IK_SOLUTION(-31)`
   - 结论：这类失败属于候选姿势本身不可解，不是执行链路语义错误
3. `grasp-1777443391158`
   - `status=failed`
   - `candidate_validation` 里 `grasp -> target -> retreat` 全部 `ok`
   - 说明该候选在 MoveIt 上可解
   - 但最终失败在执行期：`move timeout: pos_tol=8.00 rot_tol=6.00`
   - 结论：这类失败属于实机运动未在容差内到位，不属于 grasp pose 语义问题

基于这三轮结果，当前阶段结论是：

- 抓取姿势链路语义已经理顺，并且至少有 1 轮真实成功样本
- 剩余不稳定性主要分成两层：
  - 候选本身 `NO_IK_SOLUTION`
  - 候选可解，但执行期 `move timeout`
- 当前下一步应优先增强执行失败 artifact，把失败 step、目标位姿、回读位姿与误差完整落盘

## 9. 建议读文件顺序

建议按下面顺序继续接手这个工程：

1. `AGENTS.md`
2. `README.md`
3. `docs/CURRENT_STATUS.md`
4. `docs/DISTRIBUTED_RUNBOOK.md`
5. `docs/DISTRIBUTED_ARCHITECTURE.md`
6. `docs/MIGRATION_CONTRACT.md`
7. `docs/ENGINEERING_SPEC.md`
8. `docs/MIGRATION_TODO.md`
9. `docs/ROBOT_COUPLING_MAP.md`
10. 与当前任务直接相关的代码文件
