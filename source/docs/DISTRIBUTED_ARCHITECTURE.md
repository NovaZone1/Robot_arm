# Distributed Architecture

本文档定义 `robot_grasp_ros2` 当前推荐的分布式架构、节点职责、接口边界和后续演进方向。

注意：

- 当前四节点骨架已经落地并可运行
- 分布式模式是推荐主线
- 单节点模式保留为兼容 / 对照 / 调试路径
- `RunGraspPipeline.action` 目前只完成了接口定义，尚未替代当前的 `Trigger + prompt` 外部入口
- 若后文出现历史环境路径或版本信息，请以 `README.md` 和 `docs/DISTRIBUTED_RUNBOOK.md` 中已经更新到本机现状的命令为准

## 1. 目标拓扑

推荐三机拓扑，也支持同机多进程：

1. Robot Host
2. Vision Host
3. Operator Host

逻辑组件：

1. `pipeline_orchestrator_node`
2. `camera_server_node`
3. `vision_worker_node`
4. `robot_executor_node`
5. `piper_ros`
6. `rviz2` / 运维触发端

## 2. 节点职责

### `pipeline_orchestrator_node`

- 接收外部任务触发：`run/probe/stop/confirm/reject`
- 维护状态机、`run_id`、summary、diagnostics
- 负责最终抓取规划
- 在 `precenter=true` 时驱动：
  - `capture -> analyze -> recenter move -> capture`
- 在 `place_after_grasp=true` 时驱动：
  - `grasp-and-hold -> placement observation -> capture label -> verify label`
  - `place dry-run -> execute place`

### `camera_server_node`

- 管理 RealSense 生命周期
- 提供按需采集 RGBD 的服务接口
- 发布可选调试缓存：
  - `latest/color`
  - `latest/depth`
  - `latest/camera_info`

### `vision_worker_node`

- 消费采集结果
- 执行 YOLOv8-seg、点云重建、GraspNet、实例筛选
- 对指定饮料瓶增加液体颜色身份过滤
- 一次识别六个后侧竖直面盒标，并按画面从左到右输出动态槽位
- 从纸质盒标深度恢复六个 base-frame 三维点，校验 180 mm 盒列节距并计算目标盒中心
- 返回结构化分析结果
- 发布分布式 RViz 可视化 topic

### `robot_executor_node`

- 管理 `fake` / `ros2` 执行后端
- 对下只通过 `piper_ros` 与硬件交互
- 对上提供读状态、命名位姿执行、抓取计划执行和急停能力
- 对上提供经过盒尺寸、盒标、垂直净空和 IK 检查的放置计划执行
- 当前执行闭环已经补齐到：
  - `optional pregrasp -> grasp -> retreat -> optional handoff -> release -> optional home`

### `piper_ros`

- AgileX 官方 ROS2 驱动
- 负责底层 CAN 与控制消息交互

## 3. 接口边界

## 3.1 Orchestrator 对外控制面

当前稳定的 topic：

- `/grasp_pipeline/status`
- `/grasp_pipeline/summary`
- `/grasp_pipeline/diagnostics`
- `/grasp_pipeline/result_json`
- `/grasp_pipeline/run_prompt`

当前稳定的 service：

- `/grasp_pipeline/run`
- `/grasp_pipeline/probe`
- `/grasp_pipeline/stop`
- `/grasp_pipeline/confirm`
- `/grasp_pipeline/reject`

当前接口状态：

- `/grasp_pipeline/run` 仍然是 `std_srvs/Trigger`
- prompt 通过 parameter / topic 注入
- `RunGraspPipeline.action` 已定义，但还没接 action server

## 3.2 Orchestrator <-> Camera

当前服务边界：

- `/camera_server/capture`

职责：

- 返回 `color/depth/camera_info`
- 为同一次 run 生成可追踪的采集结果

## 3.3 Orchestrator <-> Vision

当前服务边界：

- `/vision_worker/analyze`
- `/vision_worker/match_item_label`

输入核心语义：

- `scene_id`
- `prompt`
- `color_image`
- `depth_image`
- `camera_info`
- `tcp_pose`
- `base_to_camera`
- `options_json`

输出核心语义：

- 感知摘要
- 候选池
- 已选抓取
- 诊断信息
- 文本 summary

## 3.4 Orchestrator <-> Robot Executor

当前服务边界：

- `/robot_executor/get_state`
- `/robot_executor/execute_named_pose`
- `/robot_executor/execute_grasp_plan`
- `/robot_executor/execute_place_plan`
- `/robot_executor/stop_robot`

语义要求：

- 上层继续使用 `mm/deg`
- 适配层负责 `mm/deg <-> m/rad`
- 真机模式下不能直接使用 `piper_sdk`
- `GraspPlan` 的 `target_pose` 表示规划姿态下的 `link6` 位姿；同时通过 `has_tool_contact_geometry`、`target_contact_point_base_m` 和 `tool_contact_offset_tool_m` 携带姿态无关的接触几何。
- `safe_top_down` 改变最终姿态时必须以接触点为约束重新求 `link6` 平移，禁止只替换 RPY 后复用原 XYZ。
- `safe_top_down` 的 RPY 变体必须逐个通过完整 waypoint IK；验证与执行使用同一选择逻辑，全部失败时拒绝执行。
- `PlacePlan` 使用 `mm/deg`，包含动态 `slot_index`、approach/release/retreat、盒外
  尺寸和盒标校验结果；
  executor 必须在 `execute=false` 时先完成同一组安全与 IK 校验。
- `label_verified=false`、槽位越界、盒尺寸不符、非垂直进退或任一路点不可达时禁止
  打开夹爪。

## 4. 可视化与产物契约

## 4.1 RViz Topic 边界

分布式模式下，RViz topic 由 `vision_worker_node` 发布：

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

这意味着：

- 分布式模式下不要再把 `/grasp_pipeline/rviz/*` 当作主线接口
- `Fixed Frame` 建议先用 `camera_color_optical_frame`

## 4.2 结果与日志契约

当前每次 distributed run 都应同时留下两类结果：

1. ROS topic 侧最后一条结果
   - 通过 transient-local QoS 保留
2. 落盘结构化产物
   - `logs/distributed_runs/<run_id>/request.json`
   - `logs/distributed_runs/<run_id>/cycles.json`
   - `logs/distributed_runs/<run_id>/final_result.json`

另外，每次启动整套分布式栈，还会留下 session 级日志：

- `ros_ws/log/distributed/<timestamp>/`

## 5. 当前状态机语义

当前主流程的实际状态边界大致为：

- `idle`
- `preflight`
- `moving_to_observation`
- `reading_robot_state`
- `capturing_scene`
- `analyzing_scene`
- `awaiting_confirmation`
- `executing`
- `moving_to_placement_observation`
- `matching_box_label`
- `placing_object`
- `done`
- `failed`
- `cancelled`

关键语义：

- `execute=false`
  - 只跑到 `plan/no_candidate`
- `execute=true` 且 `confirm=false`
  - 有计划后直接执行
- `execute=true` 且 `confirm=true`
  - 进入 `awaiting_confirmation`
  - 必须显式 `confirm` 或 `reject`

## 6. 同机部署建议

推荐顺序：

1. 同机起四个节点
2. 先用 `fake` 后端跑通前半段
3. 再切 `robot_executor` 到 `ros2`
4. 最后做真机执行闭环验证

当前已验证命令：

```bash
source /home/ybw/piper_grasp_project/source/scripts/ros_env_graspnet.sh
cd /home/ybw/piper_grasp_project/source
./scripts/run_distributed_stack_graspnet.sh --robot-backend fake
./scripts/ros2_system.sh service call /grasp_pipeline/probe std_srvs/srv/Trigger "{}"
./scripts/run_pipeline_service.sh cup
```

## 7. 多机部署建议

### 7.1 网络与中间件

- 全机统一 `ROS_DOMAIN_ID`
- 优先同一二层网络
- 优先使用 CycloneDDS
- 固定网卡白名单

### 7.2 进程放置建议

Robot Host：

- `robot_executor_node`
- `piper_ros`
- 可选 `camera_server_node`

Vision Host：

- `vision_worker_node`
- 模型依赖与 GPU

Operator Host：

- `pipeline_orchestrator_node`
- RViz
- 运维脚本与触发端

### 7.3 时间同步

- 建议全机启用 NTP / Chrony
- 所有关键消息和产物都带 `run_id`

## 8. 故障排查

### 8.1 跨机看不到节点 / 话题

检查：

1. `ROS_DOMAIN_ID` 是否一致
2. DDS 是否绑定了错误网卡
3. 网络是否允许组播 / 广播
4. 防火墙是否拦截 DDS 端口

### 8.2 任务已接收但无结果

检查：

1. orchestrator 是否发出了下游请求
2. `camera_server` 是否成功返回 RGBD
3. `vision_worker` 是否卡在模型加载或显存不足
4. `robot_executor` 是否可用

### 8.3 机器人无反馈或执行失败

检查：

1. `piper_ros` 单臂节点是否已运行
2. `can0` 是否存在
3. `/end_pose` 是否持续更新
4. stop 状态是否被遗留

### 8.4 RViz 空白

检查：

1. `Fixed Frame` 是否先设为 `camera_color_optical_frame`
2. 是否订阅了 `/vision_worker/rviz/*`
3. `/tf` 是否存在 `base_link -> camera_color_optical_frame`
4. 当前任务是否真的产出了有效点云

## 9. 演进顺序

建议继续按下面顺序演进：

1. 继续稳定 distributed fake 基线
2. 完善结果结构和产物字段
3. 把真机 `get_state`、`execute_named_pose`、`execute_grasp_plan` 做实
4. 最后再把长耗时入口逐步 action 化

维护约定：

- README 负责入口和导航
- runbook 负责命令和排障
- architecture 负责节点职责与接口边界
- 所有新接口都必须同步更新文档
