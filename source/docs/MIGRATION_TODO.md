# ROS2 Migration TODO

本文件记录 `new/robot_grasp_ros2` 的迁移 backlog、遗留问题和实施记录，不再作为项目入口文档。

注：本文件保留了较多迁移过程记录，包含历史上的 Humble/overlay 路径。当前机器上的实际运行方式以后续已更新的入口文档为准：`README.md`、`docs/CURRENT_STATUS.md`、`docs/DISTRIBUTED_RUNBOOK.md`。

开始任何改动前，先读：

- `README.md`
- `docs/CURRENT_STATUS.md`
- `docs/DISTRIBUTED_RUNBOOK.md`
- `../AGENTS.md`
- `docs/MIGRATION_CONTRACT.md`
- `docs/ENGINEERING_SPEC.md`
- `docs/ROBOT_COUPLING_MAP.md`
- `src/robot/client.py`
- `src/grasping/coordinator.py`

## Goal

把旧版主程序：

- `old/robot_grasp/src/run_grasp_pipeline.py`

迁移到：

- `new/robot_grasp_ros2/src/run_grasp_pipeline_ros2.py`

并将机械臂控制链路从：

- `PiperController + piper_sdk`

替换为：

- `RobotArmClient + Ros2PiperClient + piper_ros(humble)`

## Non-Negotiable Rules

- 规划层继续使用 `mm/deg` 语义，禁止把 ROS 单位细节泄漏进 `grasping/`
- ROS 适配层负责 `mm/deg <-> m/rad` 转换
- 禁止在新工程中直接引入 `piper_sdk`
- 感知、点云、抓取候选筛选、坐标变换优先复用旧逻辑
- 所有硬件访问必须经过 `RobotArmClient`
- 任何新增 ROS topic / service 名称都要集中写在 `RobotArmClientConfig`

## Current Status

当前主线说明：

- 推荐运行形态已经切到分布式模式
- 单节点入口和 `run_grasp_pipeline_ros2.py` 现在主要用于兼容 / 对照 / 调试
- 运行命令和排障说明以 `README.md` 和 `docs/DISTRIBUTED_RUNBOOK.md` 为准

- 已完成 Phase 1: 新工作区骨架初始化
- 已完成 Phase 2: 纯规划逻辑与基础数据结构迁移
- 已完成旧版机器人耦合点梳理
- 已完成 `agilexrobotics/piper_ros` humble 仓库本地拉取
- 已完成 `piper_msgs` 本地构建
- 已完成 `Ros2PiperClient` 基础 ROS topic / service 适配
- `GraspPipelineCoordinator` 已实现：
  - `current_tcp_pose`
  - `current_base_to_camera`
  - `_capture_fused_rgbd`
  - `_filter_scene_grasps_by_mask`
  - `capture_and_perceive`
  - `_move_to_center_target`
  - `select_best_grasp`
  - `move_to_home`
  - `move_to_observation_pose`
  - `move_to_handoff_pose`
  - `execute_grasp_plan`
  - `run_once_from_perception`
  - `run_once`
- `run_grasp_pipeline_ros2.py` 已完成：
  - `connect -> run_once -> disconnect`
  - `SIGINT -> coordinator.request_emergency_stop`
  - `--probe-robot` 与正式单次运行共存
  - 启动前体检（ROS runtime / perception imports / RealSense / checkpoint）
- `--dry-run` 已不再是“配置壳”，现在会真实进入观察位、感知和规划链路
- 已补充 `AGENTS.md`，固定必读顺序与 source-of-truth 优先级
- 已补充 `MIGRATION_CONTRACT.md`，固定迁移边界、接口契约与实施顺序
- 已补充 `ENGINEERING_SPEC.md`，固定代码规范、接口规范与读文件顺序
- 已确认本机存在 `/opt/ros/humble/setup.bash`
- 已确认本地 overlay `third_party/piper_ros_humble/install/setup.bash` 可用
- 已确认本地 overlay 可用于 Python 3.10 运行时兼容：
  - `/home/wt/Documents/handover_piper_ros/third_party/rclpy_graspnet_install/setup.bash`
  - `/home/wt/Documents/handover_piper_ros/third_party/ros_py310_overlay_ws/install/setup.bash`
- 已确认可导入：
  - `rclpy`
  - `geometry_msgs.msg`
  - `sensor_msgs.msg`
  - `rcl_interfaces.msg`
  - `piper_msgs.msg`
  - `piper_msgs.srv`
- 已确认 `ros2 interface show piper_msgs/msg/PosCmd`
- 已确认 `ros2 interface show piper_msgs/srv/Enable`
- 已确认入口脚本现在会在真正运行前做 preflight，并把缺失依赖收敛为可执行报错
- 已确认推荐统一运行时为 `graspnet` env (`Python 3.10`)
- 已确认在 `graspnet` env 中运行：
  - `python src/run_grasp_pipeline_ros2.py --robot-backend ros2 --probe-robot`
  - Python/ROS preflight 全部通过
- 已补充可直接复用的运行脚本：
  - `scripts/ros_env_graspnet.sh`
  - `scripts/probe_robot_graspnet.sh`
  - `scripts/start_piper_single_graspnet.sh`
  - `scripts/run_fake_graspnet.sh`
- 已完成第一版分布式骨架：
  - `robot_grasp_msgs/`
  - `pipeline_orchestrator_node`
  - `camera_server_node`
  - `vision_worker_node`
  - `robot_executor_node`
- 已补同机分布式运行入口：
  - `scripts/run_distributed_stack_graspnet.sh`
- 已补分布式 RViz 配置与打开脚本：
  - `rviz/distributed_grasp_pipeline.rviz`
  - `scripts/open_distributed_rviz.sh`
- 已补分布式运行手册：
  - `docs/DISTRIBUTED_RUNBOOK.md`
- 已验证同机分布式最小链路：
  - 四节点可同时拉起
  - `/grasp_pipeline/probe` 可成功返回
  - `run_pipeline_service.sh cup` 可触发一次真实分布式任务
  - 当前一次验证已走到 `capturing_scene -> analyzing_scene -> no_candidate`
- 已接通分布式确认控制面：
  - `/grasp_pipeline/confirm`
  - `/grasp_pipeline/reject`
  - `scripts/confirm_pipeline_service.sh`
  - `scripts/reject_pipeline_service.sh`
- 已补分布式单次运行产物落盘：
  - `new/log/distributed_runs/<run_id>/request.json`
  - `new/log/distributed_runs/<run_id>/cycles.json`
  - `new/log/distributed_runs/<run_id>/final_result.json`
  - `scripts/show_last_run_artifact.sh`
- 已确认 `fake` 后端在不连接机械臂时可以跑完整主流程：
  - 真实进入 observation pose、RealSense、YOLOv8-seg、GraspNet、candidate select
  - 当前帧若没有合法候选，会输出 summary + diagnostics，而不是直接异常退出
- 已支持 `--show-pointcloud`：
  - 点云重建完成后可直接弹 Open3D 窗口
  - 用于调试场景点云与实例点云质量
  - 现在会优先叠加实例级 grasp 候选，必要时回退显示 scene grasp 候选

## Observed Runtime Blocker On April 6, 2026

基于 2026-04-06 在当前工作站的只读核对，原先的 Python ABI 阻塞已经被 overlay 方案解决，当前结论应更新为：

- 已经存在统一运行时：
  - `/home/wt/.conda/envs/graspnet/bin/python`
- 已通过以下 overlay 解决 ROS Python ABI 问题：
  - `/home/wt/Documents/handover_piper_ros/third_party/piper_ros_humble/install`
  - `/home/wt/Documents/handover_piper_ros/third_party/rclpy_graspnet_install`
  - `/home/wt/Documents/handover_piper_ros/third_party/ros_py310_overlay_ws/install`
- 在该运行时中，已经确认可导入：
  - `rclpy`
  - `rcl_interfaces.msg`
  - `piper_msgs.msg`
  - `piper_msgs.srv`
- 已确认 `python src/run_grasp_pipeline_ros2.py --robot-backend ros2 --probe-robot`
  - Python/ROS preflight 全部通过
  - 进入机器人探测阶段后，卡在 `/end_pose` 无反馈
- 当前剩余阻塞聚焦为：
  - `piper_ros` 单臂节点未运行，或者
  - `CAN` 设备未配置
- 本机当前 `ip link` 看不到 `can0`

## Single-node compatibility mode

说明：

- 本节描述的是单节点兼容模式
- 当前推荐主线仍然是分布式四节点模式
- 分布式运行方式请以 `README.md` 和 `docs/DISTRIBUTED_RUNBOOK.md` 为准

- 已新增节点封装：
  - `robot_grasp_ros2/grasp_pipeline_node.py`
  - 节点名固定为 `grasp_pipeline`
  - 在 `rclpy` executor 中复用 `run_grasp_pipeline_ros2.py` 与 `GraspPipelineCoordinator`
- 已新增兼容运行入口：
  - `scripts/run_grasp_pipeline_node_graspnet.sh`
  - 实测可在 `graspnet` Python 3.10 + overlay 环境下把 node 拉起到 `idle`
- 节点已暴露以下 ROS 接口：
  - topic: `/grasp_pipeline/status`
  - topic: `/grasp_pipeline/summary`
  - topic: `/grasp_pipeline/diagnostics`
  - topic: `/grasp_pipeline/result_json`
  - topic: `/grasp_pipeline/run_prompt`
  - service: `/grasp_pipeline/run`
  - service: `/grasp_pipeline/probe`
  - service: `/grasp_pipeline/stop`
- 已新增 RViz 结果话题：
  - `/grasp_pipeline/rviz/scene_pointcloud`
  - `/grasp_pipeline/rviz/instance_pointcloud`
  - `/grasp_pipeline/rviz/candidate_grasps`
  - `/grasp_pipeline/rviz/selected_grasp`
  - `/grasp_pipeline/rviz/plan_waypoints`
  - `/grasp_pipeline/rviz/candidate_markers`
  - `/grasp_pipeline/rviz/selected_grasp_markers`
  - `/grasp_pipeline/rviz/plan_markers`
  - `/grasp_pipeline/rviz/camera_transform`
- 话题语义已固定：
  - 点云与抓取候选发布在相机系 `camera_color_optical_frame`
  - 执行位姿发布在机械臂基座系 `base_link`
  - `base_link -> camera_color_optical_frame` 现在会同步发布到标准 `/tf`
  - `/grasp_pipeline/rviz/camera_transform` 继续保留，作为最近一次结果的调试/兼容输出
- 已补齐 Python 3.10 overlay 的 TF 依赖：
  - 已引入 `third_party/ros_py310_overlay_ws/src/geometry2`
  - 已定向构建 `tf2_msgs`
  - 已验证 `TransformBroadcaster(node)` 可在 `graspnet` 环境正常创建
  - 已验证 system ROS shell 可直接 `ros2 topic echo --once /tf`
- 已验证以下 system ROS shell 命令：
  - `ros2 node list`
  - `ros2 service list | rg grasp_pipeline`
  - `ros2 topic list | rg grasp_pipeline`
  - `ros2 service call /grasp_pipeline/probe std_srvs/srv/Trigger '{}'`
  - `ros2 param set /grasp_pipeline prompt cup`
  - `ros2 topic pub --once /grasp_pipeline/run_prompt std_msgs/msg/String "{data: cup}"`
- 现阶段推荐固定成双 shell 工作流：
  - 终端 A：`source scripts/ros_env_graspnet.sh && ./scripts/run_grasp_pipeline_node_graspnet.sh`
  - 终端 B：优先直接使用 `./scripts/ros2_system.sh ...`
- 已确认当前不要在 graspnet overlay shell 里直接执行：
  - `ros2 launch robot_grasp_ros2 grasp_pipeline.launch.py`
  - 原因：`ros2` CLI 是 Python 3.14，但该 shell 会优先吃到 Python 3.10 的 `rclpy` overlay，进而触发 `_rclpy_pybind11` 导入失败
- 已补专用包装脚本：
  - `scripts/ros2_system.sh`
  - `scripts/run_pipeline_service.sh`
- 已补节点单实例保护：
  - `scripts/run_grasp_pipeline_node_graspnet.sh` 默认拒绝重复启动第二个同名节点
  - 如确有需要，需显式设置 `ALLOW_DUPLICATE_GRASP_PIPELINE=1`
- 已补 Piper 交互调试链：
  - `piper_interactive_marker_node`
  - `piper_pose_bridge_node`
  - `joint_state_feedback_relay_node`
  - `launch/piper_interactive_teleop.launch.py`
- 现阶段推荐远程触发顺序固定为：
  - `./scripts/run_pipeline_service.sh cup --robot-backend fake`
  - 若必须手动执行，则先 `ros2 param set /grasp_pipeline prompt cup`
  - 再 `ros2 service call /grasp_pipeline/run std_srvs/srv/Trigger '{}'`
- 已确认 graph 中不能同时保留多个同名 `/grasp_pipeline` 节点：
  - 否则 `/grasp_pipeline/run` 和 `/grasp_pipeline/set_parameters` 的命中目标会变得不确定
- 因此当前定位应明确为：
  - `launch/grasp_pipeline.launch.py` 先保留为包结构/参数模板
  - `run_grasp_pipeline_node_graspnet.sh` 是兼容入口，不是分布式主线

## Legacy Flow Snapshot

以下结论只来自旧实现：

- `old/robot_grasp/src/grasping/pipeline.py`
- 重点方法：
  - `run_once`
  - `capture_and_perceive`
  - `_capture_fused_rgbd`
  - `_filter_scene_grasps_by_mask`

### run_once 阶段顺序

1. `_ensure_not_stopped`
2. `move_to_observation_pose`
3. 如果 `precenter_before_grasp=True`，执行 `_move_to_center_target`
4. `capture_and_perceive`
5. `_collect_grasp_candidates`
6. 输出候选诊断与 top-k 预览
7. `plan_grasp`
8. 非 `dry_run` 时：
   - `preview_grasp`
   - 可选用户确认
   - `execute_grasp_plan`
   - 如果执行报错包含 `ANGLE_LIMIT`，按候选池顺序回退重试
9. 组装 `result` 字典与 `summary`

### capture_and_perceive 阶段顺序

1. `_capture_fused_rgbd` 采集融合后的 `color_bgr/depth_meters`
2. 根据 `depth_fusion_frames`、filter mode、对齐帧可用性决定点云重建后端：
   - `sdk`
   - `manual`
3. 可选深度滤波：
   - `bilateral`
   - `median`
   - `island`
   - `radius`
   - `none`
4. `segmenter.segment_text(color_bgr, text_prompt)` 得到实例 mask
5. `save_segmentation_outputs(...)` 生成每个实例点云与中间产物
6. 基于全场景深度重建 `scene_points`
7. `graspnet.predict(scene_points=scene_points, object_points=scene_points)` 得到全场景抓取候选
8. 针对每个 mask：
   - 重建 object points
   - 可选做 `island/radius` 点云后处理
   - 从保存后的点云计算 `object_center_camera_m`
   - 计算 `object_centers_uv`
   - 调 `_filter_scene_grasps_by_mask` 把全场景 grasp 过滤成实例级 grasp group
9. 返回 `PerceptionResult`

### _capture_fused_rgbd 输入输出

- 输入：
  - `self.camera`
  - `self.config.depth_fusion_frames`
- 输出：
  - `ok`
  - `color_bgr`
  - `depth_raw`
  - `depth_meters`
- 语义：
  - `depth_fusion_frames == 1` 时走单帧快速路径
  - `depth_fusion_frames > 1` 时抓多帧，按有效像素做 `nanmedian` 深度融合

### _filter_scene_grasps_by_mask 输入输出

- 输入：
  - `scene_grasp_group`
  - `mask_np`
  - `depth_meters`
  - 相机内参
- 输出：
  - `None` 或实例级 `GraspGroup`
- 语义：
  - 把 grasp translation 投影到图像平面
  - 要求投影点落在 mask 内或其邻域内
  - 要求投影深度与 depth map 的误差不超过 `depth_tolerance_m`
  - 保留 surviving indices，再通过 `graspnet.subset_grasp_group(...)` 返回子集

### capture_and_perceive 主要输入输出

- 输入：
  - `text_prompt`
  - `camera`
  - `segmenter`
  - `graspnet`
  - `camera.intrinsics`
  - `config.depth_fusion_frames`
  - `config.pointcloud_filter_mode`
  - `config.pointcloud_backend`
- 输出：`PerceptionResult`
  - `color_bgr`
  - `depth_meters`
  - `segmentation`
  - `pointclouds`
  - `grasp_groups`
  - `scene_grasp_count`
  - `scene_point_count`
  - `object_point_counts`
  - `object_centers_camera_m`
  - `object_centers_uv`

### 哪些阶段依赖 robot client

- 直接依赖 robot client：
  - `move_to_observation_pose`
  - `_move_to_center_target`
  - `current_tcp_pose`
  - `current_base_to_camera`
  - `preview_grasp`
  - `execute_grasp_plan`
  - `request_execution_confirmation` 之后的真实执行链路
- 不直接依赖 robot client：
  - `_capture_fused_rgbd`
  - `capture_and_perceive`
  - `_filter_scene_grasps_by_mask`
  - `_collect_grasp_candidates`
  - `plan_grasp`
- 间接依赖 robot state 的规划步骤：
  - `_approach_angle_to_vertical_deg`
  - `_build_grasp_plan_data`
  - `plan_grasp`
  - 原因：这些步骤依赖 `base_to_camera` 或当前 TCP/标定结果

### 哪些阶段可以直接迁到 coordinator.py

- 可以直接迁入 `coordinator.py` 或由其直接编排：
  - `run_once` 的阶段控制
  - `_collect_grasp_candidates`
  - `plan_grasp`
  - `preview_grasp`
  - `execute_grasp_plan`
  - `result`/`summary` 组装
- 更适合做成 coordinator 调用的感知适配器，但仍由 coordinator 负责编排：
  - `_capture_fused_rgbd`
  - `capture_and_perceive`
  - `_filter_scene_grasps_by_mask`
- 不建议把 ROS 细节塞入这些阶段：
  - 相机、分割、GraspNet 的具体 backend 初始化
  - ROS topic/service 访问

## Source Of Truth

主流程行为以旧代码为准：

- `old/robot_grasp/src/run_grasp_pipeline.py`
- `old/robot_grasp/src/grasping/pipeline.py`
- `old/robot_grasp/src/robot/piper_controller.py`

新工程中这些文件是迁移主落点：

- `src/run_grasp_pipeline_ros2.py`
- `src/grasping/coordinator.py`
- `src/robot/client.py`
- `src/robot/types.py`

## High-Priority TODO

1. 完成 `run_grasp_pipeline_ros2.py` 入口对齐
- 对齐旧版 CLI 参数
- 对齐配置构建逻辑
- 对齐日志输出
- 对齐 `connect -> run_once -> disconnect` 生命周期
- 对齐 `SIGINT -> emergency_stop` 处理

2. 核对 `piper_ros` 实际接口
- 已确认末端位姿反馈 topic: `/end_pose`
- 已确认机械臂状态 topic: `/arm_status`
- 已确认使能 / 失能 service: `/enable_srv`
- 已确认夹爪控制方式: `/joint_ctrl_single` 或 `/pos_cmd.gripper`
- 仍需在真机上确认急停 / 清错 / 复位语义是否满足旧工程要求

3. 补完 `Ros2PiperClient`
- 连接后首帧反馈等待与错误提示
- runtime topic/service 缺失诊断
- `/arm_status` 高层语义映射修正
- 真机超时和容差参数微调

4. 迁移旧主流程的执行壳
- 把旧版 `capture_and_perceive`
- `select_best_grasp`
- `plan_grasp`
- `execute_grasp_plan`
- `run_once`
  逐步迁入新 coordinator

5. 把 coordinator 补成真正可跑的一次抓取流程
- 把 `capture_and_perceive` 从旧流程迁到新工程
- 让入口真正产出 `PerceptionResult`
- 将 `move_to_observation_pose -> perception -> run_once_from_perception` 串成闭环
- 根据 `home / observe / pregrasp / grasp / retreat / handoff` 执行动作

6. 做最小联调闭环
- fake client 冒烟测试
- ROS2 dry-run 测试
- 真机单步移动测试
- 真机开合夹爪测试
- 真机单次抓取测试

## Missing In New Coordinator

当前 `src/grasping/coordinator.py` 相比旧版 `run_once` 仍缺以下闭环步骤：

- 还没有做 `RealSense + YOLOv8-seg + GraspNet` 的端到端冒烟验证
- `preview_grasp` 已补回，但还没有做 GUI/显示环境验证
- 还没有针对真机参数做动作超时、容差和观测位姿整定

## Missing In New Entry Shell

当前 `src/run_grasp_pipeline_ros2.py` 相比旧版入口仍缺：

- `--dry-run` 虽然已进入真实流程，但还没有完成带相机/模型文件的端到端验证
- 还没有完成“单臂节点已启动 + CAN 已就绪”的真机闭环验证
- 还需要把当前工作站的推荐启动顺序固定给后续 AI/工程师，避免回退到旧模型环境或系统 Python

## Next Implementation Order

按这个顺序做，返工最少：

1. 补 `preview_grasp` 和 top-k 候选调试输出
- 已完成，当前新流程已具备执行前候选摘要和 `preview_grasp`

2. 做端到端冒烟联调
- fake
- ros2 dry-run

3. 做真机单步验证
- 末端位姿移动
- 夹爪开合
- 观察位/回 home

4. 做真机单抓
- observe -> perceive -> grasp -> handoff -> home

## Method Mapping

旧版 `pipeline.py` 直接依赖的机器人能力，迁移后必须逐一保真：

- 生命周期
  - `connect`
  - `disconnect`
  - `enable`
  - `emergency_stop`

- 状态读取
  - `read_end_pose_mm_deg`
  - `format_arm_status`
  - `get_arm_status_snapshot`

- 运动
  - `move_end_pose_mm_deg`
  - `wait_until_pose_reached`
  - `pose_error`

- 夹爪
  - `open_gripper`
  - `close_gripper`
  - `wait_for_gripper`
  - `wait_for_gripper_effort`
  - `get_gripper_status`

## Confirmed piper_ros Interface

已经从 `third_party/piper_ros_humble` 源码确认：

- `piper_msgs/msg/PosCmd`
  - 话题：`/pos_cmd`
  - 单位：`x/y/z` 为 `m`，`roll/pitch/yaw` 为 `rad`
  - `gripper` 单位为 `m`

- `geometry_msgs/Pose`
  - 话题：`/end_pose`
  - 位姿反馈为四元数，需要在适配层转回 `roll/pitch/yaw`

- `piper_msgs/msg/PiperStatusMsg`
  - 话题：`/arm_status`

- `sensor_msgs/msg/JointState`
  - 话题：`/joint_states_feedback`
  - 含 7 个 joint，其中最后一个是 `gripper`
  - `position[6]` 可作为夹爪开口反馈
  - `effort[6]` 可作为夹爪力矩反馈

- `piper_msgs/srv/Enable`
  - 服务：`/enable_srv`
  - 推荐用这个服务做 enable / disable

注意：

- README 里能看到 `enable_flag`、`enable_cmd` 等历史说明，但当前默认单臂控制节点以 `piper_ctrl_single_node.py` 为准
- 新工程统一优先使用 `/enable_srv`，不要依赖 README 中的 topic 示例
- `/arm_status` 是底层原始状态码，不是高层 enable 状态；解释逻辑以源码为准

## Execution Order

推荐严格按这个顺序推进，效率最高：

1. 先读 `docs/MIGRATION_CONTRACT.md`
2. 进入统一运行时：
   ```bash
   export PATH=/home/wt/.conda/envs/graspnet/bin:$PATH
   source /opt/ros/humble/setup.bash
   source /home/wt/Documents/handover_piper_ros/third_party/rclpy_graspnet_install/setup.bash
   source /home/wt/Documents/handover_piper_ros/third_party/ros_py310_overlay_ws/install/setup.bash
   cd /home/wt/Documents/handover_piper_ros/new/robot_grasp_ros2
   ```
3. 先验证 ROS Python 导入：
   ```bash
   python - <<'PY'
   import rclpy
   from rcl_interfaces.msg import ParameterDescriptor
   from piper_msgs.msg import PosCmd
   print(rclpy.__file__)
   print(ParameterDescriptor)
   print(PosCmd)
   PY
   ```
4. 再做机器人侧 preflight：
   ```bash
   python src/run_grasp_pipeline_ros2.py --robot-backend ros2 --probe-robot
   ```
5. 真机联调时，在另一终端启动单臂节点：
   ```bash
   source /opt/ros/humble/setup.bash
   source /home/wt/Documents/handover_piper_ros/third_party/piper_ros_humble/install/setup.bash
   ros2 launch piper start_single_piper.launch.py can_port:=can0 auto_enable:=true
   ```
6. 如果卡在 `/end_pose`，先检查：
   ```bash
   ip link
   ```
7. 只有在 `can0` 出现且 `piper_ros` 单臂节点已启动后，才继续真机抓取闭环

## Done Criteria

满足以下条件才算迁移完成：

- `run_grasp_pipeline_ros2.py` 可替代旧入口运行
- 规划层不依赖 ROS 类型
- 新工程不再依赖 `piper_sdk`
- `Ros2PiperClient` 能完成状态读取、运动、夹爪控制
- 能完成一次真实的 `observe -> grasp -> handoff -> home`
