# robot_grasp_ros2 Engineering Spec

本文件是 `new/robot_grasp_ros2` 的工程规范与 API 约束。

注：本文件包含一部分历史迁移约束。若其中的环境命令、路径或版本信息与 `README.md`、`docs/CURRENT_STATUS.md`、`docs/DISTRIBUTED_RUNBOOK.md` 不一致，优先以后者为准；当前这台机器的实际运行基线是 Jazzy + system Python ROS 节点 + conda external workers。

后续任何人或任何 AI 在改这个工程前，必须先读：

1. `README.md`
2. `docs/CURRENT_STATUS.md`
3. `docs/DISTRIBUTED_RUNBOOK.md`
4. `docs/DISTRIBUTED_ARCHITECTURE.md`
5. `AGENTS.md`
6. `docs/MIGRATION_CONTRACT.md`
7. `docs/ENGINEERING_SPEC.md`
8. `docs/MIGRATION_TODO.md`
9. `docs/ROBOT_COUPLING_MAP.md`
10. `src/robot/client.py`
11. `src/grasping/coordinator.py`

## 1. 迁移目标

把旧工程：

- `old/robot_grasp`

迁移到新工程：

- `new/robot_grasp_ros2`

并把机械臂控制链从：

- `piper_sdk` 直控

替换为：

- `piper_ros` humble 的 ROS2 接口

当前主线补充说明：

- 默认推荐运行形态是分布式模式
- `src/run_grasp_pipeline_ros2.py` 和 `grasp_pipeline_node` 保留为兼容 / 对照 / 调试入口

## 1.1 Current Runtime Baseline On April 6, 2026

当前工作站已经确认存在一个可执行的统一运行时，后续 AI 和工程师默认以它为准：

- Python 解释器：
  - `/home/wt/.conda/envs/graspnet/bin/python`
- 推荐运行时名称：
  - `graspnet` env
- 不推荐使用旧的模型专用环境；本工程统一使用 `graspnet` env
- 为解决 ROS Python ABI 问题，必须额外 source：
  - `/opt/ros/humble/setup.bash`
  - `/home/wt/Documents/handover_piper_ros/third_party/piper_ros_humble/install/setup.bash`
  - `/home/wt/Documents/handover_piper_ros/third_party/rclpy_graspnet_install/setup.bash`
  - `/home/wt/Documents/handover_piper_ros/third_party/ros_py310_overlay_ws/install/setup.bash`

在上述运行时中，已经确认可导入：

- `rclpy`
- `rcl_interfaces.msg`
- `piper_msgs.msg`
- `piper_msgs.srv`

并且已经确认：

- `python src/run_grasp_pipeline_ros2.py --robot-backend ros2 --probe-robot`
  - Python/ROS preflight 全部通过
  - 后续实际阻塞点不是 import，而是 `/end_pose` 无反馈

## 2. 分层规则

必须保持下面的分层边界：

- `perception/`
  - 感知、分割、点云、抓取候选生成
  - 不允许依赖 ROS topic / service 细节

- `grasping/`
  - 抓取规划、姿态筛选、流程协调
  - 只依赖显式输入数据和 `RobotArmClient`
  - 不允许直接 import `piper_sdk`
  - 不允许直接 import `piper_msgs`

- `robot/`
  - 机器人适配层
  - 唯一允许接触 ROS topic / service / message 细节的目录
  - 唯一允许做 `mm/deg <-> m/rad` 单位换算的目录

## 3. 单位规范

这是硬约束：

- 规划层一律使用 `mm/deg`
- `piper_ros` 适配层内部使用 `m/rad`
- 夹爪上层语义统一使用 `mm` 开口、`N·m` 力矩

禁止把这些细节泄漏到 `grasping/`：

- ROS `Pose`
- `piper_msgs.msg.PosCmd`
- `sensor_msgs.msg.JointState`

## 4. 机器人 API 规范

新工程上层只允许通过 `RobotArmClient` 调机械臂。

当前约定接口如下：

- 生命周期
  - `connect()`
  - `disconnect()`
  - `enable()`
  - `disable()`
  - `emergency_stop()`
  - `recover_from_estop()`

- 状态
  - `read_end_pose_mm_deg()`
  - `get_arm_status_snapshot()`
  - `format_arm_status()`
  - `get_gripper_status()`

- 运动
  - `move_end_pose_mm_deg(...)`
  - `wait_until_pose_reached(...)`
  - `pose_error(...)`

- 夹爪
  - `open_gripper(...)`
  - `close_gripper(...)`
  - `wait_for_gripper(...)`
  - `wait_for_gripper_effort(...)`

如果后续需要新增能力，先改 `RobotArmClient` 抽象，再改具体后端。

## 5. Coordinator Expectations

- `GraspPipelineCoordinator` 现在直接实现了 `connect` / `disconnect` / `move_to_home` / `move_to_observation_pose` / `move_to_handoff_pose` / `execute_grasp_plan` / `capture_and_perceive` / `run_once` / `run_once_with_inputs`。
- 当前版本已经把 `YOLOv8-seg + RealSense + GraspNet` 主感知链路接入 coordinator，由 coordinator 统一编排观察位、可选预居中、感知、规划和执行。
- 感知组件当前采用 coordinator 内部懒加载：
  - `RealSenseRGBDCamera`
  - `YOLOSegmenter`
  - `GraspNetRunner`
  这样可以保持 `--help`、`--probe-robot`、语法检查等轻量入口不被相机/模型依赖阻塞。
- `RobotArmClientConfig` 默认 `joint_ctrl_topic` 被设置为 `/joint_ctrl_single`，以匹配 Humble 单臂 launch 的控制管道，新的夹爪/关节控制应优先通过这一路径。

### 5.1 Piper Interactive Helper Topics

为了支持 RViz 交互调试，`RobotArmClientConfig` 里新增了以下辅助 topic 默认值：

- `/interactive_piper/target_pose`
  - RViz 交互 marker 输出
- `/interactive_piper/command_pose`
  - bridge 节点归一化到 `base_link` 之后的目标位姿
- `/joint_states`
  - 供 RViz / `robot_state_publisher` 使用的显示关节话题

这些 topic 只允许出现在：

- `src/robot/client.py`
- `robot_grasp_ros2/piper_interactive_marker_node.py`
- `robot_grasp_ros2/piper_pose_bridge_node.py`
- `robot_grasp_ros2/joint_state_feedback_relay_node.py`

禁止把这些调试/交互话题散落进 `grasping/` 或 `perception/`。

### 5.2 Perception Migration Contract

新工程必须对齐旧版 `capture_and_perceive` 的语义，而不是只对齐类型名。

- source of truth:
  - `old/robot_grasp/src/grasping/pipeline.py::capture_and_perceive`
  - `old/robot_grasp/src/grasping/pipeline.py::_capture_fused_rgbd`
  - `old/robot_grasp/src/grasping/pipeline.py::_filter_scene_grasps_by_mask`
- `GraspPipelineCoordinator` 必须自己拥有完整编排能力，或者显式依赖一个“官方感知桥接器”。
- 无论实现落在 `coordinator.py` 还是单独 adapter，coordinator 都必须负责调用顺序，不允许让入口脚本散落拼接旧流程。
- 新版 `PerceptionResult` 必须保留旧语义：
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
- `grasp_groups[i]` 的语义必须是“scene grasp group 经 mask+depth 约束过滤后的实例级候选”，不能直接把全场景 grasp 复制给每个实例。
- `depth_fusion_frames > 1` 时必须保留多帧融合语义，不能退化成单帧采集。
- 点云 backend 选择必须保留旧版决策条件：
  - `sdk` 仅在 `depth_fusion_frames == 1`
  - 且 filter mode 不是 `bilateral/median`
  - 且存在 aligned depth frame
- `run_once` 的执行壳必须保留旧版阶段边界：
  - `move_to_observation_pose`
  - 可选 `precenter`
  - `capture_and_perceive`
  - `candidate collect`
  - `plan`
  - `preview`
  - `execute`
  - `ANGLE_LIMIT` fallback
  - `result/summary`

### 5.3 Target Identity And Placement Contract

- 六类比赛物品的 ID、别名、参考图、物体尺寸和放置标定只能以
  `config/item_catalog.yaml` 为 source of truth。
- 抓取指定颜色物品时，未识别到目标不得回退到其他实例或全场景 grasp。
- 盒标验证必须一次得到全部六个唯一标识，并按固定观察画面从左到右映射到
  `slot_index=0..5`；只验证“目标模板超过阈值”不足以授权松爪。
- 放置必须使用 `PlacePlan` 的 approach/release/retreat 三个位姿，并在执行前走同一条
  dry-run IK 校验。
- 动态定位必须从纸质标识取深度，不得依赖透明盒壁深度；六点必须满足盒长方向
  `180 mm` 节距约束，并以已知盒深 `132 mm` 推导盒中心。
- 动态定位失败时不得自动使用未经显式校准的静态槽位；静态模式下六个
  `slot_centers_mm`（或首尾中心）未标定必须 fail closed。
- `placement.enabled=false` 或物品缺少 `release_rpy_deg` 时必须 fail closed。
- 盒标不匹配、尺寸不符、净空不足或 IK 失败时，executor 禁止打开夹爪。
- 瓶子放置姿态不能从物块姿态推断，必须按实物和盒内净尺寸单独标定。

## 6. piper_ros Humble 真值接口

以下接口已经从：

- `third_party/piper_ros_humble/src/piper/piper/piper_ctrl_single_node.py`

确认过。

### 5.1 Launch

真机单臂常用入口：

- `third_party/piper_ros_humble/src/piper/launch/start_single_piper.launch.py`
- `third_party/piper_ros_humble/src/piper/launch/start_single_piper_rviz.launch.py`

### 5.2 Topics

- `/pos_cmd`
  - 类型：`piper_msgs/msg/PosCmd`
  - 作用：末端位姿控制输入
  - 单位：`x/y/z` 为米，`roll/pitch/yaw` 为弧度，`gripper` 为米

- `/joint_ctrl_single`
  - 类型：`sensor_msgs/msg/JointState`
  - 作用：关节控制输入，也被当前适配层用于夹爪控制

- `/end_pose`
  - 类型：`geometry_msgs/msg/Pose`
  - 作用：末端位姿反馈

- `/arm_status`
  - 类型：`piper_msgs/msg/PiperStatusMsg`
  - 作用：机械臂状态反馈
  - 注意：这里发布的是 SDK 原始状态码，不是“enabled/disabled”这种高层语义

- `/joint_states_feedback`
  - 类型：`sensor_msgs/msg/JointState`
  - 作用：关节与夹爪反馈

### 5.3 Service

- `/enable_srv`
  - 类型：`piper_msgs/srv/Enable`
  - 作用：使能 / 失能
  - 推荐作为新工程统一入口

### 5.4 不要误信的旧说明

仓库 README 和旧示例里会提到一些名字，比如：

- `enable_flag`
- `enable_cmd`

但当前 Humble 代码里真正稳定可依赖的入口，优先级应该是：

1. `/enable_srv`
2. `/pos_cmd`
3. `/joint_ctrl_single`
4. `/end_pose`
5. `/arm_status`
6. `/joint_states_feedback`

### 5.5 Current Machine Runtime And Blocker

当前工作站上的真实状态：

- `graspnet` env + Python 3.10 已能加载 ROS2 Python 依赖
- `rclpy` 来自：
  - `/home/wt/Documents/handover_piper_ros/third_party/rclpy_graspnet_install`
- `rcl_interfaces` 与 `piper_msgs` 来自：
  - `/home/wt/Documents/handover_piper_ros/third_party/ros_py310_overlay_ws/install`
- `--probe-robot` 的 preflight 已通过
- 当前剩余阻塞是：
  - `/end_pose` 没有反馈
  - 最可能原因是 `piper_ros` 单臂节点未运行，或者 `CAN` 设备未配置
- 本机当前 `ip link` 看不到 `can0`

因此，后续调试顺序应固定为：

1. 先保证统一运行时正确 source
2. 再确认 `piper_ros` 单臂节点已启动
3. 再确认 `can0` 存在
4. 最后才排查 `Ros2PiperClient` 逻辑

## 6. 迁移实现规则

### 6.1 允许复用

- 旧工程中的感知与抓取规划逻辑
- 标定、坐标变换、点云处理工具
- 旧版 CLI 参数语义

### 6.2 禁止复用

- 新工程中直接 import `piper_sdk`
- 在 `grasping/` 目录里直接发布 ROS topic
- 在 `perception/` 目录里硬编码 Piper 控制命令

### 6.3 状态解释规则

`piper_ros` 的 `/arm_status` 发布的是底层原始码：

- `ctrl_mode` 应按控制模式码解释
- `arm_status` 应按机械臂状态码解释
- `mode_feedback` 应按运动模式码解释
- `motion_status` 只能按原始码解释，不要擅自翻译成“已经使能”

## 7. 运行入口要求

当前入口分为两类：

1. 分布式主线入口
   - `scripts/run_distributed_stack_graspnet.sh`
   - 配套文档是 `README.md` 和 `docs/DISTRIBUTED_RUNBOOK.md`
2. 单节点兼容入口
   - `src/run_grasp_pipeline_ros2.py`
   - `scripts/run_grasp_pipeline_node_graspnet.sh`

默认推荐第一类；第二类用于兼容验证和局部调试。

`src/run_grasp_pipeline_ros2.py` 必须逐步对齐旧入口：

- CLI 参数尽量保持兼容
- 生命周期保持 `connect -> run_once -> disconnect`
- `SIGINT` 必须触发 `emergency_stop`
- 真机执行路径未打通前，不允许静默假执行
- 进入 `coordinator.connect()` 前必须先做启动前体检，并把错误收敛成可执行提示

### 7.1 Startup Preflight Rules

- `--probe-robot`：
  - 只检查所选 robot backend 的依赖
  - 不强制要求感知依赖存在
- 普通抓取运行：
  - 必须检查 perception 依赖是否可导入
  - 必须检查 RealSense 是否可见
  - 必须检查 GraspNet checkpoint 是否存在
- `--robot-backend ros2`：
  - 必须检查 `rclpy`、`geometry_msgs`、`sensor_msgs`、`piper_msgs`
  - 必须给出明确提示：需要 source `/opt/ros/humble/setup.bash` 与 overlay
  - 当前工作站上的推荐 overlay 是：
    - `/home/wt/Documents/handover_piper_ros/third_party/piper_ros_humble/install/setup.bash`
    - `/home/wt/Documents/handover_piper_ros/third_party/rclpy_graspnet_install/setup.bash`
    - `/home/wt/Documents/handover_piper_ros/third_party/ros_py310_overlay_ws/install/setup.bash`
- GUI/preview 失败应优先降级为 warning，不应让入口因为纯预览问题提前崩溃
- `fake` 后端的 dry-run 在感知链已跑通但未选出合法 grasp 时：
  - 应返回结构化 summary 和 diagnostics
  - 不应仅因为“当前帧没有候选”就把整条流程当成 runtime crash
- `--show-pointcloud`：
  - 应在点云重建完成后显示 Open3D 窗口
  - 至少包含场景点云；存在实例点云时应一并显示
  - 存在实例级过滤后 grasp 时，应优先叠加实例 grasp 候选
  - 若实例级 grasp 为空但全场景 GraspNet 候选存在，可回退显示 scene grasp 用于调试
  - 可视化失败时应降级为 warning，不应让主流程直接崩溃

### 7.2 Recommended Startup Order

推荐先用分布式主线：

```bash
source /home/wt/Documents/handover_piper_ros/new/robot_grasp_ros2/scripts/ros_env_graspnet.sh
cd /home/wt/Documents/handover_piper_ros/new/robot_grasp_ros2
./scripts/run_distributed_stack_graspnet.sh --robot-backend fake
```

另一个终端统一使用：

```bash
cd /home/wt/Documents/handover_piper_ros/new/robot_grasp_ros2
./scripts/ros2_system.sh service call /grasp_pipeline/probe std_srvs/srv/Trigger "{}"
./scripts/run_pipeline_service.sh cup
```

如果要手动 source 或继续做真机 / 半真机调试，再按以下顺序启动：

```bash
export PATH=/home/wt/.conda/envs/graspnet/bin:$PATH
source /opt/ros/humble/setup.bash
source /home/wt/Documents/handover_piper_ros/third_party/piper_ros_humble/install/setup.bash
source /home/wt/Documents/handover_piper_ros/third_party/rclpy_graspnet_install/setup.bash
source /home/wt/Documents/handover_piper_ros/third_party/ros_py310_overlay_ws/install/setup.bash
cd /home/wt/Documents/handover_piper_ros/new/robot_grasp_ros2
```

如无特殊需要，优先使用现成脚本而不是手写 source 顺序：

```bash
source /home/wt/Documents/handover_piper_ros/new/robot_grasp_ros2/scripts/ros_env_graspnet.sh
/home/wt/Documents/handover_piper_ros/new/robot_grasp_ros2/scripts/probe_robot_graspnet.sh
/home/wt/Documents/handover_piper_ros/new/robot_grasp_ros2/scripts/start_piper_single_graspnet.sh
/home/wt/Documents/handover_piper_ros/new/robot_grasp_ros2/scripts/run_fake_graspnet.sh cup
```

如果你在做单节点兼容调试，再使用：

```bash
source /home/wt/Documents/handover_piper_ros/new/robot_grasp_ros2/scripts/ros_env_graspnet.sh
cd /home/wt/Documents/handover_piper_ros/new/robot_grasp_ros2
./scripts/run_grasp_pipeline_node_graspnet.sh
```

推荐先验证导入：

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

再验证机器人侧 preflight：

```bash
python src/run_grasp_pipeline_ros2.py --robot-backend ros2 --probe-robot
```

真机联调时，单臂节点应在另一终端启动：

```bash
source /opt/ros/humble/setup.bash
source /home/wt/Documents/handover_piper_ros/third_party/piper_ros_humble/install/setup.bash
ros2 launch piper start_single_piper.launch.py can_port:=can0 auto_enable:=true
```

如果这一步卡在 `/end_pose`，先检查：

```bash
ip link
```

本机 2026-04-06 的已知结果是不包含 `can0`。

## 8. Perception Interface Mapping

每次评估感知与抓取工作流前，务必先读 `AGENTS.md`、`docs/MIGRATION_CONTRACT.md`、`docs/ENGINEERING_SPEC.md` 和 `docs/MIGRATION_TODO.md`，然后再看旧 pipeline 的 `capture_and_perceive` 实现。

当前接口对照情况：

| 旧 `capture_and_perceive` | 新 perception 模块 | 支撑程度 |
| `/src/grasping/pipeline.py` | `/src/perception/` | |
| - 采集 RealSense RGB+D (`RealSenseRGBDCamera.get_frames`) | `RealSenseRGBDCamera.start/get_frames` | ✅ 已由 coordinator 懒加载接入 |
| - 使用 YOLOv8-seg 生成 mask | `YOLOSegmenter.segment_text` | ✅ 可提供 `segmentation["masks"]` |
| - 过滤点云 & 保存结果 | `perception.geometry` helpers (`save_segmentation_outputs`, filters) | ✅ 可复用已有 helpers |
| - GraspNet 推理输出 `GraspGroup` | `GraspNetRunner.predict` | ✅ 已由 coordinator 调用并写入 `grasp_groups` |
| - 组合 `PerceptionResult` 供 `select_best_grasp` | `PerceptionResult` dataclass | ✅ 数据结构已定义 |

当前状态：

- `GraspPipelineCoordinator` 已经把 `RealSenseRGBDCamera`、`YOLOSegmenter` 与 `GraspNetRunner` 串起来，并能直接产出 `PerceptionResult`。
- `run_once_with_inputs` 仍保留，作为外部感知注入或离线调试入口。
- `preview_grasp` 与 top-k 候选文本预览已补回。
- 剩余缺口主要在联调验证，不是类型或编排缺口：
  - 带真实设备和模型文件的端到端验证
  - headless / GUI 环境下的 preview 行为确认

## 8. 验收标准

只有满足下面条件，才能认为迁移完成：

- 新入口可替代旧入口使用
- 规划层不依赖 ROS 类型
- 新工程不再依赖 `piper_sdk`
- `Ros2PiperClient` 可完成基本状态读取、位姿运动和夹爪控制
- 至少能完成一次 `observe -> grasp -> handoff -> home`
