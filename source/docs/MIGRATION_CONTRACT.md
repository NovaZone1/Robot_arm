# Migration Contract

本文件是 `new/robot_grasp_ros2` 的最高优先级迁移契约。

注：本契约关注迁移边界，不保证其中所有环境示例都反映当前机器的最终运行基线。实际运行命令优先看 `README.md` 与 `docs/DISTRIBUTED_RUNBOOK.md`。

每次开始实现、修改、联调前，必须按下面顺序阅读：

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

---

## 1. 迁移目标

把旧入口：

- `old/robot_grasp/src/run_grasp_pipeline.py`

迁移为新工程中的两个层次：

- 内部兼容入口：
  - `new/robot_grasp_ros2/src/run_grasp_pipeline_ros2.py`
- 系统主运行形态：
  - `pipeline_orchestrator_node + camera_server_node + vision_worker_node + robot_executor_node`

并把机器人控制后端从：

- `piper_sdk`

替换为：

- `piper_ros` humble
- `rclpy`
- `RobotArmClient -> Ros2PiperClient`

---

## 2. 不可破坏约束

以下规则是硬约束，不允许在迁移中破坏：

- 规划层继续使用 `mm/deg`
- ROS 话题层负责 `mm/deg <-> m/rad` 转换
- 新工程禁止直接依赖 `piper_sdk`
- 新工程所有机器人访问必须经过 `RobotArmClient`
- 感知、点云、候选抓取筛选、坐标变换优先复用旧代码
- 新增 ROS topic / service 名称必须集中定义在 `RobotArmClientConfig`
- 不允许把 ROS 消息类型泄漏进 `grasping/` 规划层

---

## 3. 分层边界

需要区分两层概念：

1. 系统主运行形态
   - 当前推荐主线是分布式四节点模式
   - 单节点入口保留为兼容 / 调试路径
2. 内部核心流水线结构
   - 仍然保持下面的四层边界

迁移后的内部核心流水线固定采用四层结构：

### 3.1 Entry Layer

文件：

- `src/run_grasp_pipeline_ros2.py`

补充说明：

- 这是单次运行和兼容验证入口
- 不是当前推荐的系统主运行形态
- 分布式系统入口由 orchestrator 对外控制面承接

职责：

- CLI 参数解析
- 日志与配置打印
- SIGINT 急停接管
- 创建 coordinator 与 robot client
- 执行 `connect -> run_once -> disconnect`

### 3.2 Pipeline Layer

文件：

- `src/grasping/coordinator.py`

职责：

- 组织单次抓取流程
- 串联 perception、planner、robot client
- 不直接依赖 ROS 消息类型

### 3.3 Planning Layer

文件：

- `src/grasping/models.py`
- `src/grasping/planning.py`

职责：

- 候选抓取筛选
- 坐标变换
- 工作空间检查
- 抓取位姿生成

### 3.4 Robot Adapter Layer

文件：

- `src/robot/client.py`
- `src/robot/types.py`

职责：

- ROS2 topic / service 接入
- 单位转换
- TCP 位姿读取
- 夹爪控制
- 状态反馈与等待逻辑

---

## 4. Robot API Contract

`RobotArmClient` 是唯一允许被上层调用的机器人接口。

上层只允许依赖以下语义：

- 生命周期
  - `connect`
  - `disconnect`
  - `enable`
  - `disable`
  - `emergency_stop`
  - `recover_from_estop`

- 状态读取
  - `read_end_pose_mm_deg`
  - `get_arm_status_snapshot`
  - `format_arm_status`
  - `get_gripper_status`

- 位姿运动
  - `move_end_pose_mm_deg`
  - `wait_until_pose_reached`
  - `pose_error`

- 夹爪
  - `open_gripper`
  - `close_gripper`
  - `wait_for_gripper`
  - `wait_for_gripper_effort`

任何新需求如果需要新增机器人能力，必须先补到 `RobotArmClient` 抽象层，再补具体实现。

---

## 5. 单位规范

### 5.1 规划层单位

- 平移：`m`
- TCP 输入输出：`mm`
- 欧拉角：`deg`
- 夹爪开口：`mm`
- 夹爪力矩：`N·m`

### 5.2 ROS2 层单位

根据当前本地已验证的 `piper_ros` humble 接口：

- `/pos_cmd` 的 `x/y/z` 使用 `m`
- `/pos_cmd` 的 `roll/pitch/yaw` 使用 `rad`
- `/pos_cmd.gripper` 使用 `m`
- `/end_pose` 反馈位置使用 `m`
- `/end_pose` 反馈姿态为四元数

结论：

- 规划层绝不直接处理 `rad`
- 适配层统一负责 `deg <-> rad`
- 适配层统一负责 `mm <-> m`

---

## 6. 当前已确认的 ROS2 接口

基于本地工作区 `third_party/piper_ros_humble` 源码和本机接口查询，当前确认：

- topic
  - `/pos_cmd`
  - `/end_pose`
  - `/arm_status`
  - `/joint_states_feedback`
  - `/joint_ctrl_single`

- service
  - `/enable_srv`

- message / service
  - `piper_msgs/msg/PosCmd`
  - `piper_msgs/msg/PiperStatusMsg`
  - `piper_msgs/srv/Enable`

注意：

- 历史 README 中出现过 `enable_flag` 或 `enable_cmd`
- 但当前新实现统一优先使用 `/enable_srv`
- 夹爪控制当前通过 `JointState` 发布到 `/joint_ctrl_single`

---

## 7. 代码规范

- 只在适配层处理 ROS 依赖
- 只在适配层处理单位换算
- 数据结构优先 `dataclass(slots=True)`
- 入口参数命名尽量保持和旧入口一致
- 能复用旧逻辑时优先迁移，不重写算法
- 对真机运动逻辑，优先保留旧版的等待、超时、到位判定语义

---

## 8. 实施顺序

高效顺序固定如下：

1. 先保证 `run_grasp_pipeline_ros2.py` 参数和旧入口对齐
2. 再保证 `Ros2PiperClient` 能独立完成连接、使能、位姿读取、夹爪读写
3. 再把 coordinator 串成 `observe -> perceive -> plan -> execute`
4. 再把 coordinator 能力挂入分布式四节点主线
5. 先做 distributed fake 基线验证
6. 最后做真机联调

禁止先写大而全的新逻辑，再回头适配接口。

---

## 9. 当前缺口

到当前为止，真正还没完成的是：

- 真机执行闭环还没有验收完成
- `RunGraspPipeline.action` 还没有替代当前 `Trigger + prompt` 控制面
- 分布式主线虽然已经可运行，但仍以 fake 基线为主
- 感知命中率和真机切换条件仍需继续做实

---

## 10. 完成判定

满足以下条件才算迁移完成：

- 分布式主线可以稳定替代旧工程主流程
- 单节点兼容入口仍可用于对照调试
- `new` 工程不再依赖 `piper_sdk`
- `RobotArmClient` 足以承接旧抓取流程全部机器人能力
- distributed fake 基线稳定可重复
- 完成一次真实的：
  - `observe`
  - `precenter`
  - `grasp`
  - `retreat`
  - `handoff`
  - `home`
