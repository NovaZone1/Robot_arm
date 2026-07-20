# Robot Coupling Map

本文档用于标记旧版抓取主流程里所有直接依赖机器人控制实现的区域，方便后续替换为 ROS2 适配层。

## Source Files

关键来源文件：

- [run_grasp_pipeline.py](/home/wt/Documents/handover_piper_ros/old/robot_grasp/src/run_grasp_pipeline.py)
- [pipeline.py](/home/wt/Documents/handover_piper_ros/old/robot_grasp/src/grasping/pipeline.py)
- [piper_controller.py](/home/wt/Documents/handover_piper_ros/old/robot_grasp/src/robot/piper_controller.py)

## Primary Boundary

旧工程的主耦合边界是：

- `GraspExecutionPipeline -> PiperController`

迁移目标是把这条边界改成：

- `GraspExecutionPipeline -> RobotArmClient -> Ros2PiperClient`

## Direct Robot Calls In Old Pipeline

旧版 `pipeline.py` 中直接访问 `self.robot` 的能力主要包括：

- 连接与生命周期
  - `connect()`
  - `disconnect()`
  - `enable()`
  - `emergency_stop()`

- 状态读取
  - `read_end_pose_mm_deg()`
  - `parse_arm_status()`
  - `format_arm_status()`

- 位姿运动
  - `move_end_pose_mm_deg()`
  - `wait_until_pose_reached()`
  - `pose_error()`

- 夹爪控制
  - `open_gripper()`
  - `close_gripper()`
  - `wait_for_gripper()`
  - `wait_for_gripper_effort()`
  - `get_gripper_status()`

## Already Decoupled In New Workspace

当前 `new/robot_grasp_ros2` 已经先抽离出的纯逻辑包括：

- 坐标变换
- 标定字段解析
- 点云与分割处理
- 抓取候选排序
- 工作空间检查
- TCP 位姿与抓取候选生成抓取计划

这些逻辑现在集中在：

- [planning.py](/home/wt/Documents/handover_piper_ros/new/robot_grasp_ros2/src/grasping/planning.py)
- [models.py](/home/wt/Documents/handover_piper_ros/new/robot_grasp_ros2/src/grasping/models.py)

## Remaining Work

后续 ROS2 适配层至少需要补出这些上层可调用语义：

- `connect`
- `disconnect`
- `enable`
- `disable`
- `emergency_stop`
- `recover_from_estop`
- `read_end_pose_mm_deg`
- `move_end_pose_mm_deg`
- `wait_until_pose_reached`
- `open_gripper`
- `close_gripper`
- `wait_for_gripper`
- `wait_for_gripper_effort`
- `format_arm_status`

## Important Unit Gap

这是迁移中的关键风险：

- 旧控制层大量使用 `mm/deg`
- `piper_ros` 文档中的 `/pos_cmd` 使用 `m/rad`

因此单位换算必须只出现在机器人适配层，不能污染抓取规划层。
