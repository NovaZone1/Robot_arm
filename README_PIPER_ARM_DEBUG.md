# Piper 机械臂与抓取系统调试 README

本文档是当前 Jetson 真机的机械臂调试操作卡。适用于：

- Piper 机械臂：`can1 @ 1,000,000 bit/s`
- RealSense D435
- MoveIt IK
- 分布式抓取节点
- 抓取 Dashboard：`http://127.0.0.1:8765`

当前工作目录：

```bash
/home/nvidia/auto/Robot_arm/source
```

## 1. 安全原则

1. 真机首次运行固定使用 `5%` 速度。
2. 急停必须在操作者手边，工作空间内不得有人。
3. 机械臂处于示教模式时禁止下发自动位姿。
4. 不要在高空、抓持物体或靠近桌面时直接断电。
5. 停机前先放下物体，再回 Home，最后失能和停止进程。
6. Dashboard 启动和手动分终端启动二选一，禁止混用。
7. 不要重复点击“启动真机栈”；等待状态变绿后再继续。
8. 保留 Piper 的 `ARRIVED` 到位判断。超时时先查反馈、模式和路径，不要删除到位保护。

## 2. 当前已确认参数

| 项目 | 当前值 |
| --- | --- |
| Piper CAN | `can1` |
| Piper 波特率 | `1,000,000 bit/s` |
| Scout CAN | `can2`，不要配置成 Piper 波特率 |
| 默认速度 | `5%` |
| Home | `[57, 0, 215, 0, 85, 0] mm/deg` |
| 抓取/放置观察位 | `[0, 35.5, 491.1, 180, 67.77, -89.97] mm/deg` |
| 抓后持物位 | `[256.885, 0, 315.239, 0, 84.939, 0] mm/deg` |
| 夹爪最大物理开度 | `70 mm` |

旧的高观察位 `Z≈542 mm / pitch≈80°` 可能进入不稳定关节分支并触发
`ANGLE_LIMIT`，不要再用于自动任务。

当前抓后持物位是在 Home 朝向下向前伸约 `200 mm`，不再自动左转 `90°`。

## 3. 每次启动前检查

### 3.1 检查接口编号

USB 设备重新连接后，CAN 编号可能变化。先检查：

```bash
ip -brief link
ethtool -i can1
timeout 2 candump can1 | head -20
```

Piper 常见反馈帧包含：

```text
2A1 2A2 2A3 2A4 2A5 2A6
```

如果 `can1` 不存在，或看到的不是机械臂帧，不要启动驱动。

### 3.2 配置 Piper CAN

```bash
sudo ip link set can1 down
sudo ip link set can1 type can bitrate 1000000 restart-ms 100
sudo ip link set can1 txqueuelen 10
sudo ip link set can1 up

ip -details -statistics link show can1
timeout 2 candump can1 | head -20
```

期望：

- `can state ERROR-ACTIVE`
- `bitrate 1000000`
- RX 持续增长
- `candump` 有机械臂反馈

### 3.3 确认没有旧真机栈

```bash
pgrep -af 'piper_single_ctrl|move_group|pipeline_orchestrator_node|robot_executor_node'
```

如果已有完整栈，不要再启动第二套。先使用现有 Dashboard 点 Probe。

如果只残留半套节点，先预览清理目标：

```bash
cd /home/nvidia/auto/Robot_arm/source
./scripts/clear_live_grasp_nodes.sh --dry-run
```

确认当前没有正在执行的动作、机械臂已处于安全位后，再清理：

```bash
./scripts/clear_live_grasp_nodes.sh
```

## 4. 推荐启动方式：Dashboard 一体化启动

### 4.1 启动 Dashboard

```bash
cd /home/nvidia/auto/Robot_arm/source
PIPER_CAN_PORT=can1 ./scripts/run_grasp_dashboard.py
```

浏览器打开：

```text
http://127.0.0.1:8765
```

### 4.2 启动真机栈

1. 只点击一次“启动真机栈”。
2. 等待 `pipeline/camera/vision/executor/MoveIt/driver` 全部变绿。
3. 点击 `Probe`。
4. Probe 成功后再做状态读取、夹爪或抓取测试。

启动过程可能需要几十秒。状态未变绿时再次点击会造成重复节点；Dashboard 已有防重复
保护，但操作者仍应坚持单次启动。

## 5. 备选方式：手动分终端启动

手动模式用于定位某个组件，不要同时点击 Dashboard 的“启动真机栈”。

### 5.1 终端 A：Piper 驱动

```bash
cd /home/nvidia/auto/Robot_arm/source
PIPER_CAN_PORT=can1 ./scripts/run_piper_driver.sh --can-port can1
```

### 5.2 终端 B：MoveIt IK

```bash
cd /home/nvidia/auto/Robot_arm/source
./scripts/run_piper_moveit_ik.sh
```

等待约 10 秒，让 `/compute_ik` 起稳。

### 5.3 终端 C：分布式主线

```bash
cd /home/nvidia/auto/Robot_arm/source
./scripts/run_distributed_stack_graspnet.sh \
  --robot-backend ros2 \
  --pose-execution-mode moveit_ik
```

### 5.4 终端 D：检查与触发

```bash
cd /home/nvidia/auto/Robot_arm/source
./scripts/ros2_system.sh service call \
  /grasp_pipeline/probe std_srvs/srv/Trigger "{}"
```

## 6. 启动后的只读检查

### 6.1 检查节点单实例

```bash
cd /home/nvidia/auto/Robot_arm/source
./scripts/ros2_system.sh node list
```

以下节点应各只有一份：

- `/piper_ctrl_single_node`
- `/move_group`
- `/camera_server`
- `/vision_worker`
- `/robot_executor`
- `/grasp_pipeline`
- `/base_scan_controller`

出现同名警告时不要继续运动。先停止重复栈。

### 6.2 读取当前机械臂状态

```bash
cd /home/nvidia/auto/Robot_arm/source
./scripts/ros2_system.sh service call \
  /robot_executor/get_state \
  robot_grasp_msgs/srv/GetRobotState "{}"
```

允许继续自动运动的基本条件：

- `success=True`
- `arm_status=NORMAL`
- `teach_status=0x00`
- `err_code=0x0000`
- 反馈位姿不是全零

如果 `teach_status=0x02`，先通过机械臂控制器退出示教并重新确认状态。单纯调用
`/enable_srv` 不等于退出示教。

重新上电后，驱动会先发送控制器恢复指令并切回位置速度模式，再通过 Piper SDK 的
`GetArmEnableStatus()` 核验六个电机。只有六项都为 `True` 时 `/enable_srv` 才能返回
成功。不要使用低速反馈缓存字段判断使能状态；该字段可能跨控制器掉电保留旧值，造成
“服务显示成功、控制模式切到 CAN、但所有关节完全不动”的假使能。

### 6.3 检查反馈

```bash
./scripts/ros2_system.sh topic hz /end_pose
./scripts/ros2_system.sh topic hz /joint_states_feedback
./scripts/ros2_system.sh service list | grep -E 'enable_srv|robot_executor'
```

## 7. 分级调试顺序

不要从完整抓取开始。每次更换工作区、重启硬件或修改配置后按以下顺序恢复。

### 7.1 第一级：只读状态

只调用 `/robot_executor/get_state`，不运动。

### 7.2 第二级：夹爪空载

确认夹爪周围无物体后，在 Dashboard 点击“释放夹爪”；也可以调用：

```bash
./scripts/ros2_system.sh service call \
  /robot_executor/open_gripper std_srvs/srv/Trigger "{}"
```

### 7.3 第三级：Home 空载

确认路径净空后，以 `5%` 速度移动：

```bash
./scripts/ros2_system.sh service call \
  /robot_executor/execute_named_pose \
  robot_grasp_msgs/srv/ExecuteNamedPose \
  "{name: 'home', pose: {x_mm: 57.0, y_mm: 0.0, z_mm: 215.0, roll_deg: 0.0, pitch_deg: 85.0, yaw_deg: 0.0}, speed_percent: 5.0, open_gripper_first: false}"
```

### 7.4 第四级：观察位空载

```bash
./scripts/ros2_system.sh service call \
  /robot_executor/execute_named_pose \
  robot_grasp_msgs/srv/ExecuteNamedPose \
  "{name: 'placement_observation', pose: {x_mm: 0.0, y_mm: 35.5, z_mm: 491.1, roll_deg: 180.0, pitch_deg: 67.77, yaw_deg: -89.97}, speed_percent: 5.0, open_gripper_first: false}"
```

到位后重新调用 `get_state`，不要只看 service 返回文字。

### 7.5 第五级：感知与规划，不执行

Dashboard 使用：

1. 选择明确物品，如黄色物块。
2. 速度保持 `5`。
3. 不勾选直接执行。
4. 点击“规划后确认”。
5. 检查分割图、抓取投影、候选数和目标落点。

`no_candidate` 表示当前帧没有通过筛选的候选，不等于机械臂驱动故障。

### 7.6 第六级：确认后低速执行

只有分割、目标身份、落点和路径都正确时才确认执行。首次测试保持：

- `5%` 速度
- 急停在手边
- 单个目标
- 不自动放置
- 抓取后保持夹持

### 7.7 第七级：抓后持物与放置区扫描

抓取成功后自动进入当前抓后持物位。进行放置区扫描前确认：

- 机械臂没有遮挡关键标签
- Scout `/odom` 正常
- 底盘扫描通道已清空
- 不同时运行导航或其他 `/cmd_vel` 控制源

自动放置尚未完成释放位姿标定时，扫描通过也不等于允许自动松爪。

## 8. 正确停止流程

### 8.1 停止当前任务

先点击 Dashboard 的“停止任务”，或：

```bash
./scripts/ros2_system.sh service call \
  /grasp_pipeline/stop std_srvs/srv/Trigger "{}"
```

### 8.2 放下物体并回 Home

如果正在持物：

1. 移动到人工确认的安全释放位置。
2. 打开夹爪。
3. 确认物体已脱离。
4. 回 Home。

不要在抓后高位直接失能。

### 8.3 停止真机栈

优先点击 Dashboard 的“停止真机栈”。手动模式下，在各启动终端按一次 `Ctrl+C`，
等待进程退出。

最后检查：

```bash
pgrep -af 'piper_single_ctrl|move_group|pipeline_orchestrator_node|robot_executor_node'
```

如需清理残留，先 dry-run，再执行清理脚本。

## 9. 常见故障

### 9.1 Dashboard 显示 `driver: 未运行`

检查：

```bash
ip -brief link
timeout 2 candump can1 | head
pgrep -af piper_single_ctrl
```

确认 Dashboard 是以 `PIPER_CAN_PORT=can1` 启动。

### 9.2 `move timeout` 且 actual pose 全零

优先检查：

- `/end_pose` 是否发布
- `/joint_states_feedback` 是否发布
- Piper 驱动是否仍存在
- `can1` 是否仍存在并有帧
- 是否处于示教或 STANDBY

不要通过删除 `ARRIVED` 判断掩盖问题。

### 9.3 机械臂停在半空

先停止任务，再读取状态。常见原因：

- 中间路点超时
- IK 无解
- `ANGLE_LIMIT`
- 示教模式
- 驱动反馈中断
- 实际已到位，但回读或到位窗口未满足

不要在原因未确认时连续点击执行。

### 9.4 观察位不可达

只使用当前低观察位。若低观察位仍失败：

1. 读取当前位姿和关节状态。
2. 确认不在示教模式。
3. 先回 Home。
4. 再低速走到观察位。
5. 禁止直接恢复旧高观察位。

### 9.5 `no_candidate`

检查顺序：

1. Prompt 与物品是否对应。
2. 分割掩膜是否正确。
3. 深度是否有效。
4. 工作区和桌面高度参数是否正确。
5. 候选是否被 IK 或安全规则拒绝。

### 9.6 RealSense 不在线

```bash
lsusb | grep -i Intel
lsusb -t
```

确认没有第二个程序占用相机。线缆恢复后重启相机/视觉节点或完整分布式栈。

## 10. 日志与结果

```bash
cd /home/nvidia/auto/Robot_arm/source
./scripts/show_last_distributed_snapshot.sh
./scripts/show_last_run_artifact.sh
```

重点目录：

- Session 日志：`/home/nvidia/auto/Robot_arm/ros_ws/log/distributed/<timestamp>/`
- 单次任务：`/home/nvidia/auto/Robot_arm/log/distributed_runs/<run_id>/`
- 放置扫描：`/home/nvidia/auto/Robot_arm/ros_ws/viz/placement_scan/`

## 11. 每次调试最短检查表

1. 急停可触达，工作空间清空。
2. `can1 @ 1 Mbps` 且有 Piper 帧。
3. 无旧机械臂/MoveIt/分布式进程。
4. 只启动一套 Dashboard 或手动栈。
5. Probe 成功。
6. `get_state` 正常，`teach_status=0x00`。
7. 先夹爪、Home、观察位空载测试。
8. 再做规划后确认。
9. 最后才执行抓取。
10. 停机前放物、回 Home、停止任务、停止真机栈。
