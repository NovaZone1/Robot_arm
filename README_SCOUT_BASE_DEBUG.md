# Scout Mini 底盘调试 README

本文档是当前 Jetson 真机的 Scout Mini 底盘调试操作卡。适用于：

- Scout Mini：`can2 @ 500,000 bit/s`
- ROS 2 Humble
- `/scout`
- `/base_scan_controller`
- Dashboard 底盘单向扫描

当前 Scout 工作空间：

```bash
/home/nvidia/auto/ROS2_FOR_SCOUT_MINI/third_party/scout_mini_ws
```

当前抓取工程：

```bash
/home/nvidia/auto/Robot_arm/source
```

## 1. 安全原则

1. 急停和遥控器必须在操作者手边。
2. 首次测试速度使用 `0.02 m/s`，距离使用 `0.05 m`。
3. 同一时刻只允许一个 Scout 驱动实例。
4. 同一时刻只允许一个运动控制来源实际发布非零 `/cmd_vel`。
5. 导航、键盘遥控、Dashboard 扫描和相对移动服务不能同时控制底盘。
6. 重新 launch 前必须确认旧 `scout_base_node` 已退出。
7. 不要把 `txqueuelen` 设置为 `1000`；当前使用 `10`。
8. `/odom` 不连续时禁止扫描。

## 2. 当前已确认参数

| 项目 | 当前值 |
| --- | --- |
| Scout CAN | `can2` |
| 波特率 | `500,000 bit/s` |
| CAN 队列 | `10` |
| Scout 协议 | `AGX_V2` |
| `/odom` 正常频率 | 约 `50 Hz` |
| 扫描步长 | `0.15 m` |
| 扫描速度 | `0.04 m/s` |
| 最大扫描距离 | `1.20 m` |
| 最大视角数 | `10` |
| 单段超时 | `22 s` |
| Odom 过期阈值 | `0.5 s` |

扫描从盒排左侧附近开始，只沿前进方向停车拍摄，不再执行中间点前后往返。

## 3. 当前 Scout 驱动已知问题

当前 `scout_base/ugv_sdk` 存在生命周期和异步 CAN 缺陷：

- Scout 主循环没有可靠地随 ROS shutdown 停止。
- CAN 读错误路径可能在线程内部调用 `join()` 自己，导致 `exit code -6`。
- 异步发送引用调用方栈上的 `can_frame`，存在生命周期和并发写风险。
- `auto_reconnect` 只重新请求 commanded mode，不会重建消失的 SocketCAN/USB 设备。

在这些缺陷修复并重新构建前，必须严格执行本文的单实例、停车、完整退出和 CAN 重置流程。

## 4. 每次启动前检查

### 4.1 检查 CAN 编号

```bash
ip -brief link
ethtool -i can2
readlink -f /sys/class/net/can2/device
```

`can2` 应是底盘 USB-CAN。机械臂当前使用 `can1 @ 1 Mbps`，不要混用。

如果 `can2` 不存在，先恢复 USB-CAN 枚举；不要直接 launch。

### 4.2 确认没有旧 Scout

```bash
pgrep -af scout_base_node
```

有输出时不要再次 launch。先按“正确停止流程”结束旧实例。

### 4.3 配置 CAN

```bash
sudo ip link set can2 down
sudo ip link set can2 type can bitrate 500000 restart-ms 100
sudo ip link set can2 txqueuelen 10
sudo ip link set can2 up

ip -details -statistics link show can2
timeout 3 candump can2 | head -30
```

必须确认：

- `can state ERROR-ACTIVE`
- `bitrate 500000`
- `candump` 持续有底盘帧
- RX 持续增长

没有报文时不要启动 ROS 驱动。

## 5. 正确启动 Scout

为 Scout 单独保留一个终端，整个调试期间不要关闭或重复启动：

```bash
source /opt/ros/humble/setup.bash
source /home/nvidia/auto/ROS2_FOR_SCOUT_MINI/third_party/scout_mini_ws/install/setup.bash

ros2 launch scout_base scout_mini_base.launch.py port_name:=can2
```

成功标志：

```text
Detected protocol: AGX_V2
Creating interface for Scout with AGX_V2 Protocol
Using CAN bus to talk with the robot
Robot initialized, start running ...
```

出现以下情况时不要继续：

- `Detected protocol: UNKONWN`
- `process has died`
- `exit code -6`
- `exit code -11`

## 6. 启动后验证

另开终端：

```bash
source /opt/ros/humble/setup.bash
source /home/nvidia/auto/ROS2_FOR_SCOUT_MINI/third_party/scout_mini_ws/install/setup.bash
source /home/nvidia/auto/Robot_arm/ros_ws/install/setup.bash
```

### 6.1 节点与话题

```bash
ros2 node list | grep -E 'scout|base_scan'
ros2 topic info /odom --verbose
ros2 topic info /cmd_vel --verbose
```

要求：

- `/scout` 一份
- `/base_scan_controller` 一份
- `/odom` 的 Publisher count 为 `1`
- `/cmd_vel` 的 Scout Subscriber count 为 `1`

### 6.2 里程计

```bash
timeout 5 ros2 topic hz /odom
ros2 topic echo /odom --once
```

正常频率约为 `50 Hz`。没有 `/odom` 或提示 stale 时禁止运动。

### 6.3 Scout 状态

```bash
ros2 topic echo /scout_status --once
```

如果该命令提示消息类型不可用，说明当前终端没有 source Scout 工作空间。

## 7. 分级运动测试

### 7.1 第一级：只读

先观察 `/odom` 和 `/scout_status`，不发送运动命令。

### 7.2 第二级：5 cm 低速测试

清空前后通道并准备急停：

```bash
ros2 service call /base_scan_controller/move_relative \
  robot_grasp_msgs/srv/MoveBaseRelative \
  "{distance_m: 0.05, speed_mps: 0.02, timeout_s: 8.0}"
```

要求：

- 方向正确
- `traveled_m` 正常增长
- 没有 lateral/yaw 超限
- 服务返回成功

### 7.3 第三级：反向 5 cm 测试

```bash
ros2 service call /base_scan_controller/move_relative \
  robot_grasp_msgs/srv/MoveBaseRelative \
  "{distance_m: -0.05, speed_mps: 0.02, timeout_s: 8.0}"
```

只有前后方向都确认后，才允许进入扫描。

### 7.4 第四级：Dashboard 单向扫描

开始前确认：

1. 机械臂已到当前低观察位。
2. 前方 `1.2 m` 通道清空。
3. 遥控器状态允许底盘受 ROS 控制。
4. `/odom` 约 `50 Hz`。
5. 没有导航、键盘遥控等其他非零 `/cmd_vel` 控制源。
6. Dashboard 中勾选安全确认。

点击“底盘单向扫描”后，系统会临时启用一次性安全参数，完成后自动关闭。不要手动长期保持
`base_multiview_enabled=true`。

## 8. 导航、遥控与扫描的切换

切换控制方式前先停止扫描控制器：

```bash
ros2 service call /base_scan_controller/stop \
  std_srvs/srv/Trigger "{}"
```

再发送一次零速度：

```bash
ros2 topic pub --once /cmd_vel \
  geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

确认底盘静止后，才能启动导航或人工控制。

调试导航结束后：

1. 停止导航节点。
2. 确认 `/cmd_vel` 没有额外活动控制源。
3. 重新验证 `/odom`。
4. 再使用扫描服务。

## 9. 正确停止与重新 launch

### 9.1 先停车

```bash
ros2 service call /base_scan_controller/stop \
  std_srvs/srv/Trigger "{}"

ros2 topic pub --once /cmd_vel \
  geometry_msgs/msg/Twist \
  "{linear: {x: 0.0, y: 0.0, z: 0.0}, angular: {x: 0.0, y: 0.0, z: 0.0}}"
```

观察底盘已经完全静止。

### 9.2 结束 launch

在唯一的 Scout launch 终端按一次 `Ctrl+C`，等待其退出，然后检查：

```bash
pgrep -af scout_base_node
```

必须没有输出。不要在旧进程仍存在时再次 launch。

### 9.3 重新 launch 前重置 CAN

当前驱动退出流程尚未修复，因此每次重新 launch 前执行：

```bash
sudo ip link set can2 down
sudo ip link set can2 type can bitrate 500000 restart-ms 100
sudo ip link set can2 txqueuelen 10
sudo ip link set can2 up

timeout 3 candump can2 | head -20
```

确认有帧后才能重新启动 Scout。

如果 `can2` 已从 `ip -brief link` 消失，普通重新 launch 无法恢复。需要先让 USB-CAN
重新枚举，再确认新的接口编号。

## 10. 常见故障

### 10.1 `Detected protocol: UNKONWN`

驱动在 5 秒内没有收到能识别 AGX_V2 的 `0x221/0x241` 帧。

检查：

```bash
timeout 3 candump can2 | head -30
```

无帧时不要反复 launch。

### 10.2 `Scout /odom has not been received`

```bash
ros2 node list | grep scout
ros2 topic info /odom --verbose
pgrep -af scout_base_node
```

通常表示 Scout 驱动未启动或已经退出。

### 10.3 `Scout /odom is stale`

这不是单纯扫描识别失败。检查：

```bash
timeout 5 ros2 topic hz /odom
ip -brief link
pgrep -af scout_base_node
```

- `/odom` Publisher count 为 `0`：Scout 驱动已经消失。
- `can2` 存在但无帧：CAN/控制器状态异常。
- `can2` 不存在：USB-CAN 设备已从系统消失。

### 10.4 相对移动超时且 `traveled=0`

如果 `/odom` 正常但完全不动，检查：

- 遥控器控制模式
- 急停
- `scout_status.error_code`
- 是否存在其他 `/cmd_vel` 发布者
- 是否有导航节点持续发送零速度或相反速度

### 10.5 `can2` 消失或 Scout `exit code -6`

查看：

```bash
journalctl -k --since "-5 min" --no-pager | \
  grep -E 'usb|gs_usb|can2|disconnect|xmit'
```

`USB disconnect` 只证明设备在 USB 总线上逻辑消失；原因可能是：

- USB 接触、供电或干扰
- USB-CAN 固件复位/挂死
- `gs_usb` 发送状态异常
- Scout 驱动异常关闭或异步发送缺陷触发底层问题

不要只靠重新 launch。先恢复 `can2` 枚举和报文，再启动驱动。

### 10.6 无法启用一次性底盘扫描

先检查是否有重复 `/grasp_pipeline`：

```bash
cd /home/nvidia/auto/Robot_arm/source
./scripts/ros2_system.sh node list
```

出现同名节点警告时先清理重复分布式栈。安全参数必须设置后回读成功，Dashboard 才会调用扫描。

## 11. 日志与监控

实时看 USB/CAN：

```bash
sudo journalctl -kf | grep -E 'usb|gs_usb|can2'
```

实时看接口：

```bash
watch -n 0.5 'ip -brief link'
```

实时看里程计：

```bash
timeout 30 ros2 topic hz /odom
```

Scout ROS 日志：

```bash
find /home/nvidia/.ros/log -maxdepth 2 -type f \
  -name '*scout*' -printf '%T@ %p\n' | sort -nr | head
```

## 12. 每次调试最短检查表

1. 急停和遥控器可用，运动通道清空。
2. 无旧 `scout_base_node`。
3. `can2 @ 500 kbps`、`txqueuelen=10`、有持续底盘帧。
4. 只启动一次 Scout launch。
5. 必须看到 `AGX_V2`。
6. `/odom` 必须稳定约 `50 Hz`。
7. 先做 `0.05 m @ 0.02 m/s` 测试。
8. 导航与扫描不能同时控制 `/cmd_vel`。
9. 结束时先停车，再退出 launch。
10. 重新 launch 前确认旧进程已退出并重置 CAN。
