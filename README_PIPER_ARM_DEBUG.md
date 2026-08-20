# Piper 机械臂：CAN1 与抓取 Dashboard 启动

当前固定约定：`can1` 为 Piper 机械臂 CAN 总线，速率为 **1 Mbps**。

## 1. 启动 CAN1

确认机械臂已上电、USB-CAN 已接好后，在终端执行：

```bash
sudo ip link set can1 down
sudo ip link set can1 type can bitrate 1000000 restart-ms 100
sudo ip link set can1 txqueuelen 10
sudo ip link set can1 up

ip -details -statistics link show can1
```

正常时应看到 `can1` 为 `UP, LOWER_UP`，且 CAN 状态为
`ERROR-ACTIVE`。这表示 Linux CAN 接口正常；是否能收到机械臂帧还取决于
机械臂电源、急停和 CAN 线连接状态。

可选的总线查看命令：

```bash
timeout 3 candump can1
```

## 2. 启动抓取 Dashboard

在同一终端（或已完成上一步 CAN1 配置的新终端）执行：

```bash
cd /home/nvidia/auto/Robot_arm/source
PIPER_CAN_PORT=can1 ./scripts/run_grasp_dashboard.py
```

终端出现下列信息后，在浏览器打开 <http://127.0.0.1:8765>：

```text
Grasp dashboard: http://127.0.0.1:8765
```

该终端必须保持运行。网页中点击“启动真机栈”后，等待 `driver` 状态显示
“运行”再执行 Probe、抓取或放置。

## CAN 短暂停顿保护

Piper 驱动默认允许 CAN 反馈短暂停顿最多 `2.0 s`。反馈在宽限时间内恢复时，
驱动继续发布状态，不再因为约 `0.25 s` 的瞬时零帧直接退出；持续超过宽限时间
仍会为安全起见关闭驱动，并要求人工检查机械臂和总线后重新启动真机栈。

如需诊断，可在启动 Dashboard 前覆盖宽限时间：

```bash
PIPER_CAN_PORT=can1 PIPER_CAN_LOSS_GRACE_S=3.0 ./scripts/run_grasp_dashboard.py
```

不建议将该值设得过大，因为真正断线时机械臂执行器必须及时停止等待。



cd ~/auto/Robot_arm/source
./scripts/record_placement_uv_xy.sh --item orange_bottle
