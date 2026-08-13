# robot_grasp_ros2

`robot_grasp_ros2` 是 `old/robot_grasp` 的 ROS2 迁移工作区。当前这台机器的可用运行形态：

- ROS 2 节点运行在系统 Python
- YOLOv8-seg + GraspNet + Open3D + pyrealsense2 运行在仓库根目录的 `.venv`
- 重型感知和 D435 采集通过外部 worker 子进程桥接回 ROS
- Piper 真机控制仍然走 `piper_ros`

当前主线已经明确：

- 推荐运行形态是分布式 ROS2 流程，不是单节点脚本
- 没有真机时，优先验证 `fake` 后端下的前半段数据流
- 真机控制只能经过 `piper_ros` / ROS2，不能回退到 `piper_sdk`
- 规划层继续使用 `mm/deg` 语义，ROS2 适配层负责 `mm/deg <-> m/rad`
- 当前实例分割固定使用 `yolov8n-seg.pt`，prompt 按 COCO 类别名匹配
- 固定红/黄/蓝 3D 打印物块可使用 `red block`、`yellow block`、`blue block`
  或 `物块` prompt；该路径使用 HSV 颜色实例分割，不加载新的检测模型

## 当前结论

截至 2026-04-08，仓库已经具备以下稳定基线：

- 分布式四节点主链路已落地：
  - `pipeline_orchestrator_node`
  - `camera_server_node`
  - `vision_worker_node`
  - `robot_executor_node`
- 分布式模式下已接通：
  - `/grasp_pipeline/run`
  - `/grasp_pipeline/probe`
  - `/grasp_pipeline/stop`
  - `/grasp_pipeline/confirm`
  - `/grasp_pipeline/reject`
- 在没有真机机械臂的情况下，`fake` 后端已经跑通：
  - `observation -> get_state -> capture -> analyze -> result_json -> rviz topics`
- 本机已实测打通：
  - `D435 capture -> external inference worker -> plan result`
  - `fake` 后端下完整一次分布式任务 `completed`
  - AgileX `piper_single_ctrl` 能在 `can0` 上发布 `/arm_status` 并暴露 `/enable_srv`
- 分布式运行产物会落盘到工作区：
  - `ros_ws/log/distributed/<timestamp>/`
  - `logs/distributed_runs/<run_id>/`
- 单节点模式仍保留，但现在定位为兼容 / 对照 / 调试路径，不是推荐主线
- 真实机械臂执行链路已经具备前置条件，但还没有在本轮里做带运动的最终验收
- 通用 COCO prompt 保留宽松候选策略；六种目录物品采用严格目标模式，分割为 0 时禁止全场景 grasp 兜底，避免把其他可抓物误当作指定目标

## 文档导航

建议按下面的职责来读文档：

| 文档 | 作用 |
| --- | --- |
| `README.md` | 项目入口、快速上手、文档导航 |
| `docs/CURRENT_STATUS.md` | 当前可用基线、已验证证据、下一步建议 |
| `docs/DISTRIBUTED_RUNBOOK.md` | 分布式运行手册、常用命令、排障 |
| `../README_PIPER_ARM_DEBUG.md` | 当前 Jetson 的 Piper 机械臂、抓取与 Dashboard 调试流程 |
| `../README_SCOUT_BASE_DEBUG.md` | 当前 Jetson 的 Scout Mini 底盘、导航与单向扫描调试流程 |
| `docs/DISTRIBUTED_ARCHITECTURE.md` | 分布式架构、节点职责、接口边界 |
| `docs/MIGRATION_CONTRACT.md` | 迁移硬约束、分层边界、完成判定 |
| `docs/ENGINEERING_SPEC.md` | 代码规范、API 规范、运行约束 |
| `docs/MIGRATION_TODO.md` | 迁移 backlog、遗留项、参考记录 |
| `docs/ROBOT_COUPLING_MAP.md` | 旧工程机器人耦合点梳理 |
| `docs/PIPER_LOCAL_SIM.md` | 本机 Piper 非硬件仿真 / 模型调试说明 |
| `AGENTS.md` | AI / 工程师接手本仓库的读文件顺序与工作约定 |

## 关键算法入口

如果要快速定位“候选抓取从哪里来、抓取姿势在哪里生成、执行链路从哪里接下去”，建议先看下面这些入口：

| 入口 | 作用 |
| --- | --- |
| `src/perception/external_inference_worker.py::ExternalInferenceEngine.analyze` | 外部推理 worker 主入口；负责 `YOLOv8-seg -> 点云 -> GraspNet -> 实例级抓取筛选 -> candidate_pool` |
| `src/perception/external_inference_worker.py::_filter_scene_grasps_by_mask` | 把全场景 GraspNet 抓取按实例 mask 和深度一致性筛回各个目标实例 |
| `src/grasping/planning.py::PureGraspPlanner.collect_grasp_candidates` | 把实例级抓取整理成统一 `candidate_pool`，供后续规划与执行筛选 |
| `src/grasping/planning.py::PureGraspPlanner.plan_grasp` | 由 candidate 生成 `target/pregrasp/grasp/retreat` 等抓取执行位姿 |
| `robot_grasp_ros2/pipeline_orchestrator_node.py::_run_pipeline_thread` | distributed 主流程入口；串联 observation、capture、analyze、plan、confirm、execute |
| `src/grasping/coordinator.py::run_once` | 单节点兼容主流程入口；适合对照旧链路和做局部调试 |
| `robot_grasp_ros2/robot_executor_node.py::_handle_execute_grasp_plan` | distributed 执行入口；负责 `open -> pregrasp -> grasp -> target -> close -> retreat -> handoff/home` |
| `src/robot/client.py::Ros2PiperClient` | ROS2 机器人适配层；集中处理 `/pos_cmd`、`/end_pose`、`/arm_status`、`/enable_srv` 和 `mm/deg <-> m/rad` 转换 |

当前姿势链路语义已经理顺：

- `grasp` 表示沿工具轴先下探的预接触位
- `target` 表示补偿后的实际接触/闭爪位
- `retreat` 表示抓后抬升位
- distributed 与单节点现在统一按 `pregrasp -> grasp -> target -> close_gripper -> retreat` 校验和执行

## 快速开始

当前最推荐先跑“分布式 + fake 后端”，确认感知链路。

当前 Jetson 环境入口位于仓库根目录：

```bash
cd /home/nvidia/auto/Robot_arm
source source/scripts/ros_env_graspnet.sh
```

环境脚本会自动加载 ROS 2 Humble、两个工作空间、仓库 `.venv`、GraspNet
源码和 `graspnet/checkpoint-rs.tar`，不依赖固定用户名或 Conda 路径。真机
MoveIt 链路还要求系统已安装 `ros-humble-moveit`。

终端 A：

```bash
cd /home/nvidia/auto/Robot_arm/source
source scripts/ros_env_graspnet.sh
./scripts/run_distributed_stack_graspnet.sh --robot-backend fake
```

说明：

- 这个脚本会同时拉起四个节点
- 终端保持打开是正常行为
- 关闭这个终端会一起停掉整套分布式栈
- session 日志会写到 `ros_ws/log/distributed/<timestamp>/`

终端 B：

```bash
cd /home/nvidia/auto/Robot_arm/source
./scripts/ros2_system.sh service call /grasp_pipeline/probe std_srvs/srv/Trigger "{}"
./scripts/run_pipeline_service.sh cup
./scripts/show_last_distributed_snapshot.sh
./scripts/show_last_run_artifact.sh
```

说明：

- 这台机器上不要在 `source scripts/ros_env_graspnet.sh` 后直接执行裸 `ros2 ...`
- 运行 `ros2` CLI 时，统一使用 `./scripts/ros2_system.sh`
- `./scripts/run_pipeline_service.sh cup` 会先设置 `prompt`，再调用 `/grasp_pipeline/run`
- 直接执行 `ros2 service call /grasp_pipeline/run ...` 不会自动带上 prompt
- 当前分割器固定使用 `YOLOSegmenter`，默认权重为 `yolov8n-seg.pt`
- YOLOv8-seg 只能按 COCO 类别匹配 prompt；首次联调前应确认本地已有模型权重

## 当前机器推荐用法

如果你现在就在这台真机工作站上继续联调，推荐固定按下面顺序操作。

### 0. 一键启动并直接抓一次

如果你要的是“拉起真机整套栈 + 打开 RViz + 自动触发一次抓取”，现在可以直接用：

```bash
cd /home/ybw/piper_grasp_project/source
./scripts/run_live_grasp_one_click.sh
```
默认行为：

- 默认 prompt 是 `cup`
- 自动启动 `run_piper_driver.sh`
- 自动启动 `run_piper_moveit_ik.sh`
- 自动启动 `run_distributed_stack_graspnet.sh --robot-backend ros2 --pose-execution-mode moveit_ik`
- 自动打开 RViz
- 自动等待 `/compute_ik` 和 `/grasp_pipeline/probe` ready
- ready 后再显式调用 `run_pipeline_service.sh cup`
- 默认只做观察、感知、规划和 robot validation，不执行最终抓取
- 首次任务发出后整套栈保持常驻，方便继续重复测试

常用变体：

```bash
./scripts/run_live_grasp_one_click.sh bowl
./scripts/run_live_grasp_one_click.sh cup --execute
./scripts/run_live_grasp_one_click.sh cup --enable-pregrasp --precenter
./scripts/run_live_grasp_one_click.sh cup --no-rviz
```

如果只想“一条命令打一轮任务，拿到结果后自动收掉本 wrapper 启动的进程”，使用：

```bash
./scripts/run_one_grasp_task.sh cup --robot-backend fake --plan-only --no-rviz
./scripts/run_one_grasp_task.sh cup --robot-backend ros2 --no-rviz
./scripts/run_one_grasp_task.sh cup --robot-backend ros2 --execute --no-rviz
```

说明：

- `run_one_grasp_task.sh` 是 `run_live_grasp_one_click.sh --once` 的一次任务包装
- `--robot-backend fake --plan-only` 适合先验证感知和规划
- `--robot-backend ros2` 会按真机链路启动 driver、MoveIt IK、distributed stack，但默认只规划
- 真机最终执行必须显式加 `--execute`；ros2 执行会自动进入 confirm 等待，需要另开终端调用 `./scripts/confirm_pipeline_service.sh`

如果要在浏览器里输入 prompt、触发抓取、确认 / 拒绝执行，并回看一次任务的全流程可视化，用 dashboard：

```bash
cd /home/nvidia/auto/Robot_arm/source
./scripts/run_grasp_dashboard.py
```

打开 `http://127.0.0.1:8765`，可以先点“启动真机栈”拉起 Piper driver、MoveIt IK 和分布式抓取节点；这个启动动作不会自动触发抓取。等 pipeline / camera / vision / executor 状态变绿后，可以直接设置 prompt、速度、execute / confirm / precenter / pregrasp，并触发 `run`、`confirm`、`reject`、`stop`、`probe`。页面里的 `X/Y/Z 补偿 mm` 会在每次抓取前作为 base 坐标下的目标位姿微调下发，适合先验证 2-5mm 级系统偏差。页面也会显示最近 run 的分割图、GraspNet 投影、候选验证、规划路径、执行轨迹和节点日志。

Dashboard 默认使用“标识牌自动目标”模式：机械臂先到左侧观察位，对旁桌上单独摆放的
照片标识牌连续采集三帧，从 `config/item_catalog.yaml` 的六张参考图中识别黄/红/蓝物块或
橙/深/绿色饮料瓶。只有最佳类别达到置信度阈值、且明显优于第二候选时，系统才自动选择
对应 prompt 和瓶/物块抓取策略，然后启动既有底盘单向扫描；未识别或结果歧义时立即停止，
不会沿用上次人工目标。识别仅在固定观察位标定的卡片窗口内进行，并要求至少两帧得到
同一明确类别；透明盒、夹爪、椅子和窗口外的真实物品不参与竞争。每帧原图、检测叠加图
以及最终一致性结果保存在当前 run 的 `target_card/` 下。
取消“从标识牌照片自动识别抓取目标”后，仍可使用原来的六类人工选择作为诊断模式。
“识别对应盒标并放置”会在抓取后保持夹持，移动到盒标观察位，一次识别后侧竖直面上的
全部六个标识，按画面从左到右映射到六个盒位。盒子顺序可以变化；只有六标识完整且目标
槽位明确、置信度达标时才允许执行放置。

`dynamic_box_localization=true` 时会读取纸质盒标深度，将六个三维标识中心拟合成盒子长边
方向，校验相邻中心约 `180 mm`，再从后壁朝相机方向偏移盒深的一半 `66 mm` 得到目标盒
中心；不会用透明盒壁的无效深度。静态槽位只保留为显式配置路径，深度失效时不会盲放。

抓取观察位与放置观察位现已一起切回机械臂左侧：
`[0.0, 35.5, 491.1, 180.0, 67.77, -89.97] mm/deg`。左右两套标定分别保存在
`config/calibration/observation_poses/left_side.yaml` 和
`config/calibration/observation_poses/right_side.yaml`；右侧配置未删除，后续可复用。
原 `Z≈542 mm` 高观察位因不同关节分支可能触发 `ANGLE_LIMIT`，不再用于自动任务。
放置功能仍默认安全锁定：物品 `release_rpy_deg/release_offset_mm` 尚未全部完成真机标定，
目录中六项 `placement.enabled` 都为 `false`。未标定、缺少任一盒标或目标槽位不明确时，
pipeline 会拒绝任务，不会打开夹爪。

Dashboard 的“扫描放置区”是标定入口：它只读取当前机械臂位姿、采集一帧 RGB-D、
识别六个盒标并计算盒中心，然后显示 RGB、深度、标识顺序、相邻间距和 base 坐标。
该入口不会调用命名位姿、抓取计划、放置计划或夹爪控制。

盒标识别采用融合策略：COCO YOLOv8-seg 先提供瓶形候选，再按瓶内液体颜色区分橙色、
深色和绿色瓶；红黄蓝方块使用颜色、宽高比和轮廓，并由边缘/Lab 模板复核。YOLO
在这里只负责通用“瓶形”，不是用六张参考图直接训练自定义检测模型。

六盒无法同框时可使用“底盘单向扫描”。Scout Mini 驱动需单独在线并提供
`/cmd_vel`、`/odom`；先把底盘放在盒排左侧附近，页面确认前方 `1.5 m` 扫描通道和
硬件急停后，底盘沿前进方向每 `0.15 m` 停车采图，最远扫描 `1.5 m`、最多采 24 个
视角并停在扫描终点；发现目标后仍使用 `0.07 m` 细步完成居中。
系统利用相邻画面重叠标签的二维左右关系拼出六个固定槽位顺序，不使用透明盒深度；
标签缺失、顺序冲突或视角无重叠时拒绝结果并停车。当前此入口只验证标签到槽位编号，
扫描通过后可用“对准目标盒”让底盘分段返回到目标标签最接近画面中心的采集位置；
该动作不会移动机械臂或打开夹爪。在固定释放位姿完成标定前仍不会启用自动放置。

比赛瓶子真机默认执行策略是 `center_horizontal`：YOLOv8 分割得到瓶子几何中心，保留规划层 Z/人工补偿后，以已验证的水平夹爪姿态 `[180, 85, -90] deg` 执行“观察 -> 感知 -> 原地到安全过渡高度 -> 调平/横移 -> 分段垂直下降 -> 闭爪 -> 垂直抬升”。安全过渡高度取目标上方净空与全局安全下限的较大值；观察位更高时会先原地下降，避免在极高 Z 横移造成 IK 无解。Dashboard 默认勾选“瓶子中心水平抓取”，点“直接抓取”即可一次触发；取消勾选时回退到兼容的 `safe_top_down` 路径。

红黄蓝物块应使用 Dashboard 对应的“红色物块 / 黄色物块 / 蓝色物块”快捷按钮。
快捷按钮会自动关闭瓶子专用的“中心水平抓取”。首次验证必须点击“规划后确认”检查
分割掩膜与抓取落点；识别不到指定颜色时不会回退到全场景候选。

“抓取前底盘单向扫描指定物品”对六种目录物品采用严格停车条件：必须先得到该目标的
实例分割和有效抓取候选，并进入操作员标定的瓶/物块二维中心范围后才允许停车抓取。
参考中心分别为瓶子 `(0.598, 0.485)`、物块 `(0.606, 0.619)`，容差为水平 `±0.08`、
垂直 `±0.15`，不会苛求单一像素点。目标已出现但尚未居中时，底盘从 15 cm 粗搜索切换
为 7 cm 细步；未识别到目标或仅有无类别全场景抓取点时继续扫描。扫描抓到任一目录目标
（瓶子或物块）后，抓取执行器先完成闭爪和 retreat 抬升；随后机械臂移动到放置观察位
与底盘连续前进 1.5 m 同时进行。放置松爪并退出盒体后，底盘再连续前进 1.5 m。

偏置瓶位下的垂直下降默认按 `80 mm` 分段，典型路径约 3 段，避免在瓶口附近进行过多停顿和重复姿态调整；最终落点和 TCP 补偿不变。Dashboard Speed 会真实传递到 Piper 驱动，不再固定限制为 `5%`，但网页默认仍为安全的 `5%`，真机应逐步调速。

`center_horizontal` 还会根据瓶子中心相对 `base_link` 的方位角调整夹爪 yaw：正对左侧工作区时维持已验证的 `-90 deg`，瓶子偏左或偏右时让底座与夹爪方向同步旋转，避免腕部为了维持全局固定方向在瓶口附近做大幅补偿。TCP 会按调整后的姿态重新计算，因此目标中心不变。

清理这套真机联调进程：

```bash
./scripts/clear_live_grasp_nodes.sh --dry-run
./scripts/clear_live_grasp_nodes.sh
```

说明：

- 这个脚本不会在 distributed 启动阶段直接用 `--prompt` 做 auto-start
- 它会先等 `moveit_ik` 和 `/grasp_pipeline/probe` 起稳，再显式触发第一次任务
- 如果 driver / MoveIt IK / distributed / RViz 已经完整运行，它会直接复用现有进程而不是报冲突
- 如果只残留了半套 distributed 节点，它仍会拒绝启动，避免新旧节点混跑
- wrapper 自己的日志写到 `logs/one_click/<timestamp>/`
- distributed 内部节点日志仍然写到 `ros_ws/log/distributed/<timestamp>/`

### 1. 启动 Piper 驱动

终端 A：

```bash
cd /home/ybw/piper_grasp_project/source
./scripts/run_piper_driver.sh
```

作用：

- 负责 `can0 -> piper_ros`
- 提供 `/arm_status`、`/end_pose`、`/enable_srv`

### 2. 启动 MoveIt IK 包装层

终端 B：

```bash
cd /home/ybw/piper_grasp_project/source
./scripts/run_piper_moveit_ik.sh
```

作用：

- 提供 `/compute_ik`
- 给 distributed `robot_executor` 的 `moveit_ik` 模式使用

### 3. 启动 distributed 主线

终端 C：

```bash
source /home/ybw/piper_grasp_project/source/scripts/ros_env_graspnet.sh
cd /home/ybw/piper_grasp_project/source
./scripts/run_distributed_stack_graspnet.sh --robot-backend ros2 --pose-execution-mode moveit_ik
```

说明：

- 当前 `pipeline_orchestrator.params.yaml` 已带上本机验证过的桌面几何参数
- 不需要再手动设置 `table_z_m` / `workspace_z`

### 4. 做健康检查并触发一次任务

终端 D：

```bash
cd /home/ybw/piper_grasp_project/source
./scripts/ros2_system.sh service call /grasp_pipeline/probe std_srvs/srv/Trigger "{}"
./scripts/run_pipeline_service.sh cup
./scripts/show_last_run_artifact.sh
```

怎么看结果：

- 如果是 `status=ok` 或 `status=completed`，说明当前帧已经放出了有效候选
- 如果是 `status=no_candidate` 且 diagnostics 里出现 `no grasp after mask filtering`
  说明当前阻塞点在当前帧的 YOLOv8 实例 mask / 抓取候选重叠，不是桌面几何默认值问题
- 如果是 `status=no_candidate` 且 diagnostics 里出现 `workspace` / `table_z_m`
  说明运行参数没有吃到最新配置，优先重启 distributed 栈再看

### 5. 真正执行抓取

如果只是看 plan，不要加 `--execute`。

如果要真机执行，必须先走人工确认：

```bash
source /home/ybw/piper_grasp_project/source/scripts/ros_env_graspnet.sh
cd /home/ybw/piper_grasp_project/source
./scripts/run_distributed_stack_graspnet.sh --robot-backend ros2 --pose-execution-mode moveit_ik --execute --confirm --prompt cup
```

出现 `awaiting_confirmation` 后，在另一个终端确认：

```bash
./scripts/confirm_pipeline_service.sh
```

然后在另一个终端：

```bash
./scripts/confirm_pipeline_service.sh
```

### 6. 从室内导航抓取点自动交接

导航仓库的 `run_indoor_recorded_route.sh` 已与本仓库的
`scripts/run_navigation_grasp_handoff.sh` 桥接。导航会先到抓取预备点，再以普通安全容差
到达抓取点（位置 `±0.15 m`、航向 `±0.25 rad`）；只有两个 Nav2 动作均成功后，才允许
`/grasp_pipeline` 接管标识牌识别、目标细扫与抓取。导航失败、抓取栈未就绪、
`/odom` 缺失或标识牌识别不明确时均不会启动目标扫描或抓取。

先启动并确认 Dashboard 真机栈就绪，再在已加载导航工作区的终端运行：

```bash
ros2 run scout_navigation_bringup run_indoor_recorded_route.sh
```

路线参数不再作为抓取类别传给机械臂；到达抓取区后，每次都重新拍摄照片标识牌确定目标。
`scripts/run_navigation_grasp_handoff.sh --target red_block` 仅保留为人工诊断回退。桥接会显式
启用真机执行、标识牌识别、抓取前底盘细扫以及抓取后的观察位动作，并等待最终结果。

`run_indoor03_recorded_route.sh` 会在所有预检通过后先重放已验证的红旗观察位，并等待
挥旗识别成功，之后才发送第一个导航目标。该路线进程现在常驻持有抓取节点的 ROS 服务
客户端和结果订阅：完整健康检查只在挥旗前执行一次；到达取件点后用一条批量参数请求和
一次服务调用立即开始标识牌识别/抓取，并从结果话题直接获得完成通知，不再逐条启动
`ros2 param/service` 命令或轮询结果文件。到达放置点后同样直接调用盒标扫描、对准与释放
服务。本仓库的两个 `run_navigation_*_handoff.sh` 继续保留为人工单步诊断入口，不再处于
indoor_03 到点后的主时延路径。观察位回放单独使用 `10%` 速度，近桌抓取速度保持原配置。

## 常用运行方式

### 裁判红旗启动门

室内录制路线在发出第一个 Nav2 目标前，通过 `/camera_server/capture` 监测裁判挥舞红旗。
新示教位已从实时 Piper 反馈记录，并在正常 CAN 控制下以 5% 速度验证可达。路线默认先
移动到该红旗观察位，再等待挥旗；仅台架调试可设置 `RED_FLAG_MOVE_TO_OBSERVATION=0`
跳过自动到位。
检测同时要求足够大的高饱和红色区域、三秒窗口内的明显位移、累计轨迹长度和至少一次
运动方向反转；静止红旗、棕红色椅子或红色物块不会单独触发。检测成功只放行一次当前
路线，不直接发布 `/cmd_vel`。

正常运行仍使用：

```bash
ros2 run scout_navigation_bringup run_indoor_recorded_route.sh
```

调试检测器但不启动导航：

```bash
cd /home/nvidia/auto/Robot_arm/source
./scripts/wait_for_red_flag_start.sh
```

只有台架调试时才允许临时设置 `RED_FLAG_START_ENABLED=0` 绕过启动门。标定参数和三组
现场样本位于 `config/calibration/red_flag/`，检测结果写入其 `runtime/` 子目录。

### 1. 分布式假后端

```bash
./scripts/run_distributed_stack_graspnet.sh --robot-backend fake --prompt cup
```

适合：

- 无真机验证
- 检查 `capture -> analyze -> plan/no_candidate`
- 验证 RViz 和结构化结果发布

### 2. 分布式真机执行

```bash
./scripts/run_piper_driver.sh
```

另一个终端：

```bash
./scripts/run_distributed_stack_graspnet.sh --robot-backend ros2 --prompt cup --execute --enable-pregrasp
```

如果要把 `robot_executor` 的位姿执行切到 ROS 侧 MoveIt IK，再开一个终端先启动：

```bash
./scripts/run_piper_moveit_ik.sh
```

然后用：

```bash
./scripts/run_distributed_stack_graspnet.sh --robot-backend ros2 --pose-execution-mode moveit_ik --prompt cup
```

前提：

- `./scripts/run_piper_driver.sh` 已启动
- 如果使用 `moveit_ik`，`./scripts/run_piper_moveit_ik.sh` 已启动
- `can0` 可用
- `/arm_status`、`/end_pose`、`/enable_srv` 已存在
- D435 没被别的程序占用

重要提醒：

- `--robot-backend ros2` 即使不带 `--execute`，当前 orchestrator 仍会先移动到 observation pose
- 所以真机联调前，先确认工作空间和安全区域
- 当前 `moveit_ik` 路径已经能稳定拉起 `/compute_ik`
- 本机最新实测里，直接对“当前回读位姿”和“z + 10 mm”做 `/compute_ik` 探针都已返回成功
- `robot_executor` 的 `pose_execution_mode:=moveit_ik` 已补上独立 `ReentrantCallbackGroup`，避免 service 回调里收不到 `/joint_states_feedback`
- 默认 `moveit_ik_timeout_s` 已提高到 `5.0`；真机上建议在 `run_piper_moveit_ik.sh` 启动后先预留约 `10s` 再发第一条 MoveIt IK 命令
- 2026-04-28 的最小真机抬升验证里，机械臂最终从 `z=169.502 mm` 抬到了 `z=178.684 mm`，距离原始 `+10 mm` 目标 `179.502 mm` 还差约 `0.818 mm`

### 3. 先出 plan，再人工确认执行

```bash
./scripts/run_distributed_stack_graspnet.sh --robot-backend ros2 --prompt cup --execute --confirm
```

出现 `awaiting_confirmation` 后：

```bash
./scripts/confirm_pipeline_service.sh
```

如果这次计划不执行：

```bash
./scripts/reject_pipeline_service.sh
```

### 4. 单节点兼容模式

```bash
source /home/ybw/piper_grasp_project/source/scripts/ros_env_graspnet.sh
cd /home/ybw/piper_grasp_project/source
./scripts/run_grasp_pipeline_node_graspnet.sh
```

说明：

- 这是兼容旧“单 node 串行执行”方式的路径
- 现在主要用于迁移对照和局部调试
- 不是默认推荐的系统主线

### 5. 交互式 Piper 调试

先确保已 source Jazzy 和本地 `grasp_ros/install` overlay，再启动交互式 teleop。

这条链路对应：

- `piper_interactive_marker_node`
- `piper_pose_bridge_node`
- `joint_state_feedback_relay_node`

它的作用是把 RViz 的交互 marker 转成 `PoseStamped`，再桥接到 `Ros2PiperClient`。

### 6. 本机 Piper 本地模型调试

如果当前没有 `can0`，又需要先把 Piper 模型链在本机拉起来，可以走 `piper_description` 的本地交互路径。

入口命令和限制说明见：

- `docs/PIPER_LOCAL_SIM.md`

当前这条链已经在本机验证：

- `joint_state_publisher_gui`
- `robot_state_publisher`
- `rviz2`

重要区别：

- 这是本地模型 / RViz 调试链，不是真实机械臂驱动链
- 它不能替代 `/arm_status`、`/end_pose`、`/enable_srv` 这些真实 `piper_ros` 接口

## 分布式架构摘要

当前推荐主线的节点职责如下：

1. `/grasp_pipeline`
   - 对外统一入口
   - 负责 `run_id`、状态机、最终抓取规划
2. `/camera_server`
   - 负责采集 RealSense RGBD
3. `/vision_worker`
   - 负责 YOLOv8-seg、点云重建、GraspNet、抓取候选筛选
   - 负责发布 RViz 可视化 topic
4. `/robot_executor`
   - 负责 `fake` / `ros2` 两种执行后端
   - 真机模式下只通过 `piper_ros` 接机械臂

当前已经稳定的对外控制面：

- `/grasp_pipeline/run`
- `/grasp_pipeline/probe`
- `/grasp_pipeline/stop`
- `/grasp_pipeline/confirm`
- `/grasp_pipeline/reject`
- `/grasp_pipeline/status`
- `/grasp_pipeline/summary`
- `/grasp_pipeline/diagnostics`
- `/grasp_pipeline/result_json`

当前 action 定义已经存在，但还没有替代当前入口：

- `robot_grasp_msgs/action/RunGraspPipeline.action`

也就是说，现阶段真正接通的外部入口仍然是：

- `Trigger + prompt parameter/topic`

## RViz 可视化

推荐直接打开现成配置：

```bash
cd /home/ybw/piper_grasp_project/source
./scripts/open_distributed_rviz.sh
```

分布式模式下，请看 `/vision_worker/rviz/*`，不是 `/grasp_pipeline/rviz/*`。

当前重点 topic：

- `/vision_worker/rviz/scene_pointcloud`
- `/vision_worker/rviz/instance_pointcloud`
- `/vision_worker/rviz/candidate_markers`
- `/vision_worker/rviz/selected_grasp_markers`
- `/vision_worker/rviz/plan_markers`
- `/vision_worker/rviz/camera_transform`
- `/tf`

推荐的 RViz 使用顺序：

1. `Fixed Frame` 先设为 `camera_color_optical_frame`
2. 先看点云和 marker
3. 跑过一次 pipeline 且 `/tf` 正常后，再切成 `base_link`

补充说明：

- 如果结果是 `no_candidate`，`selected_grasp*` 和 `plan_*` 为空是正常的
- 即使 `no_candidate`，点云 topic 仍应保留最后一次感知结果
- 当前 `result_json`、`status`、`summary` 等关键 topic 使用了 transient-local QoS，任务结束后仍可以回看最后一条结果

## 产物与日志

当前有两类日志和产物目录：

1. 分布式 session 日志
   - `ros_ws/log/distributed/<timestamp>/`
   - 保存每个节点的 stdout / stderr 日志
2. 单次 run 的结构化产物
   - `logs/distributed_runs/<run_id>/`
   - 当前至少包含：
     - `request.json`
     - `cycles.json`
     - `final_result.json`

常用查看脚本：

```bash
./scripts/show_last_distributed_snapshot.sh
./scripts/show_last_run_artifact.sh
```

## 当前约束

- 主写入区是 `grasp_ros/robot_grasp_ros2`
- `old/robot_grasp`、`Agilex-College`、`robotic_arm_kinematics`、`piper_ros_humble` 默认视为参考输入
- 新工程禁止直接依赖 `piper_sdk`
- 所有硬件访问必须经过 `src/robot/`
- 新增 topic / service / 参数 / 使用方式时，必须同步更新 `docs/`

## 建议读文件顺序

如果你要继续开发或接手这个仓库，建议按这个顺序建立上下文：

1. `AGENTS.md`
2. `README.md`
3. `docs/CURRENT_STATUS.md`
4. `docs/DISTRIBUTED_RUNBOOK.md`
5. `docs/DISTRIBUTED_ARCHITECTURE.md`
6. `docs/MIGRATION_CONTRACT.md`
7. `docs/ENGINEERING_SPEC.md`
8. `docs/MIGRATION_TODO.md`
9. `docs/ROBOT_COUPLING_MAP.md`
10. 与当前任务直接相关的 `src/` 或 `robot_grasp_ros2/` 文件
