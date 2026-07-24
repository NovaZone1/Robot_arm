# Robot Arm — Piper Grasp Project

> 面向无人系统竞赛的机器人抓取项目，集成 **Piper 机械臂** + **ROS2 Humble** + **GraspNet** 实现物体识别、抓取规划与机械臂执行控制的完整闭环。

[![Ubuntu](https://img.shields.io/badge/OS-Ubuntu%2022.04-orange)](https://ubuntu.com/)
[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-yellow)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-required-green)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-Proprietary-red)](LICENSE)

---

## 目录

- [项目概述](#项目概述)
- [系统架构](#系统架构)
- [仓库目录结构](#仓库目录结构)
- [硬件与软件要求](#硬件与软件要求)
- [安装指南](#安装指南)
- [快速开始](#快速开始)
- [运行模式详解](#运行模式详解)
  - [分布式模式（推荐主线）](#分布式模式推荐主线)
  - [单节点兼容模式](#单节点兼容模式)
  - [一键启动](#一键启动)
- [模块详解](#模块详解)
  - [感知层 (Perception)](#感知层-perception)
  - [抓取规划层 (Grasping)](#抓取规划层-grasping)
  - [机器人适配层 (Robot)](#机器人适配层-robot)
  - [工具层 (Utils)](#工具层-utils)
  - [ROS2 分布式节点](#ros2-分布式节点)
  - [Piper SDK 与驱动](#piper-sdk-与驱动)
  - [GraspNet 基线模型](#graspnet-基线模型)
- [配置说明](#配置说明)
- [Dashboard 操控面板](#dashboard-操控面板)
- [RViz 可视化](#rviz-可视化)
- [抓取执行流程](#抓取执行流程)
- [运行产物与回看](#运行产物与回看)
- [调试与故障排查](#调试与故障排查)
- [开发指南](#开发指南)
- [文档索引](#文档索引)
- [已知限制与下一步](#已知限制与下一步)
- [贡献](#贡献)
- [许可证](#许可证)
- [作者](#作者)

---

## 项目概述

本项目为无人系统竞赛设计，实现了一套完整的 **"感知 → 规划 → 执行"** 机器人抓取流水线。系统使用 **Intel RealSense D435** 深度相机采集 RGB-D 图像，通过 **YOLOv8-seg** 进行实例分割，利用 **GraspNet** 预测 6-DOF 抓取姿态，最终经由 **ROS2 Humble** 控制 **Piper 机械臂** 完成物体抓取。

### 核心特性

- 🎯 **ROS2 分布式架构**：四节点分离式设计，支持同机/多机部署
- 👁️ **视觉感知**：RealSense D435 RGB-D 采集 + YOLOv8-seg 实例分割
- 🤖 **GraspNet 抓取预测**：6-DOF 抓取姿态生成与筛选
- 🦾 **Piper 机械臂控制**：完整的 SDK + ROS2 驱动 + MoveIt IK
- 📊 **Web 操控面板**：内置 Dashboard 实现一键抓取与实时监控
- 📹 **RViz 可视化**：场景点云、实例点云、候选抓取位姿、执行路径全可视化
- 📝 **完整运行产物**：每次抓取自动落盘结构化 JSON 产物，支持离线回看
- 🧪 **双后端支持**：`fake`（无硬件调试）和 `ros2`（真机执行）两种模式

---

## 系统架构

### 数据流

```
┌──────────┐    ┌──────────────┐    ┌──────────────────┐    ┌──────────────┐    ┌────────────┐
│  Camera  │───▶│  Perception  │───▶│  GraspNet Pose   │───▶│   Motion     │───▶│   Piper    │
│  (D435)  │    │ (YOLOv8-seg) │    │   Prediction     │    │  Planning    │    │ Robot Arm  │
└──────────┘    └──────────────┘    └──────────────────┘    └──────────────┘    └────────────┘
                                                                                      │
                                                                                      ▼
                                                                              ┌────────────┐
                                                                              │   Object   │
                                                                              │  Grasping  │
                                                                              └────────────┘
```

### 分布式节点拓扑（推荐）

```
┌─────────────────────────┐   ┌──────────────────────┐   ┌─────────────────────────┐
│     Operator Host       │   │     Vision Host      │   │      Robot Host         │
│                         │   │                      │   │                         │
│  pipeline_orchestrator  │◀──│  vision_worker_node  │◀──│  robot_executor_node    │
│  RViz2 / Dashboard      │   │  camera_server_node  │   │  piper_ros (driver)     │
│  Trigger Scripts        │   │  YOLOv8-seg / GraspNet│  │  MoveIt IK              │
└─────────────────────────┘   └──────────────────────┘   └─────────────────────────┘
```

四个核心 ROS2 节点：

| 节点 | 服务名 | 职责 |
|------|--------|------|
| `pipeline_orchestrator_node` | `/grasp_pipeline` | 统一入口、状态机、抓取规划编排 |
| `camera_server_node` | `/camera_server` | 按需采集 RGB-D 图像 |
| `vision_worker_node` | `/vision_worker` | YOLOv8-seg 分割、点云重建、GraspNet 推理、RViz 发布 |
| `robot_executor_node` | `/robot_executor` | `fake` / `ros2` 双后端运动执行 |

### 分层架构设计

```
┌─────────────────────────┐
│   Entry / Launch        │  ← ROS2 launch、CLI、Dashboard
├─────────────────────────┤
│   Pipeline (grasping/)  │  ← 抓取编排、候选管理、流程协调
├─────────────────────────┤
│   Planning              │  ← 位姿规划、工作区校验、候选筛选（mm/deg）
├─────────────────────────┤
│   Robot Adapter         │  ← ROS2 适配层、单位转换（mm/deg ↔ m/rad）
├─────────────────────────┤
│   Perception            │  ← 相机、分割、GraspNet（无 ROS 依赖）
└─────────────────────────┘
```

**硬约束：**
- 规划层使用 `mm/deg`，ROS2 适配层负责 `mm/deg ↔ m/rad` 转换
- 新业务逻辑禁止直接依赖 `piper_sdk`
- 所有硬件访问必须经过 `src/robot/` 抽象层

---

## 仓库目录结构

```
Robot_arm/
├── source/                              # 🎯 抓取任务核心源码（主写入区）
│   ├── config/                          # 配置文件
│   │   ├── distributed/                 #   分布式节点参数
│   │   │   ├── camera_server.params.yaml
│   │   │   ├── pipeline_orchestrator.params.yaml
│   │   │   ├── robot_executor.params.yaml
│   │   │   └── vision_worker.params.yaml
│   │   ├── hand_eye/                    #   手眼标定文件
│   │   │   ├── verify_config.yaml
│   │   │   └── verify_config_eyeinhand_cam2tcp.yaml
│   │   ├── grasp_pipeline.params.yaml
│   │   └── piper_moveit_ik_joint_limits.yaml
│   │
│   ├── docs/                            # 📚 项目文档
│   │   ├── CURRENT_STATUS.md            #   当前状态与已验证基线
│   │   ├── DISTRIBUTED_RUNBOOK.md       #   分布式运行手册
│   │   ├── DISTRIBUTED_ARCHITECTURE.md  #   分布式架构设计
│   │   ├── ENGINEERING_SPEC.md          #   工程规范
│   │   ├── MIGRATION_CONTRACT.md        #   迁移契约
│   │   ├── MIGRATION_TODO.md            #   迁移待办清单
│   │   ├── CODE_STATUS_MAP.md           #   代码状态映射
│   │   ├── ROBOT_COUPLING_MAP.md        #   机器人耦合映射
│   │   └── PIPER_LOCAL_SIM.md           #   本地仿真说明
│   │
│   ├── launch/                          # ROS2 Launch 文件
│   │   ├── distributed_grasp_pipeline.launch.py
│   │   ├── grasp_pipeline.launch.py
│   │   ├── piper_interactive_teleop.launch.py
│   │   └── piper_moveit_ik.launch.py
│   │
│   ├── scripts/                         # 🛠️ 运行与工具脚本
│   │   ├── run_distributed_stack_graspnet.sh  # 分布式栈一键启动
│   │   ├── run_live_grasp_one_click.sh        # 真机一键启动
│   │   ├── run_one_grasp_task.sh              # 单次抓取任务
│   │   ├── run_grasp_dashboard.py             # Web Dashboard
│   │   ├── run_piper_driver.sh                # Piper 驱动启动
│   │   ├── run_piper_moveit_ik.sh             # MoveIt IK 包装层
│   │   ├── ros_env_graspnet.sh                # ROS + GraspNet 环境
│   │   ├── ros2_system.sh                     # ROS2 CLI 环境
│   │   ├── run_grasp_pipeline_node_graspnet.sh
│   │   ├── open_distributed_rviz.sh           # RViz 启动
│   │   ├── show_last_distributed_snapshot.sh  # 最新快照查看
│   │   ├── show_last_run_artifact.sh          # 最新产物查看
│   │   ├── clear_live_grasp_nodes.sh          # 进程清理
│   │   ├── confirm_pipeline_service.sh        # 确认执行
│   │   ├── reject_pipeline_service.sh         # 拒绝执行
│   │   ├── run_pipeline_service.sh            # 触发抓取
│   │   ├── probe_robot_graspnet.sh            # 健康探测
│   │   ├── run_fake_graspnet.sh               # Fake 后端启动
│   │   ├── start_piper_single_graspnet.sh
│   │   └── calibrate_hand_eye.py              # 手眼标定
│   │
│   ├── src/                             # 🧠 核心 Python 源码
│   │   ├── grasping/                    #   抓取规划层
│   │   │   ├── coordinator.py           #     主协调器
│   │   │   ├── models.py                #     数据模型
│   │   │   └── planning.py              #     位姿规划、候选筛选
│   │   ├── perception/                  #   感知层
│   │   │   ├── realsense_rgbd.py        #     RealSense D435 驱动
│   │   │   ├── yolo_segmenter.py        #     YOLOv8-seg 分割器
│   │   │   ├── graspnet_runner.py       #     GraspNet 推理封装
│   │   │   ├── geometry.py              #     点云处理与滤波
│   │   │   ├── external_camera_capture_worker.py
│   │   │   └── external_inference_worker.py
│   │   ├── robot/                       #   机器人适配层
│   │   │   ├── client.py                #     RobotArmClient 抽象接口
│   │   │   ├── types.py                 #     机器人数据类型
│   │   │   ├── motion_tolerances.py     #     运动容差配置
│   │   │   ├── moveit_ik.py             #     MoveIt IK 执行器
│   │   │   ├── executor_models.py       #     执行器数据模型
│   │   │   └── plan_validation.py       #     规划校验
│   │   ├── utils/                       #   工具层
│   │   │   ├── transforms.py            #     坐标变换
│   │   │   ├── calibration.py           #     标定工具
│   │   │   └── npoint_tool_offset.py    #     工具偏移计算
│   │   └── run_grasp_pipeline_ros2.py   #   单节点入口
│   │
│   ├── robot_grasp_ros2/               # 🔌 ROS2 节点实现
│   │   ├── pipeline_orchestrator_node.py    # 编排节点
│   │   ├── camera_server_node.py            # 相机服务节点
│   │   ├── vision_worker_node.py            # 视觉工作节点
│   │   ├── robot_executor_node.py           # 机器人执行节点
│   │   ├── grasp_pipeline_node.py           # 单节点兼容模式
│   │   ├── piper_pose_bridge_node.py        # 位姿桥接
│   │   ├── piper_interactive_marker_node.py # 交互式 Marker
│   │   ├── joint_state_feedback_relay_node.py
│   │   ├── rviz_visualization.py            # RViz 可视化发布
│   │   ├── distributed_utils.py             # 分布式工具
│   │   ├── clear_live_grasp_nodes.py        # 节点清理
│   │   └── live_grasp_one_click.py          # 一键启动封装
│   │
│   ├── robot_grasp_msgs/               # 📨 ROS2 消息定义
│   │   ├── msg/
│   │   │   ├── GraspCandidate.msg       #     抓取候选消息
│   │   │   ├── GraspPlan.msg            #     抓取计划消息
│   │   │   ├── PerceptionSummary.msg    #     感知摘要消息
│   │   │   ├── PipelineStatus.msg       #     流水线状态消息
│   │   │   └── Pose6D.msg               #     6D 位姿消息
│   │   └── srv/
│   │       ├── AnalyzeScene.srv         #     场景分析服务
│   │       ├── CaptureScene.srv         #     场景采集服务
│   │       ├── ExecuteGraspPlan.srv     #     抓取执行服务
│   │       ├── ExecuteNamedPose.srv     #     命名位姿执行服务
│   │       ├── GetRobotState.srv        #     机器人状态服务
│   │       └── StopRobot.srv            #     停止机器人服务
│   │
│   ├── rviz/                           # RViz 配置文件
│   │   └── distributed_grasp_pipeline.rviz
│   ├── resource/                       # 资源文件
│   │   └── robot_grasp_ros2
│   ├── test/                           # 🧪 测试文件
│   │   ├── test_clear_live_grasp_nodes.py
│   │   ├── test_coordinator_execution.py
│   │   ├── test_grasp_candidate_pose_floor.py
│   │   ├── test_hand_eye_calibration_math.py
│   │   ├── test_live_grasp_one_click_runner.py
│   │   ├── test_motion_tolerances.py
│   │   ├── test_moveit_ik_executor.py
│   │   ├── test_pipeline_orchestrator_artifacts.py
│   │   ├── test_plan_validation.py
│   │   ├── test_robot_executor_node.py
│   │   ├── test_runtime_config_defaults.py
│   │   ├── test_rviz_candidate_validation_markers.py
│   │   └── test_rviz_pose_visualization_defaults.py
│   │
│   ├── AGENTS.md                       # AI/工程师入门引导
│   ├── README.md                       # 项目入口文档
│   ├── package.xml                     # ROS2 包清单
│   ├── setup.py                        # Python 安装脚本
│   ├── setup.cfg                       # Python 包配置
│   └── yolov8n-seg.pt                  # YOLOv8-seg 分割模型权重
│
├── ros_ws/                             # ROS2 工作空间（编译入口）
│   └── src/                            #   指向 source/ 中 ROS package 的软链接
│
├── piper_ros_ws/                       # Piper ROS 驱动工作空间
│   └── src/
│       └── piper_ros/                  #   Piper ROS2 驱动包
│
├── piper_sdk/                          # Piper 底层控制 SDK
│   ├── piper_sdk/                      #   SDK 库代码
│   │   └── ...
│   ├── asserts/                        #   文档资源（接口说明、Q&A、双臂配置）
│   ├── README.MD                       #   英文用户手册
│   ├── README(ZH).MD                   #   中文用户手册
│   ├── CHANGELOG.MD                    #   更新日志
│   ├── DESCRIPTION.MD                  #   项目描述
│   ├── LICENSE                         #   SDK 许可证
│   ├── MANIFEST.in                     #   Python 包清单
│   ├── setup.py                        #   Python 安装脚本
│   └── rm_tmp.sh                       #   临时文件清理脚本
│
├── graspnet/                           # GraspNet 基线模型
│   ├── graspnetAPI/                    #   GraspNet API 库
│   ├── models/                         #   模型定义
│   ├── pointnet2/                      #   PointNet++ 算子
│   ├── knn/                            #   KNN 算子
│   ├── dataset/                        #   数据集处理
│   ├── utils/                          #   工具函数
│   ├── doc/                            #   文档
│   ├── demo.py                         #   演示脚本
│   ├── train.py                        #   训练脚本
│   ├── test.py                         #   测试/评估脚本
│   ├── command_demo.sh                 #   演示命令
│   ├── command_train.sh                #   训练命令
│   ├── command_test.sh                 #   测试命令
│   ├── requirements.txt                #   Python 依赖
│   ├── README.md                       #   GraspNet 说明
│   └── LICENSE                         #   GraspNet 许可证
│
├── logs/                               # 运行日志与产物（gitignore）
│   └── distributed_runs/<run_id>/      #   结构化单次运行产物
│       ├── request.json                #     请求参数
│       ├── cycles.json                 #     运行周期记录
│       ├── final_result.json           #     最终结果
│       ├── candidate_validation.json   #     候选验证明细
│       └── execution_trace.json        #     执行轨迹
│
├── tmp/                                # ROS 临时文件（gitignore）
├── .gitignore
└── README.md
```

---

## 硬件与软件要求

### 硬件

| 组件 | 型号/规格 |
|------|----------|
| 机械臂 | Piper Robot Arm（AgileX） |
| 深度相机 | Intel RealSense D435 |
| CAN 适配器 | USB-CAN 模块（gs_usb） |
| GPU | NVIDIA GPU（GraspNet 推理需要 CUDA） |
| 工控机/PC | x86_64，建议 ≥16GB RAM |

### 软件

| 软件 | 版本 |
|------|------|
| 操作系统 | Ubuntu 22.04 LTS |
| ROS2 | Humble (或 Jazzy) |
| Python | 3.10+ |
| CUDA | 11.x+（GraspNet 推理） |
| PyTorch | 1.6+ |
| Open3D | ≥0.8 |
| OpenCV | 4.x |

### Python 关键依赖

- **ROS2**: `rclpy`, `std_msgs`, `sensor_msgs`, `geometry_msgs`, `visualization_msgs`, `tf2_ros`, `interactive_markers`
- **MoveIt2**: `moveit_msgs`, `moveit_configs_utils`, `moveit_ros_move_group`
- **感知**: `ultralytics` (YOLOv8), `numpy`, `scipy`, `opencv-python`, `open3d`
- **机械臂**: `python-can` (≥3.3.4), `piper_sdk`
- **其他**: `pyrealsense2`, `pillow`, `tqdm`, `pyyaml`

---

## 安装指南

### 1. 克隆仓库

```bash
git clone https://github.com/NovaZone1/Robot_arm.git
cd Robot_arm
```

### 2. 安装 ROS2 Humble

```bash
# 安装 ROS2 Humble（参考官方文档）
# https://docs.ros.org/en/humble/Installation.html
sudo apt install ros-humble-desktop python3-colcon-common-extensions
```

### 3. 创建 Conda 环境（用于感知栈）

```bash
conda create -n piper python=3.10
conda activate piper
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics open3d opencv-python numpy scipy pillow tqdm pyyaml
```

### 4. 安装 Piper SDK

```bash
# 确认 conda 环境已激活
pip install python-can>=3.3.4
cd piper_sdk
pip install -e .
```

### 5. 编译 ROS2 工作空间

```bash
# 需要先安装 GraspNet 的 pointnet2 和 knn 算子
cd graspnet/pointnet2
python setup.py install
cd ../knn
python setup.py install
cd ../..

# 编译 ROS2 workspace（使用系统 Python，避开 conda）
# 先确保取消 conda 环境变量干扰
source /opt/ros/humble/setup.bash
cd ros_ws
colcon build --symlink-install
source install/setup.bash
```

### 6. 编译 Piper ROS 驱动

```bash
cd piper_ros_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install
source install/setup.bash
```

### 7. 下载模型权重

- **YOLOv8-seg**: `yolov8n-seg.pt` 会在首次运行时自动下载，也可手动放置于 `source/` 目录
- **GraspNet 预训练权重**: 推荐使用 `checkpoint-rs.tar`（RealSense 相机），从 [Google Drive](https://drive.google.com) 或 [百度网盘](https://pan.baidu.com) 下载后解压至 `graspnet/` 目录

---

## 快速开始

### 最短路径：Fake 后端验证

无需机械臂和相机，验证数据流全链路：

**终端 A — 启动分布式栈：**

```bash
source /opt/ros/humble/setup.bash
cd Robot_arm/source
./scripts/ros_env_graspnet.sh
./scripts/run_distributed_stack_graspnet.sh --robot-backend fake --prompt cup
```

**终端 B — 触发抓取任务：**

```bash
cd Robot_arm/source
./scripts/ros2_system.sh
ros2 service call /grasp_pipeline/run std_srvs/srv/Trigger "{}"
```

**终端 C（可选）— 启动 RViz 可视化：**

```bash
cd Robot_arm/source
./scripts/open_distributed_rviz.sh
```

### 查看结果

```bash
# 查看最新运行快照
./scripts/show_last_distributed_snapshot.sh

# 查看结构化产物
./scripts/show_last_run_artifact.sh
```

### 真机完整启动（4 终端方案）

**终端 A — Piper 驱动：**
```bash
./scripts/run_piper_driver.sh
```

**终端 B — MoveIt IK 包装层：**
```bash
./scripts/run_piper_moveit_ik.sh
```

**终端 C — 分布式栈：**
```bash
./scripts/ros_env_graspnet.sh
./scripts/run_distributed_stack_graspnet.sh --robot-backend ros2 --execute --confirm
```

**终端 D — Dashboard 或任务触发：**
```bash
# 启动 Web 面板
python scripts/run_grasp_dashboard.py

# 或直接触发任务
./scripts/ros2_system.sh
./scripts/run_pipeline_service.sh cup
```

---

## 运行模式详解

### 分布式模式（推荐主线）

四个 ROS2 节点组成分布式架构，这是项目推荐的标准运行模式：

| 节点 | 启动方式 | 默认后端 |
|------|---------|---------|
| `pipeline_orchestrator_node` | `run_distributed_stack_graspnet.sh` | — |
| `camera_server_node` | 同上（自动启动） | RealSense D435 |
| `vision_worker_node` | 同上（自动启动） | YOLOv8-seg + GraspNet |
| `robot_executor_node` | 同上（自动启动） | `fake` 或 `ros2` |

**常用启动参数：**

```bash
# 基础
--robot-backend fake         # 无硬件调试模式
--robot-backend ros2         # 真机模式

# 抓取配置
--prompt cup                 # 目标物体 COCO 类别名（cup / bowl / bottle ...）
--execute                    # 开启运动执行
--confirm                    # 执行前等待确认
--enable-pregrasp            # 启用预抓取位
--precenter                  # 启用预居中循环

# 比赛瓶子自动流程
# 默认启用 center_horizontal 策略：
#   - YOLO 实例中心 + GraspNet 接触点深度融合
#   - 水平夹爪姿态 [180, 85, -90] deg
#   - 80mm 级分段垂直下降
#   - 抓后保持夹持

# 跳过观察位移（机械臂已在观察位时）
--skip-observation-move
```

### 单节点兼容模式

```bash
python src/run_grasp_pipeline_ros2.py --robot-backend fake --prompt cup
```

> 单节点模式仅保留用于兼容对照和局部调试，不推荐作为日常使用方式。

### 一键启动

#### 真机一键启动

```bash
./scripts/run_live_grasp_one_click.sh
```

该脚本自动完成：驱动启动 → MoveIt IK 启动 → 分布式栈启动 → RViz 启动 → readiness 等待 → 触发首个任务。

#### 单次抓取任务

```bash
./scripts/run_one_grasp_task.sh --prompt cup --once
```

---

## 模块详解

### 感知层 (Perception)

位于 `source/src/perception/`，负责 RGB-D 数据采集、实例分割和抓取姿态预测。

| 文件 | 功能 |
|------|------|
| `realsense_rgbd.py` | RealSense D435 相机驱动：启动、配置、帧采集、内参管理 |
| `yolo_segmenter.py` | YOLOv8-seg 实例分割：根据文本 prompt 按 COCO 类别名匹配目标，输出 mask / box / score |
| `graspnet_runner.py` | GraspNet 推理封装：checkpoint 解析、模型加载、抓取位姿预测 |
| `geometry.py` | 点云处理：深度图转场景点云、双边/中值/孤岛/半径滤波、最大聚类提取、3D 可视化 |
| `external_camera_capture_worker.py` | 外部进程相机采集 Worker |
| `external_inference_worker.py` | 外部进程推理 Worker（conda 环境隔离） |

**分割模型：** 固定使用 `yolov8n-seg.pt`（YOLOv8 nano 分割模型），prompt 必须能匹配 YOLOv8 COCO 类别名（如 `cup`、`bowl`、`bottle`）。

### 抓取规划层 (Grasping)

位于 `source/src/grasping/`，负责候选生成、位姿规划与流程协调。

| 文件 | 功能 |
|------|------|
| `coordinator.py` | `GraspPipelineCoordinator` 主协调器：串联感知→规划→执行全流程，管理相机/分割器/GraspNet 懒加载，实现深度融合、居中循环、预抓取/抓取/闭爪/撤退/交接/回家等执行序列 |
| `planning.py` | `PureGraspPlanner` 纯抓取规划器：候选筛选（分数/中心偏移/角度/工作区/位姿地板）、位姿构建（target/pregrasp/grasp/retreat）、旋转变体搜索、工具接触补偿、在线偏置 |
| `models.py` | 数据模型定义：`GraspCandidate`、`GraspPlan`、`GraspExecutionConfig`、`PerceptionResult` 等 |

**候选筛选策略（2026-06-07 更新）：** 采用"宽松候选、严格执行前验证"策略：
- 默认 approach angle 放宽至 180°，grasp score 下限 0.01，中心偏移上限 0.35m，旋转增量 180°
- YOLO 0 实例时自动使用全场景 grasp 作为 pseudo-instance 兜底
- 真机验证时为同一候选尝试多个 wrist-roll / 180° 姿态变体
- 支持 robot-friendly fallback RPY（如 `[180, 60, -90]`、`[0, 120, 0]` 等）

### 机器人适配层 (Robot)

位于 `source/src/robot/`，是唯一允许接触 ROS 话题/服务/消息的层。

| 文件 | 功能 |
|------|------|
| `client.py` | `RobotArmClient` 抽象接口 + `FakeRobotArmClient` / `Ros2PiperClient` 双后端实现：生命周期（connect/disconnect/enable/disable/emergency_stop）、状态读取、位姿运动、夹爪控制 |
| `types.py` | 机器人数据类型：`EndPoseMMDeg`、`GripperStatus` 等 |
| `moveit_ik.py` | `MoveItIkExecutor`：基于 MoveIt `/compute_ik` service 的 IK 求解 + `/joint_ctrl_single` 关节下发 |
| `executor_models.py` | 执行器数据模型 |
| `motion_tolerances.py` | 运动容差与到位判定 |
| `plan_validation.py` | 规划校验：`select_first_reachable_candidate` 对候选做 IK 可达性验证 |

**双后端模式：**
- `FakeRobotArmClient`：返回模拟数据，用于无硬件调试
- `Ros2PiperClient`：通过 ROS2 与真实 `piper_ros` 驱动通信

**已确认的 ROS2 接口：**

| 接口 | 类型 | 说明 |
|------|------|------|
| `/pos_cmd` | Topic | 末端位姿命令（m/rad） |
| `/joint_ctrl_single` | Topic | 单关节控制命令 |
| `/end_pose` | Topic | 末端当前位姿（持续发布） |
| `/arm_status` | Topic | 机械臂状态 |
| `/joint_states_feedback` | Topic | 关节状态反馈 |
| `/enable_srv` | Service | 使能/失能统一入口 |

### 工具层 (Utils)

位于 `source/src/utils/`。

| 文件 | 功能 |
|------|------|
| `transforms.py` | 坐标变换工具：变换矩阵构建、RPY 转换 |
| `calibration.py` | 手眼标定加载与校验 |
| `npoint_tool_offset.py` | N 点法工具接触点标定 |

### ROS2 分布式节点

位于 `source/robot_grasp_ros2/`。

| 文件 | 功能 |
|------|------|
| `pipeline_orchestrator_node.py` | 主编排节点：`/grasp_pipeline/run`（触发）、`/probe`（健康检查）、`/stop`、`/confirm`、`/reject` 控制面；`MultiThreadedExecutor(num_threads=4)` 执行 |
| `camera_server_node.py` | 相机服务节点：按需采集 RGB-D，响应 `CaptureScene.srv` |
| `vision_worker_node.py` | 视觉工作节点：调用 YOLOv8-seg + GraspNet，响应 `AnalyzeScene.srv`，发布 RViz 可视化话题 |
| `robot_executor_node.py` | 机器人执行节点：`fake` / `ros2` 双后端，MoveIt IK / 直接关节控制两种执行模式 |
| `grasp_pipeline_node.py` | 单节点兼容模式（旧接口保留） |
| `piper_pose_bridge_node.py` | 位姿桥接 |
| `piper_interactive_marker_node.py` | RViz 交互式 Marker 控制 |
| `joint_state_feedback_relay_node.py` | 关节状态中继 |
| `distributed_utils.py` | 分布式工具：手眼变换、消息转换、QoS 设置、run_id 生成 |
| `rviz_visualization.py` | RViz Marker 构建：候选抓取位姿、选中抓取位姿、路径 waypoint |
| `live_grasp_one_click.py` | 一键启动封装逻辑 |
| `clear_live_grasp_nodes.py` | 节点清理 |

**外部控制面：**

| Service | 说明 |
|---------|------|
| `/grasp_pipeline/run` | 触发一次完整抓取流程（`std_srvs/srv/Trigger`） |
| `/grasp_pipeline/probe` | 健康探测：检查四个节点是否在线、内部服务是否可达 |
| `/grasp_pipeline/stop` | 停止当前任务 |
| `/grasp_pipeline/confirm` | 确认执行（配合 `--confirm` 模式） |
| `/grasp_pipeline/reject` | 拒绝执行（配合 `--confirm` 模式） |

**关键 Topic（transient-local QoS，支持离线回看）：**

| Topic | 说明 |
|-------|------|
| `/grasp_pipeline/status` | 流水线状态 |
| `/grasp_pipeline/result_json` | 最终结果 JSON |
| `/vision_worker/result_json` | 视觉分析结果 |
| `/vision_worker/rviz/scene_pointcloud` | 场景点云 |
| `/vision_worker/rviz/instance_pointcloud` | 实例点云 |
| `/vision_worker/rviz/candidate_grasps` | 候选抓取位姿 |
| `/vision_worker/rviz/selected_grasp` | 选中抓取位姿 |
| `/vision_worker/rviz/plan_waypoints` | 路径 waypoints |

### Piper SDK 与驱动

位于 `piper_sdk/` 和 `piper_ros_ws/`。

- **piper_sdk**: AgileX 官方 Python SDK，提供 CAN 总线通信、关节数据读取、关节/末端控制、夹爪控制等底层接口
- **piper_ros_ws**: Piper ROS2 驱动包（`piper_ros`），将 SDK 封装为 ROS2 节点，提供标准化话题与服务
- 关键依赖：`python-can ≥ 3.3.4`、CAN 接口 `can0`（通过 USB-CAN 适配器）
- CAN 波特率：`1000000`（1 Mbps）

### GraspNet 基线模型

位于 `graspnet/`。

- 基于 "GraspNet-1Billion: A Large-Scale Benchmark for General Object Grasping" (CVPR 2020)
- 输入：RGB-D 图像 → 场景点云
- 输出：6-DOF 抓取姿态（平移 + 旋转 + 宽度 + 分数）
- 包含 PointNet++ 算子、KNN 算子、数据加载 API
- 提供预训练权重：`checkpoint-rs.tar`（RealSense）、`checkpoint-kn.tar`（Kinect）

---

## 配置说明

### 分布式节点参数

节点的运行参数通过 YAML 文件配置，位于 `source/config/distributed/`：

| 文件 | 关键参数 |
|------|---------|
| `pipeline_orchestrator.params.yaml` | `table_z_m`（桌面高度）、`workspace_x/y/z`（工作区范围）、`min_gripper_table_clearance_m`（最小桌面间隙）、`pose_goal_hold_s`（到位稳定窗口） |
| `camera_server.params.yaml` | 相机配置：分辨率、帧率、裁剪距离 |
| `vision_worker.params.yaml` | 视觉配置：GraspNet checkpoint、设备选择、top-k、体素大小 |
| `robot_executor.params.yaml` | 执行配置：后端选择、速度百分比、容差、超时、MoveIt IK 参数 |

### 桌面几何参数（2026-07-11 底盘安装后）

```
base_link → 地面:  339 mm
操作平面 → 地面:   500 mm
table_z_m:        0.161  （操作平面在 base_link 上方 161 mm）
```

### 比赛瓶子自动流程关键参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `grasp_pipeline.use_object_center_contact` | `true` | 启用瓶子中心深度融合 |
| `object_center_contact_max_offset_m` | `0.08` | 中心-接触点最大偏移 |
| `top_down_vertical_step_mm` | `80` | 垂直下降分段步长 |
| `top_down_max_speed_percent` | `5` | 默认速度百分比（初次复测建议值） |
| `center_horizontal_follow_target_azimuth` | `true` | 夹爪 yaw 自适应目标方位角 |

### 观察位姿（左侧比赛工作区）

```
[0.0, 35.5, 491.1, 180.0, 67.77, -89.97] mm/deg
```

> ⚠️ 机械臂失能前必须先回到 Home；观察位只用于采图，不作为失能位姿。

### 工作区 top-down 可达性（左侧工作区）

首选腕部方向：`[180, 60, -90] deg`，备选：yaw `90` / `0` / `-90` deg。

---

## Dashboard 操控面板

启动 Web 操控面板：

```bash
python source/scripts/run_grasp_dashboard.py
```

访问 `http://127.0.0.1:8765`。

### Dashboard 功能

- 🎯 **一键抓取**：目标 prompt 输入、补偿偏置设置
- 🍼 **瓶子中心水平抓取**：独立开关，自动配置 center_horizontal 策略
- ⚡ **速度控制**：0-100% 实时调节，真实传递到 Piper 驱动 MotionCtrl_2
- 📊 **实时状态监控**：流水线状态、任务结果
- 📝 **运行日志与产物浏览**：在线查看 `logs/distributed_runs/` 内容
- 🔧 **参数调试**：`execute`、`confirm`、`precenter`、`enable_pregrasp` 等开关

> ⚠️ 默认速度 5%，初次真机复测请保持低速，确认路径安全后再逐步提高。

---

## RViz 可视化

### 启动

```bash
cd Robot_arm/source
./scripts/open_distributed_rviz.sh
```

### 可视化话题

分布式模式下，以下话题由 `vision_worker_node` 发布：

| 话题 | 类型 | 说明 |
|------|------|------|
| `/vision_worker/rviz/scene_pointcloud` | `PointCloud2` | 全场景彩色点云 |
| `/vision_worker/rviz/instance_pointcloud` | `PointCloud2` | 实例分割点云 |
| `/vision_worker/rviz/candidate_grasps` | `MarkerArray` | 候选抓取位姿（蓝色） |
| `/vision_worker/rviz/selected_grasp` | `Marker` | 选中抓取位姿（绿色） |
| `/vision_worker/rviz/plan_waypoints` | `MarkerArray` | 路径 waypoints |
| `/vision_worker/rviz/candidate_markers` | `MarkerArray` | 候选抓取标记 |
| `/vision_worker/rviz/selected_grasp_markers` | `MarkerArray` | 选中抓取标记 |
| `/vision_worker/rviz/plan_markers` | `MarkerArray` | 路径标记 |
| `/vision_worker/rviz/camera_transform` | `Marker` | 相机坐标系（base_link 中） |
| `/tf` | `TFMessage` | TF 变换树 |

### 推荐 RViz 配置

1. 初始 Fixed Frame 设为 `camera_color_optical_frame`
2. 启用场景/实例点云和候选/选中抓取标记
3. Pipeline 运行后，TF 正常时切换 Fixed Frame 为 `base_link`
4. `no_candidate` 结果时，`selected_grasp` 和 `plan_*` 为空属于正常现象

---

## 抓取执行流程

### 标准流程（单次抓取）

```
1. START
   └─ preflight（预检查）

2. moving_to_observation
   └─ 机械臂移动到观察位姿

3. (可选) precenter
   └─ 视觉居中循环：检测目标 → 计算偏移 → 移动到图像中心

4. reading_robot_state
   └─ 读取当前 TCP 位姿和机械臂状态

5. capturing_scene
   └─ camera_server 采集 RGB-D（可选深度多帧融合）

6. analyzing_scene
   └─ YOLOv8-seg 实例分割 → 点云重建
   └─ GraspNet 全场景预测 + mask 过滤
   └─ 候选生成与筛选（分数/角度/工作区/位姿地板）

7. (可选) awaiting_confirmation
   └─ 等待外部 confirm / reject

8. executing
   └─ 候选 IK 验证（多 wrist-roll / 180° 变体尝试）
   └─ pregrasp（预抓取位）→ grasp（下探）→ target（接触/闭爪）
   └─ close_gripper → retreat（抬升）
   └─ handoff（交接位）→ open_gripper → home（回零）

9. END → completed / no_candidate / failed
```

### 比赛瓶子自动流程

```
观察 → 感知 → 高位调平/横移 → 80mm 级分段下降(约3次) → 闭爪 → 抬升 80mm
```

- 抓后保持夹持状态，不自动交接或回 Home
- 停止/失能前必须先显式回 Home

---

## 运行产物与回看

### 每次运行自动落盘

```
logs/distributed_runs/<run_id>/
├── request.json                 # 请求参数（prompt、时间戳等）
├── cycles.json                  # 多周期运行记录
├── final_result.json            # 最终状态、选中候选、执行摘要
├── candidate_validation.json    # Top-K 候选的 robot validation 淘汰链路
└── execution_trace.json         # 执行下发后的逐步回读轨迹
```

### 快速查看

```bash
# 查看最新快照（topic 层面的状态摘要）
./scripts/show_last_distributed_snapshot.sh

# 查看最新结构化产物
./scripts/show_last_run_artifact.sh
```

### 产物字段解读

- **`status`**: `ok` / `completed`（有有效候选并可选执行）、`no_candidate`（正常完成但无候选）、`failed`（异常失败）
- **`candidate_validation.json`** 中每个候选包含：`candidate_index`、`candidate_score`、`translation_camera_m`、`target_base_m`、`target_rpy_deg`、`robot_validation_stage`、`ik_error_type`、`ik_error_message`、`selection_result`
- **`execution_trace.json`** 中记录每次位姿下发后的真实回读位姿和误差

---

## 调试与故障排查

### 常用调试命令

```bash
# 查看 ROS2 节点列表
ros2 node list

# 查看话题列表
ros2 topic list

# 健康检查
ros2 service call /grasp_pipeline/probe std_srvs/srv/Trigger "{}"

# 设置 prompt 参数并触发任务
ros2 param set /grasp_pipeline prompt cup
ros2 service call /grasp_pipeline/run std_srvs/srv/Trigger "{}"
```

### 常见问题

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| `_rclpy_pybind11` import 失败 | conda Python 与系统 ROS CLI 混用 | 始终使用 `ros2_system.sh` 执行 ROS CLI 命令 |
| `/grasp_pipeline/probe` 超时 | 某个节点未正常启动 | 检查 `ros_ws/log/distributed/<timestamp>/` 下各节点日志 |
| `no valid grasp candidate found` | prompt 不在视野、YOLO 未检测到实例、候选被过滤 | 正常完成；检查 prompt 匹配 COCO 类别名、桌面几何参数 |
| `capture failed` | RealSense 连接异常或被占用 | 检查 USB 连接，确认无其他进程占用相机 |
| 真机执行无反应 | `piper_ros` 驱动未运行或 `can0` 未配置 | 检查 `can0`：`ip link show can0`；检查 `/end_pose` 是否更新 |
| `MoveIt IK request timed out` | IK 服务未就绪或 warm-up 不足 | 启动 `run_piper_moveit_ik.sh` 后等待约 10s |
| `NO_IK_SOLUTION(-31)` | 候选位姿在机械臂不可达区域 | 属于正常筛选行为；检查候选位置是否过低（< 0.05m Z） |
| `rviz2: command not found` | RViz 未安装 | 非流水线问题；`sudo apt install ros-humble-rviz2` |
| `no_candidate` 但系统正常 | 候选被 workspace / table_z / pose_floor 过滤 | 检查并调整 `table_z_m`、`workspace_z`、`min_gripper_table_clearance_m` |

### 排查"Top-1 候选被拒"的标准路径

1. 看 `final_result.json` → 确认最终执行了哪个候选
2. 看 `candidate_validation.json` → 查 `candidate_index=0` 的 `robot_validation_stage`、`ik_error_type`、`waypoint_results`
3. 看 `execution_trace.json` → 确认真实执行后的逐步回读

### 日志位置

- 分布式会话日志：`ros_ws/log/distributed/<timestamp>/`
- 一键启动日志：`logs/one_click/<timestamp>/`
- 运行产物：`logs/distributed_runs/<run_id>/`

---

## 开发指南

### 必读文档顺序

开始任何开发任务前，按以下顺序建立上下文：

1. `source/AGENTS.md`
2. `source/README.md`
3. `source/docs/CURRENT_STATUS.md`
4. `source/docs/DISTRIBUTED_RUNBOOK.md`
5. `source/docs/DISTRIBUTED_ARCHITECTURE.md`
6. `source/docs/MIGRATION_CONTRACT.md`
7. `source/docs/ENGINEERING_SPEC.md`
8. `source/docs/MIGRATION_TODO.md`
9. `source/docs/ROBOT_COUPLING_MAP.md`

### 事实来源优先级

当文档、旧实现和当前代码不一致时：

1. **当前可执行代码**：`source/src/`、`source/robot_grasp_ros2/`
2. **当前状态与运行文档**：`docs/CURRENT_STATUS.md`、`docs/DISTRIBUTED_RUNBOOK.md`
3. **迁移契约与工程规范**：`docs/MIGRATION_CONTRACT.md`、`docs/ENGINEERING_SPEC.md`
4. **迁移 backlog**：`docs/MIGRATION_TODO.md`
5. **旧版行为基线**：`old/robot_grasp/src/`
6. **外部参考**：`piper_ros_humble/`、`Agilex-College/`、`robotic_arm_kinematics/`

### 开发约定

- ✅ 业务逻辑修改在 `source/` 下进行
- ✅ ROS workspace 的 `src/` 作为软链接/编译入口，不直接编辑
- ✅ 规划层继续使用 `mm/deg` 语义，ROS2 适配层负责 `mm/deg ↔ m/rad` 转换
- ✅ 新工程业务逻辑禁止直接依赖 `piper_sdk`
- ✅ 所有硬件访问必须经过 `src/robot/`
- ✅ 新增 CLI、topic、service、字段或单位语义时，必须同步更新 `docs/`
- ❌ 不要直接编辑 `build/`、`install/`、`log/` 等生成目录

### 开发流程建议

1. **优先 fake + probe**：先在不涉及真机的 `fake` 后端上验证数据流
2. **单步验证**：使用 `/grasp_pipeline/probe` 做健康检查
3. **分布式优先**：优先使用分布式模式，单节点仅用于对照
4. **产物驱动排查**：出现问题先看 `candidate_validation.json` 和 `execution_trace.json`
5. **危险操作确认**：`execute=true` 前必须先在 `execute=false` 下确认候选可达

---

## 文档索引

| 文档 | 说明 |
|------|------|
| [CURRENT_STATUS.md](source/docs/CURRENT_STATUS.md) | 当前完成工作、已验证基线、未完成项、已知坑点（**接手必读**） |
| [DISTRIBUTED_RUNBOOK.md](source/docs/DISTRIBUTED_RUNBOOK.md) | 分布式运行手册：启动步骤、参数说明、结果查看、排障 |
| [DISTRIBUTED_ARCHITECTURE.md](source/docs/DISTRIBUTED_ARCHITECTURE.md) | 分布式架构设计：节点拓扑、接口边界、状态机语义、部署建议 |
| [ENGINEERING_SPEC.md](source/docs/ENGINEERING_SPEC.md) | 工程规范：分层规则、单位约定、Robot API、Piper ROS2 接口 |
| [MIGRATION_CONTRACT.md](source/docs/MIGRATION_CONTRACT.md) | 迁移契约：10 条不可破坏约束、实施顺序、完成判定 |
| [MIGRATION_TODO.md](source/docs/MIGRATION_TODO.md) | 迁移待办清单：旧代码到新架构的方法映射 |
| [CODE_STATUS_MAP.md](source/docs/CODE_STATUS_MAP.md) | 代码状态映射 |
| [ROBOT_COUPLING_MAP.md](source/docs/ROBOT_COUPLING_MAP.md) | 旧代码与机器人控制实现的耦合点全面标记 |
| [PIPER_LOCAL_SIM.md](source/docs/PIPER_LOCAL_SIM.md) | 本地非硬件 Piper 模型显示调试说明 |

---

## 已知限制与下一步

### 已验证完成

- [x] `fake` 后端分布式全链路数据流
- [x] 四节点架构：orchestrator / camera_server / vision_worker / robot_executor
- [x] YOLOv8-seg 实例分割 + GraspNet 候选生成
- [x] RViz 场景/实例点云、候选/选中 grasp、plan waypoints 可视化
- [x] 运行产物结构化落盘与回看
- [x] Piper 真机状态接入（`/arm_status`、`/end_pose`）
- [x] `robot_executor -> piper_ros` 命令下发链路
- [x] MoveIt IK 求解链路（`/compute_ik` → `/joint_ctrl_single`）
- [x] Web Dashboard 操控面板
- [x] 真机最小抬升/位移验证
- [x] 左侧工作区 top-down 可达性验证（首选 `[180, 60, -90] deg`）
- [x] 比赛瓶子自动流程（center_horizontal 策略）
- [x] 候选筛选变体搜索（多 wrist-roll / 180° 姿态）
- [x] 分段运动夹爪保持修正
- [x] `safe_top_down` 工具接触点补偿修正
- [x] 本地 Piper 模型链（joint_state_publisher_gui + robot_state_publisher + rviz2）

### 尚未完成

- [ ] `ros2` 真机运动执行长期稳定性验证
- [ ] 单次抓取成功率达标验收
- [ ] Action 入口替代当前 Trigger + prompt 控制面
- [ ] 执行失败 artifact 增强（失败 step、目标位姿、回读位姿、误差完整落盘）
- [ ] Top-K 候选完整淘汰链路（`filtered_by_score/angle/workspace/pose_floor` 统一落盘）
- [ ] validation-only 独立调试路径
- [ ] Gazebo / MuJoCo 物理仿真
- [ ] 底盘安装后 workspace_x/y/z 重新测量

### 持续关注的真机风险

- D435 偶发首次打开返回 `Device or resource busy`（已加重试）
- `run_piper_driver.sh` 通过 `PYTHONPATH` 注入依赖，非系统级安装
- `moveit_ik` 首条命令前需 ~10s warm-up
- 毫米级补偿动作偶发 `MoveIt IK request timed out`
- 候选落入 `z < 0.05 m` 会在 planner 阶段被前置过滤

---

## 贡献

欢迎提交 Bug 修复、功能改进、算法优化和文档完善。提交 PR 时请确保：

1. 代码能够通过 `colcon build` 编译
2. ROS2 节点能够正常运行
3. 不提交大型模型文件（`.tar`、`.tar.gz`、`.zip` 已在 `.gitignore` 中排除）
4. 新增接口或参数变更同步更新 `docs/` 目录下的相关文档

---

## 许可证

本项目用于 **学习、研究以及机器人竞赛** 目的。

- Piper SDK：详见 `piper_sdk/LICENSE`
- GraspNet：仅限非商业用途，详见 `graspnet/LICENSE`
- `robot_grasp_ros2` 包：Proprietary

---

## 作者

**NovaZone1** — Robot Arm Grasp Project

---

*最后更新：2026-07-24*
