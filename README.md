# Robot Arm - Piper Grasp Project

无人系统大赛机械臂抓取项目源码。

本项目基于 **Piper 机械臂 + ROS2 + GraspNet** 构建，实现目标物体识别、抓取规划以及机械臂执行控制功能。

项目将机械臂驱动、视觉抓取算法、ROS 工作空间以及运行工具统一管理，方便开发、调试和比赛部署。

---

## ✨ 项目简介

本项目主要面向机器人自动抓取任务，包含：

* 机械臂 ROS2 控制
* 视觉感知与目标检测
* GraspNet 抓取姿态预测
* 抓取任务规划
* 自动化启动与运行管理

系统整体流程：

```
Camera
  │
  ▼
Object Perception
  │
  ▼
Grasp Pose Prediction (GraspNet)
  │
  ▼
Motion Planning
  │
  ▼
Piper Robot Arm
  │
  ▼
Object Grasping
```

---

# 📂 项目结构

```
Robot_arm/
│
├── source/
│   └── 抓取任务核心源码
│
├── ros_ws/
│   └── ROS2 工作空间
│       └── src/
│           └── 指向 source 中的 ROS package
│
├── piper_ros_ws/
│   └── Piper ROS 驱动源码及环境
│
├── piper_sdk/
│   └── Piper 底层控制 SDK
│
├── graspnet/
│   ├── GraspNet baseline
│   └── 模型 checkpoint 文件
│
├── logs/
│   └── 抓取运行日志
│
├── tmp/
│   └── ROS 临时运行文件
│
└── README.md
```

---

# 🛠️ 环境要求

推荐环境：

* Ubuntu 22.04
* ROS2 Humble
* Python 3.x
* CUDA（根据 GraspNet 推理需求配置）
* Piper Robot Arm

主要依赖：

* ROS2
* MoveIt2
* OpenCV
* PyTorch
* GraspNet
* Piper SDK

---

# 🚀 快速开始

## 1. 克隆仓库

```bash
git clone https://github.com/NovaZone1/Robot_arm.git

cd Robot_arm
```

---

# 2. 配置 ROS2 工作空间

进入 ROS 工作空间：

```bash
cd ros_ws
```

编译：

```bash
colcon build
```

加载环境：

```bash
source install/setup.bash
```

---

# 3. 配置 Piper 驱动

进入 Piper ROS 工作空间：

```bash
cd piper_ros_ws
```

编译：

```bash
colcon build
```

加载环境：

```bash
source install/setup.bash
```

---

# 4. 启动抓取系统

进入项目源码目录：

```bash
cd source
```

运行抓取控制程序：

```bash
./scripts/run_grasp_dashboard.py
```

或者使用一键启动：

```bash
./scripts/run_live_grasp_one_click.sh
```

---

# 🦾 系统模块说明

## Piper Robot Control

负责：

* 机械臂通信
* 关节控制
* 运动执行

目录：

```
piper_sdk/
piper_ros_ws/
```

---

## GraspNet Grasp Detection

负责：

* RGB-D 数据处理
* 抓取姿态预测
* 最优抓取点生成

目录：

```
graspnet/
```

---

## ROS2 Workspace

负责：

* 节点管理
* 消息通信
* 机械臂任务调度

目录：

```
ros_ws/
```

---

# 📌 开发规范

为了保持项目结构清晰：

* 业务代码统一修改：

```
source/
```

* ROS workspace 中：

```
ros_ws/src/
```

仅作为源码链接和 ROS 编译入口。

* 不建议直接修改生成文件：

```
build/
install/
log/
```

---

# 🧪 调试

查看 ROS 节点：

```bash
ros2 node list
```

查看话题：

```bash
ros2 topic list
```

查看日志：

```bash
ls logs/
```

---

# 📊 项目流程

```
启动 ROS 环境
        │
        ▼
加载 Piper 驱动
        │
        ▼
启动视觉系统
        │
        ▼
GraspNet 推理
        │
        ▼
生成抓取姿态
        │
        ▼
机械臂执行抓取
```

---

# 🤝 Contribution

欢迎提交：

* Bug 修复
* 功能优化
* 算法改进
* 文档完善

提交 Pull Request 前请确保：

* 代码能够正常编译
* ROS 节点运行正常
* 不提交大规模模型文件

---

# 📜 License

本项目仅用于学习、研究以及机器人竞赛用途。

---

# 👥 Author

NovaZone1

Robot Arm Grasp Project
