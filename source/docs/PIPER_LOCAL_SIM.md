# Piper Local Sim

更新时间：2026-04-28

本文档记录这台机器上当前已经验证过的 `piper_ros` 本地非硬件调试路径。

定位说明：

- 这是“本机临时可用的 Piper 本地模型/仿真说明”
- 重点是没有 `can0`、不能起真实驱动时，还能做什么
- 这里记录的是本机事实，不等价于项目默认正式运行方式

## 1. 当前结论

截至 2026-04-28，这台机器上已经确认：

- 真实 `piper_single_ctrl` 仍不能启动
  - 根因不是 `piper_ros` 包损坏
  - 根因是当前系统看不到 `can0`
- `piper_description` 的本地交互模型链已经可以拉起
  - `joint_state_publisher_gui`
  - `robot_state_publisher`
  - `rviz2`
- `piper_gazebo` 源码可以编译进临时 overlay，但当前机器没有 Gazebo 运行时
- `piper_mujoco` 源码可以编译进临时 overlay，但当前机器缺 `mujoco_py`

一句话总结：

当前本机已经可以做“Piper 模型显示 + 关节滑块交互 + RViz 可视化”，但还不能做 Gazebo 物理仿真、MuJoCo 仿真，也不能做真实机械臂 ROS 驱动联调。

## 2. 已验证通过的链路

### 2.1 真实驱动阻塞点

已验证：

- `./scripts/run_piper_driver.sh` 能执行到 `piper_single_ctrl` 初始化
- 进程随后退出，报错为：
  - `CAN socket can0 does not exist`
- 官方 `find_all_can_port.sh` 没有找到任何官方 USB-CAN 模块

结论：

- 这台机器当前不能把真实 `piper_ros` 驱动跑起来
- 在 `can0` 出现之前，不要把真实机械臂链路当作当前可用项

### 2.2 本地交互模型链

已验证命令：

```bash
export AMENT_PREFIX_PATH=/tmp/ros_jazzy_extra/opt/ros/jazzy:${AMENT_PREFIX_PATH:-}
export CMAKE_PREFIX_PATH=/tmp/ros_jazzy_extra/opt/ros/jazzy:${CMAKE_PREFIX_PATH:-}
export PATH=/tmp/ros_jazzy_extra/opt/ros/jazzy/bin:$PATH
export PYTHONPATH=/tmp/ros_jazzy_extra/opt/ros/jazzy/lib/python3.12/site-packages:${PYTHONPATH:-}
source /opt/ros/jazzy/setup.bash
source /home/justahorse/Document/robot_ros/piper_ros_humble/install_jazzy_sys/setup.bash
source /tmp/piper_sim_install/setup.bash
ros2 launch piper_description display_urdf.launch.py
```

本次实际看到：

- `joint_state_publisher_gui` 正常启动
- `robot_state_publisher` 正常启动
- `rviz2` 正常启动
- `joint_state_publisher_gui` 已读取到 `robot_description`

说明：

- 这是当前本机最可靠的 Piper 本地非硬件调试入口
- 可以用来做关节范围观察、模型显示、RViz 截图、基础姿态检查

### 2.3 更轻量的跟随显示链

已验证命令：

```bash
export AMENT_PREFIX_PATH=/tmp/ros_jazzy_extra/opt/ros/jazzy:${AMENT_PREFIX_PATH:-}
export CMAKE_PREFIX_PATH=/tmp/ros_jazzy_extra/opt/ros/jazzy:${CMAKE_PREFIX_PATH:-}
export PATH=/tmp/ros_jazzy_extra/opt/ros/jazzy/bin:$PATH
export PYTHONPATH=/tmp/ros_jazzy_extra/opt/ros/jazzy/lib/python3.12/site-packages:${PYTHONPATH:-}
source /opt/ros/jazzy/setup.bash
source /home/justahorse/Document/robot_ros/piper_ros_humble/install_jazzy_sys/setup.bash
source /tmp/piper_sim_install/setup.bash
ros2 launch piper_description display_urdf_follow.launch.py
```

说明：

- 这条链只需要 `robot_state_publisher + rviz2`
- 不依赖 `joint_state_publisher_gui`
- 适合做更轻量的模型显示验证

## 3. 本机临时 overlay 的来源

这次不是把依赖装进系统，而是用两个临时前缀拼出来的。

### 3.1 `/tmp/piper_sim_install`

用途：

- 保存本地临时构建的 `piper_description`
- 保存本地临时构建的 `piper_gazebo`
- 保存本地临时构建的 `piper_mujoco`

构建命令：

```bash
source /opt/ros/jazzy/setup.bash
source /home/justahorse/Document/robot_ros/piper_ros_humble/install_jazzy_sys/setup.bash
colcon build \
  --base-paths /home/justahorse/Document/robot_ros/piper_ros_humble/src \
  --packages-select piper_description piper_gazebo piper_mujoco \
  --build-base /tmp/piper_sim_build \
  --install-base /tmp/piper_sim_install \
  --symlink-install \
  --cmake-args -DPython3_EXECUTABLE=/usr/bin/python3
```

### 3.2 `/tmp/ros_jazzy_extra`

用途：

- 本地补 `xacro`
- 本地补 `joint_state_publisher`
- 本地补 `joint_state_publisher_gui`

准备命令：

```bash
mkdir -p /tmp/ros_deb_pkgs /tmp/ros_jazzy_extra
cd /tmp/ros_deb_pkgs
apt download ros-jazzy-xacro ros-jazzy-joint-state-publisher ros-jazzy-joint-state-publisher-gui
for f in *.deb; do
  dpkg-deb -x "$f" /tmp/ros_jazzy_extra
done
```

说明：

- 这是无 `sudo apt install` 权限时的本地补包方式
- 属于临时方案，不是长期系统配置

## 4. 当前不能做的事

### 4.1 不能起真实 `piper_ros` 驱动

当前阻塞：

- `can0` 不存在
- 官方 CAN 搜索脚本未发现 USB-CAN 模块

### 4.2 不能起 Gazebo 仿真

当前阻塞：

- 系统里没有 `gazebo`
- 系统里没有 `gazebo_ros` 运行时
- 当前 apt 源里也没有现成的 `ros-jazzy-gazebo-ros-pkgs` 候选

### 4.3 不能起 MuJoCo 仿真

当前阻塞：

- `piper_mujoco` 脚本直接依赖 `mujoco_py`
- 当前系统 Python 和 Conda Python 里都没有 `mujoco_py`
- 这条链还会继续依赖 MuJoCo 本体、OpenGL/OSMesa 等本机图形运行时

## 5. 使用建议

如果当前目标是“继续做本地可视化和无真机调试”，推荐顺序如下：

1. 优先用 `display_urdf.launch.py`
   - 可以直接拖动关节滑块
   - 最适合快速确认模型、关节方向和显示问题
2. 需要更轻量显示时，用 `display_urdf_follow.launch.py`
3. 不要把这条链当成真实控制链
   - 它不会给你 `/arm_status`
   - 它不会验证 `/end_pose`
   - 它不会覆盖 `robot_executor -> piper_ros` 真机接口

## 6. 当前剩余风险

- `/tmp/piper_sim_install` 和 `/tmp/ros_jazzy_extra` 都是临时目录，重启或清理 `/tmp` 后可能消失
- 当前没有把这套本地仿真入口固化成项目脚本
- 当前 README 和运行手册的主线仍然是 distributed pipeline，不是这条本地 Piper 模型链
