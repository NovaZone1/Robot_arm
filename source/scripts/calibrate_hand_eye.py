#!/usr/bin/env python3
"""自动执行眼在手上的手眼标定。

脚本流程：
  - 使能机械臂；
  - 控制机械臂移动到观察位附近预先计算的多个采样位姿；
  - 在每个采样位姿采集相机图像和 TCP 位姿；
  - 检测棋盘格或 ArUco 标定板；
  - 求解 AX=XB，并将结果保存到 config/hand_eye/verify_config.yaml。

用法：
  # 当前棋盘为横向 9 格、纵向 6 格，因此有 8×5 个内角点，方格边长为 25 mm
  python scripts/calibrate_hand_eye.py --chessboard --pattern-size 8x5 --square-size 25

  # 使用边长为 50 mm 的 ArUco 标记
  python scripts/calibrate_hand_eye.py --aruco --marker-size 50.0
"""

from __future__ import annotations

import argparse, json, math, os, re, subprocess, sys, time
from pathlib import Path

import cv2
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
OUTPUT_YAML = PROJECT / "config" / "hand_eye" / "verify_config.yaml"
ROS2 = "/opt/ros/humble/bin/ros2"

CAM_MATRIX = None  # 读取相机内参后再设置


# ── ROS2 辅助函数 ───────────────────────────────────────────────────

def ros2_svc(name, srv_type, payload="{}"):
    """加载正确的 ROS2 环境后调用指定服务。"""
    cmd = (
        f"source /opt/ros/humble/setup.bash && "
        f"source {PROJECT.parent / 'piper_ros_ws/install/setup.bash'} && "
        f"source {PROJECT.parent / 'ros_ws/install/setup.bash'} && "
        f"ros2 service call {name} {srv_type} '{payload}'"
    )
    r = subprocess.run(["bash", "-c", cmd],
                       capture_output=True, text=True, timeout=90)
    if r.returncode != 0:
        raise RuntimeError(r.stderr.strip() + "\n" + r.stdout.strip())
    return r.stdout


def enable_robot():
    ros2_svc("/enable_srv", "piper_msgs/srv/Enable", "{enable_request: true}")
    print("  Robot enabled")
    time.sleep(1.0)


def move_to_pose(x_mm, y_mm, z_mm, roll_deg, pitch_deg, yaw_deg, name="calib"):
    """控制机械臂移动到指定的绝对 TCP 位姿，单位为 mm 和 deg。"""
    payload = (
        f"{{name: {name}, "
        f"pose: {{x_mm: {x_mm:.1f}, y_mm: {y_mm:.1f}, z_mm: {z_mm:.1f}, "
        f"roll_deg: {roll_deg:.1f}, pitch_deg: {pitch_deg:.1f}, yaw_deg: {yaw_deg:.1f}}}, "
        f"speed_percent: 30.0}}"
    )
    ros2_svc("/robot_executor/execute_named_pose", "robot_grasp_msgs/srv/ExecuteNamedPose", payload)


def get_tcp_pose():
    out = ros2_svc("/robot_executor/get_state", "robot_grasp_msgs/srv/GetRobotState")
    vals = {}
    for key in ["x_mm", "y_mm", "z_mm", "roll_deg", "pitch_deg", "yaw_deg"]:
        m = re.search(rf"{key}[:\s=]+([-]?\d+\.?\d*)", out)
        if m: vals[key] = float(m.group(1))
    return (vals["x_mm"], vals["y_mm"], vals["z_mm"],
            vals["roll_deg"], vals["pitch_deg"], vals["yaw_deg"])


def get_camera_intrinsics():
    # RealSense D435 的近似默认内参，仅在尚未读取真实内参时使用
    return np.array([[616.0, 0, 320.0], [0, 616.0, 240.0], [0, 0, 1.0]], dtype=np.float64)

def stop_camera_server():
    """停止 camera_server，使本脚本能够直接访问 RealSense。"""
    subprocess.run(["pkill", "-f", "camera_server_node"], capture_output=True, timeout=5)
    time.sleep(1.0)


def capture_image():
    """直接从 RealSense 采集图像；调用前必须停止 camera_server。"""
    global CAM_MATRIX

    from src.perception.realsense_rgbd import RealSenseRGBDCamera
    cam = RealSenseRGBDCamera(width=640, height=480, fps=30)
    cam.start()
    ok, color_bgr, _, _ = cam.get_frames()
    if cam.intrinsics is not None:
        CAM_MATRIX = np.array(
            [
                [float(cam.intrinsics.fx), 0.0, float(cam.intrinsics.ppx)],
                [0.0, float(cam.intrinsics.fy), float(cam.intrinsics.ppy)],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    cam.release()
    if not ok:
        raise RuntimeError("Camera capture failed")
    return color_bgr


# ── 标定板检测 ──────────────────────────────────────────────────────

def detect_chessboard(color_bgr, square_size_m, pattern_size):
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size,
                                                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK)
    if not found: return None, None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
    cols, rows = pattern_size
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size_m
    ok, rvec, tvec = cv2.solvePnP(objp, corners, CAM_MATRIX, None)
    if not ok: return None, None
    R, _ = cv2.Rodrigues(rvec)
    return R, tvec.flatten()


def detect_aruco(color_bgr, marker_size_m):
    gray = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2GRAY)
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    det = cv2.aruco.ArucoDetector(d, cv2.aruco.DetectorParameters())
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None or 0 not in ids: return None, None
    idx = list(ids.flatten()).index(0)
    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        [corners[idx]], marker_size_m, CAM_MATRIX, None)
    R, _ = cv2.Rodrigues(rvecs[0])
    return R, tvecs[0].flatten()


# ── 标定数学计算 ────────────────────────────────────────────────────

def tcp_to_matrix(xyz_mm, rpy_deg):
    x, y, z = xyz_mm
    r, p, yw = np.deg2rad(rpy_deg)
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(yw), -np.sin(yw), 0], [np.sin(yw), np.cos(yw), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    T = np.eye(4)
    T[:3, :3] = R; T[:3, 3] = [x / 1000.0, y / 1000.0, z / 1000.0]
    return T


def matrix_to_rpy_deg(R):
    sy = math.sqrt(R[0, 0]**2 + R[1, 0]**2)
    if sy > 1e-6:
        r = math.atan2(R[2, 1], R[2, 2])
        p = math.atan2(-R[2, 0], sy)
        y = math.atan2(R[1, 0], R[0, 0])
    else:
        r = math.atan2(-R[1, 2], R[1, 1])
        p = math.atan2(-R[2, 0], sy)
        y = 0.0
    return tuple(np.rad2deg([r, p, y]))


def solve(R_c2t, t_c2t, R_b2tcp, t_b2tcp):
    # OpenCV 需要“夹爪到基座”的变换。机械臂反馈位姿为 base_T_tcp，
    # 表示把 TCP（夹爪）坐标系中的坐标变换到机械臂基座坐标系。
    R_g2b = list(R_b2tcp)
    t_g2b = list(t_b2tcp)

    methods = [
        (cv2.CALIB_HAND_EYE_TSAI, "Tsai"),
        (cv2.CALIB_HAND_EYE_PARK, "Park"),
        (cv2.CALIB_HAND_EYE_HORAUD, "Horaud"),
        (cv2.CALIB_HAND_EYE_ANDREFF, "Andreff"),
        (cv2.CALIB_HAND_EYE_DANIILIDIS, "Daniilidis"),
    ]
    best = None
    for method, name in methods:
        try:
            R, t = cv2.calibrateHandEye(R_g2b, t_g2b, R_c2t, t_c2t, method=method)
            R_c2g, t_c2g = R, t.flatten()
            xyz = [round(float(v), 1) for v in (t_c2g * 1000)]
            rpy = [round(float(v), 3) for v in matrix_to_rpy_deg(R_c2g)]
            dist = np.linalg.norm(t_c2g) * 1000
            print(f"  {name:12s}: xyz={xyz}  rpy={rpy}  dist={dist:.0f}mm")
            if best is None or dist < 300:  # 优先选择物理尺寸合理的结果（小于 300 mm）
                best = (R_c2g, t_c2g, name)
        except Exception as e:
            print(f"  {name:12s}: FAILED - {e}")
    if best is None:
        raise RuntimeError("All calibration methods failed")
    print(f"\n  Selected: {best[2]}")
    return best[0], best[1]


# ── 标定采样位姿生成 ────────────────────────────────────────────────

def calibration_poses():
    """生成适用于倾斜或竖直标定板的多组采样位姿；不可用位姿可以跳过。"""
    poses = [
        ( 50,   0, 340,   0, 120,   0),   # 从上方观察
        ( 60,   0, 310,   0, 100,   0),   # 正面观察
        ( 40,   0, 360,   0, 130,   0),   # 从更高位置观察
        ( 70,   0, 300,   0,  90,   0),   # 从较远的正面观察
        ( 30,   0, 320,   0, 110,   0),   # 从较近的上方观察
        ( 50,  30, 340,   0, 110,  15),   # 左侧观察
        ( 50, -30, 340,   0, 110, -15),   # 右侧观察
        ( 60,  20, 320,   0,  95,  10),   # 左侧偏正面观察
        ( 60, -20, 320,   0,  95, -10),   # 右侧偏正面观察
        ( 40,  20, 370,   0, 120,   5),   # 左侧较高位置观察
        ( 40, -20, 370,   0, 120,  -5),   # 右侧较高位置观察
        ( 80,   0, 340,   0, 110,   0),   # 较远位置观察
    ]
    return poses


# ── 主程序 ──────────────────────────────────────────────────────────

def main():
    global CAM_MATRIX

    parser = argparse.ArgumentParser()
    g = parser.add_mutually_exclusive_group(required=True)
    g.add_argument("--chessboard", action="store_true")
    g.add_argument("--aruco", action="store_true")
    parser.add_argument("--square-size", type=float, default=23.5, help="mm")
    parser.add_argument("--pattern-size", type=str, default="8x5", help="colsxrows inner corners")
    parser.add_argument("--marker-size", type=float, default=50.0, help="mm")
    args = parser.parse_args()

    CAM_MATRIX = get_camera_intrinsics()
    print(f"Camera: fx={CAM_MATRIX[0,0]:.1f} fy={CAM_MATRIX[1,1]:.1f}")

    # 停止 camera_server，避免其占用 RealSense
    print("Stopping camera_server for direct camera access...")
    stop_camera_server()

    if args.chessboard:
        ps = tuple(int(v) for v in args.pattern_size.split("x"))
        detect_fn = lambda img: detect_chessboard(img, args.square_size / 1000.0, ps)
        desc = f"chessboard {ps} {args.square_size}mm"
    else:
        detect_fn = lambda img: detect_aruco(img, args.marker_size / 1000.0)
        desc = f"ArUco {args.marker_size}mm"

    poses = calibration_poses()
    print(f"\nAutomated calibration: {desc}")
    print(f"{len(poses)} poses planned.\n")

    # 使能机械臂；如果已经使能，则忽略重复使能异常
    print("Enabling robot...")
    try:
        enable_robot()
    except Exception:
        print("  (may already be enabled)")

    # 先移动到第一个采样位姿，检查标定板是否在视野中
    print("Moving to first pose for validation...")
    move_to_pose(*poses[0], name="calib_0")
    time.sleep(2.0)

    # 依次遍历所有采样位姿；采用半自动流程，每次采集均保留现场确认机会
    R_c2t, t_c2t, R_b, t_b = [], [], [], []
    pose_idx = 0
    while pose_idx < len(poses):
        pose = poses[pose_idx]
        print(f"\n[Pose {pose_idx+1}/{len(poses)}]")
        print(f"  Target: ({pose[0]:.0f},{pose[1]:.0f},{pose[2]:.0f})mm"
              f" ({pose[3]:.0f},{pose[4]:.0f},{pose[5]:.0f})deg")
        move_to_pose(*pose, name=f"calib_{pose_idx}")
        time.sleep(1.5)

        try:
            tcp_pose = get_tcp_pose()
            xyz, rpy = tcp_pose[:3], tcp_pose[3:]
        except Exception as e:
            print(f"  TCP error: {e}")
            pose_idx += 1
            continue
        print(f"  Actual: ({xyz[0]:.0f},{xyz[1]:.0f},{xyz[2]:.0f})mm"
              f" ({rpy[0]:.0f},{rpy[1]:.0f},{rpy[2]:.0f})deg")

        try:
            img = capture_image()
        except Exception as e:
            print(f"  Camera error: {e}")
            pose_idx += 1
            continue

        R, t = detect_fn(img)
        if R is None:
            print(f"  NOT DETECTED → /tmp/calib_{pose_idx:02d}_fail.png  (skipped)")
            cv2.imwrite(f"/tmp/calib_{pose_idx:02d}_fail.png", img)
            pose_idx += 1
            continue

        print(f"  Detected: dist={np.linalg.norm(t):.3f}m ✓")
        T = tcp_to_matrix(xyz, rpy)
        R_c2t.append(R); t_c2t.append(t)
        R_b.append(T[:3, :3]); t_b.append(T[:3, 3])
        pose_idx += 1

    if len(R_c2t) < 3:
        print(f"\nOnly {len(R_c2t)} valid samples. Aborting.")
        return 1

    print(f"\nSolving AX=XB ({len(R_c2t)} samples)...")
    R_c2g, t_c2g = solve(R_c2t, t_c2t, R_b, t_b)
    xyz = [round(float(v), 1) for v in (t_c2g * 1000)]
    rpy = [round(float(v), 3) for v in matrix_to_rpy_deg(R_c2g)]

    print(f"\n{'='*50}")
    print(f"  camera_to_tcp_xyz_mm:  {xyz}")
    print(f"  camera_to_tcp_rpy_deg: {rpy}")
    print(f"{'='*50}")

    OUTPUT_YAML.write_text(
        "calibration:\n"
        f"  camera_to_tcp_xyz_mm: {xyz}\n"
        f"  camera_to_tcp_rpy_deg: {rpy}\n"
        '  base_frame: "base_link"\n'
        '  camera_frame: "camera_color_optical_frame"\n',
        encoding="utf-8")
    print(f"\nSaved: {OUTPUT_YAML}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
