# -*- coding: utf-8 -*-
"""UR5e + Intel RealSense D435i 眼在手上手眼标定 GUI.

运行:
    pip install numpy opencv-python pillow pyrealsense2 ur-rtde
    python ur_eyehand.py
离线算法自检:
    python ur_eyehand.py --self-test

坐标约定: T_A_B 表示 B 坐标 -> A 坐标.
输入: T_base_tcp = RTDE.getActualTCPPose(), T_cam_board = solvePnP().
输出: T_tcp_cam = calibrateHandEye(), 即彩色相机光学坐标 -> UR 当前 TCP 坐标.
如果需要 T_base_cam = T_base_tcp @ T_tcp_cam.

注意: 12x9 指棋盘方格数, 对应 11x8 个内部角点; 方格边长必须实测.
摄像机牢固固定在末端, 标定板保持静止; 标定全过程禁止改变 UR 激活的 TCP 配置.
新版本在 GUI 中提供受限的单关节增量 moveJ（默认锁定，必须主动启用）。
本软件点动与停止均不是安全级控制；操作时必须有人监护，保持示教器实体急停可及。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import queue
import threading
import time
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = ImageTk = None

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

try:
    import rtde_receive
except ImportError:
    rtde_receive = None

try:
    import rtde_control
except ImportError:
    rtde_control = None

APP_NAME = "UR5e 手眼标定 + 关节点动"
BOARD_SQUARES = (12, 9)            # 实际黑白方格数量
INNER = (11, 8)                    # OpenCV 内部角点 (列, 行)
MIN_SAMPLES = 6
MAX_RMSE_PX = 1.5
MAX_MOVING_QD = 0.025              # rad/s: 采样时各关节速度绝对值上限
MAX_MOVING_TCP_MPS = 0.01         # m/s: TCP 线速度上限
RTDE_HZ = 50.0                    # 不以 500 Hz 开额外的接收链接
VIDEO_SIZE = (640, 480)
# 这些只是额外的软件限制，不表示 UR5e 的实际关节限制或完成了碰撞检测。
# 需先在示教器上确认工作空间、有效载荷、安装与碰撞风险。
SOFT_JOINT_LIMITS_DEG = [(-180., 180.)] * 6
JOG_STEPS_DEG = (0.2, 0.5, 1.0, 2.0, 5.0)
JOG_SPEEDS_DEG_S = (2.0, 5.0, 10.0)
JOG_ACCELS_DEG_S2 = (10.0, 20.0, 40.0)
CONTROL_RTDE_HZ = 50.0
SNAPSHOT_MAX_AGE_S = 1.5
STOP_DECEL_RAD_S2 = 0.7
SETTLE_DELAY_S = 0.65


def compute_jog_target(q_rad, joint_index, direction, step_deg, speed_deg_s, accel_deg_s2):
    """无 I/O 的关节点动规划，只改变一个轴；输入检查和软件角度界限校验。"""
    q = np.asarray(q_rad, dtype=float)
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("当前关节角无效")
    if not 0 <= joint_index < 6 or direction not in (-1, 1):
        raise ValueError("关节编号或方向错误")
    if (float(step_deg) not in JOG_STEPS_DEG or
            float(speed_deg_s) not in JOG_SPEEDS_DEG_S or
            float(accel_deg_s2) not in JOG_ACCELS_DEG_S2):
        raise ValueError("步长、速度或加速度不属于允许的选项")
    goal = q.copy()
    goal[joint_index] += math.radians(direction * step_deg)
    low, high = SOFT_JOINT_LIMITS_DEG[joint_index]
    actual_deg = math.degrees(q[joint_index])
    target_deg = math.degrees(goal[joint_index])
    if not low <= target_deg <= high:
        raise ValueError(f"J{joint_index+1} 目标 {target_deg:+.2f}° 超出软件边界 [{low:+.0f}°, {high:+.0f}°]。"
                         f"当前 {actual_deg:+.2f}°，请先从示教器确认关节配置与安全性。")
    return goal.tolist(), math.radians(speed_deg_s), math.radians(accel_deg_s2)



def iso(R=None, t=None):
    T = np.eye(4, dtype=np.float64)
    if R is not None:
        T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    if t is not None:
        T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def inverse(T):
    T = np.asarray(T, dtype=np.float64)
    R = T[:3, :3]
    return iso(R.T, -R.T @ T[:3, 3])


def ur_pose_to_T(pose):
    """UR x,y,z[m], rx,ry,rz[轴角旋转向量 rad] -> T_base_tcp."""
    p = np.asarray(pose, dtype=np.float64).reshape(6)
    R = cv2.Rodrigues(p[3:].reshape(3, 1))[0]
    return iso(R, p[:3])


def rotation_deg(R):
    a = (np.trace(R) - 1.0) * 0.5
    return math.degrees(math.acos(float(np.clip(a, -1., 1.))))


def object_points(square_m):
    """X 沿角点的 11 列, Y 沿角点的 8 行, Z=0; 单位米."""
    xy = np.mgrid[0:INNER[0], 0:INNER[1]].T.reshape(-1, 2)
    pts = np.zeros((len(xy), 3), np.float32)
    pts[:, :2] = xy * float(square_m)
    return pts


def detect_and_pose(color, K, D, square_m):
    """提取角点, solvePnP, 返回结果及绘制预览; 全在图像工作线程执行."""
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    # SB 对棋盘格倾斜和光照变化更稳健; 不支持时回退经典方法.
    if hasattr(cv2, "findChessboardCornersSB"):
        flag = getattr(cv2, "CALIB_CB_NORMALIZE_IMAGE", 0)
        found, corners = cv2.findChessboardCornersSB(gray, INNER, flags=flag)
    else:
        found, corners = cv2.findChessboardCorners(
            gray, INNER, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (7, 7), (-1, -1), criteria)

    vis = color.copy()
    if not found or corners is None:
        cv2.putText(vis, "BOARD NOT FOUND (11 x 8 corners)", (12, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 40, 255), 2)
        return None, vis

    corners = np.asarray(corners, np.float32).reshape(-1, 1, 2)
    obj = object_points(square_m)
    good, rvec, tvec = cv2.solvePnP(obj, corners, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
    if not good or float(tvec[2, 0]) <= 0:
        return None, vis
    projection, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
    err = np.linalg.norm(projection.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    T_cam_board = iso(cv2.Rodrigues(rvec)[0], tvec.reshape(3))
    cv2.drawChessboardCorners(vis, INNER, corners, True)
    cv2.drawFrameAxes(vis, K, D, rvec, tvec, float(square_m) * 3, 2)
    cv2.putText(vis, f"11x8 corners | reproj RMS={rmse:.2f}px", (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 180, 0) if rmse <= MAX_RMSE_PX
                else (0, 130, 255), 2)
    return dict(corners=corners.reshape(-1, 2), rvec=rvec.reshape(3),
                tvec=tvec.reshape(3), T_cam_board=T_cam_board, rmse=rmse), vis


def mean_rotation(rotations):
    """SVD 正交化的弦均值: 对帧间角度离散度的诊断, 非概率精度估计."""
    M = np.sum(rotations, axis=0)
    U, _, Vt = np.linalg.svd(M)
    S = np.eye(3)
    S[2, 2] = np.linalg.det(U @ Vt)
    return U @ S @ Vt


def board_consistency(samples, T_tcp_cam):
    """静止棋盘格假设: T_base_board(i) 理应相同. 返回跨帧离散度."""
    transforms = [np.asarray(s["T_base_tcp"], float) @ T_tcp_cam @
                  np.asarray(s["T_cam_board"], float) for s in samples]
    xyz = np.array([T[:3, 3] for T in transforms])
    mean_t = xyz.mean(axis=0)
    mean_R = mean_rotation([T[:3, :3] for T in transforms])
    errors_mm = np.linalg.norm(xyz - mean_t, axis=1) * 1000
    errors_deg = np.array([rotation_deg(mean_R.T @ T[:3, :3]) for T in transforms])
    return {
        "translation_rms_mm": float(np.sqrt(np.mean(errors_mm ** 2))),
        "rotation_rms_deg": float(np.sqrt(np.mean(errors_deg ** 2))),
        "max_translation_mm": float(np.max(errors_mm)),
        "max_rotation_deg": float(np.max(errors_deg)),
        "T_base_board_mean": iso(mean_R, mean_t).tolist(),
        "translation_errors_mm": errors_mm.tolist(),
        "rotation_errors_deg": errors_deg.tolist(),
    }


def solve_handeye(samples):
    if len(samples) < MIN_SAMPLES:
        raise ValueError(f"至少采集 {MIN_SAMPLES} 组, 推荐 15~25 组不同朝向 (当前 {len(samples)} 组).")
    tb = [np.asarray(s["T_base_tcp"], float) for s in samples]
    ct = [np.asarray(s["T_cam_board"], float) for s in samples]
    for T in tb + ct:
        if T.shape != (4, 4) or not np.isfinite(T).all():
            raise ValueError("采样中发现无效 4x4 位姿矩阵.")
    max_rot = max(rotation_deg(tb[i][:3, :3].T @ tb[j][:3, :3])
                  for i in range(len(tb)) for j in range(i + 1, len(tb)))
    if max_rot < 8.0:
        raise ValueError(f"末端最大相对旋转仅 {max_rot:.1f}°, 旋转激励不足; 请改变末端观察倾角后再采样.")
    choices = [
        ("Park", cv2.CALIB_HAND_EYE_PARK),
        ("Tsai", cv2.CALIB_HAND_EYE_TSAI),
        ("Horaud", cv2.CALIB_HAND_EYE_HORAUD),
        ("Daniilidis", cv2.CALIB_HAND_EYE_DANIILIDIS),
    ]
    candidates = []
    failures = []
    for name, method in choices:
        try:
            R, t = cv2.calibrateHandEye(
                R_gripper2base=[T[:3, :3] for T in tb],
                t_gripper2base=[T[:3, 3] for T in tb],
                R_target2cam=[T[:3, :3] for T in ct],
                t_target2cam=[T[:3, 3] for T in ct],
                method=method,
            )
            if R is None or t is None:
                raise ValueError("算法未返回结果")
            R = np.asarray(R, float).reshape(3, 3)
            t = np.asarray(t, float).reshape(3)
            if (not np.isfinite(R).all() or not np.isfinite(t).all() or
                    abs(np.linalg.det(R) - 1) > 1e-2 or
                    np.linalg.norm(R.T @ R - np.eye(3)) > 1e-2):
                raise ValueError("返回无效旋转/平移矩阵")
            X = iso(R, t)
            c = board_consistency(samples, X)
            # 数值仅用于多算法排序, 不是实际绝对标定精度.
            score = c["translation_rms_mm"] + 2.0 * c["rotation_rms_deg"]
            candidates.append(dict(method=name, T_tcp_cam=X.tolist(),
                                   T_cam_tcp=inverse(X).tolist(),
                                   consistency=c, ranking_score=float(score)))
        except Exception as e:
            failures.append(f"{name}: {e}")
    if not candidates:
        raise RuntimeError("所有求解方法都失败: " + "; ".join(failures))
    candidates.sort(key=lambda c: c["ranking_score"])
    return dict(best=candidates[0], candidates=candidates, failures=failures,
                num_samples=len(samples), max_robot_rotation_deg=max_rot,
                mean_pnp_rmse_px=float(np.mean([s["rmse_px"] for s in samples])),
                note="方法选择依据静止棋盘格的一致性, 并非独立的绝对精度验证.")


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def compatible_intrinsics(old, new):
    if old is None:
        return True
    keys = ("width", "height", "fx", "fy", "ppx", "ppy", "coeffs", "model")
    for k in keys:
        if k not in old or k not in new:
            return False
        if k == "model":
            if old[k] != new[k]: return False
        elif not np.allclose(np.asarray(old[k], float), np.asarray(new[k], float), rtol=1e-5, atol=1e-5):
            return False
    return True


def camera_parameters(profile):
    ip = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    model = ip.model
    coeffs = list(map(float, ip.coeffs))
    # OpenCV 的畸变定义与 RealSense 某些逆向/修正畸变模型不能直接互换.
    if model not in (rs.distortion.none, rs.distortion.brown_conrady):
        if max(abs(x) for x in coeffs) > 1e-10:
            raise RuntimeError(f"RGB 内参使用 {model} 畸变模型, 不可直接交给 OpenCV solvePnP."
                               "请先完成标准针孔/常规 Brown 模型的 RGB 内参标定与适配.")
        coeffs = [0.] * 5
    K = np.array([[ip.fx, 0., ip.ppx], [0., ip.fy, ip.ppy], [0., 0., 1.]], float)
    D = np.asarray(coeffs, float)
    meta = dict(width=int(ip.width), height=int(ip.height), fx=float(ip.fx),
                fy=float(ip.fy), ppx=float(ip.ppx), ppy=float(ip.ppy),
                coeffs=coeffs, model=str(model),
                frame="RealSense RGB/color optical frame (not depth optical frame)")
    return K, D, meta


class CalibratorApp:
    def __init__(self, root):
        self.root = root
        root.title(APP_NAME)
        root.geometry("1250x840")
        root.minsize(1050, 690)
        self.ip = tk.StringVar(value="192.168.3.15")
        self.serial = tk.StringVar(value="")
        self.square_mm = tk.StringVar(value="20.0")  # 仅界面占位示例, 使用前必须实测并修改
        self.goal = tk.StringVar(value="10")
        self.save_depth = tk.BooleanVar(value=True)
        self.output_root = tk.StringVar(value=str(Path.cwd() / "handeye_data"))
        self.state_text = tk.StringVar(value="未连接")
        self.board_text = tk.StringVar(value="相机未连接")
        self.q_text = tk.StringVar(value="J1~J6: --")
        self.tcp_text = tk.StringVar(value="TCP: --")
        self.counter_text = tk.StringVar(value="0 / 10 组")
        self.queue = queue.Queue()
        self.lock = threading.Lock()
        self.latest_preview = None
        self.last_preview_id = 0
        self.photoref = None
        self.capture_event = threading.Event()
        self.stop_event = threading.Event()
        self.worker = None
        self.samples = []
        self.session_dir = None
        self.session_settings = None
        self.cam_info = None
        self.last_result = None
        self.preview_settings = None
        # 相机线程只写机器人状态快照；RTDEControl 由独立线程独占使用。
        self.robot_snapshot = None
        self.motion_thread = None
        self.motion_stop_event = threading.Event()
        self.motion_busy_event = threading.Event()
        self.motion_commands = queue.Queue()
        self.motion_armed = False
        self.motion_connecting = False
        self.capture_block_until = 0.0
        self.shutdown_requested = False
        self.jog_step = tk.StringVar(value="1.0")
        self.jog_speed = tk.StringVar(value="5.0")
        self.jog_accel = tk.StringVar(value="20.0")
        self.motion_text = tk.StringVar(value="关节控制：锁定")
        self.joint_values = [tk.StringVar(value=f"J{i+1}: --°") for i in range(6)]
        self.jog_buttons = []
        self.build_ui()
        self.refresh_devices()
        self.root.after(100, self.tick)
        root.protocol("WM_DELETE_WINDOW", self.close)

    def build_ui(self):
        pane = ttk.Frame(self.root, padding=9)
        pane.pack(fill="both", expand=True)
        cfg = ttk.LabelFrame(pane, text="连接与标定配置（改变参数请先断开连接）", padding=9)
        cfg.pack(fill="x")
        ttk.Label(cfg, text="UR IP").grid(row=0, column=0, sticky="w")
        self.ip_entry = ttk.Entry(cfg, textvariable=self.ip, width=18)
        self.ip_entry.grid(row=0, column=1, padx=(4, 16), sticky="w")
        ttk.Label(cfg, text="RealSense 序列号").grid(row=0, column=2, sticky="w")
        self.serial_combo = ttk.Combobox(cfg, textvariable=self.serial, width=20)
        self.serial_combo.grid(row=0, column=3, padx=4, sticky="w")
        ttk.Button(cfg, text="扫描相机", command=self.refresh_devices).grid(row=0, column=4, padx=6)
        ttk.Label(cfg, text="方格边长 mm（实测）").grid(row=0, column=5, sticky="w", padx=(12, 0))
        self.square_entry = ttk.Entry(cfg, textvariable=self.square_mm, width=9)
        self.square_entry.grid(row=0, column=6, padx=4)
        ttk.Label(cfg, text="采样目标").grid(row=0, column=7, sticky="w", padx=(12, 0))
        self.goal_spin = ttk.Spinbox(cfg, from_=6, to=100, increment=1, textvariable=self.goal, width=6)
        self.goal_spin.grid(row=0, column=8)
        ttk.Label(cfg, text="保存根目录").grid(row=1, column=0, sticky="w", pady=(12, 2))
        self.path_entry = ttk.Entry(cfg, textvariable=self.output_root, width=78)
        self.path_entry.grid(row=1, column=1, columnspan=5, sticky="ew", pady=(12, 2))
        ttk.Button(cfg, text="浏览", command=self.browse_dir).grid(row=1, column=6, padx=4, pady=(12, 2))
        self.btn_connect = ttk.Button(cfg, text="连接设备", command=self.connect)
        self.btn_connect.grid(row=1, column=7, padx=4, pady=(12, 2))
        self.btn_disconnect = ttk.Button(cfg, text="断开", command=self.disconnect)
        self.btn_disconnect.grid(row=1, column=8, padx=4, pady=(12, 2))
        self.depth_check = ttk.Checkbutton(
            cfg, text="同时保存对齐到彩色图的深度帧（16位 PNG；D435i 建议 USB 3.0）",
            variable=self.save_depth)
        self.depth_check.grid(row=2, column=1, columnspan=6, sticky="w", pady=(5, 0))
        cfg.columnconfigure(3, weight=1)

        status = ttk.Frame(pane)
        status.pack(fill="x", pady=7)
        ttk.Label(status, textvariable=self.state_text, foreground="#155ba1").pack(side="left", padx=(0, 30))
        ttk.Label(status, textvariable=self.board_text).pack(side="left")
        ttk.Label(status, textvariable=self.counter_text, font=("Arial", 12, "bold")).pack(side="right")

        mid = ttk.PanedWindow(pane, orient="horizontal")
        mid.pack(fill="both", expand=True)
        cam_box = ttk.LabelFrame(mid, text="RGB 实时图像：绿色轴线代表标定板坐标；录制前请等待机器人静止")
        mid.add(cam_box, weight=3)
        self.image_lbl = ttk.Label(cam_box, text="等待连接 RealSense RGB 视频", anchor="center")
        self.image_lbl.pack(expand=True, fill="both", padx=5, pady=5)
        status_cam = ttk.Frame(cam_box)
        status_cam.pack(fill="x", padx=9, pady=7)
        ttk.Label(status_cam, textvariable=self.q_text, wraplength=690).pack(anchor="w")
        ttk.Label(status_cam, textvariable=self.tcp_text, wraplength=690).pack(anchor="w")
        self.build_jog_panel(cam_box)

        right = ttk.Frame(mid, padding=(8, 2))
        mid.add(right, weight=2)
        ttk.Label(right, text="采集的数据", font=("Arial", 11, "bold")).pack(anchor="w", pady=(4, 6))
        tree_frame = ttk.Frame(right)
        tree_frame.pack(fill="both", expand=True)
        self.tree = ttk.Treeview(tree_frame, columns=("n", "time", "rmse", "position"),
                                 show="headings", height=7, selectmode="browse")
        for k, heading, width in (("n", "编号", 48), ("time", "时间", 100),
                                  ("rmse", "重投影 px", 90), ("position", "TCP XYZ (mm)", 170)):
            self.tree.heading(k, text=heading)
            self.tree.column(k, width=width, stretch=(k == "position"))
        ys = ttk.Scrollbar(tree_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=ys.set)
        self.tree.pack(side="left", fill="both", expand=True)
        ys.pack(side="right", fill="y")
        buttons = ttk.Frame(right)
        buttons.pack(fill="x", pady=8)
        self.btn_capture = ttk.Button(buttons, text="记录当前姿态 + 棋盘格", command=self.take_sample)
        self.btn_capture.pack(fill="x", pady=2)
        ttk.Button(buttons, text="删除选中的一组", command=self.remove_sample).pack(fill="x", pady=2)
        ttk.Button(buttons, text="求解手眼矩阵", command=self.calculate).pack(fill="x", pady=2)
        row = ttk.Frame(buttons)
        row.pack(fill="x")
        ttk.Button(row, text="载入历史数据", command=self.load_session).pack(side="left", expand=True, fill="x", padx=(0, 3))
        ttk.Button(row, text="新建一组数据", command=self.reset_session).pack(side="left", expand=True, fill="x", padx=(3, 0))

        res = ttk.LabelFrame(right, text="求解结果 / 质量检查", padding=6)
        res.pack(fill="both", expand=True)
        self.result_box = tk.Text(res, height=9, wrap="word", font=("Consolas", 9))
        self.result_box.pack(fill="both", expand=True)
        self.result_box.insert("1.0", "待采样。\n\n最终结果为 T_tcp_cam（彩色相机光学坐标 → UR 当前 TCP）。\n"
                                     "标定板固定，移动末端从不同方向观察，至少 10 组，建议 15~25 组。")
        self.result_box.configure(state="disabled")

        log_frame = ttk.LabelFrame(pane, text="日志 / 注意事项", padding=5)
        log_frame.pack(fill="x", pady=(8, 0))
        self.log_box = tk.Text(log_frame, height=3, wrap="word", font=("Consolas", 9))
        self.log_box.pack(fill="x")
        self.log("新增关节点动默认锁定：须主动启用；先低速检查方向与周围净空；切勿与遥操作程序争夺控制权。")
        self.log("12×9 方格 = 11×8 内角点; 方格边长 20 mm 只是初始示例, 使用前必须测量后填写.")
        self.log("相机安装不可松动, 棋盘格必须固定, UR 当前 TCP 偏置设置在整个采集期间保持不变.")

    def build_jog_panel(self, parent):
        panel = ttk.LabelFrame(parent, text="UR5e 关节增量点动（每次仅移动一个关节）", padding=6)
        panel.pack(fill="x", padx=5, pady=(0, 5))
        top = ttk.Frame(panel)
        top.pack(fill="x", pady=(0, 3))
        self.btn_arm = ttk.Button(top, text="启用运动控制", command=self.enable_motion)
        self.btn_arm.pack(side="left", padx=(0, 6))
        self.btn_stop = tk.Button(top, text="停止运动 / 锁定", bg="#b3261e", fg="white",
                                  activebackground="#8f2019", activeforeground="white",
                                  font=("Arial", 10, "bold"), command=self.stop_motion)
        self.btn_stop.pack(side="left", padx=(0, 10))
        ttk.Label(top, textvariable=self.motion_text).pack(side="left")

        group = ttk.Frame(panel)
        group.pack(fill="x", pady=(2, 4))
        for i in range(6):
            cell = ttk.Frame(group)
            cell.grid(row=0, column=i, padx=2, sticky="ew")
            group.columnconfigure(i, weight=1)
            ttk.Label(cell, textvariable=self.joint_values[i], anchor="center").pack(fill="x")
            controls = ttk.Frame(cell)
            controls.pack()
            neg = ttk.Button(controls, text="−", width=3,
                             command=lambda j=i: self.request_jog(j, -1))
            pos = ttk.Button(controls, text="+", width=3,
                             command=lambda j=i: self.request_jog(j, +1))
            neg.pack(side="left", padx=1)
            pos.pack(side="left", padx=1)
            self.jog_buttons.extend((neg, pos))

        opts = ttk.Frame(panel)
        opts.pack(fill="x")
        for title, var, choices, col in (
            ("步长 (°/次)", self.jog_step, JOG_STEPS_DEG, 0),
            ("速度 (°/s)", self.jog_speed, JOG_SPEEDS_DEG_S, 2),
            ("加速度 (°/s²)", self.jog_accel, JOG_ACCELS_DEG_S2, 4),
        ):
            ttk.Label(opts, text=title).grid(row=0, column=col, padx=(6, 2), sticky="w")
            choice = ttk.Combobox(opts, textvariable=var, width=5, state="readonly",
                                  values=[str(x) for x in choices])
            choice.grid(row=0, column=col + 1, padx=(0, 7))
        ttk.Label(panel, text="软件停止 ≠ 安全急停。请保持实体急停可及；−/+ 是关节角负/正方向，非末端 XYZ。",
                  foreground="#9b281f", wraplength=700).pack(anchor="w", pady=(3, 0))
        self.update_motion_buttons()

    def update_motion_buttons(self):
        alive = self.motion_thread is not None and self.motion_thread.is_alive()
        operating = self.motion_busy_event.is_set()
        ready = (self.motion_armed and not operating and not self.motion_stop_event.is_set() and
                 self.worker is not None and self.worker.is_alive() and self.cam_info is not None)
        for btn in self.jog_buttons:
            btn.configure(state="normal" if ready else "disabled")
        can_arm = (self.worker is not None and self.worker.is_alive() and self.cam_info is not None)
        self.btn_arm.configure(state="normal" if can_arm and not alive and not self.motion_connecting else "disabled")
        if hasattr(self, "btn_capture"):
            self.btn_capture.configure(state="disabled" if operating else "normal")
        if operating:
            self.motion_text.set("关节控制：执行中（不可采样）")
        elif self.motion_connecting:
            self.motion_text.set("关节控制：正在连接...")
        elif ready:
            self.motion_text.set("关节控制：已启用，等待点动")
        else:
            self.motion_text.set("关节控制：锁定")

    def snapshot_for_motion(self, *, require_stationary=True):
        with self.lock:
            snap = self.robot_snapshot
            data = None if snap is None else dict(snap)
        if data is None or time.monotonic() - data["time"] > SNAPSHOT_MAX_AGE_S:
            raise RuntimeError("机器人状态读取延迟/中断，请确认实时状态连接")
        if data["safety_mode"] not in (1, 2):
            raise RuntimeError(f"机器人安全模式 {data['safety_mode']} 不是正常/减速模式，不允许动作")
        if require_stationary and not data["stationary"]:
            raise RuntimeError("机器人仍在移动，必须停稳后才能发送下一条点动")
        return data

    def enable_motion(self):
        if self.shutdown_requested or self.motion_connecting or (self.motion_thread and self.motion_thread.is_alive()):
            return
        if self.worker is None or not self.worker.is_alive() or self.cam_info is None:
            messagebox.showwarning("尚未连接", "请先连接 UR5e 与相机并看到实时关节角。")
            return
        if rtde_control is None:
            messagebox.showerror("缺少模块", "请安装 ur-rtde（应包含 rtde_control 模块）。")
            return
        try:
            self.snapshot_for_motion()
        except Exception as e:
            messagebox.showerror("不可启用", str(e))
            return
        ok = messagebox.askyesno(
            "确认启用真实机器人运动",
            "此操作将连接 RTDEControl 并可能接管当前机器人运行的控制脚本。\n\n"
            "请确认：\n"
            "1. 已退出其他遥操作/回放/自动运行程序，机器人处于合适的远程控制状态；\n"
            "2. 已检查标定板、相机与所有关节的运动净空，人员远离工作区域；\n"
            "3. 有人现场监护并能立即按下实体急停；\n"
            "4. 首次用 0.2°、2°/s 小步试运行。\n\n"
            "这不是安全级碰撞保护。是否启用？")
        if not ok:
            return
        self.motion_stop_event.clear()
        self.motion_commands = queue.Queue()
        self.motion_connecting = True
        self.motion_thread = threading.Thread(target=self.motion_worker,
                                              args=(self.preview_settings["robot_ip"],), daemon=True)
        self.motion_thread.start()
        self.log("申请启用 RTDEControl 关节点动（50Hz）。")
        self.update_motion_buttons()

    def request_jog(self, joint, direction):
        if not self.motion_armed or self.motion_busy_event.is_set() or self.motion_stop_event.is_set():
            self.log("关节控制未启用或机器人尚在运动，忽略点动命令。")
            return
        try:
            snap = self.snapshot_for_motion()
            step, speed, accel = (float(self.jog_step.get()), float(self.jog_speed.get()),
                                  float(self.jog_accel.get()))
            goal, _, _ = compute_jog_target(snap["q"], joint, direction, step, speed, accel)
            if not all(np.isfinite(goal)):
                raise ValueError("运动目标无效")
        except Exception as e:
            self.log(f"拒绝点动: {e}")
            return
        # 在 GUI 回调里先加忙标志，避免快速双击叠加多条命令。
        self.motion_busy_event.set()
        self.motion_commands.put(("jog", joint, direction, step, speed, accel))
        self.log(f"申请 J{joint+1} {'+' if direction > 0 else '-'}{step:g}°，速度 {speed:g}°/s。")
        self.update_motion_buttons()

    def stop_motion(self):
        # 无论控制线程在连接、空闲还是执行运动，都锁定关节运动并请求退出。
        self.motion_armed = False
        self.motion_stop_event.set()
        self.capture_event.clear()
        if self.motion_thread is not None and self.motion_thread.is_alive():
            self.log("已请求 stopJ + 停止控制脚本；等待停止完成。紧急危险请使用实体急停！")
        self.update_motion_buttons()

    def motion_worker(self, robot_ip):
        ctrl = None
        move_active = False
        try:
            ctrl = rtde_control.RTDEControlInterface(robot_ip, CONTROL_RTDE_HZ)
            if not ctrl.isConnected():
                raise RuntimeError("RTDEControl 未连接；检查机器人远程控制及网络")
            if self.motion_stop_event.is_set():
                return
            self.queue.put(("motion_ready", None))
            while not self.motion_stop_event.is_set():
                try:
                    cmd = self.motion_commands.get(timeout=0.05)
                except queue.Empty:
                    continue
                if cmd[0] != "jog" or self.motion_stop_event.is_set():
                    continue
                _, joint, direction, step, speed, accel = cmd
                try:
                    # 发送前重新读取实测关节角，绝不把上一次目标角作为新起点。
                    snap = self.snapshot_for_motion(require_stationary=True)
                    goal, speed_r, accel_r = compute_jog_target(
                        snap["q"], joint, direction, step, speed, accel)
                    if self.motion_stop_event.is_set():
                        break
                    self.queue.put(("motion_log", f"发送 moveJ: J{joint+1} 目标={math.degrees(goal[joint]):+.2f}°"))
                    ok = ctrl.moveJ(goal, speed_r, accel_r, True)
                    if ok is False:
                        raise RuntimeError("moveJ 返回 False，机器人拒绝运动")
                    move_active = True
                    begin = time.monotonic()
                    duration_limit = max(5.0, 2.0 * math.radians(step) / speed_r +
                                         3.0 * speed_r / accel_r + 3.0)
                    settled_at = None
                    while True:
                        if self.motion_stop_event.is_set():
                            break
                        snap = self.snapshot_for_motion(require_stationary=False)
                        error = max(abs(a-b) for a, b in zip(snap["q"], goal))
                        if error < math.radians(0.15) and snap["stationary"]:
                            if settled_at is None:
                                settled_at = time.monotonic()
                            if time.monotonic() - settled_at >= 0.2:
                                break
                        else:
                            settled_at = None
                        if time.monotonic() - begin > duration_limit:
                            raise RuntimeError("运动超时或未能到达目标；请求 stopJ")
                        time.sleep(0.03)
                    if self.motion_stop_event.is_set():
                        break
                    move_active = False
                    self.capture_block_until = time.monotonic() + SETTLE_DELAY_S
                    self.queue.put(("motion_finished", f"J{joint+1} 增量点动完成；请等待相机/机械臂稳定后采样。"))
                except Exception as e:
                    self.queue.put(("motion_error", f"点动异常: {e}"))
                    self.motion_stop_event.set()
                    break
        except Exception as e:
            self.queue.put(("motion_error", f"运动控制连接/运行失败: {e}"))
            self.motion_stop_event.set()
        finally:
            if ctrl is not None:
                # 所有 RTDEControl 方法仅在这个线程中调用，避免线程间同时操作接口。
                if move_active:
                    try:
                        ctrl.stopJ(STOP_DECEL_RAD_S2, False)
                        self.queue.put(("motion_log", "已向 UR 下发 stopJ 减速停止指令。"))
                    except Exception as e:
                        self.queue.put(("motion_log", f"stopJ 发送失败: {e}。请立即检查机器人状态。"))
                try:
                    ctrl.stopScript()
                except Exception as e:
                    self.queue.put(("motion_log", f"结束 RTDE 控制脚本失败: {e}"))
                try:
                    ctrl.disconnect()
                except Exception:
                    pass
            self.motion_busy_event.clear()
            self.capture_block_until = time.monotonic() + SETTLE_DELAY_S
            self.queue.put(("motion_off", None))

    def log(self, msg):
        stamp = dt.datetime.now().strftime("%H:%M:%S")
        self.log_box.insert("end", f"[{stamp}] {msg}\n")
        self.log_box.see("end")

    def set_result(self, msg):
        self.result_box.configure(state="normal")
        self.result_box.delete("1.0", "end")
        self.result_box.insert("1.0", msg)
        self.result_box.configure(state="disabled")

    def refresh_devices(self):
        if rs is None:
            self.log("未安装 pyrealsense2，无法扫描相机。")
            return
        try:
            names = []
            ctx = rs.context()
            for dev in ctx.query_devices():
                if dev.supports(rs.camera_info.serial_number):
                    names.append(dev.get_info(rs.camera_info.serial_number))
            self.serial_combo["values"] = names
            if names and self.serial.get() not in names:
                self.serial.set(names[0])
            self.log(f"扫描到 RealSense: {names or '无'}")
        except Exception as e:
            self.log(f"扫描相机失败: {e}")

    def browse_dir(self):
        path = filedialog.askdirectory(initialdir=self.output_root.get())
        if path:
            self.output_root.set(path)

    def get_config(self):
        if not self.ip.get().strip():
            raise ValueError("UR IP 不能为空")
        if not self.serial.get().strip():
            raise ValueError("请扫描并选择 RealSense 的序列号")
        try:
            square_mm = float(self.square_mm.get())
        except ValueError:
            raise ValueError("方格边长必须是数字")
        if not (1 <= square_mm <= 200):
            raise ValueError("方格边长必须在 1~200 mm 之间（请用卡尺实测）")
        return dict(robot_ip=self.ip.get().strip(), serial=self.serial.get().strip(),
                    square_mm=square_mm, board_squares=list(BOARD_SQUARES),
                    inner_corners=list(INNER), rtde_hz=RTDE_HZ,
                    save_depth=bool(self.save_depth.get()),
                    reference_frame="UR actual TCP (not flange unless TCP offset is zero)")

    def set_config_enabled(self, enabled):
        mode = "normal" if enabled else "disabled"
        for item in (self.ip_entry, self.serial_combo, self.square_entry,
                     self.path_entry, self.goal_spin, self.depth_check):
            item.configure(state=mode)

    def connect(self):
        if (self.worker is not None and self.worker.is_alive()) or (
                self.motion_thread is not None and self.motion_thread.is_alive()):
            self.log("设备/运动控制线程仍在运行，请先停止并断开.")
            return
        if rs is None or rtde_receive is None or ImageTk is None:
            messagebox.showerror("缺少依赖", "请安装: pip install pyrealsense2 ur-rtde opencv-python numpy pillow")
            return
        try:
            conf = self.get_config()
            if self.samples and self.session_settings:
                old = self.session_settings["config"]
                if any(conf[k] != old[k] for k in ("robot_ip", "serial", "square_mm", "inner_corners")):
                    raise ValueError("已有数据的 UR IP/相机/方格尺寸不同；请新建一组数据或恢复原配置.")
        except Exception as e:
            messagebox.showerror("配置错误", str(e))
            return
        self.stop_event = threading.Event()
        self.capture_event.clear()
        self.cam_info = None
        with self.lock:
            self.robot_snapshot = None
        self.preview_settings = conf
        self.worker = threading.Thread(target=self.camera_worker, args=(conf, self.stop_event), daemon=True)
        self.worker.start()
        self.set_config_enabled(False)
        self.state_text.set("正在连接...")
        self.log(f"开始连接机器人 {conf['robot_ip']} (RTDE 50Hz) 与相机 {conf['serial']}")

    def disconnect(self):
        self.stop_motion()
        self.capture_event.clear()
        self.stop_event.set()
        self.state_text.set("正在断开...")
        self.log("已请求断开; 正等待工作线程释放相机与 RTDE.")

    def camera_worker(self, conf, stop_event):
        pipeline = None
        rtde = None
        try:
            # 该线程只通过 RTDEReceive 读机器人状态；运动由另一独立线程发送。
            rtde = rtde_receive.RTDEReceiveInterface(conf["robot_ip"], RTDE_HZ)
            if not rtde.isConnected():
                raise RuntimeError("RTDE 未连接")
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_device(conf["serial"])
            config.enable_stream(rs.stream.color, VIDEO_SIZE[0], VIDEO_SIZE[1], rs.format.bgr8, 30)
            if conf["save_depth"]:
                config.enable_stream(rs.stream.depth, VIDEO_SIZE[0], VIDEO_SIZE[1], rs.format.z16, 30)
            profile = pipeline.start(config)
            K, D, cam_meta = camera_parameters(profile)
            cam_meta["serial"] = conf["serial"]
            align = rs.align(rs.stream.color) if conf["save_depth"] else None
            depth_scale = (float(profile.get_device().first_depth_sensor().get_depth_scale())
                           if conf["save_depth"] else None)
            cam_meta["depth_scale_m_per_unit"] = depth_scale
            cam_meta["depth_aligned_to_color"] = bool(conf["save_depth"])
            self.queue.put(("ready", cam_meta))
            square_m = conf["square_mm"] / 1000.0
            while not stop_event.is_set():
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=1200)
                    color_frame = frames.get_color_frame()
                    if not color_frame:
                        continue
                except RuntimeError as e:
                    if stop_event.is_set():
                        break
                    raise RuntimeError(f"RealSense 读取失败: {e}") from e
                wants_capture = self.capture_event.is_set()
                depth_array = None
                if wants_capture and align is not None:
                    aligned = align.process(frames)
                    aligned_depth = aligned.get_depth_frame()
                    if aligned_depth:
                        depth_array = np.asanyarray(aligned_depth.get_data()).copy()
                frame = np.asanyarray(color_frame.get_data()).copy()
                frame_ms = float(color_frame.get_timestamp())
                frame_robot_time = time.time()
                q = np.asarray(rtde.getActualQ(), float)
                tcp = np.asarray(rtde.getActualTCPPose(), float)
                qd = np.asarray(rtde.getActualQd(), float)
                tcp_speed = np.asarray(rtde.getActualTCPSpeed(), float)
                safety_mode = int(rtde.getSafetyMode())
                if q.shape != (6,) or tcp.shape != (6,) or qd.shape != (6,):
                    raise RuntimeError("机器人关节/位姿数据长度异常")
                result, vis = detect_and_pose(frame, K, D, square_m)
                stationary = (np.max(np.abs(qd)) < MAX_MOVING_QD and
                              np.linalg.norm(tcp_speed[:3]) < MAX_MOVING_TCP_MPS)
                with self.lock:
                    self.robot_snapshot = dict(time=time.monotonic(), q=q.copy(),
                                               qd=qd.copy(), tcp=tcp.copy(),
                                               stationary=bool(stationary),
                                               safety_mode=safety_mode)
                safe_to_capture = (stationary and not self.motion_busy_event.is_set() and
                                   time.monotonic() >= self.capture_block_until)
                caption = "静止" if stationary else "机器人移动中 (禁止记录)"
                cv2.putText(vis, caption if caption.isascii() else
                            ("ROBOT STOPPED" if stationary else "ROBOT MOVING"),
                            (12, VIDEO_SIZE[1] - 18), cv2.FONT_HERSHEY_SIMPLEX,
                            0.6, (0, 180, 0) if stationary else (0, 40, 255), 2)
                with self.lock:
                    self.latest_preview = (vis, q.copy(), tcp.copy(),
                                           result["rmse"] if result else None, stationary)
                    self.last_preview_id += 1
                if wants_capture and self.capture_event.is_set():
                    self.capture_event.clear()
                    if conf["save_depth"] and depth_array is None:
                        self.queue.put(("info", "本帧没有有效的对齐深度图，未记录；检查 USB 3.0 连接."))
                        continue
                    if result is None:
                        self.queue.put(("info", "没有检测到完整的 11×8 内角点, 本次未记录."))
                        continue
                    if not safe_to_capture:
                        self.queue.put(("info", "机器人仍在运动/点动尚未结束或未达到稳定等待时间，请稍后再记录."))
                        continue
                    if result["rmse"] > MAX_RMSE_PX:
                        self.queue.put(("info", f"重投影 RMS {result['rmse']:.2f}px > "
                                                f"{MAX_RMSE_PX}px, 请调焦/改善照明后重试."))
                        continue
                    sample = dict(
                        timestamp_iso=dt.datetime.now().isoformat(timespec="milliseconds"),
                        host_timestamp_s=frame_robot_time,
                        camera_timestamp_ms=frame_ms,
                        joints_rad=q.tolist(), joint_speed_rad_s=qd.tolist(),
                        tcp_pose_xyz_rotvec=tcp.tolist(),
                        T_base_tcp=ur_pose_to_T(tcp).tolist(),
                        T_cam_board=result["T_cam_board"].tolist(),
                        rvec_board_in_cam=result["rvec"].tolist(),
                        tvec_board_in_cam_m=result["tvec"].tolist(),
                        rmse_px=result["rmse"],
                        corners_px=result["corners"].tolist(),
                    )
                    self.queue.put(("sample", sample, frame, depth_array))
        except Exception as e:
            self.queue.put(("error", f"设备线程异常: {e}\n{traceback.format_exc(limit=2)}"))
        finally:
            if pipeline is not None:
                try: pipeline.stop()
                except Exception: pass
            if rtde is not None:
                try: rtde.disconnect()
                except Exception: pass
            self.queue.put(("disconnected", None))

    def tick(self):
        for _ in range(25):
            try:
                item = self.queue.get_nowait()
            except queue.Empty:
                break
            event = item[0]
            if event == "ready":
                self.on_ready(item[1])
            elif event == "sample":
                self.on_sample(item[1], item[2], item[3])
            elif event == "info":
                self.log(item[1])
            elif event == "error":
                self.log(item[1])
                self.state_text.set("连接/采样失败，请检查日志")
            elif event == "disconnected":
                self.stop_motion()
                self.cam_info = None
                with self.lock:
                    self.robot_snapshot = None
                self.state_text.set("已断开")
                self.set_config_enabled(True)
                self.log("设备已断开")
            elif event == "motion_ready":
                self.motion_connecting = False
                if not self.motion_stop_event.is_set():
                    self.motion_armed = True
                    self.log("RTDEControl 已连接；关节 +/- 点动已启用。")
            elif event == "motion_finished":
                self.motion_busy_event.clear()
                self.log(item[1])
            elif event == "motion_log":
                self.log(item[1])
            elif event == "motion_error":
                self.motion_armed = False
                self.motion_connecting = False
                self.log(item[1])
            elif event == "motion_off":
                self.motion_armed = False
                self.motion_connecting = False
                self.log("RTDEControl 已关闭，关节点动锁定。")
        with self.lock:
            ident = self.last_preview_id
            payload = self.latest_preview
        if payload is not None and ident != getattr(self, "shown_preview_id", -1):
            self.shown_preview_id = ident
            frame, q, tcp, rmse, stationary = payload
            # 缩放以保持比例, 避免 GUI 图片对象被垃圾回收.
            h, w = frame.shape[:2]
            max_w = max(380, self.image_lbl.winfo_width() - 12)
            max_h = max(260, self.image_lbl.winfo_height() - 10)
            scale = min(max_w / w, max_h / h, 1.5)
            size = (int(w * scale), int(h * scale))
            scaled = cv2.resize(frame, size) if size != (w, h) else frame
            self.photoref = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)))
            self.image_lbl.configure(image=self.photoref, text="")
            self.q_text.set("关节角(°): " + ", ".join(f"{x:+.1f}" for x in np.rad2deg(q)))
            for idx, angle in enumerate(np.rad2deg(q)):
                self.joint_values[idx].set(f"J{idx+1}: {angle:+.1f}°")
            self.tcp_text.set("TCP xyz(m): " + ", ".join(f"{x:+.3f}" for x in tcp[:3]) +
                              " | 旋转向量(rad): " + ", ".join(f"{x:+.2f}" for x in tcp[3:]))
            if rmse is None:
                self.board_text.set("棋盘格未检测到")
            else:
                self.board_text.set(f"棋盘格已检测, 重投影 {rmse:.2f}px | " +
                                    ("机器人静止" if stationary else "机器人移动中"))
        self.update_motion_buttons()
        self.root.after(100, self.tick)

    def on_ready(self, info):
        if self.session_settings and not compatible_intrinsics(self.session_settings.get("camera"), info):
            self.log("当前内参与历史数据不一致！禁止混合采样。请检查分辨率/相机或新建数据组。")
            self.disconnect()
            return
        self.cam_info = info
        self.state_text.set("UR + 相机已连接")
        self.log(f"设备连接成功, RGB 内参 fx={info['fx']:.2f}, fy={info['fy']:.2f}, "
                 f"畸变模型={info['model']}，640×480")

    def take_sample(self):
        if self.worker is None or not self.worker.is_alive() or self.cam_info is None:
            self.log("请先确认设备连接成功.")
            return
        if self.motion_busy_event.is_set() or time.monotonic() < self.capture_block_until:
            self.log("关节点动执行中或尚未稳定，禁止采样；请稍后再试。")
            return
        if self.capture_event.is_set():
            self.log("已有待处理的采样请求.")
            return
        self.capture_event.set()
        self.log("请求采样：将在下一帧同时检查棋盘角点和机器人是否静止.")

    def sample_distance(self, a, b):
        A = np.asarray(a["T_base_tcp"], float)
        B = np.asarray(b["T_base_tcp"], float)
        return (np.linalg.norm(A[:3, 3] - B[:3, 3]),
                rotation_deg(A[:3, :3].T @ B[:3, :3]))

    def on_sample(self, sample, frame, depth):
        if self.samples:
            delta = [self.sample_distance(old, sample) for old in self.samples]
            if min(delta, key=lambda x: x[0] * 1000 + x[1] * 2)[0] < 0.015 and \
                    any(d < 0.015 and a < 5.0 for d, a in delta):
                self.log("新姿态与已有某组太接近（位置<15mm、旋转<5°），已跳过；请改变视角.")
                return
        if self.session_dir is None:
            folder = "eyehand_" + dt.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.session_dir = Path(self.output_root.get()).expanduser() / folder
        try:
            if self.session_settings is None:
                self.session_settings = dict(config=self.get_config(), camera=self.cam_info,
                                             created_iso=dt.datetime.now().isoformat())
            self.session_dir.mkdir(parents=True, exist_ok=True)
            sample["image_file"] = f"img_{len(self.samples)+1:03d}_{int(time.time()*1000)}.png"
            imagepath = self.session_dir / sample["image_file"]
            if not cv2.imwrite(str(imagepath), frame):
                raise IOError("OpenCV 写图片失败 (路径需可写且尽量只使用英文)")
            if depth is not None:
                sample["depth_file"] = sample["image_file"].replace("img_", "depth_", 1)
                sample["depth_scale_m_per_unit"] = self.cam_info["depth_scale_m_per_unit"]
                if not cv2.imwrite(str(self.session_dir / sample["depth_file"]), depth):
                    raise IOError("16位深度图保存失败")
            self.samples.append(sample)
            self.persist_session()
            self.last_result = None
            self.set_result("数据已发生变化, 请重新点击『求解手眼矩阵』.")
            self.update_table()
            self.log(f"已保存第 {len(self.samples)} 组, reproj RMS={sample['rmse_px']:.2f}px; "
                     f"目录: {self.session_dir}")
        except Exception as e:
            # 即使保存失败也不假装采样成功.
            if self.samples and self.samples[-1] is sample:
                self.samples.pop()
            self.log(f"保存采样失败: {e}")
            messagebox.showerror("保存失败", str(e))

    def persist_session(self):
        if self.session_dir and self.session_settings:
            save_json(self.session_dir / "session.json", dict(
                format_version=1, settings=self.session_settings, samples=self.samples,
                convention="T_A_B maps B -> A; T_base_tcp @ T_tcp_cam @ T_cam_board = T_base_board"))

    def update_table(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        for i, s in enumerate(self.samples, 1):
            xyz = np.asarray(s["tcp_pose_xyz_rotvec"][:3], float) * 1000
            self.tree.insert("", "end", iid=str(i - 1), values=(
                i, s["timestamp_iso"].split("T")[-1][:8], f"{s['rmse_px']:.2f}",
                ", ".join(f"{x:.0f}" for x in xyz)))
        self.counter_text.set(f"{len(self.samples)} / {self.goal.get()} 组")

    def remove_sample(self):
        selected = self.tree.selection()
        if not selected:
            self.log("请先在表格中选中一组.")
            return
        idx = int(selected[0])
        if not messagebox.askyesno("确认", f"删除第 {idx+1} 组记录？（原始图片会保留）"):
            return
        self.samples.pop(idx)
        self.last_result = None
        self.persist_session()
        self.update_table()
        self.set_result("采样数据已变化，需要重新求解.")
        self.log(f"已删除原第 {idx+1} 组记录 (图像不删除，供排查使用).")

    def calculate(self):
        try:
            answer = solve_handeye(self.samples)
            best = answer["best"]
            mat = np.array(best["T_tcp_cam"], float)
            c = best["consistency"]
            lines = [
                f"采样数: {answer['num_samples']}  |  方法: {best['method']}",
                f"TCP 姿态最大相对旋转: {answer['max_robot_rotation_deg']:.1f}°",
                f"平均每帧 solvePnP 重投影 RMSE: {answer['mean_pnp_rmse_px']:.3f} px",
                "", "T_tcp_cam =",
                np.array2string(mat, precision=6, suppress_small=True),
                "", "平移 (m): " + ", ".join(f"{x:.6f}" for x in mat[:3, 3]),
                f"板位姿一致性: 平移 RMS {c['translation_rms_mm']:.2f} mm; "
                f"旋转 RMS {c['rotation_rms_deg']:.2f}°",
                f"最大平移离散 {c['max_translation_mm']:.2f} mm, "
                f"最大旋转离散 {c['max_rotation_deg']:.2f}°",
                "", "其他候选 (平移RMS mm / 旋转RMS deg):",
            ]
            for item in answer["candidates"]:
                cc = item["consistency"]
                lines.append(f"{item['method']}: {cc['translation_rms_mm']:.2f} / "
                             f"{cc['rotation_rms_deg']:.2f}")
            if answer["failures"]:
                lines.append("失败方法: " + "; ".join(answer["failures"]))
            if answer["max_robot_rotation_deg"] < 25:
                lines.append("\n警告: 姿态旋转变化不足 25°, 强烈建议补采更多倾斜视角.")
            if c["translation_rms_mm"] > 8 or c["rotation_rms_deg"] > 2:
                lines.append("\n警告: 棋盘格跨帧一致性偏差较大，请检查安装、角点和 TCP 配置.")
            lines.append("\n注意：跨帧一致性≠真实绝对标定精度。建议留出验证组独立检查。")
            self.set_result("\n".join(lines))
            self.last_result = answer
            if self.session_dir is None:
                raise RuntimeError("当前数据没有关联的保存目录")
            save_json(self.session_dir / "result.json", dict(
                result=answer, settings=self.session_settings,
                result_convention="T_tcp_cam: camera color optical coordinates to robot current TCP",
                usage="p_base = T_base_tcp @ T_tcp_cam @ p_cam_color_optical"))
            np.save(self.session_dir / "T_tcp_cam.npy", mat)
            np.savetxt(self.session_dir / "T_tcp_cam.txt", mat, fmt="%.10f")
            self.log(f"求解成功 [{best['method']}]，标定矩阵已保存到 {self.session_dir}")
        except Exception as e:
            self.log(f"求解失败: {e}")
            messagebox.showerror("求解失败", str(e))

    def load_session(self):
        if (self.worker is not None and self.worker.is_alive()) or (
                self.motion_thread is not None and self.motion_thread.is_alive()):
            messagebox.showwarning("提示", "请停止运动控制并断开设备后再载入历史数据")
            return
        filepath = filedialog.askopenfilename(filetypes=[("标定记录", "session.json"), ("JSON", "*.json")])
        if not filepath:
            return
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            settings = data["settings"]
            conf = settings["config"]
            if conf["inner_corners"] != list(INNER):
                raise ValueError("记录的内角点规格不是 11×8")
            samples = data["samples"]
            for s in samples:
                if np.asarray(s["T_base_tcp"]).shape != (4, 4) or \
                        np.asarray(s["T_cam_board"]).shape != (4, 4):
                    raise ValueError("文件包含不完整的矩阵")
            self.samples = samples
            self.session_settings = settings
            self.session_dir = Path(filepath).resolve().parent
            self.last_result = None
            self.cam_info = None
            self.ip.set(conf["robot_ip"])
            self.serial.set(conf["serial"])
            self.square_mm.set(str(conf["square_mm"]))
            self.save_depth.set(bool(conf.get("save_depth", False)))
            self.update_table()
            self.set_result("已载入历史采样记录, 可离线点击『求解手眼矩阵』.")
            self.log(f"已载入 {len(self.samples)} 组: {filepath}")
        except Exception as e:
            messagebox.showerror("读取失败", str(e))

    def reset_session(self):
        if (self.worker is not None and self.worker.is_alive()) or (
                self.motion_thread is not None and self.motion_thread.is_alive()):
            messagebox.showwarning("提示", "请先停止运动控制并断开设备再新建数据组")
            return
        if self.samples and not messagebox.askyesno("确认", "新建数据组？已保存的旧数据不会被删除。"):
            return
        self.samples = []
        self.session_dir = None
        self.session_settings = None
        self.last_result = None
        self.cam_info = None
        self.update_table()
        self.set_result("新数据组已建立; 请连接 UR 和 RealSense 后开始采样.")
        self.log("新数据组已建立")

    def close(self):
        if self.shutdown_requested:
            return
        self.shutdown_requested = True
        self.stop_motion()
        self.stop_event.set()
        self.capture_event.clear()
        self.state_text.set("正在安全关闭：等待控制线程释放...")
        self.wait_shutdown()

    def wait_shutdown(self):
        if ((self.motion_thread is not None and self.motion_thread.is_alive()) or
                (self.worker is not None and self.worker.is_alive())):
            self.root.after(100, self.wait_shutdown)
            return
        self.root.destroy()


def synthetic_self_test():
    """检验矩阵方向与 OpenCV 手眼标定的连接方式, 不需要 UR/相机设备."""
    rng = np.random.default_rng(20261009)
    def random_T(angle=1.2, translation=0.5):
        axis = rng.normal(size=3)
        axis /= np.linalg.norm(axis)
        R = cv2.Rodrigues(axis * rng.uniform(-angle, angle))[0]
        return iso(R, rng.uniform(-translation, translation, 3))
    X_true = random_T(angle=1.0, translation=0.15)  # T_tcp_cam
    B_true = random_T(angle=0.6, translation=0.6)   # T_base_board
    dataset = []
    for _ in range(24):
        G = random_T()                             # T_base_tcp
        C = inverse(X_true) @ inverse(G) @ B_true  # T_cam_board
        dataset.append(dict(T_base_tcp=G.tolist(), T_cam_board=C.tolist(), rmse_px=0.0))
    result = solve_handeye(dataset)
    best = np.array(result["best"]["T_tcp_cam"])
    trans_err_mm = np.linalg.norm(best[:3, 3] - X_true[:3, 3]) * 1000
    rot_err_deg = rotation_deg(best[:3, :3].T @ X_true[:3, :3])
    print("Synthetic hand-eye check:", result["best"]["method"])
    print("  translation error (mm):", f"{trans_err_mm:.9f}")
    print("  rotation error (deg):", f"{rot_err_deg:.9f}")
    assert trans_err_mm < 1e-3 and rot_err_deg < 1e-3, "变换方向或求解逻辑有误"
    print("PASS: T_base_tcp @ T_tcp_cam @ T_cam_board = T_base_board")


def jog_self_test():
    q = np.radians([10., -35., 27., 0., 16., -20.])
    goal, v, a = compute_jog_target(q, 1, 1, 1.0, 5.0, 20.0)
    assert np.allclose(np.asarray(goal)[[0, 2, 3, 4, 5]], q[[0, 2, 3, 4, 5]])
    assert abs(math.degrees(goal[1]) + 34.0) < 1e-8
    assert abs(math.degrees(v) - 5.0) < 1e-8 and abs(math.degrees(a) - 20.0) < 1e-8
    for bad_q, j, d, step, speed, accel in [
        (np.radians([0., 179.9, 0., 0., 0., 0.]), 1, 1, 1., 5., 20.),
        (np.zeros(6), 6, 1, 1., 5., 20.),
        (np.zeros(6), 1, 1, 30., 5., 20.),
        (np.zeros(6), 1, 1, 1., 100., 20.),
        (np.zeros(6), 1, 1, 1., 5., -10.),
    ]:
        try:
            compute_jog_target(bad_q, j, d, step, speed, accel)
            raise AssertionError("危险/无效输入未被拒绝")
        except ValueError:
            pass
    print("PASS: 关节点动目标/步长/速度/加速度/边界离线检查")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="运行无需硬件的模拟手眼标定测试")
    parser.add_argument("--jog-self-test", action="store_true", help="无需硬件检查关节点动计算和软件界限")
    args = parser.parse_args()
    if args.self_test:
        synthetic_self_test()
        return
    if args.jog_self_test:
        jog_self_test()
        return
    root = tk.Tk()
    CalibratorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
