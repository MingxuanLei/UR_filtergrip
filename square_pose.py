# -*- coding: utf-8 -*-
"""UR5e + RealSense D435i 金属正方形顶面位姿检测 GUI。

检测: ROI 内 LSD 四边识别 + 已知80mm正方形 PnP；深度作为独立质量检查。
坐标: T_A_B 将 B 系中的点变换到 A 系。手眼结果 T_tcp_cam 应来自眼在手上标定。
本程序仅通过 RTDEReceive 读取机械臂状态，不发送任何运动命令。

启动: python square_pose.py
离线测试: python square_pose.py --self-test
照片调试: python square_pose.py --image 图片.png  (没有真实内参时仅做2D检测)
"""
from __future__ import annotations

import argparse
import datetime as datetime
import json
import math
import threading
import time
import traceback
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
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

APP_TITLE = "UR5e 正方形视觉定位"
EDGE_M = 0.080
GAP_M = 0.035
VIDEO_W, VIDEO_H, VIDEO_FPS = 640, 480, 30
MOVING_QD_MAX = 0.008           # rad/s, snapshot safety guard (not a safety controller)
MOVING_TCP_MAX = 0.004        # m/s
MAX_ROBOT_SAMPLE_AGE = 0.45  # s
MAX_PNP_RMSE = 3.0           # px: reject clearly ill-fitting quadrilateral
SAVE_ROOT = Path.cwd() / "square_pose_data"


def homogeneous(R=None, t=None):
    T = np.eye(4, dtype=np.float64)
    if R is not None:
        T[:3, :3] = np.asarray(R, np.float64).reshape(3, 3)
    if t is not None:
        T[:3, 3] = np.asarray(t, np.float64).reshape(3)
    return T


def ur_pose_to_T(pose):
    p = np.asarray(pose, dtype=float).reshape(6)
    return homogeneous(cv2.Rodrigues(p[3:].reshape(3, 1))[0], p[:3])


def rigid_matrix_check(T):
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.isfinite(T).all():
        raise ValueError("矩阵必须是有限的 4×4 数组")
    R = T[:3, :3]
    if not np.allclose(T[3, :], [0, 0, 0, 1], atol=1e-5):
        raise ValueError("矩阵最后一行应为 [0, 0, 0, 1]")
    if np.linalg.norm(R.T @ R - np.eye(3)) > 0.03 or abs(np.linalg.det(R) - 1) > 0.02:
        raise ValueError("旋转部分不是合法的旋转矩阵")
    return T.copy()


def normal_angle_90(angle_deg):
    """正方形的无方向边朝向，统一为 [-45, 45) 度。"""
    return (float(angle_deg) + 45.0) % 90.0 - 45.0


def equiv_delta_deg(new_angle, old_angle):
    return normal_angle_90(float(new_angle) - float(old_angle))


def canonical_corners(pts):
    """图像坐标中的四边形，返回以 TL 为起点沿图像顺时针 TL,TR,BR,BL。

    不用 x+y 简单排序整个集合，以避免旋转约45°时相邻角交换。
    """
    p = np.asarray(pts, np.float64).reshape(4, 2)
    if not np.isfinite(p).all() or len(np.unique(np.round(p, 3), axis=0)) != 4:
        raise ValueError("四角点无效")
    c = p.mean(axis=0)
    ang = np.arctan2(p[:, 1] - c[1], p[:, 0] - c[0])
    out = p[np.argsort(ang)]
    # 图像下向正： atan2 升序对应图像上的顺时针顺序
    i = np.argmin(out[:, 0] + out[:, 1])
    out = np.roll(out, -int(i), axis=0)
    if cv2.contourArea(out.astype(np.float32)) < 100:
        raise ValueError("四边形面积太小")
    if not cv2.isContourConvex(out.astype(np.float32)):
        raise ValueError("不是凸四边形")
    return out.astype(np.float64)


def quad_properties(pts):
    p = canonical_corners(pts)
    edges = np.roll(p, -1, axis=0) - p
    lens = np.linalg.norm(edges, axis=1)
    if np.min(lens) < 10 or np.max(lens) / np.min(lens) > 3.8:
        return False
    area = cv2.contourArea(p.astype(np.float32))
    if area < 300:
        return False
    for i in range(4):
        a, b = edges[i-1], edges[i]
        cosine = abs(float(np.dot(a, b))) / (np.linalg.norm(a)*np.linalg.norm(b))
        if cosine > 0.85:  # 极端尖角不可能是常规俯视金属方块
            return False
    return True


def orientation_diff_mod_pi(phi, alpha):
    return abs((phi - alpha + np.pi/2) % np.pi - np.pi/2)


def line_clusters(segments, unit_vec, image_center, min_len):
    """按同一方向的法线距离聚类，缓解金属边缘断裂、LSD 双边缘问题。"""
    n = np.array([-unit_vec[1], unit_vec[0]], np.float64)
    segs = []
    for seg in segments:
        xy1, xy2 = seg[0], seg[1]
        length = np.linalg.norm(xy2-xy1)
        if length < min_len:
            continue
        v = (xy2-xy1)/length
        err = math.acos(float(np.clip(abs(np.dot(v, unit_vec)), 0, 1)))
        if err > math.radians(19):
            continue
        rho = float(np.dot((xy1+xy2)/2 - image_center, n))
        segs.append((rho, length, np.stack([xy1, xy2]), err))
    segs.sort(key=lambda z: z[0])
    groups = []
    # 对聚类采用固定距离门限；长边在图像内可分成多段
    for r, l, ends, err in segs:
        best = None
        if groups:
            d = np.array([abs(r-g['rho']) for g in groups])
            j = int(np.argmin(d))
            if d[j] <= 8.0:
                best = groups[j]
        if best is None:
            best = {'rho': r, 'weight': 0., 'pts': [], 'rho_sum': 0.}
            groups.append(best)
        best['weight'] += l
        best['rho_sum'] += r*l
        best['rho'] = best['rho_sum']/best['weight']
        best['pts'].append(ends)
    clusters = []
    for g in groups:
        if g['weight'] < min_len * 1.7:
            continue
        pts = np.concatenate(g['pts'], axis=0).astype(np.float32)
        vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).reshape(-1)
        v = np.array([vx, vy], float)
        if np.dot(v, unit_vec) < 0:
            v = -v
        if math.acos(float(np.clip(np.dot(v, unit_vec), -1, 1))) > math.radians(20):
            v = unit_vec.copy()
        nfit = np.array([-v[1], v[0]])
        center = np.mean(pts, axis=0)
        clusters.append({'n': nfit, 'rho': float(np.dot(nfit, center)),
                         'weight': g['weight'], 'points': pts})
    return sorted(clusters, key=lambda x: -x['weight'])[:9]


def intersect_lines(a, b):
    n = np.stack([a['n'], b['n']])
    if abs(np.linalg.det(n)) < 0.13:
        return None
    return np.linalg.solve(n, [a['rho'], b['rho']])


def detect_quad_lsd(bgr, roi=None):
    """优先选取 ROI 中两组近正交的平行边，不依赖金属表面颜色或深度。

    只适合作为候选四边形：自动检测不保证每帧正确，应看可视化和质量指标。
    """
    h, w = bgr.shape[:2]
    if roi is None:
        roi = (0, 0, w, h)
    x1, y1, x2, y2 = map(int, roi)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2-x1 < 45 or y2-y1 < 45:
        return None
    crop = bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    detector = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    lines = detector.detect(gray)[0]
    if lines is None:
        return None
    seg = np.asarray(lines[:, 0, :], np.float64).reshape(-1, 2, 2)
    lengths = np.linalg.norm(seg[:, 1, :] - seg[:, 0, :], axis=1)
    min_len = max(14., min(x2-x1, y2-y1)*0.08)
    keep = lengths >= min_len
    seg = seg[keep]
    lengths = lengths[keep]
    if len(seg) < 4:
        return None
    vecs = seg[:, 1, :] - seg[:, 0, :]
    angles = np.arctan2(vecs[:, 1], vecs[:, 0])
    # 对 0°、90°、180° 等价轴使用 4θ 周期投票，以涵盖旋转后的正方形
    score = np.zeros(90)
    for idx in range(90):
        ang = math.radians(idx)
        diff = np.angle(np.exp(4j*(angles-ang)))/4
        score[idx] = np.sum(lengths * np.exp(-0.5*(diff/math.radians(11))**2))
    alpha = math.radians(int(np.argmax(score)))
    a = np.array([math.cos(alpha), math.sin(alpha)])
    b = np.array([-math.sin(alpha), math.cos(alpha)])
    center = np.array([(x2-x1)/2, (y2-y1)/2])
    clusters_a = line_clusters(seg, a, center, min_len)
    clusters_b = line_clusters(seg, b, center, min_len)
    if len(clusters_a) < 2 or len(clusters_b) < 2:
        return None
    best = None
    min_area = (x2-x1)*(y2-y1)*0.11
    for ia in range(len(clusters_a)):
        for ja in range(ia+1, len(clusters_a)):
            l1, l2 = clusters_a[ia], clusters_a[ja]
            da = abs(np.dot(l1['n'], (l2['points'][0] - l1['points'][0])))
            if da < 0.30*min(x2-x1, y2-y1):
                continue
            for ib in range(len(clusters_b)):
                for jb in range(ib+1, len(clusters_b)):
                    m1, m2 = clusters_b[ib], clusters_b[jb]
                    db = abs(np.dot(m1['n'], (m2['points'][0] - m1['points'][0])))
                    if db < 0.30*min(x2-x1, y2-y1):
                        continue
                    pts = [intersect_lines(l1,m1), intersect_lines(l1,m2),
                           intersect_lines(l2,m1), intersect_lines(l2,m2)]
                    if any(p is None for p in pts):
                        continue
                    p = np.asarray(pts, np.float64)
                    if not np.isfinite(p).all():
                        continue
                    if np.min(p[:,0]) < -17 or np.min(p[:,1]) < -17 or (np.max(p[:,0]) > x2-x1+17 or np.max(p[:,1]) > y2-y1+17):
                        continue
                    try:
                        q = canonical_corners(p)
                    except ValueError:
                        continue
                    if not quad_properties(q):
                        continue
                    area = cv2.contourArea(q.astype(np.float32))
                    if area < min_area or area > (x2-x1)*(y2-y1)*1.08:
                        continue
                    qcen = q.mean(axis=0)
                    offset = np.linalg.norm((qcen-center) / np.array([x2-x1, y2-y1]))
                    # 优先围住 ROI 中央主体，优先长且连续的真实轮廓，抑制杂散反射线
                    support = sum(math.log(1+cl['weight']) for cl in [l1,l2,m1,m2])
                    area_ratio = area / ((x2-x1)*(y2-y1))
                    ranking = support + 2*area_ratio - 4*offset
                    if best is None or ranking > best[0]:
                        best = (ranking, q + np.array([x1,y1]))
    return None if best is None else best[1]


def square_object_points(edge_m=EDGE_M):
    """局部 XY 平面，角点 TL,TR,BR,BL；图像朝下的局部Y使可见顶面外法向为局部-Z。"""
    a = edge_m / 2.0
    return np.array([[-a, -a, 0], [+a,-a,0], [+a,+a,0], [-a,+a,0]], dtype=np.float64)


def pnp_square(corners_px, K, D, edge_m=EDGE_M, T_base_cam=None):
    """平面 IPPE 双解 + 实际水平面法向约束 / 重投影评分。

    返回所有矩阵方向遵循 T_A_B: p_A = T_A_B @ p_B。
    若没有机器人 Base 位姿，则法向仅限制朝向相机，并以重投影优先。
    """
    pts = canonical_corners(corners_px)
    obj = square_object_points(edge_m)
    K = np.asarray(K, np.float64).reshape(3, 3)
    D = np.asarray(D, np.float64).reshape(-1, 1)
    if np.any(~np.isfinite(K)) or K[0,0] <= 0 or K[1,1] <= 0:
        raise ValueError("相机内参无效")
    if not quad_properties(pts):
        raise ValueError("四边形不满足基本几何要求")
    res = cv2.solvePnPGeneric(obj, pts, K, D, flags=cv2.SOLVEPNP_IPPE)
    candidates = []
    rv_tv = list(zip(res[1], res[2])) if res[0] else []
    # IPPE在严格正视、零倾角时偶尔会返回NaN，需退回迭代PnP
    if not any(np.isfinite(rv).all() and np.isfinite(tv).all() for rv,tv in rv_tv):
        good, rv, tv = cv2.solvePnP(obj, pts, K, D, flags=cv2.SOLVEPNP_ITERATIVE)
        if not good:
            raise ValueError('PnP求解失败')
        rv_tv = [(rv,tv)]
    for rvec, tvec in rv_tv:
        if not np.isfinite(rvec).all() or not np.isfinite(tvec).all():
            continue
        R = cv2.Rodrigues(rvec)[0]
        t = np.asarray(tvec, float).reshape(3)
        if t[2] <= 0 or not np.isfinite(t).all():
            continue
        reproject, _ = cv2.projectPoints(obj, rvec, tvec, K, D)
        err = float(np.sqrt(np.mean(np.sum((reproject.reshape(-1,2)-pts)**2, axis=1))))
        outward_normal_cam = -(R[:, 2])
        # 在相机前方的上表面，其法线通常指向相机
        front = max(0., float(-outward_normal_cam[2]))
        if T_base_cam is not None:
            upward = float((T_base_cam[:3,:3] @ outward_normal_cam)[2])
            # 如果有 Base，将“顶面大致水平且朝上”作为先验，用于消解 IPPE 双解
            penalty = max(0., 0.70-upward)*5.0
        else:
            upward = None
            penalty = (1-front)*1.0
        candidates.append(dict(T_cam_top=homogeneous(R,t), rvec=np.asarray(rvec).reshape(3),
                               tvec=t, rmse_px=err, score=err+penalty,
                               upward_dot=upward, normal_cam=outward_normal_cam))
    if not candidates:
        raise ValueError("四边形没有有效的相机前方位姿解")
    best = min(candidates, key=lambda c:c['score'])
    if best['rmse_px'] > MAX_PNP_RMSE:
        raise ValueError(f"PnP重投影 RMSE {best['rmse_px']:.2f}px > {MAX_PNP_RMSE}px")
    if T_base_cam is not None and best['upward_dot'] < 0.45:
        raise ValueError("顶面法向与 Base +Z 方向不一致；请检查角点、手眼矩阵或工作台是否水平")
    return best


def base_pose_from_square(T_base_cam, pnp):
    T_base_top = T_base_cam @ pnp['T_cam_top']
    edge = T_base_top[:3, 0]  # 顶面局部 X 方向在机器人 Base 中的表示
    yaw = normal_angle_90(math.degrees(math.atan2(edge[1], edge[0])))
    return dict(center_m=T_base_top[:3, 3].copy(), yaw_deg=yaw, T_base_top=T_base_top)


def depth_stats(depth_m, quad, K=None, D=None, T_base_cam=None,
                target_gap_m=GAP_M):
    """深度是辅助验证，金属表面可能无效；从不把相机 Z 差强制设为35mm。

    若已知 Base 位姿，取中心顶面内部与周边圆盘候选环带的 Base Z 中位数之差。
    只有存在足够有效样本且离散小的时候，才提示与35mm设计差值是否一致。
    """
    if depth_m is None:
        return {'valid_frac': None, 'median_m': None, 'depth_note': '无深度帧'}
    h, w = depth_m.shape
    c = np.asarray(quad, np.float64).mean(axis=0)
    quad = np.asarray(quad, np.float64)
    mask_inside = np.zeros((h,w), np.uint8)
    inner = c + 0.68 * (quad-c)
    cv2.fillConvexPoly(mask_inside, np.round(inner).astype(np.int32), 255)
    valid = (depth_m > 0.05) & (depth_m < 3.0) & np.isfinite(depth_m)
    inside = mask_inside > 0
    frac = float(np.sum(inside & valid)/max(1, int(np.sum(inside))))
    vals = depth_m[inside & valid]
    med = float(np.median(vals)) if len(vals) else None
    mad = float(np.median(np.abs(vals-med))) if len(vals) else None
    result = dict(valid_frac=frac, median_m=med, mad_m=mad,
                  depth_note=('顶面深度有效' if frac >= 0.35 else '顶面金属反光/深度缺失较多，PnP为主'),
                  expected_top_plate_gap_m=float(target_gap_m), gap_measured_mm=None)
    if K is None or T_base_cam is None:
        return result
    # 测量圆盘候选环带: 目标外沿约1.17~1.63倍范围，仅作为参考而不是精确平面分割
    outer_mask = np.zeros_like(mask_inside)
    exclude_mask = np.zeros_like(mask_inside)
    cv2.fillConvexPoly(outer_mask, np.round(c + 1.65*(quad-c)).astype(np.int32), 255)
    cv2.fillConvexPoly(exclude_mask, np.round(c + 1.17*(quad-c)).astype(np.int32), 255)
    ring = (outer_mask > 0) & (exclude_mask == 0)
    if np.sum(inside & valid) < 70 or np.sum(ring & valid) < 150:
        return result
    ys0, xs0 = np.nonzero(inside & valid)
    ys1, xs1 = np.nonzero(ring & valid)
    # 限制点数量，以减少每帧 undistortPoints 的负担
    if len(xs0) > 3500:
        sel=np.linspace(0,len(xs0)-1,3500).astype(int)
        xs0, ys0 = xs0[sel],ys0[sel]
    if len(xs1) > 4500:
        sel=np.linspace(0,len(xs1)-1,4500).astype(int)
        xs1, ys1 = xs1[sel],ys1[sel]
    def base_heights(xs,ys):
        pix=np.stack([xs,ys],axis=1).astype(np.float64).reshape(-1,1,2)
        und=cv2.undistortPoints(pix,np.asarray(K,float),np.asarray(D,float))[:,0,:]
        z=depth_m[ys,xs].astype(float)
        pcam=np.column_stack([und[:,0]*z,und[:,1]*z,z])
        return pcam @ T_base_cam[2,:3].T + T_base_cam[2,3]
    h_top=base_heights(xs0,ys0)
    h_ring=base_heights(xs1,ys1)
    top_med=float(np.median(h_top));ring_med=float(np.median(h_ring))
    mad_top=float(np.median(np.abs(h_top-top_med)))
    mad_ring=float(np.median(np.abs(h_ring-ring_med)))
    diff_mm=(top_med-ring_med)*1000.0
    # 环带可能包括其他机构和阴影，质量差时不给出误导性的35mm判断
    if mad_top<0.010 and mad_ring<0.015:
        result['gap_measured_mm']=diff_mm
        result['gap_consistent']=abs(diff_mm-target_gap_m*1000) <= 13
        result['depth_note'] += ('；估计高差与35mm接近' if result['gap_consistent']
                                 else '；估计高差不符（检查环带是否落在圆盘）')
    else:
        result['depth_note'] += '；周围圆盘深度离散大，无法可靠验证35mm'
    return result


def export_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        json.dump(obj,f,ensure_ascii=False,indent=2)


def make_jsonable(obj):
    if isinstance(obj,np.ndarray): return obj.tolist()
    if isinstance(obj,np.generic): return obj.item()
    if isinstance(obj,dict): return {str(k):make_jsonable(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)): return [make_jsonable(v) for v in obj]
    return obj


@dataclass
class ImagePacket:
    color: np.ndarray
    depth_m: np.ndarray | None
    K: np.ndarray | None
    D: np.ndarray | None
    cam_serial: str
    ts_monotonic: float
    ur: dict | None = None
    source: str = "camera"


class SquarePoseApp:
    def __init__(self, root, image_path=None):
        self.root = root
        self.root.title(APP_TITLE + "（只读定位，不控制机器人）")
        self.root.geometry('1280x810')
        self.root.minsize(1020,690)
        self.closing = False
        self.stop_cam = threading.Event()
        self.stop_ur = threading.Event()
        self.cam_thread = None
        self.ur_thread = None
        self.packet_lock = threading.Lock()
        self.packet = None
        self.ur_lock = threading.Lock()
        self.ur_state = None
        self.cam_active = False
        self.ur_active = False
        self.preview_image = None
        self.last_packet_timestamp = -1.0
        self.last_detect_key = None
        self.last_frame_serial = None
        self.manual_points = []
        self.manual_quad = None
        self.manual_quad_mono = None
        self.roi = None
        self.select_mode = None
        self.start_xy = None
        self.draw_mapping = None
        self.latest_detection = None
        self.latest_saved_detection = None
        self.reference = None
        self.T_tcp_cam = None
        self.T_tcp_cam_path = None
        self.cam_serial = tk.StringVar(value="")
        self.robot_ip = tk.StringVar(value="192.168.3.15")
        self.edge_mm = tk.StringVar(value="80.0")
        self.gap_mm = tk.StringVar(value="35.0")
        self.auto_mode = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value='请先连接D435i / 打开图像；可选择加载手眼矩阵与UR5e')
        self.pose_var = tk.StringVar(value='等待有效顶面检测')
        self.ref_var = tk.StringVar(value='尚未记录示教参考位姿')
        self.delta_var = tk.StringVar(value='ΔX / ΔY / ΔYaw: —')
        self.depth_var = tk.StringVar(value='深度质量：尚无数据')
        self.camera_intrinsics = None
        self.thread_error = None
        self.thread_ur_error = None
        self.saved_records = []
        self.last_result = None
        self.build_ui()
        self.refresh_devices()
        if image_path:
            self.load_photo(image_path)
        self.root.after(100,self.tick)
        self.root.protocol('WM_DELETE_WINDOW',self.close)

    def build_ui(self):
        main = ttk.Frame(self.root, padding=9)
        main.pack(fill='both', expand=True)
        main.columnconfigure(0,weight=4)
        main.columnconfigure(1,weight=2)
        main.rowconfigure(0,weight=1)
        self.left = ttk.Frame(main)
        self.left.grid(row=0,column=0,sticky='nsew',padx=(0,9))
        self.left.rowconfigure(1,weight=1)
        self.left.columnconfigure(0,weight=1)
        ttk.Label(self.left,text='RGB 实时预览｜绿色四边形=检测顶面｜红点=中心',font=('Microsoft YaHei UI',12,'bold')).grid(row=0,column=0,sticky='w',pady=(0,5))
        self.canvas=tk.Canvas(self.left,bg='#25313a',highlightthickness=0,cursor='crosshair')
        self.canvas.grid(row=1,column=0,sticky='nsew')
        self.canvas.bind('<ButtonPress-1>',self.mouse_down)
        self.canvas.bind('<B1-Motion>',self.mouse_move)
        self.canvas.bind('<ButtonRelease-1>',self.mouse_up)
        ttk.Label(self.left,text='先用「框选ROI」圈住顶面及其运动范围（建议略放大）；反光识别失败可用「手点四角」临时测量。',wraplength=700).grid(row=2,column=0,sticky='w',pady=5)
        self.right_canvas=tk.Canvas(main,highlightthickness=0,borderwidth=0)
        self.right_canvas.grid(row=0,column=1,sticky='nsew')
        sb=ttk.Scrollbar(main,orient='vertical',command=self.right_canvas.yview)
        sb.grid(row=0,column=2,sticky='ns')
        self.right_canvas.configure(yscrollcommand=sb.set)
        self.right=ttk.Frame(self.right_canvas,padding=(4,2,10,4))
        self.right_win=self.right_canvas.create_window((0,0),window=self.right,anchor='nw')
        self.right.bind('<Configure>',lambda e:self.right_canvas.configure(scrollregion=self.right_canvas.bbox('all')))
        self.right_canvas.bind('<Configure>',lambda e:self.right_canvas.itemconfigure(self.right_win,width=e.width))
        self.right.columnconfigure(0,weight=1)
        n=0
        f=ttk.LabelFrame(self.right,text='设备连接 / 可离线看图',padding=7)
        f.grid(row=n,column=0,sticky='ew',pady=4);n+=1
        f.columnconfigure(1,weight=1)
        ttk.Label(f,text='UR IP').grid(row=0,column=0,sticky='w')
        ttk.Entry(f,textvariable=self.robot_ip,width=19).grid(row=0,column=1,sticky='ew')
        ttk.Button(f,text='连接UR(只读)',command=self.connect_ur).grid(row=0,column=2,padx=2)
        ttk.Button(f,text='断开',command=self.disconnect_ur).grid(row=0,column=3,padx=2)
        ttk.Label(f,text='D435i').grid(row=1,column=0,sticky='w',pady=4)
        self.cam_box=ttk.Combobox(f,textvariable=self.cam_serial,width=23,state='readonly')
        self.cam_box.grid(row=1,column=1,sticky='ew')
        ttk.Button(f,text='扫描',command=self.refresh_devices).grid(row=1,column=2,padx=2)
        ttk.Button(f,text='启动',command=self.connect_cam).grid(row=1,column=3,padx=2)
        buttons=ttk.Frame(f)
        buttons.grid(row=2,column=0,columnspan=4,sticky='ew')
        ttk.Button(buttons,text='断开相机',command=self.disconnect_cam).pack(side='left',padx=2)
        ttk.Button(buttons,text='打开图片',command=self.open_image).pack(side='left',padx=2)
        ttk.Button(buttons,text='加载手眼矩阵',command=self.load_handeye).pack(side='left',padx=2)
        f=ttk.LabelFrame(self.right,text='目标参数（单位：mm）',padding=7)
        f.grid(row=n,column=0,sticky='ew',pady=4);n+=1
        p=ttk.Frame(f);p.pack(fill='x')
        ttk.Label(p,text='顶面边长').pack(side='left')
        ttk.Entry(p,textvariable=self.edge_mm,width=8).pack(side='left',padx=(4,15))
        ttk.Label(p,text='顶面高于圆盘').pack(side='left')
        ttk.Entry(p,textvariable=self.gap_mm,width=8).pack(side='left',padx=4)
        ttk.Label(f,text='35 mm仅用于几何/深度质量参考；不把相机 Z 深度差硬设为35 mm。',foreground='#666666',wraplength=420).pack(anchor='w',pady=(4,0))
        f=ttk.LabelFrame(self.right,text='顶面识别',padding=7)
        f.grid(row=n,column=0,sticky='ew',pady=4);n+=1
        a=ttk.Frame(f);a.pack(fill='x')
        ttk.Button(a,text='框选ROI',command=self.set_roi_mode).pack(side='left',padx=2)
        ttk.Button(a,text='手点四角',command=self.set_corners_mode).pack(side='left',padx=2)
        ttk.Button(a,text='清除选区',command=self.clear_selection).pack(side='left',padx=2)
        ttk.Checkbutton(f,text='自动识别ROI中的顶面四边形 (LSD)',variable=self.auto_mode,command=self.invalidate).pack(anchor='w',pady=4)
        ttk.Label(f,text='如果边缘受反光干扰：先框选较紧的ROI；必要时依次点击顶面4角（任意起点，沿周界连续）。',wraplength=430).pack(anchor='w')
        f=ttk.LabelFrame(self.right,text='顶面位姿 / 相对示教变化',padding=7)
        f.grid(row=n,column=0,sticky='ew',pady=4);n+=1
        ttk.Label(f,textvariable=self.pose_var,justify='left',wraplength=420).pack(anchor='w',pady=3)
        ttk.Label(f,textvariable=self.depth_var,justify='left',wraplength=430,foreground='#555555').pack(anchor='w',pady=3)
        ttk.Separator(f).pack(fill='x',pady=5)
        ttk.Label(f,textvariable=self.ref_var,wraplength=425).pack(anchor='w',pady=2)
        ttk.Label(f,textvariable=self.delta_var,wraplength=425,font=('Microsoft YaHei UI',11,'bold')).pack(anchor='w',pady=4)
        r=ttk.Frame(f);r.pack(fill='x',pady=4)
        ttk.Button(r,text='记录示教参考',command=self.set_reference).pack(side='left',padx=2)
        ttk.Button(r,text='保存当前检测',command=self.save_current).pack(side='left',padx=2)
        r2=ttk.Frame(f);r2.pack(fill='x')
        ttk.Button(r2,text='加载参考JSON',command=self.load_reference).pack(side='left',padx=2)
        ttk.Button(r2,text='清空参考',command=self.clear_reference).pack(side='left',padx=2)
        f=ttk.LabelFrame(self.right,text='日志',padding=7)
        f.grid(row=n,column=0,sticky='ew',pady=4);n+=1
        self.log_box=tk.Text(f,height=9,wrap='word',state='disabled',font=('Consolas',9))
        self.log_box.pack(fill='both',expand=True)
        ttk.Label(self.right,textvariable=self.status_var,wraplength=430,foreground='#246b8d').grid(row=n,column=0,sticky='ew',pady=8)
        self.log('程序不会发送UR运动指令。为获得Base坐标，请加载 T_tcp_cam 并连接UR(只读)。')

    def log(self,msg):
        if self.closing: return
        self.log_box.configure(state='normal')
        self.log_box.insert('end',f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}\n")
        self.log_box.see('end')
        self.log_box.configure(state='disabled')

    def refresh_devices(self):
        items=[]
        if rs is not None:
            try:
                ctx=rs.context()
                for dev in ctx.query_devices():
                    items.append(dev.get_info(rs.camera_info.serial_number))
            except Exception as ex:self.log(f'扫描RealSense失败: {ex}')
        self.cam_box['values']=items
        if items and self.cam_serial.get() not in items:
            self.cam_serial.set(items[0])
        self.log(f'扫描到 {len(items)} 台 RealSense；{("未安装 pyrealsense2" if rs is None else "可选序列号")}')

    def connect_cam(self):
        if rs is None:
            messagebox.showerror('缺少依赖','请先 pip install pyrealsense2')
            return
        if self.cam_thread and self.cam_thread.is_alive():return
        sn=self.cam_serial.get().strip()
        if not sn:
            messagebox.showwarning('未选择相机','先点击扫描，选择相机序列号')
            return
        self.stop_cam=threading.Event()
        self.cam_thread=threading.Thread(target=self.camera_worker,args=(sn,self.stop_cam),daemon=True)
        self.cam_thread.start()
        self.status_var.set('正在连接RealSense ...')

    def disconnect_cam(self):
        self.stop_cam.set()
        self.cam_active=False
        self.status_var.set('已请求断开相机')

    def camera_worker(self,serial,stop_event):
        pipeline=None
        try:
            pipeline=rs.pipeline()
            cfg=rs.config()
            cfg.enable_device(serial)
            cfg.enable_stream(rs.stream.color,VIDEO_W,VIDEO_H,rs.format.bgr8,VIDEO_FPS)
            cfg.enable_stream(rs.stream.depth,VIDEO_W,VIDEO_H,rs.format.z16,VIDEO_FPS)
            profile=pipeline.start(cfg)
            align=rs.align(rs.stream.color)
            scale=profile.get_device().first_depth_sensor().get_depth_scale()
            intr=profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
            K=np.array([[intr.fx,0,intr.ppx],[0,intr.fy,intr.ppy],[0,0,1]],float)
            D=np.array(intr.coeffs, float)
            with self.packet_lock:
                self.camera_intrinsics={'K':K,'D':D,'serial':serial,'depth_scale':scale}
            self.cam_active=True
            while not stop_event.is_set():
                try:
                    frames=pipeline.wait_for_frames(timeout_ms=1200)
                    frames=align.process(frames)
                    rgb=frames.get_color_frame()
                    dep=frames.get_depth_frame()
                    if not rgb:continue
                    im=np.asanyarray(rgb.get_data()).copy()
                    z=np.asanyarray(dep.get_data()).astype(np.float32)*float(scale) if dep else None
                    with self.ur_lock: ur=None if self.ur_state is None else self.ur_state.copy()
                    pkt=ImagePacket(color=im,depth_m=z,K=K,D=D,cam_serial=serial,
                                    ts_monotonic=time.monotonic(),ur=ur,source='camera')
                    with self.packet_lock:self.packet=pkt
                except RuntimeError:
                    if stop_event.is_set():break
                    continue
        except Exception as ex:
            self.thread_error=f'RealSense启动/读取错误: {ex}'
        finally:
            if pipeline is not None:
                try:pipeline.stop()
                except Exception:pass
            self.cam_active=False

    def connect_ur(self):
        if rtde_receive is None:
            messagebox.showerror('缺少依赖','请先 pip install ur-rtde')
            return
        if self.ur_thread and self.ur_thread.is_alive():return
        self.stop_ur=threading.Event()
        self.ur_thread=threading.Thread(target=self.ur_worker,args=(self.robot_ip.get().strip(),self.stop_ur),daemon=True)
        self.ur_thread.start()
        self.log('正在连接UR5e RTDEReceive，仅读取状态，不下发运动命令')

    def disconnect_ur(self):
        self.stop_ur.set()
        self.ur_active=False
        with self.ur_lock:self.ur_state=None
        self.invalidate()

    def ur_worker(self,ip,stop_event):
        conn=None
        try:
            conn=rtde_receive.RTDEReceiveInterface(ip,50.0)
            if not conn.isConnected():raise RuntimeError('RTDE未连接')
            self.ur_active=True
            while not stop_event.is_set():
                pose=np.asarray(conn.getActualTCPPose(),float)
                qd=np.asarray(conn.getActualQd(),float)
                tcp_speed=np.asarray(conn.getActualTCPSpeed(),float)
                if pose.shape!=(6,) or qd.shape!=(6,) or tcp_speed.shape!=(6,):
                    raise RuntimeError('UR RTDE返回的数据长度有误')
                if not np.isfinite(pose).all():
                    raise RuntimeError('UR RTDE返回了无效位姿')
                stationary=(np.max(np.abs(qd))<MOVING_QD_MAX and
                            np.linalg.norm(tcp_speed[:3])<MOVING_TCP_MAX)
                state={'tcp':pose.copy(),'T_base_tcp':ur_pose_to_T(pose),
                       'qd':qd.copy(),'stationary':bool(stationary),'ts':time.monotonic()}
                with self.ur_lock:self.ur_state=state
                time.sleep(0.09)
        except Exception as ex:
            self.thread_ur_error=f'UR连接/状态读取失败: {ex}'
        finally:
            self.ur_active=False
            with self.ur_lock:self.ur_state=None
            if conn is not None:
                try:conn.disconnect()
                except Exception:pass

    def open_image(self):
        path=filedialog.askopenfilename(title='打开RGB图片',filetypes=[('Image','*.png *.jpg *.jpeg *.bmp')])
        if path:self.load_photo(path)

    def load_photo(self,path):
        try:
            im=cv2.imdecode(np.fromfile(str(path),dtype=np.uint8),cv2.IMREAD_COLOR)
            if im is None:raise ValueError('图片无法读取')
            self.disconnect_cam()
            # 单张图片无内参时只提供像素级四边形预览，不伪造三维位姿
            with self.packet_lock:
                self.packet=ImagePacket(im,None,None,None,'image',time.monotonic(),None,'photo')
            self.clear_selection()
            self.log('已打开图片；无同分辨率真实内参时只进行二维检测，不输出伪造的3D坐标。')
        except Exception as ex:messagebox.showerror('图片读取失败',str(ex))

    def load_handeye(self):
        path=filedialog.askopenfilename(title='选择眼在手上标定结果 T_tcp_cam.npy',filetypes=[('Matrix','*.npy *.txt'),('All files','*.*')])
        if not path:return
        try:
            if Path(path).suffix.lower()=='.npy':T=np.load(path,allow_pickle=False)
            else:T=np.loadtxt(path)
            self.T_tcp_cam=rigid_matrix_check(T)
            self.T_tcp_cam_path=str(path)
            self.invalidate()
            self.log('加载眼在手上矩阵成功：T_tcp_cam (相机彩色光学坐标 -> 当前TCP)。')
        except Exception as ex:messagebox.showerror('矩阵加载失败',str(ex))

    def invalidate(self):
        self.last_detect_key=None

    def set_roi_mode(self):
        self.select_mode='roi'
        self.manual_quad=None
        self.manual_quad_mono=None
        self.manual_points=[]
        self.log('请在左侧画面按住左键框选正方形顶面周围区域；目标移动出ROI后需要重选。')
        self.invalidate()

    def set_corners_mode(self):
        self.select_mode='corners'
        self.manual_quad=None
        self.manual_quad_mono=None
        self.manual_points=[]
        self.log('依次点击顶部正方形四个角（顺周界点即可）。注意物体移动后手点结果必须重新选取。')
        self.invalidate()

    def clear_selection(self):
        self.roi=None;self.manual_quad=None;self.manual_quad_mono=None;self.manual_points=[];self.select_mode=None
        self.invalidate()
        self.log('已清除ROI/手工角点；请重新选择。')

    def screen_to_img(self,event):
        if self.draw_mapping is None:return None
        s,x0,y0,ww,hh=self.draw_mapping
        x=(event.x-x0)/s;y=(event.y-y0)/s
        if x<0 or x>=ww or y<0 or y>=hh:return None
        return np.array([x,y],float)

    def mouse_down(self,event):
        p=self.screen_to_img(event)
        if p is None:return
        if self.select_mode=='roi':self.start_xy=p
        elif self.select_mode=='corners':
            self.manual_points.append(p)
            if len(self.manual_points)==4:
                try:self.manual_quad=canonical_corners(self.manual_points)
                except Exception as ex:self.log(f'手点角点无效：{ex}')
                self.manual_points=[]
                self.select_mode=None
                self.invalidate()
                if self.manual_quad is not None:
                    self.manual_quad_mono=time.monotonic()
                    self.log('手点四角完成；请在5秒内记录或保存。移动工件后必须重新手点。')

    def mouse_move(self,event):
        pass

    def mouse_up(self,event):
        if self.select_mode!='roi' or self.start_xy is None:return
        p=self.screen_to_img(event)
        if p is None:return
        a=self.start_xy;b=p
        x1,y1=np.minimum(a,b).astype(int)
        x2,y2=np.maximum(a,b).astype(int)
        self.start_xy=None
        if x2-x1<45 or y2-y1<45:
            self.log('ROI过小，请重选。');return
        self.roi=(x1,y1,x2,y2)
        self.select_mode=None
        self.invalidate()
        self.log(f'已选ROI: {self.roi}，开始在ROI内自动提取四边形。')

    def current_pose(self,pkt,quad):
        result={'source':pkt.source,'capture_time_iso':datetime.datetime.now().isoformat(timespec='milliseconds'),
                'camera_serial':pkt.cam_serial,'corners_px':quad.tolist(), 'roi_px':self.roi,
                'edge_mm':float(self.edge_mm.get()),'top_plate_gap_mm':float(self.gap_mm.get())}
        if not 10 <= result['edge_mm'] <= 500 or not 0 <= result['top_plate_gap_mm'] <= 500:
            raise ValueError('顶面边长或高度差输入无效；边长范围10~500mm，高差范围0~500mm')
        K=pkt.K;D=pkt.D
        T_base_cam=None
        robot_ok=(pkt.ur is not None and pkt.ur.get('stationary') and
                  abs(pkt.ts_monotonic-pkt.ur['ts'])<MAX_ROBOT_SAMPLE_AGE and pkt.source=='camera')
        if robot_ok and self.T_tcp_cam is not None:
            T_base_cam=pkt.ur['T_base_tcp'] @ self.T_tcp_cam
        if K is not None:
            pnp=pnp_square(quad,K,D,edge_m=float(self.edge_mm.get())/1000.,T_base_cam=T_base_cam)
            result['rmse_px']=pnp['rmse_px']
            result['T_cam_top']=pnp['T_cam_top'].tolist()
            result['camera_center_m']=pnp['tvec'].tolist()
            result['camera_yaw_xy_deg']=normal_angle_90(math.degrees(math.atan2(pnp['T_cam_top'][1,0],pnp['T_cam_top'][0,0])))
            if T_base_cam is not None:
                pose=base_pose_from_square(T_base_cam,pnp)
                result['frame']='base'
                result['center_m']=pose['center_m'].tolist()
                result['yaw_deg']=pose['yaw_deg']
                result['T_base_top']=pose['T_base_top'].tolist()
                result['T_base_tcp']=pkt.ur['T_base_tcp'].tolist()
                result['T_base_cam']=T_base_cam.tolist()
                result['stationary']=True
            else:
                result['frame']='camera'
                result['center_m']=pnp['tvec'].tolist()
                result['yaw_deg']=result['camera_yaw_xy_deg']
                result['stationary']=bool(pkt.ur is None or robot_ok)
        else:
            result['frame']='pixels_only'
            result['center_px']=quad.mean(axis=0).tolist()
        result['depth_check']=depth_stats(pkt.depth_m,quad,K,D,T_base_cam,
                                          target_gap_m=float(self.gap_mm.get())/1000.)
        return result

    def calculate_delta(self,result):
        ref=self.reference
        if ref is None:return None
        if result.get('frame')!=ref.get('frame'):
            return None
        if ref.get('handeye_matrix') is not None and self.T_tcp_cam is not None:
            if not np.allclose(ref['handeye_matrix'],self.T_tcp_cam,atol=1e-7):
                return None
        if result.get('camera_serial')!=ref.get('camera_serial'):
            return None
        if result.get('frame')=='base':
            p0=np.asarray(ref['center_m']);p1=np.asarray(result['center_m'])
            return {'delta_xyz_mm':((p1-p0)*1000).tolist(),
                    'delta_yaw_deg':equiv_delta_deg(result['yaw_deg'],ref['yaw_deg']),
                    'frame':'base'}
        if result.get('frame')=='camera':
            # 不满足两次相机固定位姿时，仅相机坐标差不是真实物体运动
            return None
        return None

    def latest_fresh(self):
        if self.latest_detection is None:
            raise ValueError('当前没有可靠的顶面检测结果')
        result,pkt,vis=self.latest_detection
        if pkt.source=='camera' and time.monotonic()-pkt.ts_monotonic > 0.8:
            raise ValueError('相机画面已经过期，请检查设备连接')
        if result.get('frame')!='base':
            raise ValueError('记录参考/当前前，请加载 T_tcp_cam 并连接静止的UR5e，获得Base坐标位姿。')
        if not pkt.ur or not pkt.ur.get('stationary'):
            raise ValueError('UR5e尚未静止，请停止后再记录')
        return result,pkt,vis

    def set_reference(self):
        try:
            result,pkt,vis=self.latest_fresh()
            ref={k:result[k] for k in ['frame','camera_serial','center_m','yaw_deg','T_base_top','capture_time_iso','edge_mm']}
            ref['handeye_matrix']=self.T_tcp_cam.tolist()
            self.reference=make_jsonable(ref)
            name=datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            out=SAVE_ROOT/('reference_'+name+'.json')
            export_json(out,{'reference':self.reference,'matrix_path':self.T_tcp_cam_path,'description':'顶面中心 in UR Base, yaw modulo90'})
            self.log(f'已记录示教参考位姿，并保存到 {out}')
            self.update_ref_label()
        except Exception as ex:messagebox.showwarning('不能记录参考',str(ex))

    def save_current(self):
        try:
            result,pkt,vis=self.latest_fresh()
            now=datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            dest=SAVE_ROOT/now
            dest.mkdir(parents=True,exist_ok=True)
            bgr=vis.copy()
            rgb_name=dest/'rgb_overlay.png'
            if not cv2.imwrite(str(rgb_name),bgr):raise IOError('写入RGB图片失败')
            raw_name=dest/'rgb_raw.png'
            cv2.imwrite(str(raw_name),pkt.color)
            if pkt.depth_m is not None:
                cv2.imwrite(str(dest/'depth_aligned_z16.png'),np.clip(pkt.depth_m/0.001,0,65535).astype(np.uint16))
            d={'current':result,'reference':self.reference,'delta':self.calculate_delta(result),
               'handeye_matrix_path':self.T_tcp_cam_path,
               'K':pkt.K,'D':pkt.D,'depth_saved_unit_m':0.001,
               'notes':'顶面几何中心; Yaw 90°等价; 只读定位，不自动执行轨迹'}
            export_json(dest/'pose.json',make_jsonable(d))
            self.saved_records.append(str(dest))
            self.log(f'检测结果已保存: {dest}')
        except Exception as ex:messagebox.showwarning('保存失败',str(ex))

    def load_reference(self):
        path=filedialog.askopenfilename(title='选择参考位姿JSON',filetypes=[('JSON','*.json')])
        if not path:return
        try:
            with open(path,'r',encoding='utf-8') as f:d=json.load(f)
            ref=d.get('reference',d)
            if ref.get('frame')!='base' or len(ref.get('center_m',[]))!=3 or 'yaw_deg' not in ref:
                raise ValueError('文件不是有效的 Base 坐标参考位姿')
            if not np.isclose(float(ref.get('edge_mm',0)),float(self.edge_mm.get()),atol=0.01):
                raise ValueError('参考位姿的方块边长与当前设置不一致')
            if ref.get('handeye_matrix') is not None and self.T_tcp_cam is not None:
                if not np.allclose(ref['handeye_matrix'],self.T_tcp_cam,atol=1e-7):
                    raise ValueError('当前加载的T_tcp_cam与保存参考时不同，请使用同一标定矩阵重新记录参考')
            self.reference=ref
            self.update_ref_label()
            self.log(f'已加载参考位姿: {path}')
        except Exception as ex:messagebox.showerror('读取参考失败',str(ex))

    def clear_reference(self):
        self.reference=None
        self.update_ref_label()
        self.delta_var.set('ΔX / ΔY / ΔYaw: —')

    def update_ref_label(self):
        if self.reference is None:
            self.ref_var.set('尚未记录示教参考位姿')
        else:
            p=np.array(self.reference['center_m'])*1000
            self.ref_var.set(f'参考(Base): X={p[0]:+.1f} Y={p[1]:+.1f} Z={p[2]:+.1f} mm, Yaw={self.reference["yaw_deg"]:+.2f}°')

    def tick(self):
        if self.closing:return
        if getattr(self,'thread_error',None):
            self.log(self.thread_error);self.thread_error=None
        if getattr(self,'thread_ur_error',None):
            self.log(self.thread_ur_error);self.thread_ur_error=None
        with self.packet_lock:pkt=self.packet
        if pkt is not None:
            try:
                im=pkt.color
                vis=im.copy()
                if self.roi is not None:
                    cv2.rectangle(vis,(self.roi[0],self.roi[1]),(self.roi[2],self.roi[3]),(250,180,40),2)
                if self.manual_quad is not None and self.manual_quad_mono is not None:
                    if time.monotonic()-self.manual_quad_mono>5.0:
                        self.manual_quad=None
                        self.manual_quad_mono=None
                        self.log('手点四角已过期(5秒)；避免将过时角点当作运动后的位置，请重新选择。')
                quad=self.manual_quad
                if quad is None and self.auto_mode.get() and self.roi is not None:
                    quad=detect_quad_lsd(im,self.roi)
                if quad is not None:
                    quad=canonical_corners(quad)
                    cv2.polylines(vis,[quad.astype(np.int32)],True,(60,240,80),2,cv2.LINE_AA)
                    cen=quad.mean(axis=0)
                    cv2.circle(vis,tuple(np.round(cen).astype(int)),5,(40,40,255),-1)
                    for i,p in enumerate(quad):
                        point=tuple(np.round(p).astype(int))
                        cv2.circle(vis,point,4,(30,210,240),-1)
                        cv2.putText(vis,str(i+1),(point[0]+4,point[1]-4),cv2.FONT_HERSHEY_SIMPLEX,.52,(10,30,250),2)
                    try:
                        res=self.current_pose(pkt,quad)
                        res['detection_mode']='manual' if self.manual_quad is not None else 'automatic_lsd'
                        self.latest_detection=(res,pkt,vis.copy())
                        if res['frame']=='base':
                            p=np.array(res['center_m'])*1000
                            self.pose_var.set(f"Base 顶面中心 X={p[0]:+.2f} Y={p[1]:+.2f} Z={p[2]:+.2f} mm\n等价 Yaw={res['yaw_deg']:+.2f}°   PnP RMSE={res['rmse_px']:.2f} px")
                        elif res['frame']=='camera':
                            p=np.array(res['center_m'])*1000
                            self.pose_var.set(f"仅相机坐标 X={p[0]:+.2f} Y={p[1]:+.2f} Z={p[2]:+.2f} mm\n相机XY边方向={res['yaw_deg']:+.2f}°（不是Base桌面Yaw）\n连接UR、加载手眼矩阵后才可进行Base补偿")
                        else:
                            self.pose_var.set('已检测四边形像素位置；当前没有可用的真实RGB相机内参，不能计算三维位姿。')
                        dep=res['depth_check']
                        if dep['valid_frac'] is None:self.depth_var.set('深度：无有效帧；允许仅RGB-PnP求解')
                        else:
                            gapstr=(f" | 圆盘高差≈{dep['gap_measured_mm']:.1f}mm" if dep.get('gap_measured_mm') is not None else '')
                            medstr=(f"{dep['median_m']*1000:.1f}mm" if dep.get('median_m') is not None else '无')
                            self.depth_var.set(f"顶面有效深度 {100*dep['valid_frac']:.1f}% | 深度中位数 {medstr}{gapstr} | {dep['depth_note']}")
                        delta=self.calculate_delta(res)
                        if delta is None:
                            self.delta_var.set('ΔX / ΔY / ΔYaw：尚无同坐标系参考；或相机/UR状态不满足条件')
                        else:
                            d=delta['delta_xyz_mm']
                            self.delta_var.set(f"ΔX={d[0]:+.2f} mm | ΔY={d[1]:+.2f} mm | ΔZ={d[2]:+.2f} mm\nΔYaw(模90°)={delta['delta_yaw_deg']:+.2f}°")
                    except Exception as ex:
                        self.latest_detection=None
                        self.pose_var.set(f'四边形已找到，但三维求解无效：{str(ex)[:110]}')
                        self.delta_var.set('相对位姿：—')
                else:
                    self.latest_detection=None
                    self.pose_var.set('未识别到四条可靠边：请重新框选ROI，或手点四角。')
                    self.delta_var.set('相对位姿：—')
                if self.manual_points:
                    for p in self.manual_points:cv2.circle(vis,tuple(p.astype(int)),4,(0,0,255),-1)
                self.draw_image(vis)
                urtag='UR已连接' if self.ur_active else 'UR未连接'
                mtag='手眼矩阵已加载' if self.T_tcp_cam is not None else '未加载T_tcp_cam'
                self.status_var.set(f'{urtag} | {mtag} | {"相机运行中" if self.cam_active else "离线/相机未运行"}')
            except Exception as ex:
                self.pose_var.set(f'图像处理失败: {ex}')
        self.root.after(140,self.tick)

    def draw_image(self,im):
        if Image is None:
            self.canvas.delete('all')
            self.canvas.create_text(20,30,anchor='nw',fill='white',text='请 pip install pillow 后查看图像')
            return
        h,w=im.shape[:2]
        cw=max(100,self.canvas.winfo_width());ch=max(100,self.canvas.winfo_height())
        scale=min(cw/w,ch/h)
        sw,sh=round(w*scale),round(h*scale)
        x0=(cw-sw)//2;y0=(ch-sh)//2
        resized=cv2.resize(im,(sw,sh),interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
        self.preview_image=ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)))
        self.canvas.delete('all')
        self.canvas.create_image(x0,y0,anchor='nw',image=self.preview_image)
        self.draw_mapping=(scale,x0,y0,w,h)

    def close(self):
        self.closing=True
        self.stop_cam.set();self.stop_ur.set()
        self.root.destroy()


def offline_self_test():
    """不依赖真实设备：PnP矩阵、等价角、三维位置变换和保存测试。"""
    assert equiv_delta_deg(-43,40)==7
    assert abs(normal_angle_90(135)+45) < 1e-8
    K=np.array([[620.,0,320.],[0,621.,240.],[0,0,1.]])
    D=np.zeros(5)
    obj=square_object_points()
    true_R=cv2.Rodrigues(np.array([0.10,-0.06,0.31]))[0]
    true_t=np.array([0.008,-0.006,0.48])
    corners=cv2.projectPoints(obj,cv2.Rodrigues(true_R)[0],true_t,K,D)[0].reshape(4,2)
    corners=canonical_corners(corners)
    T_base_cam=homogeneous(np.diag([1.,-1.,-1.]), [0.3,-0.1,0.65])
    pnp=pnp_square(corners,K,D,T_base_cam=T_base_cam)
    p=base_pose_from_square(T_base_cam,pnp)
    assert pnp['rmse_px'] < 1e-4,(pnp['rmse_px'],pnp)
    assert np.linalg.norm(pnp['tvec']-true_t) < 0.003,pnp['tvec']
    assert np.all(np.isfinite(p['center_m']))
    assert np.allclose(rigid_matrix_check(np.eye(4)),np.eye(4))
    for yaw_deg in (-60., -45., -30., 0., 15., 45., 75.):
        Rz=cv2.Rodrigues(np.array([0.,0.,math.radians(yaw_deg)]))[0]
        pix=cv2.projectPoints(obj,cv2.Rodrigues(Rz)[0],np.array([0.,0.,0.45]),K,D)[0].reshape(-1,2)
        q=pnp_square(pix,K,D,T_base_cam=T_base_cam)
        expect=normal_angle_90(-yaw_deg)
        measured=base_pose_from_square(T_base_cam,q)['yaw_deg']
        assert abs(equiv_delta_deg(measured,expect)) < 0.001,(yaw_deg,measured,expect)
    print('PASS: 四角点/80mm PnP、含正视退化时的求解、矩阵变换、90°等价角')


def main():
    ap=argparse.ArgumentParser(description=APP_TITLE)
    ap.add_argument('--self-test',action='store_true')
    ap.add_argument('--image',type=str,default=None,help='加载一张图片进行离线二维边缘检测调试')
    args=ap.parse_args()
    if args.self_test:
        offline_self_test();return
    root=tk.Tk()
    SquarePoseApp(root,args.image)
    root.mainloop()

if __name__=='__main__': main()
