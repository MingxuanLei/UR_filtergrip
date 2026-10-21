# -*- coding: utf-8 -*-
"""
UR5e 视觉参考绑定 / XYZ+Yaw 位姿补偿 / 自主保存 GUI
=========================
文件夹结构（默认）：
    ur_pose_stage.py
    teach_records/
        teach_xxx.npz
        shift_outputs/            # 自动生成

运行：python ur_pose_stage.py
依赖：pip install numpy scipy matplotlib
GUI：Python 自带 tkinter（部分 Linux 环境需额外安装 python3-tk）

仅进行离线运动学计算，不连接机器人、不下发运动指令。
使用 UR5e 名义 DH 模型的 tool0（法兰）坐标，未包含实际机器人标定及夹爪 TCP。
旋转轴为基坐标系 Z 轴，仅适用于物体绕桌面法线自转；自动旋转中心基于 tool0 XY 的近似估计，需人工校验。
新增分阶段模式：按夹爪闭合—张开事件使补偿权重0→1→0，张开前回归原示教放置轨迹。
自动抬升条件仅检查名义tool0的Base-Z位移，不等于物体已离桌或碰撞安全。
未进行实际碰撞、夹持力、关节加速度及机器人安全区域验证；不可直接用于实机回放。
"""
from __future__ import annotations

import os
import json
import hashlib
import shutil
import datetime as dt
import tempfile
import queue
import re
import sys
import threading
import traceback
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import least_squares
from scipy.ndimage import median_filter
from scipy.spatial.transform import Rotation

# 标准 DH，UR5e 名义参数（非出厂专属运动学校准值）
DH_A = np.array([0.0, -0.4250, -0.3922, 0.0, 0.0, 0.0])
DH_D = np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996])
DH_ALPHA = np.array([np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0])
JOINT_LIMIT = 2 * np.pi  # 仅用于名义数值合法性检查，非现场安全限位

# Base 坐标系下的 XYZ 位移补偿范围（单位：cm；手动/视觉共用）
# 这是离线算法的输入门限，并非 UR5e 实际工作空间或碰撞安全边界。
XYZ_LIMITS_CM = np.array([12.0, 12.0, 10.0], dtype=float)


def validate_xyz_offset_cm(xyz_cm: np.ndarray) -> np.ndarray:
    """对 XYZ 三方向分别检查补偿幅度，返回以 cm 为单位的向量。"""
    values = np.asarray(xyz_cm, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError('XYZ 补偿必须是三个有限数值（单位 cm）')
    for axis, val, limit in zip('XYZ', values, XYZ_LIMITS_CM):
        if abs(val) > limit + 1e-9:
            raise ValueError(
                f'{axis} 方向补偿 {val:+.4f} cm 超出允许范围 '
                f'[-{limit:g}, +{limit:g}] cm；'
                'X/Y 各 ±12 cm，Z ±10 cm'
            )
    return values.copy()



def dh_matrix(theta: float, a: float, d: float, alpha: float) -> np.ndarray:
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st * ca, st * sa, a * ct],
        [st, ct * ca, -ct * sa, a * st],
        [0.0, sa, ca, d],
        [0.0, 0.0, 0.0, 1.0],
    ], dtype=float)


def fk_jacobian(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """base -> tool0 的变换矩阵及几何雅可比（工具0坐标，非夹爪 TCP）。"""
    T = np.eye(4)
    axes = np.empty((6, 3), dtype=float)
    origins = np.empty((6, 3), dtype=float)
    for j in range(6):
        axes[j] = T[:3, 2]
        origins[j] = T[:3, 3]
        T = T @ dh_matrix(float(q[j]), DH_A[j], DH_D[j], DH_ALPHA[j])
    J = np.empty((6, 6), dtype=float)
    for j in range(6):
        J[:3, j] = np.cross(axes[j], T[:3, 3] - origins[j])
        J[3:, j] = axes[j]
    return T, J


def pose_error(target: np.ndarray, actual: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pos = target[:3, 3] - actual[:3, 3]
    rot = Rotation.from_matrix(target[:3, :3] @ actual[:3, :3].T).as_rotvec()
    return pos, rot


def ik_near(target: np.ndarray, seed: np.ndarray,
            position_tolerance: float = 5e-5,
            rotation_tolerance: float = 5e-4) -> np.ndarray:
    """使用近邻初值的阻尼雅可比 IK；失败时启用 SciPy 优化求解。"""
    q = np.array(seed, dtype=float, copy=True)
    damping = 1e-3
    for _ in range(45):
        current, J = fk_jacobian(q)
        ep, er = pose_error(target, current)
        if np.linalg.norm(ep) < position_tolerance and np.linalg.norm(er) < rotation_tolerance:
            return q
        delta = np.linalg.solve(J.T @ J + damping**2 * np.eye(6), J.T @ np.r_[ep, er])
        mx = float(np.max(np.abs(delta)))
        if mx > 0.18:
            delta *= 0.18 / mx
        q += delta
        if not np.isfinite(q).all() or np.any(np.abs(q) > 2 * JOINT_LIMIT):
            break

    def residual(q_test):
        current, _ = fk_jacobian(q_test)
        ep, er = pose_error(target, current)
        return np.r_[ep, er]

    result = least_squares(
        residual,
        x0=np.clip(seed, -JOINT_LIMIT + 1e-8, JOINT_LIMIT - 1e-8),
        bounds=(-JOINT_LIMIT * np.ones(6), JOINT_LIMIT * np.ones(6)),
        method='trf', max_nfev=100,
        xtol=1e-11, ftol=1e-11, gtol=1e-11,
    )
    ep, er = pose_error(target, fk_jacobian(result.x)[0])
    if np.linalg.norm(ep) >= position_tolerance or np.linalg.norm(er) >= rotation_tolerance:
        raise RuntimeError(
            f'IK 未收敛：位置误差 {np.linalg.norm(ep)*1000:.3f} mm，'
            f'姿态误差 {np.linalg.norm(er):.5f} rad；该偏移可能不可达'
        )
    return result.x


def smoothstep5(s: float) -> float:
    """五次平滑曲线，s=0/1 时值为0/1，且端点的一、二阶导数为0。"""
    s = float(np.clip(s, 0, 1))
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def detect_closing_events(data: dict[str, np.ndarray],
                          source: str = 'auto') -> list[dict]:
    """识别有持续下降幅度的夹爪闭合命令，而非单帧抖动。

    gripper_*_open 的语义：数值越大越张开，数值下降代表闭合。
    事件时间使用与 UR 关节轨迹同一行号及控制频率，避免重复时间戳干扰。
    目标信号有效时优先使用 target，否则尝试实际反馈。
    这只能识别闭合动作，不能证明已经接触或成功抓住物体。
    """
    if source not in ('auto', 'target', 'actual'):
        raise ValueError('夹爪事件数据源必须为 auto、target 或 actual')
    hz = frequency_of(data)
    options = {'target': 'gripper_target_open', 'actual': 'gripper_open'}
    candidates = (('target', 'actual') if source == 'auto' else (source,))
    for label in candidates:
        key = options[label]
        if key not in data:
            continue
        raw = np.asarray(data[key], dtype=float)
        if raw.ndim != 1 or raw.size < 3 or not np.isfinite(raw).all():
            continue
        # 120 ms 中值滤波消除单个数据点的扰动，保留慢速闭合趋势
        window = max(3, int(round(hz * 0.12)) | 1)
        smooth = median_filter(raw, size=window, mode='nearest')
        if float(np.max(smooth)-np.min(smooth)) < 0.12:
            continue
        onset_drop = 0.012      # 相对前一开放峰值下降 1.2% 才可能视为开始闭合
        confirm_drop = 0.12     # 随后最多 4 s 内累计闭合 >= 12% 才确认
        release_rise = 0.15    # 回升 >= 15% 后，才重新允许检测下一次闭合
        look_ahead = max(1, int(round(hz * 4.0)))
        events = []
        armed = True
        peak_value = float(smooth[0])
        trough = peak_value
        last_event_index = -look_ahead
        i = 1
        while i < len(smooth):
            v = float(smooth[i])
            if armed:
                if v > peak_value:
                    peak_value = v
                if peak_value - v >= onset_drop:
                    j = min(len(smooth), i + look_ahead)
                    low = float(np.min(smooth[i:j]))
                    total_drop = peak_value-low
                    if total_drop >= confirm_drop and i-last_event_index >= max(1, int(hz * 0.5)):
                        events.append({
                            'index': i,
                            'time_s': float(i / hz),
                            'level_before': peak_value,
                            'level_min_next4s': low,
                            'confirmed_drop': total_drop,
                            'source': key,
                        })
                        armed = False
                        trough = v
                        last_event_index = i
            else:
                trough = min(trough, v)
                if v-trough >= release_rise:
                    armed = True
                    peak_value = v
            i += 1
        if events or source != 'auto':
            return events
    return []



def detect_gripper_cycles(data: dict[str, np.ndarray],
                          closing_events: list[dict] | None = None) -> list[dict]:
    """识别一次夹爪闭合—保持—张开循环（只识别开度变化，不证明夹持成功）。

    闭合开始沿用原有事件检测；闭合稳定优先依据真实夹爪反馈。
    张开开始优先依据目标命令（比反馈早，便于确保回退在松爪之前结束）。
    所有索引均对应原始 NPZ 行号；无法可靠判定则不输出该周期。
    """
    hz = frequency_of(data)
    events = detect_closing_events(data) if closing_events is None else closing_events
    signals = {}
    window = max(3, int(round(hz * 0.12)) | 1)
    for label, key in (('target', 'gripper_target_open'), ('actual', 'gripper_open')):
        if key not in data:
            continue
        arr = np.asarray(data[key], dtype=float)
        if arr.ndim == 1 and arr.size >= 3 and np.isfinite(arr).all():
            signals[label] = median_filter(arr, size=window, mode='nearest')
    if not signals or not events:
        return []
    n = len(next(iter(signals.values())))
    span = max(3, int(round(0.22 * hz)))
    look_ahead = max(2, int(round(4.0 * hz)))
    cycles = []

    def opening_onset(arr, begin, stop):
        """先检测低谷后的持续张开，允许部分张开但拒绝瞬时抖动。"""
        trough = float(arr[min(begin, n - 1)])
        for k in range(begin + 1, stop):
            val = float(arr[k])
            trough = min(trough, val)
            if val - trough >= 0.018:
                right = min(stop, k + look_ahead)
                if right > k and float(np.max(arr[k:right]) - trough) >= 0.15:
                    return k
        return None

    for j, ev in enumerate(events):
        close_idx = int(ev['index'])
        next_close = int(events[j + 1]['index']) if j + 1 < len(events) else n
        boundary = min(next_close, n)
        # 目标信号的张开起点优先，避免以存在执行滞后的实际反馈安排放置时刻。
        openings = []
        for name in ('target', 'actual'):
            if name in signals:
                oi = opening_onset(signals[name], close_idx, boundary)
                if oi is not None:
                    openings.append((oi, name))
        if not openings:
            continue
        open_idx, open_source = min(openings, key=lambda x: x[0])
        if open_idx <= close_idx + span + 2:
            continue

        # 夹爪真实开度可能停留在非零值（夹着工件），不要求数值等于0。
        closed_idx, closed_source = None, None
        for name in ('actual', 'target'):
            if name not in signals:
                continue
            arr = signals[name]
            initial = float(np.max(arr[max(0, close_idx - span):close_idx + 1]))
            low = float(np.min(arr[close_idx:open_idx]))
            if initial - low < 0.12:
                continue
            for k in range(close_idx + 1, open_idx - span + 1):
                segment = arr[k:k + span]
                if (float(np.ptp(segment)) <= 0.035 and
                        float(np.median(segment)) <= low + 0.035 and
                        initial - float(np.median(segment)) >= 0.12):
                    closed_idx, closed_source = k + span - 1, name
                    break
            if closed_idx is not None:
                break
        if closed_idx is None or closed_idx >= open_idx:
            continue

        # 完全张开同样是“张开动作稳定”，未必是归一化读数1.0。
        opened_idx, opened_source = None, None
        for name in ('actual', 'target'):
            if name not in signals:
                continue
            arr = signals[name]
            if boundary - open_idx < span + 1:
                continue
            low = float(np.min(arr[close_idx:open_idx + 1]))
            high = float(np.max(arr[open_idx:boundary]))
            if high - low < 0.15:
                continue
            for k in range(open_idx + 1, boundary - span + 1):
                segment = arr[k:k + span]
                if (float(np.ptp(segment)) <= 0.035 and
                        float(np.median(segment)) >= high - 0.035 and
                        float(np.median(segment)) - low >= 0.15):
                    opened_idx, opened_source = k + span - 1, name
                    break
            if opened_idx is not None:
                break

        cycles.append({
            'close_index': close_idx, 'close_time_s': close_idx / hz,
            'close_source': ev['source'],
            'closed_index': int(closed_idx), 'closed_time_s': closed_idx / hz,
            'closed_source': closed_source,
            'open_index': int(open_idx), 'open_time_s': open_idx / hz,
            'open_source': open_source,
            'opened_index': int(opened_idx) if opened_idx is not None else -1,
            'opened_time_s': opened_idx / hz if opened_idx is not None else float('nan'),
            'opened_source': opened_source or 'undetected',
        })
    return cycles


def make_staged_profile(data: dict[str, np.ndarray], cycle: dict,
                        advance_seconds: float = 1.0,
                        return_end_advance_s: float = 0.5,
                        return_start_mode: str = 'auto',
                        lift_mm: float = 30.0,
                        lift_settle_seconds: float = 0.2,
                        return_start_manual_s: float | None = None
                        ) -> tuple[np.ndarray, dict]:
    """由夹爪周期建立 0→1→0 权重曲线；未确认安全抬升时不自行猜测回退时间。

    auto 仅检查名义 tool0 在 Base Z 方向从闭合稳定时刻抬升了指定毫米数；
    这不是物体实际离桌/碰撞安全的证明。manual 要求用户确认无障碍搬运。
    """
    n = int(np.asarray(data['ur_q']).shape[0])
    hz = frequency_of(data)
    if return_start_mode not in ('auto', 'manual'):
        raise ValueError('第二阶段开始方式必须是 auto 或 manual')
    for name, value, lower, upper in (
        ('抓取提前完成时间', advance_seconds, 0.0, 10.0),
        ('放置前提前完成时间', return_end_advance_s, 0.0, 10.0),
        ('最小抬升高度', lift_mm, 5.0, 300.0),
        ('抬升后等待时间', lift_settle_seconds, 0.0, 5.0),
    ):
        if not np.isfinite(value) or not lower <= float(value) <= upper:
            raise ValueError(f'{name} 必须在 {lower:g}～{upper:g} 范围内')

    t_finish_pick = float(cycle['close_time_s']) - advance_seconds
    t_finish_return = float(cycle['open_time_s']) - return_end_advance_s
    if t_finish_pick < 0.25:
        raise ValueError('抓取补偿必须在轨迹开始至少0.25秒后完成，请减小抓取提前秒数')
    if t_finish_return <= float(cycle['closed_time_s']) + 0.5:
        raise ValueError('闭合稳定到张开前可用时间太短，不足以完成分阶段回退')

    lift_index = -1
    if return_start_mode == 'manual':
        if return_start_manual_s is None or not np.isfinite(return_start_manual_s):
            raise ValueError('手动回退模式必须填写有效的第二阶段开始时间')
        t_start_return = float(return_start_manual_s)
        reason = '人工指定：必须已确认工件离桌、无障碍且夹持稳定'
    else:
        q0 = np.asarray(data['ur_q'], dtype=float)
        closed_index = int(cycle['closed_index'])
        z_closed = float(fk_jacobian(q0[closed_index])[0][2, 3])
        target_z = z_closed + float(lift_mm) / 1000.0
        open_index = int(cycle['open_index'])
        for k in range(closed_index + 1, open_index):
            z = float(fk_jacobian(q0[k])[0][2, 3])
            if z >= target_z:
                lift_index = k
                break
        if lift_index < 0:
            raise ValueError(
                f'未检测到相对闭合稳定点抬升至少 {lift_mm:g} mm；'
                '为避免工件在桌面上横向移动，请检查示教或改为手动设置安全回退起点')
        t_start_return = lift_index / hz + float(lift_settle_seconds)
        reason = f'名义tool0抬升≥{lift_mm:g}mm后等待{lift_settle_seconds:g}s（仍需实机确认安全间隙）'

    if t_start_return < float(cycle['closed_time_s']):
        raise ValueError('回退起点早于夹爪闭合稳定时刻')
    if t_start_return <= t_finish_pick:
        raise ValueError('回退起点早于抓取阶段补偿完成时间')
    if t_finish_return - t_start_return < 0.5:
        raise ValueError(
            f'回退阶段只有 {t_finish_return-t_start_return:.3f}s（要求至少0.5s）；'
            '请提前开始回退、减小张开前提前量，或重新录制更长搬运轨迹')

    times = np.arange(n, dtype=float) / hz
    alpha = np.empty(n, dtype=float)
    # 先补偿；保持；搬运期间撤销补偿；松爪前始终恢复到原始轨迹。
    for i, t in enumerate(times):
        if t < t_finish_pick:
            alpha[i] = smoothstep5(t / t_finish_pick)
        elif t < t_start_return:
            alpha[i] = 1.0
        elif t < t_finish_return:
            alpha[i] = 1.0 - smoothstep5(
                (t - t_start_return) / (t_finish_return - t_start_return))
        else:
            alpha[i] = 0.0
    boundaries = {
        'pick_complete_s': t_finish_pick,
        'return_start_s': t_start_return,
        'return_end_s': t_finish_return,
        'lift_index': lift_index,
        'return_reason': reason,
        'cycle': cycle.copy(),
        'grasp_advance_s': float(advance_seconds),
        'release_advance_s': float(return_end_advance_s),
    }
    if not np.isclose(alpha[0], 0) or not np.all(np.isfinite(alpha)):
        raise RuntimeError('分阶段补偿比例无效')
    if float(alpha[int(cycle['close_index'])]) < 1.0 - 1e-10:
        raise RuntimeError('抓取前未完成全量补偿')
    if abs(float(alpha[int(cycle['open_index'])])) > 1e-10:
        raise RuntimeError('开始松爪时补偿未完全撤销')
    return alpha, boundaries


def compensation_alpha(i: int, n: int, hz: float, mode: str,
                       seconds: float, complete_time: float | None = None) -> float:
    """full: 全程渐变；early: 前N秒；instant: 全程平移；gripper: 闭合前完成补偿。"""
    if mode == 'full':
        return smoothstep5(i / (n - 1))
    if mode == 'early':
        return smoothstep5((i / hz) / seconds) if seconds > 0 else 1.0
    if mode == 'gripper':
        if complete_time is None or complete_time <= 0:
            raise ValueError('夹爪事件补偿缺少有效的完成时间')
        return smoothstep5((i / hz) / complete_time)
    if mode == 'instant':
        return 1.0
    raise ValueError(f'未知补偿模式: {mode}')


def load_record(path: Path) -> dict[str, np.ndarray]:
    """严格检验已有 NPZ 的关键字段，不允许 pickle 加载。"""
    with np.load(path, allow_pickle=False) as npz:
        keys = set(npz.files)
        required = {'ur_q', 'timestamp', 'gripper_open'}
        if missing := (required - keys):
            raise ValueError(f'NPZ 缺少字段：{", ".join(sorted(missing))}')
        data = {key: npz[key].copy() for key in npz.files}

    q = np.asarray(data['ur_q'], dtype=float)
    n = q.shape[0] if q.ndim == 2 else 0
    if q.ndim != 2 or q.shape[1] != 6 or n < 2:
        raise ValueError('ur_q 必须为 (N, 6)，N≥2')
    for name in ('timestamp', 'gripper_open', 'gripper_target_open'):
        if name in data:
            a = np.asarray(data[name])
            if a.ndim != 1 or len(a) != n or not np.isfinite(a).all():
                raise ValueError(f'{name} 必须为长度 N 的有限数字数组')
    if 'ur_target_q' in data:
        q_target = np.asarray(data['ur_target_q'])
        if q_target.shape != (n, 6) or not np.isfinite(q_target).all():
            raise ValueError('ur_target_q 必须为 (N, 6)，数值有限')
    if not np.isfinite(q).all():
        raise ValueError('ur_q 包含 NaN 或 Inf')
    if np.max(np.abs(q)) > JOINT_LIMIT + 1e-7:
        raise ValueError('原轨迹关节角超过 ±2π 名义范围，请人工核查单位或数据')
    if 'rtde_control_frequency_hz' in data:
        hz_data = np.asarray(data['rtde_control_frequency_hz']).ravel()
        if hz_data.size != 1 or not np.isfinite(hz_data[0]) or hz_data[0] <= 0:
            raise ValueError('rtde_control_frequency_hz 无效')
    return data


def frequency_of(data: dict[str, np.ndarray]) -> float:
    if 'rtde_control_frequency_hz' in data:
        return float(data['rtde_control_frequency_hz'].ravel()[0])
    # 只有无法取得控制频率时才从非零时间戳差估计；须人工复核
    dt = np.diff(np.asarray(data['timestamp'], dtype=float))
    valid = dt[dt > 0]
    if valid.size == 0:
        raise ValueError('无法确定采样频率：缺少 rtde_control_frequency_hz 且时间戳无有效差值')
    return 1.0 / float(np.median(valid))



def rotation_z(yaw_rad: float) -> np.ndarray:
    """基坐标系绕 +Z 的主动旋转（逆时针为正，右手定则）。"""
    c, s = np.cos(yaw_rad), np.sin(yaw_rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def compensated_pose(original: np.ndarray, xyz_m: np.ndarray, yaw_deg: float,
                     center_xy_m: np.ndarray, alpha: float) -> np.ndarray:
    """绕物体旧中心旋转，并叠加基坐标系平移。所有量由同一 alpha 渐变。

    p' = c + alpha*xyz + Rz(alpha*yaw)*(p-c)
    R' = Rz(alpha*yaw)*R
    c 是原始物体中心在机器人基坐标系的 XY（Z 不影响绕Z的旋转）。
    """
    c = np.array([center_xy_m[0], center_xy_m[1], 0.0], dtype=float)
    rot = rotation_z(np.deg2rad(float(yaw_deg)) * alpha)
    target = original.copy()
    target[:3, 3] = c + alpha * xyz_m + rot @ (original[:3, 3] - c)
    target[:3, :3] = rot @ original[:3, :3]
    return target


def pivot_from_grasp(record: dict[str, np.ndarray], selected_index: int | None = None
                     ) -> tuple[np.ndarray, int, str]:
    """仅作为物体中心近似：取闭合开始帧的 tool0 XY；准确旋转中心建议人工输入。"""
    q = np.asarray(record['ur_q'], dtype=float)
    events = detect_closing_events(record)
    if selected_index is not None:
        match = next((e for e in events if e['index'] == selected_index), None)
        if match is None:
            raise ValueError('自动中心关联的夹爪闭合事件不存在')
        index = int(match['index'])
        note = f'闭合事件 t={match["time_s"]:.2f}s 的 tool0 XY（近似）'
    elif events:
        index = int(events[0]['index'])
        note = f'首次闭合事件 t={events[0]["time_s"]:.2f}s 的 tool0 XY（近似）'
    else:
        index = len(q)-1
        note = '未检测到闭合事件，使用终点 tool0 XY（粗略近似）'
    T, _ = fk_jacobian(q[index])
    return T[:2, 3].copy(), index, note


def compute_shift(data: dict[str, np.ndarray], offset_m: np.ndarray,
                  ramp_seconds: float,
                  progress: Callable[[int, int], None] | None = None,
                  mode: str = 'full',
                  complete_time: float | None = None,
                  yaw_deg: float = 0.0,
                  pivot_xy_m: np.ndarray | None = None,
                  alpha_profile: np.ndarray | None = None):
    if mode not in ('full', 'early', 'instant', 'gripper', 'staged'):
        raise ValueError(f'无效的补偿模式: {mode}')
    if not np.isfinite(yaw_deg) or abs(yaw_deg) > 45:
        raise ValueError('绕Z旋转角度必须在 -45°～+45°')
    if pivot_xy_m is None:
        pivot_xy_m = np.zeros(2, dtype=float)
    pivot_xy_m = np.asarray(pivot_xy_m, dtype=float)
    if pivot_xy_m.shape != (2,) or not np.isfinite(pivot_xy_m).all():
        raise ValueError('原始物体旋转中心 X/Y 必须是有效数值')
    q0 = np.asarray(data['ur_q'], dtype=float)
    n = len(q0)
    hz = frequency_of(data)
    ts = np.arange(n, dtype=float) / hz
    if mode == 'staged':
        if alpha_profile is None:
            raise ValueError('分阶段模式必须提供补偿权重曲线')
        alpha_profile = np.asarray(alpha_profile, dtype=float)
        if (alpha_profile.shape != (n,) or not np.isfinite(alpha_profile).all()
                or np.any(alpha_profile < -1e-12) or np.any(alpha_profile > 1 + 1e-12)):
            raise ValueError('分阶段补偿权重不合法')
    q1 = np.empty_like(q0)
    max_pos_error = 0.0
    max_rot_error = 0.0
    last_original = last_solution = last_alpha = None

    for i, q in enumerate(q0):
        alpha = (float(alpha_profile[i]) if mode == 'staged' else
                 compensation_alpha(i, n, hz, mode, ramp_seconds, complete_time))
        if (last_original is not None and np.array_equal(q, last_original)
                and alpha == last_alpha):
            solution = last_solution
        else:
            original, _ = fk_jacobian(q)
            target = compensated_pose(original, offset_m, yaw_deg, pivot_xy_m, alpha)
            if alpha == 0.0 or (np.max(np.abs(offset_m)) < 1e-14 and abs(yaw_deg) < 1e-14):
                solution = q.copy()
            else:
                seed = q if last_solution is None else last_solution + (q - last_original)
                try:
                    solution = ik_near(target, seed)
                except Exception as exc:
                    # 首次未收敛不等同于几何不可达，改用该点原轨迹关节角再求一次。
                    try:
                        solution = ik_near(target, q)
                    except Exception as retry_exc:
                        raise RuntimeError(
                            f'第 {i+1}/{n} 个轨迹点（t={ts[i]:.3f} s）IK未满足数值精度；'
                            f'首次结果：{exc}；原始关节初值重试：{retry_exc}。'
                            '不能据此断言物理不可达；请勿直接回放。') from retry_exc
            if np.any(np.abs(solution) > JOINT_LIMIT + 1e-7):
                raise RuntimeError(f'第 {i+1} 点的新关节角超出 ±2π 名义范围，未保存结果')
            achieved, _ = fk_jacobian(solution)
            ep, er = pose_error(target, achieved)
            max_pos_error = max(max_pos_error, float(np.linalg.norm(ep)))
            max_rot_error = max(max_rot_error, float(np.linalg.norm(er)))
        q1[i] = solution
        last_original, last_solution, last_alpha = q, solution, alpha
        if progress and (i % 250 == 0 or i == n - 1):
            progress(i + 1, n)

    max_joint_jump = float(np.max(np.abs(np.diff(q1, axis=0))))
    max_joint_speed = float(np.max(np.abs(np.diff(q1, axis=0) * hz)))
    info = {
        'N': n, 'hz': hz,
        'max_pos_error_mm': max_pos_error * 1000,
        'max_rot_error_rad': max_rot_error,
        'max_joint_jump_rad': max_joint_jump,
        'max_joint_speed_rad_s': max_joint_speed,
        'mode': mode, 'yaw_deg': float(yaw_deg), 'rotation_center_xy_m': pivot_xy_m.copy(),
        'start_offset_mm': 1000.0 * float(np.linalg.norm(fk_jacobian(q1[0])[0][:3, 3] - fk_jacobian(q0[0])[0][:3, 3])),
        'end_offset_mm': 1000.0 * float(np.linalg.norm(fk_jacobian(q1[-1])[0][:3, 3] - fk_jacobian(q0[-1])[0][:3, 3])),
    }
    return q1, ts, info


def safe_part(v: float) -> str:
    # 命名使用 cm，精度到 0.001 cm，不与不同偏移量混淆
    if abs(v) < 0.0005:
        return '0'
    return ('p' if v > 0 else 'm') + f'{abs(v):.3f}'.rstrip('0').rstrip('.').replace('.', 'd')


def make_output_base(source: Path, xyz_cm: np.ndarray, folder: Path, yaw_deg: float = 0.0) -> Path:
    tag = '_'.join(axis + safe_part(float(v)) for axis, v in zip('xyz', xyz_cm))
    tag += '_rz' + safe_part(float(yaw_deg))
    stem = re.sub(r'[^\w\-]+', '_', source.stem)
    base = folder / f'{stem}_shift_{tag}'
    candidate = base
    version = 2
    while any(p.exists() for p in (
        candidate.with_suffix('.npz'),
        candidate.with_name(candidate.name + '_original_3d.png'),
        candidate.with_name(candidate.name + '_shifted_3d.png'),
    )):
        candidate = folder / f'{base.name}_v{version}'
        version += 1
    return candidate


def sampled_xyz(q: np.ndarray, indexes: np.ndarray) -> np.ndarray:
    xyz = np.empty((len(indexes), 3), dtype=float)
    for j, i in enumerate(indexes):
        xyz[j] = fk_jacobian(q[i])[0][:3, 3]
    return xyz


def draw_3d(ax, old: np.ndarray, new: np.ndarray, bounds: np.ndarray,
            compensated: bool, orientation_markers: dict | None = None):
    if compensated:
        ax.plot(*old.T, color='#97a3ae', linestyle='--', alpha=0.7,
                linewidth=1.5, label='Original reference')
        points, color, label = new, '#dc7d23', 'Compensated'
        ax.scatter(*old[0], marker='x', s=55, color='#374151', label='Original start')
        ax.scatter(*old[-1], marker='x', s=55, color='#374151', label='Original end')
        ax.plot(np.array([old[-1, 0], new[-1, 0]]),
                np.array([old[-1, 1], new[-1, 1]]),
                np.array([old[-1, 2], new[-1, 2]]),
                color='#9155b5', linewidth=2.0, label='End displacement')
    else:
        points, color, label = old, '#2678b5', 'Original'
    ax.plot(*points.T, color=color, linewidth=1.8, label=label)
    ax.scatter(*points[0], s=38, marker='o', color='#209e78', label='Start')
    ax.scatter(*points[-1], s=35, marker='^', color='#c33b56', label='End')
    # 同步显示法兰 tool0 的 X 轴方向：可看到姿态随物体 Rz 旋转。
    # 箭头代表法兰方向，不代表夹爪指尖方向。
    if orientation_markers:
        names = ('old', 'new') if compensated else ('old',)
        for name in names:
            p, direction = orientation_markers[name]
            arrow_color = '#374151' if name == 'old' else '#dc7d23'
            ax.quiver(*p, *direction, length=0.06, normalize=True,
                      color=arrow_color, linewidth=2.2,
                      label=('Original tool0 X' if name == 'old' else 'New tool0 X'))
    ax.set_xlim(*bounds[0]); ax.set_ylim(*bounds[1]); ax.set_zlim(*bounds[2])
    ax.set_box_aspect(bounds[:, 1] - bounds[:, 0])
    ax.set_xlabel('X (m)'); ax.set_ylabel('Y (m)'); ax.set_zlabel('Z (m)')
    ax.view_init(elev=25, azim=-56)
    ax.set_title('Compensated + original reference' if compensated else 'Original trajectory',
                 fontsize=10)
    ax.legend(loc='upper right', fontsize=7)


def common_bounds(old: np.ndarray, new: np.ndarray) -> np.ndarray:
    combined = np.vstack((old, new))
    lo, hi = np.min(combined, axis=0), np.max(combined, axis=0)
    mid = (lo + hi) / 2
    span = np.maximum((hi - lo) * 1.15, 0.08)
    return np.stack((mid - span / 2, mid + span / 2), axis=1)


def write_image(path: Path, old: np.ndarray, new: np.ndarray,
                bounds: np.ndarray, shifted: bool,
                orientation_markers: dict | None = None):
    # 不依赖 Tk 显示环境，可在后台线程直接保存 PNG
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    fig = Figure(figsize=(7.8, 6.1), dpi=155)
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection='3d')
    draw_3d(ax, old, new, bounds, compensated=shifted,
            orientation_markers=orientation_markers)
    fig.subplots_adjust(left=0.02, right=0.94, bottom=0.07, top=0.92)
    fig.savefig(path, dpi=155)
    fig.clear()


def generate_preview(source: Path, xyz_cm: np.ndarray, ramp: float,
                     progress: Callable[[str, int, int], None] | None = None,
                     mode: str = 'full',
                     close_event_index: int | None = None,
                     advance_seconds: float = 1.0,
                     yaw_deg: float = 0.0,
                     pivot_mode: str = 'auto',
                     pivot_xy_m: np.ndarray | None = None,
                     release_advance_seconds: float = 0.5,
                     return_start_mode: str = 'auto',
                     return_lift_mm: float = 30.0,
                     return_lift_wait_seconds: float = 0.2,
                     return_manual_time_s: float | None = None):
    """只计算，不写入磁盘：返回 NPZ 内容和两张图片共用的三维预览数据。"""
    xyz_cm = validate_xyz_offset_cm(xyz_cm)
    if mode not in ('full', 'early', 'instant', 'gripper', 'staged'):
        raise ValueError('补偿模式无效')
    if not np.isfinite(yaw_deg) or abs(yaw_deg) > 45:
        raise ValueError('绕 Z 轴的旋转角度范围为 -45°～+45°')
    if pivot_mode not in ('auto', 'manual'):
        raise ValueError('旋转中心模式必须为 auto 或 manual')
    if mode == 'early' and (not np.isfinite(ramp) or not (0 <= ramp <= 10)):
        raise ValueError('前 N 秒渐变时间必须在 0 到 10 秒范围内')
    source = Path(source).resolve()
    if not source.is_file() or source.suffix.lower() != '.npz':
        raise FileNotFoundError(f'原始 NPZ 不存在：{source}')
    record = load_record(source)
    q0 = np.asarray(record['ur_q'], dtype=float)
    hz = frequency_of(record)
    events = detect_closing_events(record)
    chosen = None
    complete_time = None
    stages = None
    alpha_profile = None
    cycles = detect_gripper_cycles(record, events) if mode == 'staged' else []
    if mode in ('gripper', 'staged'):
        if not np.isfinite(advance_seconds) or not 0 <= advance_seconds <= 10:
            raise ValueError('闭合前提前完成时间必须在 0~10 秒之间')
        if not events:
            raise ValueError('该文件没有识别到可靠的夹爪闭合事件；请检查夹爪数据或使用手动前N秒模式')
        if close_event_index is None:
            if len(events) > 1:
                raise ValueError('检测到多次夹爪闭合，请在 GUI 中选择哪一次用于抓取')
            chosen = events[0]
        else:
            chosen = next((ev for ev in events if ev['index'] == close_event_index), None)
            if chosen is None:
                raise ValueError('所选择的夹爪闭合事件不属于当前轨迹')
        complete_time = float(chosen['time_s'] - advance_seconds)
        if complete_time < 0.25:
            raise ValueError('完成时刻过早（不足0.25秒），请减小提前时间或选择其他闭合事件')
        if mode == 'staged':
            cycle = next((c for c in cycles if c['close_index'] == chosen['index']), None)
            if cycle is None:
                raise ValueError(
                    '无法将所选闭合事件匹配到可靠的“闭合稳定→张开开始”周期；'
                    '请检查原始夹爪反馈/命令数据或选择其他闭合事件')
            alpha_profile, stages = make_staged_profile(
                record, cycle, advance_seconds=advance_seconds,
                return_end_advance_s=release_advance_seconds,
                return_start_mode=return_start_mode,
                lift_mm=return_lift_mm,
                lift_settle_seconds=return_lift_wait_seconds,
                return_start_manual_s=return_manual_time_s)

    if pivot_mode == 'auto':
        # 自动中心为近似估计；基于 tool0 的 XY，而不是视觉/标定后的物体真实中心。
        pivot, pivot_ref_index, pivot_note = pivot_from_grasp(record, close_event_index)
    else:
        pivot = np.asarray(pivot_xy_m, dtype=float)
        if pivot.shape != (2,) or not np.isfinite(pivot).all():
            raise ValueError('请填写原始物体中心 X、Y（机器人基坐标系，单位 m）')
        pivot_ref_index = -1
        pivot_note = '人工输入的原始物体中心 XY（基坐标系）'

    def on_progress(i, n):
        if progress:
            progress('IK 计算中', i, n)

    q1, uniform_ts, stats = compute_shift(
        record, xyz_cm / 100, ramp, on_progress, mode=mode,
        complete_time=complete_time, yaw_deg=yaw_deg, pivot_xy_m=pivot,
        alpha_profile=alpha_profile)
    if progress:
        progress('生成预览数据', 0, 1)
    ids = np.linspace(0, len(q0) - 1, num=min(len(q0), 2400), dtype=int)
    old_xyz = sampled_xyz(q0, ids)
    new_xyz = sampled_xyz(q1, ids)
    bounds = common_bounds(old_xyz, new_xyz)
    # 标记闭合事件或最后一帧的 tool0 姿态，绘制末端方向轴。
    marker_i = chosen['index'] if chosen else (events[0]['index'] if events else len(q0)-1)
    marker_old, _ = fk_jacobian(q0[marker_i])
    marker_new, _ = fk_jacobian(q1[marker_i])
    orientation_markers = {
        'old': (marker_old[:3, 3].copy(), marker_old[:3, 0].copy()),
        'new': (marker_new[:3, 3].copy(), marker_new[:3, 0].copy()),
    }

    # 同步写入事件时间与补偿比例，便于后续核对每一个关节轨迹点。
    alpha_curve = (alpha_profile.copy() if mode == 'staged' else np.array([
        compensation_alpha(i, len(q0), hz, mode, ramp, complete_time)
        for i in range(len(q0))], dtype=float))
    # 旧版回放常用字段保持一致；旧主端数据单独存到 source_ 字段。
    output = {
        'timestamp': uniform_ts,
        'ur_q': q1,
        'ur_target_q': q1.copy(),
        'gripper_open': record['gripper_open'].copy(),
        'gripper_target_open': record.get('gripper_target_open', record['gripper_open']).copy(),
        'rtde_control_frequency_hz': np.array([stats['hz']], dtype=float),
        'rtde_receive_frequency_hz': record.get('rtde_receive_frequency_hz', np.array([np.nan])).copy(),
        'original_timestamp': record['timestamp'].copy(),
        'source_ur_q': q0.copy(),
        'source_ur_target_q': record.get('ur_target_q', q0).copy(),
        'source_master_q': record.get('master_q', np.empty((0, 6))).copy(),
        'source_tool7_pos': record.get('tool7_pos', np.empty((0,))).copy(),
        'offset_xyz_m': xyz_cm / 100,
        'xyz_compensation_limits_cm': XYZ_LIMITS_CM.copy(),
        'yaw_offset_deg': np.array([yaw_deg], dtype=float),
        'yaw_offset_rad': np.array([np.deg2rad(yaw_deg)], dtype=float),
        'rotation_center_xy_m': pivot.copy(),
        'rotation_center_mode': np.array([pivot_mode]),
        'rotation_center_reference_index': np.array([pivot_ref_index], dtype=int),
        'yaw_compensation_deg': alpha_curve * yaw_deg,
        'ramp_seconds': np.array([
            complete_time if mode in ('gripper', 'staged') else
            (ramp if mode == 'early' else (float(uniform_ts[-1]) if mode == 'full' else 0))
        ], dtype=float),
        'compensation_alpha': alpha_curve,
        'gripper_close_index': np.array([chosen['index'] if chosen else -1], dtype=int),
        'gripper_close_time_s': np.array([chosen['time_s'] if chosen else np.nan], dtype=float),
        'gripper_close_signal': np.array([chosen['source'] if chosen else 'none']),
        'gripper_advance_seconds': np.array([advance_seconds if chosen else np.nan], dtype=float),
        'compensation_complete_time_s': np.array([complete_time if chosen else np.nan], dtype=float),
        'compensation_mode': np.array([mode]),
        'nominal_dh_model': np.array(['UR5e_tool0']),
        'max_fk_position_error_m': np.array([stats['max_pos_error_mm'] / 1000]),
        'max_fk_orientation_error_rad': np.array([stats['max_rot_error_rad']]),
    }
    if stages is not None:
        cycle = stages['cycle']
        output.update({
            'staged_pick_complete_s': np.array([stages['pick_complete_s']]),
            'staged_return_start_s': np.array([stages['return_start_s']]),
            'staged_return_end_s': np.array([stages['return_end_s']]),
            'staged_lift_index': np.array([stages['lift_index']], dtype=int),
            'staged_return_start_mode': np.array([return_start_mode]),
            'staged_return_reason': np.array([stages['return_reason']]),
            'staged_lift_threshold_mm': np.array([return_lift_mm]),
            'staged_lift_wait_s': np.array([return_lift_wait_seconds]),
            'staged_release_advance_s': np.array([release_advance_seconds]),
            'gripper_closed_index': np.array([cycle['closed_index']], dtype=int),
            'gripper_closed_time_s': np.array([cycle['closed_time_s']]),
            'gripper_open_index': np.array([cycle['open_index']], dtype=int),
            'gripper_open_time_s': np.array([cycle['open_time_s']]),
            'gripper_opened_index': np.array([cycle['opened_index']], dtype=int),
            'gripper_opened_time_s': np.array([cycle['opened_time_s']]),
            'gripper_closed_signal': np.array([cycle['closed_source']]),
            'gripper_open_signal': np.array([cycle['open_source']]),
        })
        # 必须严格验证：从张开开始到最后一帧，目标 tool0 与原始轨迹一致。
        if not np.all(alpha_curve[cycle['open_index']:] == 0.0):
            raise RuntimeError('张开阶段仍然存在视觉补偿，拒绝生成轨迹')
        stats['return_start_s'] = stages['return_start_s']
        stats['return_end_s'] = stages['return_end_s']
        stats['gripper_open_time_s'] = cycle['open_time_s']
        stats['return_reason'] = stages['return_reason']
        place_old, _ = fk_jacobian(q0[cycle['open_index']])
        place_new, _ = fk_jacobian(q1[cycle['open_index']])
        place_pos_err, place_rot_err = pose_error(place_old, place_new)
        stats['placement_fk_error_mm'] = float(np.linalg.norm(place_pos_err) * 1000.0)
        stats['placement_orientation_error_deg'] = float(np.rad2deg(np.linalg.norm(place_rot_err)))
        stats['placement_joint_error_rad'] = float(np.max(np.abs(
            q1[cycle['open_index']] - q0[cycle['open_index']])))
    if progress:
        progress('预览就绪（尚未保存）', 1, 1)
    stats['gripper_close_time_s'] = chosen['time_s'] if chosen else None
    stats['compensation_complete_time_s'] = complete_time
    stats['pivot_note'] = pivot_note
    return {'yaw_deg': float(yaw_deg), 'pivot_xy_m': pivot.copy(),
            'pivot_mode': pivot_mode, 'pivot_note': pivot_note,
            'orientation_markers': orientation_markers,
            'source': source, 'xyz_cm': xyz_cm.copy(), 'ramp': ramp, 'mode': mode,
            'gripper_event': chosen, 'detected_events': events,
            'staged_boundaries': stages, 'detected_cycles': cycles,
            'gripper_time': uniform_ts,
            'gripper_target_curve': record.get('gripper_target_open', record['gripper_open']).copy(),
            'gripper_actual_curve': record['gripper_open'].copy(),
            'alpha_curve': alpha_curve,
            'npz_data': output, 'stats': stats,
            'old_xyz': old_xyz, 'new_xyz': new_xyz, 'bounds': bounds,
            'base': None, 'saved_paths': {}}


def save_selected_outputs(result: dict, output_dir: Path,
                          save_npz: bool, save_png: bool,
                          progress: Callable[[str, int, int], None] | None = None) -> dict:
    """按两个独立选项保存：NPZ 一份，PNG 一组两张；默认不覆盖旧文件。"""
    if not (save_npz or save_png):
        return {'saved': [], 'skipped': [], 'paths': dict(result['saved_paths'])}
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # 仅在用户第一次按下保存时固定文件名；之后追加保存另一类文件，沿用同一文件名前缀。
    if result['base'] is None:
        result['base'] = make_output_base(result['source'], result['xyz_cm'], output_dir, result.get('yaw_deg', 0.0))
        if result.get('mode') == 'staged':
            original_base = result['base']
            staged_base = original_base.with_name(original_base.name + '_stage')
            version = 2
            while any(x.exists() for x in (
                staged_base.with_suffix('.npz'),
                staged_base.with_name(staged_base.name + '_original_3d.png'),
                staged_base.with_name(staged_base.name + '_shifted_3d.png'),
            )):
                staged_base = original_base.with_name(f'{original_base.name}_stage_v{version}')
                version += 1
            result['base'] = staged_base
    base = result['base']
    destinations = {
        'npz': base.with_suffix('.npz'),
        'original_png': base.with_name(base.name + '_original_3d.png'),
        'compensated_png': base.with_name(base.name + '_shifted_3d.png'),
    }
    saved, skipped = [], []

    def atomic_save(key: str, writer):
        dest = destinations[key]
        if key in result['saved_paths'] and dest.is_file():
            skipped.append(dest)
            return
        if dest.exists():
            raise FileExistsError(f'目标文件已存在，程序不会覆盖：{dest}')
        fd, tempname = tempfile.mkstemp(
            prefix='.ur5e_pending_', suffix=dest.suffix, dir=output_dir)
        os.close(fd)
        temp = Path(tempname)
        try:
            writer(temp)
            if dest.exists():
                raise FileExistsError(f'目标文件已存在，程序不会覆盖：{dest}')
            os.replace(temp, dest)
            result['saved_paths'][key] = dest
            saved.append(dest)
        finally:
            temp.unlink(missing_ok=True)

    if save_npz:
        if progress:
            progress('保存 NPZ', 0, 1)
        atomic_save('npz', lambda path: np.savez_compressed(path, **result['npz_data']))
        if progress:
            progress('保存 NPZ', 1, 1)
    if save_png:
        if progress:
            progress('保存两张 PNG', 0, 2)
        # 两张轨迹图为一组：先写第一张，再写第二张；失败时回滚本次新建的 PNG。
        created_png_this_time = []
        try:
            count = len(saved)
            atomic_save('original_png', lambda path: write_image(
                path, result['old_xyz'], result['new_xyz'], result['bounds'], shifted=False,
                orientation_markers=result.get('orientation_markers')))
            created_png_this_time.extend(saved[count:])
            if progress:
                progress('保存两张 PNG', 1, 2)
            count = len(saved)
            atomic_save('compensated_png', lambda path: write_image(
                path, result['old_xyz'], result['new_xyz'], result['bounds'], shifted=True,
                orientation_markers=result.get('orientation_markers')))
            created_png_this_time.extend(saved[count:])
            if progress:
                progress('保存两张 PNG', 2, 2)
        except Exception:
            for path in created_png_this_time:
                path.unlink(missing_ok=True)
                result['saved_paths'].pop(
                    'original_png' if path == destinations['original_png'] else 'compensated_png', None)
                if path in saved:
                    saved.remove(path)
            raise
    return {'saved': saved, 'skipped': skipped, 'paths': dict(result['saved_paths'])}


# ================= 视觉任务绑定与输入数据验证 =================
# 这些函数只处理离线文件，不连接相机/UR，不发送机械臂运动指令。
VISUAL_EDGE_MM = 80.0
VISUAL_Z_WARN_M = 0.012  # 固定工作平面时的高度异常阈值；允许高度补偿时仍保留诊断
VISUAL_Z_COMPENSATION_LIMIT_M = float(XYZ_LIMITS_CM[2]) / 100.0  # 视觉Z与轨迹Z上限一致（10cm）
VISUAL_REF_CENTER_TOL_M = 1e-6
VISUAL_REF_YAW_TOL_DEG = 1e-3
TASK_MANIFEST_VERSION = 1


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def _read_json_dict(path: Path) -> dict:
    with Path(path).open('r', encoding='utf-8') as f:
        value = json.load(f)
    if not isinstance(value, dict):
        raise ValueError(f'JSON 顶层必须是对象：{path}')
    return value


def _vec3(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (3,) or not np.isfinite(arr).all():
        raise ValueError(f'{name} 必须为三个有限实数（单位 m）')
    return arr


def _matrix4(value, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (4, 4) or not np.isfinite(arr).all():
        raise ValueError(f'{name} 必须是有限的 4x4 矩阵')
    if not np.allclose(arr[3], [0., 0., 0., 1.], atol=1e-5):
        raise ValueError(f'{name} 的最后一行无效')
    R = arr[:3, :3]
    if np.linalg.norm(R.T @ R - np.eye(3)) > 0.03 or abs(np.linalg.det(R) - 1.) > 0.02:
        raise ValueError(f'{name} 不是合法的刚体转换矩阵')
    return arr


def _angle_mod90(degrees: float) -> float:
    return (float(degrees) + 45.0) % 90.0 - 45.0


def validate_visual_reference(ref: dict) -> dict:
    """解析 square_pose.py 的 reference_*.json 中 'reference' 内层。"""
    if not isinstance(ref, dict) or ref.get('frame') != 'base':
        raise ValueError('视觉参考必须是 frame=base 的有效三维位姿；不能使用仅相机坐标')
    center = _vec3(ref.get('center_m'), '参考 center_m')
    try:
        yaw = float(ref['yaw_deg'])
        edge_mm = float(ref['edge_mm'])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError('参考缺少合法的 yaw_deg / edge_mm') from e
    if not np.isfinite(yaw) or not np.isfinite(edge_mm):
        raise ValueError('参考角度或尺寸不是有限数值')
    if not np.isclose(edge_mm, VISUAL_EDGE_MM, rtol=0, atol=0.1):
        raise ValueError(f'参考方形边长应为 {VISUAL_EDGE_MM:g} mm，实际为 {edge_mm:.2f} mm')
    camera = str(ref.get('camera_serial', '')).strip()
    if not camera or camera == 'image':
        raise ValueError('参考缺少真实 RealSense 序列号，不能使用离线图片的假坐标')
    H = _matrix4(ref.get('handeye_matrix'), '参考 handeye_matrix')
    top = _matrix4(ref.get('T_base_top'), '参考 T_base_top')
    if not np.allclose(top[:3, 3], center, atol=1e-5):
        raise ValueError('参考 T_base_top 与 center_m 不一致')
    return {'raw': ref, 'center': center, 'yaw': _angle_mod90(yaw),
            'camera': camera, 'handeye': H, 'edge_mm': edge_mm,
            'capture_time': ref.get('capture_time_iso')}


def load_visual_reference(path: Path) -> dict:
    d = _read_json_dict(Path(path))
    return validate_visual_reference(d.get('reference', d))


def _same_reference(a: dict, b: dict) -> bool:
    aa = validate_visual_reference(a)
    bb = validate_visual_reference(b)
    return (aa['camera'] == bb['camera'] and
            abs(aa['edge_mm'] - bb['edge_mm']) <= 0.01 and
            np.allclose(aa['center'], bb['center'], atol=VISUAL_REF_CENTER_TOL_M, rtol=0) and
            abs(_angle_mod90(aa['yaw'] - bb['yaw'])) < VISUAL_REF_YAW_TOL_DEG and
            np.allclose(aa['handeye'], bb['handeye'], atol=1e-7, rtol=0) and
            (aa['capture_time'] is None or bb['capture_time'] is None or
             aa['capture_time'] == bb['capture_time']))


def load_visual_pair(reference_path: Path, detection_path: Path,
                     *, force_planar_z: bool = True) -> dict:
    """独立从两次 Base 位姿计算增量，并核对 pose.json 的嵌入参考及 delta。

    必须先在 square_pose 中加载同一个 reference，再保存 pose.json。
    """
    reference_path, detection_path = Path(reference_path), Path(detection_path)
    ref = load_visual_reference(reference_path)
    saved = _read_json_dict(detection_path)
    current = saved.get('current')
    if not isinstance(current, dict) or current.get('frame') != 'base':
        raise ValueError('当前检测文件必须包含 current.frame=base；请连接静止 UR 并加载手眼矩阵后重存')
    if current.get('stationary') is not True:
        raise ValueError('当前检测未确认 UR 静止，拒绝生成视觉补偿轨迹')
    if current.get('camera_serial') != ref['camera']:
        raise ValueError('示教参考与当前检测的 RealSense 序列号不同')
    if not np.isclose(float(current.get('edge_mm', np.nan)), ref['edge_mm'], atol=0.01, rtol=0):
        raise ValueError('顶面 80mm 边长不一致')
    embedded = saved.get('reference')
    if not isinstance(embedded, dict):
        raise ValueError('当前 pose.json 没有嵌入参考；请在 square_pose.py 加载绑定的 reference JSON，然后重新保存当前检测')
    if not _same_reference(ref['raw'], embedded):
        raise ValueError('当前 pose.json 内的参考与所选示教参考不是同一份！请重新在 square_pose.py 加载参考后拍摄')
    center_now = _vec3(current.get('center_m'), '当前 center_m')
    yaw_now = float(current.get('yaw_deg', np.nan))
    if not np.isfinite(yaw_now):
        raise ValueError('当前 yaw_deg 无效')
    Tnow = _matrix4(current.get('T_base_top'), '当前 T_base_top')
    if not np.allclose(Tnow[:3, 3], center_now, atol=1e-5):
        raise ValueError('当前 T_base_top 与 center_m 不一致')
    rmse = float(current.get('rmse_px', np.nan))
    if not np.isfinite(rmse) or rmse > 3.0 or rmse < 0:
        raise ValueError(f'当前 PnP 重投影误差无效或过大：{rmse}')
    delta_m = center_now - ref['center']
    yaw_deg = _angle_mod90(yaw_now - ref['yaw'])
    prior_delta = saved.get('delta')
    if prior_delta is None or prior_delta.get('frame') != 'base':
        raise ValueError('当前检测没有合法的 Base 差分；请重新加载参考并保存')
    d_xyz = _vec3(prior_delta.get('delta_xyz_mm'), '已有 delta_xyz_mm')
    stored_yaw = float(prior_delta.get('delta_yaw_deg', np.nan))
    if (not np.allclose(d_xyz, delta_m * 1000., atol=0.1, rtol=0)
            or not np.isfinite(stored_yaw)
            or abs(_angle_mod90(stored_yaw-yaw_deg)) > 0.03):
        raise ValueError('pose.json 已有差分与重新计算不一致，拒绝使用')
    ref_t = ref['capture_time']
    now_t = current.get('capture_time_iso')
    if ref_t and now_t:
        try:
            if dt.datetime.fromisoformat(now_t) < dt.datetime.fromisoformat(ref_t):
                raise ValueError('当前检测时间早于示教参考；检查文件是否选反')
        except TypeError as ex:
            raise ValueError('视觉位姿时间戳不合法') from ex
    # 两种 Z 模式均保留原 ±12 mm 检测，区别只在超阈值后的处理：
    # 默认勾选忽略 Z -> 仍视为固定平面异常并拒绝导入（原行为）。
    # 用户主动取消忽略 Z -> 允许真实工件升降，超过12mm仅警告，仍受Z±10cm补偿限制。
    z_exceeds_12mm = bool(abs(delta_m[2]) > VISUAL_Z_WARN_M + 1e-9)
    if force_planar_z and z_exceeds_12mm:
        raise ValueError(
            f'固定工作平面模式：检测到顶面高度变化 {delta_m[2]*1000:+.1f} mm，'
            '超过12mm；请检查深度/角点/工件是否抬升。'
            '若工件确实改变高度，请取消勾选「忽略视觉 ΔZ」后重新导入。'
        )
    if not force_planar_z and abs(delta_m[2]) > VISUAL_Z_COMPENSATION_LIMIT_M + 1e-9:
        raise ValueError(
            f'视觉Z补偿 {delta_m[2]*1000:+.1f} mm 超过轨迹Z补偿范围 '
            f'±{VISUAL_Z_COMPENSATION_LIMIT_M*1000:g} mm；不生成轨迹。'
        )
    offset_m = delta_m.copy()
    if force_planar_z:
        offset_m[2] = 0.0
    # 视觉文件导入时就检查逐轴范围，与手动补偿/轨迹规划采用同一组限制。
    # 固定工作平面时按处理后的Z=0检查，但原始Z超过12mm依旧会在前面拒绝。
    validate_xyz_offset_cm(offset_m * 100.0)
    z_warning = (
        f'视觉高度变化 {delta_m[2]*1000:+.1f} mm 超过原12mm阈值；'
        '已启用Z补偿，允许导入。请先核对工件实际高度、PnP测量和路径碰撞风险。'
        if z_exceeds_12mm and not force_planar_z else ''
    )
    return {'reference_center_m': ref['center'], 'current_center_m': center_now,
            'reference_yaw_deg': ref['yaw'], 'current_yaw_deg': _angle_mod90(yaw_now),
            'delta_raw_m': delta_m, 'offset_m': offset_m, 'yaw_deg': yaw_deg,
            'camera_serial': ref['camera'], 'handeye': ref['handeye'],
            'rmse_px': rmse, 'reference_path': reference_path.resolve(),
            'detection_path': detection_path.resolve(),
            'reference_capture_time': ref_t, 'current_capture_time': now_t,
            'detection_mode': current.get('detection_mode', ''),
            'force_planar_z': force_planar_z,
            'z_exceeds_12mm': z_exceeds_12mm,
            'z_warning': z_warning}


def create_visual_task(source_npz: Path, reference_path: Path, tasks_dir: Path) -> Path:
    """绑定 NPZ 与其示教前的视觉参考，并制作不可覆盖的自包含任务包。"""
    source_npz, reference_path, tasks_dir = map(Path, (source_npz, reference_path, tasks_dir))
    load_record(source_npz)
    ref = load_visual_reference(reference_path)
    tasks_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    stem = re.sub(r'[^a-zA-Z0-9_-]', '_', source_npz.stem)[:32] or 'teach'
    dirname = f'task_{timestamp}_{stem}'
    dest = tasks_dir / dirname
    if dest.exists():
        raise FileExistsError(f'任务目录已存在: {dest}')
    # 先写入临时目录；只在所有文件写完并核对后才作为完整任务出现
    stage = tasks_dir / ('.pending_' + dirname)
    if stage.exists():
        raise FileExistsError(f'临时目录已存在：{stage}')
    stage.mkdir()
    try:
        shutil.copy2(source_npz, stage / 'original.npz')
        shutil.copy2(reference_path, stage / 'reference.json')
        manifest = {
            'type': 'ur5e_visual_teach_task',
            'version': TASK_MANIFEST_VERSION,
            'created_at': dt.datetime.now().isoformat(timespec='seconds'),
            'npz_file': 'original.npz',
            'reference_file': 'reference.json',
            'npz_sha256': _sha256_file(stage / 'original.npz'),
            'reference_sha256': _sha256_file(stage / 'reference.json'),
            'source_npz_name': source_npz.name,
            'source_reference_name': reference_path.name,
            'reference_center_base_m': ref['center'].tolist(),
            'reference_yaw_deg_mod90': ref['yaw'],
            'camera_serial': ref['camera'],
            'square_edge_mm': ref['edge_mm'],
            'binding_note': '请确认示教期间物体没有移动；仅创建文件配对，不声称相机/关节逐帧同步',
        }
        (stage / 'task.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        # 确认复制的文件在新目录中仍然合法
        load_record(stage / 'original.npz')
        load_visual_reference(stage / 'reference.json')
        stage.rename(dest)
    except Exception:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    return dest / 'task.json'


def load_visual_task(task_json: Path) -> dict:
    task_json = Path(task_json).resolve()
    manifest = _read_json_dict(task_json)
    if manifest.get('type') != 'ur5e_visual_teach_task' or manifest.get('version') != TASK_MANIFEST_VERSION:
        raise ValueError('不是可识别的视觉示教任务包')
    if manifest.get('npz_file') != 'original.npz' or manifest.get('reference_file') != 'reference.json':
        raise ValueError('任务包文件名不符合安全规范')
    source = task_json.parent / 'original.npz'
    reference = task_json.parent / 'reference.json'
    if _sha256_file(source) != manifest.get('npz_sha256'):
        raise ValueError('任务包中的 NPZ 校验码不一致，文件可能被修改')
    if _sha256_file(reference) != manifest.get('reference_sha256'):
        raise ValueError('任务包中的视觉参考校验码不一致，文件可能被修改')
    load_record(source)
    ref = load_visual_reference(reference)
    if (not np.allclose(ref['center'], manifest.get('reference_center_base_m'), atol=1e-7, rtol=0)
            or abs(_angle_mod90(ref['yaw'] - float(manifest['reference_yaw_deg_mod90']))) > 1e-4
            or ref['camera'] != manifest['camera_serial']):
        raise ValueError('任务包参考元数据不一致')
    return {'path': task_json, 'npz': source, 'reference': reference, 'manifest': manifest}


def add_vision_metadata(preview: dict, vision: dict, task_manifest: Path | None = None):
    """保留原回放字段，仅增加 npz 无需 pickle 的字符串和数值字段。"""
    n = preview['npz_data']
    n['visual_mode'] = np.array(['eye_in_hand_square_80mm'])
    n['visual_reference_center_base_m'] = vision['reference_center_m'].copy()
    n['visual_current_center_base_m'] = vision['current_center_m'].copy()
    n['visual_delta_raw_xyz_m'] = vision['delta_raw_m'].copy()
    n['visual_used_offset_xyz_m'] = vision['offset_m'].copy()
    n['visual_reference_yaw_deg'] = np.array([vision['reference_yaw_deg']])
    n['visual_current_yaw_deg'] = np.array([vision['current_yaw_deg']])
    n['visual_delta_yaw_deg'] = np.array([vision['yaw_deg']])
    n['visual_reference_json_sha256'] = np.array([_sha256_file(vision['reference_path'])])
    n['visual_detection_json_sha256'] = np.array([_sha256_file(vision['detection_path'])])
    n['visual_reference_json_name'] = np.array([vision['reference_path'].name])
    n['visual_detection_json_name'] = np.array([vision['detection_path'].name])
    n['visual_task_name'] = np.array([task_manifest.parent.name if task_manifest else 'unbound'])
    n['visual_camera_serial'] = np.array([vision['camera_serial']])
    n['visual_depth_z_ignored'] = np.array([int(vision['force_planar_z'])])
    # 轨迹来源审计：记录原12mm诊断是否触发，而不修改任何原回放字段。
    n['visual_z_exceeds_12mm'] = np.array([int(vision['z_exceeds_12mm'])])
    preview['visual_info'] = vision
    preview['visual_task_manifest'] = str(task_manifest) if task_manifest else None
    return preview


class ShiftGui:
    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk
        self.tk = tk
        self.ttk = ttk
        self.root = root
        self.script_dir = Path(__file__).resolve().parent
        self.record_dir = self.script_dir / 'teach_records'
        self.output_dir = self.record_dir / 'shift_outputs'
        self.events = queue.Queue()
        self.busy = False
        self.last_result = None
        self.save_npz_var = tk.BooleanVar(value=True)
        self.save_png_var = tk.BooleanVar(value=True)
        self.records = {}
        self.root.title('UR5e XYZ + Rz 示教轨迹修正')
        self.root.geometry('1380x1000')
        self.root.minsize(1060, 760)
        self.file_name = tk.StringVar()
        self.dx = tk.StringVar(value='5.0')
        self.dy = tk.StringVar(value='0.0')
        self.dz = tk.StringVar(value='0.0')
        self.yaw = tk.StringVar(value='0.0')
        self.pivot_choice = tk.StringVar(value='auto')
        self.pivot_x = tk.StringVar(value='0.0')
        self.pivot_y = tk.StringVar(value='0.0')
        # 视觉模式独立于原有的手动补偿模式；初始仍采用手动模式。
        self.input_mode = tk.StringVar(value='manual')
        self.visual_ref_path = tk.StringVar(value='')
        self.visual_detection_path = tk.StringVar(value='')
        self.visual_task_path = tk.StringVar(value='')
        self.visual_ignore_z = tk.BooleanVar(value=True)
        self.visual_summary = tk.StringVar(value='未加载视觉文件。请先在示教前记录参考，再将 NPZ 与参考配对。')
        self.visual_task = None
        self.visual_preview = None
        self._last_input_mode = 'manual'
        self._manual_pivot_selection = 'auto'
        self._manual_snapshot = None
        self.ramp = tk.StringVar(value='2.0')
        self.mode_label = tk.StringVar(value='夹爪闭合自动补偿：抓取前完成偏移')
        self.advance = tk.StringVar(value='1.0')
        self.release_advance = tk.StringVar(value='0.5')
        self.return_start_mode = tk.StringVar(value='auto')
        self.return_lift_mm = tk.StringVar(value='30.0')
        self.return_lift_wait = tk.StringVar(value='0.2')
        self.return_manual_time = tk.StringVar(value='5.0')
        self.event_label = tk.StringVar()
        self.close_events = []
        self.gripper_cycles = []
        self.event_options = {}
        self.summary = tk.StringVar(value='请选择示教轨迹文件。')
        self.status = tk.StringVar(value='就绪（不连接 UR5e）')
        self.result_label = tk.StringVar(value='生成轨迹并查看预览后，可自主决定保存哪些文件；视觉模式在【视觉绑定】Tab导入JSON并绑定示教NPZ。')

        style = ttk.Style()
        if 'clam' in style.theme_names(): style.theme_use('clam')
        self.build_widgets()
        self.refresh_files()
        self.root.after(100, self.poll_events)
        self.on_input_mode_changed()

    def build_widgets(self):
        tk, ttk = self.tk, self.ttk
        main = ttk.Frame(self.root, padding=14)
        main.pack(fill='both', expand=True)
        ttk.Label(main, text='UR5e 示教轨迹离线补偿', font=('Microsoft YaHei UI', 17, 'bold')).pack(anchor='w')
        ttk.Label(main, text='支持手动补偿或【视觉任务绑定】：读取示教 NPZ + 参考 JSON + 当前检测 JSON，离线生成并预览轨迹（不控制真实机器人）。',
                  foreground='#4e6474').pack(anchor='w', pady=(2, 12))

        choice = ttk.LabelFrame(main, text='1 选择原始轨迹', padding=10)
        choice.pack(fill='x')
        line = ttk.Frame(choice); line.pack(fill='x')
        ttk.Label(line, text='轨迹文件').pack(side='left', padx=(0, 8))
        self.combo = ttk.Combobox(line, textvariable=self.file_name, state='readonly', width=48)
        self.combo.pack(side='left', fill='x', expand=True)
        self.combo.bind('<<ComboboxSelected>>', lambda e: self.selected_changed())
        self.refresh_button = ttk.Button(line, text='刷新列表', command=self.refresh_files)
        self.refresh_button.pack(side='left', padx=(10, 0))
        ttk.Label(choice, text=f'默认目录：{self.record_dir}', foreground='#677782').pack(anchor='w', pady=(7, 2))
        ttk.Label(choice, textvariable=self.summary, foreground='#314e69').pack(anchor='w')

        setting = ttk.LabelFrame(main, text='2 设置目标物体位姿变化（基坐标系）', padding=10)
        setting.pack(fill='x', pady=(10, 0))
        mode_line = ttk.Frame(setting)
        mode_line.pack(fill='x', pady=(0, 7))
        ttk.Label(mode_line, text='输入来源：').pack(side='left')
        ttk.Radiobutton(mode_line, text='手动输入（保留原功能）', variable=self.input_mode,
                        value='manual', command=self.on_input_mode_changed).pack(side='left', padx=(8, 18))
        ttk.Radiobutton(mode_line, text='视觉参考 + 当前检测（自动读取）', variable=self.input_mode,
                        value='vision', command=self.on_input_mode_changed).pack(side='left')
        ttk.Label(mode_line, text='视觉模式请在下方【视觉绑定】Tab 导入文件',
                  foreground='#416277').pack(side='left', padx=18)
        row = ttk.Frame(setting); row.pack(fill='x')
        self.spins = []
        for (label, value), limit in zip([('X', self.dx), ('Y', self.dy), ('Z', self.dz)], XYZ_LIMITS_CM):
            group = ttk.Frame(row); group.pack(side='left', padx=(0, 22))
            ttk.Label(group, text=f'{label}：').pack(side='left')
            spin = ttk.Spinbox(group, from_=-float(limit), to=float(limit), increment=0.5,
                               textvariable=value, width=9, justify='center')
            spin.pack(side='left')
            ttk.Label(group, text='cm').pack(side='left', padx=(4, 0))
            self.spins.append(spin)
        ttk.Label(row, text='允许范围：X ±12 cm，Y ±12 cm，Z ±10 cm（可输入小数）', foreground='#677782').pack(side='left')
        # 新增姿态与旋转中心：物体绕自身旧中心作平面 yaw 旋转。
        rotation_row = ttk.Frame(setting)
        rotation_row.pack(fill='x', pady=(9, 0))
        ttk.Label(rotation_row, text='绕基坐标系 Z 轴旋转 Rz：').pack(side='left')
        self.yaw_spin = ttk.Spinbox(rotation_row, from_=-45.0, to=45.0,
                                    increment=1.0, textvariable=self.yaw,
                                    width=8, justify='center')
        self.yaw_spin.pack(side='left')
        ttk.Label(rotation_row, text='°（-45 ～ +45，逆时针为正，从Z正方向俯视）',
                  foreground='#677782').pack(side='left', padx=(7, 0))

        pivot_row = ttk.Frame(setting)
        pivot_row.pack(fill='x', pady=(8, 0))
        ttk.Label(pivot_row, text='物体原始旋转中心：').pack(side='left')
        ttk.Radiobutton(pivot_row, text='自动估计（闭合时 tool0 XY，近似）',
                        variable=self.pivot_choice, value='auto',
                        command=self.on_pivot_changed).pack(side='left', padx=(8, 12))
        ttk.Radiobutton(pivot_row, text='手动输入物体中心 XY',
                        variable=self.pivot_choice, value='manual',
                        command=self.on_pivot_changed).pack(side='left', padx=(0, 12))
        self.pivot_entry_frame = ttk.Frame(pivot_row)
        self.pivot_entry_frame.pack(side='left')
        ttk.Label(self.pivot_entry_frame, text='X:').pack(side='left')
        self.pivot_x_spin = ttk.Spinbox(self.pivot_entry_frame, from_=-2.0, to=2.0,
                                         increment=0.01, textvariable=self.pivot_x,
                                         width=8, justify='center')
        self.pivot_x_spin.pack(side='left')
        ttk.Label(self.pivot_entry_frame, text='Y:').pack(side='left', padx=(7, 0))
        self.pivot_y_spin = ttk.Spinbox(self.pivot_entry_frame, from_=-2.0, to=2.0,
                                         increment=0.01, textvariable=self.pivot_y,
                                         width=8, justify='center')
        self.pivot_y_spin.pack(side='left')
        ttk.Label(self.pivot_entry_frame, text='m').pack(side='left', padx=(4, 0))
        ttk.Label(setting, text='旋转中心指原始示教时的物体中心：自动值仅为法兰XY近似，真实抓取请优先用标定/测量的物体中心。',
                  foreground='#9a571c').pack(anchor='w', pady=(3, 0))
        self.on_pivot_changed()
        under = ttk.Frame(setting); under.pack(fill='x', pady=(10, 0))
        ttk.Label(under, text='补偿方式：').pack(side='left')
        self.mode_labels = {
            '全程渐变：起点不变、终点完整偏移': 'full',
            '前 N 秒渐变：起点不变、N秒后完整偏移': 'early',
            '立即平移：所有点偏移（包括起点）': 'instant',
            '夹爪闭合自动补偿：抓取前完成偏移': 'gripper',
            '分阶段补偿：自适应抓取 + 原位放置': 'staged',
        }
        self.mode_combo = ttk.Combobox(
            under, textvariable=self.mode_label, state='readonly', width=38,
            values=list(self.mode_labels))
        self.mode_combo.pack(side='left', padx=(0, 12))
        self.mode_combo.bind('<<ComboboxSelected>>', lambda e: self.on_mode_changed())
        self.ramp_label = ttk.Label(under, text='提前完成时间：')
        self.ramp_label.pack(side='left')
        self.ramp_spin = ttk.Spinbox(under, from_=0.1, to=10, increment=0.5,
                                     textvariable=self.ramp, width=7, justify='center')
        self.ramp_spin.pack(side='left')
        ttk.Label(under, text='秒（仅“前 N 秒渐变”有效）',
                  foreground='#677782').pack(side='left', padx=(7, 0))
        self.grip_row = ttk.Frame(setting)
        self.grip_row.pack(fill='x', pady=(8, 0))
        ttk.Label(self.grip_row, text='闭合事件：').pack(side='left')
        self.event_combo = ttk.Combobox(self.grip_row, textvariable=self.event_label,
                                        state='readonly', width=42)
        self.event_combo.pack(side='left', padx=(4, 15))
        self.event_combo.bind('<<ComboboxSelected>>', lambda e: self.update_event_note())
        ttk.Label(self.grip_row, text='提前完成：').pack(side='left')
        self.advance_spin = ttk.Spinbox(self.grip_row, from_=0, to=10, increment=0.5,
                                        textvariable=self.advance, width=7, justify='center')
        self.advance_spin.pack(side='left')
        ttk.Label(self.grip_row, text='秒').pack(side='left', padx=(4, 10))
        self.event_note = tk.StringVar(value='将提前于夹爪开始闭合完成完整位置补偿')
        ttk.Label(self.grip_row, textvariable=self.event_note,
                  foreground='#416277').pack(side='left')
        self.advance.trace_add('write', lambda *_: self.update_event_note())

        self.stage_frame = ttk.LabelFrame(setting, text='分阶段模式：抓稳并抬升后回退，松爪前回归原放置轨迹', padding=6)
        stage1 = ttk.Frame(self.stage_frame)
        stage1.pack(fill='x')
        ttk.Label(stage1, text='回退开始：').pack(side='left')
        ttk.Radiobutton(stage1, text='自动检查抬升', variable=self.return_start_mode,
                        value='auto', command=self.on_return_mode_changed).pack(side='left', padx=(3, 9))
        ttk.Radiobutton(stage1, text='手动指定时间', variable=self.return_start_mode,
                        value='manual', command=self.on_return_mode_changed).pack(side='left', padx=(0, 8))
        ttk.Label(stage1, text='抬升至少').pack(side='left')
        self.lift_spin = ttk.Spinbox(stage1, from_=5, to=300, increment=5,
                                     textvariable=self.return_lift_mm, width=6, justify='center')
        self.lift_spin.pack(side='left')
        ttk.Label(stage1, text='mm；再等待').pack(side='left', padx=(3, 0))
        self.lift_wait_spin = ttk.Spinbox(stage1, from_=0, to=5, increment=0.1,
                                          textvariable=self.return_lift_wait, width=5, justify='center')
        self.lift_wait_spin.pack(side='left')
        ttk.Label(stage1, text='s；手动时间').pack(side='left', padx=(3, 0))
        self.return_manual_spin = ttk.Spinbox(stage1, from_=0, to=100, increment=0.1,
                                               textvariable=self.return_manual_time,
                                               width=7, justify='center')
        self.return_manual_spin.pack(side='left')
        ttk.Label(stage1, text='s').pack(side='left')
        stage2 = ttk.Frame(self.stage_frame)
        stage2.pack(fill='x', pady=(5, 0))
        ttk.Label(stage2, text='第二阶段在夹爪开始张开前').pack(side='left')
        self.release_advance_spin = ttk.Spinbox(stage2, from_=0, to=10, increment=0.1,
                                                textvariable=self.release_advance,
                                                width=6, justify='center')
        self.release_advance_spin.pack(side='left')
        ttk.Label(stage2, text='秒结束；要求回退时长≥0.5s。自动抬升仅依据名义tool0高度，需确认工件确已离桌。',
                  foreground='#9a571c').pack(side='left', padx=(7, 0))
        self.on_return_mode_changed()
        self.mode_hint = ttk.Label(setting,
                  text='提示：夹爪开合只是状态事件，不能证明抓取/离桌成功；生成的轨迹未经过实机碰撞检查。',
                  foreground='#7a5c33')
        self.mode_hint.pack(anchor='w', pady=(7, 0))
        self.on_mode_changed()

        actions = ttk.Frame(main); actions.pack(fill='x', pady=(11, 8))
        self.run_button = ttk.Button(actions, text='生成轨迹并预览（暂不保存）', command=self.start)
        self.run_button.pack(side='left')
        self.folder_button = ttk.Button(actions, text='打开输出文件夹', command=self.open_folder)
        self.folder_button.pack(side='left', padx=(10, 0))
        ttk.Label(actions, textvariable=self.status, foreground='#175b85').pack(side='right')
        save_frame = ttk.LabelFrame(main, text='3 结果保存（计算完成后可自主选择）', padding=9)
        save_frame.pack(fill='x', pady=(3, 9))
        save_row = ttk.Frame(save_frame)
        save_row.pack(fill='x')
        self.save_npz_check = ttk.Checkbutton(save_row, text='保存修正后的 NPZ',
                                             variable=self.save_npz_var, state='disabled')
        self.save_npz_check.pack(side='left', padx=(0, 22))
        self.save_png_check = ttk.Checkbutton(save_row, text='保存两张三维轨迹图（PNG）',
                                             variable=self.save_png_var, state='disabled')
        self.save_png_check.pack(side='left', padx=(0, 22))
        self.save_button = ttk.Button(save_row, text='保存所选结果',
                                      command=self.start_save, state='disabled')
        self.save_button.pack(side='left')
        ttk.Label(save_frame, text=f'保存到：{self.output_dir}（生成预览不会自动写文件）',
                  foreground='#677782').pack(anchor='w', pady=(7, 0))

        self.progress_bar = ttk.Progressbar(main, mode='determinate', maximum=100)
        self.progress_bar.pack(fill='x', pady=(0, 6))
        ttk.Label(main, textvariable=self.result_label, foreground='#3f6760', wraplength=1100).pack(anchor='w')

        notebook = ttk.Notebook(main)
        notebook.pack(fill='both', expand=True, pady=(10, 0))
        self.preview_tab = ttk.Frame(notebook)
        self.log_tab = ttk.Frame(notebook)
        self.gripper_tab = ttk.Frame(notebook)
        notebook.add(self.preview_tab, text='三维轨迹预览（两张图）')
        notebook.add(self.gripper_tab, text='夹爪事件与补偿比例')
        notebook.add(self.log_tab, text='运行记录')
        self.visual_tab = ttk.Frame(notebook)
        notebook.add(self.visual_tab, text='视觉绑定 / JSON导入')
        self.build_visual_tab()
        self.gripper_canvas = None
        self.preview_note = ttk.Label(self.preview_tab, text='计算完成后，这里会显示原始轨迹和补偿后的轨迹。',
                                      foreground='#6a7881')
        self.preview_note.pack(pady=30)
        self.canvas = None
        self.log_widget = tk.Text(self.log_tab, height=12, wrap='word', state='disabled')
        self.log_widget.pack(side='left', fill='both', expand=True)
        bar = ttk.Scrollbar(self.log_tab, command=self.log_widget.yview)
        bar.pack(side='right', fill='y')
        self.log_widget.config(yscrollcommand=bar.set)
        ttk.Label(main, text='安全提示：仅供离线实验；使用名义 DH 法兰 tool0；自动旋转中心为近似值，未校准实机运动学/TCP/碰撞/加速度。',
                  foreground='#9a571c', wraplength=1100).pack(anchor='w', pady=(7, 0))

    def build_visual_tab(self):
        """只操作离线文件；实际相机记录仍由 square_pose.py 完成。"""
        t, ttk = self.tk, self.ttk
        # Notebook 的可用高度可能很小，给视觉输入页单独提供滚动容器。
        canvas = t.Canvas(self.visual_tab, highlightthickness=0)
        bar = ttk.Scrollbar(self.visual_tab, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=bar.set)
        bar.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)
        frame = ttk.Frame(canvas, padding=14)
        visual_window = canvas.create_window((0, 0), window=frame, anchor='nw')
        frame.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>', lambda e: canvas.itemconfigure(visual_window, width=e.width))
        def _on_wheel(event):
            step = -1 if getattr(event, 'delta', 0) > 0 else 1
            canvas.yview_scroll(step * 3, 'units')
        def _on_enter(event):
            canvas.bind_all('<MouseWheel>', _on_wheel)
        def _on_leave(event):
            canvas.unbind_all('<MouseWheel>')
        canvas.bind('<Enter>', _on_enter)
        canvas.bind('<Leave>', _on_leave)
        self.visual_canvas = canvas
        ttk.Label(frame, text='推荐流程：在工件示教位置拍参考 → 遥操作录制NPZ → 绑定任务 → 移动物体拍当前检测 → 生成新NPZ',
                  foreground='#245773', wraplength=1100).pack(anchor='w', pady=(0, 10))
        box = ttk.LabelFrame(frame, text='1 建立示教轨迹与原始视觉参考的任务绑定', padding=10)
        box.pack(fill='x')
        self._visual_file_row(box, '示教参考 JSON', self.visual_ref_path, self.select_visual_reference)
        row = ttk.Frame(box); row.pack(fill='x', pady=(6, 0))
        ttk.Button(row, text='创建任务包（当前选择的NPZ + 参考JSON）',
                   command=self.create_task_ui).pack(side='left')
        ttk.Button(row, text='加载已有任务包 task.json',
                   command=self.load_task_ui).pack(side='left', padx=(8, 0))
        ttk.Label(row, text='任务包在 teach_records/visual_tasks 下；自动复制并校验原始数据',
                  foreground='#677782').pack(side='left', padx=(12, 0))
        ttk.Label(box, textvariable=self.visual_task_path, foreground='#416277',
                  wraplength=1000).pack(anchor='w', pady=(4, 0))
        current_box = ttk.LabelFrame(frame, text='2 导入工件改变位置后的最新视觉检测', padding=10)
        current_box.pack(fill='x', pady=(10, 0))
        self._visual_file_row(current_box, '当前 pose.json', self.visual_detection_path, self.select_visual_detection)
        settings = ttk.Frame(current_box); settings.pack(fill='x', pady=(8, 0))
        ttk.Checkbutton(settings, text='固定工作台面：忽略视觉 ΔZ（默认勾选；超12mm拒绝导入）',
                        variable=self.visual_ignore_z, command=self.on_visual_config_changed).pack(side='left')
        ttk.Button(settings, text='校验并导入视觉变化量', command=self.apply_visual_ui).pack(side='left', padx=(14, 0))
        ttk.Label(current_box,
                  text='取消勾选：使用视觉 ΔZ；超过12mm会警告但允许导入，X/Y 各限±12cm，Z限±10cm。'
                       '大幅升降前请独立核实高度与轨迹/环境碰撞风险。',
                  foreground='#95531b', wraplength=1080).pack(anchor='w', pady=(6, 0))
        ttk.Label(frame, textvariable=self.visual_summary, foreground='#175b85',
                  wraplength=1080, justify='left').pack(anchor='w', pady=(12, 7))
        ttk.Label(frame, text='重要：任务绑定不是硬件同步。只有 reference 对应这份 NPZ 的原始工件位置，'
                  '并且当前检测引用了同一份 reference，补偿才具有正确物理含义。'
                  '程序拒绝相机序列号、矩阵、尺寸或参考不一致的数据。',
                  foreground='#95531b', wraplength=1080).pack(anchor='w')

    def _visual_file_row(self, parent, title, variable, callback):
        row = self.ttk.Frame(parent); row.pack(fill='x', pady=3)
        self.ttk.Label(row, text=title, width=16).pack(side='left')
        self.ttk.Entry(row, textvariable=variable, state='readonly', width=82).pack(side='left', fill='x', expand=True)
        self.ttk.Button(row, text='选择文件…', command=callback).pack(side='left', padx=(8, 0))

    def on_input_mode_changed(self):
        vision = self.input_mode.get() == 'vision'
        if vision and self._last_input_mode != 'vision':
            self._manual_pivot_selection = self.pivot_choice.get()
            self._manual_snapshot = tuple(v.get() for v in
                                          (self.dx, self.dy, self.dz, self.yaw, self.pivot_x, self.pivot_y))
        if not vision and self._last_input_mode == 'vision':
            self.pivot_choice.set(self._manual_pivot_selection)
            if self._manual_snapshot is not None:
                for variable, value in zip((self.dx, self.dy, self.dz, self.yaw, self.pivot_x, self.pivot_y),
                                           self._manual_snapshot):
                    variable.set(value)
        self._last_input_mode = 'vision' if vision else 'manual'
        for item in (*self.spins, self.yaw_spin):
            item.config(state='disabled' if vision else 'normal')
        if vision:
            self.pivot_choice.set('manual')
        self.on_pivot_changed()
        if vision:
            self.pivot_x_spin.config(state='disabled')
            self.pivot_y_spin.config(state='disabled')
            self.log('已切换为视觉输入：轨迹生成时会再次读取并校验选定 JSON，不使用手动填写的 XYZ/Yaw/中心。')
        else:
            self.log('已切换为手动输入：保留以前所有补偿选项。')

    def on_visual_config_changed(self):
        self.visual_preview = None
        self.visual_summary.set('Z 处理设置已改变，请重新校验视觉数据。')

    def select_visual_reference(self):
        from tkinter import filedialog, messagebox
        path = filedialog.askopenfilename(title='选择 square_pose.py 的 reference_*.json',
                                          filetypes=[('JSON file', '*.json')])
        if not path: return
        try:
            ref = load_visual_reference(Path(path))
            # 换绑参考后，原任务包失效；禁止暗中对旧任务重新关联
            self.visual_ref_path.set(str(Path(path).resolve()))
            self.visual_task_path.set('')
            self.visual_task = None
            self.visual_preview = None
            self.visual_summary.set(
                f"参考已加载：Base 中心={np.round(ref['center']*1000, 2).tolist()}mm, "
                f"Yaw={ref['yaw']:+.2f}°, 相机={ref['camera']}。请确认这就是当前 NPZ 录制时的工件位置。")
            self.log('视觉参考已加载（尚未绑定NPZ）：'+path)
        except Exception as exc:
            messagebox.showerror('视觉参考无效', str(exc))

    def select_visual_detection(self):
        from tkinter import filedialog, messagebox
        path = filedialog.askopenfilename(title='选择移动后保存的 pose.json',
                                          filetypes=[('JSON file', '*.json')])
        if not path: return
        try:
            refpath = self._get_ref_path()
            # 立即校验，生成时也重复校验
            v = load_visual_pair(refpath, Path(path), force_planar_z=bool(self.visual_ignore_z.get()))
            self.visual_detection_path.set(str(Path(path).resolve()))
            self.visual_preview = v
            self._show_visual(v)
        except Exception as exc:
            messagebox.showerror('当前视觉检测无效', str(exc))

    def _get_ref_path(self) -> Path:
        if self.visual_task is not None:
            # 每次调用均重新检查任务包 SHA256，避免文件在绑定后被替换
            task = load_visual_task(self.visual_task['path'])
            if self.visual_ref_path.get() and Path(self.visual_ref_path.get()).resolve() != task['reference'].resolve():
                raise ValueError('当前参考不是任务包的参考文件，请重新加载 task.json')
            return task['reference']
        value = self.visual_ref_path.get().strip()
        if not value:
            raise ValueError('请先选择示教参考 reference_*.json 或加载任务包 task.json')
        return Path(value)

    def _get_vision(self) -> dict:
        p = self.visual_detection_path.get().strip()
        if not p:
            raise ValueError('请先导入移动后的 pose.json')
        ref = self._get_ref_path()
        return load_visual_pair(ref, Path(p), force_planar_z=bool(self.visual_ignore_z.get()))

    def _show_visual(self, v: dict):
        # 在手动模式下导入/预览 JSON 仅更新视觉摘要，不覆盖原来的手动参数。
        if self.input_mode.get() == 'vision':
            self.dx.set(f"{v['offset_m'][0]*100:.4f}")
            self.dy.set(f"{v['offset_m'][1]*100:.4f}")
            self.dz.set(f"{v['offset_m'][2]*100:.4f}")
            self.yaw.set(f"{v['yaw_deg']:.5f}")
            self.pivot_choice.set('manual')
            self.pivot_x.set(f"{v['reference_center_m'][0]:.7f}")
            self.pivot_y.set(f"{v['reference_center_m'][1]:.7f}")
            self.on_pivot_changed()
            self.pivot_x_spin.config(state='disabled'); self.pivot_y_spin.config(state='disabled')
        msg = (f"视觉计算成功 | ΔX={v['offset_m'][0]*1000:+.2f}mm, "
               f"ΔY={v['offset_m'][1]*1000:+.2f}mm, "
               f"视觉原始ΔZ={v['delta_raw_m'][2]*1000:+.2f}mm "
               f"({'不补偿Z' if v['force_planar_z'] else '已采用Z'}), "
               f"ΔYaw={v['yaw_deg']:+.2f}°，"
               f"原物体旋转中心Base XY={np.round(v['reference_center_m'][:2],5).tolist()}m | "
               f"PnP RMSE={v['rmse_px']:.3f}px。")
        if v.get('z_warning'):
            msg += '\n【Z高度变化提醒】' + v['z_warning']
        self.visual_summary.set(msg)
        self.log(msg)

    def apply_visual_ui(self):
        from tkinter import messagebox
        try:
            vision = self._get_vision()
            self.visual_preview = vision
            self.input_mode.set('vision')
            self.on_input_mode_changed()
            self._show_visual(vision)
        except Exception as exc:
            messagebox.showerror('视觉数据校验失败', str(exc))

    def create_task_ui(self):
        from tkinter import messagebox
        try:
            source = self.records.get(self.file_name.get())
            if source is None:
                raise ValueError('先在上方选择实际录制的 NPZ')
            if not self.visual_ref_path.get():
                raise ValueError('请先选择与该 NPZ 示教时位置对应的 reference_*.json')
            if not messagebox.askyesno('确认任务绑定',
                                       f'轨迹：{source.name}\n参考：{Path(self.visual_ref_path.get()).name}\n\n'
                                       '请确认参考拍摄后、示教录制期间，工件没有改变位置或朝向。\n'
                                       '这只是文件关联，不是相机与机器人逐帧同步。是否创建任务包？'):
                return
            task_path = create_visual_task(source, Path(self.visual_ref_path.get()),
                                           self.record_dir / 'visual_tasks')
            self._set_task(load_visual_task(task_path))
            messagebox.showinfo('绑定完成', f'视觉参考与 NPZ 已复制成任务包：\n{task_path}')
        except Exception as exc:
            messagebox.showerror('绑定失败', str(exc))

    def load_task_ui(self):
        from tkinter import filedialog, messagebox
        p = filedialog.askopenfilename(title='打开视觉任务包 task.json', filetypes=[('JSON file', '*.json')],
                                       initialdir=str(self.record_dir / 'visual_tasks'))
        if not p: return
        try:
            self._set_task(load_visual_task(Path(p)))
        except Exception as exc:
            messagebox.showerror('任务包加载失败', str(exc))

    def _set_task(self, task: dict):
        self.visual_task = task
        self.visual_task_path.set('任务包：' + str(task['path']))
        self.visual_ref_path.set(str(task['reference'].resolve()))
        # 直接将任务包 original.npz 加入可选列表：即使源 NPZ 原来不在 teach_records 也能打开。
        label = f"[视觉任务] {task['path'].parent.name}"
        self.records[label] = task['npz']
        names = list(self.combo['values'])
        if label not in names:
            self.combo['values'] = [label] + names
        self.file_name.set(label)
        self.selected_changed()
        self.visual_preview = None
        self.visual_detection_path.set('')
        self.visual_summary.set('任务绑定已载入并校验 SHA256。请在 square_pose.py 中加载任务包的 reference.json，再保存新位置 pose.json。')
        self.log('已加载视觉任务：' + str(task['path']))

    def log(self, s):
        self.log_widget.config(state='normal')
        self.log_widget.insert('end', s + '\n')
        self.log_widget.see('end')
        self.log_widget.config(state='disabled')

    def refresh_files(self):
        if self.busy: return
        if not self.record_dir.is_dir():
            self.combo['values'] = []
            self.records = {}
            self.file_name.set('')
            self.summary.set('未找到 teach_records 文件夹。请将它放在 GUI 脚本同级目录。')
            return
        paths = sorted(self.record_dir.glob('*.npz'), key=lambda p: p.stat().st_mtime, reverse=True)
        self.records = {p.name: p for p in paths}
        if self.visual_task is not None:
            task_label = f"[视觉任务] {self.visual_task['path'].parent.name}"
            self.records[task_label] = self.visual_task['npz']
        names = list(self.records)
        prior = self.file_name.get()
        self.combo['values'] = names
        self.file_name.set(prior if prior in self.records else (names[0] if names else ''))
        self.selected_changed()
        self.log(f'扫描完成：在 teach_records 找到 {len(names)} 个 NPZ 文件。')

    def selected_changed(self):
        path = self.records.get(self.file_name.get())
        if self.visual_task is not None and path is not None and Path(path).resolve() != self.visual_task['npz'].resolve():
            self.log('切换到其他 NPZ：原视觉任务绑定不再适用，生成视觉轨迹前需重新匹配。')
        if not path:
            self.close_events = []
            self.gripper_cycles = []
            self.event_options = {}
            self.event_combo['values'] = []
            self.event_label.set('')
            self.summary.set('文件夹中尚无 NPZ 文件。')
            return
        try:
            with np.load(path, allow_pickle=False) as d:
                if 'ur_q' not in d:
                    raise ValueError('缺少 ur_q')
                q = d['ur_q']
                hz = float(d['rtde_control_frequency_hz'].ravel()[0]) if 'rtde_control_frequency_hz' in d else None
                msg = f'轨迹点：{len(q)}  |  UR关节维度：{q.shape}  |  频率：{hz:g} Hz' if hz else f'轨迹点：{len(q)}  |  UR关节维度：{q.shape}'
                ev_data = {'gripper_open': d['gripper_open'].copy()} if 'gripper_open' in d else {}
                if 'gripper_target_open' in d:
                    ev_data['gripper_target_open'] = d['gripper_target_open'].copy()
                if 'rtde_control_frequency_hz' in d:
                    ev_data['rtde_control_frequency_hz'] = d['rtde_control_frequency_hz'].copy()
                elif 'timestamp' in d:
                    ev_data['timestamp'] = d['timestamp'].copy()
                self.close_events = detect_closing_events(ev_data) if ev_data else []
                self.gripper_cycles = detect_gripper_cycles(ev_data, self.close_events) if ev_data else []
                msg += (f'  |  闭合事件：{len(self.close_events)} 次'
                        f'  |  完整开合周期：{len(self.gripper_cycles)} 次')
                self.summary.set(msg)
                self.event_options = {
                    f"闭合 {i + 1}：{ev['time_s']:.2f} s（{ev['source']}）": ev['index']
                    for i, ev in enumerate(self.close_events)
                }
                names = list(self.event_options)
                self.event_combo['values'] = names
                self.event_label.set(names[0] if names else '')
                self.update_event_note()
        except Exception as e:
            self.close_events = []
            self.gripper_cycles = []
            self.event_options = {}
            self.event_combo['values'] = []
            self.event_label.set('')
            self.summary.set(f'文件可能无法读取：{e}')

    def on_mode_changed(self):
        mode = self.mode_labels.get(self.mode_label.get(), 'full')
        self.ramp_spin.config(state='normal' if mode == 'early' else 'disabled')
        if mode in ('gripper', 'staged'):
            if not self.grip_row.winfo_manager():
                self.grip_row.pack(fill='x', pady=(8, 0), before=self.mode_hint)
        else:
            self.grip_row.pack_forget()
        if mode == 'staged':
            if not self.stage_frame.winfo_manager():
                self.stage_frame.pack(fill='x', pady=(5, 0), before=self.mode_hint)
            self.mode_hint.config(text='分阶段：夹爪闭合前补偿到100%，抬升后退回到0%，在夹爪开始张开前回归原始放置轨迹。')
        else:
            self.stage_frame.pack_forget()
            self.mode_hint.config(text='注意：夹爪运动并非接触检测；单阶段补偿持续保持，原放置位置可能改变。')
        self.update_event_note()

    def on_return_mode_changed(self):
        automatic = self.return_start_mode.get() == 'auto'
        for w in (self.lift_spin, self.lift_wait_spin):
            w.configure(state='normal' if automatic else 'disabled')
        self.return_manual_spin.configure(state='disabled' if automatic else 'normal')

    def on_pivot_changed(self):
        manual = self.pivot_choice.get() == 'manual'
        for spin in (self.pivot_x_spin, self.pivot_y_spin):
            spin.config(state='normal' if manual else 'disabled')

    def update_event_note(self):
        selected = self.event_options.get(self.event_label.get())
        event = next((v for v in self.close_events if v['index'] == selected), None)
        if event is None:
            self.event_note.set('未检测到可靠闭合，请尝试手动前N秒模式')
            return
        try:
            advance = float(self.advance.get())
            if not np.isfinite(advance):
                raise ValueError
            completion = event['time_s'] - advance
            cycle = next((c for c in self.gripper_cycles if c['close_index'] == selected), None)
            if self.mode_labels.get(self.mode_label.get()) == 'staged':
                if cycle:
                    self.event_note.set(
                        f"抓取补偿完成{completion:.2f}s；闭合稳定{cycle['closed_time_s']:.2f}s，"
                        f"张开开始{cycle['open_time_s']:.2f}s")
                else:
                    self.event_note.set('该闭合事件没有可靠的张开周期，不适用于分阶段模式')
            else:
                self.event_note.set(f'预计 {completion:.2f} s 完成')
        except ValueError:
            self.event_note.set('请输入有效的提前秒数')

    def start(self):
        from tkinter import messagebox
        if self.busy: return
        src = self.records.get(self.file_name.get())
        if not src:
            messagebox.showwarning('请选择文件', '请先在列表中选择一份 teach_records 下的 NPZ 轨迹。')
            return
        vision = None
        try:
            if self.input_mode.get() == 'vision':
                if self.visual_task is None:
                    raise ValueError('视觉模式必须先创建/加载任务包，使 NPZ 与示教参考绑定')
                task = load_visual_task(self.visual_task['path'])
                if Path(src).resolve() != task['npz'].resolve():
                    raise ValueError('选择的 NPZ 与绑定的视觉任务包不一致')
                vision = self._get_vision()
                self.visual_preview = vision
                self._show_visual(vision)
                # 用精确数值计算，不通过 GUI 小数文本中转，避免截断精度
                values = vision['offset_m'] * 100.0
            else:
                values = np.array([float(self.dx.get()), float(self.dy.get()), float(self.dz.get())])
            mode = self.mode_labels.get(self.mode_label.get())
            ramp = float(self.ramp.get())
            yaw = float(vision['yaw_deg']) if vision is not None else float(self.yaw.get())
            pivot_mode = 'manual' if vision is not None else self.pivot_choice.get()
            pivot_xy = (vision['reference_center_m'][:2].copy() if vision is not None
                        else np.array([float(self.pivot_x.get()), float(self.pivot_y.get())])
                        if pivot_mode == 'manual' else None)
            if not np.isfinite(yaw) or not -45 <= yaw <= 45:
                raise ValueError('绕 Z 轴的旋转角度必须在 -45°～+45°')
            if pivot_mode not in ('auto', 'manual'):
                raise ValueError('旋转中心类型无效')
            if pivot_xy is not None and (not np.isfinite(pivot_xy).all() or np.any(np.abs(pivot_xy) > 2.0)):
                raise ValueError('物体中心 X/Y 需为基坐标系下 ±2.0 m 内的有效数值')
            values = validate_xyz_offset_cm(values)
            if mode not in ('full', 'early', 'instant', 'gripper', 'staged'):
                raise ValueError('请选择补偿方式')
            event_idx = self.event_options.get(self.event_label.get())
            advance = float(self.advance.get()) if mode in ('gripper', 'staged') else 1.0
            release_advance = float(self.release_advance.get()) if mode == 'staged' else 0.5
            start_mode = self.return_start_mode.get() if mode == 'staged' else 'auto'
            lift_mm = float(self.return_lift_mm.get()) if mode == 'staged' else 30.0
            lift_wait = float(self.return_lift_wait.get()) if mode == 'staged' else 0.2
            manual_start = float(self.return_manual_time.get()) if mode == 'staged' and start_mode == 'manual' else None
            if mode in ('gripper', 'staged'):
                if event_idx is None:
                    raise ValueError('无法识别有效的夹爪闭合事件，请检查数据或改用前N秒模式')
                if not np.isfinite(advance) or not 0 <= advance <= 10:
                    raise ValueError('提前完成时间必须在 0~10 秒之间')
                event = next((ev for ev in self.close_events if ev['index'] == event_idx), None)
                if event is None or event['time_s']-advance < 0.25:
                    raise ValueError('提前量过大，补偿完成时刻不得早于0.25秒')
                if mode == 'staged' and not any(
                        c['close_index'] == event_idx for c in self.gripper_cycles):
                    raise ValueError('所选闭合事件找不到闭合稳定→开始张开的有效周期')
            if mode == 'early' and (not np.isfinite(ramp) or not 0 < ramp <= 10):
                raise ValueError('前N秒渐变时间应在 (0, 10] 秒')
        except ValueError as e:
            messagebox.showerror('输入有误', f'请检查参数：{e}')
            return
        self.busy = True
        self.last_result = None
        self.save_button.config(state='disabled')
        self.save_npz_check.config(state='disabled')
        self.save_png_check.config(state='disabled')
        self.run_button.config(state='disabled')
        self.refresh_button.config(state='disabled')
        self.combo.config(state='disabled')
        self.progress_bar['value'] = 0
        self.status.set('计算中…')
        self.result_label.set('正在离线计算，结果仅保存在内存中，暂不写入 NPZ 或 PNG。')
        self.log(f'开始：{src.name}，XYZ = {values.tolist()} cm，Yaw={yaw:.2f}°，旋转中心={pivot_mode}，补偿方式：{self.mode_label.get()}')

        def worker():
            try:
                def cb(stage, done, total):
                    self.events.put(('progress', stage, done, total))
                result = generate_preview(src, values, ramp, cb, mode=mode,
                                          close_event_index=event_idx, advance_seconds=advance,
                                          yaw_deg=yaw, pivot_mode=pivot_mode, pivot_xy_m=pivot_xy,
                                          release_advance_seconds=release_advance,
                                          return_start_mode=start_mode,
                                          return_lift_mm=lift_mm,
                                          return_lift_wait_seconds=lift_wait,
                                          return_manual_time_s=manual_start)
                if vision is not None:
                    # 规划完成后仍记录视觉来源，供后续回放审计；不更改 ur_q / 夹爪关键字段。
                    add_vision_metadata(result, vision, self.visual_task['path'] if self.visual_task else None)
                self.events.put(('success', result))
            except Exception as e:
                self.events.put(('error', str(e), traceback.format_exc()))
        threading.Thread(target=worker, daemon=True).start()

    def start_save(self):
        from tkinter import messagebox
        if self.busy or self.last_result is None:
            return
        want_npz = bool(self.save_npz_var.get())
        want_png = bool(self.save_png_var.get())
        if not (want_npz or want_png):
            messagebox.showwarning('请选择保存内容', '请至少勾选保存 NPZ 或保存两张 PNG。\n不保存可以直接继续预览或处理下一条轨迹。')
            return
        self.busy = True
        self.run_button.config(state='disabled')
        self.refresh_button.config(state='disabled')
        self.combo.config(state='disabled')
        self.save_button.config(state='disabled')
        self.save_npz_check.config(state='disabled')
        self.save_png_check.config(state='disabled')
        self.status.set('正在保存所选文件…')
        self.log(f'保存选择：NPZ={want_npz}，两张PNG={want_png}')
        result = self.last_result

        def worker():
            try:
                def cb(stage, done, total):
                    self.events.put(('progress', stage, done, total))
                outcome = save_selected_outputs(
                    result, self.output_dir, want_npz, want_png, cb)
                self.events.put(('saved', outcome))
            except Exception as e:
                self.events.put(('save_error', str(e), traceback.format_exc()))
        threading.Thread(target=worker, daemon=True).start()

    def poll_events(self):
        from tkinter import messagebox
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == 'progress':
                    _, stage, done, total = event
                    self.status.set(f'{stage}（{done}/{total}）')
                    if stage == 'IK 计算中':
                        self.progress_bar['value'] = 85 * done / total
                    elif stage == '生成预览数据':
                        self.progress_bar['value'] = 90
                    elif stage == '预览就绪（尚未保存）':
                        self.progress_bar['value'] = 100
                elif event[0] == 'success':
                    self.finish_busy()
                    self.last_result = event[1]
                    info = self.last_result['stats']
                    self.status.set('轨迹已生成，等待保存选择')
                    self.progress_bar['value'] = 100
                    self.result_label.set(
                        f"已生成内存预览（尚未保存）：{self.last_result['source'].name}，"
                        f"XYZ = {self.last_result['xyz_cm'].tolist()} cm，Yaw={self.last_result['yaw_deg']:.2f}°。"
                        '勾选需要保存的类型，然后点击“保存所选结果”。')
                    self.log('轨迹生成成功；目前没有保存任何 NPZ 或 PNG 文件。')
                    self.log(f"旋转中心 XY={self.last_result['pivot_xy_m'].tolist()} m；{self.last_result['pivot_note']}")
                    if self.last_result.get('visual_info'):
                        vision_used = self.last_result['visual_info']
                        self.log('【视觉绑定】使用任务：' + str(self.last_result.get('visual_task_manifest')))
                        self.log(f"【视觉绑定】参考/当前={vision_used['reference_path'].name}/{vision_used['detection_path'].name}; "
                                 f"PnP重投影RMSE={vision_used['rmse_px']:.3f}px; "
                                 f"参考中心XY={vision_used['reference_center_m'][:2].tolist()}m")
                    self.log(f"完整补偿的绕Z角度={self.last_result['yaw_deg']:.3f}°（逆时针为正）")
                    self.log(f"N={info['N']}，频率={info['hz']:.2f}Hz，FK位置误差上限={info['max_pos_error_mm']:.4f}mm，"
                             f"峰值关节差={info['max_joint_jump_rad']:.4f}rad，峰值离散速度={info['max_joint_speed_rad_s']:.3f}rad/s")
                    self.log(f"首点位移={info['start_offset_mm']:.4f}mm，末点位移={info['end_offset_mm']:.4f}mm（数值模型）")
                    if self.last_result['gripper_event'] is not None:
                        ev = self.last_result['gripper_event']
                        self.log(f"检测到夹爪闭合：{ev['time_s']:.3f}s（索引{ev['index']}，{ev['source']}）；"
                                 f"在 {info['compensation_complete_time_s']:.3f}s 完成XYZ+Yaw补偿")
                    if self.last_result.get('staged_boundaries') is not None:
                        stage = self.last_result['staged_boundaries']
                        cycle = stage['cycle']
                        self.log(
                            f"【分阶段】抓取完成={stage['pick_complete_s']:.3f}s；"
                            f"夹爪闭合稳定={cycle['closed_time_s']:.3f}s；"
                            f"回退开始={stage['return_start_s']:.3f}s；"
                            f"回退结束={stage['return_end_s']:.3f}s；"
                            f"开始张开={cycle['open_time_s']:.3f}s")
                        self.log(f"【分阶段】回退依据：{stage['return_reason']}")
                        self.log(
                            f"【分阶段】张开时相对原轨迹：tool0位置偏差="
                            f"{info['placement_fk_error_mm']:.6f}mm；"
                            f"姿态偏差={info['placement_orientation_error_deg']:.6f}°；"
                            f"最大关节差={info['placement_joint_error_rad']:.6f}rad")
                        if cycle['opened_index'] < 0:
                            self.log('【提示】未确认夹爪张开稳定时刻，但已在张开开始前完成补偿；请检查夹爪数据。')
                        self.log('【警告】名义法兰上抬并不证明工件离桌；'
                                 '工件/工具碰撞、承载、实际TCP和安全余量必须单独确认。')
                    if info['max_joint_jump_rad'] > 0.08:
                        self.log('【警告】相邻轨迹点关节角差超过 0.08rad；不得直接用于实机回放！')
                    if info['max_joint_speed_rad_s'] > 2.0:
                        self.log('【警告】相邻采样点差分速度超过2rad/s，请检查真实安全速度与轨迹时序！')
                    try:
                        self.show_preview(self.last_result)
                        self.show_gripper_plot(self.last_result)
                    except Exception as e:
                        self.log(f'GUI 预览失败（可尝试保存两张 PNG 查看）：{e}')
                    self.enable_save_controls()
                    messagebox.showinfo('轨迹生成成功',
                                        '新的关节轨迹已计算完成，三维图已准备好。\n\n'
                                        '目前没有自动保存文件。\n'
                                        '请在“结果保存”区域分别选择 NPZ 和/或两张 PNG，再点击“保存所选结果”。\n\n'
                                        '注意：还没有进行实机回放安全校验。')
                elif event[0] == 'saved':
                    self.finish_busy()
                    self.enable_save_controls()
                    outcome = event[1]
                    self.status.set('已完成保存')
                    for p in outcome['saved']:
                        self.log(f'已保存：{p}')
                    for p in outcome['skipped']:
                        self.log(f'已存在、未重复保存：{p}')
                    self.result_label.set(
                        f"当前预览：{self.last_result['source'].name}；"
                        f"已保存 {len(self.last_result['saved_paths'])}/3 个文件。"
                        f"目录：{self.output_dir}")
                    if outcome['saved']:
                        messagebox.showinfo('保存成功',
                                            '已保存：\n' + '\n'.join(p.name for p in outcome['saved']) +
                                            f'\n\n所在文件夹：{self.output_dir}')
                    else:
                        messagebox.showinfo('无需重复保存', '选中的文件之前已经保存过，未覆盖文件。')
                elif event[0] == 'save_error':
                    self.finish_busy()
                    self.enable_save_controls()
                    self.status.set('保存失败，可重新尝试')
                    self.log('保存失败：' + event[1])
                    self.log(event[2])
                    messagebox.showerror('保存失败', event[1])
                elif event[0] == 'error':
                    self.finish_busy()
                    self.status.set('处理失败')
                    self.result_label.set('未生成可用的修正轨迹。请检查运行记录。')
                    self.log('处理失败：' + event[1])
                    self.log(event[2])
                    messagebox.showerror('离线修正失败', event[1])
        except queue.Empty:
            pass
        self.root.after(120, self.poll_events)

    def enable_save_controls(self):
        if self.last_result is not None and not self.busy:
            self.save_button.config(state='normal')
            self.save_npz_check.config(state='normal')
            self.save_png_check.config(state='normal')

    def finish_busy(self):
        self.busy = False
        self.run_button.config(state='normal')
        self.refresh_button.config(state='normal')
        self.combo.config(state='readonly')

    def show_preview(self, result):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
        self.preview_note.pack_forget()
        fig = Figure(figsize=(10.7, 4.8), dpi=95)
        old, new, bounds = result['old_xyz'], result['new_xyz'], result['bounds']
        for j, compensated in enumerate((False, True)):
            ax = fig.add_subplot(1, 2, j + 1, projection='3d')
            draw_3d(ax, old, new, bounds, compensated=compensated,
                    orientation_markers=result.get('orientation_markers'))
        fig.subplots_adjust(left=0, right=0.98, bottom=0.08, top=0.94, wspace=0.06)
        self.canvas = FigureCanvasTkAgg(fig, master=self.preview_tab)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def show_gripper_plot(self, result):
        """单独展示夹爪开合与补偿比例，让自动事件识别可核对。"""
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        if self.gripper_canvas:
            self.gripper_canvas.get_tk_widget().destroy()
            self.gripper_canvas = None
        fig = Figure(figsize=(10.7, 4.8), dpi=95)
        ax = fig.add_subplot(111)
        ts = result['gripper_time']
        ax.plot(ts, result['gripper_target_curve'], color='#2678b5',
                linewidth=1.25, label='Gripper target open')
        ax.plot(ts, result['gripper_actual_curve'], color='#26986c',
                linewidth=1.25, alpha=.8, label='Gripper actual open')
        if result['gripper_event'] is not None:
            ev = result['gripper_event']
            ax.axvline(ev['time_s'], color='#c33b56', linestyle='--',
                       label=f"Close onset {ev['time_s']:.2f}s")
            ax.axvline(result['stats']['compensation_complete_time_s'],
                       color='#d28b27', linestyle=':', linewidth=2,
                       label=f"Compensation done {result['stats']['compensation_complete_time_s']:.2f}s")
        stages = result.get('staged_boundaries')
        if stages:
            cycle = stages['cycle']
            marks = [
                ('Closed stable', cycle['closed_time_s'], '#b65b15'),
                ('Return start', stages['return_start_s'], '#b74a97'),
                ('Return end', stages['return_end_s'], '#1e7491'),
                ('Release start', cycle['open_time_s'], '#2b8b5a'),
            ]
            for label, time_s, color in marks:
                ax.axvline(time_s, color=color, linestyle='--', linewidth=1.1,
                           label=f'{label} {time_s:.2f}s')
            if cycle['opened_index'] >= 0:
                ax.axvline(cycle['opened_time_s'], color='#588f4e', linestyle=':',
                           linewidth=1.1, label=f"Fully opened {cycle['opened_time_s']:.2f}s")
            ax.axvspan(stages['return_start_s'], stages['return_end_s'],
                       color='#b74a97', alpha=0.05)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Open fraction (larger = more open)')
        ax.grid(alpha=0.2)
        ax2 = ax.twinx()
        ax2.plot(ts, result['alpha_curve'], color='#9a69ca', alpha=.75,
                 linewidth=1.5, label='Compensation ratio')
        ax2.set_ylabel(f"Compensation ratio (Yaw = {result.get('yaw_deg', 0.0):+.1f} deg at 100%)")
        ax2.set_ylim(-.08, 1.08)
        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)
        fig.subplots_adjust(left=0.07, right=0.92, bottom=0.14, top=0.95)
        self.gripper_canvas = FigureCanvasTkAgg(fig, master=self.gripper_tab)
        self.gripper_canvas.draw()
        self.gripper_canvas.get_tk_widget().pack(fill='both', expand=True)

    def open_folder(self):
        from tkinter import messagebox
        folder = self.output_dir if self.output_dir.is_dir() else self.record_dir
        if not folder.is_dir():
            messagebox.showwarning('目录不存在', f'找不到文件夹：\n{folder}')
            return
        try:
            if sys.platform == 'win32':
                os.startfile(str(folder))
            elif sys.platform == 'darwin':
                import subprocess
                subprocess.Popen(['open', str(folder)])
            else:
                import subprocess
                subprocess.Popen(['xdg-open', str(folder)])
        except Exception as e:
            messagebox.showerror('打开失败', str(e))


def main():
    import tkinter as tk
    root = tk.Tk()
    ShiftGui(root)
    root.mainloop()


if __name__ == '__main__':
    main()
