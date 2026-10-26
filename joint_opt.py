"""Offline, fixed-time joint trajectory NLP. No robot/camera communication.

CasADi/IPOPT optimizes one quintic B-spline over the entire trajectory.
Joint bounds use derivative control-point hulls; task poses are constrained
at every output sample and independently checked at additional midpoints.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time
import traceback

import numpy as np
from scipy.interpolate import BSpline
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr
from scipy.spatial.transform import Rotation, Slerp

VERSION = "1.0"
A = np.array([0., -.425, -.3922, 0., 0., 0.])
D = np.array([.1625, 0., 0., .1333, .0997, .0996])
ALPHA = np.array([np.pi / 2, 0., 0., np.pi / 2, -np.pi / 2, 0.])


class Cancelled(Exception):
    pass


class OptimizationError(Exception):
    def __init__(self, message, report=None):
        super().__init__(message)
        self.report = report or {"valid": False, "error": message}


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def vector(value, name, positive=False):
    a = np.asarray(value, dtype=float)
    if a.shape != (6,) or not np.isfinite(a).all() or (positive and np.any(a <= 0)):
        raise ValueError(f"{name} 必须是六个有限数字" + ("，且全部大于 0" if positive else ""))
    return a


def validate_config(value):
    cfg = copy.deepcopy(value)
    if cfg.get("version") != 1:
        raise ValueError("配置 version 必须为 1")
    lim, task, opt = cfg["limits"], cfg["task"], cfg["optimizer"]
    lo = vector(lim["q_min"], "q_min")
    hi = vector(lim["q_max"], "q_max")
    if np.any(lo >= hi):
        raise ValueError("每个 q_min 必须小于 q_max")
    for key in ("v_max", "a_max"):
        vector(lim[key], key, True)
    if lim.get("j_max") is not None:
        vector(lim["j_max"], "j_max", True)
    for key in ("free_position_mm", "critical_position_mm", "lift_position_mm"):
        x = float(task[key])
        if not math.isfinite(x) or x <= .01:
            raise ValueError(f"{key} 必须大于 0.01 mm")
    for key in ("free_orientation_deg", "critical_orientation_deg", "lift_orientation_deg"):
        x = float(task[key])
        if not math.isfinite(x) or not .01 < x < 90:
            raise ValueError(f"{key} 必须在 (0.01,90) 度内")
    for key in ("close_window_before_s", "close_window_after_s", "placement_before_s", "placement_after_s"):
        if not math.isfinite(float(task[key])) or float(task[key]) < 0:
            raise ValueError(f"{key} 必须非负")
    for key in ("check_auto_lift", "freeze_tail"):
        if type(task.get(key)) is not bool:
            raise ValueError(f"{key} 必须是 JSON true/false")
    if task.get("min_tcp_z_m") is not None:
        win = np.asarray(task.get("clearance_window_s"), float)
        if win.shape != (2,) or not np.isfinite(win).all() or not 0 <= win[0] < win[1]:
            raise ValueError("启用 min_tcp_z_m 时须设置有效的 clearance_window_s")
        if not math.isfinite(float(task["min_tcp_z_m"])):
            raise ValueError("min_tcp_z_m 无效")
    for key, low, high in (("knot_spacing_s", .04, 5), ("max_cpu_seconds", 1, 3600)):
        if not math.isfinite(float(opt[key])) or not low <= float(opt[key]) <= high:
            raise ValueError(f"{key} 必须在 [{low},{high}] 内")
    for key in ("max_iterations", "refinement_rounds", "validation_substeps"):
        if type(opt[key]) is not int:
            raise ValueError(f"{key} 必须为整数")
    if not 1 <= int(opt["max_iterations"]) <= 10000:
        raise ValueError("max_iterations 必须在 1～10000 内")
    if not 0 <= int(opt["refinement_rounds"]) <= 3 or not 2 <= int(opt["validation_substeps"]) <= 10:
        raise ValueError("refinement_rounds/validation_substeps 超出范围")
    vector(opt["max_joint_deviation_rad"], "max_joint_deviation_rad", True)
    vector(opt["jerk_scale"], "jerk_scale", True)
    for key in ("reference", "velocity", "acceleration", "jerk"):
        x = float(opt["weights"][key])
        if not math.isfinite(x) or x < 0:
            raise ValueError(f"目标权重 {key} 必须非负")
    if float(opt["weights"]["reference"]) <= 0:
        raise ValueError("reference 权重必须大于 0")
    for key in ("start_velocity", "end_velocity", "start_acceleration", "end_acceleration"):
        if cfg["boundary"].get(key) is not None:
            vector(cfg["boundary"][key], key)
    vector(cfg["model"]["tcp_xyz_rotvec"], "tcp_xyz_rotvec")
    for key in ("limits_confirmed", "task_tolerances_confirmed"):
        if type(cfg.get(key)) is not bool:
            raise ValueError(f"{key} 必须是 JSON true/false")
    return cfg


def load_config(path):
    return validate_config(json.loads(Path(path).read_text(encoding="utf-8-sig")))


def tcp_matrix(cfg):
    v = vector(cfg["model"]["tcp_xyz_rotvec"], "tcp")
    t = np.eye(4)
    t[:3, :3] = Rotation.from_rotvec(v[3:]).as_matrix()
    t[:3, 3] = v[:3]
    return t


def fk_numpy(q, tcp=None):
    """Independent batched NumPy FK; q: (N,6)."""
    q = np.atleast_2d(np.asarray(q, float))
    if q.shape[1] != 6 or not np.isfinite(q).all():
        raise ValueError("关节数组必须为有限的 N×6")
    t = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
    for k in range(6):
        c, s = np.cos(q[:, k]), np.sin(q[:, k])
        ca, sa = np.cos(ALPHA[k]), np.sin(ALPHA[k])
        m = np.zeros_like(t)
        m[:, 0, 0], m[:, 0, 1], m[:, 0, 2], m[:, 0, 3] = c, -s * ca, s * sa, A[k] * c
        m[:, 1, 0], m[:, 1, 1], m[:, 1, 2], m[:, 1, 3] = s, c * ca, -c * sa, A[k] * s
        m[:, 2, 1], m[:, 2, 2], m[:, 2, 3], m[:, 3, 3] = sa, ca, D[k], 1
        t = t @ m
    return t if tcp is None else t @ np.asarray(tcp)


def build_reference(raw_q, candidate_q, timestamps, alpha, delta_m, yaw_deg, pivot_xy, stages, event, cycle, cfg):
    t = np.asarray(timestamps, float)
    q = np.asarray(candidate_q, float)
    if q.shape != np.shape(raw_q) or q.shape != (len(t), 6) or len(t) < 4:
        raise ValueError("优化需要至少 4 个 N×6 关节点，原始/候选形状必须相同")
    if not np.isfinite(q).all():
        raise ValueError("候选关节轨迹包含非有限数值")
    if not np.isfinite(t).all() or not np.all(np.diff(t) > 0) or abs(t[0]) > 1e-8:
        raise ValueError("规划时间必须从 0 开始并严格递增")
    if not np.allclose(np.diff(t), np.diff(t)[0], atol=1e-8, rtol=1e-5):
        raise ValueError("本版要求统一的固定采样周期，不自动更改时基")
    if np.max(np.abs(np.diff(q, axis=0))) > .2:
        raise ValueError("候选相邻关节跳变超过 0.2 rad，请先核对 IK 分支/角度表示，不能盲目 unwrap")
    a = np.asarray(alpha, float)
    if a.shape != t.shape or not np.isfinite(a).all() or np.any((a < 0) | (a > 1)):
        raise ValueError("补偿权重无效")
    original = fk_numpy(raw_q)
    theta = np.deg2rad(float(yaw_deg)) * a
    rz = np.zeros((len(t), 3, 3))
    rz[:, 0, 0] = rz[:, 1, 1] = np.cos(theta)
    rz[:, 1, 0], rz[:, 0, 1], rz[:, 2, 2] = np.sin(theta), -np.sin(theta), 1
    center = np.r_[np.asarray(pivot_xy, float), 0.]
    target = original.copy()
    target[:, :3, 3] = center + a[:, None] * np.asarray(delta_m) + np.einsum("nij,nj->ni", rz, original[:, :3, 3] - center)
    target[:, :3, :3] = rz @ original[:, :3, :3]
    target = target @ tcp_matrix(cfg)
    windows = []
    policy = cfg["task"]
    if policy.get("min_tcp_z_m") is not None:
        if not 0 <= policy["clearance_window_s"][0] < policy["clearance_window_s"][1] <= t[-1]:
            raise ValueError("clearance_window_s 必须完整位于当前规划时间范围内")
    if event:
        close = float(event["time_s"])
        begin = float(stages["pick_complete_s"]) if stages else close - float(policy["close_window_before_s"])
        stable = float(cycle["closed_time_s"]) if cycle else close + float(policy["close_window_after_s"])
        windows.append({"kind": "pick", "start": max(0., begin), "end": min(t[-1], stable)})
    if stages:
        c = stages["cycle"]
        windows.append({"kind": "lift", "start": float(c["closed_time_s"]), "end": float(stages["return_start_s"])})
        release_end = float(c["opened_time_s"])
        if not math.isfinite(release_end):
            release_end = float(c["open_time_s"]) + float(policy["placement_after_s"])
        windows.append({"kind": "place", "start": float(stages["return_end_s"]), "end": min(t[-1], release_end)})
    elif cycle:
        windows.append({"kind": "place", "start": max(0., float(cycle["open_time_s"]) - float(policy["placement_before_s"])),
                        "end": min(t[-1], float(cycle["opened_time_s"]) if math.isfinite(float(cycle["opened_time_s"])) else float(cycle["open_time_s"]) + float(policy["placement_after_s"]))})
    guard = None
    if stages and policy["check_auto_lift"] and int(stages["lift_index"]) >= 0:
        guard = {"start": float(t[int(stages["lift_index"])]), "end": float(stages["return_start_s"]),
                 "closed": float(stages["cycle"]["closed_time_s"]), "height_m": float(stages["lift_mm"]) / 1000.}
    return {"time": t, "target": target, "raw_q": np.asarray(raw_q, float), "candidate_q": q,
            "windows": windows, "lift_guard": guard, "stages": stages}


def task_limits(ref, query, cfg):
    t = np.asarray(query)
    p = np.full(t.shape, float(cfg["task"]["free_position_mm"]) / 1000.)
    r = np.full(t.shape, np.deg2rad(float(cfg["task"]["free_orientation_deg"])))
    phase = np.zeros(t.shape, dtype=np.int32)
    for w in sorted(ref["windows"], key=lambda w: w["kind"] != "lift"):
        mask = (t >= w["start"] - 1e-10) & (t <= w["end"] + 1e-10)
        key = "lift" if w["kind"] == "lift" else "critical"
        p[mask] = float(cfg["task"][key + "_position_mm"]) / 1000.
        r[mask] = np.deg2rad(float(cfg["task"][key + "_orientation_deg"]))
        phase[mask] = {"pick": 1, "lift": 2, "place": 3}[w["kind"]]
    return p, r, phase


def target_at(ref, query):
    query = np.asarray(query, float)
    t, target = ref["time"], ref["target"]
    result = np.broadcast_to(np.eye(4), (len(query), 4, 4)).copy()
    for k in range(3):
        result[:, k, 3] = np.interp(query, t, target[:, k, 3])
    result[:, :3, :3] = Slerp(t, Rotation.from_matrix(target[:, :3, :3]))(query).as_matrix()
    return result


def spline_basis(t, spacing):
    degree = 5
    intervals = max(1, int(np.ceil((t[-1] - t[0]) / spacing)))
    grid = np.linspace(t[0], t[-1], intervals + 1)
    knots = np.r_[np.repeat(t[0], degree + 1), grid[1:-1], np.repeat(t[-1], degree + 1)]
    m = len(knots) - degree - 1
    identity = BSpline(knots, np.eye(m), degree)
    transforms = [csc_matrix(np.eye(m))]
    for order in (1, 2, 3):
        derivative = identity.derivative(order)
        count = len(derivative.t) - derivative.k - 1
        transforms.append(csc_matrix(derivative.c[:count]))
    return knots, degree, transforms


def basis_at(t, knots, degree):
    return csc_matrix(BSpline.design_matrix(t, knots, degree, extrapolate=False))


def seed_coefficients(ref, knots, degree):
    b = basis_at(ref["time"], knots, degree)
    c = np.column_stack([lsqr(b, ref["candidate_q"][:, j], damp=1e-7, atol=1e-9, btol=1e-9)[0] for j in range(6)])
    c[0], c[-1] = ref["candidate_q"][0], ref["candidate_q"][-1]
    return c


def sampled_metrics(q, time_axis):
    dt = float(np.diff(time_axis)[0])
    result = {}
    for order, name in ((1, "velocity"), (2, "acceleration"), (3, "jerk")):
        d = np.diff(q, n=order, axis=0) / dt**order
        result[name + "_peak"] = np.max(np.abs(d), axis=0).tolist()
        result[name + "_rms"] = np.sqrt(np.mean(d*d, axis=0)).tolist()
    return result


def validate_solution(ref, cfg, knots, degree, coefficients, solver_success=True):
    """Uses NumPy/SciPy only, including analytic spline derivatives and hulls."""
    t = ref["time"]
    steps = int(cfg["optimizer"]["validation_substeps"])
    query = np.unique(np.r_[t, np.concatenate([t[:-1] + (t[1:] - t[:-1]) * k / steps for k in range(1, steps)])])
    extra = [x for w in ref["windows"] for x in (w["start"], w["end"])]
    guard = ref["lift_guard"]
    if guard:
        extra += [guard["start"], guard["end"], guard["closed"]]
    if cfg["task"].get("min_tcp_z_m") is not None:
        extra += list(cfg["task"]["clearance_window_s"])
    query = np.unique(np.r_[query, np.clip(extra, t[0], t[-1])])
    curve = BSpline(knots, coefficients, degree)
    q, dense_q = curve(t), curve(query)
    if not np.isfinite(dense_q).all():
        return {"valid": False, "error": "优化结果出现非有限关节值"}, np.array([])
    target = target_at(ref, query)
    actual = fk_numpy(dense_q, tcp_matrix(cfg))
    ep = np.linalg.norm(actual[:, :3, 3] - target[:, :3, 3], axis=1)
    er = Rotation.from_matrix(np.swapaxes(target[:, :3, :3], 1, 2) @ actual[:, :3, :3]).magnitude()
    p_lim, r_lim, phase = task_limits(ref, query, cfg)
    failures = []
    # The documented tolerances below only absorb floating point/solver residuals.
    pos_over, rot_over = ep - p_lim, er - r_lim
    for name, over, tol in (("task_position_m", pos_over, 2e-6), ("task_orientation_rad", rot_over, 2e-6)):
        i = int(np.argmax(over))
        if over[i] > tol:
            failures.append({"constraint": name, "excess": float(over[i]), "time_s": float(query[i])})
    derivative_peaks, hull_peaks = {}, {}
    identity = BSpline(knots, np.eye(len(coefficients)), degree)
    for order, name, limits in ((0, "position", (cfg["limits"]["q_min"], cfg["limits"]["q_max"])),
                                (1, "velocity", cfg["limits"]["v_max"]),
                                (2, "acceleration", cfg["limits"]["a_max"]),
                                (3, "jerk", cfg["limits"].get("j_max"))):
        derivative = identity.derivative(order) if order else identity
        count = len(derivative.t) - derivative.k - 1
        cp = derivative.c[:count] @ coefficients
        hull_peaks[name] = np.max(np.abs(cp), axis=0).tolist()
        derivative_peaks[name] = np.max(np.abs(curve(query, nu=order)), axis=0).tolist()
        if limits is not None:
            lo, hi = (np.asarray(limits[0]), np.asarray(limits[1])) if order == 0 else (-np.asarray(limits), np.asarray(limits))
            violation = np.maximum(lo - cp, cp - hi)
            i, j = np.unravel_index(np.argmax(violation), violation.shape)
            if violation[i, j] > 1e-5:
                failures.append({"constraint": name + "_control_hull", "joint": int(j+1), "excess": float(violation[i, j])})
    dev = np.abs(q - ref["candidate_q"])
    if np.max(dev - np.asarray(cfg["optimizer"]["max_joint_deviation_rad"])) > 1e-5:
        failures.append({"constraint": "candidate_deviation", "excess": float(np.max(dev - np.asarray(cfg["optimizer"]["max_joint_deviation_rad"])) )})
    boundary_error = float(max(np.max(abs(q[0] - ref["candidate_q"][0])), np.max(abs(q[-1] - ref["candidate_q"][-1]))))
    if boundary_error > 1e-6:
        failures.append({"constraint": "endpoint_position", "excess": boundary_error})
    for key, order, time_value in (("start_velocity", 1, t[0]), ("end_velocity", 1, t[-1]),
                                   ("start_acceleration", 2, t[0]), ("end_acceleration", 2, t[-1])):
        if cfg["boundary"].get(key) is not None:
            error = float(np.max(abs(curve(time_value, nu=order) - np.asarray(cfg["boundary"][key]))))
            if error > 1e-5:
                failures.append({"constraint": key, "excess": error})
    lift_min = None
    if guard:
        mask = (query >= guard["start"] - 1e-10) & (query <= guard["end"] + 1e-10)
        baseline = fk_numpy(curve([guard["closed"]]))[0, 2, 3]
        lift_min = float(np.min(fk_numpy(dense_q[mask])[:, 2, 3] - baseline))
        if lift_min < guard["height_m"] - 2e-6:
            failures.append({"constraint": "tool0_sustained_lift", "excess": guard["height_m"] - lift_min})
    policy = cfg["task"]
    if policy.get("min_tcp_z_m") is not None:
        a, b = policy["clearance_window_s"]
        mask = (query >= a) & (query <= b)
        if not mask.any():
            failures.append({"constraint": "clearance_window", "error": "高度窗口未覆盖轨迹"})
        elif np.min(actual[mask, 2, 3]) < float(policy["min_tcp_z_m"]) - 2e-6:
            failures.append({"constraint": "min_tcp_z_m", "excess": float(policy["min_tcp_z_m"]) - float(np.min(actual[mask, 2, 3]))})
    if policy["freeze_tail"] and ref["stages"]:
        mask = t >= ref["stages"]["return_end_s"]
        if np.max(abs(q[mask] - ref["raw_q"][mask])) > 1e-6:
            failures.append({"constraint": "frozen_tail", "excess": float(np.max(abs(q[mask] - ref["raw_q"][mask])))})
    phases = {}
    for code, name in ((0, "free"), (1, "pick"), (2, "lift"), (3, "place")):
        mask = phase == code
        if mask.any():
            phases[name] = {"position_mm": float(np.max(ep[mask])*1000), "orientation_deg": float(np.rad2deg(np.max(er[mask])))}
    report = {"valid": bool(solver_success and not failures), "solver_success": bool(solver_success),
              "validation_scope": "continuous joint control-hulls; sampled nonlinear task poses and proxy clearance",
              "collision_checked": False, "dynamics_checked": False, "replay_ready": False,
              "limits_confirmed": cfg["limits_confirmed"], "task_tolerances_confirmed": cfg["task_tolerances_confirmed"],
              "failures": failures, "checked_task_samples": len(query), "phase_errors": phases,
              "max_position_mm": float(ep.max()*1000), "max_orientation_deg": float(np.rad2deg(er.max())),
              "max_candidate_deviation_rad": np.max(dev, axis=0).tolist(), "endpoint_error_rad": boundary_error,
              "tool0_lift_min_mm": None if lift_min is None else lift_min * 1000,
              "analytic_derivative_peaks": derivative_peaks, "control_hull_peaks": hull_peaks,
              "candidate_metrics": sampled_metrics(ref["candidate_q"], t), "optimized_metrics": sampled_metrics(q, t),
              "windows": ref["windows"], "numeric_allowance": {"position_m": 2e-6, "orientation_rad": 2e-6, "joint_hull": 1e-5}}
    return report, query[(pos_over > 2e-6) | (rot_over > 2e-6)]


def _casadi_fk(ca, cfg):
    q = ca.SX.sym("q", 6)
    t = ca.SX.eye(4)
    for i in range(6):
        c, s, c_a, s_a = ca.cos(q[i]), ca.sin(q[i]), float(np.cos(ALPHA[i])), float(np.sin(ALPHA[i]))
        m = ca.vertcat(ca.horzcat(c, -s*c_a, s*s_a, A[i]*c),
                       ca.horzcat(s, c*c_a, -c*s_a, A[i]*s),
                       ca.horzcat(0, s_a, c_a, D[i]), ca.horzcat(0, 0, 0, 1))
        t = ca.mtimes(t, m)
    tcp = ca.mtimes(t, ca.DM(tcp_matrix(cfg)))
    return ca.Function("task_fk", [q], [tcp[:3, 3], ca.reshape(tcp[:3, :3], 9, 1), t[2, 3]])


def optimize(ref, cfg, progress=None, cancel=None):
    cfg = validate_config(cfg)
    if cancel is not None and cancel.is_set():
        raise Cancelled("已取消")
    ends = ref["candidate_q"][[0, -1]]
    if np.any(ends < np.asarray(cfg["limits"]["q_min"]) + 1e-7) or np.any(ends > np.asarray(cfg["limits"]["q_max"]) - 1e-7):
        raise OptimizationError("固定的首/末关节点超出配置位置限值；请核对数据、边界与限值。",
                                {"valid": False, "solver_status": "preflight_endpoint_limits"})
    for key, cap in (("start_velocity", "v_max"), ("end_velocity", "v_max"),
                     ("start_acceleration", "a_max"), ("end_acceleration", "a_max")):
        if cfg["boundary"].get(key) is not None and np.any(abs(np.asarray(cfg["boundary"][key])) > np.asarray(cfg["limits"][cap]) * (1-2e-6)):
            raise OptimizationError(f"{key} 与 {cap} 硬限值冲突", {"valid": False, "solver_status": "preflight_boundary_limits"})
    try:
        import casadi as ca
    except ImportError as ex:
        raise OptimizationError("缺少 CasADi。请在当前 Python 环境运行：python -m pip install -r opt_deps.txt") from ex
    notify = progress or (lambda *args: None)
    started = time.monotonic()
    t = ref["time"]
    knots, degree, transforms = spline_basis(t, float(cfg["optimizer"]["knot_spacing_s"]))
    seed = seed_coefficients(ref, knots, degree)
    extra = [v for w in ref["windows"] for v in (w["start"], w["end"])]
    if ref["lift_guard"]:
        extra += list(ref["lift_guard"][k] for k in ("start", "end", "closed"))
    if cfg["task"].get("min_tcp_z_m") is not None:
        extra += list(cfg["task"]["clearance_window_s"])
    query = np.unique(np.r_[t, np.clip(extra, t[0], t[-1])])
    fk = _casadi_fk(ca, cfg)
    last_report = None
    for refinement in range(int(cfg["optimizer"]["refinement_rounds"]) + 1):
        if cancel is not None and cancel.is_set():
            raise Cancelled("已取消")
        notify("建立全轨迹稀疏 NLP", refinement, int(cfg["optimizer"]["refinement_rounds"]) + 1)
        op = ca.Opti()
        c = op.variable(6, len(seed))
        b = basis_at(t, knots, degree)
        q = ca.mtimes(c, ca.DM(b.T))
        lim, setting = cfg["limits"], cfg["optimizer"]
        def bound(expr, lo, hi):
            low = ca.repmat(ca.DM(np.asarray(lo).reshape(6, 1)), 1, expr.size2())
            high = ca.repmat(ca.DM(np.asarray(hi).reshape(6, 1)), 1, expr.size2())
            op.subject_to(op.bounded(ca.vec(low), ca.vec(expr), ca.vec(high)))
        bound(c, np.asarray(lim["q_min"]) + 1e-7, np.asarray(lim["q_max"]) - 1e-7)
        derivative_control = {}
        for order, name in ((1, "v_max"), (2, "a_max"), (3, "j_max")):
            dc = ca.mtimes(c, ca.DM(transforms[order].T))
            derivative_control[order] = dc
            if lim.get(name) is not None:
                cap = np.asarray(lim[name]) * (1 - 2e-6)
                bound(dc, -cap, cap)
        radius = np.asarray(setting["max_joint_deviation_rad"])
        bound(q - ca.DM(ref["candidate_q"].T), -radius, radius)
        op.subject_to(c[:, 0] == ref["candidate_q"][0])
        op.subject_to(c[:, -1] == ref["candidate_q"][-1])
        for key, order, column in (("start_velocity", 1, 0), ("end_velocity", 1, -1),
                                    ("start_acceleration", 2, 0), ("end_acceleration", 2, -1)):
            if cfg["boundary"].get(key) is not None:
                op.subject_to(derivative_control[order][:, column] == cfg["boundary"][key])
        qb = ca.mtimes(c, ca.DM(basis_at(query, knots, degree).T))
        pos, rot, ztool = fk.map(len(query))(qb)
        target = target_at(ref, query)
        p_limit, r_limit, _ = task_limits(ref, query, cfg)
        # Strict interior margins are smaller than the independent numeric allowance.
        p_limit = p_limit - 1e-6
        r_limit = r_limit - 1e-6
        pe = pos - ca.DM(target[:, :3, 3].T)
        r_target = np.column_stack([m.reshape(9, order="F") for m in target[:, :3, :3]])
        re = rot - ca.DM(r_target)
        op.subject_to(ca.vec(ca.sum1(pe**2) / ca.DM(p_limit[None, :]**2)) <= 1)
        # ||R-R_ref||_F² = 8 sin²(theta/2), exactly equivalent for theta in [0,pi].
        op.subject_to(ca.vec(ca.sum1(re**2) / ca.DM(8*np.sin(r_limit[None, :]/2)**2)) <= 1)
        guard = ref["lift_guard"]
        if guard:
            mask = np.flatnonzero((query >= guard["start"] - 1e-10) & (query <= guard["end"] + 1e-10))
            qc = ca.mtimes(c, ca.DM(basis_at([guard["closed"]], knots, degree).T))
            baseline = fk(qc)[2]
            op.subject_to(ca.vec((ztool[:, mask] - baseline) / guard["height_m"]) >= 1)
        if cfg["task"].get("min_tcp_z_m") is not None:
            a, end = cfg["task"]["clearance_window_s"]
            mask = np.flatnonzero((query >= a) & (query <= end))
            if not len(mask):
                raise OptimizationError("高度约束窗口没有覆盖当前轨迹")
            op.subject_to(ca.vec(pos[2, mask]) >= float(cfg["task"]["min_tcp_z_m"]))
        if cfg["task"]["freeze_tail"] and ref["stages"]:
            mask = np.flatnonzero(t >= ref["stages"]["return_end_s"])
            op.subject_to(ca.vec(q[:, mask]) == ca.vec(ca.DM(ref["raw_q"][mask].T)))
        weights = setting["weights"]
        cost = float(weights["reference"]) * ca.sumsqr((q - ca.DM(ref["candidate_q"].T)) / ca.repmat(ca.DM(radius), 1, len(t))) / (6*len(t))
        identity = BSpline(knots, np.eye(len(seed)), degree)
        for order, name, scale in ((1, "velocity", lim["v_max"]), (2, "acceleration", lim["a_max"]),
                                    (3, "jerk", lim.get("j_max") or setting["jerk_scale"])):
            d = identity.derivative(order)
            count = len(d.t) - d.k - 1
            db = csc_matrix(BSpline.design_matrix(t, d.t, d.k)) @ csc_matrix(d.c[:count])
            values = ca.mtimes(c, ca.DM(db.T)) / ca.repmat(ca.DM(scale), 1, len(t))
            cost += float(weights[name]) * ca.sumsqr(values) / (6*len(t))
        op.minimize(cost)
        op.set_initial(c, seed.T)
        class Iteration(ca.Callback):
            def __init__(self):
                ca.Callback.__init__(self)
                self.number = 0
                self.timed_out = False
                self.construct("joint_iteration", {})
            def get_n_in(self):
                return ca.nlpsol_n_out()
            def get_n_out(self):
                return 1
            def get_sparsity_in(self, i):
                name = ca.nlpsol_out(i)
                if name in ("x", "lam_x"):
                    return ca.Sparsity.dense(op.nx, 1)
                if name in ("g", "lam_g"):
                    return ca.Sparsity.dense(op.ng, 1)
                return ca.Sparsity.scalar() if name == "f" else ca.Sparsity(0, 1)
            def get_sparsity_out(self, _i):
                return ca.Sparsity.scalar()
            def eval(self, _args):
                self.number += 1
                if self.number == 1 or self.number % 5 == 0:
                    notify(f"关节 NLP 第 {refinement+1} 轮 / 迭代 {self.number}", self.number, int(setting["max_iterations"]))
                self.timed_out = time.monotonic() - started > float(setting["max_cpu_seconds"])
                return [int(self.timed_out or (cancel is not None and cancel.is_set()))]
        callback = Iteration()
        remaining = max(1., float(setting["max_cpu_seconds"]) - (time.monotonic() - started))
        op.solver("ipopt", {"print_time": False, "iteration_callback": callback},
                  {"print_level": 0, "sb": "yes", "max_iter": int(setting["max_iterations"]),
                   "max_cpu_time": remaining, "tol": 1e-6, "constr_viol_tol": 1e-7,
                   "acceptable_tol": 1e-5, "acceptable_constr_viol_tol": 1e-6,
                   "hessian_approximation": "limited-memory", "bound_relax_factor": 0.})
        try:
            solved = op.solve()
            stats = op.stats()
            coeff = np.asarray(solved.value(c)).T
            success = bool(stats.get("success"))
        except RuntimeError as ex:
            if cancel is not None and cancel.is_set():
                raise Cancelled("优化已取消") from ex
            stats = op.stats()
            try:
                coeff = np.asarray(op.debug.value(c)).T
                report, _ = validate_solution(ref, cfg, knots, degree, coeff, False)
            except Exception:
                report = {"valid": False, "failures": []}
            report.update(solver_status=str(stats.get("return_status", "solver_error")),
                          elapsed_seconds=time.monotonic()-started, error=str(ex)[-1800:])
            raise OptimizationError("约束优化未完成：" + report["solver_status"] + "。未自动退回候选；请查看诊断、时长、容差或样条密度。", report) from ex
        notify("独立复核关节与任务约束", 0, 1)
        report, bad_times = validate_solution(ref, cfg, knots, degree, coeff, success)
        report.update(solver="CasADi/IPOPT", casadi_version=ca.__version__, solver_status=str(stats.get("return_status")),
                      iterations=int(stats.get("iter_count", 0)), elapsed_seconds=time.monotonic()-started,
                      control_points=len(coeff), variables=len(coeff)*6, reference_samples=len(t),
                      refinement=refinement, objective=float(solved.value(cost)))
        last_report = report
        if report["valid"]:
            return {"q": BSpline(knots, coeff, degree)(t), "knots": knots, "degree": degree,
                    "coefficients": coeff, "report": report, "reference": ref}
        if len(bad_times) and refinement < int(setting["refinement_rounds"]):
            query = np.unique(np.r_[query, bad_times])
            seed = coeff
        else:
            break
    raise OptimizationError("独立复核未通过，保留候选用于分析；不会导出为优化成功轨迹。", last_report)


def run_job(job, output, cancel):
    """Spawn-process entry. Input bytes/config are frozen by the GUI."""
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[key] = "1"
    candidate = None
    def progress(stage, done, total):
        if cancel.is_set():
            raise Cancelled("已取消")
        output.put(("progress", stage, int(done), max(1, int(total))))
    try:
        import ur_pose_ui as base
        cfg = validate_config(job["config"])
        with tempfile.TemporaryDirectory(prefix="ur_opt_") as temporary:
            snapshots = {}
            hashes = {}
            for name, entry in job["files"].items():
                folder = Path(temporary) / name
                folder.mkdir()
                path = folder / Path(entry["path"]).name
                path.write_bytes(entry["bytes"])
                snapshots[name] = path
                hashes[name] = sha(entry["bytes"])
            geom = dict(job["geometry"])
            vision = None
            if job["vision"]:
                manifest = json.loads(job["manifest_bytes"].decode("utf-8-sig"))
                if hashes["source"] != manifest.get("npz_sha256") or hashes["reference"] != manifest.get("reference_sha256"):
                    raise ValueError("固定输入快照与任务包哈希不一致，请重新加载任务")
                vision = base.load_visual_pair(snapshots["reference"], snapshots["detection"], force_planar_z=job["ignore_z"])
                geom.update(xyz_cm=vision["offset_m"]*100, yaw_deg=vision["yaw_deg"],
                            pivot_mode="manual", pivot_xy_m=vision["reference_center_m"][:2])
            source = snapshots["source"]
            record = base.load_record(source)
            if np.any(np.diff(record["timestamp"]) < 0):
                raise ValueError("原始 timestamp 倒序，拒绝生成优化轨迹")
            candidate = base.generate_preview(source, progress=progress, **geom)
            if vision is not None:
                base.add_vision_metadata(candidate, vision, Path(job["task_path"]))
                vision["reference_path"] = Path(job["files"]["reference"]["path"])
                vision["detection_path"] = Path(job["files"]["detection"]["path"])
            candidate["source"] = Path(job["files"]["source"]["path"])
            stages = copy.deepcopy(candidate["staged_boundaries"])
            if stages:
                stages["lift_mm"] = geom["return_lift_mm"]
            cycles = base.detect_gripper_cycles(record)
            event = candidate["gripper_event"]
            cycle = next((c for c in cycles if event and c["close_index"] == event["index"]), None)
            ref = build_reference(record["ur_q"], candidate["npz_data"]["ur_q"], candidate["gripper_time"],
                                  candidate["alpha_curve"], candidate["xyz_cm"]/100, candidate["yaw_deg"],
                                  candidate["pivot_xy_m"], stages, event, cycle, cfg)
            if job["enable"]:
                solution = optimize(ref, cfg, progress, cancel)
                report = solution["report"]
                data = candidate["npz_data"]
                data["candidate_ur_q"] = data["ur_q"].copy()
                data["ur_q"] = solution["q"]
                data["ur_target_q"] = solution["q"].copy()
                data["opt_knots"] = solution["knots"]
                data["opt_coefficients"] = solution["coefficients"]
                data["opt_degree"] = np.array([solution["degree"]])
                data["opt_reference_T"] = ref["target"]
                pos_tol, rot_tol, phase = task_limits(ref, ref["time"], cfg)
                data["opt_position_tolerance_m"], data["opt_orientation_tolerance_rad"], data["opt_phase_id"] = pos_tol, rot_tol, phase
                data["max_fk_position_error_m"] = np.array([report["max_position_mm"] / 1000])
                data["max_fk_orientation_error_rad"] = np.deg2rad([report["max_orientation_deg"]])
                ids = np.linspace(0, len(solution["q"])-1, min(len(solution["q"]), 2400), dtype=int)
                candidate["candidate_xyz"] = candidate["new_xyz"].copy()
                candidate["new_xyz"] = base.sampled_xyz(solution["q"], ids)
                candidate["bounds"] = base.common_bounds(candidate["old_xyz"], candidate["new_xyz"])
                marker = int(event["index"]) if event else (int(candidate["detected_events"][0]["index"]) if candidate["detected_events"] else len(solution["q"])-1)
                h, _ = base.fk_jacobian(solution["q"][marker])
                candidate["orientation_markers"]["new"] = (h[:3, 3].copy(), h[:3, 0].copy())
                info = candidate["stats"]
                info.update(max_pos_error_mm=report["max_position_mm"], max_rot_error_rad=float(np.deg2rad(report["max_orientation_deg"])),
                            max_joint_jump_rad=float(np.max(abs(np.diff(solution["q"], axis=0)))),
                            max_joint_speed_rad_s=max(report["optimized_metrics"]["velocity_peak"]))
                if stages:
                    i = int(stages["cycle"]["open_index"])
                    raw, final = fk_numpy(record["ur_q"][[i]]), fk_numpy(solution["q"][[i]])
                    info.update(placement_fk_error_mm=float(np.linalg.norm(raw[0, :3, 3]-final[0, :3, 3])*1000),
                                placement_orientation_error_deg=float(np.rad2deg(Rotation.from_matrix(raw[0, :3, :3].T@final[0, :3, :3]).magnitude())),
                                placement_joint_error_rad=float(np.max(abs(record["ur_q"][i]-solution["q"][i]))))
                candidate["optimization_report"] = report
                candidate["optimization_config"] = cfg
            else:
                candidate["optimization_report"] = {"valid": False, "enabled": False, "status": "仅几何候选，未执行关节优化", "replay_ready": False}
            data = candidate["npz_data"]
            data["opt_enabled"] = np.array([int(job["enable"])])
            data["opt_validated"] = np.array([int(bool(candidate["optimization_report"].get("valid")))])
            data["opt_replay_ready"] = np.array([0])
            data["opt_report_json"] = np.array([json_text(candidate["optimization_report"])])
            data["opt_config_json"] = np.array([json_text(cfg)])
            data["opt_source_hashes_json"] = np.array([json_text(hashes)])
            data["opt_version"] = np.array([VERSION])
            candidate["job_snapshot"] = {"hashes": hashes, "task_path": job.get("task_path"), "config": cfg, "geometry": {k: np.asarray(v).tolist() if isinstance(v, np.ndarray) else v for k, v in geom.items()}}
            progress("完成独立复核", 1, 1)
            output.put(("success", candidate))
    except Cancelled:
        output.put(("cancelled", "已取消计算；没有保存轨迹"))
    except Exception as ex:
        output.put(("failed", str(ex), getattr(ex, "report", {"valid": False, "error": str(ex)}), candidate, traceback.format_exc()))
