"""Offline candidate / quintic least-squares fit / constrained NLP comparison.

Reuses the original geometry and NLP without modifying either original module.
No hardware communication. Baseline generation and constraint checks are separate.
"""
from __future__ import annotations

import copy
import csv
import datetime as dt
import io
import os
from pathlib import Path
import tempfile
import time
import traceback
import uuid

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
from scipy.interpolate import BSpline
from scipy.linalg import lstsq, null_space
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr
from scipy.spatial.transform import Rotation

import joint_opt as opt

METHODS = ("candidate", "fit", "optimize")
LABELS = {"candidate": "M3-w/oOpt", "fit": "M3-Fit", "optimize": "M3"}
TAGS = {"candidate": "cand", "fit": "fit", "optimize": "opt"}
COLORS = {"candidate": "#d28b19", "fit": "#2878b5", "optimize": "#22834f"}


def method_label(method, staged=True):
    return LABELS[method] if staged else {"candidate": "IK候选", "fit": "普通拟合", "optimize": "约束优化"}[method]


def plot_label(method, staged):
    return LABELS[method] if staged else {"candidate": "IK candidate", "fit": "Least-squares fit", "optimize": "Constrained optimization"}[method]


def check_cancel(cancel):
    if cancel is not None and cancel.is_set():
        raise opt.Cancelled("已取消计算；未保存输出")


def fit_spline(ref, cfg, progress=None, cancel=None):
    """Unweighted LS with exact common endpoint equalities; no inequality caps.

Only endpoint-supported columns enter the small equality null-space problem.
The remaining least-squares system stays sparse; no normal equations are formed.
"""
    check_cancel(cancel)
    started = time.perf_counter()
    knots, degree, _ = opt.spline_basis(ref["time"], cfg["optimizer"]["knot_spacing_s"])
    count = len(knots) - degree - 1
    identity = BSpline(knots, np.eye(count), degree)
    t = ref["time"]
    rows = [identity(t[0]), identity(t[-1])]
    values = [ref["candidate_q"][0], ref["candidate_q"][-1]]
    for key, order, when in (("start_velocity", 1, t[0]), ("end_velocity", 1, t[-1]),
                             ("start_acceleration", 2, t[0]), ("end_acceleration", 2, t[-1])):
        if cfg["boundary"].get(key) is not None:
            rows.append(identity(when, nu=order))
            values.append(cfg["boundary"][key])
    equality, target = np.asarray(rows), np.asarray(values)
    active = np.flatnonzero(np.any(equality != 0, axis=0))
    free = np.setdiff1d(np.arange(count), active)
    small = equality[:, active]
    particular, _, _, _ = lstsq(small, target, lapack_driver="gelsd")
    if np.max(abs(small @ particular - target)) > 1e-7:
        raise ValueError("样条端点等式不相容，请检查时长、节点与边界状态")
    null = null_space(small)
    transform = np.zeros((count, len(free) + null.shape[1]))
    transform[free, np.arange(len(free))] = 1.
    transform[active, len(free):] = null
    transform = csc_matrix(transform)
    coefficients = np.zeros((count, 6))
    coefficients[active] = particular
    basis = opt.basis_at(t, knots, degree)
    system = basis @ transform
    residual = ref["candidate_q"] - basis @ coefficients
    diagnostics = []
    if system.shape[1]:
        updates = []
        for j in range(6):
            check_cancel(cancel)
            if progress:
                progress("普通五次样条最小二乘拟合", j, 6)
            solved = lsqr(system, residual[:, j], damp=0., atol=1e-11, btol=1e-11,
                          conlim=1e12, iter_lim=max(200, 6 * system.shape[1]))
            if solved[1] not in (0, 1, 2, 4, 5):
                raise ValueError(f"关节 {j+1} 最小二乘未收敛，LSQR 状态 {solved[1]}")
            updates.append(solved[0])
            diagnostics.append({"joint": j+1, "stop": int(solved[1]), "iterations": int(solved[2])})
        coefficients += transform @ np.column_stack(updates)
    error = float(np.max(abs(equality @ coefficients - target)))
    q = BSpline(knots, coefficients, degree)(t)
    if not np.isfinite(q).all() or error > 1e-7:
        raise ValueError("拟合结果非有限，或未满足共同端点等式")
    check_cancel(cancel)
    return {"q": q, "knots": knots, "degree": degree, "coefficients": coefficients,
            "fit_seconds": time.perf_counter()-started,
            "fit_info": {"solver": "SciPy LSQR with endpoint elimination", "damp": 0.,
                         "endpoint_equality_error": error, "lsqr": diagnostics,
                         "objective_sse": float(np.sum((q-ref["candidate_q"])**2))}}


def query_grid(ref, cfg):
    t = ref["time"]
    steps = cfg["optimizer"]["validation_substeps"]
    parts = [t] + [t[:-1] + np.diff(t)*k/steps for k in range(1, steps)]
    extra = [x for w in ref["windows"] for x in (w["start"], w["end"])]
    if ref["lift_guard"]:
        extra += [ref["lift_guard"][k] for k in ("start", "end", "closed")]
    if cfg["task"].get("min_tcp_z_m") is not None:
        extra += list(cfg["task"]["clearance_window_s"])
    return np.unique(np.r_[np.concatenate(parts), np.clip(extra, t[0], t[-1])])


def evaluate(ref, cfg, variant):
    """Common differences + shared task grid; continuous joint hulls separate."""
    started = time.perf_counter()
    t, q = ref["time"], np.asarray(variant["q"])
    query = query_grid(ref, cfg)
    curve = BSpline(variant["knots"], variant["coefficients"], variant["degree"]) if "knots" in variant else None
    def at(x):
        return curve(x) if curve is not None else np.column_stack([np.interp(x, t, q[:, j]) for j in range(6)])
    dense_q = at(query)
    target = opt.target_at(ref, query)
    actual = opt.fk_numpy(dense_q, opt.tcp_matrix(cfg))
    ep = np.linalg.norm(actual[:, :3, 3]-target[:, :3, 3], axis=1)
    er = Rotation.from_matrix(np.swapaxes(target[:, :3, :3], 1, 2) @ actual[:, :3, :3]).magnitude()
    p_lim, r_lim, phase = opt.task_limits(ref, query, cfg)
    failures = []
    def check(name, excess, allowance, times=None):
        a = np.asarray(excess)
        index = np.unravel_index(np.argmax(a), a.shape)
        if a[index] > allowance:
            item = {"constraint": name, "excess": float(a[index])}
            if a.ndim == 2:
                item["joint"] = int(index[1]+1)
            if times is not None:
                item["time_s"] = float(times[index[0]])
            failures.append(item)
    check("task_position_m", ep-p_lim, 2e-6, query)
    check("task_orientation_rad", er-r_lim, 2e-6, query)
    lim = cfg["limits"]
    check("joint_position", np.maximum(np.asarray(lim["q_min"])-dense_q, dense_q-np.asarray(lim["q_max"])), 1e-5, query)
    metrics = opt.sampled_metrics(q, t)
    ratios = {}
    for order, name, key in ((1, "velocity", "v_max"), (2, "acceleration", "a_max"), (3, "jerk", "j_max")):
        d = np.diff(q, n=order, axis=0) / float(t[1]-t[0])**order
        cap = lim.get(key)
        ratios[name] = None if cap is None else (np.asarray(metrics[name+"_peak"])/cap).tolist()
        if cap is not None:
            check(name+"_sampled", abs(d)-np.asarray(cap), 1e-5, t[:len(d)]+order*(t[1]-t[0])/2)
    dev = abs(q-ref["candidate_q"])
    radius = np.asarray(cfg["optimizer"]["max_joint_deviation_rad"])
    check("candidate_deviation", dev-radius, 1e-5, t)
    endpoint_error = float(np.max(abs(q[[0, -1]]-ref["candidate_q"][[0, -1]])))
    check("endpoint_position", np.array([endpoint_error]), 1e-6)
    boundary_errors = {}
    for key, order, index in (("start_velocity", 1, 0), ("end_velocity", 1, -1),
                              ("start_acceleration", 2, 0), ("end_acceleration", 2, -1)):
        if cfg["boundary"].get(key) is not None:
            value = curve(t[index], nu=order) if curve is not None else (np.diff(q, n=order, axis=0)/(t[1]-t[0])**order)[index]
            boundary_errors[key] = float(np.max(abs(value-cfg["boundary"][key])))
            check(key, np.array([boundary_errors[key]]), 1e-5)
    lift_min = None
    if ref["lift_guard"]:
        guard = ref["lift_guard"]
        mask = (query >= guard["start"]-1e-10) & (query <= guard["end"]+1e-10)
        baseline = opt.fk_numpy(at([guard["closed"]]))[0, 2, 3]
        lift_min = float(np.min(opt.fk_numpy(dense_q[mask])[:, 2, 3]-baseline))
        check("tool0_sustained_lift", np.array([guard["height_m"]-lift_min]), 2e-6)
    if cfg["task"].get("min_tcp_z_m") is not None:
        a, b = cfg["task"]["clearance_window_s"]
        mask = (query >= a) & (query <= b)
        check("min_tcp_z_m", cfg["task"]["min_tcp_z_m"]-actual[mask, 2, 3], 2e-6, query[mask])
    if cfg["task"]["freeze_tail"] and ref["stages"]:
        mask = t >= ref["stages"]["return_end_s"]
        check("frozen_tail", abs(q[mask]-ref["raw_q"][mask]), 1e-6, t[mask])
    hull_failures, hull_peaks, analytic_peaks = [], {}, {}
    if curve is not None:
        for order, name, cap in ((0, "position", (lim["q_min"], lim["q_max"])),
                                  (1, "velocity", lim["v_max"]), (2, "acceleration", lim["a_max"]),
                                  (3, "jerk", lim.get("j_max"))):
            derivative = curve.derivative(order) if order else curve
            n = len(derivative.t)-derivative.k-1
            controls = derivative.c[:n]
            hull_peaks[name] = np.max(abs(controls), axis=0).tolist()
            analytic_peaks[name] = np.max(abs(derivative(query)), axis=0).tolist()
            if cap is not None:
                lo, hi = (np.asarray(cap[0]), np.asarray(cap[1])) if order == 0 else (-np.asarray(cap), np.asarray(cap))
                excess = np.maximum(lo-controls, controls-hi)
                if np.max(excess) > 1e-5:
                    hull_failures.append({"constraint": name+"_control_hull", "excess": float(np.max(excess))})
    jerk_scale = np.asarray(lim.get("j_max") or cfg["optimizer"]["jerk_scale"])
    jerk = np.diff(q, n=3, axis=0)/(t[1]-t[0])**3
    phases = {}
    for code, name in ((0, "free"), (1, "pick"), (2, "lift"), (3, "place")):
        mask = phase == code
        if mask.any():
            phases[name] = {"position_mm": float(ep[mask].max()*1000), "orientation_deg": float(np.rad2deg(er[mask].max())),
                            "position_ratio": float(np.max(ep[mask]/p_lim[mask])), "orientation_ratio": float(np.max(er[mask]/r_lim[mask]))}
    report = {"generated": True, "discrete_pass": not failures, "failures": failures,
              "hull_certified": None if curve is None else not hull_failures, "hull_failures": hull_failures,
              "metrics": metrics, "peak_ratios": ratios, "max_position_mm": float(ep.max()*1000),
              "max_orientation_deg": float(np.rad2deg(er.max())), "task_position_peak_ratio": float(np.max(ep/p_lim)),
              "task_orientation_peak_ratio": float(np.max(er/r_lim)), "phase_errors": phases,
              "max_candidate_deviation_rad": np.max(dev, axis=0).tolist(), "candidate_deviation_ratios": (np.max(dev, axis=0)/radius).tolist(),
              "endpoint_error_rad": endpoint_error, "boundary_errors": boundary_errors,
              "boundary_derivative_rule": "analytic spline" if curve is not None else "one-sided common differences",
              "tool0_lift_min_mm": None if lift_min is None else lift_min*1000,
              "jerk_normalized_rms": float(np.sqrt(np.mean((jerk/jerk_scale)**2))),
              "jerk_normalization": "j_max" if lim.get("j_max") is not None else "configured jerk_scale (not a hard limit)",
              "analytic_sampled_peaks": analytic_peaks, "control_hull_peaks": hull_peaks,
              "checked_task_samples": len(query), "evaluation_seconds": time.perf_counter()-started,
              "task_interpolation": "analytic spline" if curve is not None else "piecewise-linear joints for common task checks only",
              "replay_ready": False, "limits_confirmed": cfg["limits_confirmed"],
              "task_tolerances_confirmed": cfg["task_tolerances_confirmed"],
              "numeric_allowance": {"position_m": 2e-6, "orientation_rad": 2e-6, "joint": 1e-5}}
    arrays = {"check_time": query, "check_position_error_m": ep, "check_orientation_error_rad": er,
              "check_position_ratio": ep/p_lim, "check_orientation_ratio": er/r_lim, "check_phase_id": phase}
    return report, arrays


def run_job(job, output, cancel):
    """Windows-spawn entry; geometry is generated once using the original path."""
    def progress(stage, done=0, total=1):
        check_cancel(cancel)
        output.put(("progress", stage, done, total))
    try:
        check_cancel(cancel)
        method = job.get("method", "optimize" if job.get("enable", True) else "candidate")
        if method not in (*METHODS, "compare"):
            raise ValueError("未知关节处理方式")
        requested = list(METHODS) if method == "compare" else [method]
        terminal = []
        class CandidateSink:
            def put(self, event):
                if event[0] == "progress":
                    output.put(event)
                else:
                    terminal.append(event)
        prepared = dict(job, enable=False)
        started = time.perf_counter()
        opt.run_job(prepared, CandidateSink(), cancel)
        if not terminal or terminal[-1][0] != "success":
            if terminal:
                output.put(terminal[-1])
                return
            raise RuntimeError("共同候选生成进程未返回结果")
        result = terminal[-1][1]
        import ur_pose_ui as base
        with np.load(io.BytesIO(job["files"]["source"]["bytes"]), allow_pickle=False) as data:
            record = {k: data[k].copy() for k in data.files}
        cfg = opt.validate_config(job["config"])
        stages = copy.deepcopy(result["staged_boundaries"])
        if stages:
            stages["lift_mm"] = result["job_snapshot"]["geometry"]["return_lift_mm"]
        event = result["gripper_event"]
        cycle = next((c for c in base.detect_gripper_cycles(record) if event and c["close_index"] == event["index"]), None)
        ref = opt.build_reference(record["ur_q"], result["npz_data"]["ur_q"], result["gripper_time"],
                                  result["alpha_curve"], result["xyz_cm"]/100, result["yaw_deg"],
                                  result["pivot_xy_m"], stages, event, cycle, cfg)
        common_seconds = time.perf_counter()-started
        variants = {"candidate": {"q": ref["candidate_q"].copy(), "processing_seconds": 0.}}
        progress("共同候选评价")
        variants["candidate"]["report"], variants["candidate"]["evaluation_arrays"] = evaluate(ref, cfg, variants["candidate"])
        for name in requested:
            if name == "candidate":
                continue
            progress("普通样条拟合" if name == "fit" else "完整约束优化")
            before = time.perf_counter()
            try:
                solution = fit_spline(ref, cfg, progress, cancel) if name == "fit" else opt.optimize(ref, cfg, progress, cancel)
                processing_seconds = time.perf_counter()-before
                variant = {k: solution[k] for k in ("q", "knots", "coefficients", "degree")}
                variant["processing_seconds"] = processing_seconds
                variant["report"], variant["evaluation_arrays"] = evaluate(ref, cfg, variant)
                if name == "fit":
                    variant["report"]["fit_info"] = solution["fit_info"]
                else:
                    variant["report"]["solver_report"] = solution["report"]
                variants[name] = variant
            except opt.Cancelled:
                raise
            except Exception as ex:
                variants[name] = {"processing_seconds": time.perf_counter()-before,
                                  "report": {"generated": False, "error": str(ex), "replay_ready": False,
                                             "solver_report": getattr(ex, "report", None)}}
        check_cancel(cancel)
        case_id = job.get("case_id") or f"case_{dt.datetime.now():%Y%m%d_%H%M%S_%f}_{uuid.uuid4().hex[:6]}"
        result.update(variants=variants, requested_methods=requested, method=method, case_id=case_id,
                      common_seconds=common_seconds, run_wall_seconds=time.perf_counter()-started,
                      reference=ref, config=cfg, save_dir=None, saved_files={})
        result["job_snapshot"].update(method=method, case_id=case_id, version="ur_fit/1.0",
                                      requested_methods=requested)
        # Keep the original preview dictionary for inherited gripper and file UI.
        result["optimization_report"] = {"valid": False, "status": "分方法结果见 variants", "enabled": False}
        output.put(("success", result))
    except opt.Cancelled:
        output.put(("cancelled", "已取消计算；没有保存输出文件"))
    except Exception as ex:
        output.put(("failed", str(ex), {"error": str(ex)}, None, traceback.format_exc()))


def export_data(result, name):
    variant = result["variants"][name]
    if not variant["report"]["generated"]:
        raise ValueError("未生成的分支不能导出为轨迹")
    data = {k: np.array(v, copy=True) for k, v in result["npz_data"].items() if not k.startswith("opt_")}
    ref, cfg, report = result["reference"], result["config"], variant["report"]
    data.update(ur_q=variant["q"].copy(), ur_target_q=variant["q"].copy(), candidate_ur_q=ref["candidate_q"].copy(),
                trajectory_method=np.array([name]), paper_method=np.array([method_label(name, result["mode"] == "staged")]),
                case_id=np.array([result["case_id"]]), method_generated=np.array([1]),
                evaluation_discrete_pass=np.array([int(report["discrete_pass"])]),
                evaluation_hull_certificate=np.array([-1 if report["hull_certified"] is None else int(report["hull_certified"])]),
                evaluation_report_json=np.array([opt.json_text(report)]),
                reference_T=ref["target"], configuration_json=np.array([opt.json_text(cfg)]),
                source_hashes_json=np.array([opt.json_text(result["job_snapshot"]["hashes"])]),
                opt_enabled=np.array([int(name == "optimize")]),
                opt_validated=np.array([int(name == "optimize" and report.get("solver_report", {}).get("valid", False))]),
                opt_replay_ready=np.array([0]), fit_enabled=np.array([int(name == "fit")]))
    p, r, phase = opt.task_limits(ref, ref["time"], cfg)
    data.update(position_tolerance_m=p, orientation_tolerance_rad=r, phase_id=phase)
    data.update(variant["evaluation_arrays"])
    data["max_fk_position_error_m"] = np.array([report["max_position_mm"]/1000])
    data["max_fk_orientation_error_rad"] = np.deg2rad([report["max_orientation_deg"]])
    if "knots" in variant:
        data.update(spline_knots=variant["knots"], spline_coefficients=variant["coefficients"], spline_degree=np.array([variant["degree"]]))
    if name == "optimize":
        data.update(opt_knots=variant["knots"], opt_coefficients=variant["coefficients"], opt_degree=np.array([variant["degree"]]),
                    opt_reference_T=ref["target"], opt_position_tolerance_m=p, opt_orientation_tolerance_rad=r, opt_phase_id=phase,
                    opt_report_json=np.array([opt.json_text(report["solver_report"])]),
                    opt_config_json=np.array([opt.json_text(cfg)]), opt_source_hashes_json=data["source_hashes_json"], opt_version=np.array([opt.VERSION]))
    return data


def joint_figure(result, joint=0, analytic=False):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(10, 8), dpi=120)
    t, cfg = result["reference"]["time"], result["config"]
    for row, (order, label, cap) in enumerate(((0, "q (rad)", None), (1, "dq (rad/s)", "v_max"),
                                              (2, "ddq (rad/s2)", "a_max"), (3, "jerk (rad/s3)", "j_max")), 1):
        ax = fig.add_subplot(4, 1, row)
        for name, variant in result["variants"].items():
            if not variant["report"]["generated"] or (analytic and order and "knots" not in variant):
                continue
            if analytic and "knots" in variant:
                x, y = t, BSpline(variant["knots"], variant["coefficients"], variant["degree"])(t, nu=order)[:, joint]
            else:
                y = np.diff(variant["q"][:, joint], n=order)/(t[1]-t[0])**order
                x = t[:len(y)] + order*(t[1]-t[0])/2
            ax.plot(x, y, color=COLORS[name], linestyle={"candidate": ":", "fit": "--", "optimize": "-"}[name],
                    linewidth=1., label=plot_label(name, result["mode"] == "staged"))
        if cap and cfg["limits"].get(cap) is not None:
            for sign in (-1, 1):
                ax.axhline(sign*cfg["limits"][cap][joint], color="#888888", linestyle="--", linewidth=.7)
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
        if row == 1:
            ax.legend(ncol=3, fontsize=8)
        if row == 4:
            ax.set_xlabel("Planning time (s)")
    fig.suptitle(f"J{joint+1}: " + ("analytic splines only" if analytic else "common forward differences; midpoint times"), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .96))
    return fig


def trajectory_figure(result):
    from matplotlib.figure import Figure
    fig = Figure(figsize=(9, 7), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    ref, cfg = result["reference"], result["config"]
    ids = np.linspace(0, len(ref["time"])-1, min(1800, len(ref["time"])), dtype=int)
    points = []
    raw = opt.fk_numpy(ref["raw_q"][ids], opt.tcp_matrix(cfg))[:, :3, 3]
    ax.plot(*raw.T, color="#888888", linestyle="-.", linewidth=.9, label="Original demonstration")
    points.append(raw)
    for name, variant in result["variants"].items():
        if variant["report"]["generated"]:
            xyz = opt.fk_numpy(variant["q"][ids], opt.tcp_matrix(cfg))[:, :3, 3]
            ax.plot(*xyz.T, color=COLORS[name], linestyle={"candidate": ":", "fit": "--", "optimize": "-"}[name],
                    linewidth=1.1, label=plot_label(name, result["mode"] == "staged"))
            points.append(xyz)
    for w in ref["windows"]:
        if w["kind"] in ("pick", "place"):
            when = w["start"]
            if w["kind"] == "pick" and result.get("gripper_event"):
                when = result["gripper_event"]["time_s"]
            elif w["kind"] == "place" and ref["stages"]:
                when = ref["stages"]["cycle"]["open_time_s"]
            i = int(np.argmin(abs(ref["time"]-when)))
            p = ref["target"][i, :3, 3]
            ax.scatter(*p, marker="x", color="#333333", s=35)
            ax.text(*p, w["kind"]+" reference", fontsize=8)
    all_points = np.concatenate(points)
    center = (all_points.min(0)+all_points.max(0))/2
    radius = max(float(np.ptp(all_points, axis=0).max())*.55, .02)
    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)
    ax.set_box_aspect((1, 1, 1))
    ax.set(xlabel="Base X (m)", ylabel="Base Y (m)", zlabel="Base Z (m)", title="Nominal FK with configured TCP")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


def save_bundle(result, output_dir, save_npz=True, save_png=True):
    """Atomic per-file saves; immutable case folder and hashes permit safe retries."""
    if not (save_npz or save_png):
        return {"saved": [], "skipped": []}
    if result["save_dir"] is None:
        if Path(result["case_id"]).name != result["case_id"] or result["case_id"] in (".", ".."):
            raise ValueError("case_id 必须是单个目录名")
        directory = Path(output_dir)/result["case_id"]
        directory.mkdir(parents=True, exist_ok=False)
        result["save_dir"] = directory
    directory = Path(result["save_dir"])
    saved, skipped = [], []
    def write(name, writer):
        dest = directory/name
        if dest.exists():
            if result["saved_files"].get(name) == opt.sha(dest.read_bytes()):
                skipped.append(dest)
                return
            raise FileExistsError(f"文件已存在或被改动，不覆盖：{dest}")
        fd, temporary = tempfile.mkstemp(prefix=".fit_pending_", suffix=dest.suffix, dir=directory)
        os.close(fd)
        try:
            writer(Path(temporary))
            os.rename(temporary, dest)
            result["saved_files"][name] = opt.sha(dest.read_bytes())
            saved.append(dest)
        finally:
            Path(temporary).unlink(missing_ok=True)
    def text(name, value):
        write(name, lambda p: p.write_text(opt.json_text(value)+"\n", encoding="utf-8"))
    text("job.json", result["job_snapshot"])
    for name in result["requested_methods"]:
        variant = result["variants"][name]
        report = dict(variant["report"], processing_seconds=variant["processing_seconds"], common_seconds=result["common_seconds"])
        text(TAGS[name]+"_report.json", report)
        if save_npz and report["generated"]:
            write(TAGS[name]+".npz", lambda p, n=name: np.savez_compressed(p, **export_data(result, n)))
    def summary(path):
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            fields = ["case_id", "method", "generated", "discrete_pass", "hull_certified", "max_velocity_ratio",
                      "max_acceleration_ratio", "max_jerk_ratio", "jerk_normalized_rms", "task_position_peak_ratio",
                      "task_orientation_peak_ratio", "processing_seconds", "evaluation_seconds", "common_seconds", "total_offline_seconds", "run_wall_seconds", "error"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for name in result["requested_methods"]:
                v = result["variants"][name]
                r = v["report"]
                row = {"case_id": result["case_id"], "method": method_label(name, result["mode"] == "staged"),
                       "generated": r["generated"], "processing_seconds": v["processing_seconds"], "common_seconds": result["common_seconds"],
                       "run_wall_seconds": result["run_wall_seconds"], "error": r.get("error", "")}
                for key in ("discrete_pass", "hull_certified", "jerk_normalized_rms", "task_position_peak_ratio", "task_orientation_peak_ratio", "evaluation_seconds"):
                    row[key] = r.get(key)
                for key in ("velocity", "acceleration", "jerk"):
                    ratio = r.get("peak_ratios", {}).get(key)
                    row["max_"+key+"_ratio"] = max(ratio) if ratio is not None else None
                row["total_offline_seconds"] = result["common_seconds"]+v["processing_seconds"]+r.get("evaluation_seconds", 0.)
                writer.writerow(row)
    write("summary.csv", summary)
    if save_png:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        def image(path, fig):
            FigureCanvasAgg(fig).print_png(str(path))
            fig.clear()
        write("trajectory.png", lambda p: image(p, trajectory_figure(result)))
        for joint in range(6):
            write(f"joint_{joint+1}.png", lambda p, j=joint: image(p, joint_figure(result, j)))
    manifest = {"case_id": result["case_id"], "method": result["method"], "requested_methods": result["requested_methods"],
                "source_hashes": result["job_snapshot"]["hashes"], "replay_ready": False,
                "branches": {n: {"generated": result["variants"][n]["report"]["generated"],
                                  "discrete_pass": result["variants"][n]["report"].get("discrete_pass")}
                             for n in result["requested_methods"]}}
    text("manifest.json", manifest)
    return {"saved": saved, "skipped": skipped, "directory": directory}


def save_job(result, output_dir, save_npz, save_png, output):
    """Separate process: Agg export never collects the GUI's Tk image objects."""
    try:
        outcome = save_bundle(result, output_dir, save_npz, save_png)
        state = {"save_dir": result["save_dir"], "saved_files": result["saved_files"]}
        output.put(("saved", outcome, state))
    except Exception as ex:
        state = {"save_dir": result["save_dir"], "saved_files": result["saved_files"]}
        output.put(("save_error", str(ex), traceback.format_exc(), state))
