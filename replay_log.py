"""Replay experiment logging; no Qt, robot connections, or motion commands.

ReplayLogger receives a callable returning JointSnapshot from the existing RTDE
receiver. A background thread samples its cache; after sampling stops, the same
thread exports states_raw.npz, states.npz, states.csv, commands.csv and meta.json.
qdd_est is an estimate from actual velocity, never a measured RTDE acceleration.
All host timestamps use time.perf_counter(). Units are seconds/rad/rad/s/rad/s^2.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import re
import threading
import time
from typing import Callable, Optional
import uuid

import numpy as np


@dataclass(frozen=True)
class ReplayLogConfig:
    output_dir: str
    condition_name: str = "case01"
    receive_frequency_hz: float = 50.0
    poll_frequency_hz: float = 100.0  # Cache polling, NOT RTDE subscription rate.
    stale_timeout_s: float = 1.0
    settle_s: float = 0.3
    smoothing_window_s: float = 0.1
    gap_factor: float = 3.0
    max_rows: int = 1_000_000

    def __post_init__(self):
        if not str(self.output_dir).strip() or not str(self.condition_name).strip():
            raise ValueError("请填写保存目录和工况名称")
        for name in ("receive_frequency_hz", "poll_frequency_hz", "stale_timeout_s", "gap_factor"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.receive_frequency_hz > 500 or self.poll_frequency_hz > 500:
            raise ValueError("记录配置频率不得大于 500 Hz")
        if self.gap_factor < 1 or self.stale_timeout_s > 60:
            raise ValueError("Invalid gap factor or stale timeout")
        for name in ("settle_s", "smoothing_window_s"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 2:
                raise ValueError(f"{name} must be between 0 and 2 seconds")
        if isinstance(self.max_rows, bool) or not isinstance(self.max_rows, int) or self.max_rows < 3:
            raise ValueError("max_rows must be an integer >= 3")


@dataclass(frozen=True)
class JointSnapshot:
    robot_time: float
    host_time: float
    q_actual: tuple
    qd_actual: tuple
    q_target: Optional[tuple] = None
    qd_target: Optional[tuple] = None
    qdd_target: Optional[tuple] = None
    speed_scaling: float = math.nan
    read_duration_s: float = 0.0


def _six(values, required=True):
    if values is None and not required:
        return (math.nan,) * 6
    result = tuple(float(x) for x in values)
    if len(result) != 6 or not all(math.isfinite(x) for x in result):
        if not required:
            return (math.nan,) * 6
        raise ValueError("六关节反馈必须包含 6 个有限数值")
    return result


def _safe_condition(name):
    result = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name)).strip(" .")[:64]
    if not result or result in (".", ".."):
        result = "case"
    if result.split(".")[0].upper() in {"CON", "PRN", "AUX", "NUL", *[f"COM{i}" for i in range(1, 10)], *[f"LPT{i}" for i in range(1, 10)]}:
        result = "case_" + result
    return result


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
    os.replace(temporary, path)


def _atomic_npz(path, values):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **values)
    os.replace(temporary, path)


def estimate_acceleration(t, velocity, phase, gap_seconds, window_s=0.1):
    """Differentiate actual velocity on its actual, possibly uneven time grid.

    No derivative crosses a phase change, non-finite row, or long sampling gap.
    Raw gradients use numpy's nonuniform-grid differences. Smoothed estimates
    are slopes of local quadratic fits to velocity. Segment edges remain NaN.
    """
    t = np.asarray(t, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    phase = np.asarray(phase)
    if t.ndim != 1 or velocity.shape != (len(t), 6) or phase.shape != t.shape:
        raise ValueError("Acceleration input shapes must be N, N x 6, N")
    raw = np.full_like(velocity, np.nan)
    smooth = np.full_like(velocity, np.nan)
    valid = np.zeros(len(t), dtype=bool)
    good = np.isfinite(t) & np.isfinite(velocity).all(axis=1)
    start = 0
    while start < len(t):
        if not good[start]:
            start += 1
            continue
        end = start + 1
        while (end < len(t) and good[end] and phase[end] == phase[end - 1]
               and 0 < t[end] - t[end - 1] <= gap_seconds):
            end += 1
        if end - start >= 3:
            ts, vs = t[start:end], velocity[start:end]
            gradients = np.gradient(vs, ts, axis=0, edge_order=2)
            raw[start + 1:end - 1] = gradients[1:-1]
            for i in range(1, len(ts) - 1):
                if window_s == 0:
                    smooth[start + i] = gradients[i]
                else:
                    left = min(i - 1, int(np.searchsorted(ts, ts[i] - window_s / 2)))
                    right = max(i + 2, int(np.searchsorted(ts, ts[i] + window_s / 2, side="right")))
                    x = ts[left:right] - ts[i]
                    # Fit in normalized time to avoid ill-conditioning at short intervals.
                    scale = max(float(np.max(np.abs(x))), 1e-12)
                    z = x / scale
                    design = np.column_stack((np.ones(len(z)), z, z * z))
                    coefficients = np.linalg.lstsq(design, vs[left:right], rcond=None)[0]
                    smooth[start + i] = coefficients[1] / scale
                valid[start + i] = True
        start = end
    return raw, smooth, valid


class ReplayLogger:
    """One experiment per instance. finish() never waits or sends robot commands."""

    def __init__(self, reader: Callable[[], Optional[JointSnapshot]], config: ReplayLogConfig,
                 metadata=None, notify: Optional[Callable[[str], None]] = None):
        self.reader, self.config = reader, config
        self.metadata = dict(metadata or {})
        self.notify = notify or (lambda message: None)
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._ready = threading.Event()
        self._sampling_done = threading.Event()
        self._done = threading.Event()
        self._thread = None
        self._state = "idle"
        self._path = None
        self._samples = []
        self._commands = []
        self._events = []
        self._errors = []
        self._duplicates = 0
        self._invalid = 0
        self._finish = None
        self._deadline = math.inf
        self._cutoff = math.inf
        self._origin = 0.0
        self._replay_start = None
        self._complete = False

    def _report(self, message):
        try:
            self.notify(message)
        except Exception:
            pass  # UI notifications must never interrupt sampling or saving.

    @property
    def busy(self):
        with self._lock:
            return self._state in ("starting", "recording", "saving")

    def status(self):
        with self._lock:
            return {"state": self._state, "samples": len(self._samples),
                    "commands": len(self._commands), "path": str(self._path or ""),
                    "condition": self.config.condition_name, "complete": self._complete,
                    "error": self._errors[-1] if self._errors else ""}

    def start(self):
        with self._lock:
            if self._state != "idle":
                raise RuntimeError("每个记录实例只允许启动一次")
            self._origin = time.perf_counter()
            self._state = "starting"
        try:
            root = Path(self.config.output_dir).expanduser().resolve()
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f") + "_" + uuid.uuid4().hex[:6]
            self._path = root / _safe_condition(self.config.condition_name) / run_id
            self._path.mkdir(parents=True, exist_ok=False)
            self.metadata.update({"schema_version": 1, "condition_name": self.config.condition_name,
                                  "created_at": datetime.now().astimezone().isoformat(),
                                  "units": {"time": "s", "position": "rad", "velocity": "rad/s", "acceleration": "rad/s^2"},
                                  "rtde_receive_frequency_hz": self.config.receive_frequency_hz,
                                  "cache_poll_frequency_hz": self.config.poll_frequency_hz,
                                  "acceleration": {"source": "estimated_from_actual_qd", "method": "local_quadratic_velocity_derivative",
                                                   "window_s": self.config.smoothing_window_s, "segment_edges": "NaN",
                                                   "gap_threshold_s": self.config.gap_factor / self.config.receive_frequency_hz},
                                  "host_clock": "time.perf_counter; host_elapsed_s is relative to record start",
                                  "synchronization": "software timestamps; feedback and commands have separate time grids"})
            _atomic_json(self._path / "meta.json", {**self.metadata, "status": "starting"})
            self.mark_phase("prepare_ur")
            self._thread = threading.Thread(target=self._run, name="replay_state_log", daemon=True)
            self._thread.start()
        except Exception as exc:
            with self._lock:
                self._errors.append(str(exc))
                self._state = "error"
            self._sampling_done.set()
            self._done.set()
            raise
        if not self._ready.wait(1.5):
            self.finish("error", "记录启动时没有取得有效 RTDE 反馈", immediate=True)
            raise RuntimeError("记录启动失败：没有有效 RTDE 反馈，请检查连接")
        with self._lock:
            if self._finish is not None or self._state != "recording":
                raise RuntimeError(self._errors[-1] if self._errors else "记录启动已中止")
        self._report(f"[RUNLOG] 开始记录 {self.config.condition_name} -> {self._path}")

    def mark_phase(self, phase, host_time=None):
        host_time = time.perf_counter() if host_time is None else float(host_time)
        with self._lock:
            if self._finish is not None:
                return
            self._events.append({"phase": str(phase), "host_elapsed_s": host_time - self._origin})
            if phase == "replay":
                self._replay_start = host_time

    def append_command(self, index, q, planned_time, sent_at, returned_at, success):
        with self._lock:
            if self._finish is not None or self._state not in ("recording", "starting"):
                return
            if len(self._commands) >= self.config.max_rows:
                self._fail("指令记录超过内存行数限制")
                return
            self._commands.append((int(index), float(planned_time), float(sent_at),
                                   float(returned_at), tuple(q), bool(success)))

    def finish(self, reason="completed", detail="", *, tail_s=0.0, cutoff_host=None, immediate=False):
        """Request closure; the first reason wins. Cleanup may shorten a tail."""
        now = time.perf_counter()
        with self._lock:
            if self._state not in ("starting", "recording"):
                return
            if self._finish is None:
                self._finish = (str(reason), str(detail))
                self._deadline = now + max(0.0, float(tail_s))
                self._events.append({"phase": "settling" if tail_s > 0 else str(reason),
                                     "host_elapsed_s": now - self._origin})
            if cutoff_host is not None:
                self._cutoff = min(self._cutoff, float(cutoff_host))
                self._deadline = min(self._deadline, now)
            if immediate:
                self._deadline = min(self._deadline, now)
            self._wake.set()

    def wait_for_sampling(self, timeout=3.0):
        return self._sampling_done.wait(timeout)

    def wait_for_save(self, timeout=30.0):
        return self._done.wait(timeout)

    def _fail(self, message):
        with self._lock:
            if message not in self._errors:
                self._errors.append(message)
        self._report("[RUNLOG][ERR] " + message)
        self.finish("error", message, immediate=True)

    def _run(self):
        last_stamp = None
        last_fresh = time.perf_counter()
        try:
            while True:
                with self._lock:
                    if self._finish is not None and time.perf_counter() >= self._deadline:
                        break
                try:
                    snapshot = self.reader()
                    if snapshot is None:
                        with self._lock:
                            self._invalid += 1
                    else:
                        stamp = float(snapshot.robot_time)
                        host = float(snapshot.host_time)
                        duration = float(snapshot.read_duration_s)
                        q, qd = _six(snapshot.q_actual), _six(snapshot.qd_actual)
                        if not all(math.isfinite(x) for x in (stamp, host, duration)) or duration < 0:
                            raise ValueError("无效反馈时间戳")
                        if last_stamp is not None and stamp < last_stamp:
                            raise ConnectionError("机器人反馈时间戳回退，已结束本次记录")
                        if last_stamp == stamp:
                            with self._lock:
                                self._duplicates += 1
                        else:
                            targets = tuple(_six(getattr(snapshot, name), required=False)
                                            for name in ("q_target", "qd_target", "qdd_target"))
                            row = (stamp, host, duration, q, qd, *targets, float(snapshot.speed_scaling))
                            with self._lock:
                                if host + duration / 2 <= self._cutoff:
                                    if len(self._samples) >= self.config.max_rows:
                                        raise ConnectionError("反馈记录超过内存行数限制")
                                    self._samples.append(row)
                                    self._state = "recording"
                            last_stamp, last_fresh = stamp, time.perf_counter()
                            self._ready.set()
                except ConnectionError as exc:
                    self._fail(str(exc))
                    break
                except Exception as exc:
                    with self._lock:
                        self._invalid += 1
                    # Transient read errors may recover, but stale data is never
                    # duplicated to create a fictitious uniform 50 Hz record.
                    if time.perf_counter() - last_fresh >= self.config.stale_timeout_s:
                        self._fail("持续无法取得有效反馈: " + str(exc))
                        break
                if time.perf_counter() - last_fresh >= self.config.stale_timeout_s:
                    self._fail("RTDE 反馈时间戳停止更新")
                    break
                self._wake.wait(1.0 / self.config.poll_frequency_hz)
                self._wake.clear()
        except Exception as exc:
            self._fail("采集线程异常: " + str(exc))
        finally:
            with self._lock:
                self._state = "saving"
            self._ready.set()
            # From this point forward there are no more reader/RTDE calls.
            self._sampling_done.set()
            self._save()

    def _arrays(self):
        with self._lock:
            samples = [row for row in self._samples if row[1] + row[2] / 2 <= self._cutoff]
            events = sorted((dict(event) for event in self._events), key=lambda e: e["host_elapsed_s"])
            commands = list(self._commands)
        n = len(samples)
        robot_t = np.array([row[0] for row in samples], dtype=float)
        host_t = np.array([row[1] - self._origin for row in samples], dtype=float)
        phase = np.full(n, "prepare_ur", dtype="U24")
        for event in events:
            phase[host_t >= event["host_elapsed_s"]] = event["phase"]
        arrays = {"robot_timestamp": robot_t, "host_elapsed_s": host_t, "phase": phase,
                  "replay_time_s": np.full(n, np.nan) if self._replay_start is None else host_t - (self._replay_start - self._origin),
                  "read_duration_s": np.array([row[2] for row in samples], dtype=float),
                  "speed_scaling": np.array([row[8] for row in samples], dtype=float)}
        for offset, name in enumerate(("q_actual", "qd_actual", "q_target", "qd_target", "qdd_target"), 3):
            arrays[name] = np.array([row[offset] for row in samples], dtype=float).reshape(n, 6)
        gap = self.config.gap_factor / self.config.receive_frequency_hz
        arrays["gap_before"] = np.r_[False, np.diff(robot_t) > gap] if n else np.empty(0, dtype=bool)
        arrays.update({"command_index": np.array([r[0] for r in commands], dtype=np.int64),
                       "command_planned_time_s": np.array([r[1] for r in commands], dtype=float),
                       "command_host_send_s": np.array([r[2] - self._origin for r in commands], dtype=float),
                       "command_host_return_s": np.array([r[3] - self._origin for r in commands], dtype=float),
                       "q_cmd": np.array([r[4] for r in commands], dtype=float).reshape(len(commands), 6),
                       "command_success": np.array([r[5] for r in commands], dtype=bool)})
        return arrays, events

    def _write_csv(self, arrays):
        path = self._path / "states.csv"
        temporary = path.with_suffix(".csv.tmp")
        names = ("q_actual", "qd_actual", "qdd_est_raw", "qdd_est", "q_target", "qd_target", "qdd_target")
        units = ("rad", "rad_s", "rad_s2", "rad_s2", "rad", "rad_s", "rad_s2")
        with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["robot_timestamp_s", "host_elapsed_s", "replay_time_s", "phase", "acc_valid", "gap_before", "speed_scaling", "read_duration_s"]
                            + [f"{name}_J{j+1}_{unit}" for name, unit in zip(names, units) for j in range(6)])
            for i in range(len(arrays["robot_timestamp"])):
                writer.writerow([arrays["robot_timestamp"][i], arrays["host_elapsed_s"][i], arrays["replay_time_s"][i], arrays["phase"][i],
                                 int(arrays["acc_valid"][i]), int(arrays["gap_before"][i]), arrays["speed_scaling"][i], arrays["read_duration_s"][i]]
                                + [x for name in names for x in arrays[name][i]])
        os.replace(temporary, path)
        path = self._path / "commands.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", newline="", encoding="utf-8-sig") as stream:
            writer = csv.writer(stream)
            writer.writerow(["frame_index", "planned_time_s", "host_send_s", "host_return_s", "success"] + [f"q_cmd_J{j+1}_rad" for j in range(6)])
            for i in range(len(arrays["command_index"])):
                writer.writerow([arrays["command_index"][i], arrays["command_planned_time_s"][i], arrays["command_host_send_s"][i],
                                 arrays["command_host_return_s"][i], int(arrays["command_success"][i]), *arrays["q_cmd"][i]])
        os.replace(temporary, path)

    def _save(self):
        saved = False
        metadata = dict(self.metadata)
        try:
            arrays, events = self._arrays()
            reason, detail = self._finish or ("error", "采集意外退出")
            metadata.update({"end_reason": reason, "end_detail": detail, "events": events,
                             "sample_count": len(arrays["robot_timestamp"]),
                             "command_count": len(arrays["command_index"])})
            _atomic_npz(self._path / "states_raw.npz", arrays)
            raw, smooth, valid = estimate_acceleration(arrays["robot_timestamp"], arrays["qd_actual"], arrays["phase"],
                                                       self.config.gap_factor / self.config.receive_frequency_hz, self.config.smoothing_window_s)
            arrays.update(qdd_est_raw=raw, qdd_est=smooth, acc_valid=valid)
            _atomic_npz(self._path / "states.npz", arrays)
            self._write_csv(arrays)
            robot_t = arrays["robot_timestamp"]
            dt = np.diff(robot_t)
            replay_rows = int(np.count_nonzero(arrays["phase"] == "replay"))
            with self._lock:
                self._complete = (reason == "completed" and not self._errors and replay_rows >= 3
                                  and not np.any(arrays["gap_before"]))
            metadata.update({"status": "saved", "end_reason": reason, "end_detail": detail,
                             "complete": self._complete, "events": events,
                             "sample_count": len(robot_t), "replay_sample_count": replay_rows,
                             "command_count": len(arrays["command_index"]),
                             "effective_sample_hz": float(1 / np.mean(dt)) if len(dt) else None,
                             "max_sample_interval_s": float(np.max(dt)) if len(dt) else None,
                             "sampling_gap_count": int(np.count_nonzero(arrays["gap_before"])),
                             "acceleration_valid_rows": int(np.count_nonzero(valid)),
                             "duplicate_cache_reads": self._duplicates, "invalid_reads": self._invalid})
            saved = True
        except Exception as exc:
            with self._lock:
                self._complete = False
                self._errors.append("数据导出失败: " + str(exc))
            metadata.update(status="error", complete=False, end_reason=(self._finish or ("error", ""))[0])
            self._report("[RUNLOG][ERR] 数据导出失败，已写出的原始文件保留: " + str(exc))
        finally:
            metadata["errors"] = list(self._errors)
            metadata["finished_at"] = datetime.now().astimezone().isoformat()
            try:
                _atomic_json(self._path / "meta.json", metadata)
            except Exception as exc:
                saved = False
                with self._lock:
                    self._complete = False
                    self._errors.append("meta.json 保存失败: " + str(exc))
                self._report("[RUNLOG][ERR] " + self._errors[-1])
            with self._lock:
                self._state = "saved" if saved else "error"
            self._done.set()
            if saved:
                self._report(f"[RUNLOG] {'完整' if self._complete else '不完整'}记录已保存 -> {self._path}")
