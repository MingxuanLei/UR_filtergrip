"""Offline staged compensation + constrained joint optimization GUI.

Run: python ur_opt.py. Keep ur_pose_ui.py and joint_opt.py beside this file.
"""
from __future__ import annotations

import copy
import datetime as dt
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import tempfile
import threading
import time
import traceback

# Set before importing NumPy/CasADi in the spawned worker as well.
for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np
import ur_pose_ui as base
import joint_opt as opt


class OptGui(base.ShiftGui):
    def __init__(self, root):
        self.process = None
        self.process_queue = None
        self.cancel_event = None
        self.cancel_started = None
        self._locked_states = []
        self._closing = False
        self._terminal_received = False
        self._dead_since = None
        self.config_path = Path(__file__).with_name("opt_cfg.json")
        self.config = opt.load_config(self.config_path)
        super().__init__(root)
        self.output_dir = self.record_dir / "opt_outputs"
        root.title("UR5e 分阶段补偿 · 关节约束优化")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self.run_button.configure(text="生成候选 → 关节优化 → 独立复核")
        self.result_label.set("固定时基、夹爪时序不变；新结果需手动保存。示例限值尚未确认。")
        watched = ("file_name", "dx", "dy", "dz", "yaw", "pivot_choice", "pivot_x", "pivot_y",
                   "input_mode", "ramp", "mode_label", "advance", "release_advance", "return_start_mode",
                   "return_lift_mm", "return_lift_wait", "return_manual_time", "event_label",
                   "visual_ref_path", "visual_detection_path", "visual_task_path", "visual_ignore_z", "opt_enable")
        for name in watched:
            getattr(self, name).trace_add("write", self.invalidate)
        self.log("新流程：几何候选 → 全轨迹关节 NLP → NumPy/SciPy 独立复核；无机器人连接。")

    def build_widgets(self):
        super().build_widgets()
        tk, ttk = self.tk, self.ttk
        self.save_button.master.configure(text="5 结果保存")
        for widget in self.settings_panel.master.master.winfo_children():
            if "text" in widget.keys() and str(widget.cget("text")).startswith("仅供离线实验"):
                widget.configure(text="离线数值优化：名义 DH；TCP 按配置。未包含碰撞、动力学及实机跟踪校验。")
        settings = self.mode_combo.master.master
        section = ttk.LabelFrame(settings, text="4 全轨迹关节空间优化", padding=10)
        section.pack(fill="x", pady=(10, 0))
        self.opt_enable = tk.BooleanVar(value=True)
        ttk.Checkbutton(section, text="启用 CasADi / IPOPT 约束优化", variable=self.opt_enable).pack(anchor="w")
        self.config_label = tk.StringVar(value=str(self.config_path))
        self._layout_label(section, textvariable=self.config_label, foreground="#416277").pack(fill="x", pady=5)
        row = ttk.Frame(section)
        row.pack(fill="x")
        ttk.Button(row, text="加载配置 JSON", command=self.choose_config).pack(side="left")
        ttk.Button(row, text="编辑限值 / 高级参数", command=self.edit_config).pack(side="left", padx=(6, 0))
        self.config_summary = tk.StringVar()
        self._layout_label(section, textvariable=self.config_summary, foreground="#9a571c").pack(fill="x", pady=(6, 0))
        self.update_config_summary()
        self._layout_label(section, text="关节位置/速度/加速度/可选 jerk 为硬约束；任务位姿按阶段设置容差。取消后不保存文件。",
                           foreground="#416277").pack(fill="x", pady=(5, 0))
        self.cancel_button = ttk.Button(self.run_button.master, text="取消计算", command=self.cancel, state="disabled")
        self.cancel_button.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.diag_tab = ttk.Frame(self.result_notebook)
        self.result_notebook.insert(2, self.diag_tab, text="优化诊断 / 关节曲线")
        self.diag_panes = ttk.Panedwindow(self.diag_tab, orient="vertical")
        self.diag_panes.pack(fill="both", expand=True)
        text_frame, self.diag_plot = ttk.Frame(self.diag_panes), ttk.Frame(self.diag_panes)
        self.diag_panes.add(text_frame, weight=2)
        self.diag_panes.add(self.diag_plot, weight=3)
        bar = ttk.Scrollbar(text_frame)
        bar.pack(side="right", fill="y")
        self.diag_text = tk.Text(text_frame, height=11, width=1, wrap="word", state="disabled", yscrollcommand=bar.set)
        self.diag_text.pack(fill="both", expand=True)
        bar.configure(command=self.diag_text.yview)
        self.diag_canvas = None
        self.set_diagnostics({"status": "尚未计算；示例配置只用于离线调试。"})

    def update_config_summary(self):
        c = self.config
        self.config_summary.set(
            f"限值已核对：{'是' if c['limits_confirmed'] else '否（示例值）'}；"
            f"任务容差已核对：{'是' if c['task_tolerances_confirmed'] else '否（示例值）'}\n"
            f"关键位姿：{c['task']['critical_position_mm']} mm / {c['task']['critical_orientation_deg']}°；"
            f"样条间隔：{c['optimizer']['knot_spacing_s']} s；计算预算：{c['optimizer']['max_cpu_seconds']} s\n"
            f"TCP xyz/rotvec：{c['model']['tcp_xyz_rotvec']}")

    def invalidate(self, *_):
        if self.busy or self.last_result is None:
            return
        self.last_result = None
        for widget in (self.save_button, self.save_npz_check, self.save_png_check):
            widget.configure(state="disabled")
        self.status.set("参数已变更，请重新生成")
        self.result_label.set("图中仍为上一次结果；当前参数尚未重新计算，保存已禁用。")

    def choose_config(self):
        from tkinter import filedialog, messagebox
        if self.busy:
            return
        path = filedialog.askopenfilename(parent=self.root, title="加载优化配置", filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            cfg = opt.load_config(path)
            self.invalidate()
            self.config, self.config_path = cfg, Path(path)
            self.config_label.set(str(self.config_path))
            self.update_config_summary()
        except Exception as ex:
            messagebox.showerror("配置无效", str(ex), parent=self.root)

    def edit_config(self):
        from tkinter import filedialog, messagebox
        if self.busy:
            return
        tk, ttk = self.tk, self.ttk
        win = tk.Toplevel(self.root)
        win.title("六轴限值与优化配置（rad / s）")
        win.geometry("880x690")
        win.minsize(740, 520)
        win.transient(self.root)
        win.grab_set()
        panel = ttk.Frame(win, padding=10)
        panel.pack(fill="both", expand=True)
        ttk.Label(panel, text="轴 1～6；位置 rad，速度 rad/s，加速度 rad/s²，jerk rad/s³。示例数值需要核对。",
                  wraplength=820).pack(anchor="w", pady=(0, 8))
        table = ttk.Frame(panel)
        table.pack(fill="x")
        keys = ("q_min", "q_max", "v_max", "a_max", "j_max")
        ttk.Label(table, text="关节").grid(row=0, column=0)
        fields = {}
        for col, key in enumerate(keys, 1):
            ttk.Label(table, text=key).grid(row=0, column=col)
            fields[key] = []
            table.columnconfigure(col, weight=1)
        for j in range(6):
            ttk.Label(table, text=str(j+1)).grid(row=j+1, column=0, padx=6)
            for col, key in enumerate(keys, 1):
                values = self.config["limits"].get(key) or [100]*6
                variable = tk.StringVar(value=str(values[j]))
                fields[key].append(variable)
                ttk.Entry(table, textvariable=variable, width=12).grid(row=j+1, column=col, sticky="ew", padx=3, pady=2)
        jerk_on = tk.BooleanVar(value=self.config["limits"].get("j_max") is not None)
        ttk.Checkbutton(panel, text="启用 jerk 硬限值（不勾选时保留 jerk 平滑目标）", variable=jerk_on).pack(anchor="w", pady=5)
        ttk.Label(panel, text="其余配置 JSON（限值由上表填写；确认标志、TCP、位姿容差、边界与权重在此编辑）：",
                  wraplength=820).pack(anchor="w", pady=(6, 3))
        editor_frame = ttk.Frame(panel)
        editor_frame.pack(fill="both", expand=True)
        bar = ttk.Scrollbar(editor_frame)
        bar.pack(side="right", fill="y")
        editor = tk.Text(editor_frame, wrap="none", undo=True, yscrollcommand=bar.set)
        editor.pack(fill="both", expand=True)
        bar.configure(command=editor.yview)
        rest = copy.deepcopy(self.config)
        del rest["limits"]
        editor.insert("1.0", opt.json_text(rest))
        def collect():
            value = json.loads(editor.get("1.0", "end"))
            value["limits"] = {key: [float(v.get()) for v in fields[key]] for key in keys}
            if not jerk_on.get():
                value["limits"]["j_max"] = None
            return opt.validate_config(value)
        def apply(save=False):
            try:
                cfg = collect()
                label = "内存配置（未写回原配置）"
                if save:
                    filename = filedialog.asksaveasfilename(parent=win, title="另存优化配置", defaultextension=".json",
                                                           initialfile="opt_custom.json", filetypes=[("JSON", "*.json")])
                    if not filename:
                        return
                    atomic_text(Path(filename), opt.json_text(cfg))
                    label = filename
                self.invalidate()
                self.config = cfg
                self.config_label.set(label)
                self.update_config_summary()
                win.destroy()
            except Exception as ex:
                messagebox.showerror("配置无效", str(ex), parent=win)
        buttons = ttk.Frame(panel)
        buttons.pack(fill="x", pady=(8, 0))
        ttk.Button(buttons, text="应用到本次会话", command=apply).pack(side="left")
        ttk.Button(buttons, text="应用并另存 JSON", command=lambda: apply(True)).pack(side="left", padx=6)
        ttk.Button(buttons, text="取消", command=win.destroy).pack(side="right")

    def snapshot_job(self):
        src = self.records.get(self.file_name.get())
        if not src:
            raise ValueError("请先选择 teach_records 中的示教 NPZ 或加载视觉任务包")
        mode = self.mode_labels.get(self.mode_label.get())
        if mode is None:
            raise ValueError("补偿方式无效")
        task = None
        if self.input_mode.get() == "vision":
            if self.visual_task is None:
                raise ValueError("视觉模式须先加载或创建任务包")
            task = base.load_visual_task(self.visual_task["path"])
            if Path(src).resolve() != task["npz"].resolve():
                raise ValueError("当前 NPZ 与视觉任务包不一致")
            self._show_visual(self._get_vision())
        geometry = {
            "xyz_cm": base.validate_xyz_offset_cm([float(self.dx.get()), float(self.dy.get()), float(self.dz.get())]),
            "ramp": float(self.ramp.get()), "mode": mode,
            "close_event_index": self.event_options.get(self.event_label.get()),
            "advance_seconds": float(self.advance.get()), "yaw_deg": float(self.yaw.get()),
            "pivot_mode": self.pivot_choice.get(),
            "pivot_xy_m": [float(self.pivot_x.get()), float(self.pivot_y.get())] if self.pivot_choice.get() == "manual" else None,
            "release_advance_seconds": float(self.release_advance.get()),
            "return_start_mode": self.return_start_mode.get(), "return_lift_mm": float(self.return_lift_mm.get()),
            "return_lift_wait_seconds": float(self.return_lift_wait.get()),
            "return_manual_time_s": float(self.return_manual_time.get()) if self.return_start_mode.get() == "manual" else None,
        }
        job = {"config": opt.validate_config(self.config), "enable": bool(self.opt_enable.get()), "geometry": geometry,
               "files": {}, "vision": self.input_mode.get() == "vision", "ignore_z": bool(self.visual_ignore_z.get()),
               "task_path": None, "manifest_bytes": None}
        if job["vision"]:
            job["task_path"] = str(task["path"])
            job["manifest_bytes"] = task["path"].read_bytes()
            sources = {"source": Path(src), "reference": task["reference"], "detection": Path(self.visual_detection_path.get())}
        else:
            sources = {"source": Path(src)}
        for name, path in sources.items():
            job["files"][name] = {"path": str(path.resolve()), "bytes": path.read_bytes()}
        # Validate scalar domains here; detailed timing/IK checks remain in base.
        if not -45 <= geometry["yaw_deg"] <= 45 or not np.isfinite(geometry["yaw_deg"]):
            raise ValueError("Yaw 必须为 ±45° 内的有限数字")
        if geometry["pivot_xy_m"] is not None and (not np.isfinite(geometry["pivot_xy_m"]).all() or np.max(abs(np.asarray(geometry["pivot_xy_m"]))) > 2):
            raise ValueError("物体旋转中心须在 ±2 m 内")
        if mode in ("gripper", "staged") and geometry["close_event_index"] is None:
            raise ValueError("未找到有效闭合事件")
        return job

    def set_locked(self):
        self._locked_states = []
        def visit(parent):
            for w in parent.winfo_children():
                if w not in (self.cancel_button, self.folder_button):
                    if "state" in w.keys() and w.winfo_class() in ("TEntry", "TSpinbox", "TCombobox", "TButton", "TCheckbutton", "TRadiobutton"):
                        self._locked_states.append((w, str(w.cget("state"))))
                        w.configure(state="disabled")
                    visit(w)
        visit(self.settings_panel)
        visit(self.visual_tab)

    def start(self):
        from tkinter import messagebox
        if self.busy or self._closing:
            return
        try:
            job = self.snapshot_job()
        except Exception as ex:
            messagebox.showerror("输入无效", str(ex), parent=self.root)
            return
        self.busy, self.last_result = True, None
        self.set_locked()
        self.cancel_button.configure(state="normal")
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start(16)
        self.status.set("正在生成候选轨迹…")
        self.result_label.set("输入文件和配置已固定为本次快照；暂不写入输出文件。")
        self._terminal_received, self._dead_since, self.cancel_started = False, None, None
        context = mp.get_context("spawn")
        self.process_queue, self.cancel_event = context.Queue(), context.Event()
        self.process = context.Process(target=opt.run_job, args=(job, self.process_queue, self.cancel_event), daemon=True)
        try:
            self.process.start()
        except Exception as ex:
            self.retire_process()
            self.finish_busy()
            messagebox.showerror("启动失败", str(ex), parent=self.root)
        self.log("开始新任务：固定时基 + 原始文件快照；优化=" + str(job["enable"]))

    def cancel(self):
        if self.process is not None and self.cancel_event is not None:
            self.cancel_event.set()
            self.cancel_started = time.monotonic()
            self.cancel_button.configure(state="disabled")
            self.status.set("正在取消，等待计算进程退出…")

    def retire_process(self):
        process, channel = self.process, self.process_queue
        self.process = self.process_queue = self.cancel_event = None
        if process is not None:
            def reap():
                if process.pid is not None:
                    process.join()
                process.close()
                if channel is not None:
                    channel.close()
                    channel.join_thread()
            threading.Thread(target=reap, daemon=True).start()

    def finish_busy(self):
        self.progress_bar.stop()
        self.progress_bar.configure(mode="determinate", value=0)
        for widget, state in self._locked_states:
            widget.configure(state=state)
        self._locked_states = []
        self.cancel_button.configure(state="disabled")
        self.run_button.configure(state="normal")
        self.on_input_mode_changed()
        self.on_mode_changed()
        self.busy = False
        if self.last_result is None:
            for widget in (self.save_button, self.save_npz_check, self.save_png_check):
                widget.configure(state="disabled")
        if self._closing:
            self.root.after_idle(self.destroy_ui)

    def enable_save_controls(self):
        r = self.last_result
        if r and not self.busy and (r["optimization_report"].get("valid") or r["optimization_report"].get("enabled") is False):
            for w in (self.save_button, self.save_npz_check, self.save_png_check):
                w.configure(state="normal")

    def start_save(self):
        from tkinter import messagebox
        if self.busy or self.last_result is None:
            return
        result = self.last_result
        if not (result["optimization_report"].get("valid") or result["optimization_report"].get("enabled") is False):
            return
        want_npz, want_png = bool(self.save_npz_var.get()), bool(self.save_png_var.get())
        if not (want_npz or want_png):
            messagebox.showwarning("未选择", "请勾选 NPZ 和/或 PNG", parent=self.root)
            return
        if result["base"] is None:
            tag = "opt" if result["optimization_report"].get("valid") else "cand"
            result["base"] = self.output_dir / f"{result['source'].stem}_{tag}_{dt.datetime.now():%Y%m%d_%H%M%S_%f}"
        self.busy = True
        self.set_locked()
        self.status.set("正在保存…")
        self.result_label.set("保存 NPZ/PNG，并附带本次诊断与输入参数快照。")
        def save():
            try:
                outcome = base.save_selected_outputs(result, self.output_dir, want_npz, want_png)
                # Sidecar can be regenerated after a partial save failure without overwriting.
                for suffix, content in (("_report.json", result["optimization_report"]), ("_job.json", result["job_snapshot"])):
                    path = result["base"].with_name(result["base"].name + suffix)
                    if not path.exists():
                        atomic_text(path, opt.json_text(content), replace=False)
                        outcome["saved"].append(path)
                self.events.put(("saved", outcome))
            except Exception as ex:
                self.events.put(("save_error", str(ex), traceback.format_exc()))
        threading.Thread(target=save, daemon=True).start()

    def poll_events(self):
        from tkinter import messagebox
        if self.process_queue is not None:
            try:
                while True:
                    event = self.process_queue.get_nowait()
                    self.events.put(event)
                    if event[0] in ("success", "failed", "cancelled"):
                        self._terminal_received = True
            except queue.Empty:
                pass
            if self.process is not None:
                if self.cancel_started and self.process.is_alive() and time.monotonic() - self.cancel_started > 3:
                    self.process.terminate()
                    self._terminal_received = True
                    self.events.put(("cancelled", "计算进程已取消，没有写入输出文件"))
                    self.retire_process()
                elif not self.process.is_alive() and not self._terminal_received:
                    self._dead_since = self._dead_since or time.monotonic()
                    if time.monotonic() - self._dead_since > .8:
                        self.events.put(("failed", f"计算进程退出，退出码 {self.process.exitcode}", {"valid": False}, None, ""))
                        self.retire_process()
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "progress":
                    self.status.set(f"{event[1]}（{event[2]}/{event[3]}）")
                elif kind == "success":
                    self.retire_process()
                    self.finish_busy()
                    if self.cancel_started is not None or self._closing:
                        self.status.set("计算已取消，没有保存输出文件")
                        continue
                    self.last_result = event[1]
                    report = self.last_result["optimization_report"]
                    self.status.set("约束复核通过（离线模型）" if report.get("valid") else "仅几何候选，未执行优化")
                    self.progress_bar["value"] = 100
                    self.result_label.set("结果尚未保存；原始 / 候选 / 优化曲线可在右侧对比。通过数值复核不代表具备实机回放条件。")
                    self.log("计算完成：" + opt.json_text(report))
                    self.set_diagnostics(report, self.last_result)
                    try:
                        self.show_preview(self.last_result)
                        self.show_gripper_plot(self.last_result)
                    except Exception as ex:
                        self.log("预览绘制失败：" + str(ex))
                    self.enable_save_controls()
                elif kind == "failed":
                    self.retire_process()
                    self.finish_busy()
                    self.last_result = None
                    self.status.set("未得到通过复核的优化轨迹")
                    self.result_label.set("保存已禁用；请查看诊断并调整配置。没有自动退回候选并标记为成功。")
                    self.log(event[1] + "\n" + event[4])
                    self.set_diagnostics(event[2], event[3])
                    self.result_notebook.select(self.diag_tab)
                    if not self._closing:
                        messagebox.showerror("优化失败", event[1], parent=self.root)
                elif kind == "cancelled":
                    self.retire_process()
                    self.finish_busy()
                    self.status.set(event[1])
                    self.log(event[1])
                elif kind in ("saved", "save_error"):
                    self.finish_busy()
                    self.enable_save_controls()
                    if kind == "saved":
                        self.status.set("保存完成")
                        self.result_label.set(f"输出目录：{self.output_dir}")
                        self.log("已保存：\n" + "\n".join(str(p) for p in event[1]["saved"]))
                        if not self._closing:
                            messagebox.showinfo("保存完成", "\n".join(str(p) for p in event[1]["saved"]) or "所选文件已保存，无需重复写入", parent=self.root)
                    else:
                        self.status.set("保存未全部完成，可重试")
                        self.log(event[1] + "\n" + event[2])
                        if not self._closing:
                            messagebox.showerror("保存失败", event[1], parent=self.root)
        except queue.Empty:
            pass
        if not self._closing or self.busy:
            self.root.after(120, self.poll_events)

    def set_diagnostics(self, report, result=None):
        self.diag_text.configure(state="normal")
        self.diag_text.delete("1.0", "end")
        summary = ""
        if "candidate_metrics" in report:
            lines = ["独立复核：" + ("通过" if report.get("valid") else "未通过"),
                     f"位置最大误差 {report['max_position_mm']:.4f} mm；姿态最大误差 {report['max_orientation_deg']:.4f}°",
                     "候选 → 优化（六轴最大值；同一固定时间网格差分）"]
            for key, unit in (("velocity_peak", "rad/s"), ("acceleration_peak", "rad/s²"), ("jerk_peak", "rad/s³")):
                lines.append(f"{key}: {max(report['candidate_metrics'][key]):.4g} → {max(report['optimized_metrics'][key]):.4g} {unit}")
            lines.append("关节硬约束复核采用解析导数及控制点包络，任务位姿另加采样间点检查。")
            summary = "\n".join(lines) + "\n\n"
        self.diag_text.insert("1.0", summary + opt.json_text(report))
        self.diag_text.configure(state="disabled")
        if self.diag_canvas:
            self.diag_canvas.get_tk_widget().destroy()
            self.diag_canvas = None
        if result is None:
            return
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        fig = Figure(figsize=(10, 6), dpi=90)
        data, t = result["npz_data"], result["gripper_time"]
        candidate = data.get("candidate_ur_q", data["ur_q"])
        colors = ("#2878b5", "#e69f00", "#229954", "#b455a0", "#dc5252", "#777777")
        for row, (order, label) in enumerate(((0, "q (rad)"), (1, "dq (rad/s)"), (2, "ddq (rad/s2)"), (3, "dddq (rad/s3)")), 1):
            ax = fig.add_subplot(4, 1, row)
            for j in range(6):
                before, after = np.diff(candidate[:, j], n=order), np.diff(data["ur_q"][:, j], n=order)
                scale = float(t[1]-t[0])**order
                ax.plot(t[order:], before/scale, color=colors[j], alpha=.25, linewidth=.6)
                ax.plot(t[order:], after/scale, color=colors[j], linewidth=.9, label=f"J{j+1}")
            ax.set_ylabel(label, fontsize=8)
            ax.grid(alpha=.2)
            if row == 1:
                ax.legend(ncol=6, fontsize=7, loc="upper right")
        ax.set_xlabel("time (s); pale=candidate, solid=final", fontsize=8)
        fig.tight_layout(pad=.8)
        self.diag_canvas = FigureCanvasTkAgg(fig, master=self.diag_plot)
        self.diag_canvas.draw()
        self.diag_canvas.get_tk_widget().pack(fill="both", expand=True)

    def show_preview(self, result):
        if "candidate_xyz" not in result:
            return super().show_preview(result)
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.preview_note.pack_forget()
        fig = Figure(figsize=(11, 5), dpi=95)
        trajectories = (result["old_xyz"], result["candidate_xyz"], result["new_xyz"])
        bounds = base.common_bounds(np.vstack(trajectories[:2]), trajectories[2])
        for i, (xyz, title) in enumerate(zip(trajectories, ("Original", "Geometric candidate", "Joint optimized")), 1):
            ax = fig.add_subplot(1, 3, i, projection="3d")
            base.draw_3d(ax, result["old_xyz"], xyz, bounds, compensated=True)
            ax.set_title(title, fontsize=10)
        fig.subplots_adjust(left=.01, right=.98, bottom=.08, top=.94, wspace=.05)
        self.canvas = FigureCanvasTkAgg(fig, master=self.preview_tab)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def close(self):
        if self._closing:
            return
        self._closing = True
        if self.busy:
            self.cancel()
            self.status.set("正在退出：等待取消计算或完成当前原子保存…")
        else:
            self.destroy_ui()

    def destroy_ui(self):
        # Cancel this Tk interpreter's scheduled callbacks before closing it.
        # This also avoids stale poll callbacks when embedded in another launcher.
        for token in self.root.tk.call("after", "info"):
            try:
                self.root.after_cancel(token)
            except self.tk.TclError:
                pass
        self.root.destroy()


def atomic_text(path, content, replace=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".opt_pending_", suffix=".json", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(content + "\n")
        if replace:
            os.replace(temporary, path)
        else:
            os.rename(temporary, path)  # Windows raises if the target already exists.
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    import tkinter as tk
    from tkinter import messagebox
    root = tk.Tk()
    try:
        OptGui(root)
    except Exception as ex:
        root.withdraw()
        messagebox.showerror("启动失败", f"{ex}\n请检查 opt_cfg.json 以及依赖文件是否齐全。", parent=root)
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    mp.freeze_support()
    main()
