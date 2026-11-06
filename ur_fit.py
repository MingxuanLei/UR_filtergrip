"""Run: python ur_fit.py. Original ur_opt.py and joint_opt.py stay unchanged."""
from __future__ import annotations

import multiprocessing as mp
import queue
import time
import tkinter as tk
from tkinter import messagebox

import fit_core as core
import joint_opt as opt
import ur_opt as legacy


class FitGui(legacy.OptGui):
    def __init__(self, root):
        self._saving = False
        self.method_var = tk.StringVar(master=root, value="optimize")
        self.joint_var = tk.StringVar(master=root, value="J1")
        self.derivative_var = tk.StringVar(master=root, value="共同差分")
        super().__init__(root)
        self.output_dir = self.record_dir / "fit_outputs"
        root.title("UR5e 分阶段补偿 · M3-Fit 与约束优化")
        self.method_var.trace_add("write", self.on_method_changed)
        for variable in (self.joint_var, self.derivative_var):
            variable.trace_add("write", self.redraw_results)
        # Both initial label and stage widgets are set, not just the dropdown text.
        self.mode_label.set(next(label for label, value in self.mode_labels.items() if value == "staged"))
        self.on_mode_changed()
        self.on_method_changed()
        self.result_label.set("默认分阶段补偿；可选择候选、M3-Fit、M3 或同工况三方法对照。结果需手动保存。")
        self.log("新入口 ur_fit.py：普通拟合不依赖 CasADi；三方法共用候选、时基与评价标准。")

    def build_widgets(self):
        super().build_widgets()
        ttk = self.ttk
        settings = self.mode_combo.master.master
        section = next(w for w in settings.winfo_children()
                       if "text" in w.keys() and str(w.cget("text")).startswith("4 全轨迹"))
        # Original auto-wrap callbacks belong to labels being replaced below.
        section.unbind("<Configure>")
        for child in section.winfo_children():
            child.destroy()
        section.configure(text="4 关节轨迹处理与对照")
        self.method_buttons = {}
        for value in (*core.METHODS, "compare"):
            button = ttk.Radiobutton(section, variable=self.method_var, value=value)
            button.pack(anchor="w", pady=3)
            self.method_buttons[value] = button
        self.method_note = tk.StringVar()
        self._layout_label(section, textvariable=self.method_note, foreground="#416277").pack(fill="x", pady=5)
        self.spline_frame = ttk.Frame(section)
        self.spline_frame.pack(fill="x", pady=4)
        ttk.Label(self.spline_frame, text="共同样条：五次；节点间距（s）").pack(side="left")
        self.spacing_var = tk.StringVar(value=str(self.config["optimizer"]["knot_spacing_s"]))
        self.spacing_entry = ttk.Entry(self.spline_frame, textvariable=self.spacing_var, width=8)
        self.spacing_entry.pack(side="left", padx=5)
        self.spacing_var.trace_add("write", self.on_spacing_changed)
        self._layout_label(section, text="拟合与优化共用端点状态；候选、拟合结果即使超限也保留用于分析。",
                           foreground="#416277").pack(fill="x", pady=4)
        self._layout_label(section, textvariable=self.config_label, foreground="#416277").pack(fill="x", pady=4)
        row = ttk.Frame(section)
        row.pack(fill="x")
        ttk.Button(row, text="加载共同配置 JSON", command=self.choose_config).pack(side="left")
        ttk.Button(row, text="编辑限值 / 高级参数", command=self.edit_config).pack(side="left", padx=5)
        self._layout_label(section, textvariable=self.config_summary, foreground="#9a571c").pack(fill="x", pady=5)
        self.result_notebook.tab(self.diag_tab, text="关节曲线 / 诊断")
        toolbar = ttk.Frame(self.diag_plot)
        toolbar.pack(fill="x", padx=5, pady=4)
        ttk.Label(toolbar, text="关节").pack(side="left")
        ttk.Combobox(toolbar, textvariable=self.joint_var, values=[f"J{i}" for i in range(1, 7)],
                     state="readonly", width=5).pack(side="left", padx=5)
        ttk.Combobox(toolbar, textvariable=self.derivative_var, values=["共同差分", "样条解析导数"],
                     state="readonly", width=16).pack(side="left", padx=5)
        ttk.Label(toolbar, text="解析导数页不对离散候选虚构连续导数").pack(side="left", padx=5)
        self.check_tab = ttk.Frame(self.result_notebook)
        self.result_notebook.insert(3, self.check_tab, text="共同约束评价")
        columns = ("method", "generated", "discrete", "hull", "velocity", "acceleration", "jerk")
        self.check_tree = ttk.Treeview(self.check_tab, columns=columns, show="headings", height=5)
        for col, title in zip(columns, ("方法", "生成状态", "共同离散检查", "关节包络证书", "速度峰值比", "加速度峰值比", "jerk峰值比")):
            self.check_tree.heading(col, text=title)
            self.check_tree.column(col, width=105, minwidth=85, anchor="center")
        self.check_tree.pack(fill="x", padx=8, pady=8)
        scroll = ttk.Scrollbar(self.check_tab, orient="horizontal", command=self.check_tree.xview)
        scroll.pack(fill="x", padx=8)
        self.check_tree.configure(xscrollcommand=scroll.set)
        self.check_note = tk.StringVar(value="尚未计算。生成成功与检查通过分开显示；未获得包络证书不直接代表物理不可行。")
        self._layout_label(self.check_tab, textvariable=self.check_note).pack(fill="x", padx=8, pady=12)
        self.check_tree.bind("<<TreeviewSelect>>", self.show_check_details)
        self.save_png_check.configure(text="保存对照 PNG（轨迹 + 六轴曲线）")

    def on_spacing_changed(self, *_):
        self.invalidate()
        try:
            value = float(self.spacing_var.get())
        except ValueError:
            return
        if .04 <= value <= 5:
            self.config["optimizer"]["knot_spacing_s"] = value
            self._last_config_spacing = str(value)
            legacy.OptGui.update_config_summary(self)

    def on_mode_changed(self):
        super().on_mode_changed()
        if hasattr(self, "method_buttons"):
            self.refresh_method_labels()

    def refresh_method_labels(self):
        staged = self.mode_labels.get(self.mode_label.get()) == "staged"
        for name in core.METHODS:
            text = {"candidate": "仅候选", "fit": "普通五次样条拟合", "optimize": "完整约束优化"}[name]
            self.method_buttons[name].configure(text=text+" · "+core.method_label(name, staged))
        self.method_buttons["compare"].configure(text="三方法对照：共用同一候选与时基")

    def on_method_changed(self, *_):
        name = self.method_var.get()
        self.opt_enable.set(name in ("optimize", "compare"))
        self.invalidate()
        self.refresh_method_labels()
        labels = {"candidate": "生成候选 → 共同评价", "fit": "生成候选 → 普通拟合 → 共同评价",
                  "optimize": "生成候选 → 约束优化 → 独立复核", "compare": "生成共同候选 → 三方法对照评价"}
        self.run_button.configure(text=labels[name])
        self.save_button.configure(text="保存三方法结果 / 诊断" if name == "compare" else "保存所选方法结果 / 诊断")
        self.method_note.set({"candidate": "保留逐点 IK 候选，不执行样条拟合。",
                              "fit": "普通最小二乘拟合，仅施加共同端点等式；任务和运动限值在生成后评价。",
                              "optimize": "沿用原 CasADi / IPOPT 求解与独立复核。",
                              "compare": "同一候选只生成一次；拟合与优化分别计算，失败分支单独记录。"}[name])
        self.spacing_entry.configure(state="disabled" if name == "candidate" else "normal")
        self.update_config_summary()

    def update_config_summary(self):
        super().update_config_summary()
        if hasattr(self, "spacing_var"):
            value = str(self.config["optimizer"]["knot_spacing_s"])
            # Config edits update the shared knot setting; typing in Entry is read at snapshot time.
            if getattr(self, "_last_config_spacing", None) != value:
                self.spacing_var.set(value)
                self._last_config_spacing = value

    def snapshot_job(self):
        # The original snapshot validates and freezes visual/manual inputs and hashes.
        job = super().snapshot_job()
        cfg = job["config"]
        cfg["optimizer"]["knot_spacing_s"] = float(self.spacing_var.get())
        job["config"] = opt.validate_config(cfg)
        job["method"] = self.method_var.get()
        if job["method"] not in (*core.METHODS, "compare"):
            raise ValueError("未知处理方法")
        return job

    def start(self):
        if self.busy or self._closing:
            return
        try:
            job = self.snapshot_job()
        except Exception as ex:
            messagebox.showerror("输入无效", str(ex), parent=self.root)
            return
        self.busy, self.last_result = True, None
        for widget in (self.save_button, self.save_npz_check, self.save_png_check):
            widget.configure(state="disabled")
        self.set_locked()
        self.cancel_button.configure(state="normal")
        self.progress_bar.configure(mode="indeterminate")
        self.progress_bar.start(16)
        self.status.set("正在生成共同候选…")
        self.result_label.set("输入已固定；计算中不写输出文件。")
        self._terminal_received, self._dead_since, self.cancel_started = False, None, None
        context = mp.get_context("spawn")
        self.process_queue, self.cancel_event = context.Queue(), context.Event()
        self.process = context.Process(target=core.run_job, args=(job, self.process_queue, self.cancel_event), daemon=True)
        try:
            self.process.start()
        except Exception as ex:
            self.retire_process()
            self.finish_busy()
            messagebox.showerror("启动失败", str(ex), parent=self.root)
        self.log("开始任务："+job["method"]+"；原文件与几何配置已冻结。")

    def finish_busy(self):
        self._saving = False
        super().finish_busy()
        if not self._closing:
            self.spacing_entry.configure(state="disabled" if self.method_var.get() == "candidate" else "normal")

    def enable_save_controls(self):
        if self.last_result and "variants" in self.last_result and not self.busy:
            for widget in (self.save_button, self.save_npz_check, self.save_png_check):
                widget.configure(state="normal")

    def start_save(self):
        if self.busy or not self.last_result:
            return
        want_npz, want_png = bool(self.save_npz_var.get()), bool(self.save_png_var.get())
        if not (want_npz or want_png):
            messagebox.showwarning("未选择", "请选择 NPZ 和/或 PNG", parent=self.root)
            return
        result = self.last_result
        self.busy = True
        self._saving = True
        self.set_locked()
        self.status.set("正在保存方法结果与诊断…")
        self._terminal_received, self._dead_since, self.cancel_started = False, None, None
        context = mp.get_context("spawn")
        self.process_queue = context.Queue()
        self.process = context.Process(target=core.save_job,
                                       args=(result, self.output_dir, want_npz, want_png, self.process_queue), daemon=True)
        try:
            self.process.start()
        except Exception as ex:
            self.retire_process()
            self.finish_busy()
            self.enable_save_controls()
            messagebox.showerror("保存进程启动失败", str(ex), parent=self.root)

    def poll_events(self):
        if self.process_queue is not None:
            try:
                while True:
                    event = self.process_queue.get_nowait()
                    self.events.put(event)
                    if event[0] in ("success", "failed", "cancelled", "saved", "save_error"):
                        self._terminal_received = True
            except queue.Empty:
                pass
            if self.process is not None:
                if self.cancel_started and self.process.is_alive() and time.monotonic()-self.cancel_started > 3:
                    self.process.terminate()
                    self._terminal_received = True
                    self.events.put(("cancelled", "计算进程已取消，没有保存输出文件"))
                    self.retire_process()
                elif not self.process.is_alive() and not self._terminal_received:
                    self._dead_since = self._dead_since or time.monotonic()
                    if time.monotonic()-self._dead_since > .8:
                        self.events.put(("failed", f"计算进程退出，退出码 {self.process.exitcode}", {}, None, ""))
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
                    r = self.last_result
                    good = sum(r["variants"][n]["report"]["generated"] for n in r["requested_methods"])
                    self.status.set(f"所选方法生成 {good}/{len(r['requested_methods'])}；共同检查结果见评价页")
                    self.progress_bar["value"] = 100
                    self.result_label.set("结果尚未保存。生成状态、共同离散检查和包络证书分别报告；全部结果均无实机回放认证。")
                    self.set_diagnostics({}, r)
                    self.show_preview(r)
                    self.show_gripper_plot(r)
                    self.enable_save_controls()
                    self.log("本次计算完成："+r["case_id"])
                    self.result_notebook.select(self.check_tab)
                elif kind == "failed":
                    self.retire_process()
                    self.finish_busy()
                    self.last_result = None
                    self.status.set("共同输入或候选生成失败")
                    self.log(event[1]+"\n"+event[4])
                    if not self._closing:
                        messagebox.showerror("计算失败", event[1], parent=self.root)
                elif kind == "cancelled":
                    self.retire_process()
                    self.last_result = None
                    self.finish_busy()
                    self.status.set(event[1])
                    self.log(event[1])
                elif kind in ("saved", "save_error"):
                    self.retire_process()
                    if self.last_result:
                        self.last_result.update(event[2] if kind == "saved" else event[3])
                    self.finish_busy()
                    self.enable_save_controls()
                    if kind == "saved":
                        folder = event[1].get("directory", self.output_dir)
                        self.status.set("保存完成")
                        self.result_label.set(f"输出目录：{folder}")
                        self.log("保存完成：\n"+"\n".join(str(p) for p in event[1]["saved"]))
                        if not self._closing:
                            messagebox.showinfo("保存完成", str(folder), parent=self.root)
                    else:
                        self.status.set("保存未全部完成，可重试")
                        self.log(event[1]+"\n"+event[2])
                        if not self._closing:
                            messagebox.showerror("保存失败", event[1], parent=self.root)
        except queue.Empty:
            pass
        if not self._closing or self.busy:
            self.root.after(120, self.poll_events)

    def redraw_results(self, *_):
        if self.last_result and not self.busy:
            self.set_diagnostics({}, self.last_result)

    def set_diagnostics(self, report, result=None):
        if result is None or "variants" not in result:
            return super().set_diagnostics(report, None)
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        payload = {"case_id": result["case_id"], "method": result["method"], "common_seconds": result["common_seconds"],
                   "reports": {n: v["report"] for n, v in result["variants"].items()}}
        self.diag_text.configure(state="normal")
        self.diag_text.delete("1.0", "end")
        self.diag_text.insert("1.0", opt.json_text(payload))
        self.diag_text.configure(state="disabled")
        if self.diag_canvas:
            self.diag_canvas.get_tk_widget().destroy()
        fig = core.joint_figure(result, int(self.joint_var.get()[1:])-1, self.derivative_var.get() == "样条解析导数")
        self.diag_canvas = FigureCanvasTkAgg(fig, master=self.diag_plot)
        self.diag_canvas.draw()
        self.diag_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.check_tree.delete(*self.check_tree.get_children())
        for name, variant in result["variants"].items():
            r = variant["report"]
            ratios = r.get("peak_ratios", {})
            def ratio(key):
                return "未启用" if ratios.get(key) is None else f"{max(ratios[key]):.4g}"
            self.check_tree.insert("", "end", iid=name, values=(core.method_label(name, result["mode"] == "staged"),
                "成功" if r["generated"] else "失败", "通过" if r.get("discrete_pass") else ("未通过" if r["generated"] else "未评价"),
                "不适用" if name == "candidate" else ("获得" if r.get("hull_certified") else ("未获得" if r["generated"] else "未评价")),
                ratio("velocity"), ratio("acceleration"), ratio("jerk")))
        self.check_tree.selection_set(result["requested_methods"][-1])
        self.show_check_details()

    def show_check_details(self, *_):
        if not self.last_result or not self.check_tree.selection():
            return
        name = self.check_tree.selection()[0]
        v = self.last_result["variants"][name]
        r = v["report"]
        if not r["generated"]:
            self.check_note.set("该分支未生成轨迹："+r["error"]+"\n可保存失败诊断；不会导出为成功轨迹。")
            return
        text = (f"共同任务最大位置残差比：{r['task_position_peak_ratio']:.5g}；姿态残差比：{r['task_orientation_peak_ratio']:.5g}\n"
                f"归一化 jerk RMS：{r['jerk_normalized_rms']:.5g}；端点关节误差：{r['endpoint_error_rad']:.3g} rad\n"
                f"方法处理：{v['processing_seconds']:.3f} s；共同评价：{r['evaluation_seconds']:.3f} s\n")
        issues = r["failures"]+r["hull_failures"]
        text += "检查记录："+(opt.json_text(issues) if issues else "无超限记录")
        if name == "candidate":
            text += "\n候选仅为共同参考时，不在单方法导出清单中；控制点包络不适用。"
        self.check_note.set(text)

    def show_preview(self, result):
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        if self.canvas:
            self.canvas.get_tk_widget().destroy()
        self.preview_note.pack_forget()
        self.canvas = FigureCanvasTkAgg(core.trajectory_figure(result), master=self.preview_tab)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def close(self):
        if self.busy and self._saving:
            self._closing = True
            self.status.set("正在退出：等待保存进程完成…")
            return
        super().close()


def main():
    mp.freeze_support()
    root = tk.Tk()
    try:
        FitGui(root)
    except Exception as ex:
        root.withdraw()
        messagebox.showerror("启动失败", f"{ex}\n请保留原有 ur_opt.py、joint_opt.py、ur_pose_ui.py 与 opt_cfg.json。", parent=root)
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
