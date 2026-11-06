"""Behavior checks: python -m unittest -v test_fit. No hardware connection."""
import builtins
import importlib.util
import multiprocessing as mp
from pathlib import Path
import queue
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from scipy.interpolate import BSpline
import fit_core as core
import joint_opt as opt
from test_opt import example, job_from

HERE = Path(__file__).resolve().parent
HAS_CASADI = importlib.util.find_spec("casadi") is not None


def run(method="fit", cfg=None, ref=None):
    if cfg is None:
        cfg, ref = example()
    job = job_from(ref, cfg)
    job["method"] = method
    channel = queue.Queue()
    core.run_job(job, channel, threading.Event())
    terminal = []
    while not channel.empty():
        event = channel.get()
        if event[0] != "progress":
            terminal.append(event)
    if terminal[-1][0] != "success":
        raise AssertionError(terminal[-1])
    return terminal[-1][1]


class FitTests(unittest.TestCase):
    def test_polynomial_reproduction_and_exact_endpoint_derivatives(self):
        cfg, ref = example()
        t = ref["time"]
        s = t/t[-1]
        polynomial = 10*s**3-15*s**4+6*s**5
        ref["candidate_q"] = ref["raw_q"][:1]+polynomial[:, None]*np.linspace(.01, .06, 6)
        for key in cfg["boundary"]:
            cfg["boundary"][key] = [0.]*6
        fitted = core.fit_spline(ref, cfg)
        np.testing.assert_allclose(fitted["q"], ref["candidate_q"], atol=1e-9)
        curve = BSpline(fitted["knots"], fitted["coefficients"], fitted["degree"])
        np.testing.assert_allclose(curve(t[[0, -1]], nu=1), 0., atol=1e-9)
        np.testing.assert_allclose(curve(t[[0, -1]], nu=2), 0., atol=1e-8)

    def test_endpoint_constrained_ls_is_stationary_and_input_unchanged(self):
        cfg, ref = example()
        before = ref["candidate_q"].copy()
        s = core.fit_spline(ref, cfg)
        basis = opt.basis_at(ref["time"], s["knots"], s["degree"])
        residual = basis @ s["coefficients"]-before
        gradient = basis[:, 1:-1].T @ residual
        self.assertLess(np.max(abs(gradient)), 1e-8)
        np.testing.assert_array_equal(ref["candidate_q"], before)
        np.testing.assert_allclose(s["q"][[0, -1]], before[[0, -1]], atol=1e-10)
        knots, degree, _ = opt.spline_basis(ref["time"], cfg["optimizer"]["knot_spacing_s"])
        np.testing.assert_array_equal(s["knots"], knots)
        self.assertEqual(s["degree"], degree)

    def test_fit_requires_no_casadi_and_preserves_overlimit_result(self):
        cfg, ref = example()
        cfg["limits"]["v_max"] = [1e-6]*6
        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name == "casadi":
                raise AssertionError("Fitting must not import CasADi")
            return original(name, *args, **kwargs)
        with patch("builtins.__import__", side_effect=guarded):
            result = run("fit", cfg, ref)
        report = result["variants"]["fit"]["report"]
        self.assertTrue(report["generated"])
        self.assertFalse(report["discrete_pass"])
        self.assertFalse(report["hull_certified"])
        self.assertTrue(report["failures"])
        with tempfile.TemporaryDirectory() as temp:
            core.save_bundle(result, temp, True, False)
            with np.load(result["save_dir"]/"fit.npz", allow_pickle=False) as d:
                self.assertEqual(int(d["opt_enabled"][0]), 0)
                self.assertEqual(int(d["opt_validated"][0]), 0)
                self.assertEqual(int(d["method_generated"][0]), 1)
                self.assertEqual(int(d["evaluation_discrete_pass"][0]), 0)

    def test_compare_keeps_fit_when_optimizer_fails_and_builds_candidate_once(self):
        import ur_pose_ui as base
        original = base.generate_preview
        with patch.object(base, "generate_preview", wraps=original) as generator, \
             patch.object(opt, "optimize", side_effect=opt.OptimizationError("planned failure")):
            result = run("compare")
        self.assertEqual(generator.call_count, 1)
        self.assertTrue(result["variants"]["candidate"]["report"]["generated"])
        self.assertTrue(result["variants"]["fit"]["report"]["generated"])
        self.assertFalse(result["variants"]["optimize"]["report"]["generated"])
        with tempfile.TemporaryDirectory() as temp:
            core.save_bundle(result, temp, True, False)
            self.assertTrue((result["save_dir"]/"cand.npz").exists())
            self.assertTrue((result["save_dir"]/"fit.npz").exists())
            self.assertFalse((result["save_dir"]/"opt.npz").exists())
            self.assertTrue((result["save_dir"]/"opt_report.json").exists())

    def test_export_roundtrip_safe_retry_and_tamper_rejection(self):
        result = run()
        with tempfile.TemporaryDirectory() as temp:
            core.save_bundle(result, temp, True, False)
            folder = result["save_dir"]
            with np.load(folder/"fit.npz", allow_pickle=False) as d:
                curve = BSpline(d["spline_knots"], d["spline_coefficients"], int(d["spline_degree"][0]))
                np.testing.assert_allclose(curve(d["timestamp"]), d["ur_q"], atol=1e-12)
                np.testing.assert_array_equal(d["gripper_open"], result["npz_data"]["gripper_open"])
                np.testing.assert_array_equal(d["candidate_ur_q"], result["reference"]["candidate_q"])
                self.assertEqual(str(d["trajectory_method"][0]), "fit")
            repeated = core.save_bundle(result, temp, True, False)
            self.assertFalse(repeated["saved"])
            self.assertTrue(repeated["skipped"])
            (folder/"fit.npz").write_bytes(b"changed")
            with self.assertRaises(FileExistsError):
                core.save_bundle(result, temp, True, False)

    def test_missing_jerk_limit_is_not_reported_as_a_limit(self):
        cfg, ref = example()
        cfg["limits"]["j_max"] = None
        result = run("fit", cfg, ref)
        r = result["variants"]["fit"]["report"]
        self.assertIsNone(r["peak_ratios"]["jerk"])
        self.assertIn("not a hard limit", r["jerk_normalization"])

    def test_cancel_before_fit_and_worker(self):
        cfg, ref = example()
        stop = threading.Event()
        stop.set()
        with self.assertRaises(opt.Cancelled):
            core.fit_spline(ref, cfg, cancel=stop)
        output = queue.Queue()
        core.run_job(dict(job_from(ref, cfg), method="fit"), output, stop)
        self.assertEqual(output.get()[0], "cancelled")

    def test_windows_spawn_fit(self):
        cfg, ref = example()
        context = mp.get_context("spawn")
        output, stop = context.Queue(), context.Event()
        process = context.Process(target=core.run_job, args=(dict(job_from(ref, cfg), method="fit"), output, stop))
        process.start()
        try:
            while True:
                event = output.get(timeout=30)
                if event[0] != "progress":
                    break
            self.assertEqual(event[0], "success", str(event[1:2]))
            self.assertTrue(event[1]["variants"]["fit"]["report"]["generated"])
        finally:
            process.join(3)
            if process.is_alive():
                process.terminate()
                process.join()
            process.close()
            output.close()

    @unittest.skipUnless(HAS_CASADI, "CasADi not installed")
    def test_real_three_method_pipeline_and_optimizer_compatibility(self):
        result = run("compare")
        self.assertTrue(all(v["report"]["generated"] for v in result["variants"].values()),
                        {k:v["report"].get("error") for k,v in result["variants"].items()})
        fit, optimized = result["variants"]["fit"], result["variants"]["optimize"]
        np.testing.assert_array_equal(fit["knots"], optimized["knots"])
        self.assertTrue(optimized["report"]["solver_report"]["valid"])
        self.assertTrue(optimized["report"]["discrete_pass"])
        data = core.export_data(result, "optimize")
        self.assertEqual(int(data["opt_validated"][0]), 1)
        np.testing.assert_array_equal(data["opt_coefficients"], optimized["coefficients"])


class GuiTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        import ur_fit
        self.root = tk.Tk()
        self.root.withdraw()
        self.callback_errors = []
        self.root.report_callback_exception = lambda *args: self.callback_errors.append(args)
        self.app = ur_fit.FitGui(self.root)
        self.messages = []
        self.patches = [patch("tkinter.messagebox."+name, side_effect=lambda *a, **kw:self.messages.append(a))
                        for name in ("showinfo", "showerror", "showwarning")]
        for p in self.patches:
            p.start()
        self.root.update()

    def tearDown(self):
        if self.app.process is not None:
            self.app.cancel()
            self.pump()
        for p in self.patches:
            p.stop()
        self.app.close()

    def pump(self):
        deadline = time.monotonic()+45
        while self.app.busy and time.monotonic() < deadline:
            self.root.update()
            time.sleep(.015)
        self.root.update()
        self.assertFalse(self.app.busy)
        self.assertFalse(self.callback_errors, str(self.callback_errors))

    def test_defaults_mapping_fit_save_plots_and_invalidation(self):
        app = self.app
        self.assertEqual(app.mode_labels[app.mode_label.get()], "staged")
        self.assertEqual(app.method_var.get(), "optimize")
        self.assertFalse(self.callback_errors)
        app.method_var.set("fit")
        self.assertFalse(app.opt_enable.get())
        cfg, ref = example()
        app.snapshot_job = lambda: dict(job_from(ref, cfg), method="fit")
        app.start()
        self.pump()
        self.assertIsNotNone(app.last_result)
        self.assertTrue(app.last_result["variants"]["fit"]["report"]["generated"])
        self.assertIsNotNone(app.canvas)
        self.assertIsNotNone(app.diag_canvas)
        app.joint_var.set("J4")
        app.derivative_var.set("样条解析导数")
        self.root.update()
        self.assertFalse(self.callback_errors)
        with tempfile.TemporaryDirectory() as temp:
            app.output_dir = Path(temp)
            app.start_save()
            self.pump()
            folder = app.last_result["save_dir"]
            self.assertTrue((folder/"fit.npz").exists())
            self.assertEqual(len(list(folder.glob("*.png"))), 7)
            app.start_save()
            self.pump()
            self.assertEqual(len(list(folder.glob("*.png"))), 7)
        app.method_var.set("compare")
        self.assertIsNone(app.last_result)
        self.assertEqual(str(app.save_button.cget("state")), "disabled")
        self.assertTrue(app.opt_enable.get())
        app.spacing_var.set("0.25")
        self.assertEqual(app.config["optimizer"]["knot_spacing_s"], .25)
        app.method_var.set("fit")
        self.assertEqual(float(app.spacing_var.get()), .25)

    def test_failed_save_preserves_partial_state_for_retry(self):
        cfg, ref = example()
        self.app.snapshot_job = lambda: dict(job_from(ref, cfg), method="fit")
        self.app.start()
        self.pump()
        with tempfile.TemporaryDirectory() as temp:
            self.app.output_dir = Path(temp)
            self.app.save_png_var.set(False)
            self.app.start_save()
            self.pump()
            result = self.app.last_result
            folder = result["save_dir"]
            (folder/"fit.npz").write_bytes(b"tampered")
            self.app.start_save()
            self.pump()
            self.assertEqual(result["save_dir"], folder)
            self.assertIn("未全部完成", self.app.status.get())
            self.assertEqual((folder/"fit.npz").read_bytes(), b"tampered")

    def test_gui_cancel_never_enables_save(self):
        cfg, ref = example()
        self.app.snapshot_job = lambda: dict(job_from(ref, cfg), method="fit")
        self.app.start()
        self.app.cancel()
        self.pump()
        self.assertIsNone(self.app.last_result)
        self.assertEqual(str(self.app.save_button.cget("state")), "disabled")


if __name__ == "__main__":
    mp.freeze_support()
    unittest.main()
