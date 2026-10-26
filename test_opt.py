"""Behavior checks: python -m unittest -v test_opt (no hardware required)."""
import copy
import importlib.util
import io
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
import joint_opt as opt
import ur_pose_ui as base

HERE = Path(__file__).resolve().parent
HAS_CASADI = importlib.util.find_spec("casadi") is not None


def example():
    cfg = opt.load_config(HERE / "opt_cfg.json")
    cfg["optimizer"]["max_cpu_seconds"] = 30
    t = np.linspace(0, 2, 201)
    q = np.tile([-1.5, -1.1, -2.2, .3, 1.5, 0.], (len(t), 1))
    q[:, 0] += .05*np.sin(np.pi*t)
    q[1:-1, 1] += .001*(-1.)**np.arange(len(t)-2)
    event = {"time_s": 1.}
    ref = opt.build_reference(q, q.copy(), t, np.zeros(len(t)), [0, 0, 0], 0, [0, 0], None, event, None, cfg)
    return cfg, ref


def job_from(ref, cfg):
    buffer = io.BytesIO()
    np.savez(buffer, timestamp=ref["time"], ur_q=ref["raw_q"], gripper_open=np.ones(len(ref["time"])),
             rtde_control_frequency_hz=np.array([100.]))
    return {"config": cfg, "enable": True, "vision": False, "ignore_z": True, "task_path": None,
            "manifest_bytes": None, "files": {"source": {"path": str(HERE/"synthetic.npz"), "bytes": buffer.getvalue()}},
            "geometry": {"xyz_cm": [0, 0, 0], "ramp": 1, "mode": "full", "close_event_index": None,
                         "advance_seconds": 1, "yaw_deg": 0, "pivot_mode": "manual", "pivot_xy_m": [0, 0],
                         "release_advance_seconds": .5, "return_start_mode": "auto", "return_lift_mm": 30,
                         "return_lift_wait_seconds": .2, "return_manual_time_s": None}}


class DataTests(unittest.TestCase):
    def test_invalid_limits_and_flags_are_rejected(self):
        cfg, _ = example()
        for group, key, value in (("limits", "a_max", [0]*6), ("limits", "q_min", [20]*6),
                                  ("task", "freeze_tail", "false"), ("optimizer", "validation_substeps", 2.5)):
            bad = copy.deepcopy(cfg)
            bad[group][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                opt.validate_config(bad)

    def test_fk_agrees_with_existing_model_and_nonzero_tcp(self):
        cfg, _ = example()
        cfg["model"]["tcp_xyz_rotvec"] = [.01, -.02, .15, .1, .2, -.1]
        q = np.random.default_rng(4).uniform(-3, 3, (12, 6))
        expected = np.array([base.fk_jacobian(row)[0] @ opt.tcp_matrix(cfg) for row in q])
        np.testing.assert_allclose(opt.fk_numpy(q, opt.tcp_matrix(cfg)), expected, atol=1e-13)
        if HAS_CASADI:
            import casadi as ca
            fk = opt._casadi_fk(ca, cfg)
            for row, h in zip(q, expected):
                p, r, _ = fk(row)
                np.testing.assert_allclose(np.array(p).ravel(), h[:3, 3], atol=1e-13)
                np.testing.assert_allclose(np.array(r).reshape(3, 3, order="F"), h[:3, :3], atol=1e-13)

    def test_reference_is_geometry_not_candidate_fk(self):
        cfg, ref = example()
        raw, t = ref["raw_q"], ref["time"]
        candidate = raw + .01
        alpha = np.linspace(0, 1, len(t))
        actual = opt.build_reference(raw, candidate, t, alpha, [.01, .02, 0], 10, [.1, .2], None, None, None, cfg)
        i = 100
        expected = base.compensated_pose(base.fk_jacobian(raw[i])[0], np.array([.01, .02, 0]), 10, [.1, .2], alpha[i])
        np.testing.assert_allclose(actual["target"][i], expected, atol=1e-13)
        self.assertGreater(np.linalg.norm(actual["target"][i] - opt.fk_numpy(candidate[[i]])[0]), .01)

    def test_timestamps_and_joint_branch_jumps_rejected(self):
        cfg, ref = example()
        for mode in ("time", "joint"):
            t, q = ref["time"].copy(), ref["candidate_q"].copy()
            if mode == "time":
                t[3] = t[2]
            else:
                q[50, 0] += 2*np.pi
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                opt.build_reference(ref["raw_q"], q, t, np.zeros(len(t)), [0, 0, 0], 0, [0, 0], None, None, None, cfg)

    def test_cancel_before_start(self):
        cfg, ref = example()
        stop = threading.Event()
        stop.set()
        with self.assertRaises(opt.Cancelled):
            opt.optimize(ref, cfg, cancel=stop)

    def test_infeasible_endpoint_is_rejected_without_fallback(self):
        cfg, ref = example()
        cfg["limits"]["q_min"] = [0]*6
        with self.assertRaises(opt.OptimizationError) as context:
            opt.optimize(ref, cfg)
        self.assertFalse(context.exception.report["valid"])
        self.assertEqual(context.exception.report["solver_status"], "preflight_endpoint_limits")


@unittest.skipUnless(HAS_CASADI, "CasADi not installed; install opt_deps.txt for solver checks")
class SolverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg, cls.ref = example()
        cls.solution = opt.optimize(cls.ref, cls.cfg)

    def test_true_nlp_reduces_jerk_and_respects_pose_and_joint_limits(self):
        s, ref = self.solution, self.ref
        self.assertTrue(s["report"]["valid"])
        self.assertGreater(s["report"]["iterations"], 0)
        self.assertIn("pick", s["report"]["phase_errors"])
        np.testing.assert_allclose(s["q"][[0, -1]], ref["candidate_q"][[0, -1]], atol=1e-8)
        self.assertLess(max(s["report"]["optimized_metrics"]["jerk_peak"]), max(s["report"]["candidate_metrics"]["jerk_peak"])*.1)
        self.assertLessEqual(s["report"]["phase_errors"]["pick"]["position_mm"], self.cfg["task"]["critical_position_mm"]+.002)
        for name, key in (("velocity", "v_max"), ("acceleration", "a_max"), ("jerk", "j_max")):
            self.assertTrue(np.all(np.array(s["report"]["control_hull_peaks"][name]) <= np.array(self.cfg["limits"][key])+1e-5))

    def test_independent_validator_rejects_tampered_solution(self):
        s = self.solution
        bad = s["coefficients"].copy()
        bad[len(bad)//2, 2] += 1
        report, _ = opt.validate_solution(self.ref, self.cfg, s["knots"], s["degree"], bad)
        self.assertFalse(report["valid"])
        self.assertTrue(report["failures"])

    def test_spawn_worker_and_npz_roundtrip(self):
        context = mp.get_context("spawn")
        channel, stop = context.Queue(), context.Event()
        job = job_from(self.ref, self.cfg)
        process = context.Process(target=opt.run_job, args=(job, channel, stop))
        process.start()
        try:
            while True:
                event = channel.get(timeout=60)
                if event[0] != "progress":
                    break
            self.assertEqual(event[0], "success", str(event[1:2]))
            result = event[1]
            self.assertTrue(result["optimization_report"]["valid"])
            np.testing.assert_array_equal(result["npz_data"]["timestamp"], np.arange(len(self.ref["time"]))/100.)
            np.testing.assert_allclose(result["npz_data"]["timestamp"], self.ref["time"], atol=1e-14, rtol=0)
            np.testing.assert_array_equal(result["npz_data"]["gripper_open"], np.ones(len(self.ref["time"])))
            with tempfile.TemporaryDirectory() as temp:
                result["base"] = Path(temp)/"test_opt"
                saved = base.save_selected_outputs(result, Path(temp), True, False)
                loaded = base.load_record(saved["saved"][0])
                self.assertEqual(int(loaded["opt_validated"][0]), 1)
                self.assertEqual(int(loaded["opt_replay_ready"][0]), 0)
                np.testing.assert_allclose(loaded["ur_q"], result["npz_data"]["ur_q"])
                self.assertIn("candidate_ur_q", loaded)
                self.assertIn("source_ur_q", loaded)
        finally:
            process.join(3)
            if process.is_alive():
                process.terminate()
                process.join()
            process.close()
            channel.close()


@unittest.skipUnless(HAS_CASADI, "CasADi required for GUI integration checks")
class GuiTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        import ur_opt
        try:
            self.root = tk.Tk()
        except tk.TclError as ex:
            self.skipTest("No Tk display: " + str(ex))
        self.root.withdraw()
        self.app = ur_opt.OptGui(self.root)
        cfg, ref = example()
        self.app.snapshot_job = lambda: job_from(ref, cfg)
        self.messages = []
        self.patches = [patch("tkinter.messagebox."+name, side_effect=lambda *args, **kw: self.messages.append(args))
                        for name in ("showinfo", "showerror", "showwarning")]
        for p in self.patches:
            p.start()

    def tearDown(self):
        if hasattr(self, "app") and self.app.process is not None:
            self.app.cancel()
            self.pump()
        for p in getattr(self, "patches", []):
            p.stop()
        if hasattr(self, "root"):
            self.app.close()

    def pump(self):
        until = time.monotonic()+45
        while self.app.busy and time.monotonic() < until:
            self.root.update()
            time.sleep(.015)
        self.assertFalse(self.app.busy, "GUI job did not finish")

    def test_gui_process_preview_save_and_invalidate(self):
        app = self.app
        self.assertEqual(len(app.result_notebook.tabs()), 5)
        app.start()
        self.assertEqual(str(app.mode_combo.cget("state")), "disabled")
        self.pump()
        self.assertIsNotNone(app.last_result, self.messages)
        self.assertTrue(app.last_result["optimization_report"]["valid"])
        self.assertIsNotNone(app.canvas)
        self.assertIsNotNone(app.diag_canvas)
        with tempfile.TemporaryDirectory() as temp:
            app.output_dir = Path(temp)
            app.save_npz_var.set(True)
            app.save_png_var.set(True)
            app.start_save()
            self.pump()
            self.assertIsNotNone(app.last_result)
            self.assertEqual(len(list(Path(temp).glob("*.npz"))), 1)
            self.assertEqual(len(list(Path(temp).glob("*.png"))), 2)
            self.assertEqual(len(list(Path(temp).glob("*.json"))), 2)
            app.start_save()
            self.pump()
            self.assertEqual(len(list(Path(temp).glob("*.npz"))), 1)
        app.dx.set("1")
        self.assertIsNone(app.last_result)
        self.assertEqual(str(app.save_button.cget("state")), "disabled")

    def test_gui_cancel_does_not_enable_save(self):
        self.app.last_result = {"optimization_report": {"valid": True}}
        self.app.enable_save_controls()
        self.app.start()
        self.app.cancel()
        self.pump()
        self.assertIsNone(self.app.last_result)
        self.assertEqual(str(self.app.save_button.cget("state")), "disabled")


if __name__ == "__main__":
    mp.freeze_support()
    unittest.main()
