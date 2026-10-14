# -*- coding: utf-8 -*-
"""PySide6 视觉示教组件：复用 square_pose 的 LSD/PnP，机器人只读快照。
视觉定位仅在静止状态下触发；不参与 UR5e 控制循环。
"""
from __future__ import annotations
import datetime as dt
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QLineEdit, QDoubleSpinBox, QMessageBox, QGroupBox, QCheckBox)

import square_pose as sp


class CameraSignals(QObject):
    error = Signal(str)


class CameraFeed:
    def __init__(self):
        self.lock = threading.Lock()
        self.packet = None
        self.stop_event = threading.Event()
        self.thread = None
        self.signals = CameraSignals()

    def start(self, serial=''):
        if self.thread and self.thread.is_alive():
            raise RuntimeError('相机已经启动')
        if sp.rs is None:
            raise RuntimeError('未安装 pyrealsense2')
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run, args=(serial,), daemon=True)
        self.thread.start()

    def _run(self, serial):
        pipeline = None
        try:
            pipeline = sp.rs.pipeline()
            cfg = sp.rs.config()
            if serial:
                cfg.enable_device(serial)
            cfg.enable_stream(sp.rs.stream.color, 640, 480, sp.rs.format.bgr8, 30)
            cfg.enable_stream(sp.rs.stream.depth, 640, 480, sp.rs.format.z16, 30)
            profile = pipeline.start(cfg)
            actual_serial = profile.get_device().get_info(sp.rs.camera_info.serial_number)
            scale = profile.get_device().first_depth_sensor().get_depth_scale()
            intr = profile.get_stream(sp.rs.stream.color).as_video_stream_profile().get_intrinsics()
            K = np.array([[intr.fx, 0, intr.ppx], [0, intr.fy, intr.ppy], [0, 0, 1]], float)
            D = np.asarray(intr.coeffs, float)
            align = sp.rs.align(sp.rs.stream.color)
            while not self.stop_event.is_set():
                try:
                    frames = align.process(pipeline.wait_for_frames(timeout_ms=1000))
                except RuntimeError:
                    continue
                rgb, dep = frames.get_color_frame(), frames.get_depth_frame()
                if not rgb:
                    continue
                color = np.asanyarray(rgb.get_data()).copy()
                depth = np.asanyarray(dep.get_data()).astype(np.float32) * scale if dep else None
                pkt = sp.ImagePacket(color, depth, K, D, actual_serial, time.monotonic())
                with self.lock:
                    self.packet = pkt
        except Exception as exc:
            self.signals.error.emit(str(exc))
        finally:
            if pipeline is not None:
                try:
                    pipeline.stop()
                except Exception:
                    pass

    def latest(self):
        with self.lock:
            return self.packet

    def stop(self):
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        with self.lock:
            self.packet = None


class VisionTeachPanel(QWidget):
    log_signal = Signal(str)
    reference_changed = Signal(str)

    def __init__(self, coordinator, parent=None):
        super().__init__(parent)
        self.coordinator = coordinator
        self.feed = CameraFeed()
        self.feed.signals.error.connect(self._error)
        self.T_tcp_cam = None
        self.matrix_path = ''
        self.reference = None
        self.reference_path = None
        self.roi = None
        self.latest_result = None
        self.latest_packet = None
        self.latest_overlay = None
        self.latest_detection_at = 0.0
        self._last_frame_at = 0.0
        self._build()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_preview)
        self.timer.start(180)

    def _build(self):
        root = QVBoxLayout(self)
        row = QHBoxLayout()
        self.serial = QLineEdit()
        self.serial.setPlaceholderText('RealSense 序列号（留空自动选择）')
        row.addWidget(self.serial)
        for name, callback in [('启动相机', self.start_camera), ('停止相机', self.stop_camera), ('加载手眼矩阵', self.load_matrix)]:
            b = QPushButton(name)
            b.clicked.connect(callback)
            row.addWidget(b)
        root.addLayout(row)
        self.matrix_label = QLabel('手眼矩阵：未加载')
        root.addWidget(self.matrix_label)
        row2 = QHBoxLayout()
        self.edge = QDoubleSpinBox()
        self.edge.setRange(10, 500)
        self.edge.setValue(80)
        self.edge.setSuffix(' mm')
        row2.addWidget(QLabel('正方形边长'))
        row2.addWidget(self.edge)
        self.gap = QDoubleSpinBox()
        self.gap.setRange(0, 500)
        self.gap.setValue(35)
        self.gap.setSuffix(' mm')
        row2.addWidget(QLabel('顶面与底板高差'))
        row2.addWidget(self.gap)
        self.auto = QCheckBox('自动 LSD 检测')
        self.auto.setChecked(True)
        row2.addWidget(self.auto)
        root.addLayout(row2)
        self.image_label = QLabel('请启动 D435i')
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(640, 480)
        self.image_label.setStyleSheet('background:#222;color:white')
        self.image_label.mousePressEvent = self._mouse_press
        self.image_label.mouseReleaseEvent = self._mouse_release
        self._drag_start = None
        root.addWidget(self.image_label)
        row3 = QHBoxLayout()
        for name, callback in [('框选 ROI', self.select_roi), ('清除 ROI', self.clear_roi), ('检测静止位姿', self.detect_now),
                               ('记录视觉基准', self.save_reference), ('加载参考 JSON', self.load_reference_dialog),
                               ('保存当前 pose.json', self.save_current)]:
            b = QPushButton(name)
            b.clicked.connect(callback)
            row3.addWidget(b)
        root.addLayout(row3)
        self.pose_label = QLabel('位姿：未检测')
        self.pose_label.setWordWrap(True)
        self.ref_label = QLabel('当前任务参考：未设置')
        self.ref_label.setWordWrap(True)
        root.addWidget(self.pose_label)
        root.addWidget(self.ref_label)
        root.addWidget(QLabel('注意：先在机器人静止、物体未被触碰时记录基准；每条 NPZ 使用自己的参考。视觉检测不发送运动命令。'))

    def _error(self, message):
        self.log_signal.emit('[VISION][ERR] ' + message)
        self.pose_label.setText('相机错误：' + message)

    def start_camera(self):
        try:
            self.feed.start(self.serial.text().strip())
            self.log_signal.emit('[VISION] 正在启动 D435i')
        except Exception as e:
            QMessageBox.warning(self, '相机启动失败', str(e))

    def stop_camera(self):
        self.feed.stop()
        self.latest_result = None
        self.latest_packet = None

    def load_matrix(self):
        path, _ = QFileDialog.getOpenFileName(self, '选择 T_tcp_cam', '', '矩阵 (*.npy *.txt)')
        if not path:
            return
        try:
            T = np.load(path, allow_pickle=False) if path.lower().endswith('.npy') else np.loadtxt(path)
            self.T_tcp_cam = sp.rigid_matrix_check(T)
            self.matrix_path = path
            self.matrix_label.setText('手眼矩阵：' + path)
            self.latest_result = None
        except Exception as e:
            QMessageBox.warning(self, '矩阵错误', str(e))

    def select_roi(self):
        self._selecting_roi = True
        self.log_signal.emit('[VISION] 在图像上按下并拖动鼠标，框选正方形')

    def clear_roi(self):
        self.roi = None
        self.latest_result = None

    def _mouse_press(self, event):
        if getattr(self, '_selecting_roi', False) and event.button() == Qt.LeftButton:
            self._drag_start = (int(event.position().x()), int(event.position().y()))

    def _mouse_release(self, event):
        if self._drag_start is None:
            return
        a = self._drag_start
        b = (int(event.position().x()), int(event.position().y()))
        self._drag_start = None
        self._selecting_roi = False
        x0, x1 = sorted((max(0, min(639, a[0])), max(0, min(639, b[0]))))
        y0, y1 = sorted((max(0, min(479, a[1])), max(0, min(479, b[1]))))
        if x1-x0 < 45 or y1-y0 < 45:
            self.log_signal.emit('[VISION] ROI 太小，请重新框选')
            return
        self.roi = (x0, y0, x1, y1)
        self.latest_result = None
        self.log_signal.emit(f'[VISION] ROI={self.roi}')

    def refresh_preview(self):
        pkt = self.feed.latest()
        if pkt is None:
            return
        vis = pkt.color.copy()
        if self.roi:
            cv2.rectangle(vis, self.roi[:2], self.roi[2:], (0, 190, 255), 2)
            if self.auto.isChecked():
                try:
                    quad = sp.detect_quad_lsd(pkt.color, self.roi)
                    if quad is not None:
                        cv2.polylines(vis, [np.asarray(quad, np.int32)], True, (0, 255, 0), 2)
                except Exception:
                    pass
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        h, w, _ = rgb.shape
        self.image_label.setPixmap(QPixmap.fromImage(QImage(rgb.data, w, h, 3*w, QImage.Format_RGB888).copy()))

    def _robot_snapshot(self):
        ur = self.coordinator.ur
        if not ur.connected or ur.rtde_r is None:
            raise RuntimeError('请先在遥操作 GUI 中连接 UR5e')
        # 只在示教/回放未运行时采样，避免在控制循环中并发读取 RTDE。
        if self.coordinator.recording or self.coordinator.replay_running or self.coordinator.teleop_running:
            raise RuntimeError('正在示教记录或回放，请先停止再进行视觉定位')
        with ur.lock:
            pose = np.asarray(ur.rtde_r.getActualTCPPose(), float)
            qd = np.asarray(ur.rtde_r.getActualQd(), float)
            speed = np.asarray(ur.rtde_r.getActualTCPSpeed(), float)
        if pose.shape != (6,) or qd.shape != (6,) or speed.shape != (6,) or not np.isfinite(pose).all():
            raise RuntimeError('UR5e 返回的状态不合法')
        stationary = np.max(np.abs(qd)) < sp.MOVING_QD_MAX and np.linalg.norm(speed[:3]) < sp.MOVING_TCP_MAX
        if not stationary:
            raise RuntimeError('UR5e 未静止，拒绝保存 Base 位姿')
        return sp.ur_pose_to_T(pose)

    def detect_now(self):
        try:
            pkt = self.feed.latest()
            if pkt is None or time.monotonic()-pkt.ts_monotonic > 0.45:
                raise RuntimeError('相机画面过期或未连接')
            if self.T_tcp_cam is None:
                raise RuntimeError('请加载 T_tcp_cam 手眼矩阵')
            if self.roi is None:
                raise RuntimeError('请先框选 ROI')
            T_tcp = self._robot_snapshot()
            # 静止情况下视觉帧与 TCP 快照可近似配对；并非逐帧硬同步。
            quad = sp.detect_quad_lsd(pkt.color, self.roi)
            if quad is None:
                raise RuntimeError('ROI 内未检测到可靠正方形，请调整光照或 ROI')
            quad = sp.canonical_corners(quad)
            T_base_cam = T_tcp @ self.T_tcp_cam
            pnp = sp.pnp_square(quad, pkt.K, pkt.D, edge_m=self.edge.value()/1000, T_base_cam=T_base_cam)
            pose = sp.base_pose_from_square(T_base_cam, pnp)
            res = {'source':'camera','capture_time_iso':dt.datetime.now().isoformat(timespec='milliseconds'),
                   'camera_serial':pkt.cam_serial,'corners_px':quad.tolist(), 'roi_px':list(self.roi),
                   'edge_mm':float(self.edge.value()),'top_plate_gap_mm':float(self.gap.value()),
                   'rmse_px':float(pnp['rmse_px']), 'T_cam_top':pnp['T_cam_top'].tolist(),
                   'camera_center_m':pnp['tvec'].tolist(), 'frame':'base',
                   'center_m':pose['center_m'].tolist(),'yaw_deg':float(pose['yaw_deg']),
                   'T_base_top':pose['T_base_top'].tolist(), 'T_base_tcp':T_tcp.tolist(),
                   'T_base_cam':T_base_cam.tolist(), 'stationary':True,
                   'detection_mode':'automatic_lsd',
                   'depth_check':sp.depth_stats(pkt.depth_m, quad, pkt.K, pkt.D, T_base_cam,
                          target_gap_m=self.gap.value()/1000)}
            self.latest_result = sp.make_jsonable(res)
            self.latest_packet = pkt
            self.latest_overlay = pkt.color.copy()
            cv2.polylines(self.latest_overlay, [quad.astype(np.int32)], True, (0, 255, 0), 2)
            self.latest_detection_at = time.monotonic()
            p = np.asarray(res['center_m'])*1000
            self.pose_label.setText(f"Base: X={p[0]:+.2f} Y={p[1]:+.2f} Z={p[2]:+.2f} mm | Yaw={res['yaw_deg']:+.2f}° | RMSE={res['rmse_px']:.2f}px")
            self.log_signal.emit('[VISION] 静止位姿检测完成')
            return True
        except Exception as e:
            self.latest_result = None
            QMessageBox.warning(self, '视觉定位失败', str(e))
            return False

    def _fresh(self):
        if self.latest_result is None or time.monotonic()-self.latest_detection_at > 2.0:
            raise RuntimeError('视觉位姿已过期，请重新点击“检测静止位姿”')
        if self.latest_packet is not self.feed.latest():
            # 缓存检测结果必须仍在有效窗口内；保存前再次确认机器人静止。
            self._robot_snapshot()
        return self.latest_result

    def save_reference(self):
        try:
            res = self._fresh()
            ref = {k:res[k] for k in ('frame','camera_serial','center_m','yaw_deg','T_base_top','capture_time_iso','edge_mm')}
            ref['handeye_matrix'] = self.T_tcp_cam.tolist()
            base = Path.cwd()/'square_pose_data'
            base.mkdir(exist_ok=True)
            path = base/f"reference_{dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.json"
            sp.export_json(path, {'reference':ref,'matrix_path':self.matrix_path,
                                  'description':'顶面中心 in UR Base, yaw modulo90'})
            self.reference, self.reference_path = ref, path
            self.ref_label.setText('本次参考：' + str(path))
            self.reference_changed.emit(str(path))
            self.log_signal.emit('[VISION] 已记录视觉基准：' + str(path))
        except Exception as e:
            QMessageBox.warning(self, '无法保存参考', str(e))

    def load_reference_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, '加载示教参考', '', 'JSON (*.json)')
        if path:
            try:
                self.use_reference(Path(path))
            except Exception as e:
                QMessageBox.warning(self, '参考无效', str(e))

    def use_reference(self, path):
        path = Path(path)
        doc = json.loads(path.read_text(encoding='utf-8'))
        ref = doc.get('reference', doc)
        if ref.get('frame') != 'base' or len(ref.get('center_m', [])) != 3:
            raise ValueError('参考文件缺少 Base 坐标位姿')
        if self.T_tcp_cam is None or not np.allclose(ref.get('handeye_matrix'), self.T_tcp_cam, atol=1e-7):
            raise ValueError('请加载与参考文件一致的手眼标定矩阵')
        if abs(float(ref.get('edge_mm'))-self.edge.value()) > 0.01:
            raise ValueError('参考正方形边长与当前配置不一致')
        self.reference, self.reference_path = ref, path
        self.ref_label.setText('本次参考：' + str(path))
        self.reference_changed.emit(str(path))

    def save_current(self):
        try:
            res = self._fresh()
            if self.reference is None:
                raise RuntimeError('先加载本条 NPZ 所属任务的 reference.json')
            ref = self.reference
            if res['camera_serial'] != ref['camera_serial']:
                raise RuntimeError('当前相机序列号与参考不一致')
            if not np.isclose(res['edge_mm'], ref['edge_mm'], atol=0.01):
                raise RuntimeError('正方形尺寸不一致')
            p0, p1 = np.asarray(ref['center_m']), np.asarray(res['center_m'])
            delta = {'delta_xyz_mm':((p1-p0)*1000).tolist(),
                     'delta_yaw_deg':sp.equiv_delta_deg(res['yaw_deg'], ref['yaw_deg']), 'frame':'base'}
            base = Path.cwd()/'square_pose_data'/dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            base.mkdir(parents=True, exist_ok=False)
            cv2.imwrite(str(base/'rgb_overlay.png'), self.latest_overlay)
            cv2.imwrite(str(base/'rgb_raw.png'), self.latest_packet.color)
            if self.latest_packet.depth_m is not None:
                cv2.imwrite(str(base/'depth_aligned_z16.png'), np.clip(self.latest_packet.depth_m/0.001,0,65535).astype(np.uint16))
            doc = {'current':res,'reference':ref,'delta':delta,
                   'handeye_matrix_path':self.matrix_path,'K':self.latest_packet.K,
                   'D':self.latest_packet.D,'depth_saved_unit_m':0.001,
                   'notes':'Base顶面中心；Yaw模90；仅供离线轨迹计算'}
            sp.export_json(base/'pose.json', sp.make_jsonable(doc))
            self.log_signal.emit('[VISION] 已保存当前检测：' + str(base/'pose.json'))
            QMessageBox.information(self, '保存成功', str(base/'pose.json'))
        except Exception as e:
            QMessageBox.warning(self, '保存失败', str(e))

    def close(self):
        self.timer.stop()
        self.feed.stop()
        super().close()
