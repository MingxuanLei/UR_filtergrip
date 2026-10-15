# -*- coding: utf-8 -*-
"""PySide6 大画面视觉示教组件：自适应显示、正确 ROI 映射、后台 LSD 预览。
复用 square_pose 的 LSD/PnP；机器人仅在静止状态下读状态快照。
"""
from __future__ import annotations
import datetime as dt
import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal, QObject, QRectF, QSize
from PySide6.QtGui import QImage, QPixmap, QPainter, QPen, QColor
from PySide6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFileDialog, QLineEdit, QDoubleSpinBox, QMessageBox, QGroupBox, QCheckBox,
    QSplitter, QScrollArea, QSizePolicy, QGridLayout, QFrame)

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


def display_image_rect(view_w, view_h, image_w, image_h):
    """Contain/Fit 显示区域，返回 x0,y0,display_w,display_h（与 ROI 换算共用）。"""
    if min(view_w, view_h, image_w, image_h) <= 0:
        return None
    scale = min(float(view_w) / image_w, float(view_h) / image_h)
    dw, dh = image_w * scale, image_h * scale
    return ((view_w - dw) / 2.0, (view_h - dh) / 2.0, dw, dh)


def display_to_image(x, y, view_w, view_h, image_w, image_h):
    """鼠标窗口坐标 -> 原始图像坐标；黑边区域返回 None。"""
    rect = display_image_rect(view_w, view_h, image_w, image_h)
    if rect is None:
        return None
    ox, oy, dw, dh = rect
    if not (ox <= x <= ox + dw and oy <= y <= oy + dh):
        return None
    u = min(image_w - 1, max(0, int((x - ox) * image_w / dw)))
    v = min(image_h - 1, max(0, int((y - oy) * image_h / dh)))
    return (u, v)


class CameraView(QWidget):
    """可伸缩 4:3 相机预览；ROI 永远按原始像素返回。"""
    roi_selected = Signal(tuple)
    selection_cancelled = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(420, 315)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self._frame = None
        self._roi = None
        self._quad = None
        self._selecting = False
        self._start = None
        self._moving = None
        self._width = 640
        self._height = 480

    def sizeHint(self):
        return QSize(920, 690)

    def set_frame(self, bgr, roi=None, quad=None):
        if bgr is None:
            self._frame = None
            self.update()
            return
        h, w = bgr.shape[:2]
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        self._frame = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._width, self._height = w, h
        self._roi = tuple(roi) if roi is not None else None
        self._quad = np.asarray(quad, dtype=float).reshape(4, 2).copy() if quad is not None else None
        self.update()

    def set_selection(self, active):
        self._selecting = bool(active)
        self._start = None
        self._moving = None
        self.setCursor(Qt.CrossCursor if active else Qt.ArrowCursor)
        self.update()

    def reset_overlay(self):
        self._roi = None
        self._quad = None
        self.update()

    def _to_image(self, event):
        pos = event.position()
        return display_to_image(pos.x(), pos.y(), self.width(), self.height(),
                                self._width, self._height)

    def _project(self, u, v, rect):
        ox, oy, dw, dh = rect
        return (ox + u * dw / self._width, oy + v * dh / self._height)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor('#191d21'))
        if self._frame is None:
            p.setPen(QColor('#b6bec6'))
            p.drawText(self.rect(), Qt.AlignCenter, '请启动 D435i 以显示图像')
            return
        rect = display_image_rect(self.width(), self.height(), self._width, self._height)
        if rect is None:
            return
        ox, oy, dw, dh = rect
        p.setRenderHint(QPainter.SmoothPixmapTransform, True)
        p.drawImage(QRectF(ox, oy, dw, dh), self._frame)
        p.setRenderHint(QPainter.Antialiasing, True)
        if self._roi is not None:
            x0, y0, x1, y1 = self._roi
            a = self._project(x0, y0, rect)
            b = self._project(x1, y1, rect)
            p.setPen(QPen(QColor('#ffca45'), 2))
            p.drawRect(QRectF(a[0], a[1], b[0] - a[0], b[1] - a[1]))
        if self._quad is not None:
            p.setPen(QPen(QColor('#31f17a'), 2))
            pts = [self._project(float(x), float(y), rect) for x, y in self._quad]
            for i in range(4):
                a, b = pts[i], pts[(i + 1) % 4]
                p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
        if self._selecting and self._start is not None and self._moving is not None:
            a = self._project(*self._start, rect)
            b = self._project(*self._moving, rect)
            p.setPen(QPen(QColor('#32c8ff'), 2, Qt.DashLine))
            p.drawRect(QRectF(min(a[0], b[0]), min(a[1], b[1]),
                              abs(a[0] - b[0]), abs(a[1] - b[1])))

    def mousePressEvent(self, event):
        if self._selecting and event.button() == Qt.LeftButton and self._frame is not None:
            self._start = self._to_image(event)
            self._moving = self._start
            self.update()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._selecting and self._start is not None:
            # 拖到图像黑边外时夹取最近的有效像素，避免坐标超界。
            pos = event.position()
            rect = display_image_rect(self.width(), self.height(), self._width, self._height)
            if rect is not None:
                ox, oy, dw, dh = rect
                x = min(max(pos.x(), ox), ox + dw)
                y = min(max(pos.y(), oy), oy + dh)
                self._moving = display_to_image(x, y, self.width(), self.height(),
                                                self._width, self._height)
                self.update()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if not (self._selecting and self._start is not None and event.button() == Qt.LeftButton):
            super().mouseReleaseEvent(event)
            return
        self.mouseMoveEvent(event)
        a, b = self._start, self._moving
        self.set_selection(False)
        if a is None or b is None:
            self.selection_cancelled.emit('ROI 终点不在图像内，请重新框选')
            return
        x0, x1 = sorted((a[0], b[0]))
        y0, y1 = sorted((a[1], b[1]))
        if x1 - x0 < 45 or y1 - y0 < 45:
            self.selection_cancelled.emit('ROI 原始像素宽高应至少为 45，请重新框选')
            return
        self.roi_selected.emit((x0, y0, x1, y1))



class VisionTeachPanel(QWidget):
    log_signal = Signal(str)
    reference_changed = Signal(str)
    preview_quad_ready = Signal(object, object)

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
        self._preview_quad = None
        self._preview_quad_time = 0.0
        self._preview_roi = None
        self._lsd_busy = False
        self._last_lsd_started = 0.0
        self._closing = False
        self.preview_quad_ready.connect(self._receive_preview_quad)
        self._build()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_preview)
        self.timer.start(180)

    def _build(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 6)
        root.setSpacing(6)
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

        # 左侧约 70%：自适应相机大画面 + ROI 功能按钮。
        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        lv.setSpacing(6)
        title = QLabel('D435i 彩色图像 · ROI（显示自动缩放，保持 4:3）')
        title.setStyleSheet('font-weight:600;')
        lv.addWidget(title)
        self.image_label = CameraView()
        self.image_label.roi_selected.connect(self._roi_selected)
        self.image_label.selection_cancelled.connect(self._roi_invalid)
        lv.addWidget(self.image_label, 1)
        roi_buttons = QHBoxLayout()
        for name, callback in [('框选 ROI', self.select_roi), ('清除 ROI', self.clear_roi),
                               ('检测静止位姿', self.detect_now)]:
            b = QPushButton(name)
            b.clicked.connect(callback)
            roi_buttons.addWidget(b)
        lv.addLayout(roi_buttons)
        self.preview_tip = QLabel('黄色：ROI；绿色：LSD 四角。框选结果使用相机原始 640×480 像素坐标。')
        self.preview_tip.setWordWrap(True)
        lv.addWidget(self.preview_tip)

        # 右侧约 30%：按连接、检测参数、位姿、任务组织；窄窗口可滚动。
        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setFrameShape(QFrame.NoFrame)
        right_scroll.setMinimumWidth(315)
        right = QWidget()
        rv = QVBoxLayout(right)
        rv.setContentsMargins(6, 0, 6, 0)
        rv.setSpacing(10)

        cam_group = QGroupBox('1. 相机与手眼标定')
        cv = QVBoxLayout(cam_group)
        self.serial = QLineEdit()
        self.serial.setPlaceholderText('RealSense 序列号；留空自动选择')
        cv.addWidget(self.serial)
        cr = QHBoxLayout()
        for name, callback in [('启动相机', self.start_camera), ('停止相机', self.stop_camera)]:
            b = QPushButton(name)
            b.clicked.connect(callback)
            cr.addWidget(b)
        cv.addLayout(cr)
        btn_matrix = QPushButton('加载 T_tcp_cam 手眼矩阵')
        btn_matrix.clicked.connect(self.load_matrix)
        cv.addWidget(btn_matrix)
        self.matrix_label = QLabel('手眼矩阵：未加载')
        self.matrix_label.setWordWrap(True)
        cv.addWidget(self.matrix_label)
        rv.addWidget(cam_group)

        detect_group = QGroupBox('2. 目标检测参数')
        dv = QGridLayout(detect_group)
        self.edge = QDoubleSpinBox()
        self.edge.setRange(10, 500)
        self.edge.setValue(80)
        self.edge.setSuffix(' mm')
        self.gap = QDoubleSpinBox()
        self.gap.setRange(0, 500)
        self.gap.setValue(35)
        self.gap.setSuffix(' mm')
        self.auto = QCheckBox('预览自动 LSD 检测（后台计算）')
        self.auto.setChecked(True)
        dv.addWidget(QLabel('正方形边长'), 0, 0)
        dv.addWidget(self.edge, 0, 1)
        dv.addWidget(QLabel('顶面与底板高差'), 1, 0)
        dv.addWidget(self.gap, 1, 1)
        dv.addWidget(self.auto, 2, 0, 1, 2)
        rv.addWidget(detect_group)

        pose_group = QGroupBox('3. 当前检测结果（UR5e Base）')
        pv = QVBoxLayout(pose_group)
        self.pose_label = QLabel('位姿：未检测')
        self.pose_label.setWordWrap(True)
        self.pose_label.setMinimumHeight(44)
        pv.addWidget(self.pose_label)
        rv.addWidget(pose_group)

        task_group = QGroupBox('4. 视觉示教参考 / 任务绑定')
        tv = QVBoxLayout(task_group)
        btn_ref = QPushButton('记录视觉基准 reference.json')
        btn_ref.clicked.connect(self.save_reference)
        btn_load = QPushButton('加载任务参考 JSON')
        btn_load.clicked.connect(self.load_reference_dialog)
        btn_save = QPushButton('保存当前 pose.json')
        btn_save.clicked.connect(self.save_current)
        for btn in (btn_ref, btn_load, btn_save):
            tv.addWidget(btn)
        self.ref_label = QLabel('当前任务参考：未设置')
        self.ref_label.setWordWrap(True)
        tv.addWidget(self.ref_label)
        rv.addWidget(task_group)

        note = QLabel('注意：工件未被触碰、机器人静止时记录基准；每条 NPZ 绑定自己的参考。'
                      '视觉定位不会向机器人发送运动指令。')
        note.setWordWrap(True)
        rv.addWidget(note)
        rv.addStretch(1)
        right_scroll.setWidget(right)
        split.addWidget(left)
        split.addWidget(right_scroll)
        split.setStretchFactor(0, 7)
        split.setStretchFactor(1, 3)
        split.setSizes([1030, 420])
        root.addWidget(split, 1)
        self._vision_splitter = split

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
        self.image_label.set_frame(None)
        self._preview_quad = None
        self._preview_roi = None

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
        if self.feed.latest() is None:
            self.log_signal.emit('[VISION] 请先启动相机并等待第一帧图像')
            return
        self.image_label.set_selection(True)
        self.log_signal.emit('[VISION] 在左侧大画面按下并拖动鼠标；会自动换算到原始图像像素')

    def _roi_selected(self, roi):
        self.roi = tuple(roi)
        self.latest_result = None
        self._preview_quad = None
        self._preview_roi = None
        self.log_signal.emit(f'[VISION] ROI={self.roi}（原始相机图像坐标）')

    def _roi_invalid(self, message):
        self.log_signal.emit('[VISION] ' + message)

    def clear_roi(self):
        self.roi = None
        self.latest_result = None
        self._preview_quad = None
        self._preview_roi = None
        self.image_label.reset_overlay()
        self.image_label.set_selection(False)

    def _receive_preview_quad(self, result, stamp):
        # 只消费最新 ROI 的结果，防止 ROI 更新后显示旧四角。
        self._lsd_busy = False
        if self._closing:
            return
        roi, quad = result
        if self.roi == roi and self.auto.isChecked():
            self._preview_quad = quad
            self._preview_roi = roi
            self._preview_quad_time = time.monotonic()

    def _start_lsd_preview(self, color, roi):
        self._lsd_busy = True
        self._last_lsd_started = time.monotonic()
        color = color.copy()

        def worker():
            quad = None
            try:
                found = sp.detect_quad_lsd(color, roi)
                if found is not None:
                    quad = np.asarray(found, dtype=float).reshape(4, 2)
            except Exception:
                pass
            self.preview_quad_ready.emit((roi, quad), time.monotonic())

        threading.Thread(target=worker, name='vision_preview_lsd', daemon=True).start()

    def refresh_preview(self):
        # 预览页面不可见时不触发重型视觉计算。相机采集线程仍可独立工作。
        if not self.isVisible():
            return
        pkt = self.feed.latest()
        if pkt is None:
            return
        if self.roi is not None and self.auto.isChecked() and not self._lsd_busy:
            if time.monotonic() - self._last_lsd_started >= 0.35:
                self._start_lsd_preview(pkt.color, tuple(self.roi))
        quad = None
        if (self.auto.isChecked() and self._preview_roi == self.roi
                and time.monotonic() - self._preview_quad_time < 1.0):
            quad = self._preview_quad
        self.image_label.set_frame(pkt.color, self.roi, quad)

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

    def shutdown(self):
        """退出前停止预览与相机；后台检测结果不得再更新控件。"""
        self._closing = True
        self.timer.stop()
        self.feed.stop()

    def close(self):
        self.shutdown()
        super().close()
