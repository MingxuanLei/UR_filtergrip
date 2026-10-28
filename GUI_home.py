"""
UR5e 指定零位 / 同构遥操作 / 示教记录 / 示教回放 GUI。

本文件只负责界面和用户交互；控制逻辑位于 ur_home.py。

推荐目录结构：
    GUI_home.py
    ur_home.py
    DMMotor.py
    Robot.py
    USBCANFD.py
    GripperController.py
    zlgcan.py
    zlgcan.dll
"""

from __future__ import annotations

import math
import os
import sys
import threading

from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtGui import QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QDoubleSpinBox,
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTabWidget,
    QScrollArea,
    QFrame,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from vision_panel2 import VisionTeachPanel
from ur_pose_xy12 import create_visual_task
from pathlib import Path

from ur_home import (
    ARM_DOF,
    TOTAL_MOTOR_NUM,
    GRIPPER_FILTER_ALPHA,
    GRIPPER_SEND_DEADBAND,
    GRIPPER_TARGET_PERIOD_S,
    PREP_PV_VEL_DEFAULT,
    PREP_TOL_DEFAULT,
    PREP_TIMEOUT_DEFAULT,
    RECORD_DIR_NAME,
    REPLAY_MASTER_PV_VEL_DEFAULT,
    REPLAY_TOOL_KD_DEFAULT,
    REPLAY_TOOL_KP_DEFAULT,
    TAKEOVER_TORQUE_THRESHOLD_SCALE_DEFAULT,
    SCRIPT_DIR,
    UR_DEFAULT_IP,
    UR_HOME_Q_DEG,
    UR_HOME_SPEED_DEFAULT,
    UR_HOME_ACCEL_DEFAULT,
    UR_HOME_TIMEOUT_DEFAULT,
    Robot,
    TeleopCoordinator,
)


class MainWindow(QMainWindow):
    worker_done = Signal(str, bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("UR同构遥操作 · 指定零位 · 视觉示教 · PV回放")
        self.resize(1560, 920)
        self.setMinimumSize(1080, 680)

        self.coordinator = TeleopCoordinator()
        self.coordinator.log_signal.connect(self.append_log)
        self.coordinator.state_signal.connect(self.set_state)
        self.worker_done.connect(self.on_worker_done)

        self.worker_running = False
        self._home_worker_running = False
        self._home_stop_event = None
        self._cleanup_after_home = False
        self._pending_reference = None
        self._record_reference = None
        self._build_ui()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh_status)
        self.timer.start(150)

    # ============================================================
    # 总体布局
    # ============================================================
    def _build_ui(self):
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        # 顶部统一状态栏
        self.state_label = QLabel("状态：等待初始化")
        self.state_label.setAlignment(Qt.AlignCenter)
        self.state_label.setStyleSheet(
            "font-size: 16px; font-weight: 700; padding: 7px; "
            "border: 1px solid #999; border-radius: 4px;"
        )
        outer.addWidget(self.state_label)

        # 两个一级页面：遥操作/回放 与 大画面视觉示教，不再在下方状态 Tab 中嵌套视觉。
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)
        self.connection_label = QLabel(
            "主端：未初始化 | UR5e：未连接 | 夹爪：未连接 | "
            "遥操作：停止 | 记录：停止 | 回放：停止"
        )
        self.connection_label.setWordWrap(True)
        self.connection_label.setStyleSheet("font-weight: 600; padding: 3px;")
        top_bar.addWidget(self.connection_label, 1)
        # 切换到视觉页面后仍能使用原有安全停止动作。
        self.btn_safe_stop_global = QPushButton("安全停止全部控制")
        self.btn_safe_stop_global.setToolTip("与遥操作页中的安全停止相同：关闭控制与设备连接")
        self.btn_safe_stop_global.clicked.connect(self.safe_stop_async)
        top_bar.addWidget(self.btn_safe_stop_global)
        outer.addLayout(top_bar)

        self.main_tabs = QTabWidget()
        self.main_tabs.setDocumentMode(True)

        # ----- 主页面 1：原有遥操作、记录、PV回放及状态表 -----
        teleop_page = QWidget()
        teleop_layout = QVBoxLayout(teleop_page)
        teleop_layout.setContentsMargins(4, 8, 4, 4)
        teleop_layout.setSpacing(6)

        controls = QWidget()
        controls_grid = QGridLayout(controls)
        controls_grid.setContentsMargins(0, 0, 0, 0)
        controls_grid.setHorizontalSpacing(10)
        controls_grid.setVerticalSpacing(8)

        col_device = QWidget()
        col_device_layout = QVBoxLayout(col_device)
        col_device_layout.setContentsMargins(0, 0, 0, 0)
        col_device_layout.setSpacing(8)
        self._build_connection_group(col_device_layout)
        self._build_prepare_group(col_device_layout)
        col_device_layout.addStretch(1)

        col_teleop = QWidget()
        col_teleop_layout = QVBoxLayout(col_teleop)
        col_teleop_layout.setContentsMargins(0, 0, 0, 0)
        col_teleop_layout.setSpacing(8)
        self._build_teleop_group(col_teleop_layout)
        self._build_record_group(col_teleop_layout)
        self._build_gripper_mapping_group(col_teleop_layout)
        col_teleop_layout.addStretch(1)

        col_replay = QWidget()
        col_replay_layout = QVBoxLayout(col_replay)
        col_replay_layout.setContentsMargins(0, 0, 0, 0)
        col_replay_layout.setSpacing(8)
        self._build_replay_group(col_replay_layout)
        col_replay_layout.addStretch(1)

        controls_grid.addWidget(col_device, 0, 0)
        controls_grid.addWidget(col_teleop, 0, 1)
        controls_grid.addWidget(col_replay, 0, 2)
        controls_grid.setColumnStretch(0, 1)
        controls_grid.setColumnStretch(1, 1)
        controls_grid.setColumnStretch(2, 1)

        # 遥操作页面的参数区可滚动，避免笔记本窗口高度不足时按钮消失。
        controls_scroll = QScrollArea()
        controls_scroll.setWidgetResizable(True)
        controls_scroll.setFrameShape(QFrame.NoFrame)
        controls_scroll.setWidget(controls)
        controls_scroll.setMinimumHeight(260)

        self.status_tabs = QTabWidget()
        self._build_joint_status_tab()
        self._build_motor_status_tab()
        self.status_tabs.setMinimumHeight(175)

        teleop_splitter = QSplitter(Qt.Vertical)
        teleop_splitter.addWidget(controls_scroll)
        teleop_splitter.addWidget(self.status_tabs)
        teleop_splitter.setChildrenCollapsible(False)
        teleop_splitter.setStretchFactor(0, 3)
        teleop_splitter.setStretchFactor(1, 2)
        teleop_splitter.setSizes([480, 260])
        teleop_layout.addWidget(teleop_splitter)
        self.main_tabs.addTab(teleop_page, "遥操作 / 示教记录 / PV回放")

        # ----- 主页面 2：视觉采集、ROI、参考位姿及任务关联 -----
        self.vision_panel = VisionTeachPanel(self.coordinator)
        self.vision_panel.log_signal.connect(self.append_log)
        self.vision_panel.reference_changed.connect(self._on_reference_changed)
        self.main_tabs.addTab(self.vision_panel, "视觉示教 / 任务绑定")

        # 日志不再和视觉预览平分高度：默认只占下方一小块，支持折叠及拖拽。
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setPlaceholderText("运行日志将在这里显示……")
        self.log_box.setMinimumHeight(0)
        self.log_box.setMaximumHeight(240)
        self.log_toggle = QPushButton("隐藏日志")
        self.log_toggle.setCheckable(True)
        self.log_toggle.toggled.connect(self._toggle_log)
        self.log_toggle.setMaximumWidth(110)

        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(self.main_tabs)
        main_splitter.addWidget(self.log_box)
        main_splitter.setStretchFactor(0, 9)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setChildrenCollapsible(True)
        main_splitter.setSizes([740, 110])
        self._main_splitter = main_splitter
        outer.addWidget(main_splitter, 1)
        log_bar = QHBoxLayout()
        log_bar.addStretch(1)
        log_bar.addWidget(self.log_toggle)
        outer.addLayout(log_bar)
        self.setCentralWidget(central)

    def _toggle_log(self, hidden: bool):
        self.log_box.setVisible(not hidden)
        self.log_toggle.setText("显示日志" if hidden else "隐藏日志")

    # ============================================================
    # 第一列：设备连接与准备
    # ============================================================
    def _build_connection_group(self, parent):
        box = QGroupBox("1. 设备连接")
        g = QGridLayout(box)
        g.setColumnStretch(1, 1)

        self.ip_edit = QLineEdit(UR_DEFAULT_IP)
        self.btn_master_init = QPushButton("初始化主端 CAN + 7电机")
        self.btn_ur_connect = QPushButton("连接 UR5e RTDE")
        self.btn_gripper_connect = QPushButton("连接末端夹爪 (:54321)")
        self.btn_read_ur = QPushButton("读取 UR5e 当前关节角")

        self.btn_master_init.clicked.connect(
            lambda: self.run_async("主端初始化", self.coordinator.master.initialize)
        )
        self.btn_ur_connect.clicked.connect(self.connect_ur_async)
        self.btn_gripper_connect.clicked.connect(self.connect_gripper_async)
        self.btn_read_ur.clicked.connect(self.read_ur_now)

        g.addWidget(QLabel("UR5e IP"), 0, 0)
        g.addWidget(self.ip_edit, 0, 1)
        g.addWidget(self.btn_master_init, 1, 0, 1, 2)
        g.addWidget(self.btn_ur_connect, 2, 0, 1, 2)
        g.addWidget(self.btn_gripper_connect, 3, 0, 1, 2)
        g.addWidget(self.btn_read_ur, 4, 0, 1, 2)
        parent.addWidget(box)

    def _build_prepare_group(self, parent):
        box = QGroupBox("2. 准备模式")
        g = QGridLayout(box)
        g.setColumnStretch(1, 1)

        home_box = QGroupBox("UR5e 指定零位（角度可修改）")
        home_grid = QGridLayout(home_box)
        self.home_q_inputs = []
        for i, value in enumerate(UR_HOME_Q_DEG):
            spin = self._make_spin(-360.0, 360.0, value, 1, 1.0)
            spin.setSuffix("°")
            spin.setToolTip(f"UR5e J{i + 1} 绝对目标角度；不修改编码器零点")
            self.home_q_inputs.append(spin)
            row, col = divmod(i, 2)
            home_grid.addWidget(QLabel(f"J{i + 1}"), row, col * 2)
            home_grid.addWidget(spin, row, col * 2 + 1)
        self.home_speed = self._make_spin(0.05, 0.5, UR_HOME_SPEED_DEFAULT, 2, 0.05)
        self.home_accel = self._make_spin(0.05, 1.0, UR_HOME_ACCEL_DEFAULT, 2, 0.05)
        self.home_timeout = self._make_spin(5.0, 600.0, UR_HOME_TIMEOUT_DEFAULT, 1, 5.0)
        home_grid.addWidget(QLabel("速度(rad/s)"), 3, 0, 1, 2)
        home_grid.addWidget(self.home_speed, 3, 2, 1, 2)
        home_grid.addWidget(QLabel("加速度(rad/s²)"), 4, 0, 1, 2)
        home_grid.addWidget(self.home_accel, 4, 2, 1, 2)
        home_grid.addWidget(QLabel("回零超时(s)"), 5, 0, 1, 2)
        home_grid.addWidget(self.home_timeout, 5, 2, 1, 2)
        self.btn_ur_home = QPushButton("UR5e 回到指定零位")
        self.btn_ur_home.setEnabled(False)
        self.btn_stop_home = QPushButton("停止回零")
        self.btn_stop_home.setEnabled(False)
        self.btn_ur_home.clicked.connect(self.home_ur_async)
        self.btn_stop_home.clicked.connect(self.stop_home_now)
        home_grid.addWidget(self.btn_ur_home, 6, 0, 1, 2)
        home_grid.addWidget(self.btn_stop_home, 6, 2, 1, 2)
        home_note = QLabel("先确认回零路径无障碍。回零成功后，再点击下方主端 PV 对齐；也可直接使用原有对齐流程。")
        home_note.setWordWrap(True)
        home_grid.addWidget(home_note, 7, 0, 1, 4)
        g.addWidget(home_box, 0, 0, 1, 2)

        self.prep_vel = QDoubleSpinBox()
        self.prep_vel.setRange(0.05, 1.0)
        self.prep_vel.setDecimals(2)
        self.prep_vel.setSingleStep(0.05)
        self.prep_vel.setValue(PREP_PV_VEL_DEFAULT)

        self.prep_tol = QDoubleSpinBox()
        self.prep_tol.setRange(0.005, 0.20)
        self.prep_tol.setDecimals(3)
        self.prep_tol.setSingleStep(0.005)
        self.prep_tol.setValue(PREP_TOL_DEFAULT)

        self.prep_timeout = QDoubleSpinBox()
        self.prep_timeout.setRange(5.0, 120.0)
        self.prep_timeout.setDecimals(1)
        self.prep_timeout.setValue(PREP_TIMEOUT_DEFAULT)

        self.btn_prepare = QPushButton("读取 UR5e → 主端 PV 对齐")
        self.btn_prepare.clicked.connect(self.prepare_async)

        note = QLabel("准备阶段只对齐主端1~6轴；第7号保持MIT零力矩。")
        note.setWordWrap(True)

        g.addWidget(QLabel("主端PV速度(rad/s)"), 1, 0)
        g.addWidget(self.prep_vel, 1, 1)
        g.addWidget(QLabel("主端到位容差(rad)"), 2, 0)
        g.addWidget(self.prep_tol, 2, 1)
        g.addWidget(QLabel("主端对齐超时(s)"), 3, 0)
        g.addWidget(self.prep_timeout, 3, 1)
        g.addWidget(self.btn_prepare, 4, 0, 1, 2)
        g.addWidget(note, 5, 0, 1, 2)
        parent.addWidget(box)

    # ============================================================
    # 第二列：遥操作、记录、夹爪映射
    # ============================================================
    def _build_teleop_group(self, parent):
        box = QGroupBox("3. 遥操作")
        v = QVBoxLayout(box)

        self.btn_start_teleop = QPushButton("开始遥操作（1~6轴1:1 + 7号夹爪）")
        self.btn_stop_teleop = QPushButton("停止遥操作")
        self.btn_safe_stop = QPushButton("安全停止全部控制")

        self.btn_start_teleop.clicked.connect(
            lambda: self.run_async("启动遥操作", self.coordinator.start_teleop)
        )
        self.btn_stop_teleop.clicked.connect(
            lambda: self.run_async("停止遥操作", self.coordinator.stop_teleop)
        )
        self.btn_safe_stop.clicked.connect(self.safe_stop_async)

        note = QLabel(
            "1~6轴按相对关节变化1:1映射；第7轴 0~1 rad 映射夹爪 0~1 开度。"
        )
        note.setWordWrap(True)

        v.addWidget(self.btn_start_teleop)
        v.addWidget(self.btn_stop_teleop)
        v.addWidget(self.btn_safe_stop)
        v.addWidget(note)
        parent.addWidget(box)

    def _build_record_group(self, parent):
        box = QGroupBox("4. 遥操作示教记录")
        v = QVBoxLayout(box)

        self.btn_start_record = QPushButton("开始示教记录")
        self.btn_stop_record = QPushButton("停止记录并保存")
        self.record_file_label = QLabel("当前示教文件：-")
        self.record_file_label.setWordWrap(True)

        self.btn_start_record.clicked.connect(self.start_recording_now)
        self.task_file_label = QLabel("视觉任务包：尚未创建")
        self.task_file_label.setWordWrap(True)
        v.addWidget(self.task_file_label)
        self.btn_stop_record.clicked.connect(self.stop_recording_now)

        note = QLabel("仅在遥操作运行时记录 UR5e actual_q 与夹爪 actual_open。")
        note.setWordWrap(True)

        v.addWidget(self.btn_start_record)
        v.addWidget(self.btn_stop_record)
        v.addWidget(note)
        v.addWidget(self.record_file_label)
        parent.addWidget(box)

    def _build_gripper_mapping_group(self, parent):
        box = QGroupBox("5. 第7轴 / 夹爪状态")
        v = QVBoxLayout(box)

        self.tool_mapping_label = QLabel(
            "0.0 rad = 完全闭合；1.0 rad = 完全张开\n"
            f"低通 α={GRIPPER_FILTER_ALPHA:.2f}，死区={GRIPPER_SEND_DEADBAND:.2f}，"
            f"目标更新≤{1.0 / GRIPPER_TARGET_PERIOD_S:.0f} Hz"
        )
        self.tool_mapping_label.setWordWrap(True)

        self.tool_live_label = QLabel("M7: - | 映射: - | 滤波: - | 夹爪: -")
        self.tool_live_label.setWordWrap(True)
        self.tool_live_label.setStyleSheet("font-weight: 700;")

        v.addWidget(self.tool_mapping_label)
        v.addWidget(self.tool_live_label)
        parent.addWidget(box)

    # ============================================================
    # 第三列：回放
    # ============================================================
    def _build_replay_group(self, parent):
        box = QGroupBox("6. UR5e 示教回放 + 主端 PV 跟随")
        g = QGridLayout(box)
        g.setColumnStretch(1, 1)

        self.replay_pv_vel = self._make_spin(0.05, 1.00, REPLAY_MASTER_PV_VEL_DEFAULT, 2, 0.05)
        self.replay_tool_kp = self._make_spin(0.0, 100.0, REPLAY_TOOL_KP_DEFAULT, 2, 0.5)
        self.replay_tool_kd = self._make_spin(0.0, 5.0, REPLAY_TOOL_KD_DEFAULT, 2, 0.05)
        self.torque_threshold_scale = self._make_spin(1.00, 3.00, TAKEOVER_TORQUE_THRESHOLD_SCALE_DEFAULT, 3, 0.05)

        self.takeover_check = QCheckBox("允许主端人工介入并自动接管（力矩检测）")
        self.takeover_check.setChecked(True)

        self.btn_load_replay = QPushButton("加载示教轨迹 .npz")
        self.btn_start_replay = QPushButton("开始示教回放")
        self.btn_stop_replay = QPushButton("停止示教回放")

        self.btn_load_replay.clicked.connect(self.load_replay_file)
        self.btn_start_replay.clicked.connect(self.start_replay_async)
        self.btn_stop_replay.clicked.connect(
            lambda: self.run_async("停止示教回放", self.coordinator.stop_replay)
        )

        note = QLabel(
            "回放：UR5e先回到轨迹起点 → 主端PV对齐 → 正式回放时J1~J6保持PV高精度同步跟随；"
            "M7仍以MIT位置方式跟随记录夹爪开度。\n"
            "人工介入检测不再主要依赖位置偏差，而是200Hz读取J1~J6和M7的电机Torque，"
            "建立正常回放力矩基线并检测突变/持续力矩残差。阈值倍率>1更不敏感，<1更敏感。\n"
            "确认人工介入后：Replay立即撤权 → 主端J1~J6从PV切换到MIT+重力补偿 → "
            "UR5e进入绝对关节角1:1遥操作；不会重新建立接管零点。"
        )
        note.setWordWrap(True)

        g.addWidget(QLabel("主端PV速度上限(rad/s)"), 0, 0)
        g.addWidget(self.replay_pv_vel, 0, 1)
        g.addWidget(QLabel("M7 MIT Kp"), 1, 0)
        g.addWidget(self.replay_tool_kp, 1, 1)
        g.addWidget(QLabel("M7 MIT Kd"), 2, 0)
        g.addWidget(self.replay_tool_kd, 2, 1)
        g.addWidget(QLabel("力矩阈值倍率(>1更不敏感)"), 3, 0)
        g.addWidget(self.torque_threshold_scale, 3, 1)
        g.addWidget(self.takeover_check, 4, 0, 1, 2)
        g.addWidget(self.btn_load_replay, 5, 0, 1, 2)
        g.addWidget(self.btn_start_replay, 6, 0, 1, 2)
        g.addWidget(self.btn_stop_replay, 7, 0, 1, 2)
        g.addWidget(note, 8, 0, 1, 2)
        parent.addWidget(box)

    @staticmethod
    def _make_spin(lo, hi, value, decimals, step):
        spin = QDoubleSpinBox()
        spin.setRange(lo, hi)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setValue(value)
        return spin

    # ============================================================
    # 状态 Tab
    # ============================================================
    def _build_joint_status_tab(self):
        tab = QWidget()
        v = QVBoxLayout(tab)
        v.setContentsMargins(5, 5, 5, 5)

        self.joint_table = QTableWidget(ARM_DOF, 6)
        self.joint_table.setHorizontalHeaderLabels(
            [
                "Joint",
                "主端 DH(rad)",
                "主端(deg)",
                "UR5e(rad)",
                "UR5e(deg)",
                "主-UR等价误差(rad)",
            ]
        )
        self.joint_table.verticalHeader().setVisible(False)
        self.joint_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.joint_table.setAlternatingRowColors(True)
        for r in range(ARM_DOF):
            for c in range(6):
                item = QTableWidgetItem("")
                item.setTextAlignment(Qt.AlignCenter)
                self.joint_table.setItem(r, c, item)
        v.addWidget(self.joint_table)
        self.status_tabs.addTab(tab, "关节状态")

    def _build_motor_status_tab(self):
        tab = QWidget()
        v = QVBoxLayout(tab)
        v.setContentsMargins(5, 5, 5, 5)

        self.motor_table = QTableWidget(TOTAL_MOTOR_NUM, 7)
        self.motor_table.setHorizontalHeaderLabels(
            ["ID", "Mode", "Enable", "ERR", "Pos(rad)", "Vel", "Recv"]
        )
        self.motor_table.verticalHeader().setVisible(False)
        self.motor_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.motor_table.setAlternatingRowColors(True)
        for r in range(TOTAL_MOTOR_NUM):
            for c in range(7):
                item = QTableWidgetItem("")
                item.setTextAlignment(Qt.AlignCenter)
                self.motor_table.setItem(r, c, item)
        v.addWidget(self.motor_table)
        self.status_tabs.addTab(tab, "电机状态")

    # ============================================================
    # 通用 UI 行为
    # ============================================================
    def append_log(self, text: str):
        self.log_box.append(text)
        self.log_box.moveCursor(QTextCursor.End)

    def set_state(self, state: str):
        self.state_label.setText("状态：" + state)

    def set_action_buttons_enabled(self, enabled: bool):
        for btn in (
            self.btn_master_init,
            self.btn_ur_connect,
            self.btn_gripper_connect,
            self.btn_read_ur,
            self.btn_ur_home,
            self.btn_prepare,
            self.btn_start_teleop,
            self.btn_start_record,
            self.btn_load_replay,
            self.btn_start_replay,
            self.btn_safe_stop,
            self.btn_safe_stop_global,
        ):
            btn.setEnabled(enabled)

        self.takeover_check.setEnabled(enabled)
        for spin in self.home_q_inputs + [self.home_speed, self.home_accel, self.home_timeout]:
            spin.setEnabled(enabled)
        self.btn_stop_home.setEnabled(
            self._home_worker_running and self._home_stop_event is not None
            and not self._home_stop_event.is_set()
        )
        if self._home_worker_running:
            # 先取消回零，待工作线程结束后再执行全部设备清理。
            self.btn_safe_stop.setEnabled(True)
            self.btn_safe_stop_global.setEnabled(True)

        # 停止类按钮始终保留，便于异常时终止。
        self.btn_stop_teleop.setEnabled(True)
        self.btn_stop_record.setEnabled(True)
        self.btn_stop_replay.setEnabled(True)

    def run_async(self, label: str, func, *, homing: bool = False):
        if self.worker_running:
            self.append_log("[WARN] 上一个操作仍在执行")
            return

        self.worker_running = True
        self._home_worker_running = homing
        if homing:
            # 回零期间禁止在视觉页触发静止位姿采样。
            self.vision_panel.setEnabled(False)
        self.set_action_buttons_enabled(False)

        def worker():
            ok = False
            try:
                ok = bool(func())
            except Exception as e:
                self.coordinator.log(f"[ERR] {label} 异常: {e}")
                ok = False
            finally:
                self.worker_done.emit(label, ok)

        threading.Thread(target=worker, daemon=True).start()

    def on_worker_done(self, label: str, ok: bool):
        was_homing = self._home_worker_running
        self.worker_running = False
        self._home_worker_running = False
        self._home_stop_event = None
        if was_homing:
            self.vision_panel.setEnabled(True)
        self.set_action_buttons_enabled(True)
        self.append_log(f"[DONE] {label}: {'成功' if ok else '失败'}")
        self.refresh_status()
        if self._cleanup_after_home:
            self._cleanup_after_home = False
            self.safe_stop_async()

    # ============================================================
    # 按钮槽函数
    # ============================================================
    def connect_ur_async(self):
        ip = self.ip_edit.text().strip()
        self.run_async("连接UR5e", lambda: self.coordinator.ur.connect(ip))

    def connect_gripper_async(self):
        ip = self.ip_edit.text().strip()
        self.run_async("连接夹爪", lambda: self.coordinator.gripper.connect(ip))

    def read_ur_now(self):
        readout = self.coordinator.read_ur_configuration()
        if readout is None:
            QMessageBox.warning(self, "读取失败", "请先连接 UR5e。")
            return
        self.refresh_status()

    def prepare_async(self):
        vel = float(self.prep_vel.value())
        tol = float(self.prep_tol.value())
        timeout_s = float(self.prep_timeout.value())
        self.run_async(
            "准备模式",
            lambda: self.coordinator.run_preparation(vel, tol, timeout_s),
        )

    def home_ur_async(self):
        if self.worker_running:
            self.append_log("[WARN] 上一个操作仍在执行")
            return
        target_deg = [float(spin.value()) for spin in self.home_q_inputs]
        speed = float(self.home_speed.value())
        acceleration = float(self.home_accel.value())
        timeout_s = float(self.home_timeout.value())
        cancel = threading.Event()
        self._home_stop_event = cancel
        self.run_async(
            "UR5e回到指定零位",
            lambda: self.coordinator.run_ur_home(target_deg, speed, acceleration, timeout_s, cancel),
            homing=True,
        )

    def stop_home_now(self):
        if self._home_worker_running and self._home_stop_event is not None:
            self._home_stop_event.set()
            self.coordinator.request_ur_home_stop()
            self.btn_stop_home.setEnabled(False)
            self.append_log("[HOME] 已请求停止回零，等待运动停止；不会自动对齐主端")

    def _on_reference_changed(self, path):
        self._pending_reference = Path(path)
        self.append_log("[TASK] 已选定视觉参考：" + path)

    def start_recording_now(self):
        self._record_reference = None
        if self._pending_reference is not None and self._pending_reference.exists():
            self._record_reference = self._pending_reference
        ok = self.coordinator.start_recording()
        if ok:
            self.append_log("[TASK] 本次示教参考：" + (str(self._record_reference) if self._record_reference else "无；仅保存NPZ"))
        if not ok:
            QMessageBox.warning(
                self,
                "无法开始记录",
                "请确认遥操作已经启动，且当前没有示教回放。",
            )
        self.refresh_status()

    def stop_recording_now(self):
        ok = self.coordinator.stop_recording(save=True)
        if ok and self.coordinator.last_record_path:
            self.record_file_label.setText(
                "当前示教文件：" + self.coordinator.last_record_path
            )
        if ok and self.coordinator.last_record_path:
            if self._record_reference is not None:
                try:
                    task_json = create_visual_task(Path(self.coordinator.last_record_path), self._record_reference,
                                                   Path(SCRIPT_DIR)/"visual_tasks")
                    self.task_file_label.setText("视觉任务包：" + str(task_json))
                    self.append_log("[TASK] 已自动绑定 NPZ + 参考：" + str(task_json))
                except Exception as exc:
                    self.task_file_label.setText("视觉任务包：创建失败（原NPZ仍已保存）")
                    self.append_log("[TASK][ERR] 自动绑定失败：" + str(exc))
                    QMessageBox.warning(self, "视觉任务绑定失败", str(exc))
            else:
                self.task_file_label.setText("视觉任务包：本次没有参考，仅保存NPZ")
            self._record_reference = None
            self._pending_reference = None
            self.vision_panel.reference = None
            self.vision_panel.reference_path = None
            self.vision_panel.ref_label.setText("本条示教已结束；请重新记录下一条的视觉参考")
        self.refresh_status()

    def load_replay_file(self):
        base_dir = os.path.join(SCRIPT_DIR, RECORD_DIR_NAME)
        os.makedirs(base_dir, exist_ok=True)
        path, _ = QFileDialog.getOpenFileName(
            self,
            "选择示教轨迹",
            base_dir,
            "NumPy trajectory (*.npz)",
        )
        if not path:
            return
        if self.coordinator.load_trajectory(path):
            self.record_file_label.setText("当前示教文件：" + path)
        else:
            QMessageBox.warning(self, "加载失败", "示教轨迹文件无效。")
        self.refresh_status()

    def start_replay_async(self):
        pv_vel = float(self.replay_pv_vel.value())
        tool_kp = float(self.replay_tool_kp.value())
        tool_kd = float(self.replay_tool_kd.value())
        torque_scale = float(self.torque_threshold_scale.value())
        takeover_enabled = bool(self.takeover_check.isChecked())
        self.run_async(
            "启动示教回放",
            lambda: self.coordinator.start_replay(
                pv_vel, tool_kp, tool_kd, torque_scale, takeover_enabled
            ),
        )

    def safe_stop_async(self):
        if self._home_worker_running:
            self._cleanup_after_home = True
            self.stop_home_now()
            self.append_log("[STOP] 回零任务退出后将继续执行全部设备清理")
            return

        def task():
            self.coordinator.cleanup()
            return True

        self.run_async("安全停止", task)

    # ============================================================
    # 实时状态刷新
    # ============================================================
    def refresh_status(self):
        master_ok = self.coordinator.master.initialized
        ur_ok = self.coordinator.ur.connected
        gripper_ok = self.coordinator.gripper.connected
        self.btn_ur_home.setEnabled(
            ur_ok and not self.worker_running and not self.coordinator.home_running
            and not self.coordinator.teleop_running and not self.coordinator.replay_running
            and not self.coordinator.recording
            and self.coordinator.get_control_mode_name() == "IDLE"
        )

        takeover_text = (
            "已触发" if self.coordinator.takeover_triggered
            else ("已武装" if self.coordinator.takeover_armed else "待机")
        )
        # 人工介入开关只在未回放时允许修改；回放开始时参数会快照到协调器。
        self.takeover_check.setEnabled(
            (not self.coordinator.replay_running) and (not self.worker_running)
        )

        control_owner = self.coordinator.get_control_mode_name()
        self.connection_label.setText(
            f"主端：{'已初始化' if master_ok else '未初始化'} | "
            f"UR5e：{'已连接' if ur_ok else '未连接'} | "
            f"夹爪：{'已连接' if gripper_ok else '未连接'} | "
            f"控制权：{control_owner} | "
            f"遥操作：{'运行中' if self.coordinator.teleop_running else '停止'} | "
            f"记录：{'进行中' if self.coordinator.recording else '停止'} | "
            f"回放：{'进行中' if self.coordinator.replay_running else '停止'} | "
            f"介入检测：{takeover_text}"
        )

        if self.coordinator.loaded_trajectory_path:
            self.record_file_label.setText(
                "当前示教文件：" + self.coordinator.loaded_trajectory_path
            )

        q_master = self.coordinator.master.get_dh_q() if master_ok else None
        q_ur = self.coordinator.ur.get_actual_q() if ur_ok else None
        q_ur_equiv = (
            self.coordinator.ur_q_to_master_equivalent(q_ur)
            if q_ur is not None
            else None
        )

        for i in range(ARM_DOF):
            self.joint_table.item(i, 0).setText(f"J{i + 1}")

            if q_master is not None:
                self.joint_table.item(i, 1).setText(f"{q_master[i]:.4f}")
                self.joint_table.item(i, 2).setText(
                    f"{math.degrees(q_master[i]):.2f}"
                )
            else:
                self.joint_table.item(i, 1).setText("-")
                self.joint_table.item(i, 2).setText("-")

            if q_ur is not None:
                self.joint_table.item(i, 3).setText(f"{q_ur[i]:.4f}")
                self.joint_table.item(i, 4).setText(f"{math.degrees(q_ur[i]):.2f}")
            else:
                self.joint_table.item(i, 3).setText("-")
                self.joint_table.item(i, 4).setText("-")

            if q_master is not None and q_ur_equiv is not None:
                e = Robot.minor_arc_dir(q_ur_equiv[i], q_master[i])
                self.joint_table.item(i, 5).setText(f"{e:.4f}")
            else:
                self.joint_table.item(i, 5).setText("-")

        motors = self.coordinator.master.get_motor_snapshot() if master_ok else None
        for r in range(TOTAL_MOTOR_NUM):
            if motors is None or r >= len(motors):
                for c in range(7):
                    self.motor_table.item(r, c).setText("-")
                continue

            m = motors[r]
            vals = [
                str(m["id"]),
                str(m["mode"]),
                str(m["enable"]),
                str(m["err"]),
                f"{m['pos']:.4f}",
                f"{m['vel']:.4f}",
                str(m["recv"]),
            ]
            for c, text in enumerate(vals):
                self.motor_table.item(r, c).setText(text)

        q7 = self.coordinator.master.get_tool_position() if master_ok else None
        if q7 is not None and math.isfinite(q7):
            raw = self.coordinator.tool7_to_opening(q7)
            filt = self.coordinator.latest_tool_filtered_opening
            actual = (
                self.coordinator.gripper.get_actual_opening() if gripper_ok else None
            )
            filt_text = f"{filt:.3f}" if filt is not None else "-"
            actual_text = f"{actual:.3f}" if actual is not None else "-"
            self.tool_live_label.setText(
                f"M7: {q7:.4f} rad | 映射: {raw:.3f} | "
                f"滤波: {filt_text} | 夹爪: {actual_text}"
            )
        else:
            self.tool_live_label.setText("M7: - | 映射: - | 滤波: - | 夹爪: -")

    # ============================================================
    # 退出
    # ============================================================
    def closeEvent(self, event):
        if self.worker_running:
            if self._home_worker_running:
                self.stop_home_now()
            QMessageBox.warning(
                self,
                "提示",
                "当前操作仍在执行，请先停止当前操作或等待命令结束后退出。",
            )
            event.ignore()
            return

        reply = QMessageBox.question(
            self,
            "确认退出",
            "退出前将停止UR5e控制、关闭夹爪通信，并失能主端7个电机。\n是否继续？",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            event.ignore()
            return

        self.timer.stop()
        self.vision_panel.shutdown()
        try:
            self.coordinator.cleanup()
        except Exception as e:
            self.append_log(f"[WARN] 退出清理异常: {e}")
        event.accept()


def main():
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
