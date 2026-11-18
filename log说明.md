# GUI_log 使用说明

适用入口：`GUI_log.py`。本文依据当前 `GUI_log.py`、`ur_log.py` 和 `replay_log.py` 编写。

本程序保留遥操作、示教记录、主端 PV 跟随回放和视觉任务绑定功能，并在示教回放期间记录 UR5e 六个关节的位置、速度及估算加速度。每次启用记录的回放生成一个独立运行目录，可用于单工况实验及重复回放对比。

## 1. 文件与运行环境

项目目录：

```text
D:\Users\lmxlegion\Desktop\My personal project\UR_filtergrip
```

| 文件 | 作用 |
| --- | --- |
| [GUI_log.py](<D:/Users/lmxlegion/Desktop/My personal project/UR_filtergrip/GUI_log.py>) | 启动入口，提供记录开关、工况名称、保存位置和运行状态 |
| [ur_log.py](<D:/Users/lmxlegion/Desktop/My personal project/UR_filtergrip/ur_log.py>) | 设备通信、遥操作、示教回放，以及记录与回放阶段的衔接 |
| [replay_log.py](<D:/Users/lmxlegion/Desktop/My personal project/UR_filtergrip/replay_log.py>) | 反馈采集、时间戳检查、加速度估算和文件导出 |

三个文件应放在同一项目目录，使用时只启动 `GUI_log.py`。项目中还须保留现有的 `DMMotor.py`、`Robot.py`、`USBCANFD.py`、`GripperController.py`、`zlgcan.py`、`zlgcan.dll` 及相关 DLL 目录；界面还会导入 `vision_panel2.py`、`square_pose.py` 和 `ur_pose_xy12.py`。

主要 Python 依赖包括 PySide6、NumPy、SciPy、OpenCV、ur-rtde、minimalmodbus、pyserial。即使只做回放，界面的视觉组件导入仍需要相关基础依赖。使用 RealSense 相机时还需要 pyrealsense2；保留 Python 环境中的 tkinter 支持。

在 PowerShell 中可使用当前机器已有的解释器启动：

```powershell
Set-Location -LiteralPath 'D:\Users\lmxlegion\Desktop\My personal project\UR_filtergrip'
& 'D:\anaconda\envs\pytorch\python.exe' 'D:\Users\lmxlegion\Desktop\My personal project\UR_filtergrip\GUI_log.py'
```

若已激活包含上述依赖的 Python 环境，也可以运行：

```powershell
python 'D:\Users\lmxlegion\Desktop\My personal project\UR_filtergrip\GUI_log.py'
```

## 2. 最快完成一次回放记录

已有可回放的示教轨迹 `.npz` 时，按以下顺序操作：

1. 在“1. 设备连接”中确认 UR5e IP，默认值为 `192.168.3.15`。
2. 点击“初始化主端 CAN + 7电机”，等待成功。
3. 点击“连接 UR5e RTDE”，等待连接成功。
4. 点击“连接末端夹爪 (:54321)”，等待夹爪反馈正常。
5. 确认当前没有运行遥操作或示教记录。
6. 在“6. UR5e 示教回放 + 主端 PV 跟随”中点击“加载示教轨迹 .npz”。
7. 保持“回放时记录 UR5e 六关节状态”勾选，填写“工况名称”，例如 `case01` 或 `case01_fit`。
8. 确认“数据保存目录”，需要更换时点击“选择保存目录”。
9. 确认 UR5e 到轨迹起点及后续回放的运动路径，然后点击“开始示教回放”。该操作会驱动真实机械臂运动。
10. 等待回放结束，再等待“运行记录”显示“已保存”或“不完整，数据已保存”。点击“打开记录目录”查看结果。

**已有轨迹的回放不需要先点击“开始遥操作”，也不需要手动执行主端 PV 对齐。** 回放会依次完成 UR5e 移动到轨迹第一帧、夹爪同步、主端 PV 对齐，再执行正式轨迹。

若窗口中看不到记录设置，可在上方参数区向下滚动，或拖动分隔条扩大参数区域。

## 3. 两种“记录”的区别

| 功能 | 使用场景 | 操作与结果 |
| --- | --- | --- |
| “4. 遥操作示教记录” | 人工操作主端、生成以后可回放的轨迹 | 遥操作启动后点击“开始示教记录”；结束时点击“停止记录并保存”，生成示教 `.npz` |
| “回放时记录 UR5e 六关节状态” | 观察一次既有轨迹在真实 UR5e 上的执行情况 | 开始回放前勾选，记录随回放自动启动、收尾和保存 |

示教记录只允许在遥操作时进行，回放时不能再点击“开始示教记录”来代替运行数据采集。

运行日志中的 `states.npz`、`states_raw.npz` 是分析数据，**不能作为“加载示教轨迹 .npz”的输入**。回放输入需要包含 `timestamp`、`ur_q`、`gripper_open` 等示教轨迹字段。

如果需要先制作示教轨迹：完成设备连接，点击“读取 UR5e → 主端 PV 对齐”，待准备完成后启动遥操作，再开始示教记录；结束记录并保存后，停止遥操作，随后按第 2 节回放。

本入口的“准备模式”用于让主端对齐 UR5e 当前构型。它不把 UR5e 移动到预设零位；本入口也没有另外一版零位 GUI 的回零按钮。

## 4. 回放区域参数

| 控件 | 当前默认值 | 含义 |
| --- | --- | --- |
| 主端 PV 速度上限 | `1.00 rad/s` | 限制主端 J1～J6 的跟随速度，不直接设置 UR5e 轨迹回放速度 |
| M7 MIT Kp / Kd | `3.0 / 0.15` | 主端第 7 电机在回放时跟随夹爪开度的参数 |
| 力矩阈值倍率 | `1.500` | 人工介入检测阈值倍率，界面范围为 `1.000～3.000`；倍率越大越不敏感 |
| 允许主端人工介入并自动接管 | 勾选 | 触发后结束回放控制，切换到人工遥操作 |
| 回放时记录 UR5e 六关节状态 | 勾选 | 启用本次回放的关节状态记录 |
| 工况名称 | `case01` | 用于分组输出目录；同名工况重复运行会创建不同运行目录 |
| 数据保存目录 | 项目目录下的 `replay_logs` | 记录输出的根目录 |

UR5e 正式回放根据加载轨迹的时间轴发送目标。比较不同轨迹或重复实验时，应同时核对轨迹时长、机器人速度缩放、负载、工具和上述控制参数。`meta.json` 会保存部分运行参数，外部工况仍需自行记录。

如果只想沿用普通回放而不记录数据，在启动前取消“回放时记录 UR5e 六关节状态”。记录开关、工况和目录在本次运行中固定，不能通过回放中途修改来追加记录。

## 5. 采集频率与时间轴

| 项目 | 当前配置 | 说明 |
| --- | --- | --- |
| RTDE Control | `125 Hz`，周期 `0.008 s` | 原有控制配置 |
| RTDE Receive | `50 Hz`，名义间隔 `0.02 s` | 六关节实际反馈的配置频率 |
| 记录器缓存轮询 | `100 Hz` | 读取已有接收器缓存，不增加 RTDE 连接，不代表取得 100 Hz 的独立反馈 |
| GUI 刷新 | 每 `150 ms` | 用于界面状态显示，不作为数据采样时钟 |

记录器只接收新的机器人时间戳；重复读取同一反馈缓存不会重复计入有效样本。实际有效频率可能因调度、控制器或通信情况低于配置值，应结合 `effective_sample_hz`、`max_sample_interval_s`、`sampling_gap_count` 查看。

`robot_timestamp` 是机器人启动后的时间；`host_elapsed_s` 是相对本次采集启动的电脑单调时钟时间。`replay_time_s` 是相对正式回放起点的电脑时间，准备阶段一般为负数；若从未进入正式回放，则为 `NaN`。

加速度计算使用机器人时间戳的实际间隔。指令与反馈各有时间轴，不应按数组行号或 CSV 行号直接一一对应；这是软件时间戳记录，不是硬件同步测量。

## 6. 输出目录与文件

默认根目录为：

```text
D:\Users\lmxlegion\Desktop\My personal project\UR_filtergrip\replay_logs
```

每次运行的目录结构如下，其中运行编号由日期时间和随机标识组成：

```text
replay_logs/
  工况名称/
    运行编号/
      states_raw.npz
      states.npz
      states.csv
      commands.csv
      meta.json
```

工况名称会转换为适合 Windows 文件夹使用的名称，原始名称仍保存在元数据中。

| 文件 | 内容 |
| --- | --- |
| `states_raw.npz` | 加速度处理前的反馈、时间轴、阶段和指令数组备份 |
| `states.npz` | 上述数据加上估算加速度和加速度有效性标记 |
| `states.csv` | 每行一个反馈样本，可使用 Excel、Origin、MATLAB 或 Python 读取 |
| `commands.csv` | 每行一条记录到的回放指令，含帧索引、发送/返回时间和接口返回状态 |
| `meta.json` | 工况、源轨迹路径及 SHA256、控制参数、阶段事件、采样质量和结束原因 |

写文件期间可能暂时看到 `.tmp` 文件，应等待保存状态结束。`meta.json` 在启动阶段就会创建，文件存在本身不代表记录已经成功完成。运行中的原始采样主要缓存在内存里，`states_raw.npz` 在收尾时写出，不是逐帧持久化日志。

## 7. 数据字段与阶段

### 7.1 反馈字段

以下为 `states.npz` 中的主要字段。设有效反馈数量为 `N`，六关节顺序为 J1～J6。

| 字段 | 形状 | 含义与单位 |
| --- | --- | --- |
| `robot_timestamp` | `N` | 机器人时间戳，s；CSV 中列名为 `robot_timestamp_s` |
| `host_elapsed_s` | `N` | 相对采集开始的电脑时间，s |
| `replay_time_s` | `N` | 相对正式回放开始的电脑时间，s |
| `phase` | `N` | 当前阶段，见下表 |
| `q_actual` | `N × 6` | UR5e 实际关节角，rad |
| `qd_actual` | `N × 6` | UR5e 实际关节速度，rad/s |
| `qdd_est_raw` | `N × 6` | 实际速度经差分得到的加速度估计，rad/s² |
| `qdd_est` | `N × 6` | 实际速度经局部二次拟合求导得到的加速度估计，rad/s² |
| `acc_valid` | `N` | 该行估算加速度是否有效；CSV 使用 0/1 |
| `q_target` / `qd_target` / `qdd_target` | 各为 `N × 6` | 机器人控制器内部的目标位置、速度、加速度 |
| `speed_scaling` | `N` | 接收器提供的速度缩放值 |
| `gap_before` | `N` | 当前行之前是否存在超过阈值的反馈时间间隙 |
| `read_duration_s` | `N` | 读取反馈快照所用时间，s |

可选目标字段或速度缩放读取不可用时可能为 `NaN`，不应将其填成 0。CSV 将向量展开为六列，例如 `q_actual_J1_rad`、`qd_actual_J1_rad_s`、`qdd_est_J1_rad_s2`，一直到 J6。

### 7.2 阶段选择

| `phase` 值 | 含义 |
| --- | --- |
| `prepare_ur` | 回放准备，包括 UR5e 移动到轨迹起点等过程 |
| `prepare_master` | 主端 PV 对齐及进入正式跟随前的准备 |
| `replay` | 正式回放阶段 |
| `settling` | 正式回放退出、减速停止及收尾采集阶段 |

结束事件中还可能出现 `takeover`、`error`、`user_stopped`、`completed`、`shutdown` 等名称；是否存在相应数据行取决于结束边界和最后采样时间。

**单工况正式回放曲线先筛选 `phase == "replay"`。** 分析加速度时，再叠加 `acc_valid == True`。不要直接把准备移动和停止收尾阶段混入正式回放统计。

### 7.3 指令字段

`states.npz` 也包含指令数组，其长度为 `M`，通常与反馈行数 `N` 不同：

- `command_index`：源轨迹的帧索引，从 0 开始；CSV 列名为 `frame_index`。
- `command_planned_time_s`：该帧相对源轨迹首帧的规划时间；CSV 列名为 `planned_time_s`。
- `command_host_send_s` / `command_host_return_s`：相对采集开始的接口调用开始/返回时间；CSV 列名为 `host_send_s` / `host_return_s`。
- `q_cmd`：`M × 6` 的 PC 发送目标关节角，rad；CSV 展开为 `q_cmd_J1_rad`～`q_cmd_J6_rad`。
- `command_success`：接口返回状态；CSV 列名为 `success`。它不代表机械臂已经准确到达对应关节位置。

`q_cmd` 是 PC 给出的目标，`q_target` 是控制器反馈的内部目标，两者也不应直接视为同一个量。

## 8. 加速度的解释与默认参数

本程序的加速度来自实际关节速度对时间求导。`qdd_est_raw` 使用实际时间网格上的差分，`qdd_est` 使用局部二次拟合求导，默认时间窗口为 `0.1 s`。

**`qdd_est` 是估算值；`qdd_target` 是控制器目标加速度。两者都不能标注为“直接测得的实际关节加速度”。**

计算不会跨越阶段变化、非有限数据或超过阈值的采样间隙。默认间隙阈值为 `3 / 50 = 0.06 s`，代码在间隔大于该值时分段；分段首末点和不足 3 个样本的短段加速度为 `NaN`，`acc_valid` 为假。这些空值不表示机械臂的加速度为 0。

平滑窗口会影响峰值和曲线细节。比较多次实验时，应保持算法和窗口一致，并保留实际速度与原始加速度估计，便于复核。

下列为 `replay_log.py` 中 `ReplayLogConfig` 的当前默认值，普通使用无需修改：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `receive_frequency_hz` | `50.0` | 与现有 RTDE Receive 配置一致 |
| `poll_frequency_hz` | `100.0` | 本机缓存轮询频率 |
| `stale_timeout_s` | `1.0` | 持续没有新有效反馈时结束采集 |
| `settle_s` | `0.3` | 正常完成或主动停止后的额外收尾采集时间；设备清理可缩短它 |
| `smoothing_window_s` | `0.1` | 局部拟合的时间窗口 |
| `gap_factor` | `3.0` | 时间间隙阈值倍数 |
| `max_rows` | `1_000_000` | 反馈和指令缓冲各自的最大行数 |

GUI 没有这些高级参数的输入框。需要调整时，先核对记录配置与控制模块的一致性；仅修改记录器的频率数值不会改变已有 RTDE 连接，控制模块也会拒绝与当前 Receive 频率不一致的记录配置。

## 9. 完整性与结束原因

运行记录可能显示“待机”“启动中”“采集中”“计算并保存中”“已保存”“不完整，数据已保存”或“保存/记录异常”。采集、收尾和保存期间，相关配置及新回放入口会暂时禁用。

当前代码在以下条件全部满足时，将 `meta.json` 的 `complete` 设为 `true`：正常完成、记录器没有登记错误、正式回放阶段至少有 3 个反馈样本、全流程没有超过阈值的采样间隙。

因此 `complete` 是程序的数据完整性标记，不是轨迹跟踪精度或运动质量的合格证明。即使间隙只出现在准备阶段，也可能使整次记录被标为不完整。

| `end_reason` | 含义 |
| --- | --- |
| `completed` | 正常完成回放 |
| `user_stopped` | 主动停止回放 |
| `takeover` | 人工介入并触发接管，本次纯回放记录在接管边界结束 |
| `error` | 采集、回放或初始化等过程发生错误 |
| `shutdown` | 设备清理请求结束采集；若此前已记录结束原因，则通常保留先前原因 |

检查数据时，同时查看 `status`、`complete`、`end_reason`、`end_detail` 和 `errors`，再检查 `sample_count`、`replay_sample_count`、`effective_sample_hz`、`max_sample_interval_s`、`sampling_gap_count`、`invalid_reads`。

`duplicate_cache_reads` 统计本机读到重复缓存的次数，在 100 Hz 轮询、50 Hz 反馈时出现属于正常现象，不等于网络丢包次数。`sampling_gap_count` 也只说明时间戳存在间隙，不能单凭它判断具体丢样原因。

## 10. 用 Python 读取正式回放数据

将下面的 `run_dir` 替换为界面“本次记录目录”显示的完整路径；这段代码只读取文件，不连接机械臂。

```python
from pathlib import Path
import json
import numpy as np

run_dir = Path(r"D:\Users\lmxlegion\Desktop\My personal project\UR_filtergrip\replay_logs\case01\替换为实际运行编号")
meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
print("结束原因:", meta.get("end_reason"))
print("完整性:", meta.get("complete"))
print("记录错误:", meta.get("errors", []))

with np.load(run_dir / "states.npz", allow_pickle=False) as data:
    replay = data["phase"] == "replay"
    robot_t = data["robot_timestamp"][replay]
    q = data["q_actual"][replay]       # rad，N_replay × 6
    qd = data["qd_actual"][replay]     # rad/s
    qdd = data["qdd_est"][replay]     # rad/s²，估算加速度
    acc_ok = data["acc_valid"][replay]

if len(robot_t) == 0:
    raise ValueError("本次记录没有正式回放阶段，请检查结束原因")

t = robot_t - robot_t[0]  # 以第一个正式回放反馈样本为零点
q_deg = np.rad2deg(q)    # 仅用于角度制显示
good_acc = acc_ok & np.isfinite(qdd).all(axis=1)
print("正式回放反馈数:", len(t))
print("位置数组形状:", q.shape)

if good_acc.any():
    print("六关节加速度绝对峰值(rad/s^2):",
          np.max(np.abs(qdd[good_acc]), axis=0))
else:
    print("有效加速度样本不足，不计算峰值")
```

这里的 `t` 是反馈时间轴，不能直接当作 `commands.csv` 的时间轴。需要比较指令与反馈时，应使用共同的电脑相对时间字段进行时间对齐，并保留时间偏差；不要按行直接相减。

CSV 可直接导入 Excel 或 Origin。先筛选 `phase` 列为 `replay`，加速度统计再筛选 `acc_valid` 为 1，并确认相关列没有 `NaN`。

## 11. 常见问题

| 现象 | 检查方法 |
| --- | --- |
| 启动提示找不到 PySide6、cv2、scipy 等模块 | 核对当前 Python 解释器，优先使用包含原项目依赖的环境，不要混用不同环境的 `python` 和 `pip` |
| 启动或初始化时找不到 DLL | 保留原项目 DLL 和依赖目录，检查所用解释器与原驱动环境是否匹配 |
| 点击回放后提示主端或夹爪未连接 | 当前是主从同步回放入口，主端初始化、UR5e 连接、夹爪连接及有效夹爪反馈都需要满足 |
| 提示记录初始化失败，机械臂尚未开始回放 | 检查工况名称、保存目录权限、源轨迹文件，以及 RTDE 是否能提供有效反馈；启用记录时，初始化失败会阻止本次回放启动 |
| 只有 `meta.json`，没有最终 CSV/NPZ | 检查是否仍在采集或保存、是否在启动阶段失败；查看元数据状态和 GUI 日志 |
| 只有 `states_raw.npz` 等部分文件 | 可能在后处理或后续文件导出阶段失败；保留已有文件，并检查 `errors` 和磁盘状态 |
| 显示“不完整，数据已保存” | 先看结束原因；主动停止和人工接管通常不是完整回放，再检查反馈间隙、错误和正式回放样本数 |
| 加速度存在空值或较大的尖峰 | 先看 `acc_valid`、分段边界、采样间隙、速度原始数据及拟合窗口，不将空值替换为 0 来计算指标 |
| “打开记录目录”提示尚未生成目录 | 完成一次启用记录的回放，或确认填写的根目录已经存在 |
| 回放结束后开始按钮仍暂时变灰 | 等待短暂收尾采集和后台保存完成，观察“运行记录”状态 |
| 回放时视觉检测被拒绝 | 静止位姿采样需要在遥操作、示教记录和回放停止后执行；实验采集与保存也结束后再操作更便于管理 |

遇到 RTDE `End of file`、接收断开或时间戳停止更新时，保留本次输出和完整终端日志，核对控制器状态、网络、其他控制程序及实际连接配置。当前 100 Hz 是缓存轮询，不是把机器人反馈提高到了 100 Hz，也没有新建额外 RTDE 连接。

采集异常会结束记录并尝试保存已有数据；记录模块本身不发送运动或停止指令。应根据回放、遥操作和控制权状态确认机械臂当前行为，不能仅根据“数据已保存”判断机械臂已经停止。

## 12. 停止、退出与视觉功能

- “停止示教回放”用于主动结束回放。触发人工接管后，程序已切换为遥操作，普通“停止示教回放”不会中断新遥操作；应使用“停止遥操作”或“安全停止全部控制”。
- “安全停止全部控制”会停止控制并清理设备连接，下一次使用需要重新连接和初始化相应设备。
- 回放结束后，等待数据保存状态完成再退出。正常退出会执行设备清理并等待记录器结束；若提示保存或线程尚未结束，应等待处理完成，不把强制结束进程当作保存操作。
- “视觉示教 / 任务绑定”沿用原功能：连接相机、加载手眼矩阵、设置尺寸和 ROI，在机械臂静止时检测并记录视觉基准，再进行遥操作示教。保存示教时可将轨迹与参考 JSON 关联；未选参考时只保存示教轨迹。
- 若仅回放已有轨迹并记录六关节数据，无需启动相机或重新记录视觉参考。视觉任务绑定与回放运行日志是两个独立用途。

三个新代码文件此前已通过语法、GUI 构建及模拟场景检查；这些检查不等同于真实硬件上的连接稳定性和回放运动验证。本文的采样、字段及完整性规则以当前代码为准。
