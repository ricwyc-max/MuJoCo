"""全局实验配置。

改这里就能改整个作业的实验条件：目标构型、控制器增益、仿真时长、输出路径。
"""

from __future__ import annotations

import pathlib

import numpy as np

# ---------------------------------------------------------------- 路径
PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
VIDEOS_DIR = RESULTS_DIR / "videos"
TRIALS_DIR = RESULTS_DIR / "trials"
METRICS_PATH = RESULTS_DIR / "metrics.json"

# ---------------------------------------------------------------- 机器人与仿真
N_JOINTS = 7  # Panda 七轴
TIMESTEP = 0.002  # 2 ms
SIM_TIME = 2.0  # 单次实验仿真时长（秒）
RENDER_STRIDE = 2  # 每 N 个物理步渲染一帧
VIDEO_FPS = 60
RENDER_WIDTH, RENDER_HEIGHT = 640, 480

CAMERA_NAME = "cam_main"
EE_TOLERANCE = 5e-3  # 末端"到达目标"判据（米）
SETTLE_WINDOW = 0.5  # 稳态统计窗口（秒，取末尾这段时间）

# ---------------------------------------------------------------- 目标构型
# 非零的关节目标位置 qpos（rad）。取一个手臂向前伸展、离开竖直姿态的构型，
# 这样重力在关节 2/3 上产生明显力矩，"无控制下垂"和"稳态误差"都容易观测。
Q_TARGET = np.array([0.00, -0.30, 0.00, -1.20, 0.00, 1.60, 0.50])

# 初始构型 Q_START：与目标有明显差距，让控制器有事可做。
Q_START = np.array([0.00, 0.00, 0.00, -0.60, 0.00, 1.00, 0.00])

# ---------------------------------------------------------------- 控制器增益
# 依据 Franka 惯量分布选：前四个关节带动整臂（惯量大），后三个只带动腕部（惯量小）。
# Kd ≈ 0.8~0.9 * sqrt(Kp * 惯量)，保证阻尼比接近 1（临界阻尼附近，无超调）。
KP = np.array([600.0, 600.0, 600.0, 600.0, 300.0, 150.0, 50.0])
KD = np.array([50.0, 50.0, 50.0, 50.0, 25.0, 10.0, 5.0])

# ---------------------------------------------------------------- 派生模型里的执行器参数
# 力矩模型：Panda 官方关节力矩上限 (N·m)
TORQUE_CTRL_RANGE = [
    (-87.0, 87.0),
    (-87.0, 87.0),
    (-87.0, 87.0),
    (-87.0, 87.0),
    (-12.0, 12.0),
    (-12.0, 12.0),
    (-12.0, 12.0),
]

# 位置伺服模型：内部 PD 增益，刻意与上面 KP/KD 保持一致，
# 这样实验 B（MuJoCo 自带伺服）与实验 C（我们自己写的 PD）应当给出完全相同的轨迹。
SERVO_KP = KP.copy()
SERVO_KV = KD.copy()

# ---------------------------------------------------------------- 扰动实验
# t = 0.8 s 起给关节 4（重力力矩最大的关节之一，目标构型处约 14.9 N·m）
# 加一个持续的 -20 N·m 外部负载（等价于末端抓了个重物），一直加到仿真结束。
# 负号是为了让外负载与重力同向叠加，误差逐项单调变大，看得更清楚。
#
# 稳态平衡条件是 qfrc_bias = τ_actuator + τ_applied，于是：
#   纯 PD                     τ_actuator = Kp·e          -> e = (g - τ_ext) / Kp ≈ (14.9 + 20) / 600 ≈ 0.058 rad
#   PD + 重力补偿             τ_actuator = Kp·e + g      -> e = -τ_ext / Kp ≈ 20 / 600 ≈ 0.033 rad
#   PD + 重力补偿 + 负载前馈  再加上 τ_ff = -τ_ext        -> e = 0
# 结论：重力补偿只抵消"模型已知"的扰动；未知外负载还得靠前馈 / 积分 / 扰动观测器。
DISTURBANCE = dict(time=0.8, duration=None, dof=3, torque=-20.0)

# ---------------------------------------------------------------- 增益扫描
# Kp 扫描 4 档；Kd 用"阻尼缩放因子"给出：Kd = scale · Kd_ref · sqrt(Kp / Kp_ref)。
# 这样无论 Kp 取多少，阻尼比 ζ = Kd/(2√(Kp·I)) 都保持同一量级，
# 不会出现 Kd 随 Kp 线性放大、把系统阻尼到数值上"冻住"的情况。
GAIN_SWEEP_KP = [200.0, 400.0, 600.0, 900.0]
GAIN_SWEEP_KD_SCALE = [0.5, 1.0, 1.5]


def ensure_dirs() -> None:
    """确保输出目录存在。"""
    for d in (MODELS_DIR, RESULTS_DIR, FIGURES_DIR, VIDEOS_DIR, TRIALS_DIR):
        d.mkdir(parents=True, exist_ok=True)