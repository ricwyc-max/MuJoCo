"""实验 D（作业核心答案）：PD + 重力补偿。

    tau = Kp * (q* - q) - Kd * qdot + g(q)

其中重力项直接取 MuJoCo 算好的偏置广义力 ``data.qfrc_bias``。
在固定基座下 ``qfrc_bias = C(q,qdot) + g(q)``，机械臂静止时它就等于重力力矩。

为什么稳态误差变成 0：
加入重力补偿后，重力被前馈完全抵消，控制律只剩下"把当前位形拉向目标"的
PD 作用，稳态时 q̇ = 0、q̈ = 0，于是 Kp*(q* - q) = 0，即 q = q*，误差为 0。

另一条等价路线是实验 F：用质心雅可比自己算 Σ Jᵀ(mg) 代替 qfrc_bias。
"""

from __future__ import annotations

import pathlib
import sys

# 允许以脚本方式直接运行（python src/experiments/xxx.py）
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from src.experiments._common import finish
from src.controllers import PDGravityController
from src.recorder import run_trial

KEY = "exp_D_pd_gravity"


def run(render: bool = True):
    return run_trial(KEY, PDGravityController(), model_kind="torque", render=render)


def main() -> None:
    trial, metrics = finish(run(), KEY)

    # 交叉验证：稳态时 PD 项趋于 0，施加力矩应当≈重力力矩
    tail = trial.t >= trial.t[-1] - 0.5
    tau_ss = trial.tau[tail][-1]
    g_ss = trial.qfrc_bias[tail][-1]
    print("\n稳态施加力矩 tau_ss (N·m) :", np.array2string(tau_ss, precision=3))
    print("稳态重力力矩 g(q) (N·m)   :", np.array2string(g_ss, precision=3))
    print("两者最大偏差 (N·m)        :", f"{np.max(np.abs(tau_ss - g_ss)):.3e}")

    ok = metrics["ss_joint_err_max_rad"] < 1e-3 and metrics["ss_ee_err_m"] < 5e-3
    if ok:
        print(f"[验收] 通过：稳态最大关节误差 {metrics['ss_joint_err_max_rad']} rad，"
              f"末端误差 {metrics['ss_ee_err_m']} m —— 构型被稳定保持")
    else:
        print(f"[验收] 失败：稳态关节误差 {metrics['ss_joint_err_max_rad']} rad，"
              f"末端误差 {metrics['ss_ee_err_m']} m，未达到保持要求")


if __name__ == "__main__":
    main()