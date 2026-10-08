"""实验 C：用 ``data.ctrl`` 直接施加关节力矩，实现纯 PD 控制。

    tau = Kp * (q* - q) - Kd * qdot

对应作业要求第 3 条的前半段。能把手臂拉回目标构型，
但因为重力是恒定扰动、而比例增益有限，稳态时必然残留误差：

    稳态：tau = g(q)  ->  Kp*(q* - q) = g(q)  ->  误差 = g(q)/Kp

这个残留误差正是引出实验 D（重力补偿）的动机。
"""

from __future__ import annotations

import pathlib
import sys

# 允许以脚本方式直接运行（python src/experiments/xxx.py）
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.experiments._common import finish
from src.controllers import PDController
from src.recorder import run_trial

KEY = "exp_C_pd_torque"


def run(render: bool = True):
    return run_trial(KEY, PDController(), model_kind="torque", render=render)


def main() -> None:
    _, metrics = finish(run(), KEY)
    if metrics["ss_joint_err_max_rad"] <= 1e-3:
        print(f"[验收] 失败：纯 PD 的稳态误差 {metrics['ss_joint_err_max_rad']} rad 太小，"
              "没有体现出'PD 无法消除恒定扰动'这一现象")
    else:
        print(f"[验收] 通过：纯 PD 稳态最大关节误差 {metrics['ss_joint_err_max_rad']} rad，"
              "即重力造成的稳态偏差")


if __name__ == "__main__":
    main()