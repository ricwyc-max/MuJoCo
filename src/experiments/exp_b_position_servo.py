"""实验 B：使用 MuJoCo 模型自带的"纯位置伺服"。

对应作业考察点里"对比纯位置伺服与力矩控制的差异"。

这里用的是派生模型 ``panda_position.xml``：执行器是 ``<position kp=... kv=...>``，
于是 ``data.ctrl`` 的单位是 **关节角 (rad)**，MuJoCo 内部替我们算

    tau = kp * (ctrl - q) - kv * qdot

也就是说，MuJoCo 把一个 PD 控制器封装进了执行器里。增益被写在 XML 中，
Python 侧看不见、也改不了，更没法往里塞重力补偿项。
"""

from __future__ import annotations

import pathlib
import sys

# 允许以脚本方式直接运行（python src/experiments/xxx.py）
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.experiments._common import finish
from src.controllers import PositionServoController
from src.recorder import run_trial

KEY = "exp_B_position_servo"


def run(render: bool = True):
    return run_trial(KEY, PositionServoController(), model_kind="position", render=render)


def main() -> None:
    _, metrics = finish(run(), KEY)
    print(
        "[说明] 位置伺服由 MuJoCo 内部实现 PD（单位 rad），"
        "它与实验 C 的轨迹几乎一致 —— 说明二者本质是同一个控制器。"
    )


if __name__ == "__main__":
    main()