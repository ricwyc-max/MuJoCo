"""实验 A：不做任何控制，观察机械臂在重力下下垂。

这是全部实验的基线 —— 用来回答作业要求第 2 条：
"如果不做任何控制，运行仿真并观察机械臂是否会因重力下垂"。

关键点：力矩型执行器下 ``data.ctrl = 0`` 就是"电机不通电"，
机械臂完全靠重力塌下去。
"""

from __future__ import annotations

import pathlib
import sys

# 允许以脚本方式直接运行（python src/experiments/xxx.py）
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.experiments._common import finish
from src.controllers import ZeroController
from src.recorder import run_trial

KEY = "exp_A_no_control"


def run(render: bool = True):
    return run_trial(KEY, ZeroController(), model_kind="torque", render=render)


def main() -> None:
    _, metrics = finish(run(), KEY)
    if metrics["sag_max_m"] <= 0.05:
        print(f"[验收] 失败：最大下垂量仅 {metrics['sag_max_m']} m，未观察到明显下垂")
    else:
        print(f"[验收] 通过：无控制下末端最大下垂 {metrics['sag_max_m']} m，重力下垂现象清晰")


if __name__ == "__main__":
    main()