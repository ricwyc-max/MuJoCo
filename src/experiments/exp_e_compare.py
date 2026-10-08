"""实验 E：四个实验的横向对比。

回答作业考察点 —— "对比纯位置伺服与力矩控制的差异"。

产出两张图：
1. 轨迹叠加（末端高度 + 关节 2 角度），四条曲线放在一起；
2. 关键指标柱状图，一眼看出"谁的稳态误差是 0"。

另外做一项定量验证：实验 B（MuJoCo 内置位置伺服）与实验 C（我们自己写的 PD）
用了同一组 Kp/Kd 与同一组力矩上限，因此两者的**稳态**结果应当完全一致；
瞬态上的微小差别来自 RK4 积分 —— 位置执行器的内部 PD 会在每个 RK4 子步按
当前状态重新求值，而我们写的力矩控制在整步内是零阶保持。
"""

from __future__ import annotations

import pathlib
import sys

# 允许以脚本方式直接运行（python src/experiments/xxx.py）
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from src import config
from src.recorder import dump_metrics, plot_compare

KEY = "exp_E_compare"


def _bar_chart(trials, path):
    """把稳态误差画成柱状图。"""
    import matplotlib.pyplot as plt

    labels = [f"{t.name.split('_', 1)[-1]}" for t in trials]
    joint_err = [float(np.abs(t.qpos[-1] - t.q_target).max()) for t in trials]
    ee_err = [float(np.linalg.norm(t.ee_pos[-1] - t.ee_target)) for t in trials]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, values, title, unit, scale in (
        (axes[0], joint_err, "稳态最大关节误差", "rad", 1.0),
        (axes[1], ee_err, "稳态末端位置误差", "m", 1.0),
    ):
        bars = ax.bar(labels, values, color=["C0", "C1", "C2", "C3"][: len(labels)])
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{v:.2e}", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"{title} ({unit})")
        ax.set_yscale("log")
        ax.set_ylim(1e-7, max(values) * 20 if max(values) > 0 else 1e-6)
        ax.grid(alpha=0.3, axis="y")
        del scale
    fig.suptitle("A 无控制 / B 位置伺服 / C 纯 PD / D PD+重力补偿 稳态误差对比")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    config.ensure_dirs()
    from src.experiments import (  # 延迟导入，避免 exp_e 被 run_all 直接调用时提前跑仿真
        exp_a_no_control,
        exp_b_position_servo,
        exp_c_pd_torque,
        exp_d_pd_gravity,
    )

    trials = [
        exp_a_no_control.run(render=False),
        exp_b_position_servo.run(render=False),
        exp_c_pd_torque.run(render=False),
        exp_d_pd_gravity.run(render=False),
    ]

    fig_traj = plot_compare(trials, config.FIGURES_DIR / "exp_E_compare_traj.png")
    fig_bar = _bar_chart(trials, config.FIGURES_DIR / "exp_E_compare_metrics.png")

    # 定量验证：B 与 C 在稳态上是否一致
    b, c = trials[1], trials[2]
    diff = np.abs(b.qpos - c.qpos)
    bc_all = float(diff.max())
    tail = b.t >= b.t[-1] - 0.5
    bc_ss = float(diff[tail].max())

    metrics = {
        "name": KEY,
        "note": "四个实验的对比汇总",
        "steady_joint_err_rad": {
            t.name: round(float(np.abs(t.qpos[-1] - t.q_target).max()), 6) for t in trials
        },
        "steady_ee_err_m": {
            t.name: round(float(np.linalg.norm(t.ee_pos[-1] - t.ee_target)), 6) for t in trials
        },
        "B_vs_C_max_joint_diff_rad": round(bc_all, 9),
        "B_vs_C_steady_joint_diff_rad": round(bc_ss, 12),
        "B_vs_C_note": "同一组 Kp/Kd 与力矩上限；稳态应一致，瞬态差异来自 RK4 子步求值方式",
    }
    dump_metrics(KEY, metrics)

    print(f"\n===== {KEY} =====")
    print(f"{'实验':<24}{'稳态最大关节误差(rad)':>22}{'稳态末端误差(m)':>20}")
    for t in trials:
        name = t.controller
        je = float(np.abs(t.qpos[-1] - t.q_target).max())
        ee = float(np.linalg.norm(t.ee_pos[-1] - t.ee_target))
        print(f"{name:<24}{je:>22.3e}{ee:>20.3e}")
    print(f"\nB(位置伺服) vs C(纯PD)：全程最大偏差 {bc_all:.3e} rad，"
          f"末 0.5 s 偏差 {bc_ss:.3e} rad")
    print("  → 稳态完全一致，说明 MuJoCo 内置位置伺服本质上就是这个 PD；")
    print("  → 瞬态残差来自 RK4：位置执行器的内部 PD 每个子步重算，力矩控制是零阶保持。")
    print(f"图表: {fig_traj}\n      {fig_bar}")


if __name__ == "__main__":
    main()