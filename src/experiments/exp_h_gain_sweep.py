"""实验 H：PD 增益扫描（超调量 / 收敛时间 / 稳态精度）。

回答"增益怎么选"这个必然会被问到的问题：
- Kp 越大，纯 PD 的稳态误差越小（误差 ≈ g/Kp），但控制力矩峰值也越大，
  力矩一旦顶到 ctrlrange 就会饱和，反而收敛更慢；
- Kd 提供阻尼，接近临界阻尼时超调最小；
- 有了重力补偿，稳态误差与增益彻底解耦，恒为 0。

扫描 Kp ∈ {200,400,600,900} × 阻尼缩放因子 {0.5,1.0,1.5}，其中

    Kd = Kd_ref · scale · sqrt(Kp / Kp_ref)

保证各档的阻尼比 ζ = Kd / (2√(Kp·I)) 处于同一量级；否则 Kd 随 Kp 线性放大会把系统
阻尼到数值上"冻住"（dt=2 ms 根本跟不上）。分别用纯 PD 和 PD+重力补偿各跑一遍。
"""

from __future__ import annotations

import pathlib
import sys

# 允许以脚本方式直接运行（python src/experiments/xxx.py）
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import csv

import numpy as np

from src import config
from src.controllers import PDController, PDGravityController
from src.recorder import run_trial

KEY = "exp_H_gain_sweep"
SIM_TIME = 1.5
TRACK_JOINT = 1  # 跟踪关节 2：初始误差大、重力矩大，最能体现增益差别
KP_REF = float(config.KP[TRACK_JOINT])
KD_REF = float(config.KD[TRACK_JOINT])


def _overshoot(q, q0, q_star) -> float:
    """相对初始误差方向的超调量（正数表示冲过了目标）。"""
    err = q - q_star
    err0 = q0 - q_star
    if abs(err0) < 1e-9:
        return 0.0
    return float(max(0.0, -np.sign(err0) * np.min(err) / abs(err0)))


def _settle(t, err, tol=5e-3) -> float:
    """误差稳定到"自身终值"±tol 所需的时间。

    不能拿"到目标的误差 < tol"当判据：纯 PD 的稳态误差本来就有 ~0.02 rad，
    那样永远判不出收敛。这里把两件事分开：
      settle_time_s —— 动态收敛速度（本函数）
      ss_joint_err  —— 稳态精度（另行给出）
    """
    deviation = np.abs(err - err[-1])
    outside = np.flatnonzero(deviation > tol)
    if len(outside) == 0:
        return 0.0
    return float(t[outside[-1]])


def saturated(trial) -> np.ndarray:
    """布尔矩阵：实际施加的关节力矩是否已经顶到力矩上限。"""
    lo, hi = trial.torque_limit[0], trial.torque_limit[1]
    return (trial.tau >= hi - 1e-6) | (trial.tau <= lo + 1e-6)


def sweep(kp_values=None, kd_scales=None, use_gravity=(False, True)):
    kp_values = kp_values or config.GAIN_SWEEP_KP
    kd_scales = kd_scales or config.GAIN_SWEEP_KD_SCALE

    rows = []
    for use_gc in use_gravity:
        cls = PDGravityController if use_gc else PDController
        label = "PD+重力补偿" if use_gc else "纯PD"
        for kp in kp_values:
            for scale in kd_scales:
                kd = KD_REF * scale * float(np.sqrt(kp / KP_REF))
                ctrl = cls(kp=np.full(config.N_JOINTS, kp), kd=np.full(config.N_JOINTS, kd))
                trial = run_trial(f"sweep_{'gc' if use_gc else 'pd'}_kp{kp:g}_s{scale:g}",
                                  ctrl, model_kind="torque", render=False, sim_time=SIM_TIME)
                err = trial.qpos[:, TRACK_JOINT] - trial.q_target[TRACK_JOINT]
                tail = trial.t >= trial.t[-1] - 0.3
                rows.append({
                    "controller": label,
                    "kp": kp,
                    "kd": round(kd, 2),
                    "kd_scale": scale,
                    "overshoot": round(_overshoot(trial.qpos[:, TRACK_JOINT],
                                                  trial.q_start[TRACK_JOINT],
                                                  trial.q_target[TRACK_JOINT]), 4),
                    "settle_time_s": round(_settle(trial.t, err), 4),
                    "ss_joint_err_rad": round(float(np.abs(err[tail]).mean()), 6),
                    "tau_peak_j2": round(float(np.abs(trial.tau[:, TRACK_JOINT]).max()), 2),
                    "cmd_peak_j2": round(float(np.abs(trial.ctrl[:, TRACK_JOINT]).max()), 2),
                    "sat_ratio": round(float(saturated(trial).mean()), 4),
                })
    return rows


def _heatmap(rows, field, controller, kp_values, kd_scales, path):
    import matplotlib.pyplot as plt

    grid = np.full((len(kp_values), len(kd_scales)), np.nan)
    for r in rows:
        if r["controller"] != controller:
            continue
        grid[kp_values.index(r["kp"]), kd_scales.index(r["kd_scale"])] = r[field]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    im = ax.imshow(grid, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(kd_scales)), [f"{s:g}" for s in kd_scales])
    ax.set_yticks(range(len(kp_values)), [f"{v:g}" for v in kp_values])
    for i in range(len(kp_values)):
        for j in range(len(kd_scales)):
            value = grid[i, j]
            ax.text(j, i, "n/a" if np.isnan(value) else f"{value:.3g}",
                    ha="center", va="center", color="w", fontsize=9)
    ax.set_xlabel("Kd 阻尼缩放因子")
    ax.set_ylabel("Kp")
    ax.set_title(f"{controller}: {field}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def main() -> None:
    config.ensure_dirs()
    from src.recorder import dump_metrics

    rows = sweep()
    kp_values = config.GAIN_SWEEP_KP
    kd_scales = config.GAIN_SWEEP_KD_SCALE

    csv_path = config.RESULTS_DIR / f"{KEY}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    figs = [
        _heatmap(rows, "overshoot", "纯PD", kp_values, kd_scales,
                 config.FIGURES_DIR / f"{KEY}_overshoot_pd.png"),
        _heatmap(rows, "settle_time_s", "纯PD", kp_values, kd_scales,
                 config.FIGURES_DIR / f"{KEY}_settle_pd.png"),
        _heatmap(rows, "ss_joint_err_rad", "纯PD", kp_values, kd_scales,
                 config.FIGURES_DIR / f"{KEY}_sserr_pd.png"),
        _heatmap(rows, "ss_joint_err_rad", "PD+重力补偿", kp_values, kd_scales,
                 config.FIGURES_DIR / f"{KEY}_sserr_gc.png"),
    ]
    dump_metrics(KEY, {"name": KEY, "kp_ref": KP_REF, "kd_ref": KD_REF, "rows": rows})

    print(f"\n===== {KEY}（跟踪关节 {TRACK_JOINT + 1}，Kp_ref={KP_REF:g} Kd_ref={KD_REF:g}）=====")
    print(f"{'控制器':<14}{'Kp':>6}{'Kd':>8}{'超调':>8}{'收敛(s)':>10}"
          f"{'稳态误差(rad)':>16}{'指令峰值':>10}{'力矩峰值':>10}{'饱和占比':>10}")
    for r in rows:
        print(f"{r['controller']:<14}{r['kp']:>6g}{r['kd']:>8.1f}{r['overshoot']:>8.3f}"
              f"{r['settle_time_s']:>10.4f}{r['ss_joint_err_rad']:>16.3e}"
              f"{r['cmd_peak_j2']:>10.1f}{r['tau_peak_j2']:>10.1f}{r['sat_ratio']:>10.3f}")
    print("\n观察：纯 PD 的稳态误差随 Kp 增大而下降（约 ∝ 1/Kp），但力矩峰值上升、"
          "饱和占比变高；\n      PD+重力补偿的稳态误差恒为 0，与增益无关。")
    print(f"CSV  : {csv_path}")
    for p in figs:
        print(f"热力图: {p}")


if __name__ == "__main__":
    main()