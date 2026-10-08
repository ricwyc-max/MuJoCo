"""实验 F：解析重力补偿 vs MuJoCo 的 qfrc_bias（理解执行器 / 广义力模型的验证）。

作业考察点要求理解 MuJoCo 的执行器模型。本实验把"重力力矩到底是什么"
从定义层面推一遍：

    每个刚体的重力 F_b = m_b · g（g 取 model.opt.gravity）
    映射到关节空间：tau_b = J_b(q)^T · F_b，J_b 由 mj_jacBody 给出
    总重力力矩：tau_g = Σ_b tau_b

在 q̇ = 0 时，MuJoCo 的 data.qfrc_bias = C(q,q̇) + g(q) 退化为纯重力项，
于是两者应当在数值误差量级上完全一致。

如果换用模型自带的 ``<general>`` 执行器做闭环（而不是自己写 data.ctrl），
就无法把这个量单独取出来做前馈——这正是"力矩控制"相对"位置伺服"的价值。
"""

from __future__ import annotations

import pathlib
import sys

# 允许以脚本方式直接运行（python src/experiments/xxx.py）
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from src import config
from src.analytic_gravity import compare_with_qfrc_bias, gravity_torque
from src.model_utils import get_refs, load_model, make_data

KEY = "exp_F_analytic_gravity"
N_SAMPLES = 200


def _sample_errors(n_samples: int = N_SAMPLES, seed: int = 0):
    import mujoco

    model = load_model("torque")
    refs = get_refs("torque")
    data = make_data(model)

    rng = np.random.default_rng(seed)
    lo, hi = model.jnt_range[refs.qpos_adr, 0], model.jnt_range[refs.qpos_adr, 1]

    errors, taus, poses = [], [], []
    for _ in range(n_samples):
        q = rng.uniform(lo, hi)
        data.qpos[refs.qpos_adr] = q
        data.qvel[:] = 0.0  # q̇=0 时 qfrc_bias 只剩重力项
        mujoco.mj_forward(model, data)
        errors.append(compare_with_qfrc_bias(model, data, refs))
        taus.append(gravity_torque(model, data, refs))
        poses.append(q)

    return np.array(errors), np.array(taus), np.array(poses)


def main() -> None:
    config.ensure_dirs()
    import mujoco

    from src.recorder import L, dump_metrics

    errors, taus, poses = _sample_errors()

    # 目标构型处的重力力矩，用于和实验 D 的稳态控制力矩对照
    model = load_model("torque")
    refs = get_refs("torque")
    data = make_data(model)
    data.qpos[refs.qpos_adr] = config.Q_TARGET
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    tau_target = gravity_torque(model, data, refs)

    metrics = {
        "name": KEY,
        "n_samples": int(len(errors)),
        "max_abs_err": float(errors.max()),
        "mean_abs_err": float(errors.mean()),
        "gravity_tau_at_target": [round(float(x), 4) for x in tau_target],
        "gravity_tau_max_over_samples": [round(float(x), 3) for x in np.abs(taus).max(axis=0)],
    }
    dump_metrics(KEY, metrics)

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(errors, lw=0.8)
    axes[0].set_yscale("log")
    axes[0].axhline(1e-9, color="r", ls="--", lw=1,
                    label=L("1e-9 阈值", "1e-9 threshold"))
    axes[0].set_title(L(f"{len(errors)} 个随机构型上 ‖Σ J^T m g − qfrc_bias‖∞",
                        f"‖Σ J^T m g − qfrc_bias‖∞ over {len(errors)} random poses"))
    axes[0].set_xlabel(L("采样序号", "sample index"))
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(poses[:, 1], taus[:, 1], ".", ms=2, label=L("关节 2", "joint 2"))
    axes[1].plot(poses[:, 2], taus[:, 2], ".", ms=2, label=L("关节 3", "joint 3"))
    axes[1].plot(poses[:, 0], taus[:, 0], ".", ms=2, label=L("关节 1", "joint 1"))
    axes[1].axvline(config.Q_TARGET[1], color="k", ls="--", lw=1)
    axes[1].set_title(L("重力力矩随构型变化（竖线为目标构型）",
                        "gravity torque vs configuration (dashed = target)"))
    axes[1].set_xlabel(L("关节角 (rad)", "joint angle (rad)"))
    axes[1].set_ylabel(L("重力力矩 (N·m)", "gravity torque (N·m)"))
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    path = config.FIGURES_DIR / f"{KEY}.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)

    print(f"\n===== {KEY} =====")
    print(f"随机采样构型数        : {len(errors)}")
    print(f"最大绝对偏差          : {errors.max():.3e}")
    print(f"平均绝对偏差          : {errors.mean():.3e}")
    print("目标构型处重力力矩(N·m):", np.array2string(tau_target, precision=3))
    print(f"图表: {path}")

    if errors.max() < 1e-9:
        print("[验收] 通过：解析重力补偿与 data.qfrc_bias 完全一致（< 1e-9）")
    else:
        print(f"[验收] 失败：最大偏差 {errors.max():.3e} 超过 1e-9")


if __name__ == "__main__":
    main()