"""实验 G：外部负载下的保持能力 —— 重力补偿能做什么、不能做什么。

从 t = 0.8 s 起，给关节 4 加一个持续的 +20 N·m 外部负载（通过 ``data.qfrc_applied``），
相当于末端抓了个重物，一直加到仿真结束。

稳态时三种控制器的误差可以逐项拆开核对：

    纯 PD                    误差 = (g(q*) + τ_ext) / Kp ≈ (14.9 + 20) / 600 ≈ 0.058 rad
    PD + 重力补偿            误差 = τ_ext / Kp = 20 / 600 ≈ 0.033 rad
    PD + 重力补偿 + 负载前馈  误差 = 0

结论（也是本实验真正想说的事）：**重力补偿只能抵消"模型已经知道"的扰动**。
外部负载是 ``qfrc_bias`` 之外的一条独立通路（``data.qfrc_applied``），
前馈项里没有它，误差依然存在；只有把负载也前馈进去（理想扰动观测器）
才能真正做到零误差。真机上的积分项 / 扰动观测器解决的就是这个问题。
"""

from __future__ import annotations

import pathlib
import sys

# 允许以脚本方式直接运行（python src/experiments/xxx.py）
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import mujoco
import numpy as np

from src import config
from src.analytic_gravity import gravity_torque
from src.controllers import PDController, PDGravityController, PDGravityLoadController
from src.experiments._common import finish
from src.model_utils import get_refs, load_model, make_data
from src.recorder import run_trial

KEY_PD = "exp_G_disturbance_pd"
KEY_GC = "exp_G_disturbance_gravity"
KEY_FF = "exp_G_disturbance_feedforward"

DOF = config.DISTURBANCE["dof"]
LOAD_TIME = config.DISTURBANCE["time"]
LOAD = config.DISTURBANCE["torque"]


def _gravity_at_target() -> np.ndarray:
    """目标构型下的重力力矩（复用实验 F 的解析实现）。"""
    model = load_model("torque")
    refs = get_refs("torque")
    data = make_data(model)
    data.qpos[refs.qpos_adr] = config.Q_TARGET
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)
    return gravity_torque(model, data, refs)


def _disturbances(sim_time: float):
    end = sim_time if config.DISTURBANCE["duration"] is None else (
        LOAD_TIME + config.DISTURBANCE["duration"]
    )
    return [(LOAD_TIME, end, DOF, LOAD)]


def run(render: bool = True, sim_time: float | None = None):
    """跑三种控制器：纯 PD、PD+重力补偿、PD+重力补偿+负载前馈。"""
    sim_time = config.SIM_TIME if sim_time is None else sim_time
    disturbances = _disturbances(sim_time)

    load_ff = np.zeros(config.N_JOINTS)
    load_ff[DOF] = -LOAD  # 稳态 qfrc_bias = τ_actuator + τ_applied，要顶住负载得出反向力矩

    trials = [
        run_trial(KEY_PD, PDController(), model_kind="torque", render=render,
                  sim_time=sim_time, disturbances=disturbances),
        run_trial(KEY_GC, PDGravityController(), model_kind="torque", render=render,
                  sim_time=sim_time, disturbances=disturbances),
        run_trial(KEY_FF, PDGravityLoadController(extra_tau=load_ff), model_kind="torque",
                  render=render, sim_time=sim_time, disturbances=disturbances),
    ]
    return trials


def _load_metrics(trial, gravity_tau: float) -> dict:
    """加负载后的表现：峰值偏差 + 加载后的稳态误差，并给出理论预期。

    稳态平衡条件：qfrc_bias = τ_actuator + τ_applied
      纯 PD        τ_actuator = Kp·e         ->  e = (g - τ_ext) / Kp
      PD+重力补偿  τ_actuator = Kp·e + g     ->  e = -τ_ext / Kp
      再加负载前馈  τ_actuator = Kp·e + g - τ_ext -> e = 0
    """
    err = np.abs(trial.qpos[:, DOF] - trial.q_target[DOF])
    tail = trial.t >= trial.t[-1] - 0.3
    kp = config.KP[DOF]
    if trial.controller.startswith("C"):  # 纯 PD：重力和外负载都得由比例项扛
        theory = abs(gravity_tau - LOAD) / kp
    else:  # 重力已被补偿，只剩外负载（再加前馈则为 0）
        theory = abs(LOAD) / kp if trial.controller.startswith("D PD") else 0.0
    return {
        "peak_joint_err_after_load_rad": round(float(err[trial.t >= LOAD_TIME].max()), 6),
        "ss_joint_err_after_load_rad": round(float(err[tail].mean()), 6),
        "ss_joint_err_max_all_rad": round(float(err[tail].max()), 6),
        "theory_ss_err_rad": round(theory, 6),
        "tau_ss_joint4": round(float(trial.tau[tail][:, DOF].mean()), 3),
    }


def main() -> None:
    config.ensure_dirs()
    from src.recorder import L, dump_metrics

    gravity_tau = _gravity_at_target()
    trials = run()
    for t, key in zip(trials, (KEY_PD, KEY_GC, KEY_FF)):
        finish(t, key)

    per_trial = [_load_metrics(t, gravity_tau[DOF]) for t in trials]
    dump_metrics("exp_G_disturbance", {
        "load": config.DISTURBANCE,
        "gravity_tau_joint4_at_target": round(float(gravity_tau[DOF]), 3),
        KEY_PD: per_trial[0],
        KEY_GC: per_trial[1],
        KEY_FF: per_trial[2],
        "conclusion": "重力补偿只抵消模型已知的扰动；未知外负载需要前馈/积分/扰动观测器",
    })

    import matplotlib.pyplot as plt

    colors = ["C1", "C3", "C2"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    for t, m, color in zip(trials, per_trial, colors):
        axes[0].plot(t.t, t.qpos[:, DOF] - t.q_target[DOF], lw=1.6, color=color,
                     label=f"{t.controller}（实测 {m['ss_joint_err_after_load_rad']:.4f}）")
    axes[0].axvline(LOAD_TIME, color="r", ls="--", lw=1.2,
                    label=L(f"t={LOAD_TIME}s 起加 {LOAD:g} N·m 持续负载",
                            f"sustained {LOAD:g} N·m load from t={LOAD_TIME}s"))
    axes[0].axhline(0.0, color="k", lw=0.8)
    axes[0].set_xlabel(L("时间 (s)", "time (s)"))
    axes[0].set_ylabel(L(f"关节 {DOF + 1} 误差 (rad)", f"joint {DOF + 1} error (rad)"))
    axes[0].set_title(L("外部负载下的关节误差", "Joint error under external load"))
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    names = [t.controller for t in trials]
    measured = [m["ss_joint_err_after_load_rad"] for m in per_trial]
    theory = [m["theory_ss_err_rad"] for m in per_trial]
    x = np.arange(len(names))
    axes[1].bar(x - 0.2, measured, 0.4, color=colors, label=L("实测", "measured"))
    axes[1].bar(x + 0.2, theory, 0.4, color="gray", alpha=0.7,
                label=L("理论 (扰动/Kp)", "theory (disturbance/Kp)"))
    axes[1].set_xticks(x, [n.replace("PD + ", "PD+\n+ ") if len(n) > 14 else n for n in names],
                       fontsize=8)
    axes[1].set_ylabel(L("加载后稳态误差 (rad)", "steady error after load (rad)"))
    axes[1].set_title(L(f"关节 {DOF + 1} 加载后稳态误差：实测 vs 理论",
                        f"joint {DOF + 1} steady error: measured vs theory"))
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path = config.FIGURES_DIR / "exp_G_disturbance.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)

    print(f"\n===== 外部负载下的保持能力（关节 {DOF + 1}）=====")
    print(f"目标构型处该关节重力力矩 = {gravity_tau[DOF]:.3f} N·m，外部负载 = {LOAD:g} N·m，"
          f"Kp = {config.KP[DOF]:g}")
    print(f"{'控制器':<28}{'峰值误差(rad)':>15}{'加载后稳态误差':>17}{'理论值':>10}")
    for t, m in zip(trials, per_trial):
        print(f"{t.controller:<28}{m['peak_joint_err_after_load_rad']:>15.4f}"
              f"{m['ss_joint_err_after_load_rad']:>17.4f}{m['theory_ss_err_rad']:>10.4f}")
    print("\n结论：重力补偿把'重力'这一项扰动消掉了，但外部负载不在 qfrc_bias 里，"
          "前馈项看不到它；\n      把负载也前馈进去（第三行）误差才归零 —— "
          "这正是真机上积分项 / 扰动观测器的职责。")
    print(f"图表: {path}")

    if per_trial[2]["ss_joint_err_max_all_rad"] < 1e-3:
        print("[验收] 通过：加入负载前馈后，加载期间稳态误差 < 1e-3 rad")
    else:
        print(f"[验收] 失败：加载后稳态误差 {per_trial[2]['ss_joint_err_max_all_rad']} rad 过大")


if __name__ == "__main__":
    main()