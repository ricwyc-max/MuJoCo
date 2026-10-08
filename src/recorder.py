"""仿真试运行、指标计算、绘图与视频输出。

``run_trial`` 是所有实验共享的主循环骨架，也是作业的核心代码：

    data.ctrl[refs.act_ids] = controller.compute(model, data, refs)   # 写控制量
    data.qfrc_applied[dof] += dist                                     # （可选）加扰动
    mujoco.mj_step(model, data)                                        # 推进一个物理步
"""

from __future__ import annotations

import dataclasses
import json
import pathlib

import matplotlib

matplotlib.use("Agg")  # 无显示环境，必须在 pyplot 之前指定

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from src import config  # noqa: E402
from src import model_utils  # noqa: E402

# ---------------------------------------------------------------- 中文字体


def _setup_cjk_font() -> bool:
    """挑一个系统里有的中文字体；都没有就返回 False，图表降级为英文标签。"""
    available = {f.name for f in font_manager.fontManager.ttflist}
    for candidate in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"):
        if candidate in available:
            matplotlib.rcParams["font.sans-serif"] = [candidate]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return True
    matplotlib.rcParams["axes.unicode_minus"] = False
    return False


HAS_CJK_FONT = _setup_cjk_font()


def L(zh: str, en: str) -> str:
    """按字体可用性选择中文或英文标签。"""
    return zh if HAS_CJK_FONT else en


# ---------------------------------------------------------------- 试运行


@dataclasses.dataclass
class Trial:
    """一次仿真的完整记录。"""

    name: str
    controller: str
    model_kind: str
    ctrl_unit: str
    q_start: np.ndarray  # (7,)
    q_target: np.ndarray  # (7,)
    ee_target: np.ndarray  # (3,) 目标构型下的末端位置
    ctrlrange: np.ndarray  # (2, 7) 执行器 ctrlrange（力矩模型是 N·m，位置模型是 rad）
    torque_limit: np.ndarray  # (2, 7) 实际关节力矩的上限，N·m
    t: np.ndarray  # (N,)
    qpos: np.ndarray  # (N, 7)
    qvel: np.ndarray  # (N, 7)
    ctrl: np.ndarray  # (N, 7) 实际写入 data.ctrl 的值（原始指令，可能超出 ctrlrange）
    tau: np.ndarray  # (N, 7) MuJoCo 真正施加到关节上的力矩 data.qfrc_actuator（已被 ctrlrange 截断）
    qfrc_bias: np.ndarray  # (N, 7) 当步的偏置广义力
    ee_pos: np.ndarray  # (N, 3)

    @property
    def n_steps(self) -> int:
        return len(self.t)

    def save(self, stem: str) -> None:
        config.TRIALS_DIR.mkdir(parents=True, exist_ok=True)
        path = config.TRIALS_DIR / f"{stem}.npz"
        np.savez_compressed(path, **{f.name: getattr(self, f.name) for f in dataclasses.fields(self)})

    @staticmethod
    def load(path) -> "Trial":
        with np.load(path, allow_pickle=False) as data:
            fields = {f.name: data[f.name] for f in dataclasses.fields(Trial)}
        return Trial(**fields)


def run_trial(
    trial_name: str,
    controller,
    *,
    model_kind: str = "torque",
    q_start: np.ndarray | None = None,
    q_target: np.ndarray | None = None,
    sim_time: float | None = None,
    render: bool = True,
    disturbances=(),
) -> Trial:
    """跑一次仿真并记录全过程。

    disturbances: 可选的扰动列表，元素为 ``(t_start, t_end, dof_index, torque)``。
    """
    import mujoco

    model = model_utils.load_model(model_kind)
    refs = model_utils.get_refs(model_kind)
    q_start = config.Q_START.copy() if q_start is None else np.asarray(q_start, float)
    q_target = config.Q_TARGET.copy() if q_target is None else np.asarray(q_target, float)
    sim_time = config.SIM_TIME if sim_time is None else sim_time

    n_steps = max(1, int(round(sim_time / config.TIMESTEP)))

    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    data.qpos[refs.qpos_adr] = q_start
    mujoco.mj_forward(model, data)

    ee_target = model_utils.ee_position_at(model, refs, q_target)
    ctrlrange = np.array([model.actuator_ctrlrange[a] for a in refs.act_ids], dtype=float).T
    # 力矩真正被截断在哪个值上：forcerange 优先，没有才用 ctrllimited 的 ctrlrange。
    # 位置伺服模型的 ctrlrange 是关节角（rad），拿它判力矩饱和会得到毫无意义的数字。
    forcelimited = np.array([model.actuator_forcelimited[a] for a in refs.act_ids], dtype=bool)
    forcerange = np.array([model.actuator_forcerange[a] for a in refs.act_ids], dtype=float)
    torque_limit = np.where(forcelimited.T, forcerange.T, ctrlrange)

    buf = {
        key: np.empty((n_steps, dim))
        for key, dim in (
            ("qpos", refs.n),
            ("qvel", refs.n),
            ("ctrl", refs.n),
            ("tau", refs.n),
            ("qfrc_bias", refs.n),
            ("ee_pos", 3),
        )
    }
    times = np.arange(n_steps + 1) * config.TIMESTEP

    renderer = model_utils.make_renderer(model_kind) if render else None
    frames = []

    for step in range(n_steps):
        # ---- 1. 读状态、算控制量、写 data.ctrl（作业考察点）----
        data.ctrl[refs.act_ids] = controller.compute(model, data, refs)

        # ---- 2.（可选）施加外部扰动力矩 data.qfrc_applied ----
        # data.qfrc_applied 是用户字段，MuJoCo 不会自动清零，必须每步先归零，
        # 否则扰动会一直累加，把机械臂直接甩飞。
        data.qfrc_applied[:] = 0.0
        t_now = times[step]
        for t0, t1, dof, torque in disturbances:
            if t0 <= t_now < t1:
                data.qfrc_applied[refs.dof_adr[dof]] += torque

        # ---- 3. 推进仿真 ----
        mujoco.mj_step(model, data)

        # ---- 4. 记录 ----
        buf["qpos"][step] = data.qpos[refs.qpos_adr]
        buf["qvel"][step] = data.qvel[refs.dof_adr]
        buf["ctrl"][step] = data.ctrl[refs.act_ids]
        buf["tau"][step] = data.qfrc_actuator[refs.dof_adr]
        buf["qfrc_bias"][step] = data.qfrc_bias[refs.dof_adr]
        buf["ee_pos"][step] = model_utils.ee_position(data, refs)

        if renderer is not None and step % config.RENDER_STRIDE == 0:
            frames.append(model_utils.render(renderer, model, data))

    trial = Trial(
        name=trial_name,
        controller=controller.name,
        model_kind=model_kind,
        ctrl_unit=controller.unit,
        q_start=q_start,
        q_target=q_target,
        ee_target=ee_target,
        ctrlrange=ctrlrange,
        torque_limit=torque_limit,
        t=times[:n_steps],
        **buf,
    )

    if frames:
        save_video(frames, config.VIDEOS_DIR / f"{trial_name}.mp4")
    return trial


# ---------------------------------------------------------------- 指标


def _settling_time(t: np.ndarray, err: np.ndarray, tol: float) -> float:
    """误差最后一次超出容差的时间；从不超出则返回 0。"""
    outside = np.flatnonzero(err > tol)
    if len(outside) == 0:
        return 0.0
    idx = outside[-1]
    return float(t[idx]) if idx + 1 < len(t) else float(t[-1])


def compute_metrics(trial: Trial) -> dict:
    """把一条轨迹压缩成可比较的量化指标。"""
    t = trial.t
    ee_err = np.linalg.norm(trial.ee_pos - trial.ee_target, axis=1)
    z_err = trial.ee_target[2] - trial.ee_pos[:, 2]  # 正值 = 比目标低

    tail = t >= (t[-1] - config.SETTLE_WINDOW)
    joint_err = np.abs(trial.qpos - trial.q_target)

    lo = trial.torque_limit[0]
    hi = trial.torque_limit[1]
    # 饱和要看真正施加的力矩 data.qfrc_actuator，而不是我们写进去的指令：
    # MuJoCo 会把超出上限的指令截断后才作用到关节上。
    saturated = (trial.tau >= hi - 1e-6) | (trial.tau <= lo + 1e-6)

    return {
        "name": trial.name,
        "controller": trial.controller,
        "model_kind": trial.model_kind,
        "ctrl_unit": trial.ctrl_unit,
        "sim_time_s": round(float(t[-1]), 4),
        # 末端高度
        "ee_target_z_m": round(float(trial.ee_target[2]), 5),
        "ee_final_z_m": round(float(trial.ee_pos[-1, 2]), 5),
        "ee_min_z_m": round(float(trial.ee_pos[:, 2].min()), 5),
        "sag_max_m": round(float(z_err.max()), 5),  # 最大下垂量
        # 稳态（末尾 0.5 s）
        "ss_joint_err_max_rad": round(float(joint_err[tail].max()), 6),
        "ss_joint_err_rms_rad": round(float(np.sqrt((joint_err[tail] ** 2).mean())), 6),
        "ss_ee_err_m": round(float(ee_err[tail][-1]), 6),
        "ss_ee_err_mean_m": round(float(ee_err[tail].mean()), 6),
        # 动态
        "settle_time_s": round(_settling_time(t, ee_err, config.EE_TOLERANCE), 4),
        "peak_ee_err_m": round(float(ee_err.max()), 5),
        "final_ee_drift_m": round(float(trial.ee_pos[-1, 2] - trial.ee_pos[0, 2]), 6),
        # 力矩
        "cmd_peak_abs": [round(float(x), 2) for x in np.abs(trial.ctrl).max(axis=0)],
        "tau_peak_abs": [round(float(x), 2) for x in np.abs(trial.tau).max(axis=0)],
        "sat_ratio": round(float(saturated.mean()), 4),
        # 重力项参考：稳态时 tau ≈ qfrc_bias（PD 残差趋零）
        "ss_bias_tau_max": [round(float(x), 3) for x in np.abs(trial.qfrc_bias[tail]).max(axis=0)],
    }


METRIC_LABELS = {
    "sag_max_m": ("最大下垂量 (m)", "max sag (m)"),
    "ss_joint_err_max_rad": ("稳态最大关节误差 (rad)", "ss max joint err (rad)"),
    "ss_ee_err_m": ("稳态末端误差 (m)", "ss EE err (m)"),
    "settle_time_s": ("收敛时间 (s)", "settling time (s)"),
    "final_ee_drift_m": ("末端 z 净漂移 (m)", "final EE z drift (m)"),
    "sat_ratio": ("力矩饱和占比", "torque saturation ratio"),
}


# ---------------------------------------------------------------- 绘图


def plot_trial(trial: Trial, path=None) -> pathlib.Path:
    """三联图：关节角 / 末端高度 / 控制量。"""
    path = pathlib.Path(path) if path else config.FIGURES_DIR / f"{trial.name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(10, 12), sharex=True)
    t = trial.t

    ax = axes[0]
    for k in range(trial.qpos.shape[1]):
        ax.plot(t, trial.qpos[:, k], lw=1.4, label=f"q{k + 1}")
    for k in range(trial.qpos.shape[1]):
        ax.axhline(trial.q_target[k], color=f"C{k}", ls="--", lw=0.8, alpha=0.7)
    ax.set_ylabel(L("关节角 (rad)", "joint angle (rad)"))
    ax.legend(ncol=7, fontsize=8, loc="upper right")
    ax.set_title(
        L(
            f"{trial.name}｜{trial.controller}（ctrl 单位：{trial.ctrl_unit}）",
            f"{trial.name} | {trial.controller} (ctrl unit: {trial.ctrl_unit})",
        )
    )
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(t, trial.ee_pos[:, 2], color="C2", lw=1.6, label=L("末端 z", "EE z"))
    ax.axhline(trial.ee_target[2], color="k", ls="--", lw=1.0, label=L("目标 z", "target z"))
    ax.set_ylabel(L("末端高度 (m)", "EE height (m)"))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    for k in range(trial.tau.shape[1]):
        ax.plot(t, trial.tau[:, k], lw=1.2, label=f"τ{k + 1}")
    lo, hi = trial.torque_limit[0].min(), trial.torque_limit[1].max()
    ax.axhline(hi, color="k", ls=":", lw=0.9, alpha=0.7, label=L("力矩上限", "torque limit"))
    ax.axhline(lo, color="k", ls=":", lw=0.9, alpha=0.7)
    ax.set_ylabel(L("实际关节力矩 (N·m)", "applied joint torque (N·m)"))
    ax.legend(ncol=5, fontsize=8, loc="upper right")
    ax.set_title(L("虚线 = 执行器力矩上限（贴到即为饱和）", "dotted = actuator torque limit (saturated)"))
    ax.grid(alpha=0.3)

    ax = axes[3]
    for k in range(trial.ctrl.shape[1]):
        ax.plot(t, trial.ctrl[:, k], lw=1.2, label=f"{trial.ctrl_unit}[{k + 1}]")
    ax.set_ylabel(
        L("控制指令：关节力矩 (N·m)", "command: joint torque (N·m)")
        if trial.ctrl_unit == "N·m"
        else L("控制指令：关节角 (rad)", "command: joint angle (rad)")
    )
    ax.set_xlabel(L("时间 (s)", "time (s)"))
    ax.legend(ncol=4, fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


def plot_compare(trials, path=None, keys=("ee", "joint")) -> pathlib.Path:
    """多实验叠加对比图。``keys`` 决定画末端高度还是关节角。"""
    path = pathlib.Path(path) if path else config.FIGURES_DIR / "compare.png"
    path.parent.mkdir(parents=True, exist_ok=True)

    colors = ["C0", "C1", "C2", "C3", "C4"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    ax = axes[0]
    for i, tr in enumerate(trials):
        ax.plot(tr.t, tr.ee_pos[:, 2], lw=1.6, color=colors[i % len(colors)],
                label=f"{tr.name}: {tr.controller}")
    ax.axhline(trials[0].ee_target[2], color="k", ls="--", lw=1.0, label=L("目标 z", "target z"))
    ax.set_ylabel(L("末端高度 z (m)", "EE height z (m)"))
    ax.set_title(L("末端高度对比", "End-effector height comparison"))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    joint_idx = 1 if trials[0].qpos.shape[1] > 1 else 0
    for i, tr in enumerate(trials):
        ax.plot(tr.t, tr.qpos[:, joint_idx], lw=1.6, color=colors[i % len(colors)],
                label=f"{tr.name}: {tr.controller}")
    ax.axhline(trials[0].q_target[joint_idx], color="k", ls="--", lw=1.0,
               label=L("目标角", "target angle"))
    ax.set_ylabel(L("关节 2 角度 (rad)", "joint 2 angle (rad)"))
    ax.set_xlabel(L("时间 (s)", "time (s)"))
    ax.set_title(L("关节 2 角度对比", "Joint 2 angle comparison"))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- 视频


def save_video(frames, path, fps: int | None = None) -> pathlib.Path:
    import imageio.v2 as imageio

    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(path, frames, fps=fps or config.VIDEO_FPS, codec="libx264",
                    quality=8, macro_block_size=16)
    return path


# ---------------------------------------------------------------- 指标落盘


def dump_metrics(key: str, metrics: dict) -> dict:
    """把单个实验的指标合并进 results/metrics.json。"""
    config.ensure_dirs()
    all_metrics = {}
    if config.METRICS_PATH.is_file():
        try:
            all_metrics = json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            all_metrics = {}
    all_metrics[key] = metrics
    config.METRICS_PATH.write_text(
        json.dumps(all_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return all_metrics


def load_all_metrics() -> dict:
    if not config.METRICS_PATH.is_file():
        return {}
    try:
        return json.loads(config.METRICS_PATH.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def print_summary(trial: Trial, metrics: dict, video=None, figure=None) -> None:
    print(f"\n===== {trial.name} =====")
    print(f"控制器      : {trial.controller}")
    print(f"模型/执行器 : {trial.model_kind} / ctrl 单位 {trial.ctrl_unit}")
    print(f"起始 qpos   : {np.array2string(trial.q_start, precision=3)}")
    print(f"目标 qpos   : {np.array2string(trial.q_target, precision=3)}")
    for key in ("sag_max_m", "ss_joint_err_max_rad", "ss_ee_err_m", "settle_time_s",
                "final_ee_drift_m", "sat_ratio"):
        zh, en = METRIC_LABELS[key]
        print(f"{L(zh, en):<22}: {metrics[key]}")
    print(f"{'指令峰值 |ctrl|':<22}: {metrics['cmd_peak_abs']}")
    print(f"{'力矩峰值 |τ|':<22}: {metrics['tau_peak_abs']}")
    if figure:
        print(f"图表        : {figure}")
    if video:
        print(f"视频        : {video}")