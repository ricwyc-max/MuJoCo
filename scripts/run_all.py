"""一键跑完全部实验，并做定量验收。

    .venv/Scripts/python.exe scripts/run_all.py

每个实验单独跑（带视频渲染），跑完打印验收结论。
任何单个实验失败不会中断整体，最后统一汇总。
"""

from __future__ import annotations

import pathlib
import sys
import time
import traceback

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src import config  # noqa: E402
from src.recorder import load_all_metrics  # noqa: E402

# (脚本模块, 说明)
EXPERIMENTS = [
    ("exp_a_no_control", "A 无控制：观察重力下垂（作业要求 2）"),
    ("exp_b_position_servo", "B 位置伺服：MuJoCo 内置 <position> 伺服（考察点对照）"),
    ("exp_c_pd_torque", "C 纯 PD：自己写 data.ctrl 施加力矩（作业要求 3 前半）"),
    ("exp_d_pd_gravity", "D PD + 重力补偿：稳态零误差（作业要求 3 后半 / 核心答案）"),
    ("exp_e_compare", "E 四组横向对比：位置伺服 vs 力矩控制（考察点）"),
    ("exp_f_analytic_gravity", "F 解析重力补偿 J^T·m·g 与 qfrc_bias 对拍（理解广义力模型）"),
    ("exp_g_disturbance", "G 外部负载下的保持能力与重力补偿的边界"),
    ("exp_h_gain_sweep", "H PD 增益扫描：超调 / 收敛时间 / 稳态精度"),
]


def _acceptance() -> dict:
    """每个实验的验收判据，不满足就在最后标记 FAIL。

    所有判据都接收完整的 metrics 字典，方便跨实验互相对照。
    """

    def a(m):
        sag = m["exp_A_no_control"]["sag_max_m"]
        return sag > 0.05, f"无控制下垂量 {sag} m > 0.05 m，重力下垂现象清晰"

    def b(m):
        jb = m["exp_B_position_servo"]["ss_joint_err_max_rad"]
        jc = m["exp_C_pd_torque"]["ss_joint_err_max_rad"]
        ok = jb < 1e-1 and abs(jb - jc) / jc < 1e-3
        return ok, f"位置伺服稳态误差 {jb:.4e} rad ≈ 纯 PD 的 {jc:.4e} rad（内部 PD 同源）"

    def c(m):
        e = m["exp_C_pd_torque"]["ss_joint_err_max_rad"]
        return e > 1e-3, f"纯 PD 保留 {e:.4e} rad 稳态误差（重力无法被 PD 消除）"

    def d(m):
        e = m["exp_D_pd_gravity"]["ss_joint_err_max_rad"]
        ee = m["exp_D_pd_gravity"]["ss_ee_err_m"]
        ok = e < 1e-3 and ee < 5e-3
        return ok, f"稳态关节误差 {e:.2e} rad、末端误差 {ee:.2e} m，构型被保持"

    def e(m):
        diff = m["exp_E_compare"]["B_vs_C_steady_joint_diff_rad"]
        ok = diff < 1e-6
        return ok, f"B/C 稳态轨迹偏差 {diff:.2e} rad，证实位置伺服 ≡ PD"

    def f(m):
        err = m["exp_F_analytic_gravity"]["max_abs_err"]
        return err < 1e-9, f"解析重力 vs qfrc_bias 最大偏差 {err:.2e}"

    def g(m):
        err = m["exp_G_disturbance"]["exp_G_disturbance_feedforward"]["ss_joint_err_max_all_rad"]
        return err < 1e-3, f"负载前馈后加载期间稳态误差 {err:.2e} rad"

    return {
        "exp_A_no_control": a,
        "exp_B_position_servo": b,
        "exp_C_pd_torque": c,
        "exp_D_pd_gravity": d,
        "exp_E_compare": e,
        "exp_F_analytic_gravity": f,
        "exp_G_disturbance": g,
    }


def main() -> int:
    from src import build_models

    print("=" * 78)
    print("MuJoCo 任务二：重力补偿与保持 —— 全量实验")
    print("=" * 78)
    build_models.build_all()

    results: dict[str, tuple[bool, str]] = {}
    failed_scripts: list[str] = []

    for module_name, description in EXPERIMENTS:
        print(f"\n{'-' * 78}\n>>> {module_name} — {description}\n{'-' * 78}")
        t0 = time.perf_counter()
        try:
            __import__(f"src.experiments.{module_name}")
            sys.modules[f"src.experiments.{module_name}"].main()
        except Exception:  # noqa: BLE001 - 单个实验失败不应中断整体
            failed_scripts.append(module_name)
            traceback.print_exc()
        print(f"[耗时] {module_name}: {time.perf_counter() - t0:.1f} s")

    from src.model_utils import close_renderers

    close_renderers()

    print(f"\n{'=' * 78}\n验收汇总\n{'=' * 78}")
    metrics = load_all_metrics()
    checks = _acceptance()
    n_pass = 0
    for key, check in checks.items():
        if key not in metrics:
            results[key] = (False, "指标缺失（实验未成功完成）")
            print(f"  [FAIL] {key:<24} 指标缺失")
            continue
        try:
            ok, message = check(metrics)
        except (KeyError, TypeError) as exc:
            ok, message = False, f"指标结构异常：{exc}"
        results[key] = (ok, message)
        n_pass += ok
        print(f"  [{'PASS' if ok else 'FAIL'}] {key:<24} {message}")

    if failed_scripts:
        print(f"\n执行异常的脚本：{', '.join(failed_scripts)}")

    print(f"\n合计 {n_pass}/{len(checks)} 项验收通过")
    print(f"指标汇总: {config.METRICS_PATH}")
    print(f"图表    : {config.FIGURES_DIR}")
    print(f"视频    : {config.VIDEOS_DIR}")
    return 0 if n_pass == len(checks) and not failed_scripts else 1


if __name__ == "__main__":
    raise SystemExit(main())