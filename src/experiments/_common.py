"""实验脚本的公共收尾逻辑。

每个实验都可以单独运行（脚本自己在 import 前把项目根加进 sys.path）：
    .venv/Scripts/python.exe src/experiments/exp_d_pd_gravity.py
"""

from __future__ import annotations

from src import config
from src.recorder import (
    Trial,
    compute_metrics,
    dump_metrics,
    plot_trial,
    print_summary,
)


def finish(trial: Trial, key: str) -> tuple[Trial, dict]:
    """统一收尾：算指标 → 存轨迹 → 出图 → 落盘指标 → 打印摘要。"""
    config.ensure_dirs()
    metrics = compute_metrics(trial)
    figure = plot_trial(trial)
    trial.save(key)
    dump_metrics(key, metrics)
    video = config.VIDEOS_DIR / f"{key}.mp4"
    print_summary(trial, metrics, video=video if video.is_file() else None, figure=figure)
    return trial, metrics