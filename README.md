# MuJoCo 课程作业 —— 任务二：重力补偿与保持

> 让 Franka Emika **Panda** 七轴机械臂在重力作用下稳定保持在一个指定的关节构型上。

作业要求见 `微信图片_20261007163858_5_56.jpg`。

| 交付物 | 说明 |
|---|---|
| **[MuJoCo课程作业报告.docx](MuJoCo课程作业报告.docx)** | 论文排版标准的 Word 版（含封面、目录、三线表、图题），可直接提交 |
| [REPORT.md](REPORT.md) | 同一份报告的 Markdown 源文件，改完重新生成 docx 即可 |

---

## 一、这个项目做了什么

在 MuJoCo 里搭了一套 Panda 机械臂仿真，用**自己写进 `data.ctrl` 的关节力矩**控制它，
逐级回答作业的四个要求：

| 实验 | 内容 | 对应作业要求 |
|---|---|---|
| **A** | `data.ctrl ≡ 0`，完全不通电 → 观察重力下垂 | 要求 2 |
| **B** | 用 MuJoCo 内置 `<position>` 位置伺服 | 考察点（伺服对照） |
| **C** | 自己写 PD：`τ = Kp(q*-q) - Kd·q̇` | 要求 3 前半 |
| **D** | **PD + 重力补偿**：`τ = Kp(q*-q) - Kd·q̇ + qfrc_bias` | **要求 3 后半（核心答案）** |
| **E** | A/B/C/D 四组横向对比 | 考察点（伺服 vs 力矩） |
| **F** | 解析重力 `−Σ Jᵀ(mg)` 与 `data.qfrc_bias` 对拍 | 考察点（广义力模型） |
| **G** | 外部持续负载下三种控制器的保持能力 | 深入理解"保持"的边界 |
| **H** | PD 增益扫描：超调 / 收敛时间 / 稳态精度 | 增益怎么选 |

**核心结论**：任何有限的 P 增益都无法消除重力造成的稳态误差（误差 ≈ g(q)/Kp ≈ 0.026 rad）；
把重力作为前馈项加进控制律后，稳态关节误差降到 **2×10⁻⁶ rad**、末端误差 **0 m**，
构型被严格保持。实验 F 进一步验证解析算出的重力矩与 MuJoCo 内部值一致到 **1.4×10⁻¹⁴**。

---

## 二、环境搭建（项目内虚拟环境 + 国内镜像）

```bash
bash scripts/setup_venv.sh
```

脚本会做三件事：创建项目根目录下的 `.venv`、按清华 → 阿里云 → 北外的顺序自动挑镜像装依赖、
打印装好的版本。也可以手动执行：

```bash
D:/python3.11/python.exe -m venv .venv
.venv/Scripts/python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -U pip
.venv/Scripts/python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

> 依赖：`mujoco`（3.15）、`numpy`、`matplotlib`、`imageio`、`imageio-ffmpeg`。
> 生成 Word 报告另需 `python-docx`（不在 `requirements.txt` 里，单独装即可）。

### 关于模型来源

MuJoCo 从 3.3 起不再把机器人模型打进 pip 包，官方仓库里也已移除。
本项目使用 **MuJoCo 官方 Menagerie** 的 `franka_emika_panda`，已完整复制到
`models/official/`（含 67 个 STL/OBJ 网格，约 31 MB），项目因此完全自包含、不依赖外部下载。

---

## 三、运行

```bash
# 一键跑完 A~H，并做定量验收
.venv/Scripts/python.exe scripts/run_all.py
```

单个实验也可以独立运行：

```bash
.venv/Scripts/python.exe src/experiments/exp_d_pd_gravity.py
```

改完 `REPORT.md` 后重新生成 Word 报告：

```bash
.venv/Scripts/python.exe -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple python-docx
.venv/Scripts/python.exe scripts/make_report_docx.py
```

脚本顶部可改学号/姓名/班级；打开 docx 时 Word 会提示更新目录域，选"是"即可。
若不提示，`Ctrl+A` 后按 `F9` 手动更新。

首次运行会自动从 `models/official/panda.xml` 生成两个派生模型：
`models/panda_torque.xml`（力矩执行器）与 `models/panda_position.xml`（位置伺服）。
只要基线模型或生成脚本有改动就会自动重建。

---

## 四、目录结构

```
.
├── models/
│   ├── official/              官方 Menagerie Panda 模型 + 网格资源
│   ├── panda_torque.xml       派生模型：<motor> 力矩执行器（ctrl 单位 N·m）
│   └── panda_position.xml     派生模型：<position> 伺服（ctrl 单位 rad）
├── src/
│   ├── config.py              目标构型、PD 增益、仿真时长等全部可调参数
│   ├── build_models.py        从官方 XML 生成两个派生模型
│   ├── model_utils.py         关节/执行器/末端索引解析、离屏渲染
│   ├── controllers.py         Zero / PD / PD+重力补偿 / 位置伺服 / 负载前馈
│   ├── recorder.py            仿真主循环、指标计算、绘图、视频
│   ├── analytic_gravity.py    解析重力补偿 τ = −Σ J_comᵀ(mg)
│   └── experiments/           八个实验脚本，每个都可独立运行
├── scripts/
│   ├── setup_venv.sh          建环境（国内镜像）
│   ├── run_all.py             一键跑完 + 验收
│   └── make_report_docx.py    REPORT.md → 论文格式 Word
└── results/
    ├── videos/                7 个 mp4（videos 已 gitignore）
    ├── figures/               15 张 png
    ├── trials/                每条轨迹的 npz
    ├── metrics.json           全部量化指标
    └── exp_H_gain_sweep.csv   增益扫描原始数据
```

---

## 五、结果索引

| 图表 / 视频 | 内容 |
|---|---|
| `results/videos/exp_A_no_control.mp4` | 无控制：手臂在重力下塌到地面 |
| `results/videos/exp_B_position_servo.mp4` | 位置伺服：被拉回但有稳态偏差 |
| `results/videos/exp_C_pd_torque.mp4` | 纯 PD：被拉回但有稳态偏差 |
| `results/videos/exp_D_pd_gravity.mp4` | **PD+重力补偿：全程严格保持** |
| `results/videos/exp_G_disturbance_*.mp4` | 外部负载下三种控制器的表现 |
| `results/figures/exp_E_compare_traj.png` | 四组末端高度 / 关节角叠加对比 |
| `results/figures/exp_E_compare_metrics.png` | 四组稳态误差柱状对比 |
| `results/figures/exp_F_analytic_gravity.png` | 解析重力 vs `qfrc_bias`，及重力矩随构型变化 |
| `results/figures/exp_G_disturbance.png` | 外负载下误差曲线 + 实测/理论稳态误差 |
| `results/figures/exp_H_gain_sweep_*.png` | 增益扫描热力图 |

完整数据见 `results/metrics.json`。

---

## 六、常见问题

**没有视频 / 渲染器创建失败**
MuJoCo 需要 OpenGL 上下文。渲染失败时实验仍会跑完并出图。想解决可试
`set MUJOCO_GL=osmesa`（需另装 osmesa），或在带图形界面的环境下运行。

**中文图表变成方块**
`src/recorder.py` 会自动在 Microsoft YaHei / SimHei / Noto CJK 中挑一个；
都没有时图表标签自动降级为英文，不会报错。

**终端中文乱码**
所有脚本在导入 `src` 时都会把 stdout/stderr 切到 UTF-8（见 `src/__init__.py`）。