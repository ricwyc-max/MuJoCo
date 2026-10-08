"""四种控制器：零控制 / 纯 PD / PD + 重力补偿 / 位置伺服。

统一接口：``controller.compute(model, data, refs) -> np.ndarray(7)``，
返回值直接写进 ``data.ctrl[refs.act_ids]``。
"""

from __future__ import annotations

import dataclasses

import numpy as np

from src import config


@dataclasses.dataclass
class Controller:
    """控制器基类。``name`` 用于日志与图例，``unit`` 说明 ctrl 的物理量纲。"""

    name: str = "base"
    unit: str = "N·m"
    description: str = ""

    def compute(self, model, data, refs) -> np.ndarray:  # pragma: no cover - 抽象
        raise NotImplementedError


@dataclasses.dataclass
class ZeroController(Controller):
    """实验 A：不做任何控制，data.ctrl 全零，看机械臂如何在重力下下垂。"""

    name: str = "A 无控制"
    unit: str = "N·m"
    description: str = "data.ctrl ≡ 0，纯开环观察重力下垂"

    def compute(self, model, data, refs) -> np.ndarray:
        return np.zeros(refs.n)


@dataclasses.dataclass
class PDController(Controller):
    """实验 C：标准 PD 关节空间控制。

        tau = Kp * (q* - q) - Kd * qdot

    稳态时重力力矩需要由比例项独自承担，误差不为零：tau = g(q) -> q* - q = g(q)/Kp。
    """

    name: str = "C 纯 PD"
    unit: str = "N·m"
    description: str = "tau = Kp(q*-q) - Kd·qdot，无重力项，存在稳态误差"
    q_target: np.ndarray = dataclasses.field(default_factory=lambda: config.Q_TARGET.copy())
    kp: np.ndarray = dataclasses.field(default_factory=lambda: config.KP.copy())
    kd: np.ndarray = dataclasses.field(default_factory=lambda: config.KD.copy())
    # 额外的前馈力矩（实验 G 用来模拟"已知外负载"），None 表示没有
    extra_tau: np.ndarray | None = None

    def compute(self, model, data, refs) -> np.ndarray:
        q = data.qpos[refs.qpos_adr]
        qd = data.qvel[refs.dof_adr]
        tau = self.kp * (self.q_target - q) - self.kd * qd
        if self.extra_tau is not None:
            tau = tau + self.extra_tau
        return tau


@dataclasses.dataclass
class PDGravityController(PDController):
    """实验 D：PD + 重力补偿（作业核心答案）。

        tau = Kp * (q* - q) - Kd * qdot + g(q)

    其中 ``data.qfrc_bias`` 就是 MuJoCo 算好的偏置广义力 C(q,qdot) + g(q)；
    静止时它恰等于重力力矩，于是比例项只需负责纠正初始偏差，稳态误差趋于 0。
    """

    name: str = "D PD + 重力补偿"
    unit: str = "N·m"
    description: str = "tau = Kp(q*-q) - Kd·qdot + qfrc_bias，稳态误差≈0"

    def compute(self, model, data, refs) -> np.ndarray:
        return super().compute(model, data, refs) + data.qfrc_bias[refs.dof_adr]


@dataclasses.dataclass
class PositionServoController(Controller):
    """实验 B：使用 MuJoCo 自带的 position 执行器。

    此时 ``data.ctrl`` 的单位是 **关节角（rad）**，MuJoCo 内部按

        tau = kp * (ctrl - q) - kv * qdot

    计算力矩。我们只写位置，增益被封装在 XML 里。
    """

    name: str = "B 位置伺服"
    unit: str = "rad"
    description: str = "ctrl 是关节角，MuJoCo 内部实现 tau = kp(ctrl-q) - kv·qdot"
    q_target: np.ndarray = dataclasses.field(default_factory=lambda: config.Q_TARGET.copy())

    def compute(self, model, data, refs) -> np.ndarray:
        return self.q_target.copy()


@dataclasses.dataclass
class PDGravityLoadController(PDGravityController):
    """实验 G 用：在重力补偿之外，再把**已知**的外部负载也加进前馈。

    重力补偿只能抵消"模型已经知道"的扰动；把已知外负载也前馈进去，
    才能把稳态误差压到 0 —— 这正是扰动观测器 / 前馈-反馈复合控制的思想。
    """

    name: str = "PD + 重力补偿 + 负载前馈"
    description: str = "在重力补偿基础上再前馈已知外部负载，稳态误差归零"


@dataclasses.dataclass
class AnalyticGravityController(PDController):
    """实验 F 用：把重力力矩换成自己解析计算的 Jᵀ·m·g，而不是直接读 qfrc_bias。

    用于验证"解析推导 == MuJoCo 内部计算"。
    """

    name: str = "F PD + 解析重力补偿"
    unit: str = "N·m"
    description: str = "tau = Kp(q*-q) - Kd·qdot + Σ Jᵀ(mg)，重力项由质心雅可比自己算"

    def compute(self, model, data, refs) -> np.ndarray:
        from src.analytic_gravity import gravity_torque

        return super().compute(model, data, refs) + gravity_torque(model, data, refs)