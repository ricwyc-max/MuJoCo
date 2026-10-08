"""解析重力补偿：τ_g = −Σ_b J_com,b(q)ᵀ · (m_b · g)。

MuJoCo 的 ``data.qfrc_bias`` = C(q, q̇) + g(q)，静止时只剩重力项。
这里不直接用它，而是按刚体力学的定义自己算一遍，再和 MuJoCo 的结果对拍，
用来验证对"执行器 / 广义力"模型的理解（作业考察点之一）。

两个容易踩的坑，都在下面的实现里处理了：

1) **作用点在质心，不在 body 坐标系原点。**
   ``mj_jacBody`` 给的是 body 原点的雅可比，而重力作用在质心上。
   设 body 的世界姿态为 R、质心在该 body 坐标系下的偏移为 ipos，则

       v_com = v_origin + ω × (R·ipos)
       J_com = J_origin + skew(ω) · (R·ipos)

2) **符号。** 重力是施加在系统上的**外力**，它在关节空间产生的广义力是
   ``J_comᵀ(m·g)``；而 ``qfrc_bias`` 里存的是与之相反的量（见运动方程
   ``M q̈ + qfrc_bias = τ_actuator + τ_applied``）。所以

       τ_gravity_compensation = qfrc_bias|_q̇=0 = −Σ J_comᵀ(m·g)

   这个 τ 才是"要保持住位形，电机得给出多大的力矩"，可以直接前馈进控制律。
"""

from __future__ import annotations

import numpy as np


def gravity_torque(model, data, refs, out: np.ndarray | None = None) -> np.ndarray:
    """返回重力补偿力矩（前馈项），单位 N·m，形状 (nv,)。

    静止时它应当与 ``data.qfrc_bias`` 的对应分量完全相等。
    """
    import mujoco

    if out is None:
        out = np.zeros(model.nv)

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    gravity = np.asarray(model.opt.gravity, dtype=float)
    out[:] = 0.0

    for body_id in range(1, model.nbody):  # 0 是 world
        mass = float(model.body_mass[body_id])
        if mass <= 0.0:
            continue
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        rot = np.asarray(data.xmat[body_id]).reshape(3, 3)
        com_offset = rot @ np.asarray(model.body_ipos[body_id], dtype=float)
        # J_com = J_origin + ω × r = J_origin − (r × ω)
        jac_com = jacp - np.cross(com_offset[:, None], jacr, axis=0)
        out[:] -= jac_com.T @ (mass * gravity)  # 重力是外力，取负号得到保持力矩

    return out[refs.dof_adr]


def compare_with_qfrc_bias(model, data, refs) -> float:
    """返回解析重力补偿力矩与 MuJoCo ``qfrc_bias`` 的最大绝对偏差。

    仅在 q̇ = 0 时二者应当相等（此时 qfrc_bias 只剩重力项）。
    """
    qd = np.asarray(data.qvel, dtype=float)
    if np.max(np.abs(qd)) > 1e-12:
        raise ValueError("qfrc_bias 在 q̇≠0 时还包含科氏/离心项，两者不应直接比较")

    tau_g = gravity_torque(model, data, refs)
    return float(np.max(np.abs(tau_g - np.asarray(data.qfrc_bias)[refs.dof_adr])))