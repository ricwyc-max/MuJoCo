"""MuJoCo 模型的加载与常用查询。

这一层解决两件事：
1. 把模型里的关节 / 执行器 / 末端解析成索引，避免在业务代码里硬编码 id；
2. 提供末端位置查询与离屏渲染封装。
"""

from __future__ import annotations

import dataclasses
import re

import numpy as np

from src import config

OFFICIAL_DIR = config.MODELS_DIR / "official"

# 匹配 "joint1" / "panda_joint1" 这类主关节名
_JOINT_NAME_RE = re.compile(r"^(?:panda_)?joint(\d+)$")

# ---------------------------------------------------------------- 模型加载

_MODEL_CACHE: dict[str, object] = {}
_REFS_CACHE: dict[str, "PandaRefs"] = {}
_RENDERER_CACHE: dict[str, object] = {}


@dataclasses.dataclass(frozen=True)
class PandaRefs:
    """Panda 各关键元素的 id 索引。"""

    qpos_adr: np.ndarray  # (7,) 七个手臂关节在 data.qpos 中的起始下标
    dof_adr: np.ndarray  # (7,) 七个手臂关节在 data.qvel / qfrc_* 中的起始下标
    act_ids: np.ndarray  # (7,) 与上述关节一一对应的执行器 id
    hand_body: int  # 末端连杆（hand）的 body id
    ee_site: int  # 末端 site id，没有则为 -1

    @property
    def n(self) -> int:
        return len(self.qpos_adr)


def build_refs(model) -> PandaRefs:
    """从模型中解析出关节 / 执行器 / 末端的索引。"""
    import mujoco

    # --- 七个主关节：按名称里的序号排序，不依赖在 XML 中的出现顺序 ---
    found: dict[int, int] = {}
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
        m = _JOINT_NAME_RE.match(name)
        if m and 1 <= int(m.group(1)) <= config.N_JOINTS:
            found[int(m.group(1))] = i
    missing = sorted(set(range(1, config.N_JOINTS + 1)) - set(found))
    if missing:
        raise RuntimeError(f"模型中缺少主关节 {missing}，无法解析关节索引")

    jnt_ids = [found[k] for k in range(1, config.N_JOINTS + 1)]
    qpos_adr = model.jnt_qposadr[jnt_ids].astype(int)
    dof_adr = model.jnt_dofadr[jnt_ids].astype(int)

    # --- 执行器：按其驱动的关节重排，保证 ctrl[k] 对应第 k 个手臂关节 ---
    actuator_of_joint: dict[int, int] = {}
    for a in range(model.nu):
        if model.actuator_trntype[a] != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        jid = int(model.actuator_trnid[a, 0])
        if jid in jnt_ids:
            actuator_of_joint[jnt_ids.index(jid)] = a
    if len(actuator_of_joint) != config.N_JOINTS:
        raise RuntimeError(
            f"期望 {config.N_JOINTS} 个关节执行器，实际找到 {len(actuator_of_joint)} 个"
        )
    act_ids = np.array([actuator_of_joint[k] for k in range(config.N_JOINTS)], dtype=int)

    # --- 末端 body：优先 hand，否则取名字带 panda_/link 的最大编号连杆 ---
    names = {i: (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "") for i in range(model.nbody)}
    hand_body = next(
        (i for i, n in names.items() if "hand" in n),
        max((i for i, n in names.items() if n.startswith(("panda_", "link"))), default=-1),
    )
    if hand_body < 0:
        raise RuntimeError("模型中找不到末端 body")

    # --- 末端 site：有就用 site，没有就退化到 body 原点 ---
    ee_site = -1
    for i in range(model.nsite):
        name = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i) or "").lower()
        if any(k in name for k in ("attach", "tcp", "eef", "grasp")):
            ee_site = i
            break

    return PandaRefs(qpos_adr=qpos_adr, dof_adr=dof_adr, act_ids=act_ids,
                     hand_body=int(hand_body), ee_site=int(ee_site))


def load_model(kind: str = "torque"):
    """加载派生模型。kind 取 'torque'（力矩执行器）或 'position'（位置伺服）。"""
    import mujoco

    if kind not in ("torque", "position"):
        raise ValueError(f"未知模型类型 {kind!r}，只支持 'torque' / 'position'")
    if kind in _MODEL_CACHE:
        return _MODEL_CACHE[kind]

    # 保证派生模型与当前生成逻辑一致（已是最新时只是一次 stat 比较）
    from src import build_models

    build_models.build_all()

    model = mujoco.MjModel.from_xml_path(str(config.MODELS_DIR / f"panda_{kind}.xml"))
    model.opt.timestep = config.TIMESTEP
    _MODEL_CACHE[kind] = model
    _REFS_CACHE[kind] = build_refs(model)
    return model


def get_refs(kind: str = "torque") -> PandaRefs:
    load_model(kind)
    return _REFS_CACHE[kind]


def make_data(model):
    """新建 MjData 并完成一次正运动学，保证 qfrc_bias 等派生量已就绪。"""
    import mujoco

    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return data


def ee_position(data, refs: PandaRefs) -> np.ndarray:
    """末端位置（世界坐标，米）。有 site 用 site，否则用 hand body 原点。"""
    if refs.ee_site >= 0:
        return np.array(data.site_xpos[refs.ee_site], dtype=float)
    return np.array(data.xpos[refs.hand_body], dtype=float)


def ee_position_at(model, refs: PandaRefs, q: np.ndarray) -> np.ndarray:
    """给定关节角 q，返回该构型下的末端位置（正运动学）。"""
    import mujoco

    data = make_data(model)
    data.qpos[refs.qpos_adr] = q
    mujoco.mj_forward(model, data)
    return ee_position(data, refs)


# ---------------------------------------------------------------- 渲染


def make_renderer(kind: str = "torque"):
    """创建离屏渲染器；失败时返回 None（实验仍可跑完并出图）。"""
    if kind in _RENDERER_CACHE:
        return _RENDERER_CACHE[kind]

    import mujoco

    model = load_model(kind)
    renderer = None
    try:
        try:
            renderer = mujoco.Renderer(model, height=config.RENDER_HEIGHT,
                                      width=config.RENDER_WIDTH)
        except TypeError:  # 极老版本只接受位置参数
            renderer = mujoco.Renderer(model, config.RENDER_WIDTH, config.RENDER_HEIGHT)
    except Exception as exc:  # noqa: BLE001 - 无 OpenGL 上下文时不该中断实验
        print(f"[model_utils] 渲染器创建失败，本次不输出视频：{exc}")
        print("[model_utils] 如需视频，可尝试设置环境变量 MUJOCO_GL=osmesa")
        renderer = None

    _RENDERER_CACHE[kind] = renderer
    return renderer


def render(renderer, model, data, camera: str = config.CAMERA_NAME) -> np.ndarray:
    renderer.update_scene(data, camera=camera)
    return renderer.render()


def close_renderers() -> None:
    for renderer in _RENDERER_CACHE.values():
        if renderer is not None:
            try:
                renderer.close()
            except Exception:  # noqa: BLE001
                pass
    _RENDERER_CACHE.clear()