"""从官方 Panda MJCF 生成两个派生模型。

- ``models/panda_torque.xml``   ：``motor`` 执行器，``data.ctrl`` 单位 N·m（核心实验用）
- ``models/panda_position.xml`` ：``position`` 执行器，``data.ctrl`` 单位 rad，MuJoCo 内部做 PD

基线模型来自 MuJoCo 官方 Menagerie 的 ``franka_emika_panda/panda.xml``（已复制到
``models/official/``，含全部 STL/OBJ 网格资源）。

为什么要改写：
1. 原始执行器是 ``<general>`` 伺服（力矩被封装在 gainprm/biasprm 里），
   拿不到"我写进去多少 N·m"这个直观量，不利于讲清楚执行器模型；
2. 原始模型的关节限位、阻尼（damping=1）、转子惯量（armature=0.1）都保留，
   这样重力力矩与真机一致，实验才有意义。

这里用 :mod:`xml.etree.ElementTree` 做结构化改写（不用正则，避免破坏 XML 结构）。
"""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET

from src import config

OFFICIAL_DIR = config.MODELS_DIR / "official"
OFFICIAL_PANDA = OFFICIAL_DIR / "panda.xml"
OFFICIAL_ASSETS = OFFICIAL_DIR / "assets"

# 固定机位相机：斜俯视整个机械臂，取景范围约 1.44 m 高（z 从 -0.22 到 1.22）。
# xyaxes 是相机自身 x/y 轴在世界系下的方向，由 pos 指向 (0,0,0.5) 的视线叉乘得到。
CAMERA_XML = {
    "name": config.CAMERA_NAME,
    "pos": "1.2 -1.2 0.9",
    "xyaxes": "0.7071 0.7071 0 -0.1622 0.1622 0.9727",
    "fovy": "45",
}

# 纯视觉参考地面：关掉碰撞，保证实验 A 看到的下垂完全由重力造成。
FLOOR_XML = {
    "name": "floor",
    "type": "plane",
    "size": "6 6 0.1",
    "pos": "0 0 0",
    "rgba": "0.86 0.88 0.92 1",
    "contype": "0",
    "conaffinity": "0",
}

OPTION_XML = {
    "timestep": str(config.TIMESTEP),
    "gravity": "0 0 -9.81",
    "integrator": "RK4",  # PD 控制较刚，RK4 比默认 Euler 更稳
    "iterations": "50",
    "ls_iterations": "20",
}


def _check_official_assets() -> None:
    if not OFFICIAL_PANDA.is_file():
        raise FileNotFoundError(
            f"缺少基线模型 {OFFICIAL_PANDA}。"
            "它来自 MuJoCo 官方 Menagerie 的 franka_emika_panda，"
            "请参考 README 重新获取。"
        )
    if not (OFFICIAL_ASSETS / "link0.stl").is_file():
        raise FileNotFoundError(
            f"缺少网格资源目录 {OFFICIAL_ASSETS}，基线模型无法编译。"
        )


def _arm_joint_names_and_ranges():
    """真正加载一次基线模型，从 MuJoCo 里读七个手臂关节的名称与限位。"""
    import mujoco

    _check_official_assets()
    model = mujoco.MjModel.from_xml_path(str(OFFICIAL_PANDA))
    names, ranges = [], []
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) or ""
        if name.startswith("joint") and name[5:].isdigit():
            idx = int(name[5:])
            if 1 <= idx <= config.N_JOINTS:
                names.append((idx, name))
                ranges.append((float(model.jnt_range[i, 0]), float(model.jnt_range[i, 1])))
    if len(names) != config.N_JOINTS:
        raise RuntimeError(f"基线模型中解析到 {len(names)} 个手臂关节，期望 {config.N_JOINTS}")

    order = sorted(range(len(names)), key=lambda k: names[k][0])
    return [names[k][1] for k in order], [ranges[k] for k in order]


def _strip_free_joints(root: ET.Element) -> int:
    """删除所有自由关节，把基座焊死在 world 上（保证 nq = nv，执行器与关节一一对应）。"""
    removed = 0
    for parent in list(root.iter()):
        for child in list(parent):
            if child.tag == "freejoint" or (child.tag == "joint" and child.get("type") == "free"):
                parent.remove(child)
                removed += 1
    return removed


def _replace_option(root: ET.Element) -> None:
    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)
    for key, value in OPTION_XML.items():
        option.set(key, value)


def _fix_meshdir(root: ET.Element) -> None:
    """派生模型与 official/ 同级，网格目录要指向 official/assets。"""
    compiler = root.find("compiler")
    if compiler is not None:
        compiler.set("meshdir", "official/assets")


def _add_visual_extras(root: ET.Element) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        worldbody = ET.SubElement(root, "worldbody")
    existing = {child.get("name") for child in worldbody}
    if "floor" not in existing:
        worldbody.append(ET.Element("geom", dict(FLOOR_XML)))
    if config.CAMERA_NAME not in existing:
        worldbody.append(ET.Element("camera", dict(CAMERA_XML)))


def _replace_actuators(root: ET.Element, joint_names, joint_ranges, kind: str) -> None:
    """删掉原始 <general> 伺服，按需要重建 motor 或 position 执行器。"""
    old = root.find("actuator")
    if old is not None:
        root.remove(old)

    actuator = ET.SubElement(root, "actuator")
    for k, jname in enumerate(joint_names):
        common = {
            "name": f"actuator{k + 1}",
            "joint": jname,
            "ctrllimited": "true",
            "ctrlrange": f"{config.TORQUE_CTRL_RANGE[k][0]:g} {config.TORQUE_CTRL_RANGE[k][1]:g}",
        }
        if kind == "torque":
            # data.ctrl 直接就是关节力矩（N·m），由我们自己的控制器写入
            ET.SubElement(actuator, "motor", common)
        else:
            # 位置伺服：MuJoCo 内部算 tau = kp*(ctrl - q) - kv*qdot，ctrl 单位 rad。
            # 增益与实验 C 完全一致，forcerange 也取同一组力矩上限，
            # 这样"位置伺服"与"纯 PD"在数值上应当给出同一条轨迹（实验 E 会验证）。
            lo, hi = joint_ranges[k]
            force_lo, force_hi = config.TORQUE_CTRL_RANGE[k]
            attrs = dict(
                common,
                ctrlrange=f"{lo:g} {hi:g}",
                forcerange=f"{force_lo:g} {force_hi:g}",
                kp=f"{config.SERVO_KP[k]:g}",
                kv=f"{config.SERVO_KV[k]:g}",
            )
            ET.SubElement(actuator, "position", attrs)


def _drop_keyframes(root: ET.Element) -> None:
    """基线模型的 keyframe 里带着旧的 ctrl，向量化换模型后会失效。"""
    keyframe = root.find("keyframe")
    if keyframe is not None:
        root.remove(keyframe)


def _newest_input_mtime() -> float:
    """基线模型与本脚本中较新的修改时间。

    本脚本变了也必须重建，否则改了生成逻辑却沿用旧产物。
    """
    return max(OFFICIAL_PANDA.stat().st_mtime, pathlib.Path(__file__).stat().st_mtime)


def _all_up_to_date() -> bool:
    newest = _newest_input_mtime()
    return all(
        (config.MODELS_DIR / f"panda_{kind}.xml").is_file()
        and (config.MODELS_DIR / f"panda_{kind}.xml").stat().st_mtime >= newest
        for kind in ("torque", "position")
    )


def build_one(kind: str, joint_names, joint_ranges, force: bool = False) -> pathlib.Path:
    out_path = config.MODELS_DIR / f"panda_{kind}.xml"
    if out_path.is_file() and not force and out_path.stat().st_mtime >= _newest_input_mtime():
        print(f"[build_models] {out_path.name} 已是最新，跳过生成")
        return out_path

    _check_official_assets()
    tree = ET.parse(OFFICIAL_PANDA)
    root = tree.getroot()

    n_removed = _strip_free_joints(root)
    _replace_option(root)
    _fix_meshdir(root)
    _add_visual_extras(root)
    _drop_keyframes(root)
    _replace_actuators(root, joint_names, joint_ranges, kind)
    root.set("model", f"panda_{kind}")

    tree.write(out_path, encoding="utf-8", xml_declaration=True)
    print(f"[build_models] 已生成 {out_path.name}（移除 {n_removed} 个自由关节）")
    return out_path


def build_all(force: bool = False) -> list:
    if not force and _all_up_to_date():
        print("[build_models] 派生模型均为最新，跳过生成")
        return [config.MODELS_DIR / f"panda_{k}.xml" for k in ("torque", "position")]

    joint_names, joint_ranges = _arm_joint_names_and_ranges()
    paths = [
        build_one("torque", joint_names, joint_ranges, force),
        build_one("position", joint_names, joint_ranges, force),
    ]
    print("[build_models] 手臂关节:", ", ".join(joint_names))
    print("[build_models] 关节限位:", ", ".join(f"[{lo:.3f},{hi:.3f}]" for lo, hi in joint_ranges))
    return paths


def summarize(kind: str = "torque") -> None:
    """打印模型的 nq/nv/nu 与执行器参数，用于人工核对。"""
    import mujoco

    from src.model_utils import get_refs, load_model

    model = load_model(kind)
    refs = get_refs(kind)
    print(f"[build_models] {kind}: nq={model.nq} nv={model.nv} nu={model.nu}")
    print(f"[build_models] 手臂关节 qpos 下标 {refs.qpos_adr.tolist()}，末端 body={refs.hand_body}")
    for k, a in enumerate(refs.act_ids):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, int(a))
        print(
            f"  ctrl[{k}] -> {name}: ctrlrange="
            f"[{model.actuator_ctrlrange[a, 0]:g}, {model.actuator_ctrlrange[a, 1]:g}]"
        )


if __name__ == "__main__":
    import sys

    build_all(force="--force" in sys.argv)
    for kind in ("torque", "position"):
        summarize(kind)