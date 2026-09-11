"""Build a MuJoCo model of the xArm Lite 6 with the Ability Hand as gripper.

Standalone MuJoCo, not Gazebo or RViz: one model, one viewer, and the same
physics the safety layer already runs on.

The arm is generated directly from the kinematics that xarm_description ships
(`config/kinematics/default/lite6_default_kinematics.yaml`) plus the STL meshes
under `meshes/lite6/`. Every xArm joint rotates about its own local Z, so the
chain is just six fixed origins with a hinge at each - no DH table and no
reconstructed link lengths.

The hand is compiled from ah_urdf as usual, dumped to MJCF, and its bodies are
spliced onto the arm's last link at the mount transform. That keeps one source
of truth per robot: the arm's geometry stays with xarm_description and the
hand's with ah_urdf.

MuJoCo does not read DAE, so the .stl variants are used.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

from ah_mujoco.arm_teleop import LITE6_LIMITS, LITE6_ORIGINS

ARM_JOINT_NAMES = [f"arm_joint{i + 1}" for i in range(6)]
LINK_MESHES = ["link_base", "link1", "link2", "link3", "link4", "link5", "link6"]


def _euler_str(roll, pitch, yaw):
    return f"{roll} {pitch} {yaw}"


def find_xarm_description(hint=None):
    """Locate xarm_description, whether installed or sitting in src/."""
    if hint:
        p = Path(hint)
        if (p / "meshes" / "lite6").exists():
            return p
    try:
        from ament_index_python.packages import get_package_share_directory

        p = Path(get_package_share_directory("xarm_description"))
        if (p / "meshes" / "lite6").exists():
            return p
    except Exception:
        pass
    for base in [Path("/ws/src"), Path.cwd(), Path(__file__).resolve().parents[4]]:
        for cand in base.rglob("xarm_description"):
            if (cand / "meshes" / "lite6").exists():
                return cand
    raise FileNotFoundError(
        "xarm_description not found. Pass xarm_description_path, or make sure "
        "the repo is in the workspace."
    )


def build_arm_mjcf(xarm_path, hand_mjcf=None, mount_xyz=(0, 0, 0),
                   mount_rpy=(0, 0, 0), kp=200.0, kv_ratio=0.02,
                   timestep=0.002):
    """Return MJCF text for the Lite 6, optionally with a hand spliced on.

    hand_mjcf : MJCF string of the hand (from mj_saveLastXML on the compiled
        ah_urdf model). Its assets and its root body's children are merged in.
    """
    xarm_path = Path(xarm_path)
    visual = xarm_path / "meshes" / "lite6" / "visual"
    collision = xarm_path / "meshes" / "lite6" / "collision"

    assets = []
    for name in LINK_MESHES:
        vis = visual / f"{name}.stl"
        col = collision / f"{name}.stl"
        if vis.exists():
            assets.append(
                f'<mesh name="arm_{name}" file="{vis}"/>'
            )
        if col.exists():
            assets.append(
                f'<mesh name="arm_{name}_col" file="{col}"/>'
            )

    # Nest the six links, each at its fixed origin with a hinge about local Z
    body_open, body_close = [], []
    for i, (x, y, z, r, p, yw) in enumerate(LITE6_ORIGINS):
        lo, hi = LITE6_LIMITS[i]
        mesh = f"link{i + 1}"
        body_open.append(
            f'<body name="arm_link{i + 1}" pos="{x} {y} {z}" '
            f'euler="{_euler_str(r, p, yw)}">'
            f'<joint name="{ARM_JOINT_NAMES[i]}" type="hinge" axis="0 0 1" '
            f'range="{lo} {hi}" damping="2.0" armature="0.05"/>'
            f'<geom type="mesh" mesh="arm_{mesh}" class="armvis"/>'
            f'<geom type="mesh" mesh="arm_{mesh}_col" class="armcol"/>'
        )
        body_close.append("</body>")

    # Mount frame at the end of link6, where the hand attaches
    mx, my, mz = mount_xyz
    hand_block = (
        f'<body name="hand_mount" pos="{mx} {my} {mz}" '
        f'euler="{_euler_str(*mount_rpy)}">{{HAND}}</body>'
    )

    actuators = "".join(
        f'<position name="act_{n}" joint="{n}" kp="{kp}" '
        f'kv="{kp * kv_ratio}" ctrlrange="{lo} {hi}" forcerange="-60 60"/>'
        for n, (lo, hi) in zip(ARM_JOINT_NAMES, LITE6_LIMITS)
    )

    mjcf = f"""<mujoco model="lite6_ability_hand">
  <!-- eulerseq XYZ = extrinsic, which is what URDF rpy means. MuJoCo's
       default is intrinsic "xyz"; using it here puts the flange 0.56 m from
       where the kinematics say it should be. -->
  <compiler angle="radian" autolimits="true" eulerseq="XYZ"/>
  <option timestep="{timestep}" integrator="implicitfast" cone="elliptic"/>
  <default>
    <default class="armvis">
      <geom contype="0" conaffinity="0" group="2" rgba="0.75 0.76 0.78 1"/>
    </default>
    <default class="armcol">
      <geom contype="1" conaffinity="1" group="3" rgba="0.5 0.5 0.5 0"/>
    </default>
  </default>
  <asset>
    <texture name="ah_sky" type="skybox" builtin="flat"
             rgb1="0.071 0.078 0.094" width="8" height="8"/>
    {''.join(assets)}
  </asset>
  <worldbody>
    <light pos="0 0 2" dir="0 0 -1" diffuse="0.6 0.6 0.6"/>
    <body name="arm_base" pos="0 0 0">
      <geom type="mesh" mesh="arm_link_base" class="armvis"/>
      <geom type="mesh" mesh="arm_link_base_col" class="armcol"/>
      {''.join(body_open)}
      {hand_block}
      {''.join(body_close)}
    </body>
  </worldbody>
  <actuator>{actuators}</actuator>
</mujoco>"""
    return mjcf


def _extract_hand(hand_mjcf):
    """Pull the hand's assets and body subtree out of its own MJCF.

    Mesh paths are made absolute: they are relative to the hand's own
    <compiler meshdir=...>, which does not survive the merge into a model that
    has its own meshdir for the arm.
    """
    root = ET.fromstring(hand_mjcf)

    comp = root.find("compiler")
    meshdir = comp.get("meshdir", "") if comp is not None else ""

    asset = root.find("asset")
    if asset is not None and meshdir:
        for mesh in asset.iter("mesh"):
            f = mesh.get("file")
            if f and not Path(f).is_absolute():
                mesh.set("file", str((Path(meshdir) / f).resolve()))
    asset_xml = "".join(ET.tostring(c, encoding="unicode")
                        for c in (asset if asset is not None else []))

    world = root.find("worldbody")
    bodies = list(world) if world is not None else []
    body_xml = "".join(ET.tostring(b, encoding="unicode") for b in bodies)

    extras = ""
    for tag in ("equality", "contact"):
        el = root.find(tag)
        if el is not None:
            extras += ET.tostring(el, encoding="unicode")
    return asset_xml, body_xml, extras


def build_combined_model(hand_urdf, xarm_description_path=None,
                         mount_xyz=(0.0, 0.0, 0.0), mount_rpy=(0.0, 0.0, 0.0),
                         timestep=0.002, save_xml_to=None):
    """Compile arm + hand into one MjModel.

    Returns (model, mjcf_text).
    """
    from ah_mujoco.physics_model import build_physics_model

    # Compile the hand on its own first, so it keeps its mimic constraints,
    # actuators and contact exclusions
    hand_model, _ = build_physics_model(hand_urdf, timestep=timestep)
    tmp = "/tmp/_ah_hand_for_arm.xml"
    mujoco.mj_saveLastXML(tmp, hand_model)
    hand_mjcf = Path(tmp).read_text()
    hand_assets, hand_bodies, hand_extras = _extract_hand(hand_mjcf)

    xarm_path = find_xarm_description(xarm_description_path)
    mjcf = build_arm_mjcf(
        xarm_path, mount_xyz=mount_xyz, mount_rpy=mount_rpy, timestep=timestep
    )
    mjcf = mjcf.replace("{HAND}", hand_bodies)
    mjcf = mjcf.replace("</asset>", hand_assets + "</asset>")
    mjcf = mjcf.replace("</mujoco>", hand_extras + "</mujoco>")

    # The hand's own actuators live in its MJCF; merge them too
    hand_root = ET.fromstring(hand_mjcf)
    hand_act = hand_root.find("actuator")
    if hand_act is not None:
        act_xml = "".join(ET.tostring(c, encoding="unicode") for c in hand_act)
        mjcf = mjcf.replace("</actuator>", act_xml + "</actuator>")

    if save_xml_to:
        Path(save_xml_to).write_text(mjcf)
    return mujoco.MjModel.from_xml_string(mjcf), mjcf
