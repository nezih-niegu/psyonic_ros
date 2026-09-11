"""Build a physics-capable Ability Hand model from the ah_urdf URDF.

The URDF alone is not simulatable: MuJoCo's URDF importer produces a model
with no actuators and drops the <mimic> tags (no equality constraints). That
is fine for a passive viewer that writes qpos directly, but useless as a plant.

This module compiles the URDF, dumps the resulting MJCF, and injects:

  * an <equality><joint> constraint per URDF <mimic> tag, so q2 tracks q1
    the way the real linkage does
  * a <position> actuator on each of the 6 driven joints, with ctrlrange
    taken from the URDF joint limits and forcerange from the URDF effort
  * damping / armature on the driven joints so the sim is stable at 1 kHz

Driven joints (6 DOF, matching the hardware):
    index_q1, middle_q1, ring_q1, pinky_q1, thumb_q1, thumb_q2

index_q2 .. pinky_q2 are NOT actuated - they follow via the equality
constraint, exactly like the real mechanism.
"""

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco

# The 6 actuated joints, in the hardware order the driver uses:
# [index, middle, ring, pinky, thumb_flexor, thumb_rotator]
# Note thumb_q2 is the flexor and thumb_q1 the rotator, matching ah_node.
DRIVEN_JOINTS = [
    "index_q1",
    "middle_q1",
    "ring_q1",
    "pinky_q1",
    "thumb_q2",  # thumb flexor
    "thumb_q1",  # thumb rotator
]


def _parse_mimic(urdf_text):
    """child -> (parent, multiplier, offset) from URDF <mimic> tags."""
    mimics = {}
    for block in re.findall(r"<joint\b.*?</joint>", urdf_text, re.DOTALL):
        name = re.search(r'<joint[^>]*\bname="([^"]+)"', block)
        mim = re.search(r"<mimic\b([^>]*)>", block)
        if not name or not mim:
            continue
        parent = re.search(r'joint="([^"]+)"', mim.group(1))
        if not parent:
            continue
        mult = re.search(r'multiplier="([^"]+)"', mim.group(1))
        off = re.search(r'offset="([^"]+)"', mim.group(1))
        mimics[name.group(1)] = (
            parent.group(1),
            float(mult.group(1)) if mult else 1.0,
            float(off.group(1)) if off else 0.0,
        )
    return mimics


def _urdf_efforts(urdf_text):
    """joint name -> effort limit, for actuator forcerange."""
    efforts = {}
    for block in re.findall(r"<joint\b.*?</joint>", urdf_text, re.DOTALL):
        name = re.search(r'<joint[^>]*\bname="([^"]+)"', block)
        eff = re.search(r'<limit[^>]*\beffort="([^"]+)"', block)
        if name and eff:
            efforts[name.group(1)] = float(eff.group(1))
    return efforts


def _prepare_urdf(urdf_path):
    urdf_path = Path(urdf_path)
    raw = urdf_path.read_text()
    text = re.sub(r"package://ah_urdf/models/", "", raw)
    meshdir = (urdf_path.parent / ".." / "models").resolve()
    block = (
        f'<mujoco><compiler meshdir="{meshdir}" balanceinertia="true" '
        'discardvisual="false" fusestatic="false"/></mujoco>'
    )
    text = re.sub(r"(<robot[^>]*>)", r"\1\n" + block, text, count=1)
    return raw, text


def build_physics_model(
    urdf_path,
    timestep=0.001,
    kp=100.0,
    kv_ratio=0.01,
    damping=0.004,
    armature=1e-5,
    save_xml_to=None,
):
    """Return (MjModel, mjcf_string) with actuators and mimic constraints."""
    raw_urdf, patched_urdf = _prepare_urdf(urdf_path)
    base = mujoco.MjModel.from_xml_string(patched_urdf)

    tmp_xml = "/tmp/_ah_base_%d.xml" % id(base)
    mujoco.mj_saveLastXML(tmp_xml, base)
    root = ET.parse(tmp_xml).getroot()

    mimics = _parse_mimic(raw_urdf)
    efforts = _urdf_efforts(raw_urdf)

    # --- simulation options -------------------------------------------------
    opt = root.find("option")
    if opt is None:
        opt = ET.SubElement(root, "option")
    opt.set("timestep", str(timestep))
    opt.set("integrator", "implicitfast")
    opt.set("cone", "elliptic")     # better behaved friction for grasping

    # --- joint damping / armature on driven joints --------------------------
    joint_els = {j.get("name"): j for j in root.iter("joint") if j.get("name")}
    for name in DRIVEN_JOINTS:
        j = joint_els.get(name)
        if j is not None:
            j.set("damping", str(damping))
            j.set("armature", str(armature))
    # Mimic joints need a little damping too or the constraint rings
    for child in mimics:
        j = joint_els.get(child)
        if j is not None:
            j.set("damping", str(damping * 0.5))
            j.set("armature", str(armature))

    # --- equality constraints replacing URDF <mimic> ------------------------
    # MuJoCo: joint1 = polycoef[0] + polycoef[1]*joint2 + ...
    if mimics:
        eq = root.find("equality")
        if eq is None:
            eq = ET.SubElement(root, "equality")
        for child, (parent, mult, offset) in mimics.items():
            if child not in joint_els or parent not in joint_els:
                continue
            ET.SubElement(
                eq,
                "joint",
                {
                    "name": f"mimic_{child}",
                    "joint1": child,
                    "joint2": parent,
                    "polycoef": f"{offset} {mult} 0 0 0",
                    "solimp": "0.99 0.9999 0.0001",
                    "solref": "0.002 1",
                },
            )

    # --- position actuators on the driven joints ----------------------------
    act = root.find("actuator")
    if act is None:
        act = ET.SubElement(root, "actuator")
    for name in DRIVEN_JOINTS:
        j = joint_els.get(name)
        if j is None:
            continue
        rng = j.get("range")
        attrs = {
            "name": f"act_{name}",
            "joint": name,
            "kp": str(kp),
            # kv/kp IS the closed-loop time constant, independent of kp.
            # 0.06 gives a 60 ms lag on every joint; 0.01 gives 10 ms.
            "kv": str(kp * kv_ratio),
        }
        if rng:
            attrs["ctrlrange"] = rng
        eff = efforts.get(name)
        if eff:
            attrs["forcerange"] = f"{-abs(eff)} {abs(eff)}"
        ET.SubElement(act, "position", attrs)

    # --- contact exclusions -------------------------------------------------
    # The URDF's collision hulls overlap at every joint (the proximal segment
    # hull intersects the palm hull, and each segment intersects the next).
    # MuJoCo does not auto-exclude bodies connected by a joint, so without
    # these the fingers jam against the palm at ~10 degrees and never close.
    body_parent = {}

    def _walk(el, parent_name):
        for b in el.findall("body"):
            name = b.get("name")
            if name:
                body_parent[name] = parent_name
                _walk(b, name)

    world = root.find("worldbody")
    if world is not None:
        _walk(world, None)

    contact_el = root.find("contact")
    if contact_el is None:
        contact_el = ET.SubElement(root, "contact")
    for child, parent in body_parent.items():
        if parent:
            ET.SubElement(
                contact_el,
                "exclude",
                {"name": f"excl_{parent}_{child}", "body1": parent, "body2": child},
            )

    mjcf = ET.tostring(root, encoding="unicode")
    if save_xml_to:
        Path(save_xml_to).write_text(mjcf)

    model = mujoco.MjModel.from_xml_string(mjcf)
    return model, mjcf


def actuator_indices(model):
    """Map DRIVEN_JOINTS -> actuator id, in hardware order."""
    ids = []
    for name in DRIVEN_JOINTS:
        aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_{name}"
        )
        ids.append(aid)
    return ids
