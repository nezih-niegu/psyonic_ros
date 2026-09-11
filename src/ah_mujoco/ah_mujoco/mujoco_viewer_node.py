"""MuJoCo viewer node for the PSYONIC Ability Hand.

Drop-in replacement for the RViz visualization path. Instead of
joint_state_publisher -> robot_state_publisher -> RViz, this node:

    /joint_states_ah (sensor_msgs/JointState) --> MuJoCo passive viewer

The URDF from ah_urdf is loaded directly by MuJoCo (MuJoCo's compiler
understands URDF). At load time we resolve the ROS "package://" mesh URIs
to real paths and inject a <mujoco> compiler block, all in memory - the
URDF files on disk are never modified.

By default the node runs in kinematic "mirror" mode: incoming joint
positions are written straight into qpos and mj_forward() updates the
scene, exactly matching what RViz displayed. FSR touch feedback from
/ability_hand/<side>/feedback/touch is visualized by tinting the
fingertip geoms.
"""

import re
import threading
from pathlib import Path

import mujoco
import mujoco.viewer
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray


def parse_mimic_joints(urdf_text: str) -> dict:
    """Extract URDF <mimic> relations: child -> (parent, multiplier, offset).

    MuJoCo's URDF parser ignores <mimic> (the compiled model has no equality
    constraints), so the viewer has to enforce these itself. robot_state_publisher
    did this in the RViz path, which is why the discrepancy between the URDF's
    mimic relation and ah_node's published q2 never showed up before.
    """
    mimics = {}
    for joint_block in re.findall(r"<joint\b.*?</joint>", urdf_text, re.DOTALL):
        name_m = re.search(r'<joint[^>]*\bname="([^"]+)"', joint_block)
        mimic_m = re.search(r"<mimic\b([^>]*)>", joint_block)
        if not name_m or not mimic_m:
            continue
        attrs = mimic_m.group(1)
        parent_m = re.search(r'joint="([^"]+)"', attrs)
        if not parent_m:
            continue
        mult_m = re.search(r'multiplier="([^"]+)"', attrs)
        off_m = re.search(r'offset="([^"]+)"', attrs)
        mimics[name_m.group(1)] = (
            parent_m.group(1),
            float(mult_m.group(1)) if mult_m else 1.0,
            float(off_m.group(1)) if off_m else 0.0,
        )
    return mimics


def _add_skybox(model, rgb=(0.071, 0.078, 0.094)):
    """Recompile a model with a flat skybox texture.

    The 3D background colour comes from the skybox; without one MuJoCo clears
    the viewport to black, which leaves a hole in a light-themed UI. The URDF
    <mujoco> extension ignores <asset>, so the model is dumped to MJCF, the
    texture injected, and the result recompiled.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as fh:
        tmp = fh.name
    mujoco.mj_saveLastXML(tmp, model)
    xml = Path(tmp).read_text()
    tex = (
        '<asset><texture name="ah_sky" type="skybox" builtin="flat" '
        f'rgb1="{rgb[0]} {rgb[1]} {rgb[2]}" width="8" height="8"/></asset>'
    )
    if "<asset>" in xml:
        xml = xml.replace(
            "<asset>",
            '<asset><texture name="ah_sky" type="skybox" builtin="flat" '
            f'rgb1="{rgb[0]} {rgb[1]} {rgb[2]}" width="8" height="8"/>',
            1,
        )
    else:
        xml = re.sub(r"(<mujoco[^>]*>)", r"\1\n" + tex, xml, count=1)
    return mujoco.MjModel.from_xml_string(xml)


def load_ability_hand_model(urdf_path: str, skybox: bool = False):
    """Compile an ah_urdf URDF into a MuJoCo model.

    Resolves package:// mesh URIs relative to the URDF location and
    injects the <mujoco> extension block so visual meshes are kept.

    Returns (model, mimic_map).
    """
    urdf_path = Path(urdf_path)
    raw = urdf_path.read_text()

    # package://ah_urdf/models/foo.STL -> foo.STL, resolved via meshdir
    urdf = re.sub(r"package://ah_urdf/models/", "", raw)
    meshdir = (urdf_path.parent / ".." / "models").resolve()

    mujoco_block = (
        f'<mujoco><compiler meshdir="{meshdir}" balanceinertia="true" '
        'discardvisual="false" fusestatic="false"/></mujoco>'
    )
    urdf = re.sub(r"(<robot[^>]*>)", r"\1\n" + mujoco_block, urdf, count=1)

    model = mujoco.MjModel.from_xml_string(urdf)
    if skybox:
        model = _add_skybox(model)
    return model, parse_mimic_joints(raw)


class MujocoViewerNode(Node):
    # Same ordering the ah_node uses when publishing /joint_states_ah
    JOINT_NAMES = [
        "index_q1", "index_q2",
        "middle_q1", "middle_q2",
        "ring_q1", "ring_q2",
        "pinky_q1", "pinky_q2",
        "thumb_q1", "thumb_q2",
    ]

    # Digits in the order the FSR array reports them (6 sensors each).
    # Touch is shown on the distal visual geoms, e.g. 'index_mesh_2'.
    FINGERTIP_GEOM_HINTS = ["index", "middle", "ring", "pinky", "thumb"]

    # If no /joint_states_ah arrives within this window, follow targets instead
    JS_TIMEOUT_S = 0.5

    def __init__(self):
        super().__init__("mujoco_viewer_node")
        self.declare_parameter("hand_side", "Right")
        self.declare_parameter("hand_size", "Large")
        self.declare_parameter("model_path", "")
        self.declare_parameter("render_hz", 60.0)
        self.declare_parameter("show_touch", True)
        self.declare_parameter("follow_targets", True)
        # Recompute mimic joints from the URDF instead of trusting incoming q2.
        # ah_node publishes q2 = 1.0585*q1 + 0.7235, but the URDF declares
        # q2 = 0.814*q1. RViz used the URDF value; match that behavior.
        self.declare_parameter("enforce_mimic", True)
        # "touch": tint by FSR feedback. "msj": tint by movement smoothness
        # (green on the MSJ reference, through yellow/orange to red).
        # "none": leave the model colours alone.
        self.declare_parameter("color_mode", "touch")
        self.declare_parameter("msj_reference", 524.0)
        self.declare_parameter("msj_spread", 0.0)
        self.declare_parameter("msj_direction", "both")
        # Load the reference from the saved teleop calibration when present,
        # so the viewer and the teleop preview agree on what "good" means.
        self.declare_parameter("calibration_file", "")

        side = (
            self.get_parameter("hand_side").get_parameter_value().string_value.lower()
        )
        size = (
            self.get_parameter("hand_size").get_parameter_value().string_value.lower()
        )
        model_path = (
            self.get_parameter("model_path").get_parameter_value().string_value
        )
        self.render_hz = (
            self.get_parameter("render_hz").get_parameter_value().double_value
        )
        self.show_touch = (
            self.get_parameter("show_touch").get_parameter_value().bool_value
        )
        self.follow_targets = (
            self.get_parameter("follow_targets").get_parameter_value().bool_value
        )
        self.enforce_mimic = (
            self.get_parameter("enforce_mimic").get_parameter_value().bool_value
        )
        self.color_mode = (
            self.get_parameter("color_mode")
            .get_parameter_value()
            .string_value.lower()
        )
        self.msj_reference = (
            self.get_parameter("msj_reference").get_parameter_value().double_value
        )
        spread = self.get_parameter("msj_spread").get_parameter_value().double_value
        self.msj_spread = spread if spread > 0 else None
        self.msj_direction = (
            self.get_parameter("msj_direction")
            .get_parameter_value()
            .string_value.lower()
        )
        if self.color_mode == "msj":
            try:
                from ah_mujoco.calibration import load_calibration

                cal = load_calibration(
                    self.get_parameter("calibration_file")
                    .get_parameter_value()
                    .string_value
                    or None
                )
                if cal and cal.get("msj_reference") is not None:
                    self.msj_reference = float(cal["msj_reference"])
                    if cal.get("msj_spread"):
                        self.msj_spread = float(cal["msj_spread"])
            except Exception as exc:
                self.get_logger().warn(
                    f"Could not load calibration for MSJ reference: {exc}"
                )

        if not model_path:
            model_path = self._resolve_urdf(side, size)
        self.get_logger().info(f"Loading MuJoCo model from: {model_path}")

        self.model, self.mimic = load_ability_hand_model(model_path)
        self.data = mujoco.MjData(self.model)
        if self.enforce_mimic and self.mimic:
            self.get_logger().info(
                "Enforcing URDF mimic relations (as robot_state_publisher did): "
                + ", ".join(
                    f"{c}={m:g}*{p}{f'{o:+g}' if o else ''}"
                    for c, (p, m, o) in self.mimic.items()
                )
            )

        # Map each incoming joint name to its qpos address once, up front
        self.qpos_addr = {}
        for name in self.JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self.qpos_addr[name] = self.model.jnt_qposadr[jid]
            else:
                self.get_logger().warn(f"Joint '{name}' not found in model")

        self._lock = threading.Lock()
        self._latest = {}
        self._touch = None
        self._last_js_time = None
        self._target_deg = None
        self._msj = None

        self.create_subscription(
            JointState, "/joint_states_ah", self._joint_state_cb, 10
        )
        if self.follow_targets:
            from ah_messages.msg import Digits

            self.create_subscription(
                Digits,
                f"/ability_hand/{side}/target/position",
                self._target_cb,
                10,
            )
            self.get_logger().info(
                f"Following /ability_hand/{side}/target/position when no "
                "joint states are present (virtual hand mode)."
            )
        if self.show_touch and self.color_mode == "touch":
            self.create_subscription(
                Float32MultiArray,
                f"/ability_hand/{side}/feedback/touch",
                self._touch_cb,
                10,
            )
        if self.color_mode == "msj":
            self.create_subscription(
                Float32MultiArray,
                f"/ability_hand/{side}/quality/msj",
                self._msj_cb,
                10,
            )
            self.get_logger().info(
                f"Colouring joints by MSJ (reference {self.msj_reference:.1f}"
                + (
                    f" +/- {self.msj_spread:.1f}, thresholds at 1/2/5 sigma"
                    if self.msj_spread
                    else ", thresholds at 50/100/250"
                )
                + "): green on the calibrated baseline, red as it diverges."
            )

        self._viewer_thread = threading.Thread(target=self._viewer_loop, daemon=True)
        self._viewer_thread.start()

    # ------------------------------------------------------------------ ROS

    def _resolve_urdf(self, side: str, size: str) -> str:
        filename = f"ability_hand_{side}_{size}.urdf"
        try:
            from ament_index_python.packages import get_package_share_directory

            share = Path(get_package_share_directory("ah_urdf"))
            candidate = share / "urdf" / filename
            if candidate.exists():
                return str(candidate)
        except Exception:
            pass
        # Fallback for running from the source tree without installing ah_urdf
        here = Path(__file__).resolve()
        for parent in here.parents:
            candidate = parent / "ah_urdf" / "urdf" / filename
            if candidate.exists():
                return str(candidate)
        raise FileNotFoundError(
            f"Could not locate {filename}; set the 'model_path' parameter."
        )

    def _joint_state_cb(self, msg: JointState):
        import time

        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                self._latest[name] = pos
            self._last_js_time = time.monotonic()

    def _target_cb(self, msg):
        """Cache target positions (degrees, HAND_JOINT_ORDER) from the driver API.

        Order matches what ah_node expects:
        [index, middle, ring, pinky, thumb_flexor, thumb_rotator]
        """
        with self._lock:
            self._target_deg = list(msg.data)

    def _targets_to_qpos(self, target_deg):
        """Convert 6 target degrees to the 10 URDF joint angles.

        Replicates ah_node.publish_joint_states exactly, including the
        q2 mimic relation and the thumb index swap.
        """
        from math import radians

        if len(target_deg) < 6:
            return {}

        out = {}
        for i, digit in enumerate(["index", "middle", "ring", "pinky"]):
            q1 = radians(target_deg[i])
            out[f"{digit}_q1"] = q1
            out[f"{digit}_q2"] = q1 * 1.05851325 + 0.72349796
        # ah_node maps the last two entries crossed over
        out["thumb_q1"] = radians(target_deg[-1])
        out["thumb_q2"] = radians(target_deg[-2])
        return out

    def _touch_cb(self, msg: Float32MultiArray):
        with self._lock:
            self._touch = list(msg.data)

    def _msj_cb(self, msg: Float32MultiArray):
        with self._lock:
            self._msj = list(msg.data)

    # --------------------------------------------------------------- Viewer

    def _viewer_loop(self):
        import time

        period = 1.0 / max(self.render_hz, 1.0)
        if self.color_mode == "msj":
            color_geoms = self._collect_digit_geoms()
        elif self.color_mode == "touch":
            color_geoms = self._collect_touch_geoms()
        else:
            color_geoms = []

        with mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=False, show_right_ui=False
        ) as viewer:
            # Frame the hand nicely on startup
            viewer.cam.distance = 0.45
            viewer.cam.elevation = -25
            viewer.cam.azimuth = 130

            while viewer.is_running() and rclpy.ok():
                t0 = time.perf_counter()

                with self._lock:
                    latest = dict(self._latest)
                    touch = list(self._touch) if self._touch else None
                    msj = list(self._msj) if self._msj else None
                    js_time = self._last_js_time
                    target_deg = (
                        list(self._target_deg) if self._target_deg else None
                    )

                # Real feedback wins; fall back to targets if it goes stale
                js_fresh = js_time is not None and (
                    time.monotonic() - js_time
                ) < self.JS_TIMEOUT_S
                if not js_fresh and target_deg is not None:
                    latest = self._targets_to_qpos(target_deg)

                for name, pos in latest.items():
                    addr = self.qpos_addr.get(name)
                    if addr is not None:
                        self.data.qpos[addr] = pos

                if self.enforce_mimic:
                    self._apply_mimic()

                mujoco.mj_forward(self.model, self.data)

                if self.color_mode == "msj" and msj is not None and color_geoms:
                    self._apply_msj_tint(msj, color_geoms)
                elif (
                    self.color_mode == "touch"
                    and touch is not None
                    and color_geoms
                ):
                    self._apply_touch_tint(touch, color_geoms)

                viewer.sync()

                dt = time.perf_counter() - t0
                if dt < period:
                    time.sleep(period - dt)

        # Viewer window closed -> shut the node down
        if rclpy.ok():
            self.get_logger().info("MuJoCo viewer closed, shutting down.")
            rclpy.try_shutdown()

    def _apply_mimic(self):
        """Overwrite mimic joints from their parent, per the URDF relation."""
        for child, (parent, mult, offset) in self.mimic.items():
            c_addr = self.qpos_addr.get(child)
            p_addr = self.qpos_addr.get(parent)
            if c_addr is None or p_addr is None:
                continue
            self.data.qpos[c_addr] = self.data.qpos[p_addr] * mult + offset

    def _collect_touch_geoms(self):
        """Group distal visual geoms by digit and snapshot base colors."""
        groups = [[] for _ in self.FINGERTIP_GEOM_HINTS]
        self._base_rgba = {}
        for gid in range(self.model.ngeom):
            name = (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            ).lower()
            if "coll" in name or "hull" in name:
                continue  # collision geoms stay untouched
            if not name.endswith("_2"):
                continue  # only distal segments carry the FSRs
            for i, hint in enumerate(self.FINGERTIP_GEOM_HINTS):
                if name.startswith(hint):
                    groups[i].append(gid)
                    self._base_rgba[gid] = self.model.geom_rgba[gid].copy()
                    break
        return groups

    def _collect_digit_geoms(self):
        """All visual geoms of each digit, for whole-finger colouring."""
        groups = [[] for _ in self.FINGERTIP_GEOM_HINTS]
        if not hasattr(self, "_base_rgba"):
            self._base_rgba = {}
        for gid in range(self.model.ngeom):
            name = (
                mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
            ).lower()
            if "coll" in name or "hull" in name:
                continue
            for i, hint in enumerate(self.FINGERTIP_GEOM_HINTS):
                if name.startswith(hint):
                    groups[i].append(gid)
                    self._base_rgba.setdefault(
                        gid, self.model.geom_rgba[gid].copy()
                    )
                    break
        return groups

    def _apply_msj_tint(self, msj, color_geoms):
        """Colour each digit by how far its MSJ sits from the reference."""
        from ah_mujoco.signal_quality import msj_color

        # msj arrives in hardware order [index, middle, ring, pinky,
        # thumb_flexor, thumb_rotator]; the geom groups are per digit, so the
        # thumb takes the worse of its two DOF.
        per_digit = list(msj[:4]) + [max(msj[4:6], key=abs)] if len(msj) >= 6 \
            else list(msj)
        for i, geoms in enumerate(color_geoms):
            if i >= len(per_digit):
                break
            r, g, b = msj_color(
                per_digit[i],
                self.msj_reference,
                spread=self.msj_spread,
                direction=self.msj_direction,
            )
            for gid in geoms:
                base = self._base_rgba[gid]
                self.model.geom_rgba[gid] = [r, g, b, base[3]]

    def _apply_touch_tint(self, touch, touch_geoms):
        """Tint each digit red proportionally to its max FSR reading."""
        n_per_digit = max(len(touch) // len(touch_geoms), 1)
        for i, geoms in enumerate(touch_geoms):
            digit_vals = touch[i * n_per_digit:(i + 1) * n_per_digit]
            level = min(max(digit_vals, default=0.0) / 100.0, 1.0)
            for gid in geoms:
                base = self._base_rgba[gid]  # always tint from the original color
                self.model.geom_rgba[gid] = [
                    base[0] * (1 - level) + 1.0 * level,
                    base[1] * (1 - level),
                    base[2] * (1 - level),
                    base[3],
                ]


def main(args=None):
    rclpy.init(args=args)
    node = MujocoViewerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
