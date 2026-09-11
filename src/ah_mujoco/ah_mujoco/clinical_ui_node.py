"""GLFW window hosting the unified clinical UI.

Subscribes to everything the pipeline already publishes and draws it in one
window, replacing the separate OpenCV preview and MuJoCo passive viewer.

    /joint_states_ah                      -> 3D hand pose
    /ability_hand/<side>/camera/image_raw -> tracking view
    /ability_hand/<side>/quality/msj      -> per-joint colour + gauge
    /ability_hand/<side>/quality/mae      -> acquisition noise readout
    /ability_hand/<side>/target/position  -> joint command bars

Mouse: drag to orbit, right-drag to pan, scroll to zoom.
Keys:  o / c / b run the calibration steps (relayed to teleop, which owns the
       session), k restarts calibration, s saves, t toggles dark/light,
       space toggles the arm clutch, r resets the camera, q quits.
"""

import math
import threading

import glfw
import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, Float32MultiArray, String

from ah_messages.msg import Digits

from ah_mujoco.clinical_ui import VIEWER_JOINTS, ClinicalUI
from ah_mujoco.mujoco_viewer_node import load_ability_hand_model


class ClinicalUINode(Node):
    def __init__(self):
        super().__init__("clinical_ui_node")
        self.declare_parameter("hand_side", "Right")
        self.declare_parameter("hand_size", "Large")
        self.declare_parameter("model_path", "")
        self.declare_parameter("window_width", 1400)
        self.declare_parameter("window_height", 800)
        self.declare_parameter("msj_reference", 524.0)
        self.declare_parameter("msj_spread", 0.0)
        self.declare_parameter("calibration_file", "")
        self.declare_parameter("enforce_mimic", True)
        self.declare_parameter("render_hz", 60.0)
        self.declare_parameter("theme", "dark")   # "dark" or "light"
        # Arm mode: show the Lite 6 with the hand mounted, driven by
        # /xarm/target_joints, instead of the hand on its own.
        self.declare_parameter("arm", False)
        self.declare_parameter("xarm_description_path", "")
        self.declare_parameter("mount_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("mount_rpy", [0.0, 0.0, 0.0])
        # Real robot feedback. The view should show where the arm IS, not
        # only where teleop wants it - otherwise the safe-pose move is
        # invisible and the model looks stuck at zero.
        self.declare_parameter("arm_state_topic", "/ufactory/joint_states")

        g = self.get_parameter
        side = g("hand_side").get_parameter_value().string_value.lower()
        size = g("hand_size").get_parameter_value().string_value.lower()
        model_path = g("model_path").get_parameter_value().string_value
        self.W = g("window_width").get_parameter_value().integer_value
        self.H = g("window_height").get_parameter_value().integer_value
        self.render_hz = g("render_hz").get_parameter_value().double_value
        self.enforce_mimic = g("enforce_mimic").get_parameter_value().bool_value

        if not model_path:
            model_path = self._resolve_urdf(side, size)
        self.arm_mode = g("arm").get_parameter_value().bool_value
        if self.arm_mode:
            from ah_mujoco.arm_model import ARM_JOINT_NAMES, build_combined_model

            self.arm_joint_names = ARM_JOINT_NAMES
            self.model, _ = build_combined_model(
                model_path,
                xarm_description_path=g("xarm_description_path")
                .get_parameter_value().string_value or None,
                mount_xyz=list(
                    g("mount_xyz").get_parameter_value().double_array_value
                ) or (0.0, 0.0, 0.0),
                mount_rpy=list(
                    g("mount_rpy").get_parameter_value().double_array_value
                ) or (0.0, 0.0, 0.0),
            )
            # the combined model already carries mimic equalities
            self.mimic = {}
            self.get_logger().info("arm mode: Lite 6 + Ability Hand")
        else:
            self.arm_joint_names = []
            self.model, self.mimic = load_ability_hand_model(
                model_path, skybox=True
            )
        # The offscreen buffer must be at least the window size
        self.model.vis.global_.offwidth = max(self.W, 1920)
        self.model.vis.global_.offheight = max(self.H, 1080)

        self.ui = ClinicalUI(
            self.model,
            theme=g("theme").get_parameter_value().string_value,
        )
        self.qpos_addr = {}
        for name in list(VIEWER_JOINTS) + list(self.arm_joint_names):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self.qpos_addr[name] = self.model.jnt_qposadr[jid]

        if self.arm_mode:
            self.ui.cam.distance = 1.2
            self.ui.cam.elevation = -18
            self.ui.cam.lookat[:] = [0.0, 0.1, 0.35]

        self._load_reference(side)

        self._have_arm_state = False
        # Same biomechanical metric as the hand, applied to the arm's joint
        # trajectories. The dimensionless form is amplitude and unit free, so
        # the reference calibrated on the hand is meaningful here too:
        # measured 13.9 for a finger and 15.4 for the shoulder on the same
        # smooth profile.
        from ah_mujoco.signal_quality import MSJMonitor

        self._arm_msj_monitor = MSJMonitor(6, 1.0 / 60.0, window=90)
        self._lock = threading.Lock()
        ns = f"/ability_hand/{side}"
        self.create_subscription(JointState, "/joint_states_ah", self._js_cb, 10)
        if self.arm_mode:
            self.create_subscription(
                JointState, "/xarm/target_joints", self._arm_js_cb, 10
            )
            self.create_subscription(
                String, "/xarm/bridge_status", self._bridge_cb, 10
            )
            state_topic = (
                g("arm_state_topic").get_parameter_value().string_value
            )
            if state_topic:
                self.create_subscription(
                    JointState, state_topic, self._arm_state_cb, 10
                )
                self.get_logger().info(f"mirroring arm state from {state_topic}")
        self.create_subscription(
            Bool, f"{ns}/clutch", self._clutch_state_cb, 10
        )
        self.create_subscription(Image, f"{ns}/camera/image_raw", self._img_cb, 1)
        self.create_subscription(
            Float32MultiArray, f"{ns}/quality/msj", self._msj_cb, 10
        )
        self.create_subscription(
            Float32MultiArray, f"{ns}/quality/mae", self._mae_cb, 10
        )
        self.create_subscription(
            Float32MultiArray, f"{ns}/quality/latency_ms", self._lat_cb, 10
        )
        self.create_subscription(
            Float32MultiArray, f"{ns}/quality/reference", self._ref_cb, 10
        )
        self.create_subscription(
            Digits, f"{ns}/target/position", self._target_cb, 10
        )
        self.create_subscription(
            String, f"{ns}/calibration/status", self._calib_cb, 10
        )

        self.pub_calib_cmd = self.create_publisher(
            String, f"{ns}/calibration/command", 10
        )
        self._Bool = Bool
        self.pub_clutch = self.create_publisher(Bool, f"{ns}/clutch", 10)
        self._clutch = False

        self._render_thread = threading.Thread(target=self._loop, daemon=True)
        self._render_thread.start()
        self.get_logger().info(
            "Clinical UI running.  o/c/b calibrate, k recalibrates, "
            "s saves, t toggles theme, r resets view, q quits."
        )

    # ------------------------------------------------------------------ setup

    def _resolve_urdf(self, side, size):
        from pathlib import Path

        fn = f"ability_hand_{side}_{size}.urdf"
        try:
            from ament_index_python.packages import get_package_share_directory

            p = Path(get_package_share_directory("ah_urdf")) / "urdf" / fn
            if p.exists():
                return str(p)
        except Exception:
            pass
        for parent in Path(__file__).resolve().parents:
            p = parent / "ah_urdf" / "urdf" / fn
            if p.exists():
                return str(p)
        raise FileNotFoundError(f"Could not find {fn}")

    def _load_reference(self, side):
        g = self.get_parameter
        self.ui.msj_reference = g("msj_reference").get_parameter_value().double_value
        spread = g("msj_spread").get_parameter_value().double_value
        self.ui.msj_spread = spread if spread > 0 else None
        try:
            from ah_mujoco.calibration import load_calibration

            cal = load_calibration(
                g("calibration_file").get_parameter_value().string_value or None
            )
            if cal and cal.get("msj_reference") is not None:
                self.ui.msj_reference = float(cal["msj_reference"])
                if cal.get("msj_spread"):
                    self.ui.msj_spread = float(cal["msj_spread"])
                self.get_logger().info(
                    f"MSJ reference {self.ui.msj_reference:.1f} from calibration"
                )
        except Exception as exc:
            self.get_logger().warn(f"No calibration reference: {exc}")

    # -------------------------------------------------------------- callbacks

    def _js_cb(self, msg: JointState):
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                addr = self.qpos_addr.get(name)
                if addr is not None:
                    self.ui.data.qpos[addr] = pos
            if self.enforce_mimic:
                for child, (parent, mult, off) in self.mimic.items():
                    ca, pa = self.qpos_addr.get(child), self.qpos_addr.get(parent)
                    if ca is not None and pa is not None:
                        self.ui.data.qpos[ca] = self.ui.data.qpos[pa] * mult + off
            self.ui.connected = True

    def _arm_state_cb(self, msg: JointState):
        """Actual robot joints. Takes priority over the commanded target."""
        with self._lock:
            for name, pos in zip(msg.name, msg.position):
                addr = self.qpos_addr.get(f"arm_{name}") or self.qpos_addr.get(name)
                if addr is not None:
                    self.ui.data.qpos[addr] = pos
            self._have_arm_state = True
            self._update_arm_joints()

    def _do_command(self, cmd, label=""):
        """One place for both button clicks and keystrokes."""
        if cmd == "theme":
            name = self.ui.toggle_theme(self._con)
            self.get_logger().info(f"theme: {name}")
        elif cmd == "clutch":
            self._clutch = not self._clutch
            m = self._Bool()
            m.data = self._clutch
            self.pub_clutch.publish(m)
            self.get_logger().info(
                f"clutch {'ENGAGED' if self._clutch else 'released'}"
            )
        else:
            m = String()
            m.data = cmd
            self.pub_calib_cmd.publish(m)
            self.get_logger().info(f"sent '{cmd}' {label}")

    def _update_arm_joints(self):
        """Cache the arm joints and score their smoothness the same way the
        hand's are scored."""
        q = [
            self.ui.data.qpos[self.qpos_addr[f"arm_joint{i+1}"]]
            for i in range(6)
            if f"arm_joint{i+1}" in self.qpos_addr
        ]
        if len(q) != 6:
            self.ui.arm_joints = None
            return
        self.ui.arm_joints = q
        self.ui.arm_msj = self._arm_msj_monitor.update(np.degrees(q))

    def _bridge_cb(self, msg: String):
        with self._lock:
            self.ui.arm_status = msg.data or ""

    def _clutch_state_cb(self, msg):
        with self._lock:
            self.ui.clutch = bool(msg.data)

    def _arm_js_cb(self, msg: JointState):
        """Commanded joints from arm_teleop, used when the robot is absent."""
        with self._lock:
            self.ui.arm_active = True
            if getattr(self, "_have_arm_state", False):
                return          # real feedback wins
            for name, pos in zip(msg.name, msg.position):
                # arm_teleop publishes joint1..joint6; the model calls them
                # arm_joint1..arm_joint6
                addr = self.qpos_addr.get(f"arm_{name}") or self.qpos_addr.get(name)
                if addr is not None:
                    self.ui.data.qpos[addr] = pos
            self._update_arm_joints()

    def _img_cb(self, msg: Image):
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            arr = arr.reshape(msg.height, msg.width, 3)
            if msg.encoding == "bgr8":
                arr = arr[:, :, ::-1]
            # OpenGL origin is bottom-left
            with self._lock:
                self.ui.frame = np.ascontiguousarray(np.flipud(arr))
        except Exception as exc:
            self.get_logger().warn(
                f"bad image message: {exc}", throttle_duration_sec=5.0
            )

    def _msj_cb(self, msg):
        with self._lock:
            d = list(msg.data)
            if len(d) >= 6:
                self.ui.msj = np.array(d[:6])

    def _mae_cb(self, msg):
        with self._lock:
            if msg.data:
                self.ui.mae_mm = float(np.mean(msg.data)) * 1000.0

    def _ref_cb(self, msg):
        """Live MSJ reference from teleop, so a calibration done during the
        session takes effect immediately instead of at the next restart."""
        d = list(msg.data)
        if len(d) >= 2 and d[0] > 0:
            with self._lock:
                if abs(self.ui.msj_reference - d[0]) > 1e-6:
                    self.get_logger().info(
                        f"MSJ reference updated: {d[0]:.1f} +/- {d[1]:.1f}"
                    )
                self.ui.msj_reference = float(d[0])
                self.ui.msj_spread = float(d[1]) if d[1] > 0 else None

    def _lat_cb(self, msg):
        with self._lock:
            if msg.data:
                self.ui.latency_ms = float(msg.data[0])

    def _target_cb(self, msg):
        with self._lock:
            d = list(msg.data)
            if len(d) >= 6:
                self.ui.targets = np.array(d[:6])

    def _calib_cb(self, msg: String):
        with self._lock:
            text = msg.data or ""
            self.ui.calibrating = bool(text)
            self.ui.calib_prompt = text

    # ------------------------------------------------------------------- loop

    def _loop(self):
        import time

        if not glfw.init():
            self.get_logger().error("glfw init failed - is DISPLAY set?")
            rclpy.try_shutdown()
            return

        glfw.window_hint(glfw.SAMPLES, 4)
        window = glfw.create_window(
            self.W, self.H, "Ability Hand  ·  Teleop", None, None
        )
        if not window:
            glfw.terminate()
            self.get_logger().error("could not create window")
            rclpy.try_shutdown()
            return

        glfw.make_context_current(window)
        glfw.swap_interval(1)
        con = mujoco.MjrContext(
            self.model, mujoco.mjtFontScale.mjFONTSCALE_150
        )
        mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW, con)
        self._con = con
        self.ui._apply_theme(con)      # upload the themed skybox

        self._install_input(window)

        period = 1.0 / max(self.render_hz, 1.0)
        last = time.perf_counter()
        frames, fps_t0 = 0, time.perf_counter()

        while not glfw.window_should_close(window) and rclpy.ok():
            t0 = time.perf_counter()
            w, h = glfw.get_framebuffer_size(window)

            with self._lock:
                mujoco.mj_forward(self.model, self.ui.data)
                self.ui.draw(con, w, h)

            glfw.swap_buffers(window)
            glfw.poll_events()

            frames += 1
            if t0 - fps_t0 >= 0.5:
                self.ui.fps = frames / (t0 - fps_t0)
                frames, fps_t0 = 0, t0

            dt = time.perf_counter() - t0
            if dt < period:
                time.sleep(period - dt)
            last = t0

        glfw.terminate()
        rclpy.try_shutdown()

    def _install_input(self, window):
        state = {"lx": 0.0, "ly": 0.0, "left": False, "right": False}

        def hit_button(win):
            """Which button is under the cursor, if any.

            GLFW cursor coords are top-left origin; the UI draws bottom-left,
            so y is flipped before testing.
            """
            x, y = glfw.get_cursor_pos(win)
            fw, fh = glfw.get_framebuffer_size(win)
            ww, wh = glfw.get_window_size(win)
            sx = fw / max(ww, 1)
            sy = fh / max(wh, 1)
            px, py = x * sx, fh - y * sy
            for bx, by, bw, bh, cmd, label in self.ui.buttons:
                if bx <= px <= bx + bw and by <= py <= by + bh:
                    return cmd, label
            return None, None

        def on_mouse_button(win, button, act, mods):
            if button == glfw.MOUSE_BUTTON_LEFT and act == glfw.PRESS:
                cmd, label = hit_button(win)
                if cmd is not None:
                    self._do_command(cmd, label)
                    return          # a button press is not a camera drag
            state["left"] = (
                glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS
            )
            state["right"] = (
                glfw.get_mouse_button(win, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS
            )
            state["lx"], state["ly"] = glfw.get_cursor_pos(win)

        def on_cursor(win, x, y):
            dx, dy = x - state["lx"], y - state["ly"]
            state["lx"], state["ly"] = x, y
            cam = self.ui.cam
            if state["left"]:
                cam.azimuth -= dx * 0.3
                cam.elevation = max(min(cam.elevation - dy * 0.3, 89), -89)
            elif state["right"]:
                cam.lookat[0] += dx * 0.0004
                cam.lookat[2] += dy * 0.0004

        def on_scroll(win, xo, yo):
            self.ui.cam.distance = max(
                0.05, min(self.ui.cam.distance * (1.0 - 0.08 * yo), 3.0)
            )

        def on_key(win, key, scancode, act, mods):
            if act != glfw.PRESS:
                return
            if key == glfw.KEY_Q or key == glfw.KEY_ESCAPE:
                glfw.set_window_should_close(win, True)
            elif key in (glfw.KEY_O, glfw.KEY_C, glfw.KEY_B, glfw.KEY_L,
                         glfw.KEY_K, glfw.KEY_S):
                self._do_command(chr(key).lower())
            elif key == glfw.KEY_SPACE:
                self._do_command("clutch")
            elif key == glfw.KEY_T:
                self._do_command("theme")
            elif key == glfw.KEY_R:
                c = self.ui.cam
                c.distance, c.elevation, c.azimuth = 0.30, -14, 152
                c.lookat[:] = [0.0, 0.0, 0.07]

        glfw.set_mouse_button_callback(window, on_mouse_button)
        self._glfw = glfw
        glfw.set_cursor_pos_callback(window, on_cursor)
        glfw.set_scroll_callback(window, on_scroll)
        glfw.set_key_callback(window, on_key)


def main(args=None):
    rclpy.init(args=args)
    node = ClinicalUINode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
