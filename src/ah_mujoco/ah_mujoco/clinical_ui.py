"""Single-window clinical UI: camera tracking and MuJoCo hand side by side.

Replaces the two separate windows (an OpenCV preview and the MuJoCo passive
viewer) with one OpenGL surface. Everything is drawn through MuJoCo's own
renderer primitives - mjr_render for the 3D hand, mjr_drawPixels for the camera
feed, mjr_rectangle and mjr_text for the panels - so there is no PyOpenGL or Qt
dependency beyond what MuJoCo already ships.

    +--------------------------------------------------------------+
    |  ABILITY HAND / TELEOP            status pills          time  |
    +----------------------------+---------------------------------+
    |                            |                                 |
    |      camera + landmarks    |        MuJoCo hand              |
    |                            |        (commanded state)        |
    +----------------------------+---------------------------------+
    |  per-joint bars, MSJ colour coded    |  signal quality panel  |
    +--------------------------------------------------------------+

The layout is resolution independent: every panel is computed from the current
framebuffer size, so the window can be resized freely.
"""

import math
import threading
import time

import mujoco
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32MultiArray

from ah_mujoco.mujoco_viewer_node import load_ability_hand_model
from ah_mujoco.signal_quality import msj_band, msj_color

# ---------------------------------------------------------------- themes
# Minimalist: two surface tones, one hairline, two text weights, and a single
# accent. All status meaning is carried by the MSJ gradient, so the chrome
# stays neutral and out of the way.


class Theme:
    def __init__(self, name, bg, surface, hairline, text, text_dim, accent,
                 sim_bg, marker):
        self.name = name
        self.bg = bg                # window background
        self.surface = surface      # panel fill, one step from bg
        self.hairline = hairline    # 1 px rules; the only border used
        self.text = text
        self.text_dim = text_dim
        self.accent = accent
        self.sim_bg = sim_bg        # MuJoCo scene background
        self.marker = marker        # gauge needle


DARK = Theme(
    name="dark",
    bg=(0.055, 0.063, 0.078),
    surface=(0.086, 0.098, 0.118),
    hairline=(0.169, 0.192, 0.227),
    text=(0.918, 0.933, 0.949),
    text_dim=(0.451, 0.486, 0.541),
    accent=(0.361, 0.722, 0.949),
    sim_bg=(0.071, 0.078, 0.094),
    marker=(1.0, 1.0, 1.0),
)

LIGHT = Theme(
    name="light",
    bg=(0.976, 0.980, 0.984),
    surface=(1.000, 1.000, 1.000),
    hairline=(0.867, 0.882, 0.898),
    text=(0.086, 0.106, 0.133),
    text_dim=(0.420, 0.451, 0.502),
    accent=(0.086, 0.396, 0.702),
    sim_bg=(0.937, 0.945, 0.953),
    marker=(0.086, 0.106, 0.133),
)

THEMES = {"dark": DARK, "light": LIGHT}

JOINT_LABELS = ["INDEX", "MIDDLE", "RING", "PINKY", "TH FLEX", "TH ROT"]
ARM_LABELS = ["J1 BASE", "J2 SHLDR", "J3 ELBOW", "J4 ROLL", "J5 PITCH",
              "J6 ROLL"]
# Joint limits in radians, for the bar fill. From lite6_robot_macro.xacro.
ARM_RANGES = [
    (-3.11, 3.11), (-2.62, 2.62), (-0.061, 3.11),
    (-3.11, 3.11), (-2.16, 2.16), (-3.11, 3.11),
]
VIEWER_JOINTS = [
    "index_q1", "index_q2", "middle_q1", "middle_q2",
    "ring_q1", "ring_q2", "pinky_q1", "pinky_q2",
    "thumb_q1", "thumb_q2",
]


def rect(l, b, w, h):
    r = mujoco.MjrRect(0, 0, 0, 0)
    r.left, r.bottom, r.width, r.height = int(l), int(b), int(w), int(h)
    return r


class ClinicalUI:
    """Pure drawing layer. Holds no ROS state, so it can be unit rendered."""

    def __init__(self, model, theme="dark"):
        self.model = model
        self.theme = THEMES.get(theme, DARK)
        self.data = mujoco.MjData(model)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()
        self.scene = mujoco.MjvScene(model, maxgeom=2000)
        mujoco.mjv_defaultCamera(self.cam)
        self.cam.distance = 0.30
        self.cam.elevation = -14
        self.cam.azimuth = 152
        self.cam.lookat[:] = [0.0, 0.0, 0.07]
        self._apply_theme()

        self.frame = None            # HxWx3 uint8 RGB, already flipped
        self.msj = np.zeros(6)
        self.mae_mm = 0.0
        self.targets = np.zeros(6)
        self.msj_reference = 524.0
        self.msj_spread = None
        self.calibrating = False
        self.calib_prompt = ""
        self.fps = 0.0
        self.latency_ms = 0.0
        self.connected = False
        self.clutch = False
        self.arm_status = ""      # from /xarm/bridge_status
        self.arm_active = False   # target joints arriving
        # Clickable regions, rebuilt each draw: (x, y, w, h, command, label).
        # The node hit-tests these on mouse press.
        self.buttons = []
        # Arm joint angles in radians, shown alongside the hand's DOFs so the
        # whole commanded system is visible in one place.
        self.arm_joints = None
        # Per-joint MSJ for the arm, same metric and same reference as the
        # hand's. None until enough samples have accumulated.
        self.arm_msj = None

    def set_theme(self, name, con=None):
        self.theme = THEMES.get(name, self.theme)
        self._apply_theme(con)

    def toggle_theme(self, con=None):
        self.set_theme(
            "light" if self.theme.name == "dark" else "dark", con
        )
        return self.theme.name

    def _apply_theme(self, con=None):
        """Recolour the scene background and lighting for the active theme.

        The 3D background is the skybox texture: MuJoCo clears the viewport to
        black otherwise, which leaves a hole in the light theme. The texture
        bytes are rewritten here and re-uploaded to the GL context.
        """
        th = self.theme
        light = th.name == "light"
        self.model.vis.headlight.ambient[:] = (
            [0.50, 0.50, 0.50] if light else [0.30, 0.30, 0.30]
        )
        self.model.vis.headlight.diffuse[:] = (
            [0.62, 0.62, 0.62] if light else [0.55, 0.55, 0.55]
        )
        self._paint_skybox(con)

    def _paint_skybox(self, con=None):
        tid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_TEXTURE, "ah_sky")
        if tid < 0:
            return
        c = np.clip(np.array(self.theme.sim_bg) * 255.0, 0, 255).astype(np.uint8)
        adr = int(self.model.tex_adr[tid])
        nch = int(getattr(self.model, "tex_nchannel", [3])[tid]) if hasattr(
            self.model, "tex_nchannel"
        ) else 3
        n = int(self.model.tex_width[tid]) * int(self.model.tex_height[tid])
        block = self.model.tex_data[adr: adr + n * nch].reshape(-1, nch)
        block[:, :3] = c
        if nch == 4:
            block[:, 3] = 255
        if con is not None:
            mujoco.mjr_uploadTexture(self.model, con, tid)

    # ------------------------------------------------------------- drawing

    def draw(self, con, W, H):
        th = self.theme
        header_h = int(H * 0.075)
        footer_h = int(H * 0.215)
        body_b = footer_h
        body_h = H - header_h - footer_h
        split = int(W * 0.42)

        mujoco.mjr_rectangle(rect(0, 0, W, H), *th.bg, 1)
        self._header(con, W, H, header_h)

        self.buttons = []
        if self.calibrating and self.calib_prompt:
            bh = int(H * 0.036)
            banner = rect(0, H - header_h - bh, W, bh)
            mujoco.mjr_rectangle(banner, *th.accent, 1)
            bg = th.bg if th.name == "dark" else (1, 1, 1)
            txt = "CALIBRATING   " + self.calib_prompt
            avail = int((W - 24) / self.CHAR_W)
            self._text_px(con, banner, txt[:avail], 12,
                          max((bh - self.CHAR_H) / 2, 1), bg)
            body_h -= bh

            bar_h = int(H * 0.062)
            self._button_bar(con, 0, H - header_h - bh - bar_h, W, bar_h,
                             self._calibration_buttons())
            body_h -= bar_h
        else:
            bar_h = int(H * 0.052)
            self._button_bar(con, 0, H - header_h - bar_h, W, bar_h,
                             [("k", "RECALIBRATE"), ("s", "SAVE"),
                              ("clutch", "CLUTCH"), ("theme", "THEME")],
                             subtle=True)
            body_h -= bar_h

        self._camera_panel(con, 0, body_b, split, body_h)
        self._sim_panel(con, split, body_b, W - split, body_h)
        self._footer(con, W, footer_h, split)

    def _calibration_buttons(self):
        """One primary action for the current step, plus a restart."""
        step = ""
        p = self.calib_prompt.upper()
        if "STEP 1" in p:
            step = ("o", "CAPTURE OPEN HAND")
        elif "STEP 2" in p:
            step = ("c", "CAPTURE FIST")
        elif "STEP 3" in p:
            step = ("b", "CAPTURE BASELINE")
        elif "STEP 4" in p:
            step = ("l", "MEASURE ARM")
        else:
            step = ("o", "CAPTURE")
        return [step, ("k", "RESTART")]

    def _button_bar(self, con, l, b, w, h, items, subtle=False):
        """Draw buttons and record their rects for hit testing."""
        th = self.theme
        if not items:
            return
        mujoco.mjr_rectangle(rect(l, b, w, h), *th.bg, 1)
        pad = int(w * 0.012)
        bw = min(int((w - pad * (len(items) + 1)) / len(items)),
                 int(w * 0.22))
        bh = int(h * 0.62)
        by = b + (h - bh) // 2
        x = l + pad

        for i, (cmd, label) in enumerate(items):
            primary = (i == 0) and not subtle
            fill = th.accent if primary else th.surface
            mujoco.mjr_rectangle(rect(x, by, bw, bh), *fill, 1)
            if not primary:
                mujoco.mjr_rectangle(rect(x, by, bw, 1), *th.hairline, 1)
                mujoco.mjr_rectangle(rect(x, by + bh - 1, bw, 1),
                                     *th.hairline, 1)
            txt_col = (th.bg if th.name == "dark" else (1, 1, 1)) \
                if primary else th.text
            r = rect(x, by, bw, bh)
            ty = max((bh - self.CHAR_H) / 2.0, 1.0)
            lab = label
            while self.text_width(lab) > bw - 16 and len(lab) > 3:
                lab = lab[:-1]
            self._text_px(con, r, lab, 10, ty, txt_col)
            self.buttons.append((x, by, bw, bh, cmd, label))
            x += bw + pad

    # --- header -----------------------------------------------------------

    def _header(self, con, W, H, h):
        th = self.theme
        top = H - h
        r = rect(0, top, W, h)
        mujoco.mjr_rectangle(r, *th.bg, 1)
        mujoco.mjr_rectangle(rect(0, top, W, 1), *th.hairline, 1)

        # Two rows, sized from the font rather than from fractions of h.
        row1 = h / 2.0 + (h / 2.0 - self.CHAR_H) / 2.0
        row2 = (h / 2.0 - self.CHAR_H) / 2.0

        state = "CALIBRATING" if self.calibrating else (
            "TRACKING" if self.connected else "NO HAND")
        state_col = th.accent if (self.connected or self.calibrating) \
            else th.text_dim
        clutch_txt = "CLUTCH ON" if self.clutch else "CLUTCH OFF"
        clutch_col = th.accent if self.clutch else th.text_dim
        arm_txt = "ARM " + (
            self.arm_status.upper() if self.arm_status
            else ("STREAMING" if self.arm_active else "IDLE")
        )

        # Right-aligned stacks, packed in pixels so they cannot collide.
        left1 = self._row_right(con, r, [
            (state, state_col),
            (clutch_txt, clutch_col),
            (f"{self.fps:.0f} FPS", th.text_dim),
        ], row1)
        left2 = self._row_right(con, r, [
            ("[SPACE]" if not self.clutch else "", th.text_dim),
            (arm_txt, th.text_dim),
            (f"{self.latency_ms:.0f} MS", th.text_dim),
        ], row2)

        # The title only goes in if there is room left for it.
        title = "ABILITY HAND"
        if self.text_width(title) + 32 < min(left1, left2):
            self._text_px(con, r, title, 16, row1, th.text)

        # --- camera -----------------------------------------------------------

    def _camera_panel(self, con, l, b, w, h):
        th = self.theme
        pad = int(min(w, h) * 0.05)
        inner = rect(l + pad, b + pad, w - 2 * pad, h - 2 * pad)
        mujoco.mjr_rectangle(inner, *th.surface, 1)

        label_h = 24
        if self.frame is not None:
            fh, fw = self.frame.shape[:2]
            avail_h = inner.height - label_h
            scale = min(inner.width / fw, avail_h / fh)
            dw, dh = max(int(fw * scale), 1), max(int(fh * scale), 1)
            vp = rect(
                inner.left + (inner.width - dw) // 2,
                inner.bottom + (avail_h - dh) // 2,
                dw, dh,
            )
            # mjr_drawPixels does no scaling: it consumes exactly
            # viewport.width * viewport.height pixels and uses the viewport
            # width as the row stride, so resize to the viewport first.
            buf = self._fit(self.frame, dw, dh)
            mujoco.mjr_drawPixels(buf.reshape(-1), None, vp, con)
        else:
            msg = "WAITING FOR CAMERA"
            self._text_px(con, inner, msg,
                          (inner.width - self.text_width(msg)) / 2,
                          inner.height / 2, th.text_dim)

        lab = rect(inner.left, inner.bottom + inner.height - label_h,
                   inner.width, label_h)
        self._text_px(con, lab, "TRACKING", 8, 5, th.text_dim)

    # --- simulation -------------------------------------------------------

    def _sim_panel(self, con, l, b, w, h):
        th = self.theme
        pad = int(min(w, h) * 0.05)
        inner = rect(l + pad, b + pad, w - 2 * pad, h - 2 * pad)
        mujoco.mjr_rectangle(inner, *th.surface, 1)

        label_h = 24
        scene_vp = rect(inner.left, inner.bottom,
                        inner.width, inner.height - label_h)
        mujoco.mjv_updateScene(
            self.model, self.data, self.opt, None, self.cam,
            mujoco.mjtCatBit.mjCAT_ALL, self.scene,
        )
        mujoco.mjr_render(scene_vp, self.scene, con)

        lab = rect(inner.left, inner.bottom + inner.height - label_h,
                   inner.width, label_h)
        self._text_px(con, lab, "COMMANDED", 8, 5, th.text_dim)

    # --- footer -----------------------------------------------------------

    def _footer(self, con, W, h, split):
        th = self.theme
        mujoco.mjr_rectangle(rect(0, 0, W, h), *th.bg, 1)
        mujoco.mjr_rectangle(rect(0, h - 1, W, 1), *th.hairline, 1)
        self._joint_bars(con, 0, 0, split, h)
        self._quality_panel(con, split, 0, W - split, h)

    def _joint_bars(self, con, l, b, w, h):
        if self.arm_joints is not None:
            half = w // 2
            # "above" for the arm: a stationary joint has MSJ near zero, and
            # holding still is normal for a robot arm - it should not read as
            # a fault. For the HAND, a collapse to zero means tracking was
            # lost, which is worth flagging, so that side stays symmetric.
            self._bar_group(con, l, b, half, h, "ARM", ARM_LABELS,
                            self._arm_values(), self._arm_fracs(),
                            msj=self.arm_msj, direction="above")
            mujoco.mjr_rectangle(rect(l + half, b + int(h * 0.12), 1,
                                      int(h * 0.76)), *self.theme.hairline, 1)
            self._bar_group(con, l + half, b, half, h, "HAND", JOINT_LABELS,
                            [float(v) for v in self.targets],
                            [min(abs(float(v)) / 100.0, 1.0)
                             for v in self.targets], msj=self.msj)
            return
        self._bar_group(con, l, b, w, h, "HAND", JOINT_LABELS,
                        [float(v) for v in self.targets],
                        [min(abs(float(v)) / 100.0, 1.0)
                         for v in self.targets], msj=self.msj)

    def _arm_values(self):
        return [float(np.degrees(v)) for v in self.arm_joints[:6]]

    def _arm_fracs(self):
        """How far each joint sits through its own range."""
        out = []
        for v, (lo, hi) in zip(self.arm_joints[:6], ARM_RANGES):
            out.append(min(max((float(v) - lo) / (hi - lo), 0.0), 1.0))
        return out

    def _bar_group(self, con, l, b, w, h, title, labels, values, fracs,
                   msj=None, direction="both"):
        th = self.theme
        pad = 14
        n = len(labels)
        # Column widths come from the longest string actually present, so a
        # label can never run under a bar and a value can never run off the
        # panel.
        lab_w = max(self.text_width(s) for s in labels) + pad
        val_w = max(self.text_width(f"{v:+.0f}") for v in values) + pad
        self._text_px(con, rect(l, b, w, h), title, pad, h - self.CHAR_H - 6,
                      th.text_dim)
        top = b + h - int(self.CHAR_H) - 14
        bottom = b + 8
        row = (top - bottom) / n
        bar_l = l + pad + lab_w
        bar_w = w - (bar_l - l) - val_w - pad
        if bar_w < 20:                     # too narrow for bars: text only
            bar_w = 0
            bar_l = l + pad + lab_w

        for i, label in enumerate(labels):
            row_b = int(top - (i + 1) * row)
            rowr = rect(l, row_b, w, int(row))
            bh = max(int(row * 0.26), 2)
            by = row_b + int((row - bh) / 2)

            frac = fracs[i]
            # Both groups are coloured by the same biomechanical metric.
            # Dimensionless MSJ is amplitude and unit free, so one calibrated
            # reference applies to a finger and a shoulder alike.
            if msj is not None and i < len(msj):
                col = msj_color(float(msj[i]), self.msj_reference,
                                spread=self.msj_spread, direction=direction)
            else:
                col = th.text_dim
            if bar_w:
                mujoco.mjr_rectangle(rect(bar_l, by, bar_w, bh),
                                     *th.hairline, 1)
                if frac > 0.001:
                    mujoco.mjr_rectangle(
                        rect(bar_l, by, max(int(bar_w * frac), 2), bh),
                        *col, 1
                    )
            ty = max((row - self.CHAR_H) / 2.0, 1.0)
            self._text_px(con, rowr, label, pad, ty, th.text_dim)
            # Value is right-aligned to the panel edge, so it stays put as
            # the number changes width between -100 and +7.
            vtxt = f"{values[i]:+.0f}"
            self._text_px(con, rowr, vtxt,
                          w - pad - self.text_width(vtxt), ty,
                          col if bar_w == 0 else th.text)

    def _quality_panel(self, con, l, b, w, h):
        th = self.theme
        r = rect(l, b, w, h)
        pad = 16

        total = float(np.mean(self.msj))
        band = msj_band(total, self.msj_reference, self.msj_spread)
        col = msj_color(total, self.msj_reference, spread=self.msj_spread)

        big_txt = f"{total:.0f}"
        band_txt = f"MSJ  {band.upper()}"
        ref = f"REF {self.msj_reference:.0f}"
        if self.msj_spread:
            ref += f" +/- {self.msj_spread:.0f}"
        mae_txt = f"MAE {self.mae_mm:.2f} MM"

        # Four stacked lines, spaced by the font's own height so they cannot
        # run into each other however short the panel is.
        y = h - self.CHAR_H_BIG - 10
        self._text_px(con, r, big_txt, pad, y, col, big=True)
        y -= self.CHAR_H + 8
        self._text_px(con, r, band_txt, pad, y, th.text_dim)
        y -= self.CHAR_H + 4
        self._text_px(con, r, ref, pad, y, th.text_dim)
        y -= self.CHAR_H + 4
        self._text_px(con, r, mae_txt, pad, y, th.text_dim)

        left_w = max(
            self.text_width(big_txt, big=True),
            self.text_width(band_txt),
            self.text_width(ref),
            self.text_width(mae_txt),
        )
        gl = l + pad + int(left_w) + 24
        gw = w - (gl - l) - pad
        if gw < 80:                       # no room for the gauge
            return
        gy = b + int(h * 0.42)
        gh = max(int(h * 0.055), 4)
        seg = max(gw // 48, 1)
        for i in range(gw // seg):
            tpos = i / max(gw / seg - 1, 1)
            dev = (tpos - 0.5) * 2.0
            scale = (self.msj_spread * 5.0) if self.msj_spread else 250.0
            c = msj_color(
                self.msj_reference + dev * scale, self.msj_reference,
                spread=self.msj_spread,
            )
            mujoco.mjr_rectangle(rect(gl + i * seg, gy, seg - 1, gh), *c, 1)

        scale = (self.msj_spread * 5.0) if self.msj_spread else 250.0
        pos = 0.5 + (total - self.msj_reference) / (2.0 * scale)
        pos = min(max(pos, 0.0), 1.0)
        mx = gl + int(pos * (gw - 2))
        mujoco.mjr_rectangle(rect(mx, gy - 5, 2, gh + 10), *th.marker, 1)

        gr = rect(gl, gy - int(h * 0.26), gw, int(h * 0.2))
        self._text(con, gr, "DEVIATION FROM BASELINE", 0.0, 0.3, th.text_dim)

    def _fit(self, frame, w, h):
        """Resize an RGB frame to exactly (w, h), cached between draws."""
        key = (frame.shape, w, h, frame.__array_interface__["data"][0])
        if getattr(self, "_fit_key", None) == key:
            return self._fit_buf
        try:
            import cv2

            out = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        except Exception:
            ys = (np.arange(h) * frame.shape[0] // h).clip(0, frame.shape[0] - 1)
            xs = (np.arange(w) * frame.shape[1] // w).clip(0, frame.shape[1] - 1)
            out = frame[ys][:, xs]
        out = np.ascontiguousarray(out, dtype=np.uint8)
        self._fit_key = key
        self._fit_buf = out
        return out

    # MuJoCo's bitmap fonts are fixed width. These were MEASURED at
    # FONTSCALE_150 by rendering a known string and reading back the pixel
    # extent - guessed values were 30% low, which made items that were
    # calculated to fit overlap by half a string.
    CHAR_W = 13.9
    CHAR_H = 17.0
    CHAR_W_BIG = 24.8
    CHAR_H_BIG = 33.0

    @classmethod
    def text_width(cls, txt, big=False):
        return len(txt) * (cls.CHAR_W_BIG if big else cls.CHAR_W)

    @staticmethod
    def _text(con, r, txt, xf, yf, col, big=False):
        """Draw text at normalised (xf, yf) inside rect r."""
        mujoco.mjr_rectangle(r, 0, 0, 0, 0)   # sets the GL viewport only
        font = mujoco.mjtFont.mjFONT_BIG if big else mujoco.mjtFont.mjFONT_NORMAL
        mujoco.mjr_text(font, txt, con, xf, yf, *col)

    @classmethod
    def _text_px(cls, con, r, txt, x_px, y_px, col, big=False):
        """Draw text at pixel offsets inside rect r.

        Everything that shares a row is placed this way: normalised fractions
        put items at fixed proportions of the width, so at a narrower window
        the strings run into each other.
        """
        if r.width <= 0 or r.height <= 0:
            return
        cls._text(con, r, txt, x_px / r.width, y_px / r.height, col, big)

    @classmethod
    def _row_right(cls, con, r, items, y_px, gap=18.0):
        """Right-align a sequence of (text, colour), packing right to left.

        Returns the x of the leftmost item drawn, so a caller can check
        whether something to its left would collide.
        """
        x = r.width - gap
        left = x
        for txt, col in reversed(items):
            wpx = cls.text_width(txt)
            x -= wpx
            if x < 0:
                break
            cls._text_px(con, r, txt, x, y_px, col)
            left = x
            x -= gap
        return left
