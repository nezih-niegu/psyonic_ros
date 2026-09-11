"""Retarget MediaPipe hand landmarks onto the Ability Hand's 6 DOF.

MediaPipe HandLandmarker returns 21 landmarks (points). The Ability Hand has
6 actuated degrees of freedom:

    index, middle, ring, pinky, thumb_flexor, thumb_rotator

so this is a dimensionality reduction, not a joint-for-joint copy. Each finger's
three interior angles are collapsed into a single "curl" scalar, which is what
the hand's one motor per finger actually controls. The distal joint (q2) is not
commanded at all — it follows mechanically through the URDF mimic relation.

Landmark indices follow HandPose's LANDMARK_NAMES ordering.

Everything here is pure numpy so it can be unit tested without a camera.
"""

import numpy as np

from ah_mujoco.signal_quality import OneEuroFilter

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

# Finger chains, in the hand's own joint order
FINGER_CHAINS = {
    "index": (INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP),
    "middle": (MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP),
    "ring": (RING_MCP, RING_PIP, RING_DIP, RING_TIP),
    "pinky": (PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP),
}

# Ability Hand command limits, in the hardware order the driver expects:
# [index, middle, ring, pinky, thumb_flexor, thumb_rotator], degrees.
# Derived from the URDF joint limits: finger q1 and thumb_q2 are 0..+114 deg,
# thumb_q1 (the rotator) runs 0..-114 deg.
HAND_LIMITS_DEG = np.array(
    [
        [0.0, 100.0],    # index
        [0.0, 100.0],    # middle
        [0.0, 100.0],    # ring
        [0.0, 100.0],    # pinky
        [0.0, 100.0],    # thumb_flexor  (maps to thumb_q2, 0..+)
        [-100.0, 0.0],   # thumb_rotator (maps to thumb_q1, -..0)
    ],
    dtype=np.float32,
)


def _angle_deg(a, b, c):
    """Interior angle at b, in degrees. 180 = straight."""
    v1 = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    v2 = np.asarray(c, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return 180.0
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def finger_curl_deg(lm, chain):
    """Total curl of one finger: how far it is from fully extended.

    Sums the flexion at MCP, PIP and DIP. Roughly 0 when straight, and
    around 200-260 for a tight fist.
    """
    mcp, pip, dip, tip = chain
    at_mcp = _angle_deg(lm[WRIST], lm[mcp], lm[pip])
    at_pip = _angle_deg(lm[mcp], lm[pip], lm[dip])
    at_dip = _angle_deg(lm[pip], lm[dip], lm[tip])
    return (180.0 - at_mcp) + (180.0 - at_pip) + (180.0 - at_dip)


def thumb_flex_deg(lm):
    """Thumb flexion: bend along the thumb chain, ignoring rotation."""
    at_mcp = _angle_deg(lm[THUMB_CMC], lm[THUMB_MCP], lm[THUMB_IP])
    at_ip = _angle_deg(lm[THUMB_MCP], lm[THUMB_IP], lm[THUMB_TIP])
    return (180.0 - at_mcp) + (180.0 - at_ip)


def thumb_oppose_deg(lm):
    """Thumb opposition: angle between the thumb and index metacarpals.

    Large when the thumb is spread away from the palm, small when it swings
    across toward the little finger. This is the rotator axis, which is
    separate from flexion and is what makes a real grasp possible.
    """
    return _angle_deg(lm[THUMB_TIP], lm[WRIST], lm[INDEX_MCP])


def _map_range(value, src_lo, src_hi, dst_lo, dst_hi):
    if abs(src_hi - src_lo) < 1e-9:
        return dst_lo
    t = (value - src_lo) / (src_hi - src_lo)
    t = min(max(t, 0.0), 1.0)
    return dst_lo + t * (dst_hi - dst_lo)


class HandRetargeter:
    """Landmarks -> 6 Ability Hand joint targets in degrees.

    The calibration ranges matter more than anything else here: hands differ
    in size and the camera sees them at different distances, so the raw curl
    values for "open" and "closed" vary between people. Defaults are sane for
    an adult hand at arm's length; call observe() during a calibration pose to
    adapt them.
    """

    def __init__(
        self,
        curl_open=15.0,
        curl_closed=200.0,
        thumb_flex_open=10.0,
        thumb_flex_closed=90.0,
        thumb_opp_open=50.0,
        thumb_opp_closed=15.0,
        smoothing=0.0,
        one_euro=True,
        min_cutoff=0.1,
        beta=0.005,
        dt=1.0 / 30.0,
    ):
        self.curl_open = curl_open
        self.curl_closed = curl_closed
        self.thumb_flex_open = thumb_flex_open
        self.thumb_flex_closed = thumb_flex_closed
        self.thumb_opp_open = thumb_opp_open
        self.thumb_opp_closed = thumb_opp_closed
        self.smoothing = smoothing
        self._prev = None
        # One Euro adapts its cutoff to the speed of the signal, so jitter at
        # rest and lag during motion can be traded almost independently. A
        # fixed EMA cannot: enough smoothing to kill jitter always costs lag.
        # Measured through the full pipeline at 30 Hz: 1.42 deg residual
        # jitter at 0 ms added lag, against 4.07 deg for EMA alpha=0.5 (which
        # also added 34 ms) and 7.20 deg for the alpha=0.15 it replaced.
        # beta is unit-dependent: it multiplies speed in deg/s, so retuning is
        # needed if the input scale changes.
        self.one_euro = OneEuroFilter(
            6, dt, min_cutoff=min_cutoff, beta=beta
        ) if one_euro else None

    def raw_features(self, lm):
        """The six scalars before any range mapping. Useful for calibration."""
        lm = np.asarray(lm, dtype=np.float64).reshape(21, 3)
        feats = [finger_curl_deg(lm, FINGER_CHAINS[f])
                 for f in ("index", "middle", "ring", "pinky")]
        feats.append(thumb_flex_deg(lm))
        feats.append(thumb_oppose_deg(lm))
        return np.array(feats, dtype=np.float32)

    def __call__(self, lm, dt=None):
        return self.retarget(lm, dt)

    def retarget(self, lm, dt=None):
        """Return 6 joint targets in degrees, hardware order.

        dt is the interval since the previous sample; passing the real value
        keeps the filter correct when the source rate varies, which it does
        with a camera topic.
        """
        f = self.raw_features(lm)

        out = np.zeros(6, dtype=np.float32)
        for i in range(4):  # index, middle, ring, pinky
            lo, hi = HAND_LIMITS_DEG[i]
            out[i] = _map_range(f[i], self.curl_open, self.curl_closed, lo, hi)

        lo, hi = HAND_LIMITS_DEG[4]
        out[4] = _map_range(
            f[4], self.thumb_flex_open, self.thumb_flex_closed, lo, hi
        )

        # Rotator runs negative; opposition decreases as the thumb swings in,
        # so the source range is inverted on purpose.
        lo, hi = HAND_LIMITS_DEG[5]
        out[5] = _map_range(
            f[5], self.thumb_opp_open, self.thumb_opp_closed, 0.0, lo
        )

        out = np.clip(out, HAND_LIMITS_DEG[:, 0], HAND_LIMITS_DEG[:, 1])

        # MediaPipe jitters frame to frame and the motors should not chase it.
        if self.one_euro is not None:
            out = self.one_euro.update(out, dt)
        elif self.smoothing > 0.0:
            if self._prev is None:
                self._prev = out
            else:
                a = self.smoothing
                out = a * self._prev + (1.0 - a) * out
                self._prev = out
        return np.asarray(out, dtype=np.float32)

    def reset(self):
        self._prev = None
        if self.one_euro is not None:
            self.one_euro.reset()
