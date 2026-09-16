"""Analysis of how well operator motion transmits to the robot.

Two questions, kept separate because they have different answers and different
consequences:

  BIOFIDELITY - does the robot reproduce what the operator did? Lag, gain,
  residual error after accounting for both, the frequency band that survives,
  and whether the smoothness of the original motion is preserved.

  BIOMECHANICAL STRESS - what does the operator pay to drive it? Joint angles
  against comfortable range, time spent outside it, static holding, and
  cumulative jerk exposure.

A teleoperation link can score well on one and badly on the other. A rigid,
high-gain mapping transmits faithfully while forcing the operator into extreme
postures; a heavily filtered one is comfortable and transmits almost nothing.
Reporting a single number would hide that trade.

Everything here is pure numpy/scipy so it can be tested without a robot.
"""

import numpy as np

# Comfortable ranges for the shoulder and elbow, in degrees. These are the
# conventional "neutral / low-risk" bands used in ergonomic screening
# (RULA/REBA style), not anatomical limits - a joint can reach much further
# than is comfortable to hold.
COMFORT = {
    "shoulder_elevation": (0.0, 60.0),   # beyond 60 deg loading rises sharply
    "shoulder_abduction": (0.0, 45.0),
    "elbow_flexion": (20.0, 100.0),      # very straight or very bent both load
}


# --------------------------------------------------------------- biofidelity

def estimate_lag(a, b, dt, max_lag_s=1.0, use_velocity=True):
    """Lag of b behind a, in seconds, by cross-correlation.

    Correlation is taken on the DERIVATIVES by default. Teleoperation signals
    are smooth and slow, and the correlation of two smooth position traces has
    a very broad peak - measured errors of 80-100 ms against a known lag.
    Differentiating sharpens the peak because it emphasises the transitions,
    which are what actually carry timing information.

    De-meaned first: a constant offset would otherwise dominate.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    if use_velocity and a.size > 2 and b.size > 2:
        a = np.diff(a)
        b = np.diff(b)
    n = min(a.size, b.size)
    a, b = a[:n] - np.mean(a[:n]), b[:n] - np.mean(b[:n])
    if n < 4 or np.allclose(a, 0) or np.allclose(b, 0):
        return float("nan"), float("nan")

    max_lag = int(min(max_lag_s / dt, n - 1))
    corr = np.correlate(b, a, mode="full")
    mid = n - 1
    lo, hi = mid - max_lag, mid + max_lag + 1
    window = corr[lo:hi]
    k = int(np.argmax(window)) - max_lag
    denom = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    peak = float(window[int(np.argmax(window))] / denom) if denom > 0 else 0.0
    return k * dt, peak


def estimate_gain(a, b):
    """Least-squares slope of b against a, through the origin after de-meaning.

    This is the fraction of the operator's motion that reaches the robot. It
    should come out at `position_scale`; a value well below it means something
    downstream - rate limiting, IK failure, workspace clamping - is eating
    motion that the operator produced.
    """
    a = np.asarray(a, dtype=float).ravel()
    b = np.asarray(b, dtype=float).ravel()
    n = min(a.size, b.size)
    a, b = a[:n] - np.mean(a[:n]), b[:n] - np.mean(b[:n])
    denom = float(np.sum(a * a))
    if denom < 1e-12:
        return float("nan")
    return float(np.sum(a * b) / denom)


def align(a, b, lag_samples):
    """Shift b back by lag_samples and trim both to the overlap."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    k = int(lag_samples)
    if k > 0:
        return a[:len(a) - k], b[k:]
    if k < 0:
        return a[-k:], b[:len(b) + k]
    n = min(len(a), len(b))
    return a[:n], b[:n]


def tracking_error(operator, robot, dt, gain=None):
    """RMSE between operator and robot after removing lag and gain.

    Residual error is what is left once the two known, correctable effects are
    accounted for. Lag can be compensated by prediction and gain is a setting;
    what remains is the part of the operator's motion the link genuinely
    failed to convey.
    """
    operator = np.asarray(operator, dtype=float)
    robot = np.asarray(robot, dtype=float)
    lag_s, peak = estimate_lag(operator, robot, dt)
    if not np.isfinite(lag_s):
        return {"lag_s": float("nan"), "gain": float("nan"),
                "rmse": float("nan"), "correlation": float("nan")}

    a, b = align(operator, robot, round(lag_s / dt))
    g = estimate_gain(a, b) if gain is None else gain
    if not np.isfinite(g) or abs(g) < 1e-9:
        return {"lag_s": lag_s, "gain": g, "rmse": float("nan"),
                "correlation": peak}

    # Put the robot back on the operator's scale before comparing
    resid = (b - np.mean(b)) / g - (a - np.mean(a))
    return {
        "lag_s": float(lag_s),
        "gain": float(g),
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "correlation": float(peak),
    }


def transmission_bandwidth(operator, robot, dt, drop_db=-3.0):
    """Frequency at which the link stops passing the operator's motion.

    Measured from the transfer-function magnitude |Pxy / Pxx|, normalised by
    its low-frequency value, and reported at the -3 dB crossing.

    Coherence is the obvious candidate and is the wrong tool here: for a
    noiseless linear filter, magnitude-squared coherence stays near 1 at every
    frequency no matter how much the filter attenuates. A 0.5 Hz low-pass
    measured a 0.94 Hz "bandwidth" that way, and a 5 Hz one measured 12.9 Hz.
    Coherence answers "is the output linearly related to the input", not "does
    any of the input get through" - so it is returned alongside, as a
    confidence measure on the gain estimate rather than as the bandwidth.
    """
    try:
        from scipy.signal import coherence, csd, welch
    except ImportError:
        return {"bandwidth_hz": float("nan"), "f": np.array([]),
                "gain_db": np.array([]), "coherence": np.array([])}

    operator = np.asarray(operator, dtype=float).ravel()
    robot = np.asarray(robot, dtype=float).ravel()
    n = min(operator.size, robot.size)
    if n < 64:
        return {"bandwidth_hz": float("nan"), "f": np.array([]),
                "gain_db": np.array([]), "coherence": np.array([])}
    operator, robot = operator[:n], robot[:n]

    nperseg = int(min(512, max(64, n // 6)))
    f, pxx = welch(operator, fs=1.0 / dt, nperseg=nperseg)
    _, pxy = csd(operator, robot, fs=1.0 / dt, nperseg=nperseg)
    _, cxy = coherence(operator, robot, fs=1.0 / dt, nperseg=nperseg)

    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.abs(pxy) / np.maximum(pxx, 1e-20)

    # Reference gain from the lowest frequencies that carry real energy
    valid = pxx > pxx.max() * 1e-3
    ref = np.median(h[valid][:5]) if np.any(valid) else (h[1] if h.size > 1 else 1.0)
    if not np.isfinite(ref) or ref <= 0:
        ref = 1.0
    gain_db = 20.0 * np.log10(np.maximum(h / ref, 1e-6))

    bw = float("nan")
    for i in range(1, len(f)):
        if gain_db[i] < drop_db:
            # Linear interpolation onto the crossing
            g0, g1 = gain_db[i - 1], gain_db[i]
            if g1 != g0:
                frac = (drop_db - g0) / (g1 - g0)
                bw = float(f[i - 1] + frac * (f[i] - f[i - 1]))
            else:
                bw = float(f[i])
            break
    if not np.isfinite(bw):
        bw = float(f[-1])

    return {"bandwidth_hz": bw, "f": f, "gain_db": gain_db, "coherence": cxy}


# Kept under the old name so callers that want coherence still get it
def coherence_bandwidth(operator, robot, dt, threshold=0.5):
    return transmission_bandwidth(operator, robot, dt)


def smoothness_ratio(operator, robot, dt):
    """Dimensionless jerk of each signal, and the robot / operator ratio.

    Above 1 the robot moves less smoothly than the operator did - the link
    added jerk, usually rate limiting or IK stepping between branches. Below 1
    it smoothed the motion, which is filtering and shows up as lag.
    """
    from ah_mujoco.signal_quality import MSJMonitor

    def score(x):
        x = np.asarray(x, dtype=float).ravel()
        m = MSJMonitor(1, dt, window=10 ** 6, normalize=True)
        for v in x:
            m.update([v])
        return m.msj_total()

    so = score(operator)
    sr = score(robot)
    return {
        "operator": float(so),
        "robot": float(sr),
        "ratio": float(sr / so) if so > 1e-9 else float("nan"),
    }


# -------------------------------------------------------- biomechanics

def joint_angles(shoulder, elbow, wrist, up=np.array([0.0, -1.0, 0.0])):
    """Shoulder elevation, shoulder abduction and elbow flexion, in degrees.

    `up` is the direction of the operator's trunk axis in the camera frame.
    The default suits a camera whose Y axis points down, which is the usual
    optical-frame convention.
    """
    shoulder = np.asarray(shoulder, dtype=float)
    elbow = np.asarray(elbow, dtype=float)
    wrist = np.asarray(wrist, dtype=float)

    upper = elbow - shoulder
    fore = wrist - elbow
    nu = np.linalg.norm(upper, axis=-1, keepdims=True)
    nf = np.linalg.norm(fore, axis=-1, keepdims=True)
    upper_u = upper / np.maximum(nu, 1e-9)
    fore_u = fore / np.maximum(nf, 1e-9)

    up = up / np.linalg.norm(up)
    # Elevation: angle of the upper arm away from hanging straight down.
    # Zero is the arm at rest by the side, which is what the comfort bands
    # are referenced to.
    cos_elev = np.clip(np.sum(upper_u * (-up), axis=-1), -1.0, 1.0)
    elevation = np.degrees(np.arccos(cos_elev))

    # Abduction: how far the upper arm swings out of the sagittal plane
    lateral = upper_u - np.sum(upper_u * up, axis=-1, keepdims=True) * up
    abduction = np.degrees(np.arctan2(
        np.abs(lateral[..., 0]), np.abs(lateral[..., 2]) + 1e-9
    ))

    # Elbow flexion: 0 is straight
    cos_el = np.clip(np.sum(-upper_u * fore_u, axis=-1), -1.0, 1.0)
    flexion = 180.0 - np.degrees(np.arccos(cos_el))

    return {
        "shoulder_elevation": elevation,
        "shoulder_abduction": abduction,
        "elbow_flexion": flexion,
    }


def exposure(series, lo, hi, dt):
    """Time and fraction spent outside a comfortable band."""
    x = np.asarray(series, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"outside_s": 0.0, "fraction": 0.0, "p95": float("nan"),
                "median": float("nan"), "max": float("nan")}
    outside = np.sum((x < lo) | (x > hi))
    return {
        "outside_s": float(outside * dt),
        "fraction": float(outside / x.size),
        "p95": float(np.percentile(x, 95)),
        "median": float(np.median(x)),
        "max": float(np.max(x)),
    }


def static_loading(speed, dt, still_threshold=0.02, min_hold_s=3.0):
    """Holding still under load: the isometric part of the work.

    Sustained low-velocity periods are what fatigues an operator holding a
    posture, and they are invisible in a range-of-motion summary - the joint
    angle is simply constant. Returns total held time and the longest hold.
    """
    speed = np.asarray(speed, dtype=float).ravel()
    still = speed < still_threshold
    holds, run = [], 0
    for s in still:
        if s:
            run += 1
        else:
            if run:
                holds.append(run)
            run = 0
    if run:
        holds.append(run)
    holds = [h * dt for h in holds if h * dt >= min_hold_s]
    return {
        "total_held_s": float(sum(holds)),
        "longest_hold_s": float(max(holds)) if holds else 0.0,
        "n_holds": len(holds),
        "fraction": float(sum(holds) / (speed.size * dt)) if speed.size else 0.0,
    }


def jerk_exposure(series, dt):
    """Cumulative absolute jerk, a proxy for repetitive-load exposure."""
    x = np.asarray(series, dtype=float).ravel()
    if x.size < 4:
        return {"mean_abs_jerk": float("nan"), "cumulative": float("nan")}
    j = np.diff(x, n=3) / (dt ** 3)
    return {
        "mean_abs_jerk": float(np.mean(np.abs(j))),
        "cumulative": float(np.sum(np.abs(j)) * dt),
    }


def rula_arm_score(elevation, flexion, abduction):
    """Crude RULA-style upper-arm and lower-arm score, 1 (best) to 6.

    Not a validated RULA assessment - that needs wrist, neck, trunk, load and
    muscle-use scoring by a trained observer. It is a coarse posture flag to
    show when a session is drifting into territory a real assessment would
    care about.
    """
    e = np.asarray(elevation, dtype=float)
    upper = np.ones_like(e)
    upper = np.where(e > 20, 2, upper)
    upper = np.where(e > 45, 3, upper)
    upper = np.where(e > 90, 4, upper)
    upper = np.where(np.asarray(abduction, dtype=float) > 45, upper + 1, upper)

    f = np.asarray(flexion, dtype=float)
    lower = np.where((f >= 60) & (f <= 100), 1, 2)
    return np.clip(upper + lower - 1, 1, 6)
