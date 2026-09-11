"""Signal quality and smoothness metrics for tracked hand motion.

Follows the preprocessing used in Morales et al., "Biomechanical evaluation as
basis for the development of predictive models in microsurgery robots":

  * a Kalman filter on the raw tracked positions (the paper found it preferable
    to Savitzky-Golay, preserving movement characteristics with less error)
  * Mean Absolute Error between raw and filtered signals as the measure of how
    noisy the acquisition is
  * Mean Squared Jerk as a biomechanically motivated realism score

Jerk is the third derivative of position. Human reaching movements are close to
minimum-jerk trajectories, so an implausibly high MSJ means the commanded
motion is not something a human limb would produce - usually tracking noise
rather than real movement.

    J_t = (p_t - 3 p_{t-1} + 3 p_{t-2} - p_{t-3}) / dt^3
    MSJ = (1/N) sum ||J_i||^2

IMPORTANT - units. MSJ scales as (unit/s^3)^2, so its numeric value depends
entirely on the position unit and the sample rate. A reference of 524 is only
meaningful alongside the units and dt it was measured with. Everything here
keeps the unit explicit and the reference configurable; calibrate the reference
on your own signal rather than assuming it transfers.
"""

import numpy as np

# Deviation thresholds from the MSJ reference, and their colours.
# Continuous interpolation between the stops, green -> red.
MSJ_COLOR_STOPS = [
    (0.0, (0.15, 0.80, 0.20)),    # green   - on reference
    (50.0, (0.95, 0.90, 0.15)),   # yellow
    (100.0, (0.98, 0.55, 0.10)),  # orange
    (250.0, (0.90, 0.12, 0.12)),  # red     - and beyond
]


class KalmanPosition:
    """Constant-acceleration Kalman filter, applied per coordinate.

    State per coordinate: [position, velocity, acceleration]. A constant-
    acceleration model is used rather than constant-velocity because the
    downstream metric is jerk, and a CV model would bias acceleration toward
    zero and flatten exactly the signal being measured.

    All coordinates are advanced together with batched (dim, 3, 3) array
    operations rather than a Python loop over coordinates. Because the
    observation model is H = [1, 0, 0], the usual matrix products collapse to
    column indexing: H P H^T is P[:, 0, 0] and P H^T is P[:, :, 0], so no
    3x3 inverse is ever formed.

    Parameters
    ----------
    dim : number of independent coordinates to track
    dt : sample period in seconds
    process_var : trust in the model; raise it to track fast motion, lower it
        to smooth harder
    measurement_var : assumed variance of the tracker's position noise
    xp : array module. Defaults to numpy. Pass cupy to run on GPU, which is
        only worth it for very large dim - see the note in the class docstring
        of the module README.
    """

    def __init__(self, dim, dt, process_var=1e-2, measurement_var=1e-3, xp=None):
        self.xp = xp if xp is not None else np
        xp = self.xp

        self.dim = dim
        self.dt = dt
        self.q = process_var
        self.r = measurement_var

        self.x = xp.zeros((dim, 3))
        self.P = xp.tile(xp.eye(3), (dim, 1, 1))
        self._I = xp.eye(3)

        self.F = xp.asarray(
            [[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]]
        )

        # Continuous white-noise-jerk process noise
        dt3, dt4, dt5 = dt ** 3, dt ** 4, dt ** 5
        self.Q = self.q * xp.asarray(
            [
                [dt5 / 20.0, dt4 / 8.0, dt3 / 6.0],
                [dt4 / 8.0, dt3 / 3.0, dt * dt / 2.0],
                [dt3 / 6.0, dt * dt / 2.0, dt],
            ]
        )
        self._initialised = False

    def reset(self):
        xp = self.xp
        self.x = xp.zeros((self.dim, 3))
        self.P = xp.tile(xp.eye(3), (self.dim, 1, 1))
        self._initialised = False

    def update(self, z):
        """Filter one measurement vector. Returns the filtered positions."""
        xp = self.xp
        z = xp.asarray(z).reshape(self.dim)

        if not self._initialised:
            self.x[:, 0] = z
            self._initialised = True
            return z.copy()

        # Predict, batched over coordinates
        x = self.x @ self.F.T
        P = self.F @ self.P @ self.F.T + self.Q

        # Update. H = [1,0,0] collapses the products to column 0.
        y = z - x[:, 0]
        S = P[:, 0, 0] + self.r
        K = P[:, :, 0] / S[:, None]

        self.x = x + K * y[:, None]
        KH = xp.zeros((self.dim, 3, 3))
        KH[:, :, 0] = K
        self.P = (self._I - KH) @ P
        return self.x[:, 0].copy()


class MAETracker:
    """Running Mean Absolute Error between raw and filtered signals.

    This is the paper's measure of acquisition quality: how much the filter had
    to move the signal is a proxy for how noisy the raw signal was.
    """

    def __init__(self, dim, window=120):
        self.window = window
        self.dim = dim
        self._buf = []

    def update(self, raw, filtered):
        err = np.abs(np.asarray(raw, dtype=np.float64)
                     - np.asarray(filtered, dtype=np.float64))
        self._buf.append(err)
        if len(self._buf) > self.window:
            self._buf.pop(0)
        return self.mae()

    def mae(self):
        if not self._buf:
            return np.zeros(self.dim)
        return np.mean(np.asarray(self._buf), axis=0)

    def total(self):
        return float(np.mean(self.mae())) if self._buf else 0.0


def jerk_from_history(p, dt):
    """Third-difference jerk from 4 consecutive samples.

    p : array shaped (4, dim), oldest first, i.e. [p_{t-3}, p_{t-2}, p_{t-1}, p_t]
    """
    p = np.asarray(p, dtype=np.float64)
    return (p[3] - 3.0 * p[2] + 3.0 * p[1] - p[0]) / (dt ** 3)


class MSJMonitor:
    """Rolling Mean Squared Jerk, per coordinate and overall.

    Two flavours, selected by `normalize`:

    raw (normalize=False)
        MSJ exactly as written: mean of ||J||^2. Scales as amplitude^2 / dt^6,
        so its numeric value is meaningful only alongside the units, sample
        rate and movement scale it was measured at. Degrees at 30 Hz put even
        an ideal minimum-jerk finger curl above 10^6.

    dimensionless (normalize=True, the default)
        The standard normalisation from the motor-control literature
        (Balasubramanian et al.): the jerk integral scaled by movement duration
        and amplitude,

            DJ = sqrt( (T^5 / (2 A^2)) * integral |J(t)|^2 dt )

        This cancels the units, the sample rate and the size of the movement,
        so a fixed reference such as 524 means the same thing across setups.
        A smooth minimum-jerk movement lands near a constant regardless of how
        fast or far it moved; noise pushes it up by orders of magnitude.
    """

    def __init__(self, dim, dt, window=90, normalize=True):
        self.dim = dim
        self.dt = dt
        self.window = window
        self.normalize = normalize
        self._hist = []
        self._jerks = []
        self._pos = []

    def update(self, p):
        p = np.asarray(p, dtype=np.float64).reshape(self.dim)
        self._hist.append(p)
        if len(self._hist) > 4:
            self._hist.pop(0)
        self._pos.append(p)
        if len(self._pos) > self.window + 4:
            self._pos.pop(0)
        if len(self._hist) == 4:
            j = jerk_from_history(self._hist, self.dt)
            self._jerks.append(j * j)          # squared per coordinate
            if len(self._jerks) > self.window:
                self._jerks.pop(0)
        return self.msj()

    def _amplitude(self):
        """Peak-to-peak movement over the window, per coordinate."""
        if len(self._pos) < 2:
            return np.ones(self.dim)
        a = np.asarray(self._pos)
        amp = a.max(axis=0) - a.min(axis=0)
        return np.maximum(amp, 1e-6)

    def msj(self):
        """Per-coordinate score (dimensionless if normalize is set)."""
        if not self._jerks:
            return np.zeros(self.dim)
        raw = np.mean(np.asarray(self._jerks), axis=0)
        if not self.normalize:
            return raw
        n = len(self._jerks)
        T = max(n * self.dt, 1e-6)
        amp = self._amplitude()
        # integral |J|^2 dt = mean(|J|^2) * T
        return np.sqrt(0.5 * raw * T * (T ** 5) / (amp ** 2))

    def msj_total(self):
        """Scalar score over all coordinates."""
        if not self._jerks:
            return 0.0
        if not self.normalize:
            return float(np.sum(np.asarray(self._jerks), axis=1).mean())
        return float(np.mean(self.msj()))

    def reset(self):
        self._hist.clear()
        self._jerks.clear()
        self._pos.clear()


def msj_color(
    msj_value,
    reference=524.0,
    stops=MSJ_COLOR_STOPS,
    spread=None,
    direction="both",
):
    """Map an MSJ value to an RGB colour by its deviation from reference.

    The reference is what the user's own motion measured during calibration,
    so sitting on it is "good" and is drawn green; departing from it fades
    through yellow and orange to red.

    Parameters
    ----------
    reference : the calibrated baseline, i.e. the green centre
    stops : (deviation, rgb) anchor points, interpolated continuously
    spread : if given, the standard deviation of MSJ measured during
        calibration. The stop deviations are then interpreted in units of that
        spread rather than absolute, so the same thresholds mean the same
        thing whether the baseline measured 12 or 500. The default stops
        (50/100/250) are scaled to (1/2/5) sigma.
    direction : "both" flags departure in either direction; "above" flags only
        jerkier-than-baseline motion and leaves smoother motion green.

    Returns (r, g, b) floats in 0..1.
    """
    delta = float(msj_value) - float(reference)

    if direction == "above" and delta < 0.0:
        return stops[0][1]          # smoother than baseline stays green
    dev = abs(delta)

    if spread is not None and spread > 1e-9:
        # Reinterpret the stops as multiples of the measured spread:
        # 50 -> 1 sigma, 100 -> 2 sigma, 250 -> 5 sigma
        sigma_stops = [
            (d / 50.0 * spread, c) for d, c in stops
        ]
        stops = sigma_stops

    if dev <= stops[0][0]:
        return stops[0][1]
    if dev >= stops[-1][0]:
        return stops[-1][1]

    for (d0, c0), (d1, c1) in zip(stops, stops[1:]):
        if d0 <= dev <= d1:
            t = (dev - d0) / (d1 - d0) if d1 > d0 else 0.0
            return tuple(a + (b - a) * t for a, b in zip(c0, c1))
    return stops[-1][1]


def msj_band(msj_value, reference=524.0, spread=None, direction="both"):
    """Human-readable band name for logging."""
    delta = float(msj_value) - float(reference)
    if direction == "above" and delta < 0.0:
        return "good"
    dev = abs(delta)
    scale = (spread / 50.0) if (spread is not None and spread > 1e-9) else 1.0
    if dev < 50.0 * scale:
        return "good"
    if dev < 100.0 * scale:
        return "fair"
    if dev < 250.0 * scale:
        return "poor"
    return "bad"


class OneEuroFilter:
    """Adaptive low-pass filter (Casiez, Roussel & Vogel, CHI 2012).

    A fixed low-pass forces a single trade: enough smoothing to kill jitter at
    rest always costs lag during fast motion. One Euro varies its cutoff with
    the observed speed of the signal - heavy smoothing when the hand is still,
    light smoothing when it moves - so jitter and lag can be tuned almost
    independently.

        cutoff = min_cutoff + beta * |dx/dt|

    Parameters
    ----------
    min_cutoff : cutoff in Hz when the signal is still. Lower is smoother at
        rest and does not affect fast motion much.
    beta : how fast the cutoff opens up with speed. Higher tracks quick motion
        with less lag, at the cost of passing more noise during it.
    d_cutoff : cutoff for the derivative estimate itself; rarely needs changing.
    """

    def __init__(self, dim, dt, min_cutoff=1.0, beta=0.3, d_cutoff=1.0):
        self.dim = dim
        self.dt = dt
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self._x_prev = None
        self._dx_prev = np.zeros(dim)

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2.0 * np.pi * np.maximum(cutoff, 1e-6))
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self._x_prev = None
        self._dx_prev = np.zeros(self.dim)

    def update(self, x, dt=None):
        x = np.asarray(x, dtype=np.float64).reshape(self.dim)
        dt = float(dt if dt else self.dt)
        if dt <= 0.0:
            dt = self.dt

        if self._x_prev is None:
            self._x_prev = x.copy()
            return x.copy()

        # Filter the derivative first, or its own noise drives the cutoff
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        self._dx_prev = dx_hat

        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        return x_hat
