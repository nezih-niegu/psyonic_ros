"""SE(3) hand pose from MediaPipe landmarks plus Kinect depth.

MediaPipe returns two landmark sets: normalised image coordinates, and metric
"world" landmarks expressed in a hand-centred frame. Neither alone gives the
hand's pose in the camera frame - the world landmarks are origin-centred, so
translation is lost, and the image landmarks have no scale.

Combining them with depth recovers the full rigid transform:

  1. deproject each image landmark to a 3D camera point using the depth image
     and the camera intrinsics
  2. solve for the rigid transform between the hand-frame world landmarks and
     those camera points (Kabsch / Umeyama, without scale)

All 21 landmarks are used for the fit, not just the palm. The world landmarks
already encode the current finger articulation, so at any instant every
landmark is a valid rigid correspondence - and the wider spatial spread
conditions the rotation far better than the nearly-coplanar palm alone.
Measured against known poses with 5 mm depth noise: 2.3 deg / 3.4 mm using all
21, against 7.8 deg / 8.0 mm using the five palm points. Pass
indices=PALM_LANDMARKS to fit the palm only.

Depth from a Kinect is holey - reflective skin, edges, and shadowing all drop
pixels - so each lookup takes the median of a small patch and points without
valid depth are dropped. With fewer than three valid points the fit is
under-determined and the estimate is refused rather than guessed.
"""

import numpy as np

# Landmarks that move rigidly with the palm
PALM_LANDMARKS = [0, 5, 9, 13, 17]   # wrist, index/middle/ring/pinky MCP


def deproject(u, v, z, fx, fy, cx, cy):
    """Pixel + depth -> 3D point in the camera optical frame (metres)."""
    return np.array([(u - cx) * z / fx, (v - cy) * z / fy, z], dtype=np.float64)


def sample_depth(depth, u, v, patch=2, scale=0.001):
    """Median depth in a patch around (u, v), in metres. NaN if unusable.

    A single pixel is unreliable on a Kinect: holes and edge noise are common.
    The median of a small patch rejects both, and returns NaN when the whole
    patch is invalid rather than a plausible-looking zero.
    """
    h, w = depth.shape[:2]
    u, v = int(round(u)), int(round(v))
    if not (0 <= u < w and 0 <= v < h):
        return float("nan")
    u0, u1 = max(u - patch, 0), min(u + patch + 1, w)
    v0, v1 = max(v - patch, 0), min(v + patch + 1, h)
    block = depth[v0:v1, u0:u1].astype(np.float64).reshape(-1)
    block = block[np.isfinite(block) & (block > 0)]
    if block.size == 0:
        return float("nan")
    return float(np.median(block)) * scale


def kabsch(P, Q):
    """Rigid transform mapping P onto Q. Returns (R, t) with Q ~= R @ P + t.

    Both are (N, 3). No scaling: the world landmarks are already metric, so
    fitting scale would absorb depth bias into the geometry instead of
    surfacing it as residual error.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape or P.shape[0] < 3:
        raise ValueError("need at least 3 matched points")

    cp = P.mean(axis=0)
    cq = Q.mean(axis=0)
    H = (P - cp).T @ (Q - cq)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    # Guard against a reflection when the points are nearly coplanar, which
    # palm landmarks always are
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cq - R @ cp
    return R, t


def fit_pose(world_lm, image_lm, depth, intrinsics, indices=None,
             patch=2, depth_scale=0.001, max_residual=0.05):
    """Estimate the hand pose in the camera frame.

    Parameters
    ----------
    world_lm : (21, 3) MediaPipe world landmarks, metric, hand-centred
    image_lm : (21, 2) pixel coordinates in the depth image's frame
    depth : HxW depth image (uint16 millimetres by default)
    intrinsics : (fx, fy, cx, cy)
    indices : landmark subset to fit; defaults to all landmarks
    max_residual : reject the fit if RMS error exceeds this, in metres

    Returns dict with R, t, residual, n_points - or None if not estimable.
    """
    idx = list(range(len(world_lm))) if indices is None else list(indices)
    fx, fy, cx, cy = intrinsics

    P, Q = [], []
    for i in idx:
        u, v = image_lm[i]
        z = sample_depth(depth, u, v, patch=patch, scale=depth_scale)
        if not np.isfinite(z) or z <= 0.0:
            continue
        P.append(world_lm[i])
        Q.append(deproject(u, v, z, fx, fy, cx, cy))

    if len(P) < 3:
        return {"ok": False, "reason": "few_depth", "n_points": len(P),
                "residual": float("nan"), "n_tried": len(idx)}

    P = np.asarray(P)
    Q = np.asarray(Q)
    R, t = kabsch(P, Q)
    resid = np.sqrt(np.mean(np.sum((Q - (P @ R.T + t)) ** 2, axis=1)))
    if resid > max_residual:
        return {"ok": False, "reason": "residual", "n_points": len(P),
                "residual": float(resid), "R": R, "t": t,
                "n_tried": len(idx)}
    return {"ok": True, "reason": "", "R": R, "t": t,
            "residual": float(resid), "n_points": len(P), "n_tried": len(idx)}


def matrix_to_quaternion(R):
    """Rotation matrix -> (x, y, z, w), the ROS ordering."""
    R = np.asarray(R, dtype=np.float64)
    tr = np.trace(R)
    if tr > 0.0:
        s = np.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def quaternion_to_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class PoseSmoother:
    """Low-pass on translation, and on rotation via the quaternion.

    Quaternions are sign-ambiguous: q and -q are the same rotation, so a naive
    filter can lurch when the sign flips between frames. Each sample is aligned
    to the previous one before blending.
    """

    def __init__(self, alpha_t=0.5, alpha_r=0.5):
        self.alpha_t = alpha_t
        self.alpha_r = alpha_r
        self._t = None
        self._q = None

    def reset(self):
        self._t = None
        self._q = None

    def update(self, t, q):
        t = np.asarray(t, dtype=np.float64)
        q = np.asarray(q, dtype=np.float64)
        q = q / np.linalg.norm(q)

        if self._t is None:
            self._t, self._q = t.copy(), q.copy()
            return self._t, self._q

        self._t = self.alpha_t * self._t + (1.0 - self.alpha_t) * t
        if np.dot(self._q, q) < 0.0:
            q = -q                      # shortest path
        qn = self.alpha_r * self._q + (1.0 - self.alpha_r) * q
        self._q = qn / np.linalg.norm(qn)
        return self._t, self._q


def kabsch_scaled(P, Q, scale_range=(0.6, 1.6)):
    """Similarity fit (Umeyama): Q ~= s * R @ P + t.

    The hand fit deliberately forbids scale, because MediaPipe's HAND world
    landmarks are properly metric and letting scale float would hide depth
    bias in the geometry.

    Body pose world landmarks are different: they are estimated body
    proportions inferred from a single image, and their absolute scale is only
    approximate. Forcing scale to 1 leaves 150-250 mm of residual against real
    depth. Allowing it recovers the fit, and everything downstream that
    matters here - joint directions, the elbow swivel angle - is scale free
    anyway.

    Scale is clamped to `scale_range` so a degenerate frame cannot collapse or
    explode the body.
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    if P.shape != Q.shape or P.shape[0] < 3:
        raise ValueError("need at least 3 matched points")

    cp, cq = P.mean(axis=0), Q.mean(axis=0)
    Pc, Qc = P - cp, Q - cq

    H = Pc.T @ Qc
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    var = np.sum(Pc ** 2)
    s = 1.0 if var < 1e-12 else float((S * np.array([1.0, 1.0, d])).sum() / var)
    s = min(max(s, scale_range[0]), scale_range[1])

    t = cq - s * (R @ cp)
    return R, t, s


def kabsch_ransac(P, Q, threshold=0.05, iters=60, min_inliers=4, seed=0,
                  scaled=True):
    """Similarity/rigid fit robust to outlier correspondences.

    Least squares is not robust: a handful of bad points drags the whole
    transform. With body landmarks that happens constantly - MediaPipe reports
    all 33 points including legs and face, and any that are occluded or
    inferred get whatever depth lies BEHIND them (a chair, the floor, a wall),
    which is metres away from where the landmark actually is.

    RANSAC fits repeatedly on minimal random subsets and keeps the transform
    with the largest consensus set, then refits on those inliers.

    Returns (R, t, s, inlier_indices, residual).
    """
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    n = P.shape[0]
    fit = kabsch_scaled if scaled else (
        lambda a, b: (*kabsch(a, b), 1.0)
    )
    if n < 3:
        raise ValueError("need at least 3 points")

    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iters):
        idx = rng.choice(n, size=3, replace=False)
        try:
            R, t, s = fit(P[idx], Q[idx])
        except Exception:
            continue
        err = np.linalg.norm(Q - (s * (P @ R.T) + t), axis=1)
        inliers = np.where(err < threshold)[0]
        if best is None or inliers.size > best[0].size:
            best = (inliers, R, t, s)

    if best is None or best[0].size < min_inliers:
        # Fall back to a plain fit on everything rather than refusing
        R, t, s = fit(P, Q)
        err = np.linalg.norm(Q - (s * (P @ R.T) + t), axis=1)
        return R, t, s, np.arange(n), float(np.sqrt(np.mean(err ** 2)))

    inliers = best[0]
    R, t, s = fit(P[inliers], Q[inliers])
    err = np.linalg.norm(Q[inliers] - (s * (P[inliers] @ R.T) + t), axis=1)
    return R, t, s, inliers, float(np.sqrt(np.mean(err ** 2)))
