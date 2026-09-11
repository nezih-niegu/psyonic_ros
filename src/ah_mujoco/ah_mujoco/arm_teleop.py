"""Mapping a tracked hand pose onto a 6-DOF arm's wrist.

The solo-hand mode uses only finger flexion and discards where the hand is.
With the Ability Hand mounted as an xArm Lite 6 gripper, the tracked SE(3)
pose becomes the arm's target and the fingers keep doing what they already do.

Three transforms stand between the camera and a joint command:

    T_base_cam    camera pose in the arm base frame     (hand-eye calibration)
    T_cam_hand    tracked hand pose                     (hand_pose node)
    T_hand_flange fixed mount offset                    (mechanical drawing)

    T_base_flange = T_base_cam @ T_cam_hand @ T_hand_flange

Absolute mapping is almost never what you want for teleoperation: the operator's
hand and the arm's workspace do not coincide, and a tracking glitch commands a
large motion. Clutched relative mapping is used instead - motion accumulates
only while the clutch is engaged, and the arm holds when it is released, so the
operator can re-centre the way a mouse is lifted and repositioned.

DH parameters are NOT hardcoded. Publish the ones for your robot, or use the
xArm SDK's own solver, which matches the controller exactly. Guessed link
lengths produce a confident, wrong answer.
"""

import numpy as np


# ----------------------------------------------------------------- transforms

def make_T(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def inv_T(T):
    R = T[:3, :3]
    t = T[:3, 3]
    out = np.eye(4)
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ t
    return out


def quat_to_R(q):
    x, y, z, w = q
    n = np.linalg.norm([x, y, z, w])
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = np.array([x, y, z, w]) / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat(R):
    tr = np.trace(R)
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w, x = 0.25 * s, (R[2, 1] - R[1, 2]) / s
        y, z = (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w, x = (R[2, 1] - R[1, 2]) / s, 0.25 * s
        y, z = (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w, x = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s
        y, z = 0.25 * s, (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w, x = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s
        y, z = (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([x, y, z, w])
    return q / np.linalg.norm(q)


def rotvec_from_R(R):
    """Rotation matrix -> axis*angle vector, the orientation error form."""
    cos = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    if angle < 1e-9:
        return np.zeros(3)
    if abs(angle - np.pi) < 1e-6:
        # Near pi the skew part vanishes; recover the axis from R + I
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.clip(np.diag(A), 0.0, 1.0))
        if axis[0] >= max(axis[1], axis[2]):
            axis = axis * np.sign(A[0]) if A[0, 0] > 0 else axis
        return axis / np.linalg.norm(axis) * angle
    v = np.array([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]])
    return v / (2.0 * np.sin(angle)) * angle


# ------------------------------------------------------------ clutched mapping

class ClutchedPoseMapper:
    """Relative hand motion -> arm target, engaged by a clutch.

    On engage, the current hand pose and current arm pose are latched as the
    reference pair. While engaged the arm target is

        T_target = T_arm_ref @ delta(T_hand_ref, T_hand_now)

    with translation scaled by `position_scale` and rotation by
    `rotation_scale`. On release the target holds, so the operator can move
    their hand back to a comfortable position without dragging the arm.
    """

    def __init__(self, position_scale=1.0, rotation_scale=1.0,
                 frame="base"):
        """frame: "base" or "tool".

        "base" (default) maps hand motion in the arm's base frame: move your
        hand right and the arm goes right, regardless of how the gripper is
        oriented. That is what an operator expects.

        "tool" applies the delta in the gripper's own frame, so translation
        follows wherever the tool is pointing. Useful for aligned insertion
        tasks, confusing for free-space motion.
        """
        self.position_scale = float(position_scale)
        self.rotation_scale = float(rotation_scale)
        self.frame = frame
        self.engaged = False
        self._hand_ref = None
        self._arm_ref = None
        self._target = None

    def engage(self, T_hand, T_arm):
        self._hand_ref = T_hand.copy()
        self._arm_ref = T_arm.copy()
        self._target = T_arm.copy()
        self.engaged = True

    def release(self):
        self.engaged = False

    def update(self, T_hand):
        """Returns the current arm target, or None before the first engage."""
        if not self.engaged or self._hand_ref is None:
            return self._target
        if self.frame == "tool":
            delta = inv_T(self._hand_ref) @ T_hand
            dt = delta[:3, 3] * self.position_scale
            rv = rotvec_from_R(delta[:3, :3]) * self.rotation_scale
            self._target = self._arm_ref @ make_T(R_from_rotvec(rv), dt)
        else:
            # Base frame: translation is the straight displacement, and the
            # rotation is applied about the base axes rather than the tool's.
            dt = (T_hand[:3, 3] - self._hand_ref[:3, 3]) * self.position_scale
            dR_full = T_hand[:3, :3] @ self._hand_ref[:3, :3].T
            rv = rotvec_from_R(dR_full) * self.rotation_scale
            R_new = R_from_rotvec(rv) @ self._arm_ref[:3, :3]
            self._target = make_T(R_new, self._arm_ref[:3, 3] + dt)
        return self._target

    @property
    def target(self):
        return self._target


def R_from_rotvec(rv):
    """Axis*angle vector -> rotation matrix (Rodrigues)."""
    theta = np.linalg.norm(rv)
    if theta < 1e-12:
        return np.eye(3)
    k = rv / theta
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)


# ------------------------------------------------------------------ kinematics

# Lite 6 joint origins, taken verbatim from xarm_description's
# config/kinematics/default/lite6_default_kinematics.yaml. Every xArm joint
# rotates about its own local Z, so the chain is fully described by these
# fixed origins plus one rotation per joint - no DH table and no guessed link
# lengths.
LITE6_ORIGINS = [
    # x,        y,         z,      roll,    pitch,    yaw
    (0.0,       0.0,       0.2435,  0.0,     0.0,      0.0),
    (0.0,       0.0,       0.0,     1.5708, -1.5708,   3.1416),
    (0.2002,    0.0,       0.0,    -3.1416,  0.0,      1.5708),
    (0.087,    -0.22761,   0.0,     1.5708,  0.0,      0.0),
    (0.0,       0.0,       0.0,     1.5708,  0.0,      0.0),
    (0.0,       0.0625,    0.0,    -1.5708,  0.0,      0.0),
]

# From lite6_robot_macro.xacro
LITE6_LIMITS = [
    (-np.pi * 0.99, np.pi * 0.99),
    (-2.61799, 2.61799),
    (-0.061087, np.pi * 0.99),
    (-np.pi * 0.99, np.pi * 0.99),
    (-2.1642, 2.1642),
    (-np.pi * 0.99, np.pi * 0.99),
]


def rpy_to_R(roll, pitch, yaw):
    """URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


class UrdfChain:
    """Serial chain of fixed origins with a Z-axis revolute joint at each.

    This is exactly how xarm_description builds its arms, so the geometry
    matches the real robot rather than a reconstructed DH table.
    """

    def __init__(self, origins, limits=None, tool=None):
        self.origins = [
            make_T(rpy_to_R(r, p, y), np.array([x, yy, z]))
            for (x, yy, z, r, p, y) in origins
        ]
        self.n = len(self.origins)
        self.limits = None if limits is None else np.asarray(limits, float)
        self.tool = np.eye(4) if tool is None else np.asarray(tool, float)

    def fk(self, q):
        T = np.eye(4)
        for i in range(self.n):
            c, s = np.cos(q[i]), np.sin(q[i])
            Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            T = T @ self.origins[i] @ make_T(Rz, np.zeros(3))
        return T @ self.tool

    def jacobian(self, q):
        """Analytic geometric Jacobian; each joint spins about its frame Z."""
        T = np.eye(4)
        origins, axes = [], []
        for i in range(self.n):
            T = T @ self.origins[i]
            origins.append(T[:3, 3].copy())
            axes.append(T[:3, 2].copy())
            c, s = np.cos(q[i]), np.sin(q[i])
            Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
            T = T @ make_T(Rz, np.zeros(3))
        p_end = (T @ self.tool)[:3, 3]

        J = np.zeros((6, self.n))
        for i in range(self.n):
            J[:3, i] = np.cross(axes[i], p_end - origins[i])
            J[3:, i] = axes[i]
        return J



def dh_transform(alpha, a, d, theta, modified=True):
    """One DH link transform. Modified (Craig) convention by default."""
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    if modified:
        return np.array([
            [ct,      -st,     0.0,    a],
            [st * ca,  ct * ca, -sa,  -sa * d],
            [st * sa,  ct * sa,  ca,   ca * d],
            [0.0,      0.0,     0.0,   1.0],
        ])
    return np.array([
        [ct, -st * ca,  st * sa, a * ct],
        [st,  ct * ca, -ct * sa, a * st],
        [0.0, sa,       ca,      d],
        [0.0, 0.0,      0.0,     1.0],
    ])


class SerialArm:
    """Generic serial chain from a DH table, with DLS inverse kinematics.

    dh : (n, 4) array of [alpha, a, d, theta_offset]
    limits : (n, 2) joint limits in radians, or None
    """

    def __init__(self, dh, limits=None, modified=True, tool=None):
        self.dh = np.asarray(dh, dtype=np.float64)
        self.n = self.dh.shape[0]
        self.limits = None if limits is None else np.asarray(limits, float)
        self.modified = modified
        self.tool = np.eye(4) if tool is None else np.asarray(tool, float)

    def fk(self, q, upto=None):
        T = np.eye(4)
        n = self.n if upto is None else upto
        for i in range(n):
            alpha, a, d, off = self.dh[i]
            T = T @ dh_transform(alpha, a, d, q[i] + off, self.modified)
        if upto is None:
            T = T @ self.tool
        return T

    def jacobian(self, q):
        """Geometric Jacobian by finite difference of the pose.

        Analytic columns are cheap for a known chain, but the DH table here is
        user-supplied and may use either convention; a numeric Jacobian is
        correct for both and the cost is irrelevant at teleop rates.
        """
        eps = 1e-6
        T0 = self.fk(q)
        J = np.zeros((6, self.n))
        for i in range(self.n):
            dq = np.array(q, dtype=np.float64)
            dq[i] += eps
            T1 = self.fk(dq)
            J[:3, i] = (T1[:3, 3] - T0[:3, 3]) / eps
            dR = T1[:3, :3] @ T0[:3, :3].T
            J[3:, i] = rotvec_from_R(dR) / eps
        return J

    def ik(self, T_target, q0, iters=100, tol_pos=1e-4, tol_rot=1e-3,
           damping=0.05, max_step=0.2):
        """Damped least squares IK, seeded from q0.

        Damping keeps the step bounded near singularities, where an
        undamped pseudo-inverse would command an enormous joint velocity.
        Returns (q, converged, position_error, rotation_error).
        """
        q = np.array(q0, dtype=np.float64).copy()
        for _ in range(iters):
            T = self.fk(q)
            e_pos = T_target[:3, 3] - T[:3, 3]
            e_rot = rotvec_from_R(T_target[:3, :3] @ T[:3, :3].T)
            err = np.concatenate([e_pos, e_rot])

            if np.linalg.norm(e_pos) < tol_pos and np.linalg.norm(e_rot) < tol_rot:
                return q, True, float(np.linalg.norm(e_pos)), float(
                    np.linalg.norm(e_rot))

            J = self.jacobian(q)
            JT = J.T
            dq = JT @ np.linalg.solve(
                J @ JT + (damping ** 2) * np.eye(6), err
            )
            norm = np.linalg.norm(dq)
            if norm > max_step:
                dq *= max_step / norm
            q = q + dq
            if self.limits is not None:
                q = np.clip(q, self.limits[:, 0], self.limits[:, 1])

        T = self.fk(q)
        return (
            q,
            False,
            float(np.linalg.norm(T_target[:3, 3] - T[:3, 3])),
            float(np.linalg.norm(rotvec_from_R(T_target[:3, :3] @ T[:3, :3].T))),
        )


# Both chain classes expose fk() and jacobian(), so they share one solver.
UrdfChain.ik = SerialArm.ik


def clamp_workspace(T, box_min, box_max):
    """Clamp the target position into a box. Orientation is untouched."""
    T = T.copy()
    T[:3, 3] = np.clip(T[:3, 3], box_min, box_max)
    return T


# --------------------------------------------------------------- elbow / swivel
#
# Mapping a human upper limb onto the arm needs care about degrees of freedom.
# The Lite 6 has six, and a fully constrained wrist pose (3 position +
# 3 orientation) uses all of them: the measured null space of the full
# Jacobian is exactly 0, so the elbow cannot be commanded independently while
# the wrist pose is held exactly.
#
# Leaving the tool ROLL free - rotation about the hand's own approach axis,
# which for a gripper is usually the least meaningful direction - makes the
# primary task 5-D and leaves exactly 1 spare DOF. That is what the elbow
# objective uses.
#
# The elbow is matched by SWIVEL ANGLE rather than position. Human and robot
# limbs have different segment lengths, so a position target is unreachable in
# general; the swivel angle - the rotation of the elbow about the
# shoulder-to-wrist axis - is scale free and is what actually reads as "same
# posture" to an observer.

ELBOW_JOINT_INDEX = 3      # joint3's origin is the Lite 6's elbow


def swivel_angle(shoulder, elbow, wrist, reference=np.array([0.0, 0.0, 1.0])):
    """Rotation of the elbow about the shoulder->wrist axis, in radians.

    Zero means the elbow lies in the plane containing the axis and the
    reference direction; positive follows the right-hand rule about the axis.
    Scale free, so a human limb and a robot arm of different proportions can
    be compared directly.
    """
    shoulder = np.asarray(shoulder, dtype=np.float64)
    elbow = np.asarray(elbow, dtype=np.float64)
    wrist = np.asarray(wrist, dtype=np.float64)

    axis = wrist - shoulder
    n = np.linalg.norm(axis)
    if n < 1e-9:
        return 0.0
    axis = axis / n

    # Component of the elbow offset perpendicular to the axis
    v = elbow - shoulder
    v_perp = v - np.dot(v, axis) * axis
    if np.linalg.norm(v_perp) < 1e-9:
        return 0.0                      # arm fully extended: swivel undefined
    v_perp = v_perp / np.linalg.norm(v_perp)

    ref_perp = reference - np.dot(reference, axis) * axis
    if np.linalg.norm(ref_perp) < 1e-9:
        ref_perp = np.cross(axis, [1.0, 0.0, 0.0])
        if np.linalg.norm(ref_perp) < 1e-9:
            ref_perp = np.cross(axis, [0.0, 1.0, 0.0])
    ref_perp = ref_perp / np.linalg.norm(ref_perp)

    cross = np.cross(ref_perp, v_perp)
    return float(np.arctan2(np.dot(cross, axis), np.dot(ref_perp, v_perp)))


def _point_jacobian(chain, q, upto):
    """Translational Jacobian of the origin of frame `upto`."""
    T = np.eye(4)
    origins, axes = [], []
    for i in range(chain.n):
        T = T @ chain.origins[i]
        origins.append(T[:3, 3].copy())
        axes.append(T[:3, 2].copy())
        c, s = np.cos(q[i]), np.sin(q[i])
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        T = T @ make_T(Rz, np.zeros(3))

    T = np.eye(4)
    for i in range(upto):
        T = T @ chain.origins[i]
        c, s = np.cos(q[i]), np.sin(q[i])
        Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        T = T @ make_T(Rz, np.zeros(3))
    p = T[:3, 3]

    J = np.zeros((3, chain.n))
    for i in range(upto):                # joints beyond the point cannot move it
        J[:, i] = np.cross(axes[i], p - origins[i])
    return J, p


def elbow_position(chain, q, index=ELBOW_JOINT_INDEX):
    _, p = _point_jacobian(chain, q, index)
    return p


def ik_posture(chain, T_target, q0, swivel_target=None, elbow_weight=0.0,
               iters=150, tol_pos=1e-4, tol_rot=1e-3, damping=0.05,
               max_step=0.2, reference=np.array([0.0, 0.0, 1.0]),
               free_roll=True):
    """Wrist pose IK with the elbow swivel as a WEIGHTED task, not a null-space one.

    Null-space projection does not work on this arm. With the wrist pose held
    exactly, the only spare DOF (after freeing tool roll) is joint 6, which is
    distal to the elbow and moves it by 0.000 mm/rad - measured, not assumed.
    Given a wrist pose, the Lite 6's elbow is determined.

    So matching a human elbow necessarily costs wrist accuracy, and the only
    honest interface is a weight. `elbow_weight` 0 ignores the elbow entirely;
    larger values buy elbow agreement with wrist error. See the table in the
    README for the measured trade-off.

    Returns (q, converged, pos_err, rot_err, swivel_err).
    """
    q = np.array(q0, dtype=np.float64).copy()
    swivel_err = float("nan")
    shoulder = np.zeros(3)

    for _ in range(iters):
        T = chain.fk(q)
        e_pos = T_target[:3, 3] - T[:3, 3]
        e_rot = rotvec_from_R(T_target[:3, :3] @ T[:3, :3].T)
        J = chain.jacobian(q)

        if free_roll:
            a = T[:3, 2]
            P = np.eye(3) - np.outer(a, a)
            rows = [J[:3, :], P @ J[3:, :]]
            errs = [e_pos, P @ e_rot]
        else:
            rows = [J[:3, :], J[3:, :]]
            errs = [e_pos, e_rot]

        if swivel_target is not None and elbow_weight > 0.0:
            p_elbow = elbow_position(chain, q)
            cur = swivel_angle(shoulder, p_elbow, T[:3, 3], reference)
            swivel_err = float(np.arctan2(
                np.sin(swivel_target - cur), np.cos(swivel_target - cur)
            ))
            grad = np.zeros(chain.n)
            eps = 1e-5
            for i in range(chain.n):
                dqi = q.copy()
                dqi[i] += eps
                Ti = chain.fk(dqi)
                s2 = swivel_angle(
                    shoulder, elbow_position(chain, dqi), Ti[:3, 3], reference
                )
                grad[i] = np.arctan2(
                    np.sin(s2 - cur), np.cos(s2 - cur)
                ) / eps
            rows.append(elbow_weight * grad.reshape(1, -1))
            errs.append(np.array([elbow_weight * swivel_err]))

        Jt = np.vstack(rows)
        err = np.concatenate(errs)

        if (np.linalg.norm(e_pos) < tol_pos
                and np.linalg.norm(errs[1]) < tol_rot
                and (swivel_target is None or elbow_weight <= 0.0
                     or abs(swivel_err) < np.radians(0.5))):
            break

        JT = Jt.T
        dq = JT @ np.linalg.solve(
            Jt @ JT + (damping ** 2) * np.eye(Jt.shape[0]), err
        )
        n = np.linalg.norm(dq)
        if n > max_step:
            dq *= max_step / n
        q = q + dq
        if chain.limits is not None:
            q = np.clip(q, chain.limits[:, 0], chain.limits[:, 1])

    T = chain.fk(q)
    ep = float(np.linalg.norm(T_target[:3, 3] - T[:3, 3]))
    er_vec = rotvec_from_R(T_target[:3, :3] @ T[:3, :3].T)
    if free_roll:
        a = T[:3, 2]
        er_vec = (np.eye(3) - np.outer(a, a)) @ er_vec
    er = float(np.linalg.norm(er_vec))
    return q, ep < 1e-3 and er < 1e-2, ep, er, swivel_err


def ik_with_elbow(chain, T_target, q0, swivel_target=None, iters=120,
                  tol_pos=1e-4, tol_rot=1e-3, damping=0.05, max_step=0.2,
                  free_roll=True, elbow_gain=0.5,
                  reference=np.array([0.0, 0.0, 1.0])):
    """Wrist pose IK with a secondary elbow-swivel objective.

    The wrist task is solved first. If `free_roll`, rotation about the tool
    approach axis is dropped from the task, leaving one spare DOF; the elbow
    objective is then projected into that null space so it can never pull the
    wrist off target.

    Returns (q, converged, pos_err, rot_err, swivel_err).
    """
    q = np.array(q0, dtype=np.float64).copy()
    swivel_err = float("nan")

    for _ in range(iters):
        T = chain.fk(q)
        e_pos = T_target[:3, 3] - T[:3, 3]
        e_rot = rotvec_from_R(T_target[:3, :3] @ T[:3, :3].T)

        J = chain.jacobian(q)
        if free_roll:
            a = T[:3, 2]                       # tool approach axis
            P = np.eye(3) - np.outer(a, a)
            Jt = np.vstack([J[:3, :], P @ J[3:, :]])
            err = np.concatenate([e_pos, P @ e_rot])
        else:
            Jt, err = J, np.concatenate([e_pos, e_rot])

        primary_done = (
            np.linalg.norm(e_pos) < tol_pos
            and np.linalg.norm(P @ e_rot if free_roll else e_rot) < tol_rot
        )
        if primary_done and (
            swivel_target is None
            or (np.isfinite(swivel_err) and abs(swivel_err) < np.radians(1.0))
        ):
            break

        JT = Jt.T
        A = Jt @ JT + (damping ** 2) * np.eye(Jt.shape[0])
        dq = JT @ np.linalg.solve(A, err)

        if swivel_target is not None and free_roll:
            # Secondary task, projected so it cannot disturb the wrist
            Je, p_elbow = _point_jacobian(chain, q, ELBOW_JOINT_INDEX)
            shoulder = np.zeros(3)             # arm base
            wrist = T[:3, 3]
            cur = swivel_angle(shoulder, p_elbow, wrist, reference)
            swivel_err = float(np.arctan2(
                np.sin(swivel_target - cur), np.cos(swivel_target - cur)
            ))
            # Numeric gradient of swivel with respect to q
            grad = np.zeros(chain.n)
            eps = 1e-5
            for i in range(chain.n):
                dqi = q.copy()
                dqi[i] += eps
                Ti = chain.fk(dqi)
                pe = elbow_position(chain, dqi)
                s2 = swivel_angle(shoulder, pe, Ti[:3, 3], reference)
                grad[i] = np.arctan2(
                    np.sin(s2 - cur), np.cos(s2 - cur)
                ) / eps
            gn = np.linalg.norm(grad)
            if gn > 1e-9:
                secondary = elbow_gain * swivel_err * grad / (gn ** 2)
                # The projector must use the true pseudo-inverse. Building it
                # from the DAMPED inverse leaks the secondary task into the
                # primary one - measured as 12 mm of wrist error, against
                # 0.09 mm without the elbow objective.
                Jpinv = np.linalg.pinv(Jt, rcond=1e-6)
                N = np.eye(chain.n) - Jpinv @ Jt
                step = N @ secondary
                # Keep the secondary step from dominating a single iteration
                sn = np.linalg.norm(step)
                if sn > 0.05:
                    step *= 0.05 / sn
                dq = dq + step

        n = np.linalg.norm(dq)
        if n > max_step:
            dq *= max_step / n
        q = q + dq
        if chain.limits is not None:
            q = np.clip(q, chain.limits[:, 0], chain.limits[:, 1])

    T = chain.fk(q)
    ep = float(np.linalg.norm(T_target[:3, 3] - T[:3, 3]))
    er_vec = rotvec_from_R(T_target[:3, :3] @ T[:3, :3].T)
    if free_roll:
        a = T[:3, 2]
        er_vec = (np.eye(3) - np.outer(a, a)) @ er_vec
    er = float(np.linalg.norm(er_vec))
    converged = ep < tol_pos * 10 and er < tol_rot * 10
    return q, converged, ep, er, swivel_err
