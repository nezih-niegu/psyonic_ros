"""Hand pose -> xArm Lite 6 wrist target -> joint angles.

    /ability_hand/<side>/hand_pose   (camera frame, from hand_pose_node)
              |
              |  T_base_cam       hand-eye calibration
              |  T_hand_flange    mount offset
              |  clutch + scaling
              v
    /xarm/target_pose              PoseStamped in the arm base frame
    /xarm/target_joints            JointState, 6 joints

The Ability Hand keeps doing fingers; the arm carries the wrist. Since the
hand has no wrist DOF of its own, every rotation the operator makes has to be
performed by the arm's last three joints.

Kinematics come from xarm_description's own
config/kinematics/default/lite6_default_kinematics.yaml, so the chain matches
the real robot. IK is damped least squares seeded from the previous solution,
which is well conditioned for teleoperation because consecutive targets are
close together.

Two transforms must be measured, not guessed:

  T_base_cam    where the camera sits in the arm's base frame. This is a
                hand-eye calibration. An error here maps directly into
                every command.
  T_hand_flange how the Ability Hand is bolted to the flange, from the
                mechanical drawing.

Both default to identity, which is certainly wrong for your setup.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray

from ah_mujoco.arm_teleop import (
    LITE6_LIMITS,
    ik_posture,
    LITE6_ORIGINS,
    ClutchedPoseMapper,
    UrdfChain,
    R_to_quat,
    clamp_workspace,
    make_T,
    quat_to_R,
    rpy_to_R,
)

JOINT_NAMES = [f"joint{i+1}" for i in range(6)]


class ArmTeleopNode(Node):
    def __init__(self):
        super().__init__("arm_teleop_node")
        self.declare_parameter("hand_side", "Right")
        # Hand-eye: camera pose in the arm base frame
        self.declare_parameter("base_cam_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("base_cam_rpy", [0.0, 0.0, 0.0])
        # Mount: flange pose in the tracked-hand frame
        self.declare_parameter("hand_flange_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("hand_flange_rpy", [0.0, 0.0, 0.0])
        self.declare_parameter("position_scale", 1.0)
        self.declare_parameter("rotation_scale", 1.0)
        self.declare_parameter("mapping_frame", "base")   # "base" or "tool"
        # Shoulder anchoring: express the hand pose relative to the operator's
        # tracked shoulder instead of the camera, so leaning or stepping
        # sideways does not drag the robot. This is the main practical use of
        # upper-limb tracking; the elbow itself cannot be commanded (see
        # upper_limb_node for the measurements).
        self.declare_parameter("anchor_to_shoulder", False)
        # Elbow swivel as a weighted IK objective. Zero by default: bringing
        # swivel error from 22 to 7 degrees costs 93 mm of wrist error on this
        # arm, which is a bad trade for manipulation.
        self.declare_parameter("elbow_weight", 0.0)
        # Workspace box in the base frame; a hard limit on where the target
        # may go regardless of what the tracker reports
        self.declare_parameter("box_min", [-0.5, -0.5, 0.05])
        self.declare_parameter("box_max", [0.5, 0.5, 0.7])
        self.declare_parameter("max_joint_step", 0.05)   # rad per solve
        self.declare_parameter("publish_hz", 50.0)
        # Must match xarm_bridge's safe_pose, or the first clutch engage
        # solves from a configuration the arm is not actually in.
        self.declare_parameter(
            "start_joints", [0.0, -0.5236, 0.8727, 0.0, 0.6981, 0.0]
        )
        # Track the arm's real joints when available, so IK is always seeded
        # from where the robot actually is.
        self.declare_parameter("joint_states_topic", "/ufactory/joint_states")

        g = self.get_parameter
        side = g("hand_side").get_parameter_value().string_value.lower()
        self.max_joint_step = (
            g("max_joint_step").get_parameter_value().double_value
        )

        def vec(name):
            return np.asarray(
                g(name).get_parameter_value().double_array_value, dtype=float
            )

        bc_xyz, bc_rpy = vec("base_cam_xyz"), vec("base_cam_rpy")
        hf_xyz, hf_rpy = vec("hand_flange_xyz"), vec("hand_flange_rpy")
        self.T_base_cam = make_T(rpy_to_R(*bc_rpy), bc_xyz)
        self.T_hand_flange = make_T(rpy_to_R(*hf_rpy), hf_xyz)
        self.box_min, self.box_max = vec("box_min"), vec("box_max")

        if not np.any(bc_xyz) and not np.any(bc_rpy):
            self.get_logger().warn(
                "base_cam transform is identity - the camera pose in the arm "
                "base frame has not been calibrated. Targets will be wrong "
                "until it is."
            )

        self.arm = UrdfChain(LITE6_ORIGINS, LITE6_LIMITS)
        self.q = np.asarray(
            g("start_joints").get_parameter_value().double_array_value,
            dtype=float,
        )
        if self.q.size != 6:
            self.q = np.zeros(6)

        self.mapper = ClutchedPoseMapper(
            position_scale=g("position_scale").get_parameter_value().double_value,
            rotation_scale=g("rotation_scale").get_parameter_value().double_value,
            frame=g("mapping_frame").get_parameter_value().string_value.lower(),
        )

        self._T_hand = None
        self._shoulder = None
        self._human_swivel = None
        self.anchor_to_shoulder = (
            g("anchor_to_shoulder").get_parameter_value().bool_value
        )
        self.elbow_weight = g("elbow_weight").get_parameter_value().double_value
        self.create_subscription(
            PoseStamped, f"/ability_hand/{side}/hand_pose", self._pose_cb, 10
        )
        self.create_subscription(
            Bool, f"/ability_hand/{side}/clutch", self._clutch_cb, 10
        )
        self.create_subscription(
            Float32MultiArray, f"/ability_hand/{side}/upper_limb",
            self._limb_cb, 10,
        )
        self.create_subscription(
            Float32MultiArray, f"/ability_hand/{side}/upper_limb/swivel",
            self._swivel_cb, 10,
        )
        js_topic = g("joint_states_topic").get_parameter_value().string_value
        if js_topic:
            self.create_subscription(
                JointState, js_topic, self._arm_state_cb, 10
            )
        self._have_state = False

        self.pub_pose = self.create_publisher(PoseStamped, "/xarm/target_pose", 10)
        self.pub_joints = self.create_publisher(
            JointState, "/xarm/target_joints", 10
        )
        self.pub_diag = self.create_publisher(
            Float32MultiArray, "/xarm/ik_diagnostics", 10
        )

        self.create_timer(
            1.0 / max(g("publish_hz").get_parameter_value().double_value, 1.0),
            self._tick,
        )
        self.get_logger().info(
            "Arm teleop ready. Publish True on "
            f"/ability_hand/{side}/clutch to engage; the arm holds when "
            "released."
        )

    # -------------------------------------------------------------- callbacks

    def _pose_cb(self, msg: PoseStamped):
        q = [msg.pose.orientation.x, msg.pose.orientation.y,
             msg.pose.orientation.z, msg.pose.orientation.w]
        t = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        T_cam_hand = make_T(quat_to_R(q), np.asarray(t))

        if self.anchor_to_shoulder and self._shoulder is not None:
            # Subtract the shoulder so the pose is "hand relative to the
            # operator's own body" rather than "hand relative to the camera".
            # Orientation is untouched: only the origin moves.
            T_cam_hand = T_cam_hand.copy()
            T_cam_hand[:3, 3] = T_cam_hand[:3, 3] - self._shoulder

        # Into the arm base frame, then out to the flange
        self._T_hand = self.T_base_cam @ T_cam_hand @ self.T_hand_flange

    def _arm_state_cb(self, msg: JointState):
        """Real joint feedback. Used only while the clutch is released, so the
        target does not fight the controller mid-motion."""
        if self.mapper.engaged or len(msg.position) < 6:
            return
        idx = {n: i for i, n in enumerate(msg.name)}
        q = []
        for n in JOINT_NAMES:
            if n in idx:
                q.append(msg.position[idx[n]])
        if len(q) == 6:
            self.q = np.asarray(q, dtype=float)
            self._have_state = True

    def _limb_cb(self, msg):
        d = np.asarray(msg.data, dtype=float)
        if d.size >= 9:
            self._shoulder = d[:3]          # camera frame

    def _swivel_cb(self, msg):
        d = list(msg.data)
        if len(d) >= 2 and d[1] > 0.5:
            self._human_swivel = float(d[0])

    def _clutch_cb(self, msg: Bool):
        if msg.data and not self.mapper.engaged:
            if self._T_hand is None:
                self.get_logger().warn("cannot engage: no hand pose yet")
                return
            self.mapper.engage(self._T_hand, self.arm.fk(self.q))
            src = "robot feedback" if self._have_state else "start_joints"
            self.get_logger().info(f"clutch ENGAGED (seeded from {src})")
        elif not msg.data and self.mapper.engaged:
            self.mapper.release()
            self.get_logger().info("clutch released - arm holding")

    # ------------------------------------------------------------------- loop

    def _tick(self):
        if self._T_hand is None or self.mapper.target is None:
            return

        T_target = self.mapper.update(self._T_hand)
        T_target = clamp_workspace(T_target, self.box_min, self.box_max)

        if self.elbow_weight > 0.0 and self._human_swivel is not None:
            q_new, converged, ep, er, sw = ik_posture(
                self.arm, T_target, self.q,
                swivel_target=self._human_swivel,
                elbow_weight=self.elbow_weight,
            )
        else:
            q_new, converged, ep, er = self.arm.ik(T_target, self.q)

        # Rate limit in joint space: an unreachable target otherwise produces
        # a large jump the moment IK finds a distant branch.
        dq = q_new - self.q
        n = np.max(np.abs(dq))
        if n > self.max_joint_step:
            dq *= self.max_joint_step / n
        self.q = np.clip(
            self.q + dq,
            self.arm.limits[:, 0], self.arm.limits[:, 1],
        )

        if not converged:
            self.get_logger().warn(
                f"IK did not converge: {ep*1000:.1f} mm, "
                f"{np.degrees(er):.1f} deg. Target may be out of reach.",
                throttle_duration_sec=2.0,
            )

        stamp = self.get_clock().now().to_msg()

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = "link_base"
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = (
            float(v) for v in T_target[:3, 3]
        )
        qq = R_to_quat(T_target[:3, :3])
        (pose.pose.orientation.x, pose.pose.orientation.y,
         pose.pose.orientation.z, pose.pose.orientation.w) = (
            float(v) for v in qq)
        self.pub_pose.publish(pose)

        js = JointState()
        js.header.stamp = stamp
        js.name = JOINT_NAMES
        js.position = [float(v) for v in self.q]
        self.pub_joints.publish(js)

        diag = Float32MultiArray()
        diag.data = [float(ep), float(er), float(converged)]
        self.pub_diag.publish(diag)


def main(args=None):
    rclpy.init(args=args)
    node = ArmTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
