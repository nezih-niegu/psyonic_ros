"""Record a teleoperation session for later analysis.

Subscribes to what the pipeline already publishes and writes one .npz at the
end. Nothing is analysed live - the point is to capture the session cheaply and
think about it afterwards.

    ros2 run ah_mujoco session_recorder --ros-args -p out:=/ws/vendor/session1.npz

Stop it with Ctrl-C and it writes the file. Then:

    ros2 run ah_mujoco session_report --ros-args \\
        -p session:=/ws/vendor/session1.npz \\
        -p out:=/ws/vendor/session1.html

Everything is sampled on one timer rather than on each topic's own callback, so
every channel shares a timebase. Comparing an operator signal against a robot
signal only means something if both were sampled at the same instants, and
topics arriving at 15, 30, 60 and 120 Hz do not give that for free.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float32MultiArray

from ah_messages.msg import Digits


class SessionRecorderNode(Node):
    def __init__(self):
        super().__init__("session_recorder_node")
        self.declare_parameter("hand_side", "Right")
        self.declare_parameter("out", "/ws/vendor/session.npz")
        self.declare_parameter("rate_hz", 60.0)
        self.declare_parameter("max_minutes", 30.0)

        g = self.get_parameter
        side = g("hand_side").get_parameter_value().string_value.lower()
        self.out = g("out").get_parameter_value().string_value
        self.rate = g("rate_hz").get_parameter_value().double_value
        max_min = g("max_minutes").get_parameter_value().double_value
        self.max_samples = int(self.rate * 60 * max_min)

        ns = f"/ability_hand/{side}"
        self._limb = None          # shoulder, elbow, wrist (9)
        self._wrist = None         # SE(3) position from hand_pose
        self._targets = None       # 6 hand joint commands
        self._arm = None           # 6 arm joints
        self._clutch = False
        self._msj = None
        self._swivel = None

        self.create_subscription(
            Float32MultiArray, f"{ns}/upper_limb", self._limb_cb, 10)
        self.create_subscription(
            PoseStamped, f"{ns}/hand_pose", self._wrist_cb, 10)
        self.create_subscription(
            Digits, f"{ns}/target/position", self._target_cb, 10)
        self.create_subscription(
            JointState, "/ufactory/joint_states", self._arm_cb, 10)
        self.create_subscription(
            JointState, "/xarm/target_joints", self._arm_cmd_cb, 10)
        self.create_subscription(Bool, f"{ns}/clutch", self._clutch_cb, 10)
        self.create_subscription(
            Float32MultiArray, f"{ns}/quality/msj", self._msj_cb, 10)
        self.create_subscription(
            Float32MultiArray, f"{ns}/upper_limb/swivel", self._swivel_cb, 10)

        self.rows = []
        self.create_timer(1.0 / max(self.rate, 1.0), self._tick)
        self.get_logger().info(
            f"Recording at {self.rate:.0f} Hz to {self.out}. "
            "Ctrl-C to stop and write."
        )

    # ------------------------------------------------------------- callbacks

    def _limb_cb(self, m):
        d = list(m.data)
        if len(d) >= 9:
            self._limb = d[:9]

    def _wrist_cb(self, m):
        self._wrist = [m.pose.position.x, m.pose.position.y, m.pose.position.z]

    def _target_cb(self, m):
        d = list(m.data)
        if len(d) >= 6:
            self._targets = d[:6]

    def _arm_cb(self, m):
        if len(m.position) >= 6:
            self._arm = [float(v) for v in m.position[:6]]

    def _arm_cmd_cb(self, m):
        # Only used when there is no real feedback
        if self._arm is None and len(m.position) >= 6:
            self._arm = [float(v) for v in m.position[:6]]

    def _clutch_cb(self, m):
        self._clutch = bool(m.data)

    def _msj_cb(self, m):
        d = list(m.data)
        if len(d) >= 6:
            self._msj = d[:6]

    def _swivel_cb(self, m):
        d = list(m.data)
        if d:
            self._swivel = d[0]

    # ------------------------------------------------------------------ loop

    def _tick(self):
        if len(self.rows) >= self.max_samples:
            self.get_logger().warn("recording limit reached", once=True)
            return
        nan3, nan6, nan9 = [np.nan] * 3, [np.nan] * 6, [np.nan] * 9
        t = self.get_clock().now().nanoseconds * 1e-9
        self.rows.append(
            [t]
            + (self._limb or nan9)
            + (self._wrist or nan3)
            + (self._targets or nan6)
            + (self._arm or nan6)
            + [1.0 if self._clutch else 0.0]
            + (self._msj or nan6)
            + [self._swivel if self._swivel is not None else np.nan]
        )

    def write(self):
        if not self.rows:
            self.get_logger().warn("nothing recorded")
            return
        a = np.asarray(self.rows, dtype=float)
        t = a[:, 0] - a[0, 0]
        np.savez_compressed(
            self.out,
            t=t,
            limb=a[:, 1:10],
            wrist=a[:, 10:13],
            hand_targets=a[:, 13:19],
            arm_joints=a[:, 19:25],
            clutch=a[:, 25],
            msj=a[:, 26:32],
            swivel=a[:, 32],
            rate_hz=self.rate,
        )
        valid = int(np.sum(np.isfinite(a[:, 10])))
        self.get_logger().info(
            f"wrote {self.out}: {len(self.rows)} samples, "
            f"{t[-1]:.1f} s, {valid} with a wrist pose"
        )


def main(args=None):
    rclpy.init(args=args)
    node = SessionRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.write()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
