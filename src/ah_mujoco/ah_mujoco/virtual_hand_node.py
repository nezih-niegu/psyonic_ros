"""Virtual Ability Hand — emulates the hardware side of the driver API.

Closed-loop nodes such as ah_tests/real_validation_node need position,
velocity AND touch feedback before they will produce any output. With no
physical hand attached those topics are silent, so the control loop stalls
publishing zeros forever.

This node stands in for the hardware:

    target/position (Digits, deg)  -->  [ first-order plant ]  -->  feedback/position (deg)
                                                                   feedback/velocity (deg/s)
                                                                   feedback/touch    (30 floats)
                                                                   /joint_states_ah  (rad)

It intentionally does NOT replace ah_node: run this *instead of* ah_node when
no hardware is present. The joint state output uses the same mapping ah_node
uses, so the MuJoCo viewer works unchanged.

Joint order throughout is the hardware order the driver uses:
    [index, middle, ring, pinky, thumb_flexor, thumb_rotator]
"""

import math

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray, UInt16

from ah_messages.msg import Digits

HAND_JOINT_ORDER = [
    "index", "middle", "ring", "pinky", "thumb_flexor", "thumb_rotator",
]
N_TOUCH = 30  # 5 digits x 6 FSRs, matches pipeline's reshape(5, 6)


class VirtualHandNode(Node):
    JOINT_NAMES = [
        "index_q1", "index_q2",
        "middle_q1", "middle_q2",
        "ring_q1", "ring_q2",
        "pinky_q1", "pinky_q2",
        "thumb_q1", "thumb_q2",
    ]

    def __init__(self):
        super().__init__("virtual_hand_node")
        self.declare_parameter("hand_side", "Right")
        self.declare_parameter("rate_hz", 60.0)
        # Degrees per second the virtual joints can slew toward the target
        self.declare_parameter("max_speed_dps", 250.0)
        self.declare_parameter("publish_joint_states", True)
        # The ONNX policy has a fixed point at exactly zero with zero contact:
        # it emits negative finger deltas that clip against the lower limit and
        # never moves. Seed a non-zero pose to get it going.
        self.declare_parameter("initial_position_deg", [0.0] * 6)

        self.side = (
            self.get_parameter("hand_side").get_parameter_value().string_value.lower()
        )
        self.rate_hz = self.get_parameter("rate_hz").get_parameter_value().double_value
        self.max_speed = (
            self.get_parameter("max_speed_dps").get_parameter_value().double_value
        )
        self.js_enabled = (
            self.get_parameter("publish_joint_states")
            .get_parameter_value()
            .bool_value
        )

        n = len(HAND_JOINT_ORDER)
        initial = list(
            self.get_parameter("initial_position_deg")
            .get_parameter_value()
            .double_array_value
        )
        if len(initial) != n:
            if initial:
                self.get_logger().warn(
                    f"initial_position_deg has {len(initial)} values, "
                    f"expected {n}; starting at zero."
                )
            initial = [0.0] * n

        self.position = list(initial)   # degrees
        self.velocity = [0.0] * n       # deg/s
        self.target = list(initial)     # degrees
        self.touch = [0.0] * N_TOUCH

        ns = f"/ability_hand/{self.side}"
        self.pub_pos = self.create_publisher(
            Float32MultiArray, f"{ns}/feedback/position", 10
        )
        self.pub_vel = self.create_publisher(
            Float32MultiArray, f"{ns}/feedback/velocity", 10
        )
        self.pub_touch = self.create_publisher(
            Float32MultiArray, f"{ns}/feedback/touch", 10
        )
        self.pub_current = self.create_publisher(
            Float32MultiArray, f"{ns}/feedback/current", 10
        )
        self.pub_hot_cold = self.create_publisher(
            UInt16, f"{ns}/feedback/hot_cold", 10
        )
        if self.js_enabled:
            self.pub_js = self.create_publisher(JointState, "/joint_states_ah", 10)
            # ah_policy_node subscribes to plain /joint_states, which used to be
            # produced by joint_state_publisher relaying joint_states_ah via its
            # source_list. RViz is gone, so publish it here as well.
            self.pub_js_plain = self.create_publisher(JointState, "/joint_states", 10)

        self.create_subscription(
            Digits, f"{ns}/target/position", self._target_cb, 10
        )

        self.dt = 1.0 / self.rate_hz
        self.create_timer(self.dt, self._tick)
        self.get_logger().info(
            f"Virtual Ability Hand ready on {ns} "
            f"({self.rate_hz:.0f} Hz). No hardware required."
        )

    def _target_cb(self, msg: Digits):
        data = list(msg.data)
        if len(data) != len(self.target):
            self.get_logger().warn(
                f"Ignoring target with {len(data)} values, "
                f"expected {len(self.target)}",
                throttle_duration_sec=2.0,
            )
            return
        self.target = [float(v) for v in data]

    def _tick(self):
        # First-order plant: slew toward the target, speed limited
        max_step = self.max_speed * self.dt
        for i, (cur, tgt) in enumerate(zip(self.position, self.target)):
            err = tgt - cur
            step = max(-max_step, min(max_step, err))
            self.position[i] = cur + step
            self.velocity[i] = step / self.dt

        self._publish_array(self.pub_pos, self.position)
        self._publish_array(self.pub_vel, self.velocity)
        self._publish_array(self.pub_touch, self.touch)
        self._publish_array(self.pub_current, [0.0] * len(self.position))

        hc = UInt16()
        hc.data = 0
        self.pub_hot_cold.publish(hc)

        if self.js_enabled:
            self._publish_joint_states()

    def _publish_array(self, pub, values):
        msg = Float32MultiArray()
        msg.data = [float(v) for v in values]
        pub.publish(msg)

    def _publish_joint_states(self):
        """Same mapping ah_node.publish_joint_states uses."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.JOINT_NAMES
        js = [0.0] * 10
        for i in range(4):
            js[i * 2] = math.radians(self.position[i])
            js[i * 2 + 1] = js[i * 2] * 1.05851325 + 0.72349796
        js[-1] = math.radians(self.position[-2])
        js[-2] = math.radians(self.position[-1])
        msg.position = js
        self.pub_js.publish(msg)
        self.pub_js_plain.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VirtualHandNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
