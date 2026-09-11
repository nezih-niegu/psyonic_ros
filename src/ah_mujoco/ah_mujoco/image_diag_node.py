"""Diagnose why an image topic is not being received.

Subscribes to the same topic twice, once best-effort and once reliable, counts
what each delivers, and reports the publishers' advertised QoS. That separates
the three failure modes which all look identical from the outside:

    nothing published            -> no publishers listed
    QoS incompatibility          -> publishers listed, one subscription gets
                                    frames and the other gets none
    transport / size problem     -> publishers listed, neither gets frames

    ros2 run ah_mujoco image_diag --ros-args -p topic:=/rgb/image_raw
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)
from sensor_msgs.msg import Image


class ImageDiagNode(Node):
    def __init__(self):
        super().__init__("image_diag_node")
        self.declare_parameter("topic", "/rgb/image_raw")
        self.declare_parameter("seconds", 5.0)

        self.topic = self.get_parameter("topic").get_parameter_value().string_value
        self.seconds = (
            self.get_parameter("seconds").get_parameter_value().double_value
        )

        self.counts = {"best_effort": 0, "reliable": 0}
        self.first = {}

        reliable = QoSProfile(depth=1)
        reliable.reliability = ReliabilityPolicy.RELIABLE

        self.create_subscription(
            Image, self.topic,
            lambda m: self._got("best_effort", m), qos_profile_sensor_data,
        )
        self.create_subscription(
            Image, self.topic, lambda m: self._got("reliable", m), reliable,
        )

        self.get_logger().info(
            f"Listening on {self.topic} for {self.seconds:.0f} s "
            "(best-effort and reliable simultaneously)..."
        )
        self.create_timer(self.seconds, self._report)

    def _got(self, which, msg):
        self.counts[which] += 1
        if which not in self.first:
            self.first[which] = (
                msg.width, msg.height, msg.encoding,
                len(msg.data), msg.header.frame_id,
            )

    def _report(self):
        log = self.get_logger().info
        log("=" * 62)

        pubs = self.get_publishers_info_by_topic(self.topic)
        if not pubs:
            log(f"NO PUBLISHERS on {self.topic}.")
            log("  The driver is not running, or it is on a different")
            log("  ROS_DOMAIN_ID. Check: echo $ROS_DOMAIN_ID  and  ros2 node list")
        else:
            log(f"{len(pubs)} publisher(s) on {self.topic}:")
            for p in pubs:
                try:
                    rel = p.qos_profile.reliability.name
                    dur = p.qos_profile.durability.name
                    log(f"  {p.node_name}: reliability={rel} durability={dur}")
                except Exception:
                    log(f"  {p.node_name}")

        log("")
        for which in ("best_effort", "reliable"):
            n = self.counts[which]
            rate = n / self.seconds
            log(f"  {which:12s} received {n:4d} msgs ({rate:5.1f} Hz)")
            if which in self.first:
                w, h, enc, nbytes, frame = self.first[which]
                log(f"               {w}x{h} {enc}, {nbytes/1e6:.1f} MB/frame, "
                    f"frame_id='{frame}'")

        log("")
        be, rl = self.counts["best_effort"], self.counts["reliable"]
        if be == 0 and rl == 0:
            log("VERDICT: no frames on either profile.")
            log("  If publishers are listed above, this is transport, not QoS.")
            log("  Large frames over CycloneDDS often need a bigger receive")
            log("  buffer; try a lower color_resolution first.")
        elif be and not rl:
            log("VERDICT: publisher is BEST-EFFORT only.")
            log("  Use image_qos:=sensor (the default 'auto' also works).")
        elif rl and not be:
            log("VERDICT: only the reliable subscription received frames.")
            log("  Use image_qos:=reliable.")
        else:
            log("VERDICT: both profiles work. QoS is not the problem.")
            log("  If teleop still sees nothing, check image_topic spelling")
            log("  and that teleop was rebuilt after the last change.")

        enc = self.first.get("best_effort") or self.first.get("reliable")
        if enc:
            e = enc[2].lower()
            if e not in ("bgr8", "rgb8", "mono8", "bgra8", "rgba8"):
                log("")
                log(f"WARNING: encoding '{enc[2]}' is not decoded by teleop.")
                log("  Supported: bgr8, rgb8, mono8, bgra8, rgba8.")
        log("=" * 62)
        rclpy.try_shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = ImageDiagNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
