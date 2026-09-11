"""Bridge between arm_teleop and the real xArm Lite 6.

Owns the startup sequence, which matters more than the streaming:

    1. motion_enable                 power the joints
    2. set_mode 0, set_state 0       position mode, ready
    3. set_servo_angle(safe_pose)    move to a known configuration and WAIT
    4. set_mode 1, set_state 0       servo mode, for streaming
    5. stream set_servo_angle_j      from /xarm/target_joints

Step 3 is the point. Wherever the arm was left - folded, mid-task, against a
limit - it goes to one known pose before anything else happens, and the motion
is a normal position-mode move, not a servo command. Streaming only starts
afterwards, and only once the clutch is engaged, so the arm cannot lurch from
an arbitrary configuration toward a teleoperation target.

Servo mode (mode 1) has no interpolation: every command is executed as-is at
high rate, so a jump in the stream is a jump at the joints. The rate limiting
in arm_teleop and the gating here are what keep that safe.

The driver itself is xarm_api; this node only calls its services. Start it
with the robot's address:

    ros2 launch xarm_api lite6_driver.launch.py robot_ip:=192.168.1.154
"""

import threading

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String

from xarm_msgs.srv import MoveJoint, SetInt16, SetInt16ById

# Elbow up, wrist bent: 0, -30, 50, 0, 40, 0 degrees.
#
# Chosen by measurement, not taste. The all-zeros home pose is SINGULAR
# (manipulability 0), so IK seeded there converges for only 156/200 nearby
# targets. This pose measures 0.0041 and converges 199/200, with at least
# 0.93 rad of margin on every joint limit and the flange 0.42 m up, clear of
# the table.
DEFAULT_SAFE_POSE = [0.0, -0.5236, 0.8727, 0.0, 0.6981, 0.0]


class XArmBridgeNode(Node):
    def __init__(self):
        super().__init__("xarm_bridge_node")
        self.declare_parameter("robot_ip", "192.168.1.154")
        self.declare_parameter("hand_side", "Right")
        self.declare_parameter("safe_pose", DEFAULT_SAFE_POSE)
        self.declare_parameter("safe_speed", 0.35)        # rad/s
        self.declare_parameter("safe_acc", 3.0)
        # The Lite 6 driver namespaces its services under /ufactory, not /xarm.
        # Check with: ros2 service list | grep motion_enable
        self.declare_parameter("service_prefix", "/ufactory")
        self.declare_parameter("stream_hz", 100.0)
        self.declare_parameter("require_clutch", True)
        self.declare_parameter("go_safe_on_start", True)
        self.declare_parameter("dry_run", False)

        g = self.get_parameter
        self.robot_ip = g("robot_ip").get_parameter_value().string_value
        side = g("hand_side").get_parameter_value().string_value.lower()
        self.prefix = g("service_prefix").get_parameter_value().string_value
        self.safe_pose = list(
            g("safe_pose").get_parameter_value().double_array_value
        ) or DEFAULT_SAFE_POSE
        self.safe_speed = g("safe_speed").get_parameter_value().double_value
        self.safe_acc = g("safe_acc").get_parameter_value().double_value
        self.require_clutch = g("require_clutch").get_parameter_value().bool_value
        self.dry_run = g("dry_run").get_parameter_value().bool_value

        self.ready = False          # safe pose reached, servo mode active
        self.clutch = False
        self.target = None

        # Service responses are delivered by the executor. If the clients sit
        # in the default mutually-exclusive group and the startup sequence
        # blocks waiting for a response, the callback that would deliver it
        # cannot run - the call times out with "no response" no matter what
        # the driver did.
        from rclpy.callback_groups import ReentrantCallbackGroup

        self._cbg = ReentrantCallbackGroup()
        self.cli_enable = self.create_client(
            SetInt16ById, f"{self.prefix}/motion_enable",
            callback_group=self._cbg,
        )
        self.cli_mode = self.create_client(
            SetInt16, f"{self.prefix}/set_mode", callback_group=self._cbg
        )
        self.cli_state = self.create_client(
            SetInt16, f"{self.prefix}/set_state", callback_group=self._cbg
        )
        self.cli_angle = self.create_client(
            MoveJoint, f"{self.prefix}/set_servo_angle",
            callback_group=self._cbg,
        )
        self.cli_servo_j = self.create_client(
            MoveJoint, f"{self.prefix}/set_servo_angle_j",
            callback_group=self._cbg,
        )

        self.create_subscription(
            JointState, "/xarm/target_joints", self._target_cb, 10
        )
        self.create_subscription(
            Bool, f"/ability_hand/{side}/clutch", self._clutch_cb, 10
        )
        self.pub_status = self.create_publisher(String, "/xarm/bridge_status", 10)

        self.get_logger().info(
            f"xArm bridge for {self.robot_ip}. The driver must already be "
            f"running: ros2 launch xarm_api lite6_driver.launch.py "
            f"robot_ip:={self.robot_ip}"
        )

        # Set before the thread can read it: the startup thread races
        # __init__, so every attribute it touches must already exist.
        self._startup_done = False

        if g("go_safe_on_start").get_parameter_value().bool_value:
            threading.Thread(target=self._startup_once, daemon=True).start()
        else:
            self.ready = True

        self.create_timer(
            1.0 / max(g("stream_hz").get_parameter_value().double_value, 1.0),
            self._stream,
        )

    # ----------------------------------------------------------------- startup

    def _call(self, client, request, what, timeout=10.0):
        if self.dry_run:
            self.get_logger().info(f"[dry run] {what}")
            return True
        if not client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(
                f"service {client.srv_name} unavailable - is the xarm_api "
                f"driver running with robot_ip:={self.robot_ip}?"
            )
            return False
        # Do NOT spin here: the executor is already spinning this node, and a
        # nested spin never services the future. Just wait for it.
        import time

        future = client.call_async(request)
        deadline = time.time() + timeout
        while not future.done() and time.time() < deadline and rclpy.ok():
            time.sleep(0.01)
        if not future.done() or future.result() is None:
            self.get_logger().error(
                f"{what}: no response after {timeout:.0f}s. The service exists "
                "but the driver did not answer - check the robot for an "
                "active error (an engaged emergency stop will do this)."
            )
            return False
        ret = future.result().ret
        if ret != 0:
            self.get_logger().error(f"{what}: ret={ret} {future.result().message}")
            return False
        self.get_logger().info(f"{what}: ok")
        return True

    def _startup_once(self):
        if self._startup_done:
            return
        self._startup_done = True
        # Give the executor a moment to come up before calling services
        import time

        time.sleep(1.0)
        self._status("starting")

        req = SetInt16ById.Request()
        req.id, req.data = 8, 1
        if not self._call(self.cli_enable, req, "motion_enable"):
            self._status("failed: motion_enable")
            return

        for mode, what in ((0, "set_mode 0 (position)"),):
            r = SetInt16.Request()
            r.data = mode
            if not self._call(self.cli_mode, r, what):
                self._status("failed: set_mode")
                return

        r = SetInt16.Request()
        r.data = 0
        if not self._call(self.cli_state, r, "set_state 0 (ready)"):
            self._status("failed: set_state")
            return

        # The move that matters: a normal position-mode move to a known pose,
        # waiting for completion before anything is streamed.
        self.get_logger().info(
            f"moving to safe pose {np.round(self.safe_pose, 3).tolist()} rad "
            "and waiting..."
        )
        self._status("moving to safe pose")
        mj = MoveJoint.Request()
        mj.angles = [float(v) for v in self.safe_pose]
        mj.speed = float(self.safe_speed)
        mj.acc = float(self.safe_acc)
        mj.wait = True
        mj.timeout = 30.0
        if not self._call(self.cli_angle, mj, "set_servo_angle(safe)", timeout=40.0):
            self._status("failed: safe pose")
            return

        # Only now switch to servo mode for streaming
        r = SetInt16.Request()
        r.data = 1
        if not self._call(self.cli_mode, r, "set_mode 1 (servo)"):
            self._status("failed: servo mode")
            return
        r = SetInt16.Request()
        r.data = 0
        if not self._call(self.cli_state, r, "set_state 0"):
            self._status("failed: state after servo")
            return

        self.ready = True
        self._status("ready")
        self.get_logger().info(
            "safe pose reached, servo mode active. "
            + ("Engage the clutch to move the arm."
               if self.require_clutch else "Streaming.")
        )

    # ------------------------------------------------------------------ stream

    def _clutch_cb(self, msg: Bool):
        self.clutch = bool(msg.data)

    def _target_cb(self, msg: JointState):
        if len(msg.position) >= 6:
            self.target = [float(v) for v in msg.position[:6]]

    def _stream(self):
        if not self.ready or self.target is None:
            return
        if self.require_clutch and not self.clutch:
            return
        if self.dry_run:
            return
        req = MoveJoint.Request()
        req.angles = self.target
        req.speed = 0.0
        req.acc = 0.0
        req.mvtime = 0.0
        self.cli_servo_j.call_async(req)

    def _status(self, text):
        m = String()
        m.data = text
        self.pub_status.publish(m)


def main(args=None):
    rclpy.init(args=args)
    node = XArmBridgeNode()
    from rclpy.executors import MultiThreadedExecutor

    ex = MultiThreadedExecutor(num_threads=3)
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
