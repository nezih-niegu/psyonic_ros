"""MuJoCo physics in the command path, between teleop and the hand.

    teleop  --raw/position-->  [ mujoco_safety ]  --target/position-->  ah_node
                                     |                                     |
                                     |  simulates + constrains             v
                                     +-- /joint_states_ah --> viewer    HARDWARE

Raw commands from any source (webcam teleop, a policy, a script) are treated as
*requests*. This node applies them to a physics simulation of the hand, and what
comes out of the simulation is what gets sent to the hardware. So the hand is
only ever commanded to a pose the simulation actually reached.

Constraints applied, in order:

  1. joint limits   - clamped to the URDF ranges
  2. rate limit     - max deg/s slew, so a tracking glitch cannot step the
                      hand instantly across its range
  3. physics        - the sim's own joint limits, mimic coupling and contacts
  4. torque guard   - if an actuator exceeds max_torque, the command for that
                      joint stops advancing (a rough self-collision / jam guard)
  5. contact guard  - the real hand's FSR feedback stops a digit from closing
                      further once it is pressing on something. This is the
                      only stage that knows about the actual world; the
                      simulation only knows about the hand itself.

Add further constraints in _apply_safety(); it is deliberately the only place
that modifies commands.

The node publishes /joint_states_ah from the SIMULATED state, so the MuJoCo
viewer shows what is actually being commanded rather than the raw request.
"""

import math
import threading

import numpy as np
import mujoco
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32MultiArray

from ah_messages.msg import Digits

from ah_mujoco.physics_model import (
    DRIVEN_JOINTS,
    actuator_indices,
    build_physics_model,
)

# URDF joint names in the order the viewer / ah_node expect
JOINT_NAMES = [
    "index_q1", "index_q2",
    "middle_q1", "middle_q2",
    "ring_q1", "ring_q2",
    "pinky_q1", "pinky_q2",
    "thumb_q1", "thumb_q2",
]


class MujocoSafetyNode(Node):
    def __init__(self):
        super().__init__("mujoco_safety_node")
        self.declare_parameter("hand_side", "Right")
        self.declare_parameter("hand_size", "Large")
        self.declare_parameter("model_path", "")
        self.declare_parameter("sim_hz", 500.0)
        self.declare_parameter("publish_hz", 120.0)
        # 200 deg/s means a 90 deg move takes 450 ms. The URDF velocity limit
        # is 8.07 rad/s = 462 deg/s; going near it keeps the guard meaningful
        # without dominating the latency budget.
        self.declare_parameter("max_speed_dps", 450.0)
        # The URDF effort limit is 6 Nm per finger joint. A guard set below
        # the actuator's own saturation fires during ordinary fast closing,
        # not just on a jam - which is what "torque limit hit on index_q1"
        # once a second means. Sit just under the real limit instead.
        self.declare_parameter("max_torque", 5.5)
        # Contact guard, driven by the hand's FSR feedback.
        # engage: a digit at or above this stops closing further.
        # release: it may close again once it falls below this. The gap is
        # hysteresis - a single threshold chatters as the reading crosses it.
        self.declare_parameter("contact_engage", 25.0)
        self.declare_parameter("contact_release", 12.0)
        self.declare_parameter("contact_backoff_deg", 0.0)
        self.declare_parameter("use_contact_guard", True)
        self.declare_parameter("kp", 100.0)
        # kv/kp is the closed-loop time constant. 0.06 cost 146 ms of rise
        # time regardless of kp; 0.005 gives 9 ms with no overshoot.
        self.declare_parameter("kv_ratio", 0.005)
        self.declare_parameter("raw_topic", "")
        self.declare_parameter("publish_targets", True)

        g = self.get_parameter
        side = g("hand_side").get_parameter_value().string_value.lower()
        size = g("hand_size").get_parameter_value().string_value.lower()
        model_path = g("model_path").get_parameter_value().string_value
        self.sim_hz = g("sim_hz").get_parameter_value().double_value
        self.publish_hz = g("publish_hz").get_parameter_value().double_value
        self.max_speed = g("max_speed_dps").get_parameter_value().double_value
        self.max_torque = g("max_torque").get_parameter_value().double_value
        self.contact_engage = (
            g("contact_engage").get_parameter_value().double_value
        )
        self.contact_release = (
            g("contact_release").get_parameter_value().double_value
        )
        self.contact_backoff = (
            g("contact_backoff_deg").get_parameter_value().double_value
        )
        self.use_contact_guard = (
            g("use_contact_guard").get_parameter_value().bool_value
        )
        kp = g("kp").get_parameter_value().double_value
        kv_ratio = g("kv_ratio").get_parameter_value().double_value
        self.publish_targets = (
            g("publish_targets").get_parameter_value().bool_value
        )
        raw_topic = g("raw_topic").get_parameter_value().string_value
        if not raw_topic:
            raw_topic = f"/ability_hand/{side}/raw/position"

        if not model_path:
            model_path = self._resolve_urdf(side, size)
        self.get_logger().info(f"Building physics model from {model_path}")

        self.model, _ = build_physics_model(
            model_path, timestep=1.0 / self.sim_hz, kp=kp, kv_ratio=kv_ratio
        )
        self.data = mujoco.MjData(self.model)
        self.act_ids = actuator_indices(self.model)

        self.qpos_addr = {}
        for name in JOINT_NAMES:
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self.qpos_addr[name] = self.model.jnt_qposadr[jid]

        # Command state, in hardware order (degrees):
        # [index, middle, ring, pinky, thumb_flexor, thumb_rotator]
        self.n = len(DRIVEN_JOINTS)
        self.requested = np.zeros(self.n, dtype=np.float64)
        self.commanded = np.zeros(self.n, dtype=np.float64)
        self.ctrl_limits_deg = np.degrees(
            self.model.jnt_range[
                [
                    mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, n)
                    for n in DRIVEN_JOINTS
                ]
            ]
        )
        self._lock = threading.Lock()
        self._clamped_warned = 0.0
        # Targets are withheld until a raw command actually arrives. Without
        # this the node would publish its own zero state at startup and drive
        # a real hand to the zero pose before teleop has said anything.
        self._got_raw = False
        # Latched per-digit contact state, with hysteresis
        self._in_contact = np.zeros(5, dtype=bool)
        self._touch = None
        self._contact_warned = 0.0

        ns = f"/ability_hand/{side}"
        self.create_subscription(Digits, raw_topic, self._raw_cb, 10)
        if self.use_contact_guard:
            self.create_subscription(
                Float32MultiArray, f"{ns}/feedback/touch", self._touch_cb, 10
            )
        self.pub_target = self.create_publisher(
            Digits, f"{ns}/target/position", 10
        )
        self.pub_js = self.create_publisher(JointState, "/joint_states_ah", 10)
        self.pub_js_plain = self.create_publisher(JointState, "/joint_states", 10)
        self.pub_status = self.create_publisher(
            Float32MultiArray, f"{ns}/safety/actuator_force", 10
        )
        self.pub_contact = self.create_publisher(
            Float32MultiArray, f"{ns}/safety/contact", 10
        )

        self._sim_thread = threading.Thread(target=self._sim_loop, daemon=True)
        self._sim_thread.start()
        self.create_timer(1.0 / max(self.publish_hz, 1.0), self._publish)

        self.get_logger().info(
            f"Safety layer active: {raw_topic} -> physics -> "
            f"{ns}/target/position  "
            f"(max {self.max_speed:.0f} deg/s, {self.max_torque:.1f} Nm)"
        )

    # ------------------------------------------------------------------ setup

    def _resolve_urdf(self, side, size):
        from pathlib import Path

        filename = f"ability_hand_{side}_{size}.urdf"
        try:
            from ament_index_python.packages import get_package_share_directory

            p = Path(get_package_share_directory("ah_urdf")) / "urdf" / filename
            if p.exists():
                return str(p)
        except Exception:
            pass
        for parent in Path(__file__).resolve().parents:
            p = parent / "ah_urdf" / "urdf" / filename
            if p.exists():
                return str(p)
        raise FileNotFoundError(f"Could not find {filename}")

    # -------------------------------------------------------------- callbacks

    # Which way each DOF closes. Fingers and the thumb flexor run 0..+100;
    # the thumb rotator runs 0..-100, so "closing" is negative for it.
    CLOSING_SIGN = np.array([1.0, 1.0, 1.0, 1.0, 1.0, -1.0])

    # Digit index (of the 5 FSR groups) that gates each of the 6 DOF.
    # Both thumb DOF are gated by the thumb's sensors.
    DOF_TO_DIGIT = [0, 1, 2, 3, 4, 4]

    def _touch_cb(self, msg: Float32MultiArray):
        """FSR feedback: 30 values, 6 sensors per digit."""
        data = np.asarray(msg.data, dtype=np.float64)
        if data.size < 5:
            return
        n_per = max(data.size // 5, 1)
        per_digit = np.array(
            [np.max(data[i * n_per:(i + 1) * n_per]) for i in range(5)]
        )
        with self._lock:
            self._touch = per_digit
            # Hysteresis: latch on above engage, release only below release
            self._in_contact = np.where(
                self._in_contact,
                per_digit > self.contact_release,
                per_digit >= self.contact_engage,
            )

    def _raw_cb(self, msg: Digits):
        data = list(msg.data)
        if len(data) != self.n:
            self.get_logger().warn(
                f"raw command has {len(data)} values, expected {self.n}",
                throttle_duration_sec=2.0,
            )
            return
        with self._lock:
            self.requested = np.array(data, dtype=np.float64)
            if not self._got_raw:
                self._got_raw = True
                # Start the plant at the commanded pose rather than sweeping
                # to it from zero at the rate limit
                self.commanded = self.requested.copy()
                self.get_logger().info(
                    "first raw command received; commanding the hand"
                )

    # ------------------------------------------------------------ safety core

    def _apply_safety(self, requested, dt):
        """Turn a requested pose into a safe commanded pose.

        This is the single place where commands are modified. Add new
        constraints here.
        """
        # 1. joint limits
        lo = self.ctrl_limits_deg[:, 0]
        hi = self.ctrl_limits_deg[:, 1]
        target = np.clip(requested, lo, hi)

        # 2. rate limit
        max_step = self.max_speed * dt
        delta = np.clip(target - self.commanded, -max_step, max_step)
        commanded = self.commanded + delta

        # 5. contact guard: a digit already pressing on something may not
        #    close further. Opening is always allowed, or a hand that grasps
        #    an object could never let go.
        if self.use_contact_guard and self._touch is not None:
            contact = self._in_contact
            for i in range(self.n):
                if not contact[self.DOF_TO_DIGIT[i]]:
                    continue
                closing = (commanded[i] - self.commanded[i]) * self.CLOSING_SIGN[i]
                if closing > 0.0:
                    commanded[i] = self.commanded[i]        # hold position
                if self.contact_backoff > 0.0:
                    commanded[i] -= self.CLOSING_SIGN[i] * self.contact_backoff
            if np.any(contact):
                now = self.get_clock().now().nanoseconds * 1e-9
                if now - self._contact_warned > 2.0:
                    self._contact_warned = now
                    names = [
                        ["index", "middle", "ring", "pinky", "thumb"][d]
                        for d in np.where(contact)[0]
                    ]
                    self.get_logger().info(f"contact: {names} - holding")

        # 4. torque guard: back off joints that are pushing too hard
        forces = self.data.actuator_force[self.act_ids]
        over = np.abs(forces) > self.max_torque
        if np.any(over):
            # Freeze the offending joints at their current simulated position
            for i in np.where(over)[0]:
                jname = DRIVEN_JOINTS[i]
                commanded[i] = math.degrees(
                    self.data.qpos[self.qpos_addr[jname]]
                )
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self._clamped_warned > 1.0:
                self._clamped_warned = now
                names = [DRIVEN_JOINTS[i] for i in np.where(over)[0]]
                self.get_logger().warn(f"torque limit hit on {names}")

        return commanded

    def _sim_loop(self):
        import time

        dt = 1.0 / self.sim_hz
        while rclpy.ok():
            t0 = time.perf_counter()
            with self._lock:
                requested = self.requested.copy()

            self.commanded = self._apply_safety(requested, dt)

            # 3. physics: the sim itself enforces limits, mimic coupling
            #    and contacts
            for i, aid in enumerate(self.act_ids):
                if aid >= 0:
                    self.data.ctrl[aid] = math.radians(self.commanded[i])
            mujoco.mj_step(self.model, self.data)

            elapsed = time.perf_counter() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

    # -------------------------------------------------------------- publishing

    def _sim_positions_deg(self):
        """Simulated joint positions in hardware order."""
        return np.array(
            [
                math.degrees(self.data.qpos[self.qpos_addr[n]])
                for n in DRIVEN_JOINTS
            ],
            dtype=np.float64,
        )

    def _publish(self):
        sim_deg = self._sim_positions_deg()

        if self.publish_targets and self._got_raw:
            msg = Digits()
            msg.reply_mode = 1
            # Send what the simulation actually reached, not the raw request
            msg.data = [float(v) for v in sim_deg]
            self.pub_target.publish(msg)

        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = JOINT_NAMES
        js.position = [
            float(self.data.qpos[self.qpos_addr[n]]) for n in JOINT_NAMES
        ]
        self.pub_js.publish(js)
        self.pub_js_plain.publish(js)

        status = Float32MultiArray()
        status.data = [float(v) for v in self.data.actuator_force[self.act_ids]]
        self.pub_status.publish(status)

        contact = Float32MultiArray()
        touch = self._touch if self._touch is not None else np.zeros(5)
        contact.data = [float(v) for v in self._in_contact.astype(float)] + [
            float(v) for v in touch
        ]
        self.pub_contact.publish(contact)


def main(args=None):
    rclpy.init(args=args)
    node = MujocoSafetyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
