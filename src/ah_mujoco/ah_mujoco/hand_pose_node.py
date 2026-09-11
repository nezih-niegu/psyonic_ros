"""Publish the hand's SE(3) pose in the camera frame.

    /ability_hand/<side>/landmarks   (from mediapipe_teleop)
    /depth_to_rgb/image_raw          (depth registered to the RGB frame)
    /rgb/camera_info                 (intrinsics)
                        |
                        v
    /ability_hand/<side>/hand_pose   geometry_msgs/PoseStamped
    + TF: <camera frame> -> <side>_hand

Teleop already runs MediaPipe, so landmarks are consumed from its topic rather
than running inference a second time.

Use the REGISTERED depth topic (`/depth_to_rgb/image_raw`), not the raw one.
The Kinect's depth and colour cameras sit a few centimetres apart with
different intrinsics; landmark pixels come from the colour image, so looking
them up in unregistered depth samples the wrong points entirely.

The Ability Hand itself has no degrees of freedom that consume this - it is a
hand, not an arm. This is for driving a wrist or arm, for recording trajectories
in a world frame, or for logging alongside the joint data.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray

from ah_mujoco.hand_pose import (
    PALM_LANDMARKS,
    PoseSmoother,
    fit_pose,
    matrix_to_quaternion,
)


class HandPoseNode(Node):
    def __init__(self):
        super().__init__("hand_pose_node")
        self.declare_parameter("hand_side", "Right")
        self.declare_parameter("depth_topic", "/depth_to_rgb/image_raw")
        self.declare_parameter("camera_info_topic", "/rgb/camera_info")
        self.declare_parameter("camera_frame", "")
        self.declare_parameter("depth_scale", 0.001)     # uint16 mm -> m
        self.declare_parameter("patch", 2)
        self.declare_parameter("max_residual", 0.05)
        self.declare_parameter("palm_only", False)
        self.declare_parameter("alpha_t", 0.5)
        self.declare_parameter("alpha_r", 0.5)
        self.declare_parameter("publish_tf", True)

        g = self.get_parameter
        self.side = g("hand_side").get_parameter_value().string_value.lower()
        depth_topic = g("depth_topic").get_parameter_value().string_value
        info_topic = g("camera_info_topic").get_parameter_value().string_value
        self.camera_frame = g("camera_frame").get_parameter_value().string_value
        self.depth_scale = g("depth_scale").get_parameter_value().double_value
        self.patch = g("patch").get_parameter_value().integer_value
        self.max_residual = g("max_residual").get_parameter_value().double_value
        self.palm_only = g("palm_only").get_parameter_value().bool_value
        self.publish_tf = g("publish_tf").get_parameter_value().bool_value

        self.smoother = PoseSmoother(
            alpha_t=g("alpha_t").get_parameter_value().double_value,
            alpha_r=g("alpha_r").get_parameter_value().double_value,
        )

        self.intrinsics = None
        self.depth = None
        self.depth_frame = ""
        self._logged_first = False

        ns = f"/ability_hand/{self.side}"
        self.create_subscription(
            Float32MultiArray, f"{ns}/landmarks", self._landmarks_cb, 5
        )
        self.create_subscription(
            Image, depth_topic, self._depth_cb, qos_profile_sensor_data
        )
        self.create_subscription(
            CameraInfo, info_topic, self._info_cb, qos_profile_sensor_data
        )

        self.pub_pose = self.create_publisher(PoseStamped, f"{ns}/hand_pose", 10)
        self.pub_diag = self.create_publisher(
            Float32MultiArray, f"{ns}/hand_pose/diagnostics", 10
        )

        self.tf_broadcaster = None
        if self.publish_tf:
            try:
                from tf2_ros import TransformBroadcaster

                self.tf_broadcaster = TransformBroadcaster(self)
            except ImportError:
                self.get_logger().warn("tf2_ros unavailable; not publishing TF")

        self.get_logger().info(
            f"Hand pose: {ns}/landmarks + {depth_topic} -> {ns}/hand_pose"
        )

    # -------------------------------------------------------------- callbacks

    def _info_cb(self, msg: CameraInfo):
        if self.intrinsics is None:
            k = msg.k
            self.intrinsics = (k[0], k[4], k[2], k[5])
            self.get_logger().info(
                f"intrinsics fx={k[0]:.1f} fy={k[4]:.1f} "
                f"cx={k[2]:.1f} cy={k[5]:.1f} ({msg.width}x{msg.height})"
            )

    def _depth_cb(self, msg: Image):
        enc = (msg.encoding or "").lower()
        try:
            if enc in ("16uc1", "mono16"):
                d = np.frombuffer(msg.data, dtype=np.uint16)
                scale = self.depth_scale
            elif enc == "32fc1":
                d = np.frombuffer(msg.data, dtype=np.float32)
                scale = 1.0            # already metres
            else:
                self.get_logger().warn(
                    f"unsupported depth encoding '{msg.encoding}'; "
                    "expected 16UC1, mono16 or 32FC1",
                    throttle_duration_sec=5.0,
                )
                return
            self.depth = d.reshape(msg.height, msg.width)
            self._active_scale = scale
            self.depth_frame = msg.header.frame_id
        except Exception as exc:
            self.get_logger().warn(
                f"bad depth image: {exc}", throttle_duration_sec=5.0
            )

    def _landmarks_cb(self, msg: Float32MultiArray):
        if self.intrinsics is None or self.depth is None:
            self.get_logger().warn(
                "waiting for camera_info and depth",
                throttle_duration_sec=5.0,
            )
            return

        data = np.asarray(msg.data, dtype=np.float64)
        if data.size != 21 * 5:
            self.get_logger().warn(
                f"landmarks message has {data.size} values, expected 105",
                throttle_duration_sec=5.0,
            )
            return

        # layout: 21 x (u, v) pixels, then 21 x (x, y, z) world metres
        image_lm = data[: 21 * 2].reshape(21, 2)
        world_lm = data[21 * 2:].reshape(21, 3)

        out = fit_pose(
            world_lm, image_lm, self.depth, self.intrinsics,
            indices=PALM_LANDMARKS if self.palm_only else None,
            patch=self.patch,
            depth_scale=getattr(self, "_active_scale", self.depth_scale),
            max_residual=self.max_residual,
        )
        # Always publish diagnostics, including on failure: a silent topic
        # gives no way to tell "no valid depth" from "geometry inconsistent".
        diag = Float32MultiArray()
        diag.data = [
            float(out.get("residual", float("nan"))),
            float(out.get("n_points", 0)),
            float(bool(out.get("ok", False))),
        ]
        self.pub_diag.publish(diag)

        if not out.get("ok", False):
            if out["reason"] == "few_depth":
                self.get_logger().warn(
                    f"no pose: only {out['n_points']}/{out['n_tried']} "
                    "landmarks had valid depth. The hand is probably outside "
                    "the depth camera's field of view (narrower than the "
                    "colour image) or too close/far.",
                    throttle_duration_sec=2.0,
                )
            else:
                self.get_logger().warn(
                    f"no pose: fit residual {out['residual']*1000:.0f} mm "
                    f"exceeds max_residual "
                    f"({self.max_residual*1000:.0f} mm) with "
                    f"{out['n_points']} points. Depth and landmark geometry "
                    "disagree; raise max_residual to see the pose anyway.",
                    throttle_duration_sec=2.0,
                )
            return

        q = matrix_to_quaternion(out["R"])
        t, q = self.smoother.update(out["t"], q)

        if not self._logged_first:
            self._logged_first = True
            self.get_logger().info(
                f"first pose: t=({t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}) m, "
                f"{out['n_points']} points, residual {out['residual']*1000:.1f} mm"
            )

        frame = self.camera_frame or self.depth_frame or "camera_base"
        stamp = self.get_clock().now().to_msg()

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = frame
        pose.pose.position.x = float(t[0])
        pose.pose.position.y = float(t[1])
        pose.pose.position.z = float(t[2])
        pose.pose.orientation.x = float(q[0])
        pose.pose.orientation.y = float(q[1])
        pose.pose.orientation.z = float(q[2])
        pose.pose.orientation.w = float(q[3])
        self.pub_pose.publish(pose)

        if self.tf_broadcaster is not None:
            tf = TransformStamped()
            tf.header.stamp = stamp
            tf.header.frame_id = frame
            tf.child_frame_id = f"{self.side}_hand"
            tf.transform.translation.x = float(t[0])
            tf.transform.translation.y = float(t[1])
            tf.transform.translation.z = float(t[2])
            tf.transform.rotation = pose.pose.orientation
            self.tf_broadcaster.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = HandPoseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
