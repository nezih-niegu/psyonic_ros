"""Shoulder and elbow tracking, same pipeline as the hand.

MediaPipe Pose landmarks + registered depth -> 3D shoulder, elbow and wrist in
the camera frame, filtered and published the same way hand_pose publishes the
wrist.

    /rgb/image_raw  +  /depth_to_rgb/image_raw  +  /rgb/camera_info
        -> /ability_hand/<side>/upper_limb        (shoulder, elbow, wrist)
        -> /ability_hand/<side>/upper_limb/swivel (elbow swivel, radians)
        -> TF: <camera> -> <side>_shoulder, <side>_elbow

WHAT THIS IS AND IS NOT USED FOR
--------------------------------
It is tempting to map the human shoulder and elbow onto the arm's shoulder and
elbow joints. On a Lite 6 that does not work, and the reason is measurable
rather than a matter of taste:

  * the full Jacobian has rank 6 everywhere, so a fully constrained wrist pose
    consumes every degree of freedom - null space dimension 0
  * freeing tool roll leaves exactly one spare DOF, but that DOF is joint 6,
    which is distal to the elbow and moves it by 0.000 mm/rad
  * solving from many seeds and keeping the branch with the best elbow match
    changes mean swivel error by under 2 degrees
  * forcing the elbow with a weighted objective works, but costs 93 mm of
    wrist error to bring swivel from 22 to 7 degrees

Given a wrist pose, the arm's elbow is determined. So this node does three
things that are actually useful instead:

  1. SHOULDER ANCHORING. The hand pose is expressed relative to the operator's
     shoulder rather than the camera, so leaning or stepping sideways no
     longer drags the robot. This is the main reason to track the body.
  2. A POSTURE DIAGNOSTIC. Human swivel versus the arm's swivel, published for
     display, so the operator can see when the robot's posture has diverged
     even though the wrist is correct.
  3. RECORDING. Full upper-limb kinematics alongside the joint data, which is
     what a biomechanical dataset needs.

The optional elbow weighting in arm_teleop consumes the swivel topic if you
want it, but it defaults to zero for the reason measured above.
"""

import threading

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TransformStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray

from ah_mujoco.hand_pose import deproject, kabsch_ransac, sample_depth
from ah_mujoco.signal_quality import OneEuroFilter

# MediaPipe Pose landmark indices
POSE_LEFT_SHOULDER, POSE_RIGHT_SHOULDER = 11, 12
POSE_LEFT_ELBOW, POSE_RIGHT_ELBOW = 13, 14
POSE_LEFT_WRIST, POSE_RIGHT_WRIST = 15, 16


def swivel_from_points(shoulder, elbow, wrist,
                       reference=np.array([0.0, 0.0, -1.0])):
    """Elbow rotation about the shoulder->wrist axis. Scale free."""
    axis = wrist - shoulder
    n = np.linalg.norm(axis)
    if n < 1e-6:
        return float("nan")
    axis = axis / n
    v = elbow - shoulder
    v_perp = v - np.dot(v, axis) * axis
    if np.linalg.norm(v_perp) < 1e-6:
        return float("nan")
    v_perp /= np.linalg.norm(v_perp)
    ref = reference - np.dot(reference, axis) * axis
    if np.linalg.norm(ref) < 1e-6:
        ref = np.cross(axis, [1.0, 0.0, 0.0])
    ref /= np.linalg.norm(ref)
    return float(np.arctan2(
        np.dot(np.cross(ref, v_perp), axis), np.dot(ref, v_perp)
    ))


class UpperLimbNode(Node):
    def __init__(self):
        super().__init__("upper_limb_node")
        self.declare_parameter("hand_side", "Right")
        self.declare_parameter("image_topic", "/rgb/image_raw")
        self.declare_parameter("depth_topic", "/depth_to_rgb/image_raw")
        self.declare_parameter("camera_info_topic", "/rgb/camera_info")
        self.declare_parameter("model_path", "pose_landmarker.task")
        # Inference rate: how often MediaPipe Pose actually runs. Heavy.
        self.declare_parameter("rate_hz", 15.0)
        # Publish rate: how often the limb is emitted. Should match the hand
        # pipeline, or the limb lags it and the shoulder anchor updates a
        # quarter as often as the pose it is correcting.
        #
        # These can differ because the two parts of the estimate change at
        # very different speeds. The body-to-camera transform (where the torso
        # is) drifts slowly and only needs re-fitting at the inference rate.
        # The wrist moves fast, and comes from hand_pose at the hand's own
        # rate. So the cached transform is re-applied with each fresh wrist,
        # giving limb output at the hand's rate without running inference
        # any more often.
        self.declare_parameter("publish_hz", 60.0)
        self.declare_parameter("patch", 4)
        self.declare_parameter("min_visibility", 0.3)
        # 250 mm, not 60. The threshold should match what the output is USED
        # for, and the main use is shoulder anchoring - removing gross body
        # motion so leaning does not drag the robot. A shoulder located to
        # within 100 mm does that job completely. The earlier 60 mm was copied
        # from the hand fit, where precision genuinely matters, and it threw
        # away perfectly usable body estimates.
        self.declare_parameter("max_residual", 0.25)
        self.declare_parameter("inlier_threshold", 0.05)
        # Fit only the upper body by default. Legs are usually occluded by a
        # desk or out of the depth camera's view, and contribute nothing to
        # shoulder/elbow/wrist.
        self.declare_parameter("upper_body_only", True)
        self.declare_parameter("min_cutoff", 0.1)
        self.declare_parameter("beta", 0.01)
        self.declare_parameter("publish_tf", True)
        # The hand's own SE(3) estimate is far more accurate than the body fit
        # (a few mm against ~100 mm), and the wrist is the SAME physical point
        # in both. Use it as the chain's endpoint instead of predicting it.
        self.declare_parameter("use_hand_wrist", True)
        self.declare_parameter("hand_wrist_weight", 6)

        g = self.get_parameter
        self.side = g("hand_side").get_parameter_value().string_value.lower()
        self.patch = g("patch").get_parameter_value().integer_value
        self.min_vis = g("min_visibility").get_parameter_value().double_value
        self.max_residual = g("max_residual").get_parameter_value().double_value
        self.inlier_threshold = (
            g("inlier_threshold").get_parameter_value().double_value
        )
        self.upper_body_only = (
            g("upper_body_only").get_parameter_value().bool_value
        )
        # Shoulders, elbows, wrists, hips: the part of the body that is
        # actually visible and relevant to an arm anchor.
        self.fit_indices = list(range(11, 25)) if self.upper_body_only else None
        self._n_points, self._residual, self._scale = 0, float("nan"), 1.0
        self._wrist_correction = float("nan")
        self.rate = g("rate_hz").get_parameter_value().double_value
        self.publish_hz = g("publish_hz").get_parameter_value().double_value
        model_path = g("model_path").get_parameter_value().string_value

        self.right = self.side.startswith("r")
        self.idx = (
            (POSE_RIGHT_SHOULDER, POSE_RIGHT_ELBOW, POSE_RIGHT_WRIST)
            if self.right else
            (POSE_LEFT_SHOULDER, POSE_LEFT_ELBOW, POSE_LEFT_WRIST)
        )

        import os

        if not os.path.exists(model_path):
            raise SystemExit(
                f"Pose model not found: {model_path}\n"
                "Download it with:\n"
                "  wget -O pose_landmarker.task https://storage.googleapis.com/"
                "mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/"
                "1/pose_landmarker_lite.task"
            )

        self.use_hand_wrist = g("use_hand_wrist").get_parameter_value().bool_value
        self.hand_wrist_weight = (
            g("hand_wrist_weight").get_parameter_value().integer_value
        )
        self._hand_wrist = None
        self._fit = None
        self._body_pts = None
        self._fit_stamp = 0
        self._fit_lock = threading.Lock()
        # A fit older than this is not re-used: the operator has moved and the
        # cached torso transform no longer describes where they are.
        self.max_fit_age_ns = int(0.5e9)
        self.intrinsics = None
        self.depth = None
        self.depth_scale = 0.001
        self.frame = None
        self.frame_id = ""
        self._seq = 0
        self._last_seq = -1

        # Same filtering as the hand path: 3 points x 3 coords
        dt = 1.0 / max(self.rate, 1.0)
        self.filter = OneEuroFilter(
            9, dt,
            min_cutoff=g("min_cutoff").get_parameter_value().double_value,
            beta=g("beta").get_parameter_value().double_value,
        )

        self._init_landmarker(model_path)

        ns = f"/ability_hand/{self.side}"
        self.create_subscription(
            Image, g("image_topic").get_parameter_value().string_value,
            self._image_cb, qos_profile_sensor_data,
        )
        self.create_subscription(
            Image, g("depth_topic").get_parameter_value().string_value,
            self._depth_cb, qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            g("camera_info_topic").get_parameter_value().string_value,
            self._info_cb, qos_profile_sensor_data,
        )

        self.pub_limb = self.create_publisher(
            Float32MultiArray, f"{ns}/upper_limb", 10
        )
        self.pub_swivel = self.create_publisher(
            Float32MultiArray, f"{ns}/upper_limb/swivel", 10
        )
        if self.use_hand_wrist:
            self.create_subscription(
                PoseStamped, f"{ns}/hand_pose", self._hand_pose_cb, 10
            )

        self.pub_shoulder = self.create_publisher(
            PoseStamped, f"{ns}/shoulder", 10
        )
        # 2D landmarks for the UI overlay. The node that owns the image is
        # mediapipe_teleop, so the pixels have to travel to it.
        self.pub_pixels = self.create_publisher(
            Float32MultiArray, f"{ns}/upper_limb/pixels", 5
        )

        self.tf = None
        if g("publish_tf").get_parameter_value().bool_value:
            try:
                from tf2_ros import TransformBroadcaster

                self.tf = TransformBroadcaster(self)
            except ImportError:
                self.get_logger().warn("tf2_ros unavailable; no TF")

        self.create_timer(dt, self._tick)                     # inference
        self.create_timer(
            1.0 / max(self.publish_hz, 1.0), self._publish_tick
        )
        self.get_logger().info(
            f"Upper limb tracking ({self.side}) -> {ns}/upper_limb"
        )

    def _init_landmarker(self, model_path):
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.cv2 = cv2
        self.mp = mp
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(opts)

    # -------------------------------------------------------------- callbacks

    def _info_cb(self, msg: CameraInfo):
        if self.intrinsics is None:
            k = msg.k
            self.intrinsics = (k[0], k[4], k[2], k[5])
            self.get_logger().info(
                f"intrinsics fx={k[0]:.1f} cx={k[2]:.1f} ({msg.width}x{msg.height})"
            )

    def _depth_cb(self, msg: Image):
        enc = (msg.encoding or "").lower()
        try:
            if enc in ("16uc1", "mono16"):
                self.depth = np.frombuffer(msg.data, np.uint16).reshape(
                    msg.height, msg.width
                )
                self.depth_scale = 0.001
            elif enc == "32fc1":
                self.depth = np.frombuffer(msg.data, np.float32).reshape(
                    msg.height, msg.width
                )
                self.depth_scale = 1.0
        except Exception as exc:
            self.get_logger().warn(f"bad depth: {exc}", throttle_duration_sec=5.0)

    def _hand_pose_cb(self, msg: PoseStamped):
        """Wrist position from hand_pose, in the camera frame."""
        self._hand_wrist = np.array([
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        ])

    def _publish_tick(self):
        """Emit at the publish rate using the cached fit and the latest wrist.

        Between inferences the torso transform is held and only the wrist
        moves, which is exactly how the real arm behaves over 15-60 ms: the
        shoulder is near enough stationary while the hand is not.
        """
        with self._fit_lock:
            if self._fit is None or self._body_pts is None:
                return
            R, tvec, s = self._fit
            body = self._body_pts.copy()
            age_ns = self.get_clock().now().nanoseconds - self._fit_stamp

        if age_ns > self.max_fit_age_ns:
            return                      # the fit is stale; say nothing

        pts = np.array([s * (R @ p) + tvec for p in body])
        if self.use_hand_wrist and self._hand_wrist is not None:
            correction = self._hand_wrist - pts[2]
            pts[2] = self._hand_wrist
            pts[1] = pts[1] + 0.5 * correction
        self._emit(pts, filtered=False)

    def _emit(self, pts, filtered=True):
        m = Float32MultiArray()
        m.data = [float(v) for v in np.asarray(pts).reshape(-1)]
        self.pub_limb.publish(m)

    def _image_cb(self, msg: Image):
        enc = (msg.encoding or "bgr8").lower()
        try:
            buf = np.frombuffer(msg.data, np.uint8)
            if enc in ("bgr8", "rgb8"):
                img = buf.reshape(msg.height, msg.width, 3)
                rgb = img if enc == "rgb8" else img[:, :, ::-1]
            elif enc in ("bgra8", "rgba8"):
                img = buf.reshape(msg.height, msg.width, 4)[:, :, :3]
                rgb = img if enc == "rgba8" else img[:, :, ::-1]
            else:
                return
            self.frame = np.ascontiguousarray(rgb)
            self.frame_id = msg.header.frame_id
            self._seq += 1
        except Exception as exc:
            self.get_logger().warn(f"bad image: {exc}", throttle_duration_sec=5.0)

    # ------------------------------------------------------------------- loop

    def _tick(self):
        if self.frame is None or self.depth is None or self.intrinsics is None:
            self.get_logger().warn(
                "waiting for image, depth and camera_info",
                throttle_duration_sec=5.0,
            )
            return
        if self._seq == self._last_seq:
            return
        self._last_seq = self._seq

        mp_image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB, data=self.frame
        )
        result = self.landmarker.detect_for_video(
            mp_image, int(self._seq * (1000.0 / max(self.rate, 1.0)))
        )
        if not result.pose_landmarks:
            self.get_logger().warn("no pose detected", throttle_duration_sec=5.0)
            return

        lm = result.pose_landmarks[0]
        h, w = self.frame.shape[:2]
        fx, fy, cx, cy = self.intrinsics

        # Requiring depth at exactly the three joints is brittle: one dropout
        # on the shoulder loses the whole estimate. Instead fit the body's
        # metric world landmarks to whatever points DO have depth - the same
        # approach hand_pose uses - then read the three joints off the fitted
        # transform. Any three valid points anywhere on the body are enough.
        world = result.pose_world_landmarks[0] if result.pose_world_landmarks \
            else None
        if world is None:
            self.get_logger().warn(
                "pose has no world landmarks", throttle_duration_sec=5.0
            )
            return

        P, Q = [], []
        candidates = (
            self.fit_indices if self.fit_indices is not None
            else range(len(lm))
        )
        for i in candidates:
            p = lm[i]
            if getattr(p, "visibility", 1.0) < self.min_vis:
                continue
            u, v = p.x * w, p.y * h
            z = sample_depth(self.depth, u, v, self.patch, self.depth_scale)
            if not np.isfinite(z) or z <= 0:
                continue
            P.append([world[i].x, world[i].y, world[i].z])
            Q.append(deproject(u, v, z, fx, fy, cx, cy))


        # Anchor the fit to the hand's wrist: it is the same point the body
        # model calls the wrist, measured far more accurately. Repeating the
        # correspondence weights it without changing the solver.
        wrist_idx = self.idx[2]
        if self.use_hand_wrist and self._hand_wrist is not None:
            wp = [world[wrist_idx].x, world[wrist_idx].y, world[wrist_idx].z]
            for _ in range(max(self.hand_wrist_weight, 1)):
                P.append(wp)
                Q.append(self._hand_wrist)

        if len(P) < 3:
            self.get_logger().warn(
                f"only {len(P)} body landmarks had valid depth (need 3). "
                "The colour camera sees more of you than the depth camera "
                "does: at 720P the RGB field of view is wider than "
                "NFOV_UNBINNED depth, and uncovered pixels come back as zero. "
                "depth_mode:=WFOV_2X2BINNED covers roughly 120 deg instead of "
                "75 and runs at the same 30 fps.",
                throttle_duration_sec=5.0,
            )
            return

        # Similarity, not rigid: body world landmarks are estimated
        # proportions, so their absolute scale is approximate. A rigid fit
        # leaves 150-250 mm of residual against real depth.
        #
        # RANSAC on top of that: landmarks which are occluded or inferred get
        # whatever depth lies BEHIND them - a chair, the floor - which is
        # metres from the true point. Least squares lets a few such points
        # drag the whole transform.
        R, tvec, s, inliers, resid = kabsch_ransac(
            np.asarray(P), np.asarray(Q),
            threshold=self.inlier_threshold, seed=0,
        )
        if resid > self.max_residual:
            self.get_logger().warn(
                f"body fit rejected: residual {resid*1000:.0f} mm from "
                f"{len(P)} points (limit {self.max_residual*1000:.0f} mm)",
                throttle_duration_sec=3.0,
            )
            return

        # Publish, but say plainly how good it is. Quality belongs in the
        # data, not in a threshold that silently discards it.
        if resid > 0.10:
            self.get_logger().info(
                f"body fit {resid*1000:.0f} mm from {len(inliers)} points - "
                "coarse, fine for shoulder anchoring, not for posture "
                "measurement",
                throttle_duration_sec=10.0,
            )

        # The three joints we care about, from the fitted body transform
        # Cache the slow part: the fitted transform and the body-frame joint
        # positions. _emit() re-applies these at the publish rate.
        with self._fit_lock:
            self._fit = (R, tvec, s)
            self._body_pts = np.array([
                [world[i].x, world[i].y, world[i].z] for i in self.idx
            ])
            self._fit_stamp = self.get_clock().now().nanoseconds

        pts = np.array([
            s * (R @ p) + tvec for p in self._body_pts
        ])

        if self.use_hand_wrist and self._hand_wrist is not None:
            # The chain ENDS at the hand's wrist rather than at the body
            # model's guess, so the limb and the hand agree by construction
            # instead of drifting apart. The elbow is shifted by the same
            # correction, scaled down along the limb, so the upper arm is not
            # pulled out of shape.
            correction = self._hand_wrist - pts[2]
            pts[2] = self._hand_wrist
            pts[1] = pts[1] + 0.5 * correction
            self._wrist_correction = float(np.linalg.norm(correction))
        else:
            self._wrist_correction = float("nan")
        self._n_points, self._residual, self._scale = len(inliers), resid, s

        pts = self.filter.update(np.asarray(pts).reshape(-1)).reshape(3, 3)
        shoulder, elbow, wrist = pts

        self._emit(pts)

        # Normalised pixel coords of shoulder, elbow, wrist so the overlay is
        # independent of whatever resolution the UI is showing.
        px = Float32MultiArray()
        px.data = [
            float(v) for i in self.idx for v in (lm[i].x, lm[i].y)
        ]
        self.pub_pixels.publish(px)

        sw = swivel_from_points(shoulder, elbow, wrist)
        s = Float32MultiArray()
        s.data = [float(sw) if np.isfinite(sw) else 0.0,
                  float(np.isfinite(sw)),
                  float(self._n_points), float(self._residual),
                  float(self._scale), float(self._wrist_correction)]
        self.pub_swivel.publish(s)

        stamp = self.get_clock().now().to_msg()
        frame = self.frame_id or "rgb_camera_link"

        ps = PoseStamped()
        ps.header.stamp = stamp
        ps.header.frame_id = frame
        ps.pose.position.x, ps.pose.position.y, ps.pose.position.z = (
            float(v) for v in shoulder
        )
        ps.pose.orientation.w = 1.0
        self.pub_shoulder.publish(ps)

        if self.tf is not None:
            for name, p in (("shoulder", shoulder), ("elbow", elbow)):
                tf = TransformStamped()
                tf.header.stamp = stamp
                tf.header.frame_id = frame
                tf.child_frame_id = f"{self.side}_{name}"
                tf.transform.translation.x = float(p[0])
                tf.transform.translation.y = float(p[1])
                tf.transform.translation.z = float(p[2])
                tf.transform.rotation.w = 1.0
                self.tf.sendTransform(tf)


def main(args=None):
    rclpy.init(args=args)
    node = UpperLimbNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
