"""Webcam hand tracking -> Ability Hand targets.

Runs MediaPipe HandLandmarker on a live webcam feed, retargets the 21
landmarks onto the hand's 6 DOF, and publishes them as
/ability_hand/<side>/target/position.

Because it publishes targets rather than joint states, it works unchanged
against either backend:

    teleop --targets--> virtual_hand --> mujoco_viewer     (no hardware)
    teleop --targets--> ah_node ------> real hand          (hardware)

Model bundle: run HandPose's get_model.sh, or point --model at any
hand_landmarker.task file.

Calibration matters. Hand size and camera distance change the raw curl values,
so the open/closed ranges are per-user. With the preview window focused:

    o : capture current pose as OPEN   (flat hand, fingers straight, thumb out)
    c : capture current pose as CLOSED (fist, thumb across the palm)
    s : print the calibration as ROS params you can paste into a launch file
    r : reset to defaults
    q : quit
"""

import os

import numpy as np
import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray

from ah_messages.msg import Digits

from ah_mujoco.hand_retarget import HandRetargeter, HAND_LIMITS_DEG
from ah_mujoco.calibration import (
    DEFAULT_PATH,
    CalibrationError,
    CalibrationSession,
    describe,
    load_calibration,
    save_calibration,
)
from ah_mujoco.signal_quality import (
    KalmanPosition,
    MAETracker,
    MSJMonitor,
    msj_band,
    msj_color,
)

JOINT_LABELS = ["index", "middle", "ring", "pinky", "thumb_flex", "thumb_rot"]


class MediapipeTeleopNode(Node):
    def __init__(self):
        super().__init__("mediapipe_teleop_node")
        self.declare_parameter("hand_side", "Right")
        self.declare_parameter("camera_index", 0)
        self.declare_parameter("camera_width", 640)
        self.declare_parameter("camera_height", 480)
        self.declare_parameter("camera_fps", 60.0)
        self.declare_parameter("mjpg", True)
        # Source: an image topic if set, otherwise the local camera device.
        # A topic lets a camera driver, a bag, or another machine feed the
        # pipeline without teleop owning /dev/video*.
        self.declare_parameter("image_topic", "")
        # QoS for image_topic. "sensor" is best-effort (what most camera
        # drivers publish); "reliable" matches drivers that publish with the
        # default profile. A best-effort subscriber gets nothing from some
        # configurations, which looks exactly like a dead camera, so this is
        # switchable. "auto" subscribes both ways and keeps whichever delivers.
        self.declare_parameter("image_qos", "auto")
        # --- unified skeleton -------------------------------------------------
        # Run BOTH landmarkers on the SAME frame in the same tick, so the hand
        # and the upper limb are one anatomy sampled once rather than two
        # estimates from different frames exchanged over topics. The wrist is
        # detected once and shared: it is simultaneously the hand's origin and
        # the limb's endpoint.
        self.declare_parameter("upper_limb", False)
        self.declare_parameter("pose_model_path", "")
        self.declare_parameter("depth_topic", "/depth_to_rgb/image_raw")
        self.declare_parameter("camera_info_topic", "/rgb/camera_info")
        # Pose inference is heavier than hands. Run it every Nth frame; the
        # torso transform is held between times and re-applied with each fresh
        # wrist, so the OUTPUT is still one skeleton per frame.
        self.declare_parameter("pose_every_n", 3)
        # Publish raw landmarks so hand_pose_node can estimate SE(3) without
        # running MediaPipe a second time.
        self.declare_parameter("publish_landmarks", False)
        # Publish frames and calibration prompts for the unified UI. When the
        # UI is used, set preview:=False so cv2 does not open a second window.
        self.declare_parameter("publish_image", False)
        self.declare_parameter("image_width", 480)
        self.declare_parameter("model_path", "hand_landmarker.task")
        self.declare_parameter("publish_hz", 60.0)
        self.declare_parameter("preview", True)
        self.declare_parameter("mirror", True)
        # EMA smoothing is a first-order lag: 0.6 at 60 Hz costs ~33 ms before
        # the command even leaves this node. Kept low for fine motion.
        self.declare_parameter("smoothing", 0.0)      # legacy EMA fallback
        # One Euro: adaptive low-pass. min_cutoff sets smoothing when still,
        # beta how quickly it opens up with speed.
        self.declare_parameter("one_euro", True)
        self.declare_parameter("min_cutoff", 0.1)
        self.declare_parameter("beta", 0.005)
        self.declare_parameter("det_conf", 0.5)
        # Where to publish. Default is the raw topic so mujoco_safety_node can
        # filter the commands before they reach hardware. Set to
        # "target" to bypass the safety layer and command the hand directly.
        self.declare_parameter("output", "raw")
        # --- signal quality / smoothness -----------------------------------
        self.declare_parameter("kalman", True)
        self.declare_parameter("kalman_process_var", 1e-1)
        self.declare_parameter("kalman_measurement_var", 1e-4)
        self.declare_parameter("msj_reference", 524.0)
        self.declare_parameter("msj_window", 90)
        self.declare_parameter("msj_normalize", True)
        self.declare_parameter("quality_log_hz", 1.0)
        # --- calibration ----------------------------------------------------
        self.declare_parameter("calibration_file", "")
        self.declare_parameter("force_calibration", False)
        # "both": departing from the calibrated baseline in either direction is
        # flagged. "above": only jerkier-than-baseline motion is flagged.
        self.declare_parameter("msj_direction", "both")
        self.declare_parameter("msj_spread", 0.0)
        # Calibration, overridable from a launch file
        self.declare_parameter("curl_open", 15.0)
        self.declare_parameter("curl_closed", 200.0)
        self.declare_parameter("thumb_flex_open", 10.0)
        self.declare_parameter("thumb_flex_closed", 90.0)
        self.declare_parameter("thumb_opp_open", 50.0)
        self.declare_parameter("thumb_opp_closed", 15.0)

        g = self.get_parameter
        side = g("hand_side").get_parameter_value().string_value.lower()
        self.cam_index = g("camera_index").get_parameter_value().integer_value
        self.cam_width = g("camera_width").get_parameter_value().integer_value
        self.cam_height = g("camera_height").get_parameter_value().integer_value
        self.cam_fps = g("camera_fps").get_parameter_value().double_value
        self.mjpg = g("mjpg").get_parameter_value().bool_value
        self.image_topic = g("image_topic").get_parameter_value().string_value
        self.image_qos = (
            g("image_qos").get_parameter_value().string_value.lower()
        )
        self.publish_landmarks = (
            g("publish_landmarks").get_parameter_value().bool_value
        )
        self.upper_limb = g("upper_limb").get_parameter_value().bool_value
        self.pose_every_n = max(
            g("pose_every_n").get_parameter_value().integer_value, 1
        )
        self._pose_model = (
            g("pose_model_path").get_parameter_value().string_value
        )
        self._depth = None
        self._depth_scale = 0.001
        self._intrinsics = None
        self._fit = None
        self._body_pts = None
        self._limb_pixels = None
        self.publish_image = g("publish_image").get_parameter_value().bool_value
        self.image_width = g("image_width").get_parameter_value().integer_value
        model_path = g("model_path").get_parameter_value().string_value
        self.publish_hz = g("publish_hz").get_parameter_value().double_value
        self.preview = g("preview").get_parameter_value().bool_value
        self.mirror = g("mirror").get_parameter_value().bool_value
        self.det_conf = g("det_conf").get_parameter_value().double_value

        # Nominal sample period; used by the filters below. The real
        # inter-frame interval is passed per tick where it matters.
        dt = 1.0 / max(self.publish_hz, 1.0)

        self.calibration_file = (
            g("calibration_file").get_parameter_value().string_value or None
        )
        force_cal = g("force_calibration").get_parameter_value().bool_value

        saved = None
        if not force_cal:
            try:
                saved = load_calibration(self.calibration_file)
            except CalibrationError as exc:
                # An unusable file must not silently fall back to defaults -
                # but it also must not be a dead end. Recalibrating is the
                # obvious next step, so do that instead of refusing to start.
                self.get_logger().warn(
                    f"Existing calibration cannot be used: {exc}\n"
                    "Starting a fresh calibration."
                )
                saved = None

        self.retargeter = HandRetargeter(
            smoothing=g("smoothing").get_parameter_value().double_value,
            one_euro=g("one_euro").get_parameter_value().bool_value,
            min_cutoff=g("min_cutoff").get_parameter_value().double_value,
            beta=g("beta").get_parameter_value().double_value,
            dt=dt,
        )
        self.session = None

        if saved and not force_cal:
            for k, v in saved.items():
                if hasattr(self.retargeter, k):
                    setattr(self.retargeter, k, float(v))
            self._saved_msj_reference = saved.get("msj_reference")
            self._saved_msj_spread = saved.get("msj_spread")
            self.body_scale = saved.get("body_scale")
            self.get_logger().info(f"Loaded calibration: {describe(saved)}")
        else:
            self._saved_msj_reference = None
            self._saved_msj_spread = None
            self.body_scale = None
            self.session = CalibrationSession()
            reason = "forced" if force_cal else "no saved calibration found"
            if not self.preview and not self.publish_image:
                # Nothing can capture the poses: no cv2 window and no UI
                # listening for remote calibration commands.
                raise SystemExit(
                    f"Calibration required ({reason}) but there is no preview "
                    "window and no UI attached, so the poses cannot be "
                    "captured.\nRun with preview:=True, or run the clinical "
                    "UI (publish_image:=True), or point calibration_file at "
                    f"an existing file ({self.calibration_file or DEFAULT_PATH})."
                )
            self.get_logger().warn(
                f"Calibration required ({reason}). No hand commands will be "
                "published until it is complete. " + self.session.prompt()
            )

        self.use_kalman = g("kalman").get_parameter_value().bool_value
        self.msj_reference = g("msj_reference").get_parameter_value().double_value
        if getattr(self, "_saved_msj_reference", None) is not None:
            self.msj_reference = float(self._saved_msj_reference)
        self.msj_direction = (
            g("msj_direction").get_parameter_value().string_value.lower()
        )
        spread_param = g("msj_spread").get_parameter_value().double_value
        self.msj_spread = (
            float(self._saved_msj_spread)
            if getattr(self, "_saved_msj_spread", None)
            else (spread_param if spread_param > 0 else None)
        )
        # Kalman runs on the 21 landmarks x 3 coords, before retargeting, so
        # the filter sees the raw tracker output the way the paper does.
        self.kf = KalmanPosition(
            63,
            dt,
            process_var=g("kalman_process_var").get_parameter_value().double_value,
            measurement_var=g("kalman_measurement_var")
            .get_parameter_value()
            .double_value,
        )
        self.mae = MAETracker(63)
        # MSJ is computed on the 6 retargeted joint commands, which is what
        # actually reaches the hand.
        self.msj = MSJMonitor(
            6,
            dt,
            window=g("msj_window").get_parameter_value().integer_value,
            normalize=g("msj_normalize").get_parameter_value().bool_value,
        )
        self.last_msj = np.zeros(6)
        self._last_quality_log = 0.0
        self.quality_log_hz = (
            g("quality_log_hz").get_parameter_value().double_value
        )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"HandLandmarker model not found: {model_path}\n"
                "Get it with HandPose's get_model.sh, then pass "
                "-p model_path:=/abs/path/hand_landmarker.task"
            )

        output = g("output").get_parameter_value().string_value.lower()
        if output == "target":
            topic = f"/ability_hand/{side}/target/position"
            self.get_logger().warn(
                "publishing straight to target/position - the MuJoCo safety "
                "layer is bypassed"
            )
        else:
            topic = f"/ability_hand/{side}/raw/position"
        self.pub = self.create_publisher(Digits, topic, 10)
        self._topic = topic
        self._F32 = Float32MultiArray
        self.pub_msj = self.create_publisher(
            Float32MultiArray, f"/ability_hand/{side}/quality/msj", 10
        )
        self.pub_mae = self.create_publisher(
            Float32MultiArray, f"/ability_hand/{side}/quality/mae", 10
        )
        self.pub_latency = self.create_publisher(
            Float32MultiArray, f"/ability_hand/{side}/quality/latency_ms", 10
        )
        # The MSJ reference has to travel to whoever colours by it. The UI
        # reads the calibration file once at startup, so a calibration done
        # DURING the session was invisible to it and every bar stayed red
        # against the stale default.
        self.pub_reference = self.create_publisher(
            Float32MultiArray, f"/ability_hand/{side}/quality/reference", 10
        )
        if self.publish_landmarks:
            self.pub_landmarks = self.create_publisher(
                Float32MultiArray, f"/ability_hand/{side}/landmarks", 5
            )
        # Upper-limb overlay: upper_limb_node detects the body, but this node
        # owns the image, so the 2D points come back here to be drawn.
        self._limb_px = None
        self.create_subscription(
            Float32MultiArray, f"/ability_hand/{side}/upper_limb/pixels",
            self._limb_px_cb, 5,
        )
        if self.publish_image and self.image_topic == (
            f"/ability_hand/{side}/camera/image_raw"
        ):
            self.publish_image = False
            self.get_logger().warn(
                "publish_image disabled: image_topic is the same topic teleop "
                "would publish to, which would feed back on itself"
            )
        if self.publish_image:
            from sensor_msgs.msg import Image
            from std_msgs.msg import String

            self._Image = Image
            self.pub_img = self.create_publisher(
                Image, f"/ability_hand/{side}/camera/image_raw", 1
            )
            self.pub_calib = self.create_publisher(
                String, f"/ability_hand/{side}/calibration/status", 10
            )
            self._String = String
            self.create_subscription(
                String,
                f"/ability_hand/{side}/calibration/command",
                self._calib_cmd_cb,
                10,
            )

        if self.image_topic:
            self._init_image_topic()
        else:
            self._init_camera()
        self._init_landmarker(model_path)
        if self.upper_limb:
            self._init_pose_landmarker()

        self.last_targets = np.zeros(6, dtype=np.float32)
        self.frame_idx = 0
        self._last_seq = -1
        self._last_landmarks = None
        self.source_age_ms = 0.0
        self._side_is_right = (
            self.get_parameter("hand_side")
            .get_parameter_value().string_value.lower().startswith("r")
        )
        self.create_timer(1.0 / max(self.publish_hz, 1.0), self._tick)
        self.get_logger().info(
            f"Teleop running on "
            f"{('topic ' + self.image_topic) if self.image_topic else ('camera ' + str(self.cam_index))}"
            f" -> {self._topic}. "
            "Press 'o' (open) and 'c' (closed) in the preview to calibrate."
        )

    def _init_camera(self):
        import threading

        import cv2

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(self.cam_index)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {self.cam_index}. In Docker, pass the "
                "device through: devices: ['/dev/video0:/dev/video0']"
            )
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.cam_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.cam_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.cam_fps)
        # Only keep the newest frame. The V4L2 default queues several, and a
        # queued frame is pure latency: by the time it is read it is already
        # 1-4 frame periods old.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.mjpg:
            # YUYV at 640x480x30 saturates USB2 and the driver throttles;
            # MJPG is compressed on the camera and arrives sooner.
            self.cap.set(
                cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")
            )

        self._frame = None
        self._frame_seq = 0
        self._frame_stamp = None
        self._frame_lock = threading.Lock()
        self._grab_running = True
        self._grab_thread = threading.Thread(
            target=self._grab_loop, daemon=True
        )
        self._grab_thread.start()
        self.get_logger().info(
            f"Camera {self.cam_index}: {self.cam_width}x{self.cam_height} "
            f"@{self.cam_fps:.0f} fps, buffersize=1"
            + (", MJPG" if self.mjpg else "")
        )

    def _init_image_topic(self):
        """Take frames from a ROS topic instead of a local camera device."""
        import threading

        import cv2
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image as ImageMsg

        self.cv2 = cv2
        self.cap = None
        self._frame = None
        self._frame_seq = 0
        self._frame_stamp = None
        self._frame_lock = threading.Lock()
        self._grab_running = False

        from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
        from rclpy.qos import QoSProfile, ReliabilityPolicy

        # Inference runs in the timer callback and takes tens of milliseconds.
        # On the default single-threaded executor that can starve the image
        # subscription, which looks exactly like a camera that never delivers.
        self._image_cbg = MutuallyExclusiveCallbackGroup()

        reliable = QoSProfile(depth=1)
        reliable.reliability = ReliabilityPolicy.RELIABLE

        subs = []
        if self.image_qos in ("sensor", "best_effort", "auto"):
            subs.append(("best-effort", qos_profile_sensor_data))
        if self.image_qos in ("reliable", "default", "auto"):
            subs.append(("reliable", reliable))
        if not subs:
            subs = [("best-effort", qos_profile_sensor_data)]

        for label, profile in subs:
            self.create_subscription(
                ImageMsg, self.image_topic, self._source_image_cb, profile,
                callback_group=self._image_cbg,
            )
        self._first_frame_logged = False
        self.get_logger().info(
            f"Image source: topic {self.image_topic} "
            f"({', '.join(l for l, _ in subs)} QoS)"
        )

    def _source_image_cb(self, msg):
        """Decode an incoming frame to BGR and keep only the newest."""
        try:
            enc = (msg.encoding or "bgr8").lower()
            buf = np.frombuffer(msg.data, dtype=np.uint8)
            if enc in ("bgr8", "rgb8"):
                img = buf.reshape(msg.height, msg.width, 3)
                if enc == "rgb8":
                    img = img[:, :, ::-1]
            elif enc in ("mono8", "8uc1"):
                img = self.cv2.cvtColor(
                    buf.reshape(msg.height, msg.width), self.cv2.COLOR_GRAY2BGR
                )
            elif enc in ("bgra8", "rgba8"):
                img = buf.reshape(msg.height, msg.width, 4)[:, :, :3]
                if enc == "rgba8":
                    img = img[:, :, ::-1]
            else:
                self.get_logger().warn(
                    f"unsupported image encoding '{msg.encoding}'; expected "
                    "bgr8, rgb8, mono8, bgra8 or rgba8",
                    throttle_duration_sec=5.0,
                )
                return
        except Exception as exc:
            self.get_logger().warn(
                f"could not decode image: {exc}", throttle_duration_sec=5.0
            )
            return

        if not getattr(self, "_first_frame_logged", False):
            self._first_frame_logged = True
            self.get_logger().info(
                f"first frame received: {msg.width}x{msg.height} "
                f"{msg.encoding}"
            )

        with self._frame_lock:
            self._frame = np.ascontiguousarray(img)
            self._frame_seq += 1
            self._frame_stamp = msg.header.stamp

    def _grab_loop(self):
        """Continuously drain the driver queue, keeping only the newest frame.

        Reading in the timer callback instead would let frames pile up
        whenever inference runs slower than the camera, and every queued frame
        adds a full frame period of latency.
        """
        while self._grab_running:
            ok, frame = self.cap.read()
            if not ok:
                continue
            with self._frame_lock:
                self._frame = frame
                self._frame_seq += 1

    def _latest_frame(self):
        with self._frame_lock:
            if self._frame is None:
                return None, self._frame_seq
            return self._frame.copy(), self._frame_seq

    def _init_pose_landmarker(self):
        """Second model, same frame. One anatomy, one sample."""
        import os

        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CameraInfo
        from sensor_msgs.msg import Image as ImageMsg
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        if not self._pose_model or not os.path.exists(self._pose_model):
            raise SystemExit(
                f"upper_limb is on but pose_model_path is not a file: "
                f"{self._pose_model!r}\n"
                "wget -O pose_landmarker.task https://storage.googleapis.com/"
                "mediapipe-models/pose_landmarker/pose_landmarker_lite/"
                "float16/1/pose_landmarker_lite.task"
            )

        import threading

        self._pose_busy = False
        self._pose_input = None
        self._pose_result = None
        self._pose_running = True
        self._pose_event = threading.Event()

        self.pose_landmarker = vision.PoseLandmarker.create_from_options(
            vision.PoseLandmarkerOptions(
                base_options=mp_python.BaseOptions(
                    model_asset_path=self._pose_model
                ),
                running_mode=vision.RunningMode.VIDEO,
                num_poses=1,
            )
        )

        g = self.get_parameter
        self.create_subscription(
            ImageMsg, g("depth_topic").get_parameter_value().string_value,
            self._depth_cb, qos_profile_sensor_data,
        )
        self.create_subscription(
            CameraInfo,
            g("camera_info_topic").get_parameter_value().string_value,
            self._info_cb, qos_profile_sensor_data,
        )

        from geometry_msgs.msg import PoseStamped

        self._PoseStamped = PoseStamped
        side = g("hand_side").get_parameter_value().string_value.lower()
        ns = f"/ability_hand/{side}"
        self.pub_skeleton = self.create_publisher(
            Float32MultiArray, f"{ns}/skeleton", 5
        )
        self.pub_limb = self.create_publisher(
            Float32MultiArray, f"{ns}/upper_limb", 5
        )
        self.pub_limb_px = self.create_publisher(
            Float32MultiArray, f"{ns}/upper_limb/pixels", 5
        )
        self.pub_swivel = self.create_publisher(
            Float32MultiArray, f"{ns}/upper_limb/swivel", 5
        )
        threading.Thread(target=self._pose_worker, daemon=True).start()
        self.get_logger().info(
            "unified skeleton: hand + upper limb from the same frame, "
            f"pose inference every {self.pose_every_n} frames "
            "(off-thread, so the hand loop never waits on it)"
        )

    def _info_cb(self, msg):
        if self._intrinsics is None:
            k = msg.k
            self._intrinsics = (k[0], k[4], k[2], k[5])

    def _depth_cb(self, msg):
        enc = (msg.encoding or "").lower()
        try:
            if enc in ("16uc1", "mono16"):
                self._depth = np.frombuffer(msg.data, np.uint16).reshape(
                    msg.height, msg.width
                )
                self._depth_scale = 0.001
            elif enc == "32fc1":
                self._depth = np.frombuffer(msg.data, np.float32).reshape(
                    msg.height, msg.width
                )
                self._depth_scale = 1.0
        except Exception:
            pass

    def _init_landmarker(self, model_path):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self.mp = mp
        base = mp_python.BaseOptions(model_asset_path=model_path)
        opts = vision.HandLandmarkerOptions(
            base_options=base,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=self.det_conf,
        )
        self.landmarker = vision.HandLandmarker.create_from_options(opts)

    # ------------------------------------------------------------------ loop

    def _tick(self):
        cv2 = self.cv2
        frame, seq = self._latest_frame()
        if frame is None:
            if self.image_topic:
                self.get_logger().warn(
                    f"no frames on {self.image_topic}. Check the publisher is "
                    f"running (ros2 topic hz {self.image_topic}); if it "
                    "publishes from another container or host, check DDS "
                    "discovery reaches this one.",
                    throttle_duration_sec=5.0,
                )
            else:
                self.get_logger().warn(
                    f"no frames from camera {self.cam_index}. Check the "
                    "device is present (ls -l /dev/video*) and forwarded into "
                    "the container.",
                    throttle_duration_sec=5.0,
                )
            return
        if seq == self._last_seq:
            return          # no new frame yet; do not re-run inference
        self._last_seq = seq

        with self._frame_lock:
            stamp = self._frame_stamp
        if stamp is not None:
            now = self.get_clock().now().nanoseconds
            age_ms = (now - (stamp.sec * 10**9 + stamp.nanosec)) / 1e6
            if 0.0 <= age_ms < 5000.0:
                self.source_age_ms = age_ms

        if self.mirror:
            frame = cv2.flip(frame, 1)
            if self.publish_landmarks and not getattr(self, "_mirror_warned", False):
                self._mirror_warned = True
                self.get_logger().warn(
                    "mirror is on while publishing landmarks: pixel "
                    "coordinates will not match the depth image. Set "
                    "mirror:=False when using hand_pose_node."
                )

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = self.mp.Image(
            image_format=self.mp.ImageFormat.SRGB, data=rgb
        )
        self.frame_idx += 1
        ts_ms = int(self.frame_idx * (1000.0 / max(self.publish_hz, 1.0)))
        result = self.landmarker.detect_for_video(mp_image, ts_ms)

        landmarks = None
        pixel_lm = None
        if result.hand_landmarks:
            h, w = frame.shape[:2]
            pixel_lm = np.array(
                [[p.x * w, p.y * h] for p in result.hand_landmarks[0]],
                dtype=np.float64,
            )
        if result.hand_world_landmarks:
            # World landmarks are metric and origin-centred on the hand, so
            # they are invariant to where the hand sits in frame.
            landmarks = np.array(
                [[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]],
                dtype=np.float64,
            )

        if landmarks is not None:
            raw_flat = landmarks.reshape(-1)
            if self.use_kalman:
                filt_flat = self.kf.update(raw_flat)
                self.mae.update(raw_flat, filt_flat)
                landmarks = filt_flat.reshape(21, 3)
            now_ns = self.get_clock().now().nanoseconds
            prev = getattr(self, "_last_tick_ns", None)
            frame_dt = (now_ns - prev) / 1e9 if prev else None
            self._last_tick_ns = now_ns
            if frame_dt is not None and not (0.001 < frame_dt < 1.0):
                frame_dt = None          # ignore stalls and clock jumps
            self.last_targets = self.retargeter.retarget(landmarks, frame_dt)
            self.last_msj = self.msj.update(self.last_targets)
            self._last_features = self.retargeter.raw_features(landmarks)
            self._last_landmarks = landmarks
            if self.session is None:
                self._publish(self.last_targets)
            elif self.session.step == "baseline":
                self.session.observe_baseline(float(np.mean(self.last_msj)))
            self._publish_quality()
            if self.publish_landmarks and pixel_lm is not None:
                self._publish_landmarks(pixel_lm, landmarks)
            if self.upper_limb:
                # Same frame, same tick, same wrist. The hand's own wrist
                # pixel IS the limb's endpoint - it does not get predicted
                # twice and cannot disagree with itself.
                self._update_upper_limb(mp_image, frame, pixel_lm)

        if self.preview or self.publish_image:
            annotated = self._annotate(frame, result)
            if self.publish_image:
                self._publish_image(annotated)
                self._publish_calib_status()
            if self.preview:
                self._show(annotated, landmarks)

    def _publish_quality(self):
        m = self._F32()
        m.data = [float(v) for v in self.last_msj]
        self.pub_msj.publish(m)
        self._publish_latency()
        self._publish_reference()
        e = self._F32()
        e.data = [float(v) for v in self.mae.mae()]
        self.pub_mae.publish(e)

        if self.quality_log_hz > 0:
            now = self.get_clock().now().nanoseconds * 1e-9
            if now - self._last_quality_log > 1.0 / self.quality_log_hz:
                self._last_quality_log = now
                total = float(np.mean(self.last_msj))
                self.get_logger().info(
                    f"MSJ={total:8.1f} "
                    f"({msj_band(total, self.msj_reference, self.msj_spread, self.msj_direction)}) "
                    f"dev={abs(total - self.msj_reference):7.1f} | "
                    f"landmark MAE={self.mae.total() * 1000:.2f} mm"
                )

    def _publish(self, targets):
        msg = Digits()
        msg.reply_mode = 1
        msg.data = [float(v) for v in targets]
        self.pub.publish(msg)

    # --------------------------------------------------------------- preview

    def _publish_image(self, frame):
        cv2 = self.cv2
        h, w = frame.shape[:2]
        if w != self.image_width:
            scale = self.image_width / w
            frame = cv2.resize(
                frame, (self.image_width, max(int(h * scale), 1)),
                interpolation=cv2.INTER_AREA,
            )
        h, w = frame.shape[:2]
        msg = self._Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height, msg.width = h, w
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = w * 3
        msg.data = np.ascontiguousarray(frame).tobytes()
        self.pub_img.publish(msg)

    def _calib_cmd_cb(self, msg):
        """Calibration keys forwarded from the clinical UI.

        The UI owns the only window when preview is off, so it relays 'o',
        'c', 'b', 'k' and 's' here rather than the session being unreachable.
        """
        cmd = (msg.data or "").strip().lower()[:1]
        if not cmd:
            return
        lm = getattr(self, "_last_landmarks", None)
        if lm is None:
            self.get_logger().warn(
                f"'{cmd}' ignored: no hand detected. Hold your hand in view "
                "of the camera and try again."
            )
            return
        step = self.session.step if self.session is not None else None
        self.get_logger().info(f"calibration key '{cmd}' (step: {step})")
        self._handle_key(ord(cmd), lm)

    def _update_upper_limb(self, mp_image, frame, hand_pixels):
        """Fit the torso, then attach the limb to the hand's own wrist."""
        from ah_mujoco.hand_pose import deproject, kabsch_ransac, sample_depth
        from ah_mujoco.upper_limb_node import swivel_from_points

        if self._depth is None or self._intrinsics is None:
            self.get_logger().warn(
                f"upper limb idle: depth={'yes' if self._depth is not None else 'NO'} "
                f"camera_info={'yes' if self._intrinsics else 'NO'}. "
                "The Kinect must publish depth_to_rgb and camera_info.",
                throttle_duration_sec=5.0,
            )
            return

        h, w = frame.shape[:2]
        fx, fy, cx, cy = self._intrinsics
        right = self._side_is_right
        idx = (12, 14, 16) if right else (11, 13, 15)

        # Pose inference runs OFF this thread. Inline it would stall the hand
        # loop for its whole duration every Nth frame, which shows up as the
        # UI stuttering and the finger targets arriving late. The torso is the
        # slow-changing part, so a result that lands a frame or two later
        # costs nothing.
        if (self.frame_idx % self.pose_every_n == 0
                and not self._pose_busy):
            self._pose_busy = True
            self._pose_input = (mp_image, frame.shape[:2], self.frame_idx)
            self._pose_event.set()

        result = self._pose_result
        self._pose_result = None
        if result is not None:
            if not (result.pose_landmarks and result.pose_world_landmarks):
                self.get_logger().warn(
                    "upper limb: no body detected. Pose needs your torso and "
                    "both shoulders in frame - considerably more of you than "
                    "hand tracking does.",
                    throttle_duration_sec=5.0,
                )
            if result.pose_landmarks and result.pose_world_landmarks:
                lm = result.pose_landmarks[0]
                world = result.pose_world_landmarks[0]
                P, Q = [], []
                for i in range(11, 25):
                    if getattr(lm[i], "visibility", 1.0) < 0.3:
                        continue
                    u, v = lm[i].x * w, lm[i].y * h
                    z = sample_depth(self._depth, u, v, 4, self._depth_scale)
                    if not np.isfinite(z) or z <= 0:
                        continue
                    P.append([world[i].x, world[i].y, world[i].z])
                    Q.append(deproject(u, v, z, fx, fy, cx, cy))

                if len(P) < 3:
                    self.get_logger().warn(
                        f"upper limb: only {len(P)} body landmarks had valid "
                        "depth (need 3). The colour camera sees more of you "
                        "than NFOV depth does; depth_mode:=WFOV_2X2BINNED is "
                        "wider at the same frame rate.",
                        throttle_duration_sec=5.0,
                    )
                if len(P) >= 3:
                    try:
                        if self.body_scale:
                            # Scale is known from calibration, so only R and t
                            # are solved. One fewer unknown matters a lot when
                            # 3-6 points carry depth: measured 40 mm shoulder
                            # error against 54 mm with a free scale.
                            R, tv, s, inl, res = kabsch_ransac(
                                np.asarray(P) * self.body_scale,
                                np.asarray(Q), threshold=0.05, scaled=False,
                            )
                            s = self.body_scale
                            R_use, t_use = R, tv
                        else:
                            R, tv, s, inl, res = kabsch_ransac(
                                np.asarray(P), np.asarray(Q), threshold=0.05
                            )
                            R_use, t_use = R, tv
                        if res >= 0.25:
                            self.get_logger().warn(
                                f"upper limb: fit rejected, residual "
                                f"{res*1000:.0f} mm from {len(P)} points",
                                throttle_duration_sec=5.0,
                            )
                        if res < 0.25:
                            if not getattr(self, "_limb_ok_logged", False):
                                self._limb_ok_logged = True
                                self.get_logger().info(
                                    f"upper limb tracking: {res*1000:.0f} mm "
                                    f"from {len(inl)} points"
                                )
                            self._fit = (R_use, t_use, s, len(inl), res)
                            self._body_pts = np.array([
                                [world[i].x, world[i].y, world[i].z]
                                for i in idx
                            ])
                            self._limb_pixels = [
                                lm[i].x for i in idx
                            ], [lm[i].y for i in idx]
                    except Exception as exc:
                        self.get_logger().warn(
                            f"upper limb fit failed: {exc}",
                            throttle_duration_sec=5.0,
                        )

        if self._fit is None or self._body_pts is None:
            return

        R, tv, s, n_in, res = self._fit
        pts = np.array([s * (R @ p) + tv for p in self._body_pts])

        if self.session is not None and self.session.step == "limb":
            self.session.observe_limb(
                s,
                float(np.linalg.norm(pts[1] - pts[0])),
                float(np.linalg.norm(pts[2] - pts[1])),
            )

        # The hand's wrist, from THIS frame's hand landmarks and this frame's
        # depth. Not a separate estimate arriving late over a topic.
        if hand_pixels is not None:
            u, v = hand_pixels[0]
            z = sample_depth(self._depth, u, v, 3, self._depth_scale)
            if np.isfinite(z) and z > 0:
                wrist = deproject(u, v, z, fx, fy, cx, cy)
                correction = wrist - pts[2]
                pts[2] = wrist
                pts[1] = pts[1] + 0.5 * correction
                if self._limb_pixels is not None:
                    xs, ys = self._limb_pixels
                    xs = list(xs); ys = list(ys)
                    xs[2], ys[2] = u / w, v / h     # draw at the hand's wrist
                    self._limb_pixels = (xs, ys)

        m = Float32MultiArray()
        m.data = [float(x) for x in pts.reshape(-1)]
        self.pub_limb.publish(m)

        if self._limb_pixels is not None:
            xs, ys = self._limb_pixels
            px = Float32MultiArray()
            px.data = [float(v) for pair in zip(xs, ys) for v in pair]
            self.pub_limb_px.publish(px)
            self._limb_px = px.data          # drawn by _annotate in this node

        sw = swivel_from_points(pts[0], pts[1], pts[2])
        s_msg = Float32MultiArray()
        s_msg.data = [
            float(sw) if np.isfinite(sw) else 0.0,
            float(np.isfinite(sw)), float(n_in), float(res), float(s),
        ]
        self.pub_swivel.publish(s_msg)

        # One message, one timestamp: shoulder, elbow, wrist, then the 6 hand
        # joint targets. The whole anatomy as a single sample.
        sk = Float32MultiArray()
        sk.data = [float(x) for x in pts.reshape(-1)] + [
            float(x) for x in self.last_targets
        ]
        self.pub_skeleton.publish(sk)

    def _pose_worker(self):
        """Background pose inference. Publishes nothing; only hands back a
        result for the main loop to fit."""
        import threading

        while self._pose_running:
            if not self._pose_event.wait(timeout=0.2):
                continue
            self._pose_event.clear()
            job = self._pose_input
            if job is None:
                self._pose_busy = False
                continue
            mp_image, shape, idx = job
            try:
                self._pose_result = self.pose_landmarker.detect_for_video(
                    mp_image, int(idx * 1000.0 / max(self.publish_hz, 1.0))
                )
            except Exception as exc:
                self.get_logger().warn(
                    f"pose inference failed: {exc}", throttle_duration_sec=5.0
                )
            finally:
                self._pose_busy = False

    def _limb_px_cb(self, msg):
        d = list(msg.data)
        self._limb_px = d if len(d) >= 6 else None

    def _publish_landmarks(self, pixel_lm, world_lm):
        """21 x (u, v) pixels then 21 x (x, y, z) world metres, flattened.

        Pixel coordinates are in the frame teleop received, before any
        downscaling for the UI, so they index the depth image directly.
        """
        m = self._F32()
        m.data = [float(v) for v in pixel_lm.reshape(-1)] + [
            float(v) for v in np.asarray(world_lm).reshape(-1)
        ]
        self.pub_landmarks.publish(m)

    def _publish_reference(self):
        m = self._F32()
        m.data = [float(self.msj_reference),
                  float(self.msj_spread or 0.0)]
        self.pub_reference.publish(m)

    def _publish_latency(self):
        m = self._F32()
        m.data = [float(self.source_age_ms)]
        self.pub_latency.publish(m)

    def _publish_calib_status(self):
        m = self._String()
        if self.session is None:
            m.data = ""
        else:
            s = self.session
            n = len(s.STEPS)
            i = min(s.step_index + 1, n)
            text = f"STEP {i}/{n}  {s.prompt()}"
            if s.step == "baseline":
                text += f"   [{s.baseline_count}/30 samples]"
            m.data = text
        self.pub_calib.publish(m)

    def _annotate(self, frame, result):
        """Draw the landmark skeleton; shared by the preview and the UI feed."""
        cv2 = self.cv2
        frame = frame.copy()
        h, w = frame.shape[:2]
        CONNECTIONS = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17),
        ]
        if result.hand_landmarks:
            lm = result.hand_landmarks[0]
            for a, b in CONNECTIONS:
                pa = (int(lm[a].x * w), int(lm[a].y * h))
                pb = (int(lm[b].x * w), int(lm[b].y * h))
                cv2.line(frame, pa, pb, (255, 255, 255), 2)
            for p in lm:
                cv2.circle(frame, (int(p.x * w), int(p.y * h)), 4, (0, 0, 255), -1)
        else:
            cv2.putText(frame, "no hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

        # Upper limb: shoulder -> elbow -> wrist, drawn in a different colour
        # so it reads as a separate signal from the hand skeleton.
        if self._limb_px is not None:
            pts = [
                (int(self._limb_px[i] * w), int(self._limb_px[i + 1] * h))
                for i in range(0, 6, 2)
            ]
            for a, b in ((0, 1), (1, 2)):
                cv2.line(frame, pts[a], pts[b], (255, 200, 60), 3,
                         cv2.LINE_AA)
            for p, label in zip(pts, ("S", "E", "W")):
                cv2.circle(frame, p, 7, (255, 200, 60), -1, cv2.LINE_AA)
                cv2.circle(frame, p, 7, (30, 30, 30), 1, cv2.LINE_AA)
                cv2.putText(frame, label, (p[0] + 10, p[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 200, 60), 1,
                            cv2.LINE_AA)
        return frame

    def _show(self, frame, world_lm):
        cv2 = self.cv2
        h, w = frame.shape[:2]

        # Target bars
        for i, (label, val) in enumerate(zip(JOINT_LABELS, self.last_targets)):
            lo, hi = HAND_LIMITS_DEG[i]
            frac = abs(val - (lo if hi > 0 else hi)) / max(abs(hi - lo), 1e-6)
            y = h - 130 + i * 20
            cv2.putText(frame, f"{label:>10} {val:7.1f}", (10, y + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            c = msj_color(
                float(self.last_msj[i]),
                self.msj_reference,
                spread=self.msj_spread,
                direction=self.msj_direction,
            )
            bgr = (int(c[2] * 255), int(c[1] * 255), int(c[0] * 255))
            cv2.rectangle(frame, (170, y), (170 + int(160 * frac), y + 12),
                          bgr, -1)
            cv2.rectangle(frame, (170, y), (330, y + 12), (90, 90, 90), 1)

        if self.session is not None:
            cv2.rectangle(frame, (0, 0), (w, 58), (0, 0, 0), -1)
            cv2.putText(frame, "CALIBRATING - commands paused", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            cv2.putText(frame, self.session.prompt(), (10, 46),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow("Ability Hand teleop  [k]recalibrate  [s]ave  [q]uit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key != 255 and world_lm is not None:
            self._handle_key(key, world_lm)
        elif key == ord("q"):
            rclpy.try_shutdown()

    def _handle_key(self, key, world_lm):
        r = self.retargeter
        f = r.raw_features(world_lm)

        if key == ord("q"):
            rclpy.try_shutdown()
            return

        if self.session is not None:
            self._calibration_key(key, f)
            return

        # Already calibrated: 'k' starts a fresh session, 's' re-saves
        if key == ord("k"):
            self.session = CalibrationSession()
            self.get_logger().warn(
                "Recalibrating. Commands paused. " + self.session.prompt()
            )
        elif key == ord("s"):
            self._save_current()

    EXPECTED_KEY = {"open": "o", "closed": "c", "baseline": "b", "limb": "l"}

    def _calibration_key(self, key, features):
        s = self.session
        step = s.step
        want = self.EXPECTED_KEY.get(step)
        got = chr(key).lower()
        if want and got != want:
            self.get_logger().warn(
                f"'{got}' does not apply here. Current step is '{step}': "
                f"{s.prompt()}  (press '{want}')"
            )
            return
        if step == "open" and key == ord("o"):
            s.capture_open(features)
            self.get_logger().info(
                f"OPEN captured (curl {s.values['curl_open']:.1f}, "
                f"thumb opp {s.values['thumb_opp_open']:.1f}). "
                f"NEXT: {s.prompt()}"
            )
        elif step == "closed" and key == ord("c"):
            s.capture_closed(features)
            problems = s.validate()
            for p in problems:
                self.get_logger().warn(f"calibration: {p}")
            # Apply the ranges now so the baseline step is measured on the
            # same mapping that will be used afterwards
            for k, v in s.values.items():
                if hasattr(self.retargeter, k):
                    setattr(self.retargeter, k, v)
            self.msj.reset()
            self.get_logger().info(
                f"CLOSED captured (curl {s.values['curl_closed']:.1f}, "
                f"thumb opp {s.values['thumb_opp_closed']:.1f}). "
                f"NEXT: {s.prompt()}"
            )
        elif step == "limb" and key == ord("l"):
            if s.limb_count < 15:
                self.get_logger().warn(
                    f"only {s.limb_count}/15 limb samples. Face the camera "
                    "with your arm visible for a few seconds."
                )
                return
            s.capture_limb()
            self._finish_calibration()
        elif step == "baseline" and key == ord("b"):
            if s.baseline_count < 30:
                self.get_logger().warn(
                    f"Only {s.baseline_count}/30 samples so far. Keep moving "
                    "your hand for a few seconds, then press 'b' again."
                )
                return
            s.capture_baseline()
            self.get_logger().info(f"BASELINE captured. NEXT: {s.prompt()}")

    def _finish_calibration(self):
        s = self.session
        for k, v in s.values.items():
            if hasattr(self.retargeter, k):
                setattr(self.retargeter, k, v)
        if "msj_reference" in s.values:
            self.msj_reference = s.values["msj_reference"]
        if "msj_spread" in s.values:
            self.msj_spread = s.values["msj_spread"]
        if "body_scale" in s.values:
            self.body_scale = s.values["body_scale"]
            self.get_logger().info(
                f"body scale {self.body_scale:.3f}, upper arm "
                f"{s.values.get('upper_arm_m', 0)*100:.1f} cm, forearm "
                f"{s.values.get('forearm_m', 0)*100:.1f} cm"
            )
        path = save_calibration(s.values, self.calibration_file)
        self.session = None
        self.retargeter.reset()
        self.get_logger().info(
            f"Calibration complete and saved to {path}. "
            f"MSJ reference {self.msj_reference:.1f} "
            f"+/- {self.msj_spread:.1f} from {s.values.get('msj_samples', 0)} "
            "samples; that motion is now the green centre. Commands resumed."
        )

    def _save_current(self):
        r = self.retargeter
        vals = {
            "curl_open": r.curl_open,
            "curl_closed": r.curl_closed,
            "thumb_flex_open": r.thumb_flex_open,
            "thumb_flex_closed": r.thumb_flex_closed,
            "thumb_opp_open": r.thumb_opp_open,
            "thumb_opp_closed": r.thumb_opp_closed,
            "msj_reference": self.msj_reference,
            "msj_spread": self.msj_spread or 0.0,
        }
        path = save_calibration(vals, self.calibration_file)
        self.get_logger().info(f"Calibration saved to {path}")

    def destroy_node(self):
        try:
            self._pose_running = False
            self._grab_running = False
            if self.cap is not None:
                self.cap.release()
            if self.preview:
                self.cv2.destroyAllWindows()
        except Exception:
            pass
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MediapipeTeleopNode()
    from rclpy.executors import MultiThreadedExecutor

    executor = MultiThreadedExecutor(num_threads=3)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
