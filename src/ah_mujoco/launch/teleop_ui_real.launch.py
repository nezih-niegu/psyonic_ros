"""Unified clinical UI driving the REAL hand.

    mediapipe_teleop --raw--> mujoco_safety --target--> ah_node --> HARDWARE
            |                        |
            +- camera/image_raw      +- /joint_states_ah -> clinical_ui

mujoco_safety is the ONLY publisher on target/position. Do not run
virtual_hand or real_validation_node alongside this.

Calibration is done in the UI window: o (flat hand), c (fist), b (move
naturally). Until it completes, teleop publishes nothing and the safety node
withholds targets, so the hand does not move.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("port", default_value="/dev/ttyUSB0",
                              description="check with ls -l /dev/ttyUSB* /dev/ttyACM*"),
        DeclareLaunchArgument("baud_rate", default_value="0"),
        DeclareLaunchArgument("hand_side", default_value="Right",
                              choices=["Right", "Left"]),
        DeclareLaunchArgument("hand_size", default_value="Large",
                              choices=["Small", "Large"]),
        DeclareLaunchArgument("camera_index", default_value="0"),
        DeclareLaunchArgument("hand_pose", default_value="False",
                              choices=["True", "False"],
                              description="publish landmarks for hand_pose_node"),
        DeclareLaunchArgument("mirror", default_value="True",
                              choices=["True", "False"],
                              description="set False when using hand_pose"),
        DeclareLaunchArgument("image_qos", default_value="auto",
                              choices=["auto", "sensor", "reliable"],
                              description="auto subscribes both ways"),
        DeclareLaunchArgument(
            "image_topic", default_value="",
            description="If set, take frames from this sensor_msgs/Image "
                        "topic instead of opening the local camera"),
        DeclareLaunchArgument("model_path", default_value="hand_landmarker.task"),
        DeclareLaunchArgument("force_calibration", default_value="False",
                              choices=["True", "False"],
                              description="ignore any saved calibration and "
                                          "run a fresh session"),
        DeclareLaunchArgument(
            "calibration_file",
            default_value="/ws/src/ah_mujoco/config/teleop_calibration.json"),
        DeclareLaunchArgument("theme", default_value="dark",
                              choices=["dark", "light"]),
        DeclareLaunchArgument("window_width", default_value="1400"),
        DeclareLaunchArgument("window_height", default_value="800"),
        DeclareLaunchArgument("preview", default_value="False",
                              choices=["True", "False"],
                              description="extra cv2 window; not needed, the "
                                          "UI can calibrate"),
        # Start conservative on hardware and raise once it feels right.
        DeclareLaunchArgument("max_speed_dps", default_value="300.0",
                              description="URDF limit is ~462 deg/s"),
        DeclareLaunchArgument("max_torque", default_value="3.0"),
        DeclareLaunchArgument("contact_engage", default_value="25.0",
                              description="FSR level at which a digit stops "
                                          "closing further"),
        DeclareLaunchArgument("contact_release", default_value="12.0"),
        DeclareLaunchArgument("use_contact_guard", default_value="True",
                              choices=["True", "False"]),

        Node(package="ah_ros_py", executable="ah_node", name="ah_node",
             output="screen",
             parameters=[
                 {"port": LaunchConfiguration("port")},
                 {"baud_rate": LaunchConfiguration("baud_rate")},
                 {"hand_side": LaunchConfiguration("hand_side")},
                 {"write_thread": False},
                 # mujoco_safety publishes joint states from the simulated
                 # state, so ah_node must not publish competing ones
                 {"js_publisher": False},
                 {"simulated_hand": False},
                 {"reply_mode": 1},
             ]),
        Node(package="ah_mujoco", executable="mediapipe_teleop",
             name="mediapipe_teleop", output="screen",
             parameters=[
                 {"hand_side": LaunchConfiguration("hand_side")},
                 {"camera_index": LaunchConfiguration("camera_index")},
                 {"image_topic": LaunchConfiguration("image_topic")},
                 {"image_qos": LaunchConfiguration("image_qos")},
                 {"publish_landmarks": LaunchConfiguration("hand_pose")},
                 {"mirror": LaunchConfiguration("mirror")},
                 {"model_path": LaunchConfiguration("model_path")},
                 {"calibration_file": LaunchConfiguration("calibration_file")},
                 {"force_calibration":
                  LaunchConfiguration("force_calibration")},
                 {"preview": LaunchConfiguration("preview")},
                 {"publish_image": True},
                 {"output": "raw"},
             ]),
        Node(package="ah_mujoco", executable="mujoco_safety",
             name="mujoco_safety", output="screen",
             parameters=[
                 {"hand_side": LaunchConfiguration("hand_side")},
                 {"hand_size": LaunchConfiguration("hand_size")},
                 {"max_speed_dps": LaunchConfiguration("max_speed_dps")},
                 {"max_torque": LaunchConfiguration("max_torque")},
                 {"contact_engage": LaunchConfiguration("contact_engage")},
                 {"contact_release": LaunchConfiguration("contact_release")},
                 {"use_contact_guard":
                  LaunchConfiguration("use_contact_guard")},
             ]),
        Node(package="ah_mujoco", executable="clinical_ui",
             name="clinical_ui", output="screen",
             parameters=[
                 {"hand_side": LaunchConfiguration("hand_side")},
                 {"hand_size": LaunchConfiguration("hand_size")},
                 {"calibration_file": LaunchConfiguration("calibration_file")},
                 {"theme": LaunchConfiguration("theme")},
                 {"window_width": LaunchConfiguration("window_width")},
                 {"window_height": LaunchConfiguration("window_height")},
             ]),
    ])
