"""Unified clinical UI: webcam tracking + MuJoCo physics in one window.

    mediapipe_teleop --raw--> mujoco_safety --target--> (no hardware)
            |                        |
            +-- camera/image_raw ----+-- /joint_states_ah --> clinical_ui

Teleop runs with preview:=False so cv2 does not open a second window; its
camera feed and calibration prompts are published for the UI instead.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
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
                              description="cv2 window; needed to calibrate"),

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
