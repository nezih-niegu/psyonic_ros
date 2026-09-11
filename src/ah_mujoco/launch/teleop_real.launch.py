"""Webcam hand tracking -> REAL Ability Hand, mirrored in MuJoCo.

Do not run virtual_hand alongside this: ah_node owns the serial port and
publishes the real feedback topics.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("baud_rate", default_value="0"),
        DeclareLaunchArgument("hand_side", default_value="Right",
                              choices=["Right", "Left"]),
        DeclareLaunchArgument("hand_size", default_value="Large",
                              choices=["Small", "Large"]),
        DeclareLaunchArgument("camera_index", default_value="0"),
        DeclareLaunchArgument("force_calibration", default_value="False",
                              choices=["True", "False"],
                              description="ignore any saved calibration and "
                                          "run a fresh session"),
        DeclareLaunchArgument(
            "calibration_file",
            default_value="/ws/src/ah_mujoco/config/teleop_calibration.json",
            description="Kept in the mounted workspace so it survives "
                        "container rebuilds"),
        DeclareLaunchArgument("model_path", default_value="hand_landmarker.task"),
        Node(package="ah_ros_py", executable="ah_node", name="ah_node",
             output="screen",
             parameters=[
                 {"port": LaunchConfiguration("port")},
                 {"baud_rate": LaunchConfiguration("baud_rate")},
                 {"hand_side": LaunchConfiguration("hand_side")},
                 {"write_thread": False},
                 {"js_publisher": True},
                 {"simulated_hand": False},
                 {"reply_mode": 1},
             ]),
        Node(package="ah_mujoco", executable="mediapipe_teleop",
             name="mediapipe_teleop", output="screen",
             parameters=[
                 {"hand_side": LaunchConfiguration("hand_side")},
                 {"camera_index": LaunchConfiguration("camera_index")},
                 {"model_path": LaunchConfiguration("model_path")},
                 {"calibration_file": LaunchConfiguration("calibration_file")},
                 {"force_calibration":
                  LaunchConfiguration("force_calibration")},
             ]),
        Node(package="ah_mujoco", executable="mujoco_viewer",
             name="mujoco_viewer", output="screen",
             parameters=[
                 {"hand_side": LaunchConfiguration("hand_side")},
                 {"hand_size": LaunchConfiguration("hand_size")},
                 {"follow_targets": False},
                 {"calibration_file": LaunchConfiguration("calibration_file")},
             ]),
    ])
