"""Webcam hand tracking -> virtual hand -> MuJoCo. No hardware needed."""

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
        DeclareLaunchArgument(
            "model_path", default_value="hand_landmarker.task",
            description="HandLandmarker .task bundle (get_model.sh)"),
        DeclareLaunchArgument(
            "calibration_file",
            default_value="/ws/src/ah_mujoco/config/teleop_calibration.json"),
        DeclareLaunchArgument("force_calibration", default_value="False",
                              choices=["True", "False"],
                              description="ignore any saved calibration"),
        DeclareLaunchArgument("preview", default_value="True",
                              choices=["True", "False"]),
        Node(package="ah_mujoco", executable="mediapipe_teleop",
             name="mediapipe_teleop", output="screen",
             parameters=[
                 {"hand_side": LaunchConfiguration("hand_side")},
                 {"camera_index": LaunchConfiguration("camera_index")},
                 {"model_path": LaunchConfiguration("model_path")},
                 {"calibration_file": LaunchConfiguration("calibration_file")},
                 {"force_calibration":
                  LaunchConfiguration("force_calibration")},
                 {"preview": LaunchConfiguration("preview")},
             ]),
        Node(package="ah_mujoco", executable="virtual_hand",
             name="virtual_hand", output="screen",
             parameters=[{"hand_side": LaunchConfiguration("hand_side")}]),
        Node(package="ah_mujoco", executable="mujoco_viewer",
             name="mujoco_viewer", output="screen",
             parameters=[
                 {"hand_side": LaunchConfiguration("hand_side")},
                 {"hand_size": LaunchConfiguration("hand_size")},
             ]),
    ])
