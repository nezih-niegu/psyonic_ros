"""Webcam -> MuJoCo physics/safety -> viewer. No hardware.

    mediapipe_teleop --raw--> mujoco_safety --target--> (no hardware)
                                    |
                                    +-- /joint_states_ah --> mujoco_viewer

The viewer shows the SIMULATED state, i.e. the commands that would actually
be sent to the hand after safety filtering.
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
        DeclareLaunchArgument("max_speed_dps", default_value="200.0"),
        DeclareLaunchArgument("max_torque", default_value="4.0"),
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
                 {"output": "raw"},
             ]),
        Node(package="ah_mujoco", executable="mujoco_safety",
             name="mujoco_safety", output="screen",
             parameters=[
                 {"hand_side": LaunchConfiguration("hand_side")},
                 {"hand_size": LaunchConfiguration("hand_size")},
                 {"max_speed_dps": LaunchConfiguration("max_speed_dps")},
                 {"max_torque": LaunchConfiguration("max_torque")},
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
