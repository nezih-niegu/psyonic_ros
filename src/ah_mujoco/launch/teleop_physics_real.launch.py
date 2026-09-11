"""Full real-time chain: webcam -> MuJoCo physics/safety -> REAL hand.

    mediapipe_teleop --raw--> mujoco_safety --target--> ah_node --> HARDWARE
                                    |
                                    +-- /joint_states_ah --> mujoco_viewer

mujoco_safety is the ONLY publisher on target/position. Do not run
real_validation_node or virtual_hand alongside this: both would publish
competing commands on the same topic.
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
        DeclareLaunchArgument("max_speed_dps", default_value="150.0"),
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
                 # The safety node publishes joint states; ah_node need not
                 {"js_publisher": False},
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
        Node(package="ah_mujoco", executable="mujoco_viewer",
             name="mujoco_viewer", output="screen",
             parameters=[
                 {"hand_side": LaunchConfiguration("hand_side")},
                 {"hand_size": LaunchConfiguration("hand_size")},
                 {"follow_targets": False},
                 {"calibration_file": LaunchConfiguration("calibration_file")},
             ]),
    ])
