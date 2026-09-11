"""SE(3) hand pose from Kinect depth, alongside the teleop pipeline.

Adds hand_pose to the UI stack. Teleop publishes its landmarks so MediaPipe
runs once, and hand_pose combines them with registered depth.

Use the REGISTERED depth topic (/depth_to_rgb/image_raw): landmark pixels come
from the colour image, and unregistered depth has different intrinsics and a
few cm of baseline, so lookups would sample the wrong points.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("hand_side", default_value="Right",
                              choices=["Right", "Left"]),
        DeclareLaunchArgument("depth_topic",
                              default_value="/depth_to_rgb/image_raw"),
        DeclareLaunchArgument("camera_info_topic",
                              default_value="/rgb/camera_info"),
        DeclareLaunchArgument("palm_only", default_value="False",
                              choices=["True", "False"],
                              description="fit palm landmarks only; less "
                                          "accurate, see README"),
        DeclareLaunchArgument("alpha_t", default_value="0.5"),
        DeclareLaunchArgument("alpha_r", default_value="0.5"),
        DeclareLaunchArgument("publish_tf", default_value="True",
                              choices=["True", "False"]),

        Node(package="ah_mujoco", executable="hand_pose",
             name="hand_pose", output="screen",
             parameters=[
                 {"hand_side": LaunchConfiguration("hand_side")},
                 {"depth_topic": LaunchConfiguration("depth_topic")},
                 {"camera_info_topic":
                  LaunchConfiguration("camera_info_topic")},
                 {"palm_only": LaunchConfiguration("palm_only")},
                 {"alpha_t": LaunchConfiguration("alpha_t")},
                 {"alpha_r": LaunchConfiguration("alpha_r")},
                 {"publish_tf": LaunchConfiguration("publish_tf")},
             ]),
    ])
