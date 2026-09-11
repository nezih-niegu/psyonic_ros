"""Shoulder and elbow tracking alongside the hand pipeline.

Needs the MediaPipe Pose model:

    cd /ws/src/HandPose
    wget -O pose_landmarker.task \
      https://storage.googleapis.com/mediapipe-models/pose_landmarker/\
pose_landmarker_lite/float16/1/pose_landmarker_lite.task
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("hand_side", default_value="Right",
                              choices=["Right", "Left"]),
        DeclareLaunchArgument("image_topic", default_value="/rgb/image_raw"),
        DeclareLaunchArgument("depth_topic",
                              default_value="/depth_to_rgb/image_raw"),
        DeclareLaunchArgument(
            "model_path",
            default_value="/ws/src/HandPose/pose_landmarker.task"),
        DeclareLaunchArgument("publish_hz", default_value="60.0",
                              description="output rate; match the hand "
                                          "pipeline so the limb does not lag"),
        DeclareLaunchArgument("rate_hz", default_value="15.0",
                              description="pose inference is heavier than "
                                          "hands; 15 Hz is plenty for a "
                                          "shoulder anchor"),
        Node(package="ah_mujoco", executable="upper_limb",
             name="upper_limb", output="screen",
             parameters=[
                 {"hand_side": LaunchConfiguration("hand_side")},
                 {"image_topic": LaunchConfiguration("image_topic")},
                 {"depth_topic": LaunchConfiguration("depth_topic")},
                 {"model_path": LaunchConfiguration("model_path")},
                 {"rate_hz": LaunchConfiguration("rate_hz")},
                 {"publish_hz": LaunchConfiguration("publish_hz")},
             ]),
    ])
