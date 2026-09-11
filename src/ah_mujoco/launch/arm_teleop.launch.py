"""Hand pose -> xArm Lite 6 wrist target.

Runs alongside teleop_ui_real + hand_pose. The hand keeps controlling fingers;
the arm carries the wrist, because the Ability Hand has no wrist DOF.

The two transforms below MUST be measured for your setup. Defaults are
identity, which is certainly wrong:

  base_cam_*    camera pose in the arm base frame (hand-eye calibration)
  hand_flange_* how the hand is bolted to the flange (mechanical drawing)
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("hand_side", default_value="Right",
                              choices=["Right", "Left"]),
        DeclareLaunchArgument("position_scale", default_value="1.0",
                              description="hand travel -> arm travel"),
        DeclareLaunchArgument("rotation_scale", default_value="1.0"),
        DeclareLaunchArgument("max_joint_step", default_value="0.05",
                              description="rad per solve, joint-space rate cap"),
        DeclareLaunchArgument("publish_hz", default_value="50.0"),

        Node(package="ah_mujoco", executable="arm_teleop",
             name="arm_teleop", output="screen",
             parameters=[
                 {"hand_side": LaunchConfiguration("hand_side")},
                 {"position_scale": LaunchConfiguration("position_scale")},
                 {"rotation_scale": LaunchConfiguration("rotation_scale")},
                 {"max_joint_step": LaunchConfiguration("max_joint_step")},
                 {"publish_hz": LaunchConfiguration("publish_hz")},
                 # MEASURE THESE:
                 {"base_cam_xyz": [0.0, 0.0, 0.0]},
                 {"base_cam_rpy": [0.0, 0.0, 0.0]},
                 {"hand_flange_xyz": [0.0, 0.0, 0.0]},
                 {"hand_flange_rpy": [0.0, 0.0, 0.0]},
             ]),
    ])
