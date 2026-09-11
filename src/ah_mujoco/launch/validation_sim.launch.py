"""Run the validation pipeline against MuJoCo with no hardware attached.

    virtual_hand  <--target/position--  real_validation_node
         |                                      ^
         +--- feedback/{position,velocity,touch}+
         |
         +--- /joint_states_ah ---> mujoco_viewer
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "hand_side",
                default_value="Right",
                description="Ability Hand Side",
                choices=["Right", "Left"],
            ),
            DeclareLaunchArgument(
                "hand_size",
                default_value="Large",
                description="Ability Hand Size",
                choices=["Small", "Large"],
            ),
            DeclareLaunchArgument(
                "validation",
                default_value="True",
                description="Also start ah_tests/real_validation_node",
                choices=["True", "False"],
            ),
            # Stands in for the hardware + ah_node
            Node(
                package="ah_mujoco",
                executable="virtual_hand",
                name="virtual_hand",
                output="screen",
                parameters=[{"hand_side": LaunchConfiguration("hand_side")}],
            ),
            Node(
                package="ah_mujoco",
                executable="mujoco_viewer",
                name="mujoco_viewer",
                output="screen",
                parameters=[
                    {"hand_side": LaunchConfiguration("hand_side")},
                    {"hand_size": LaunchConfiguration("hand_size")},
                ],
            ),
            Node(
                package="ah_tests",
                executable="real_validation_node",
                name="real_validation_node",
                output="screen",
                condition=IfCondition(LaunchConfiguration("validation")),
            ),
        ]
    )
