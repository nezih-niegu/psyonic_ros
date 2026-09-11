"""Run the trained ONNX policy against the virtual hand, visualized in MuJoCo.

    virtual_hand  <--target/position--  ah_policy_node
         |                                    ^
         +-- feedback/{position,velocity,touch}+
         +-- /joint_states  (mimic q2 source) -+
         |
         +-- /joint_states_ah ---> mujoco_viewer

Note the non-zero initial pose: the policy has a fixed point at exactly zero
with zero contact, so a hand parked at zero never starts moving.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
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
            Node(
                package="ah_mujoco",
                executable="virtual_hand",
                name="virtual_hand",
                output="screen",
                parameters=[
                    {"hand_side": LaunchConfiguration("hand_side")},
                    # Seeded off the policy's zero fixed point
                    {
                        "initial_position_deg": [
                            10.0, 10.0, 10.0, 10.0, -10.0, 10.0
                        ]
                    },
                ],
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
                package="ah_driver",
                executable="ah_policy_node",
                name="ah_policy_node",
                output="screen",
            ),
        ]
    )
