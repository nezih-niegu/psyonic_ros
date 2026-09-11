"""Validation against the REAL hand, mirrored in MuJoCo.

    ah_node <--target/position-- real_validation_node
       |                                  ^
       |  (serial)                        |
       v                                  |
    HARDWARE --feedback/{position,velocity,touch}--+
       |
       +-- /joint_states_ah --> mujoco_viewer

Do NOT run virtual_hand alongside this: it publishes on the same feedback
topics and the validation node would follow the fake hand instead.

ah_node must have js_publisher:=True for the viewer to move, and reply_mode 1
(position, velocity, touch) for the validation node to unblock.
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
                "port",
                default_value="/dev/ttyUSB0",
                description="Serial port the hand enumerates on "
                "(often /dev/ttyACM0)",
            ),
            DeclareLaunchArgument(
                "baud_rate", default_value="0", description="Baud Rate"
            ),
            DeclareLaunchArgument(
                "hand_side",
                default_value="Right",
                description="Ability Hand Side. Note: real_validation_node "
                "hardcodes the right-hand topics.",
                choices=["Right", "Left"],
            ),
            DeclareLaunchArgument(
                "hand_size",
                default_value="Large",
                description="Ability Hand Size",
                choices=["Small", "Large"],
            ),
            DeclareLaunchArgument(
                "write_thread",
                default_value="False",
                description="False means ah_node sends a command on each "
                "incoming target message",
                choices=["True", "False"],
            ),
            DeclareLaunchArgument(
                "validation",
                default_value="True",
                description="Also start ah_tests/real_validation_node",
                choices=["True", "False"],
            ),
            # Real driver: owns the serial port
            Node(
                package="ah_ros_py",
                executable="ah_node",
                name="ah_node",
                output="screen",
                parameters=[
                    {"port": LaunchConfiguration("port")},
                    {"baud_rate": LaunchConfiguration("baud_rate")},
                    # ah_node.launch.py omits hand_side; pass it explicitly
                    {"hand_side": LaunchConfiguration("hand_side")},
                    {"write_thread": LaunchConfiguration("write_thread")},
                    {"js_publisher": True},
                    {"simulated_hand": False},
                    {"reply_mode": 1},
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
                    # Mirror real feedback only; don't fall back to targets
                    {"follow_targets": False},
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
