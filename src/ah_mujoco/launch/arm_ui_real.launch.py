"""Full arm mode: hand tracking -> Lite 6 + Ability Hand, all in MuJoCo.

Standalone MuJoCo throughout - no Gazebo, no RViz. The UI shows the arm with
the hand mounted, driven by the same physics the safety layer runs.

    mediapipe_teleop --raw--> mujoco_safety --target--> ah_node --> HAND
            |                                 |
            +--> hand_pose --> arm_teleop --> /xarm/target_joints --> UI
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("port", default_value="/dev/ttyUSB0"),
        DeclareLaunchArgument("hand_side", default_value="Right",
                              choices=["Right", "Left"]),
        DeclareLaunchArgument("hand_size", default_value="Large",
                              choices=["Small", "Large"]),
        DeclareLaunchArgument("model_path",
                              default_value="/ws/HandPose/hand_landmarker.task"),
        DeclareLaunchArgument("image_topic", default_value="/rgb/image_raw"),
        DeclareLaunchArgument("depth_topic",
                              default_value="/depth_to_rgb/image_raw"),
        DeclareLaunchArgument(
            "calibration_file",
            default_value="/ws/src/ah_mujoco/config/teleop_calibration.json"),
        DeclareLaunchArgument("force_calibration", default_value="False",
                              choices=["True", "False"]),
        DeclareLaunchArgument("xarm_description_path", default_value=""),
        DeclareLaunchArgument("upper_limb", default_value="True",
                              choices=["True", "False"],
                              description="track shoulder and elbow in the "
                                          "SAME node and frame as the hand"),
        DeclareLaunchArgument(
            "pose_model_path",
            default_value="/ws/src/HandPose/pose_landmarker.task"),
        DeclareLaunchArgument("pose_every_n", default_value="3"),
        DeclareLaunchArgument("theme", default_value="dark",
                              choices=["dark", "light"]),
        DeclareLaunchArgument("position_scale", default_value="0.3",
                              description="start small on hardware"),
        DeclareLaunchArgument("robot_ip", default_value="192.168.1.154"),
        DeclareLaunchArgument("arm_state_topic",
                              default_value="/ufactory/joint_states",
                              description="real robot joint feedback; check "
                                          "with ros2 topic list | grep joint"),
        DeclareLaunchArgument("service_prefix", default_value="/ufactory",
                              description="xarm_api namespace; the Lite 6 "
                                          "driver uses /ufactory"),
        DeclareLaunchArgument("use_arm_hardware", default_value="True",
                              choices=["True", "False"],
                              description="run the xArm bridge; False is "
                                          "MuJoCo only"),
        DeclareLaunchArgument("safe_pose",
                              default_value="[0.0, -0.5236, 0.8727, 0.0, 0.6981, 0.0]",
                              description="joint angles in radians"),

        Node(package="ah_ros_py", executable="ah_node", name="ah_node",
             output="screen",
             parameters=[
                 {"port": LaunchConfiguration("port")},
                 {"hand_side": LaunchConfiguration("hand_side")},
                 {"write_thread": False}, {"js_publisher": False},
                 {"simulated_hand": False}, {"reply_mode": 1},
             ]),
        Node(package="ah_mujoco", executable="mediapipe_teleop",
             name="mediapipe_teleop", output="screen",
             parameters=[
                 {"hand_side": LaunchConfiguration("hand_side")},
                 {"model_path": LaunchConfiguration("model_path")},
                 {"image_topic": LaunchConfiguration("image_topic")},
                 {"calibration_file": LaunchConfiguration("calibration_file")},
                 {"force_calibration":
                  LaunchConfiguration("force_calibration")},
                 {"publish_image": True}, {"publish_landmarks": True},
                 {"upper_limb": LaunchConfiguration("upper_limb")},
                 {"pose_model_path": LaunchConfiguration("pose_model_path")},
                 {"pose_every_n": LaunchConfiguration("pose_every_n")},
                 {"depth_topic": LaunchConfiguration("depth_topic")},
                 {"mirror": False}, {"preview": False}, {"output": "raw"},
             ]),
        Node(package="ah_mujoco", executable="mujoco_safety",
             name="mujoco_safety", output="screen",
             parameters=[
                 {"hand_side": LaunchConfiguration("hand_side")},
                 {"hand_size": LaunchConfiguration("hand_size")},
             ]),
        Node(package="ah_mujoco", executable="hand_pose",
             name="hand_pose", output="screen",
             parameters=[
                 {"hand_side": LaunchConfiguration("hand_side")},
                 {"depth_topic": LaunchConfiguration("depth_topic")},
             ]),
        Node(package="ah_mujoco", executable="arm_teleop",
             name="arm_teleop", output="screen",
             parameters=[
                 {"hand_side": LaunchConfiguration("hand_side")},
                 {"position_scale": LaunchConfiguration("position_scale")},
                 # MEASURE THESE for your rig:
                 {"base_cam_xyz": [0.0, 0.0, 0.0]},
                 {"base_cam_rpy": [0.0, 0.0, 0.0]},
                 {"hand_flange_xyz": [0.0, 0.0, 0.0]},
                 {"hand_flange_rpy": [0.0, 0.0, 0.0]},
             ]),
        Node(package="ah_mujoco", executable="xarm_bridge",
             name="xarm_bridge", output="screen",
             condition=IfCondition(LaunchConfiguration("use_arm_hardware")),
             parameters=[
                 {"robot_ip": LaunchConfiguration("robot_ip")},
                 {"service_prefix": LaunchConfiguration("service_prefix")},
                 {"hand_side": LaunchConfiguration("hand_side")},
                 {"safe_pose": ParameterValue(
                     LaunchConfiguration("safe_pose"),
                     value_type=None)},
             ]),
        Node(package="ah_mujoco", executable="clinical_ui",
             name="clinical_ui", output="screen",
             parameters=[
                 {"hand_side": LaunchConfiguration("hand_side")},
                 {"hand_size": LaunchConfiguration("hand_size")},
                 {"calibration_file": LaunchConfiguration("calibration_file")},
                 {"theme": LaunchConfiguration("theme")},
                 {"arm": True},
                 {"arm_state_topic": LaunchConfiguration("arm_state_topic")},
                 {"xarm_description_path":
                  LaunchConfiguration("xarm_description_path")},
             ]),
    ])
