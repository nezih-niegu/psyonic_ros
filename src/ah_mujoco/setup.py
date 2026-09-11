import os
from glob import glob

from setuptools import find_packages, setup

package_name = "ah_mujoco"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.launch.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.json")),
    ],
    install_requires=["setuptools", "mujoco", "mediapipe", "opencv-python", "glfw"],
    zip_safe=True,
    maintainer="psyonic_ros",
    maintainer_email="user@todo.todo",
    description="MuJoCo visualization for the PSYONIC Ability Hand (RViz replacement)",
    license="MIT",
    entry_points={
        "console_scripts": [
            "mujoco_viewer = ah_mujoco.mujoco_viewer_node:main",
            "virtual_hand = ah_mujoco.virtual_hand_node:main",
            "mediapipe_teleop = ah_mujoco.mediapipe_teleop_node:main",
            "mujoco_safety = ah_mujoco.mujoco_safety_node:main",
            "clinical_ui = ah_mujoco.clinical_ui_node:main",
            "hand_pose = ah_mujoco.hand_pose_node:main",
            "image_diag = ah_mujoco.image_diag_node:main",
            "arm_teleop = ah_mujoco.arm_teleop_node:main",
            "xarm_bridge = ah_mujoco.xarm_bridge_node:main",
            "upper_limb = ah_mujoco.upper_limb_node:main",
        ],
    },
)
