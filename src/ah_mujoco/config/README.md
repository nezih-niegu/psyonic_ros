Calibration files live here.

This directory is inside the workspace, which docker-compose bind-mounts as
`../src:/ws/src`. Files written here appear on the host and survive container
rebuilds. The default calibration path (`~/.ros/`) does not: it sits in the
container's writable layer and is lost on `docker compose down`, `build`, or
`up --force-recreate`.

Keep one file per person or per rig:

    ros2 launch ah_mujoco teleop_physics_sim.launch.py \
        calibration_file:=/ws/src/ah_mujoco/config/nezih.json
