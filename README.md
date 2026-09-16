# Ability Hand teleoperation

Webcam hand tracking driving a PSYONIC Ability Hand and an xArm Lite 6, with
MuJoCo physics in the command path as a safety filter.

A camera watches your hand. MediaPipe finds the landmarks, they are filtered
and retargeted onto the hand's six actuated joints, and the resulting command
is applied to a simulation of the hand before anything reaches the hardware —
what the robot receives is a pose the simulation actually reached. With the
hand mounted on a Lite 6, the tracked wrist pose drives the arm through inverse
kinematics as well.

Everything is visualised in one MuJoCo window. No Gazebo, no RViz.

```
camera ─┬─ hand landmarks ─ Kalman ─ retarget ─ One Euro ─┐
        │                                                 ├─ MuJoCo safety ─ Ability Hand
        └─ body landmarks ─ depth ─ shoulder / elbow ──┐   │
                                                       │   │
           wrist pose (SE3) ─ clutch ─ IK ─────────────┴───┴─ xArm Lite 6
```

## What is here

| package | what it does |
|---|---|
| `ah_mujoco` | everything added for this project: tracking, retargeting, MuJoCo physics and safety, the UI, IK, the xArm bridge |
| `ah_ros_py` | the Ability Hand serial driver (`ah_node`) |
| `ah_messages`, `ah_urdf`, `ah_driver`, `ah_tests` | messages, model, ONNX policy, validation |
| `xarm_ros2` | UFACTORY's driver and description (submodule) |
| `Azure_Kinect_ROS_Driver` | Kinect driver (submodule) |
| `HandPose` | MediaPipe model download (submodule) |

`src/ah_mujoco/README.md` documents the pipeline in detail, including the
measurements behind most of the design choices. This file is how to get it
running.

## Setup

```bash
git clone --recursive <this repo>
cd psyonic_ros

cd src/HandPose && ./get_model.sh                 # hand_landmarker.task
wget -O pose_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
cd ../..

cd docker
docker compose build            # slow: K4A SDK, MediaPipe, MuJoCo
docker compose up -d
docker compose exec ros2_ah bash
```

Inside the container:

```bash
cd /ws
colcon build --packages-skip xarm_controller xarm_planner xarm_moveit_servo \
                             xarm_gazebo d435i_xarm_setup realsense_gazebo_plugin
source install/setup.bash
```

Those skips are packages needing MoveIt, Gazebo or RealSense, none of which are
used — the visualisation is MuJoCo. Leaving them in aborts the whole build.

Check it worked:

```bash
ros2 pkg list | grep -E "ah_mujoco|xarm_api|azure_kinect"
python3 -c "import mediapipe, cv2, mujoco, glfw; print('ok')"
```

## Running

Three terminals, each `docker compose exec ros2_ah bash` then
`source /ws/install/setup.bash`.

**1 — camera**

```bash
ros2 launch azure_kinect_ros_driver driver.launch.py \
    color_resolution:=720P fps:=30 depth_mode:=NFOV_UNBINNED
```

**2 — arm driver** (skip for hand-only)

```bash
ros2 launch xarm_api lite6_driver.launch.py robot_ip:=192.168.1.154
```

**3 — everything else**

```bash
ros2 launch ah_mujoco arm_ui_real.launch.py \
    robot_ip:=192.168.1.154 \
    xarm_description_path:=/ws/src/xarm_ros2/xarm_description \
    model_path:=/ws/src/HandPose/hand_landmarker.task \
    calibration_file:=/ws/vendor/teleop_calibration.json \
    anchor_to_shoulder:=True
```

**The arm moves to its safe pose when terminal 3 starts.** Clearance, and keep
the estop in reach.

### Other entry points

| launch | what it runs |
|---|---|
| `arm_ui_real.launch.py` | the full thing: arm, hand, tracking, UI |
| `teleop_ui_real.launch.py` | hand only, real hardware |
| `teleop_ui_sim.launch.py` | hand only, no hardware |
| `arm_ui_real.launch.py use_arm_hardware:=False` | everything except commanding the arm |
| `validation_sim.launch.py` | the scripted validation sweep, no hardware |
| `policy_sim.launch.py` | the trained ONNX policy against a simulated hand |

Anything ending `_sim` needs no robot at all, which is the right place to start.

## Calibration

Four steps, and all four are required. Buttons in the UI, or the keys in
brackets.

| step | pose | button |
|---|---|---|
| 1 | flat hand, fingers straight, thumb spread | CAPTURE OPEN HAND `o` |
| 2 | fist, thumb across the palm | CAPTURE FIST `c` |
| 3 | move at your intended teleop speed | CAPTURE BASELINE `b` |
| 4 | face the camera, arm visible | MEASURE ARM `l` |

Steps 1 and 2 set the retargeting range — hold the two poses distinctly, or
every finger saturates. Step 3 sets what counts as smooth motion for you; that
becomes the green centre of the colour scale. Step 4 measures your body
proportions, which stops the upper-limb fit solving for scale on every frame.

Calibration is per person and per camera position. It is saved and reused; pass
`force_calibration:=True` to redo it.

**Keep the file outside the package.** `/ws/vendor/` is bind-mounted to the
repo, so it survives container rebuilds and `rm -rf src/ah_mujoco`.

## The UI

One window. Camera with the tracked skeleton on the left, the MuJoCo arm and
hand on the right, all twelve joints along the bottom.

- **space** or the CLUTCH button engages arm motion. Released, the arm holds —
  so you can reposition your hand the way you lift a mouse.
- Joint bars are coloured by Mean Squared Jerk against your calibrated
  baseline: green is motion as smooth as yours, red is not.
- **t** switches dark and light. **r** resets the camera. Drag to orbit,
  scroll to zoom.

## Safety

Four things sit between the tracker and the hardware, in this order:

1. joint limits from the URDF
2. a joint-space rate limit, so a bad tracking frame cannot step the hand
   across its range
3. MuJoCo physics — the command is applied to a simulation and what the
   simulation reached is what gets sent
4. the FSR contact guard: a digit already pressing on something stops closing
   further, though opening is always allowed

For the arm: a workspace box, a joint-space rate cap, and the clutch. Targets
are withheld entirely until a real command arrives, so nothing moves on
startup beyond the deliberate safe-pose move.

`position_scale` (0.3 by default on hardware) scales your motion down. Raise it
once the mapping feels right.

## Things that will catch you out

**The camera FOV is not the depth FOV.** At 720P the colour image is wider than
`NFOV_UNBINNED` depth, so your hand can be tracked perfectly while having no
depth at all. `depth_mode:=WFOV_2X2BINNED` is roughly 120° instead of 75° at the
same frame rate. This is the single most common cause of the upper limb and the
wrist pose going quiet.

**Body tracking needs much more of you than hand tracking does.** Torso and
both shoulders. Seated close to the camera with an arm extended, MediaPipe
frequently finds no body at all.

**Anything outside a bind mount is destroyed by `docker compose down`.** That
is why `vendor/`, `install/`, `build/` and `.ros/` are mounted. Downloads and
manually installed packages belong in `src/vendor/`.

**Two nodes cannot both publish `target/position`.** `mujoco_safety` owns it.
Running `virtual_hand` or `real_validation_node` alongside teleop gives the
hand an interleaved mix of two command streams.

**The xArm services are under `/ufactory`, not `/xarm`.** Check with
`ros2 service list | grep motion_enable` if a service call hangs.

## Session analysis

Record an exercise and get a report on how well the motion transmitted and
what it cost the operator:

```bash
ros2 run ah_mujoco session_recorder --ros-args -p out:=/ws/vendor/session1.npz
# Ctrl-C when done

ros2 run ah_mujoco session_report --ros-args \
    -p session:=/ws/vendor/session1.npz -p out:=/ws/vendor/session1.html
```

One self-contained HTML file of Plotly figures: transmission lag and gain,
usable bandwidth, smoothness preservation, and the operator's shoulder and
elbow against ergonomic comfort bands. `src/ah_mujoco/README.md` has the
detail, including why velocity correlation and transfer-function magnitude are
used instead of the obvious alternatives.

## Diagnostics

```bash
ros2 run ah_mujoco image_diag --ros-args -p topic:=/rgb/image_raw
```

Subscribes twice, best-effort and reliable, counts what each delivers, lists
publishers with their QoS, and gives a verdict. Separates "nothing is
publishing" from "QoS mismatch" from "transport problem", which look identical
from the outside.

Useful topics:

```bash
ros2 topic hz  /ability_hand/right/raw/position        # tracking alive
ros2 topic hz  /ability_hand/right/hand_pose           # wrist SE(3)
ros2 topic hz  /xarm/target_joints                     # IK solving
ros2 topic echo /ability_hand/right/upper_limb/swivel  # [swivel, valid, points, residual, scale]
ros2 topic echo /xarm/bridge_status                    # arm startup state
```

Most nodes say why they are idle rather than staying silent. If something is
not working, the terminal it runs in usually names the reason.

## Credit

The Ability Hand driver, URDF, messages and validation code are PSYONIC's.
`xarm_ros2` is UFACTORY's. The Azure Kinect driver is Microsoft's. The
biomechanical smoothness metric follows Morales et al., *Biomechanical
evaluation as basis for the development of predictive models in microsurgery
robots*.
