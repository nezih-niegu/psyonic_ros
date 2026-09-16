# ah_mujoco

MuJoCo visualization for the PSYONIC Ability Hand — a drop-in replacement for
the RViz pipeline (`joint_state_publisher` → `robot_state_publisher` → RViz).

The viewer subscribes to the same `/joint_states_ah` topic that `ah_node`
already publishes, loads the URDF from `ah_urdf` directly into MuJoCo (no MJCF
conversion step — `package://` mesh URIs are resolved in memory), and mirrors
the hand in a passive MuJoCo viewer window. FSR touch feedback tints each
fingertip red as pressure increases.

## Install

```bash
pip install mujoco   # also added to requirements/requirements.txt
colcon build --packages-select ah_mujoco
source install/setup.bash
```

## Run

Equivalent of the old `ah_node.launch.py` (driver + visualization):

```bash
ros2 launch ah_mujoco ah_mujoco.launch.py hand_side:=Right hand_size:=Large
```

Or attach the viewer to an already-running system:

```bash
ros2 run ah_mujoco mujoco_viewer --ros-args -p hand_side:=Right -p hand_size:=Large
```

Note: `ah_node` must run with `js_publisher:=True` (the launch file sets this).

## Parameters

| Parameter    | Default   | Description                                             |
|--------------|-----------|---------------------------------------------------------|
| `hand_side`  | `Right`   | `Right` / `Left` — picks the URDF variant               |
| `hand_size`  | `Large`   | `Small` / `Large`                                       |
| `model_path` | *(empty)* | Explicit URDF path; overrides side/size resolution      |
| `render_hz`  | `60.0`    | Viewer refresh rate                                     |
| `show_touch` | `True`    | Tint fingertips using `/ability_hand/<side>/feedback/touch` |
| `follow_targets` | `True` | Follow `target/position` when no joint states arrive (virtual hand) |

## Notes

- The node runs in kinematic mirror mode: incoming positions are written to
  `qpos` and `mj_forward()` updates the scene — same behavior RViz gave you,
  including the q2 mimic values `ah_node` computes.
- Headless machines need a working GL backend for the interactive viewer; the
  model loading itself has no display requirement.

## Virtual hand mode

With no hardware attached, the viewer follows `/ability_hand/<side>/target/position`
(`ah_messages/msg/Digits`, 6 values in degrees, order:
`index, middle, ring, pinky, thumb_flexor, thumb_rotator`) whenever
`/joint_states_ah` has been silent for 0.5 s. Real feedback always takes
priority when present. The degree-to-radian conversion, the q2 mimic relation
and the thumb index mapping match `ah_node.publish_joint_states` exactly.

Drive it by hand:

```bash
ros2 topic pub -r 10 /ability_hand/right/target/position \
  ah_messages/msg/Digits "{reply_mode: 1, data: [40,40,40,40,-30,30]}"
```

Set `follow_targets:=False` to restore strict feedback-only behavior.

## Running the validation pipeline without hardware

`ah_tests/real_validation_node` is a closed loop: its `_loop` returns early
until position, velocity AND touch feedback have all arrived. With no hand
attached those topics are silent, so it publishes zeros forever.

`virtual_hand` stands in for the hardware — it accepts targets, integrates a
speed-limited first-order plant, and publishes all the feedback topics plus
`/joint_states_ah`. Run it *instead of* `ah_node`:

```bash
ros2 launch ah_mujoco validation_sim.launch.py
```

That starts `virtual_hand` + `mujoco_viewer` + `real_validation_node`, and the
hand runs the pipeline's 4-second flex cycle (0 deg -> ~46 deg -> 0 deg).
Pass `validation:=False` to leave the validation node out.

Parameters: `hand_side`, `rate_hz` (60), `max_speed_dps` (250),
`publish_joint_states` (True).

Note: touch is published as 30 zeros, so `_compute_contact` always reports no
contact. Contact only becomes meaningful with a physics model and an object to
grasp.

## Running the trained ONNX policy

`ah_driver/ah_policy_node` subscribes to plain `/joint_states` for the q2 mimic
positions. That topic used to come from `joint_state_publisher` relaying
`joint_states_ah` (see `source_list` in urdf_launch's display.launch.py).
With RViz gone, `virtual_hand` publishes it directly.

```bash
ros2 launch ah_mujoco policy_sim.launch.py
```

The policy is a 21-input / 6-output network: 6 positions + 6 velocities +
4 mimic positions + 5 contact flags, all in SIM_JOINT_ORDER
(`index, middle, pinky, ring, thumb_flexor, thumb_rotator`).

**Gotcha:** the policy has a fixed point at exactly zero with zero contact —
it emits negative finger deltas that clip against the lower limit, so a hand
parked at zero never moves. `policy_sim.launch.py` seeds a 10 degree pose to
avoid this. From any non-zero pose it converges to a grasp-like posture.

## Mimic joints (important)

The URDF declares `q2 = 0.814 * q1` via `<mimic>` tags, but
`ah_node.publish_joint_states` computes `q2 = 1.0585*q1 + 0.7235`. These
disagree by 42 deg at rest, rising to 66 deg at full flex.

This never showed in RViz because `robot_state_publisher` honors `<mimic>` and
overrides whatever q2 the driver publishes. MuJoCo ignores URDF `<mimic>`
entirely (the compiled model has zero equality constraints), so incoming q2
values are used verbatim and the fingers render permanently hooked.

The viewer now parses the `<mimic>` tags itself and recomputes q2 from q1,
reproducing the RViz behavior. Disable with `enforce_mimic:=False` to see the
raw published values.

## Real hardware vs simulation — pick the right launch

| Launch | Serial port owner | Use when |
|---|---|---|
| `validation_sim.launch.py`  | `virtual_hand` (fake) | No hardware attached |
| `validation_real.launch.py` | `ah_node` (real)      | Hand plugged in |
| `policy_sim.launch.py`      | `virtual_hand` (fake) | Testing the ONNX policy |
| `ah_mujoco.launch.py`       | `ah_node` (real)      | Driver + viewer, no validation |

`validation_sim.launch.py` starts `virtual_hand`, NOT `ah_node`. Nothing in it
owns the serial port, so it can never command real hardware — and because
`virtual_hand` publishes the same feedback topics, the validation node follows
the fake hand even when a real one is plugged in. Never run both.

```bash
ros2 launch ah_mujoco validation_real.launch.py port:=/dev/ttyACM0
```

Note `ah_node.launch.py` declares a `hand_side` argument but never passes it to
the node (see its `parameters=` list), so `hand_side:=Left` is silently ignored
there and the driver stays on Right. The launch files in this package pass it
explicitly.

Also note `real_validation_node` hardcodes the `/ability_hand/right/...` topics,
so it only works with a right hand regardless of the launch argument.

## Webcam teleop (MediaPipe)

`mediapipe_teleop` tracks your hand with a webcam and publishes Ability Hand
targets. It uses the same MediaPipe HandLandmarker model as the HandPose repo.

### 21 landmarks are not 21 joints

MediaPipe returns 21 **landmarks** (3D points). The Ability Hand has **6
actuated DOF**: four finger flexors plus thumb flexion and thumb rotation. The
distal joint of each finger is not independently controllable — it follows
through the URDF mimic relation. So this is a retargeting (21 points -> 6 DOF),
not a joint-for-joint copy.

`hand_retarget.py` collapses each finger's three interior angles (MCP, PIP, DIP)
into one curl scalar, and derives thumb flexion and thumb opposition separately.
World landmarks are used rather than pixel coordinates, so the mapping is
invariant to where your hand sits in frame.

### Setup

```bash
pip install mediapipe opencv-python
cd /path/to/HandPose-main && ./get_model.sh    # downloads hand_landmarker.task
```

### Run

No hardware — tracking drives the virtual hand and MuJoCo:

```bash
ros2 launch ah_mujoco teleop_sim.launch.py \
    model_path:=/abs/path/HandPose-main/hand_landmarker.task
```

Real hand, mirrored in MuJoCo:

```bash
ros2 launch ah_mujoco teleop_real.launch.py port:=/dev/ttyACM0 \
    model_path:=/abs/path/HandPose-main/hand_landmarker.task
```

### Calibration is mandatory (and saved)

Curl values depend on hand size and camera distance, so the defaults will not
fit everyone (the thumb especially). With the preview window focused:

| key | action |
|---|---|
| `o` | capture current pose as OPEN (flat hand, thumb out) |
| `c` | capture current pose as CLOSED (fist, thumb across palm) |
| `s` | print the calibration as ROS params to paste into a launch |
| `r` | reset to defaults |
| `q` | quit |

Hold each pose steady, press the key, then paste the `s` output into your
launch file so you don't recalibrate every session.

### Parameters

`camera_index` (0), `model_path`, `publish_hz` (30), `preview` (True),
`mirror` (True), `smoothing` (0.6, exponential — raise it for calmer motion,
lower for more responsive), `det_conf` (0.5), and the six calibration values.

### Webcam in Docker

The compose file does not forward a camera. Add to `docker-compose.yaml`:

```yaml
    devices:
      - /dev/video0:/dev/video0
```

Check the camera is visible inside the container with
`ls -l /dev/video*` before launching.

## Physics in the loop (webcam -> MuJoCo -> real hand)

`mujoco_safety` puts the simulation *in the command path* rather than beside
it. Raw commands are treated as requests; the hand only ever receives a pose
the simulation actually reached.

```
mediapipe_teleop --raw/position--> mujoco_safety --target/position--> ah_node --> HARDWARE
                                        |
                                        +-- /joint_states_ah --> mujoco_viewer
```

```bash
# no hardware
ros2 launch ah_mujoco teleop_physics_sim.launch.py \
    model_path:=/ws/HandPose/hand_landmarker.task

# real hand
ros2 launch ah_mujoco teleop_physics_real.launch.py \
    port:=/dev/ttyACM0 model_path:=/ws/HandPose/hand_landmarker.task
```

### The physics model

The URDF alone is not simulatable: MuJoCo's importer gives it no actuators and
drops the `<mimic>` tags. `physics_model.py` compiles the URDF, dumps the MJCF,
and injects:

* an `<equality><joint>` per mimic tag, so q2 tracks q1 as the real linkage does
  (verified: ratio holds at 0.8140)
* a `<position>` actuator on each of the 6 driven joints, with `ctrlrange` from
  the URDF limits and `forcerange` from the URDF effort
* `<contact><exclude>` for every parent-child body pair

That last one is not optional. The URDF's collision hulls overlap at each joint,
and MuJoCo does not auto-exclude bodies connected by a joint, so without the
exclusions the fingers jam against the palm at ~10 degrees and never close.

Runs at ~107x real time (19 us/step at 500 Hz), so there is plenty of budget
for further constraints.

### Adding safety constraints

`_apply_safety()` in `mujoco_safety_node.py` is the single place commands are
modified. It currently does:

1. joint limits — clamp to URDF ranges
2. rate limit — `max_speed_dps`, so a bad tracking frame cannot step the hand
   across its range instantly
3. physics — the sim's own limits, mimic coupling and contacts
4. torque guard — joints exceeding `max_torque` stop advancing

Add yours there. Contact forces are available as `data.contact` /
`data.actuator_force`; actuator forces are also published on
`/ability_hand/<side>/safety/actuator_force` for logging.

### One publisher rule

`mujoco_safety` is the only node that may publish `target/position`. Running
`virtual_hand`, `real_validation_node` or teleop with `output:=target` at the
same time puts two publishers on one topic, and the hand receives an
interleaved mix of both. Teleop defaults to `output:=raw` for this reason.

## Signal quality and movement realism (MSJ)

Following Morales et al., *Biomechanical evaluation as basis for the development
of predictive models in microsurgery robots*, `signal_quality.py` provides:

* **Kalman filter** on the 21 raw landmarks before retargeting (the paper found
  it preferable to Savitzky-Golay: less error, movement characteristics kept)
* **MAE** between raw and filtered landmarks, published on
  `/ability_hand/<side>/quality/mae` — how noisy the acquisition is
* **Mean Squared Jerk** on the 6 retargeted joint commands, published on
  `/ability_hand/<side>/quality/msj`

```
J_t = (p_t - 3 p_{t-1} + 3 p_{t-2} - p_{t-3}) / dt^3
MSJ = (1/N) sum ||J_i||^2
```

Verified against a cubic, where jerk is analytically 6a: error 1.2e-14.

### Read this before trusting the 524 reference

Raw MSJ scales as amplitude^2 / dt^6, so its numeric value means nothing
without the units, sample rate and movement scale it was measured at. In
degrees at 30 Hz an *ideal* minimum-jerk finger curl already scores 4.1e6 — a
reference of 524 is unreachable on that scale.

`MSJMonitor(normalize=True)` (the default) therefore reports the standard
dimensionless jerk instead:

```
DJ = sqrt( (T^5 / (2 A^2)) * integral |J|^2 dt )
```

Measured invariance for ideal minimum-jerk motion: identical score across
amplitudes of 10, 90 and 500 degrees. On this scale:

| signal | score |
|---|---|
| ideal minimum-jerk | ~12 |
| + 0.1 deg tracker noise | ~79 |
| + 0.3 deg | ~259 |
| + 1.0 deg | ~700 |

So 524 corresponds to roughly 0.7 degrees of noise. **Calibrate the reference
on your own signal** rather than assuming 524 transfers — run teleop, watch the
logged MSJ during motion you consider good, and set `msj_reference` to that.
Set `msj_normalize:=False` for the raw formula.

### Colouring

The calibrated baseline is the green centre: whatever your own motion measured
during calibration is "good", and departing from it fades continuously through
yellow and orange to red.

| deviation | colour |
|---|---|
| 0 | green |
| 50 (1 sigma) | yellow |
| 100 (2 sigma) | orange |
| 250+ (5 sigma) | red |

**Thresholds scale with the calibrated spread.** The baseline step records the
standard deviation of your motion as well as its mean, and the stops are then
read as 1/2/5 sigma rather than absolute numbers. A fixed +/-50 means something
very different against a baseline of 12 than against 520; in sigma units the
same thresholds behave identically at either.

If no spread is available (hand-set `msj_reference`, or an older calibration
file) the absolute 50/100/250 stops are used instead.

**Direction.** `msj_direction` defaults to `both`: departing from the baseline
in either direction is flagged. Set it to `above` to flag only jerkier-than-
baseline motion. Be aware of what that hides — if tracking freezes, position
stops changing, jerk goes to ~0 and the score collapses far *below* baseline.
Under `both` that shows red; under `above` it shows green. Lost tracking is
exactly the failure you want the colour to catch, which is why `both` is the
default.

```bash
ros2 run ah_mujoco mujoco_viewer --ros-args \
    -p color_mode:=msj -p msj_reference:=524.0
```

Each digit is tinted by its own joint's score; the thumb takes the worse of its
two DOF. The teleop preview bars use the same colours. `color_mode` accepts
`touch` (FSR, the default), `msj`, or `none`.

### Parameters

`kalman` (True), `kalman_process_var` (1e-2), `kalman_measurement_var` (1e-4),
`msj_reference` (524.0), `msj_window` (90 samples), `msj_normalize` (True),
`quality_log_hz` (1.0).

Tuning note: MAE wants fidelity and MSJ wants smoothness, and they pull in
opposite directions — jerk is a third derivative, so it amplifies tracker noise
by 1/dt^3 (about 2e5 at 60 Hz). Raising `kalman_process_var` tracks fast motion
and raises MSJ; lowering it smooths harder, lowers MSJ, and adds lag.

## Calibration policy

Teleop **always calibrates on first run**, then reuses the saved calibration.
It will not publish hand commands until it has either loaded a saved file or
completed a fresh session — defaults are not a safe fallback, because the
retargeting ranges depend on hand size and camera distance and the MSJ
reference depends on your rate and filter tuning.

Startup logic:

1. `force_calibration:=True` → always run a new session
2. otherwise, load `calibration_file` (default `~/.ros/ah_teleop_calibration.json`)
3. no file → run a guided session, save it, then start commanding

A corrupt, incomplete, wrong-schema or degenerate file never falls back to
defaults silently - defaults would command a real hand with the wrong mapping.
It logs why the file cannot be used and starts a fresh calibration, which is
the obvious next step. An earlier version exited instead, which turned an old
calibration file into a dead end.

### The guided session

Three steps, prompted in the preview window. Commands stay paused throughout
and a "CALIBRATING" banner is shown.

| step | pose | key |
|---|---|---|
| open | flat hand, fingers straight, thumb spread | `o` |
| closed | fist, thumb across the palm | `c` |
| baseline | move naturally at your intended teleop speed | `b` |

The baseline step accumulates every frame while it is active, so the reference
is the mean of several seconds of real motion rather than one instant, and its
spread is measured at the same time. It requires at least 30 samples before
`b` is accepted.

The baseline step is what sets `msj_reference`: whatever your own smooth motion
scores becomes the green centre, so later motion is judged against you rather
than against a number from someone else's setup.

After calibration: `k` restarts a session, `s` re-saves the current values,
`q` quits.

### Reusing a calibration

```bash
# first run - calibrates and saves
ros2 launch ah_mujoco teleop_physics_sim.launch.py \
    model_path:=/ws/HandPose/hand_landmarker.task

# later runs - loads the saved file, no calibration
ros2 launch ah_mujoco teleop_physics_sim.launch.py \
    model_path:=/ws/HandPose/hand_landmarker.task

# a specific file (per user, per rig)
ros2 run ah_mujoco mediapipe_teleop --ros-args \
    -p calibration_file:=/ws/src/ah_mujoco/config/nezih.json
```

Headless (`preview:=False`) cannot calibrate, since the session is driven by
keypresses. If no saved calibration exists the node exits immediately with an
explanation instead of running with no output.

## Kalman filter performance

The filter was originally a Python loop over the 63 coordinates. It is now
batched: all coordinates advance together with `(dim, 3, 3)` array operations.
Because the observation model is `H = [1, 0, 0]`, the matrix products collapse
to column indexing (`H P Hᵀ` is `P[:, 0, 0]`, `P Hᵀ` is `P[:, :, 0]`), so no
3x3 inverse is ever formed.

| version | us/frame | % of 30 Hz budget |
|---|---|---|
| per-coordinate loop | 971.5 | 2.91 % |
| batched | 26.1 | 0.078 % |

37x faster, and numerically identical to the loop (max difference 3.3e-16).

### On GPU acceleration

Not worth it at this size, and it would be slower. A CUDA kernel launch costs
roughly 5-10 us and a host-device round trip a few us more; the entire batched
CPU computation is 26 us, so the launch overhead alone is a large fraction of
the work. Scaling on CPU:

| coordinates | us/frame | max rate |
|---|---|---|
| 63 (this pipeline) | 27 | 37 kHz |
| 630 (10 hands) | 142 | 7 kHz |
| 6,300 (100 hands) | 1,196 | 836 Hz |
| 63,000 (1000 hands) | 13,368 | 75 Hz |

GPU starts to make sense somewhere around the 6,300+ row — hundreds of hands
tracked at once, or a much larger state per coordinate. At 30 Hz with one hand,
the CPU version uses 0.078 % of the frame budget and MediaPipe inference
dominates the loop by orders of magnitude.

`KalmanPosition` takes an `xp` argument if you do want to try it:

```python
import cupy
kf = KalmanPosition(63, dt, xp=cupy)
```

Note on tooling: cuSignal was deprecated after RAPIDS v23.08 and archived
during the v23.10 cycle; its functionality moved into CuPy v13 as
`cupyx.scipy.signal`. It is a scipy.signal port (filtering, resampling,
spectral analysis) and has no Kalman implementation to call. CUSP is a sparse
linear-algebra library, also not applicable here. The relevant primitive for a
GPU Kalman would be cuBLAS batched routines, which is what CuPy's batched
matmul already dispatches to.

## Calibration and Docker

The default calibration path is `~/.ros/ah_teleop_calibration.json`, which
inside the container is `/root/.ros/...` — part of the container's writable
layer, not a mount. It survives `docker compose stop` / `start` / `restart`,
but is **lost** on `docker compose down`, `build`, or `up --force-recreate`.
That is the same reason the `install/` space disappears after a rebuild.

The compose file mounts only `../src:/ws/src`, so put the calibration there:

```bash
mkdir -p /ws/src/ah_mujoco/config

ros2 launch ah_mujoco teleop_physics_sim.launch.py \
    model_path:=/ws/HandPose/hand_landmarker.task \
    calibration_file:=/ws/src/ah_mujoco/config/nezih.json
```

The file then appears on the host under
`psyonic_ros-master/src/ah_mujoco/config/` and survives rebuilds. All four
teleop launch files take `calibration_file` and default it to
`/ws/src/ah_mujoco/config/teleop_calibration.json`; the same value is passed
to the viewer so it picks up the MSJ reference and spread.

To persist the default `~/.ros` location instead, add a volume:

```yaml
      - ../.ros:/root/.ros
```

The model bundle has the same issue if you keep it outside the mount. Anything
you want to survive a rebuild belongs under `src/`, or needs its own volume.

## Latency

Measured 90 % step response through the chain. The defaults were tuned for
safety and smoothness, which cost roughly a second of lag - unusable for
dexterous work. Current defaults:

| stage | before | after | what changed |
|---|---|---|---|
| camera queue | 100 ms | 17 ms | `CAP_PROP_BUFFERSIZE=1`, MJPG, grab thread |
| inference + publish period | 33 ms | 17 ms | `publish_hz` 30 -> 60 |
| retargeter EMA | 167 ms | 32 ms | `smoothing` 0.6 -> 0.15 |
| safety rate limit (90 deg) | 600 ms | 200 ms | `max_speed_dps` 150 -> 450 |
| MuJoCo actuator | 146 ms | 9 ms | `kv_ratio` 0.06 -> 0.005 |
| safety publish period | 17 ms | 8 ms | `publish_hz` 60 -> 120 |
| **total** | **1063 ms** | **282 ms** | |

For a small 20 degree fine adjustment, which is what dexterous manipulation
actually consists of: **596 ms -> 127 ms**.

### The two that mattered most

**`kv_ratio`.** The actuator had `kv = kp * 0.06` hardcoded. In a MuJoCo
position actuator, kv/kp *is* the closed-loop time constant, so every joint
carried a 60 ms lag no matter what kp was set to - sweeping kp from 100 to 3000
changed the rise time by 0 ms. At 0.005 the same step takes 9 ms with no
overshoot.

**Camera buffering.** `cap.read()` returns the *oldest* queued frame, and V4L2
queues several by default, so each one is already 1-4 frame periods old. A
background thread now drains the queue continuously and inference always runs
on the newest frame. Frames are also skipped rather than reprocessed when
inference outruns the camera.

Also worth knowing: `armature` was 0.005 against link inertias of ~1e-6, so the
simulated fingers were about 5000x heavier than the real ones. Now 1e-5.

### Tuning further

```bash
ros2 launch ah_mujoco teleop_physics_real.launch.py \
    port:=/dev/ttyACM0 \
    max_speed_dps:=600 \
    camera_fps:=120 publish_hz:=90 \
    smoothing:=0.05
```

- `smoothing` 0 removes the EMA entirely. Jitter then passes straight through,
  which the MSJ colouring will show as red - a useful way to find the point
  where you have traded away too much filtering.
- `max_speed_dps` above ~460 exceeds the URDF's own velocity limit
  (8.07 rad/s), so the guard stops being physically meaningful.
- A 60 or 120 fps camera helps more than any other single change; at 30 fps
  the frame period alone is 33 ms and no amount of tuning recovers it.
- `kalman_process_var` up (try 1.0) tracks fast motion with less lag, at the
  cost of passing more noise through.

Recalibrate after changing `publish_hz`: the MSJ reference depends on the
sample rate.

## Unified UI

`clinical_ui` replaces the two separate windows (the OpenCV preview and the
MuJoCo passive viewer) with one OpenGL surface:

```
+--------------------------------------------------------------+
|  ABILITY HAND / TELEOP          status            fps · lat   |
+----------------------------+---------------------------------+
|     camera + landmarks     |   MuJoCo hand (commanded state)  |
+----------------------------+---------------------------------+
|  joint bars, MSJ coloured  |  signal quality + deviation gauge|
+--------------------------------------------------------------+
```

Minimalist by design: two surface tones, hairline rules, two text weights, one
accent. All status meaning is carried by the MSJ gradient, so the chrome stays
neutral. Dark and light themes ship; `t` toggles at runtime, or set
`theme:=light` at launch.

Everything is drawn with MuJoCo's own renderer primitives — `mjr_render` for
the 3D hand, `mjr_drawPixels` for the camera feed, `mjr_rectangle` and
`mjr_text` for panels — so the only new dependency is `glfw`, which MuJoCo
already pulls in. The layout is computed from the framebuffer size, so the
window resizes freely.

```bash
pip install glfw          # usually already present with mujoco

ros2 launch ah_mujoco teleop_ui_sim.launch.py \
    model_path:=/ws/HandPose/hand_landmarker.task

ros2 launch ah_mujoco teleop_ui_real.launch.py \
    port:=/dev/ttyACM0 model_path:=/ws/HandPose/hand_landmarker.task
```

Mouse: drag to orbit, right-drag to pan, scroll to zoom. `r` resets the view,
`t` toggles dark/light, `q` quits.

The 3D background colour comes from a flat skybox texture: without one MuJoCo
clears the viewport to black, which leaves a hole in the light theme. The URDF
`<mujoco>` extension ignores `<asset>` blocks, so `load_ability_hand_model(...,
skybox=True)` dumps the compiled model to MJCF, injects the texture and
recompiles. Switching theme rewrites the texture bytes and re-uploads them.

The joint bars are tinted by each joint's MSJ, and the gauge on the right
shows deviation from the calibrated baseline with a marker for the current
value — green centre, red edges, same scale as the 3D hand colouring.

### Calibration and the UI

Calibration works from the UI window directly: `o`, `c` and `b` are relayed to
teleop over `/ability_hand/<side>/calibration/command`, since teleop owns the
session and the landmarks. `k` restarts calibration, `s` re-saves. The prompt
for the current step appears as a banner across the top.

So a first run needs no special handling — launch the UI and calibrate in it:

```bash
ros2 launch ah_mujoco teleop_ui_sim.launch.py \
    model_path:=/ws/HandPose/hand_landmarker.task
```

Teleop still refuses to publish hand commands until calibration completes, and
now only aborts at startup when there is neither a preview window nor a UI
attached (`preview:=False` and `publish_image:=False`), since in that case
nothing can capture the poses.

The cv2 preview remains available if you prefer it:

```bash
# first: calibrate (cv2 window)
ros2 run ah_mujoco mediapipe_teleop --ros-args \
    -p model_path:=/ws/HandPose/hand_landmarker.task \
    -p calibration_file:=/ws/src/ah_mujoco/config/nezih.json

# then: UI, reusing it
ros2 launch ah_mujoco teleop_ui_sim.launch.py \
    model_path:=/ws/HandPose/hand_landmarker.task \
    calibration_file:=/ws/src/ah_mujoco/config/nezih.json
```

Running with `preview:=True` shows both windows, which is only useful while
calibrating. A calibration in progress is shown as an amber banner across the
UI, and teleop publishes the prompt on
`/ability_hand/<side>/calibration/status`.

### Notes

The camera feed travels as `sensor_msgs/Image` on
`/ability_hand/<side>/camera/image_raw`, downscaled to `image_width` (480 px by
default) to keep the topic cheap. Two processes cannot share a webcam, so the
UI never opens the camera itself.

`mjr_drawPixels` does no scaling: it consumes exactly
`viewport.width * viewport.height` pixels and uses the viewport width as the
row stride. Frames are resized to the viewport before drawing; a mismatch
shears the image into diagonal noise.

## Real hand with the UI

```bash
ros2 launch ah_mujoco teleop_ui_real.launch.py \
    port:=/dev/ttyACM0 \
    model_path:=/ws/HandPose/hand_landmarker.task \
    calibration_file:=/ws/src/ah_mujoco/config/teleop_calibration.json
```

```
mediapipe_teleop --raw--> mujoco_safety --target--> ah_node --> HARDWARE
        |                       |
        +- camera/image_raw     +- /joint_states_ah -> clinical_ui
```

Calibrate in the UI window on first run: `o` flat hand, `c` fist, `b` move
naturally. Until that completes teleop publishes nothing, so the hand stays
where it is.

### Startup safety

`mujoco_safety` withholds `target/position` until a raw command actually
arrives. Without that it would publish its own zero state at 120 Hz from
startup and drive the hand to the zero pose before teleop had said anything.
On the first command the plant is seeded at that pose rather than sweeping to
it from zero at the rate limit.

`ah_node` runs with `js_publisher:=False` here: the safety node publishes
`/joint_states_ah` from the simulated state, and two publishers on one topic
would make the UI flicker between real and commanded poses.

### Limits

The real launch defaults to `max_speed_dps:=300` and `max_torque:=3.0`, below
the tuned sim values (450 / 4.0), because a tracking glitch on hardware is
worth more caution than one in simulation. Raise them once the motion feels
right:

```bash
ros2 launch ah_mujoco teleop_ui_real.launch.py port:=/dev/ttyACM0 \
    max_speed_dps:=450 max_torque:=4.0 theme:=light
```

The URDF velocity limit is 8.07 rad/s (~462 deg/s); past that the rate guard
stops being physically meaningful.

### Sanity checks

```bash
ros2 topic hz /ability_hand/right/raw/position      # teleop is publishing
ros2 topic hz /ability_hand/right/target/position   # safety is passing through
ros2 topic echo /ability_hand/right/safety/actuator_force --once
```

No traffic on `raw/position` means calibration has not finished. Traffic there
but none on `target/position` means the safety node has not seen a raw command
yet.

## Camera from a ROS topic

Set `image_topic` and teleop subscribes to a `sensor_msgs/Image` stream instead
of opening `/dev/video*` itself. Useful when a camera driver already owns the
device, when replaying a bag, or when the camera is on another machine.

```bash
ros2 launch ah_mujoco teleop_ui_real.launch.py \
    port:=/dev/ttyACM0 \
    model_path:=/ws/HandPose/hand_landmarker.task \
    image_topic:=/camera/color/image_raw
```

Leave `image_topic` empty (the default) to keep using the local device.

Details worth knowing:

- **QoS is sensor-data (best effort).** Camera drivers almost always publish
  best-effort; a reliable subscription would silently receive nothing from
  them, which looks exactly like a dead camera.
- **Encodings:** `bgr8`, `rgb8`, `mono8`, `bgra8`, `rgba8`. Anything else
  (`yuyv`, `bayer_*`) is rejected with a warning rather than reinterpreted as
  garbage — put a decoding node upstream.
- **Newest frame wins.** Frames are not queued; if inference is slower than the
  publisher, older frames are dropped rather than processed late.
- **Feedback guard.** If `image_topic` is set to the same topic teleop
  publishes for the UI, `publish_image` is disabled automatically and a warning
  is logged, since it would otherwise feed back on itself.
- **Real latency measurement.** With a topic source the frames carry a header
  stamp, so the age of each frame is measured and published on
  `/ability_hand/<side>/quality/latency_ms`. The UI header shows this instead
  of a placeholder — it is the true source-to-inference latency, including
  driver and transport.

A bag replay works the same way:

```bash
ros2 bag play my_session.bag
ros2 launch ah_mujoco teleop_ui_sim.launch.py \
    model_path:=/ws/HandPose/hand_landmarker.task \
    image_topic:=/camera/color/image_raw
```

## Troubleshooting a topic source

`no frames on <topic>` means the subscription is up but nothing is arriving.
In order of likelihood:

1. **Nothing is publishing.** `ros2 topic hz /rgb/image_raw` in the same shell
   as teleop. No output means the camera driver is not running.
2. **ROS domain mismatch.** The compose file sets `ROS_DOMAIN_ID=100`. A
   publisher started outside that domain (default is 0) is invisible even on a
   shared host network, and `ros2 topic list` will simply not show it. Start
   the driver with the same `ROS_DOMAIN_ID`, or change the container's.
3. **Wrong topic name.** `ros2 topic list | grep -i image`.
4. **Encoding.** `ros2 topic echo /rgb/image_raw --field encoding --once`.
   Only `bgr8`, `rgb8`, `mono8`, `bgra8` and `rgba8` are decoded; anything else
   warns rather than rendering garbage.

### Azure Kinect as the source

The k4a driver's `/rgb/image_raw` runs at whatever resolution the driver was
started with, and at 1536p that is a few Hz — far too slow for teleop, and it
dominates the latency budget on its own. Start the driver at a low colour
resolution and a high frame rate:

```bash
ros2 launch azure_kinect_ros_driver driver.launch.py \
    color_resolution:=720P fps:=30 depth_enabled:=false
```

Check what you are actually getting before blaming the pipeline:

```bash
ros2 topic hz /rgb/image_raw            # want 30, not 4
ros2 topic info /rgb/image_raw --verbose | grep -A2 Reliability
```

If the publisher is RELIABLE and teleop subscribes best-effort only, no frames
arrive at all — which looks exactly like a dead camera. `image_qos` defaults to
`auto`, which subscribes both ways and keeps whichever delivers; force it with
`image_qos:=reliable` or `image_qos:=sensor` if you prefer.

Teleop logs `first frame received: WxH encoding` on the first successful frame,
so a working link is unambiguous.

## Smoothing

The EMA was replaced with a One Euro filter (Casiez, Roussel & Vogel, CHI 2012)
on the six retargeted joint targets. A fixed low-pass forces one trade: enough
smoothing to kill jitter at rest always costs lag during motion. One Euro
varies its cutoff with the observed speed,

    cutoff = min_cutoff + beta * |dx/dt|

so it smooths hard when the hand is still and opens up when it moves. Jitter
and lag become nearly independent knobs.

Measured through the full pipeline at 30 Hz (Kalman on landmarks, then
retargeting, then smoothing):

| filter | jitter | added lag | MSJ |
|---|---|---|---|
| none | 8.72° | 0 ms | 472571 |
| EMA α=0.15 (old default) | 7.20° | 0 ms | 370431 |
| EMA α=0.5 | 4.07° | 34 ms | 184310 |
| EMA α=0.8 | 1.73° | 134 ms | 66211 |
| **One Euro (new default)** | **1.42°** | **0 ms** | **58907** |

**5.1x less jitter and 6.3x lower MSJ than before, with no added lag** — and
still smoother than EMA α=0.8, which cost 134 ms.

### Tuning

| parameter | default | effect |
|---|---|---|
| `min_cutoff` | 0.1 | smoothing when still. Lower = smoother at rest, barely affects fast motion |
| `beta` | 0.005 | how fast the cutoff opens with speed. Lower = smoother but laggier during motion |
| `kalman_process_var` | 0.1 | landmark stage. Raised from 0.01: with One Euro downstream the Kalman no longer needs to smooth, and a higher value removes its lag |

Still too noisy? Drop `beta` first — 0.002 gives 1.00° at 167 ms added lag.
Too sluggish? Raise it — 0.01 gives 1.84° at 34 ms.

```bash
ros2 launch ah_mujoco teleop_ui_real.launch.py port:=/dev/ttyUSB0 \
    model_path:=/ws/HandPose/hand_landmarker.task \
    image_topic:=/rgb/image_raw \
    beta:=0.002 min_cutoff:=0.05
```

**`beta` is unit-dependent.** It multiplies speed in deg/s, so a value tuned at
one noise level or frame rate does not transfer. Retune if you change the
source resolution or rate.

The real inter-frame interval is passed to the filter each tick rather than the
nominal period, which matters with a camera topic where the rate varies. Set
`one_euro:=False` to fall back to the EMA via `smoothing`.

Recalibrate after changing any of this: the MSJ reference is measured through
the filter, so a smoother pipeline shifts it.

## SE(3) hand pose from depth

`hand_pose` estimates the hand's full rigid pose in the camera frame and
publishes it as `geometry_msgs/PoseStamped` plus TF.

```
/ability_hand/<side>/landmarks  (from mediapipe_teleop)
/depth_to_rgb/image_raw         (registered depth)
/rgb/camera_info                (intrinsics)
        -> /ability_hand/<side>/hand_pose   +   TF camera -> <side>_hand
```

```bash
# teleop, publishing landmarks and NOT mirroring
ros2 launch ah_mujoco teleop_ui_real.launch.py \
    port:=/dev/ttyUSB0 model_path:=/ws/HandPose/hand_landmarker.task \
    image_topic:=/rgb/image_raw hand_pose:=True mirror:=False

# pose estimator
ros2 launch ah_mujoco hand_pose.launch.py
```

### How it works

MediaPipe's world landmarks are metric but hand-centred, so translation is
lost; the image landmarks have no scale. Depth supplies the missing piece: each
image landmark is deprojected to a 3D camera point, then a rigid transform is
fitted between the hand-frame landmarks and those points (Kabsch/Umeyama, no
scale — the world landmarks are already metric, and fitting scale would hide
depth bias in the geometry).

### All 21 landmarks, not just the palm

The obvious choice is to fit only the palm (wrist and the four MCP joints),
since fingers articulate. That is wrong here: the world landmarks already
encode the current articulation, so at any instant every landmark is a valid
rigid correspondence, and the wider spread conditions rotation far better than
the nearly-coplanar palm. Measured against known poses at 0.6 m:

| landmarks | 2 mm depth noise | 5 mm | 10 mm |
|---|---|---|---|
| palm (5) | 5.1° / 6.2 mm | 7.8° / 8.0 mm | 15.9° / 16.8 mm |
| **all 21** | **1.2° / 1.3 mm** | **2.3° / 3.4 mm** | **3.4° / 4.8 mm** |

Rotation is always the weaker axis — it is far more sensitive to depth noise
than translation. `palm_only:=True` is available for comparison.

### Two things that will bite

**Use registered depth.** `/depth_to_rgb/image_raw`, not `/depth/image_raw`.
Landmark pixels come from the colour image; the Kinect's depth camera has
different intrinsics and sits a few centimetres away, so unregistered lookups
sample the wrong points and produce a confident, wrong pose.

**Turn mirroring off.** Teleop mirrors the preview by default, which flips
pixel coordinates while the depth image stays unmirrored. `mirror:=False`
whenever landmarks are published; teleop warns if you forget.

### Robustness

Kinect depth is holey — reflective skin, edges and shadowing all drop pixels.
Each lookup takes the median of a small patch (`patch`, default 2 → 5x5), and
landmarks without valid depth are dropped. Below three valid points, or with
RMS residual above `max_residual` (5 cm), the estimate is refused rather than
published. Residual and point count go to
`/ability_hand/<side>/hand_pose/diagnostics` — a rising residual usually means
depth dropouts or a partly occluded hand.

Pose is smoothed by `alpha_t` and `alpha_r` (0.5). Quaternions are sign
ambiguous, so each sample is aligned to the previous before blending; without
that the filter lurches whenever the sign flips.

### What consumes it

Nothing yet. The Ability Hand has no wrist or arm DOF, so this drives no
existing output — it is for a wrist/arm stage, for recording trajectories in a
world frame, or for logging alongside the joint data. View it with:

```bash
ros2 topic echo /ability_hand/right/hand_pose
ros2 run tf2_ros tf2_echo rgb_camera_link right_hand
```

## Diagnosing "no frames" from a topic

```bash
ros2 run ah_mujoco image_diag --ros-args -p topic:=/rgb/image_raw
```

Subscribes twice at once, best-effort and reliable, counts what each delivers,
lists the publishers with their advertised QoS, and prints a verdict. That
separates three failures which look identical from the outside:

| symptom | meaning |
|---|---|
| no publishers listed | driver not running, or different `ROS_DOMAIN_ID` |
| one profile receives, the other does not | QoS mismatch → set `image_qos` |
| publishers listed, neither receives | transport or frame size, not QoS |

It also reports resolution, encoding and MB/frame, and warns if the encoding is
not one teleop decodes.

### Executor starvation

MediaPipe inference runs in teleop's timer callback and takes tens of
milliseconds. On rclpy's default single-threaded executor a long callback
blocks subscription callbacks, so frames can go undelivered even when QoS and
transport are fine — indistinguishable from a dead camera. The image
subscription now sits in its own callback group and teleop spins a
`MultiThreadedExecutor`.

### Frame size

At 3072p BGRA a Kinect frame is about 25 MB. CycloneDDS over UDP needs a large
receive buffer for that, and the default is often too small — frames are
fragmented and silently never reassembled. `ros2 topic hz` may still show a
rate while a second subscriber gets nothing. Dropping to `720P` (3.7 MB) avoids
the problem entirely and is plenty for hand landmarks.

## Contact guard (FSR feedback into the safety layer)

The safety layer's torque guard only knows about the *simulated* hand, which
contains nothing but the hand itself — it can catch a jam or self-collision,
but not the hand pressing on a real object. The contact guard closes that gap
using the hand's FSR feedback:

```
/ability_hand/<side>/feedback/touch  ->  mujoco_safety  ->  target/position
```

Once a digit's peak FSR reading reaches `contact_engage` (25), that digit stops
closing further. Anything that would increase flexion is held at the current
position; **opening is always allowed**, or a hand that grasped an object could
never let go.

Each digit is gated independently, so a grasp settles around the object's shape
rather than stopping the whole hand at first touch. Simulated grasp of a
cylinder where digits meet the surface at different depths:

```
requested   [ 95   95   95   95   85  -75 ]
commanded   [ 25.2 35.4 45.0 55.2 30.0 -30.0 ]
```

Both thumb DOF are gated by the thumb's sensors, and the rotator is held in its
own (negative) closing direction.

### Hysteresis

`contact_engage` (25) latches the guard on; it releases only below
`contact_release` (12). A single threshold chatters as the reading crosses it —
measured on a signal hovering around 25, one threshold gave 11 state changes
where the hysteresis pair gave 3.

### Parameters

| parameter | default | meaning |
|---|---|---|
| `use_contact_guard` | True | disable to fall back to the torque guard alone |
| `contact_engage` | 25.0 | FSR level that stops further closing |
| `contact_release` | 12.0 | level below which closing resumes |
| `contact_backoff_deg` | 0.0 | degrees to retreat on contact; 0 just holds |

Tune `contact_engage` against your own readings — FSR output is not calibrated
in newtons:

```bash
ros2 topic echo /ability_hand/right/feedback/touch
ros2 topic echo /ability_hand/right/safety/contact
```

`safety/contact` publishes 11 values: five contact flags then the five
per-digit peak readings.

**This only works with real hardware.** `virtual_hand` publishes touch as
zeros, so in simulation the guard never engages. Contact in simulation would
need objects in the MuJoCo scene, which is a separate piece of work.

## Arm mode: xArm Lite 6 with the hand as gripper

In solo-hand mode the tracked SE(3) pose is discarded. With the hand mounted on
a Lite 6, that pose becomes the arm's target and the fingers carry on as
before. The hand has no wrist DOF, so **every rotation the operator makes has
to be performed by the arm's last three joints.**

```
hand_pose (camera frame)
   -> T_base_cam @ T_cam_hand @ T_hand_flange
   -> clutch + scaling + workspace clamp
   -> IK  ->  /xarm/target_joints
```

```bash
ros2 launch ah_mujoco arm_teleop.launch.py
```

### Kinematics come from your repo, not from a guess

`LITE6_ORIGINS` is copied verbatim from
`xarm_description/config/kinematics/default/lite6_default_kinematics.yaml`, and
the limits from `lite6_robot_macro.xacro`. Every xArm joint rotates about its
own local Z, so those fixed origins plus one rotation per joint describe the
chain exactly — no DH table, no reconstructed link lengths.

Verified against that chain:

| check | result |
|---|---|
| analytic vs numeric Jacobian | max difference 1.4e-08 |
| IK round-trip, warm seed | 286/300 converged, mean error 69 µm / 0.0005° |
| IK from home seed | 161/200 converged |
| tracking a 10 cm arc | worst error 0.099 mm |

IK is damped least squares seeded from the previous solution — well
conditioned for teleoperation, since consecutive targets are close together.
Damping keeps steps bounded near singularities.

### Two transforms you must measure

| parameter | what it is |
|---|---|
| `base_cam_xyz` / `base_cam_rpy` | camera pose in the arm base frame — a hand-eye calibration |
| `hand_flange_xyz` / `hand_flange_rpy` | how the hand is bolted to the flange — from the drawing |

Both default to identity, which is certainly wrong for your rig; the node warns
at startup. An error in `base_cam` maps directly into every command, so this is
the first thing to get right and the first thing to suspect.

### Clutch

Absolute mapping is the wrong model: the operator's hand and the arm's
workspace do not coincide, and a tracking glitch would command a large motion.
Motion accumulates only while the clutch is engaged, and the arm holds when it
is released — so the operator can re-centre the way a mouse is lifted and
repositioned. Press **space** in the UI window, or publish
`std_msgs/Bool` on `/ability_hand/<side>/clutch`.

Verified: engaging never jumps the arm, releasing freezes it, and re-engaging
from a different hand position resumes without a jump.

### Mapping frame

`mapping_frame:=base` (default) moves the arm the way the hand moved, in the
base frame:

```
hand moves [0.08, 0.00, 0.05]  ->  arm moves [0.08, 0.00, 0.05]
```

`mapping_frame:=tool` applies the delta in the gripper's frame, so the same
hand motion becomes `[0.031, 0.000, 0.089]` — translation follows wherever the
tool points. Useful for aligned insertion, confusing in free space.

### Safety

- `position_scale` below 1.0 damps operator motion; start at 0.3 on hardware
- `box_min` / `box_max` clamp the target in the base frame regardless of what
  the tracker says
- `max_joint_step` (0.05 rad per solve) caps joint-space rate, so an
  unreachable target cannot produce a jump when IK finds a distant branch
- joint limits from the xacro are enforced on every solve
- `/xarm/ik_diagnostics` carries position error, rotation error and a
  converged flag

### Sending it to the robot

The node publishes targets; it does not command the arm. Bridge
`/xarm/target_joints` to the controller you prefer — `xarm_planner`, MoveIt
Servo, or the SDK's own `set_servo_angle_j`. Consider using the SDK's
`get_inverse_kinematics` instead of this solver if you want the controller's
exact branch selection; the solver here is for a target the arm can follow,
not a certified motion plan.

**Test in simulation first**, with `xarm_moveit_config`'s fake controller or
Gazebo. Position error in the hand-eye transform shows up as the arm confidently
moving to the wrong place.

## Standalone MuJoCo for the arm too

No Gazebo, no RViz. `arm_model.build_combined_model()` produces one MuJoCo
model containing the Lite 6 and the Ability Hand mounted on its flange, and the
UI renders it.

```bash
ros2 launch ah_mujoco arm_ui_real.launch.py \
    xarm_description_path:=/ws/src/xarm_ros2/xarm_description
```

Leave `xarm_description_path` empty and it searches the ament index and the
workspace.

### How the model is built

The arm comes from xarm_description's own data — joint origins from
`config/kinematics/default/lite6_default_kinematics.yaml`, limits from
`lite6_robot_macro.xacro`, and the `meshes/lite6/*.stl` link meshes (MuJoCo
cannot read the .dae variants). The hand is compiled from ah_urdf exactly as
before, keeping its mimic equalities, actuators and contact exclusions, then
spliced onto the arm's last link at the mount transform.

One source of truth per robot: arm geometry stays with xarm_description, hand
geometry with ah_urdf, and neither is transcribed by hand.

Result: 16 DOF, 12 actuators, 4 mimic constraints.

### The convention that will bite you

MuJoCo's default `eulerseq` is intrinsic `xyz`; URDF `rpy` is extrinsic. Using
the default put the flange **0.56 m** from where the kinematics said it should
be — a model that looks plausible and is wrong. The generated MJCF sets
`eulerseq="XYZ"`.

Verified afterwards against the analytic chain used for IK, over 200 random
configurations:

| | error |
|---|---|
| flange position | < 0.001 µm |
| flange orientation | 4.2 microdegrees |

The simulation and the IK solver are the same robot. That check is worth
repeating any time the model changes — if they diverge, the arm goes somewhere
the solver never intended.

### Mounting the hand

`mount_xyz` / `mount_rpy` place the hand on the flange, and default to
identity. This is the same transform as `hand_flange_*` in `arm_teleop`, and
they must agree, or the simulation shows one thing while the IK targets
another.

## Real xArm: startup, safe pose, IP

`xarm_bridge` owns the startup sequence and streams joint targets to the robot.
The driver itself is `xarm_api`; this node only calls its services.

```bash
# 1. the xArm driver, with the robot's address
ros2 launch xarm_api lite6_driver.launch.py robot_ip:=192.168.1.154

# 2. everything else
ros2 launch ah_mujoco arm_ui_real.launch.py robot_ip:=192.168.1.154
```

### The startup sequence

```
1. motion_enable                power the joints
2. set_mode 0, set_state 0      position mode, ready
3. set_servo_angle(safe_pose)   move to a known pose and WAIT for it
4. set_mode 1, set_state 0      servo mode
5. stream set_servo_angle_j     from /xarm/target_joints
```

Step 3 is the point of the node. Wherever the arm was left — folded, mid-task,
against a limit — it goes to one known configuration first, as a normal
position-mode move with `wait=true`. Streaming begins only afterwards, and only
once the clutch is engaged, so the arm cannot lurch from an arbitrary
configuration toward a teleoperation target.

Servo mode has no interpolation: each command executes as-is at high rate, so a
jump in the stream is a jump at the joints. The joint-space rate limit in
`arm_teleop` and the clutch gate here are what make that safe.

### The safe pose was chosen by measurement

`[0, -30, 50, 0, 40, 0]` degrees — elbow up, wrist bent.

**The all-zeros home pose is singular** (manipulability 0). IK seeded there
converges for only 156/200 nearby targets, and near a singularity small
Cartesian motions demand large joint velocities. Candidates measured:

| pose (deg) | manipulability | IK convergence | flange height |
|---|---|---|---|
| all zeros | 0.00000 | 156/200 | 0.154 m |
| `0,-30,50,0,40,0` (chosen) | 0.00412 | **199/200** | 0.415 m |
| `0,20,60,0,-40,0` | 0.00871 | 199/200 | 0.302 m |
| `0,-45,90,0,45,0` | 0.00136 | 171/200 | 0.608 m |

The chosen pose keeps at least 0.93 rad of margin on every joint limit and
holds the flange 0.42 m up, clear of a table. Override with `safe_pose`.

`arm_teleop`'s `start_joints` defaults to the same values and must stay in
step with it, otherwise the first clutch engage solves from a configuration the
arm is not in. `arm_teleop` also subscribes to `/xarm/joint_states` and reseeds
from real feedback whenever the clutch is released, so the two cannot drift.

### Before touching hardware

```bash
# no robot: the whole pipeline in MuJoCo only
ros2 launch ah_mujoco arm_ui_real.launch.py use_arm_hardware:=False

# robot connected, but log the sequence instead of executing it
ros2 run ah_mujoco xarm_bridge --ros-args -p dry_run:=True
```

`/xarm/bridge_status` reports `starting`, `moving to safe pose`, `ready`, or a
`failed: ...` reason. If a service call times out, the driver is not running or
the IP is wrong.

## Upper limb: shoulder and elbow

`upper_limb` runs MediaPipe Pose on the same image and depth topics and
publishes 3D shoulder, elbow and wrist, filtered exactly like the hand path.

```bash
cd /ws/src/HandPose
wget -O pose_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task

ros2 launch ah_mujoco upper_limb.launch.py
```

### The elbow cannot be mapped to the arm's elbow

The obvious plan — human shoulder to the arm's shoulder joints, human elbow to
the arm's elbow — does not work on a 6-DOF Lite 6. Measured, not assumed:

| test | result |
|---|---|
| rank of the full Jacobian | 6 everywhere → **null space dimension 0** |
| spare DOF after freeing tool roll | 1, but it is joint 6 |
| elbow motion from that spare DOF | **0.000 mm/rad** (joint 6 is distal to the elbow) |
| best-of-12 IK branches by elbow match | swivel error 21.0° vs 21.3° for one seed |
| forcing it with a weighted objective | swivel 22° → 7° costs **93 mm** of wrist error |

Given a wrist pose, this arm's elbow is determined. A human arm has 7 DOF and a
genuine elbow swivel freedom; the Lite 6 does not. Anything that claims to
track both is quietly sacrificing wrist accuracy.

### What the tracking is used for instead

**1. Shoulder anchoring — the main reason to do this.** The hand pose is
expressed relative to the operator's shoulder rather than the camera, so
leaning, swivelling in a chair or stepping sideways no longer drags the robot.

```bash
ros2 launch ah_mujoco arm_ui_real.launch.py anchor_to_shoulder:=True ...
```

**2. A posture diagnostic.** Human swivel is published on
`/ability_hand/<side>/upper_limb/swivel`, so the arm's posture can be compared
against the operator's even though it cannot be commanded.

**3. Recording.** Full upper-limb kinematics alongside joint data, in the
camera frame, which is what a biomechanical dataset needs.

### If you want the elbow objective anyway

It is implemented and off by default:

```bash
ros2 launch ah_mujoco arm_ui_real.launch.py elbow_weight:=0.05 ...
```

| elbow_weight | wrist position error | swivel error |
|---|---|---|
| 0.00 | 0.07 mm | 22.5° |
| 0.05 | 3.6 mm | 18.3° |
| 0.10 | 8.2 mm | 16.2° |
| 0.30 | 44.3 mm | 12.2° |
| 1.00 | 93.5 mm | 6.9° |

For a manipulation task 0 is the right value. For a demonstration where posture
matters more than precision, 0.05–0.1 is a reasonable compromise.

### The threshold should match the use, not the sensor

An early version rejected any fit above 60 mm residual, copied from the hand
path where precision matters. That threw away estimates that were perfectly
good for the actual job. Shoulder anchoring only has to remove gross body
motion - an operator shifts their shoulder by 100-300 mm when leaning or
turning, so an anchor good to ~100 mm removes most of it.

| points with depth | residual | shoulder error | usable as an anchor |
|---|---|---|---|
| 12 | 8 mm | 3 mm | yes |
| 8 | 27 mm | 17 mm | yes |
| 5 | 51 mm | 59 mm | yes |
| 4 | 82 mm | 106 mm | yes |
| 3 | 87 mm | 237 mm | marginal |

`max_residual` therefore defaults to 250 mm, and the node logs when a fit is
coarse rather than discarding it. Use the residual and inlier count on
`/ability_hand/<side>/upper_limb/swivel` to decide whether a given estimate is
good enough for what you are doing with it - posture measurement needs far
better than anchoring does.

### The Kinect is not stereo

It has one RGB camera and one time-of-flight depth camera. Depth is measured
from light travel time, not from disparity between two images, so there is no
second view to fuse. `/depth_to_rgb/image_raw` is that single depth image
warped into the colour frame.

This is why a torso clearly visible in the colour image can still have no
depth: at 720P the colour field of view is wider than NFOV_UNBINNED depth, and
pixels the depth camera never covered come back as zero.
`depth_mode:=WFOV_2X2BINNED` covers roughly 120 degrees instead of 75, at the
same 30 fps.

### Notes

Pose inference is heavier than hand inference, so this runs at 15 Hz by
default and in its own node — it does not slow the hand path. ### Robustness to depth dropouts

Requiring depth at exactly the shoulder, elbow and wrist is brittle - one
dropout on the shoulder loses the whole estimate, and dark clothing at the
shoulder is a common dropout. The node instead fits the body's metric world
landmarks to whatever landmarks DO have depth (the same rigid fit hand_pose
uses on the hand) and reads the three joints off the fitted transform.

Measured recovery of the three joints as depth coverage falls:

| body landmarks with depth | shoulder | elbow | wrist |
|---|---|---|---|
| 33 | 0.8 mm | 0.7 mm | 0.6 mm |
| 12 | 0.5 mm | 1.5 mm | 1.8 mm |
| 8 | 0.6 mm | 0.8 mm | 1.0 mm |
| 4 | 1.3 mm | 2.0 mm | 2.9 mm |
| 2 | refused | | |

Any three landmarks anywhere on the body are enough - the joints are recovered
even when their own depth is missing.

### Body landmarks need a SIMILARITY fit, not a rigid one

The hand fit forbids scale on purpose: MediaPipe's hand world landmarks are
properly metric, and letting scale float would hide depth bias in the geometry.

Body pose world landmarks are not the same thing. They are estimated body
proportions inferred from a single image, and their absolute scale is only
approximate. Forcing scale to 1 leaves 150-250 mm of residual against real
depth - which is exactly what a rigid fit reported in practice.

Measured, fitting points whose true scale differs from the model's:

| true scale | rigid fit residual | similarity fit | recovered scale |
|---|---|---|---|
| 0.85 | 68 mm | 6.0 mm | 0.854 |
| 1.00 | 6 mm | 6.2 mm | 1.000 |
| 1.15 | 79 mm | 6.9 mm | 1.146 |
| 1.30 | 144 mm | 6.6 mm | 1.300 |

### And RANSAC, because a couple of bad depths is normal

Scale alone took the residual from 144-265 mm down to 80-85 mm. The rest is
outliers: landmarks that are occluded or inferred get whatever depth lies
BEHIND them - a chair, the floor, a wall - which is metres from the true
point, and least squares lets a few such points drag the whole transform.

Measured, with outliers injected as background depth:

| outliers | least squares | RANSAC | inliers kept |
|---|---|---|---|
| 0 | 8 mm | 8.4 mm | 20 |
| 1 | 52 mm | 7.1 mm | 19 |
| 2 | **86 mm** | 7.7 mm | 18 |
| 4 | 122 mm | 7.4 mm | 16 |
| 7 | 194 mm | 10.0 mm | 13 |

Two bad points reproduce the observed 80-85 mm almost exactly. Error here is
measured on the good points only, so it shows how far the outliers dragged the
transform rather than how well it fits the noise.

The node uses `kabsch_ransac` (similarity fit, Umeyama scale clamped to
0.6-1.6, RANSAC consensus at `inlier_threshold`, default 50 mm) and fits only
landmarks 11-24 - shoulders, elbows, wrists and hips. Legs are usually behind
a desk or outside the depth camera's view and contribute nothing to an arm
anchor. Set `upper_body_only:=False` to fit the whole body.

`/ability_hand/<side>/upper_limb/swivel` carries
`[swivel, valid, n_points, residual, scale]`, so coverage, fit quality and the
recovered body scale are all visible while running.

If it still reports `no pose detected`, MediaPipe cannot find a body at all:
it needs considerably more of you in frame than hand tracking does. Stand back
until your torso and both shoulders are visible. `WFOV_2X2BINNED` widens the
depth field at the same frame rate and helps when the colour image sees you
but the depth image does not.

### Overlay

`upper_limb` publishes the shoulder/elbow/wrist pixel positions on
`/ability_hand/<side>/upper_limb/pixels` (normalised, so the overlay is
independent of display resolution), and `mediapipe_teleop` draws them on the
tracking image it already annotates. The limb is drawn in cyan against the
hand's white-and-red skeleton so the two read as separate signals, with S/E/W
labels.

Both nodes must be running: the body is detected by `upper_limb`, but the
image belongs to `mediapipe_teleop`.

### Watch the scale value

`scale` in the swivel message pinning at exactly 0.600 means the lower clamp is
binding - the fit wants to shrink the body model further than allowed. Residual
can still look good because a handful of points are easy to fit. It usually
means few inliers and a poorly conditioned fit rather than a genuinely tiny
body. Treat a pinned scale as a warning that the estimate is weakly determined,
and prefer frames where it sits between roughly 0.7 and 1.3.

### The limb chain ends at the hand's wrist

The body's wrist and the hand's wrist are the same physical point, estimated
twice with very different accuracy: `hand_pose` locates it to a few mm, the
body fit to roughly 100 mm. Estimating it twice makes the two disagree
visibly - the drawn limb ends somewhere the hand skeleton is not.

`upper_limb` therefore subscribes to `hand_pose` and uses that wrist both as a
weighted correspondence IN the fit and as the chain's endpoint after it. The
elbow takes half the correction so the upper arm is not pulled out of shape.

Measured, with realistic body-depth noise and two background outliers:

| hand wrist weight | shoulder | elbow | wrist |
|---|---|---|---|
| 0 (off) | 56 mm | 62 mm | 62 mm |
| 1 | 48 mm | 50 mm | 8 mm |
| **6 (default)** | **47 mm** | **41 mm** | **8 mm** |
| 12 | 44 mm | 45 mm | 8 mm |

The wrist improves eightfold, and the elbow and shoulder improve too - a
trustworthy correspondence constrains the whole fit, not only its own point.
The swivel angle benefits as well, since the shoulder-to-wrist axis it is
measured about is now anchored at one end.

`use_hand_wrist:=False` disables it. The swivel message gains a sixth field,
the size of the wrist correction in metres: large values mean the body fit and
the hand disagree badly and the estimate is weak.

### Inference rate and publish rate are separate

The hand pipeline runs at 60 Hz and pose inference cannot. Publishing the limb
at the inference rate makes it lag the hand and updates the shoulder anchor a
quarter as often as the pose it is correcting.

The two parts of the estimate change at very different speeds, so they are
decoupled:

* the **body-to-camera transform** (where the torso is) is re-fitted at
  `rate_hz` (15 Hz) - it drifts slowly
* the **wrist** comes from `hand_pose` at the hand's own rate
* the cached transform is re-applied with each fresh wrist at `publish_hz`
  (60 Hz), so the limb is emitted in step with the hand

Cost of holding the torso transform between inferences:

| inference Hz | hold time | torso drift @0.2 m/s | @0.5 m/s |
|---|---|---|---|
| 5 | 200 ms | 40 mm | 100 mm |
| **15** | **67 ms** | **13 mm** | 33 mm |
| 30 | 33 ms | 7 mm | 17 mm |

At 15 Hz the shoulder is held for 67 ms, so a torso moving 0.2 m/s drifts
13 mm - well inside the ~47 mm the fit itself achieves. Nothing fast is being
held: the wrist, which moves an order of magnitude quicker, is refreshed every
publish tick.

A fit older than 0.5 s is not re-used - by then the operator has moved and the
cached transform no longer describes where they are.

## One anatomy, one node, one frame

The hand and the upper limb are not two things. The wrist is a single joint
that belongs to both, and estimating it twice - in two nodes, from two frames,
exchanged over topics - makes the two halves disagree and refresh at different
rates. No amount of rate matching fixes that, because the two nodes are
processing different images.

`mediapipe_teleop` now runs BOTH landmarkers on the same frame in the same
tick, with `upper_limb:=True`:

```
one frame -> hand landmarker  -> finger targets
          -> pose landmarker  -> torso fit
          -> one depth lookup -> one wrist, shared by both
          -> /ability_hand/<side>/skeleton   (one message, one timestamp)
```

The wrist comes from the hand's own landmark and this frame's depth. It is not
predicted by the body model at all - it is measured once and used as the limb's
endpoint, so the drawn limb ends exactly where the hand skeleton begins by
construction.

Pose inference is heavier than hands, so it runs every `pose_every_n` frames
(default 3) while the torso transform is held between times and re-applied with
each fresh wrist. The OUTPUT is still one complete skeleton per frame, at the
hand's rate.

```bash
ros2 launch ah_mujoco arm_ui_real.launch.py \
    robot_ip:=192.168.1.154 \
    xarm_description_path:=/ws/src/xarm_ros2/xarm_description \
    model_path:=/ws/src/HandPose/hand_landmarker.task \
    anchor_to_shoulder:=True
```

That single launch now covers the arm, the hand, the tracking and the upper
limb. The standalone `upper_limb` node is still there for a second camera, but
it is no longer needed for this.

`/ability_hand/<side>/skeleton` carries shoulder, elbow, wrist (9 values) then
the 6 hand joint targets - the whole anatomy as one sample.

## Limb calibration (fourth step)

The calibration sequence gained a fourth step. Face the camera with your arm
visible, hold a few seconds, press `l`.

It measures your **body scale** and segment lengths (upper arm, forearm) from
the depth-backed fit, taking the median over the samples so one badly fitted
frame cannot drag the result.

### Why it helps

Without it, body scale is solved for on every frame. With only 3-6 landmarks
carrying valid depth that is under-constrained, which is why `scale` was
observed pinned at its 0.6 clamp. Measuring it once removes an unknown from
every subsequent fit:

| points with depth | free scale (shoulder / elbow) | calibrated scale |
|---|---|---|
| 4 | 62 / 60 mm | 57 / 56 mm |
| 6 | 54 / 52 mm | 40 / 40 mm |
| 10 | 43 / 43 mm | 31 / 31 mm |
| 14 | 35 / 33 mm | 23 / 22 mm |

25-35 % better wherever there is enough data to matter, and it stops the scale
running to its clamp.

The step is optional: an older calibration without it still loads, and the
hand pipeline does not use it at all.

### Sequence

| step | pose | key |
|---|---|---|
| open | flat hand, thumb spread | `o` |
| closed | fist, thumb across palm | `c` |
| baseline | move at your teleop speed | `b` |
| **limb** | **face camera, arm visible** | **`l`** |

## Keeping the UI responsive

Pose inference is about 18 ms, hand inference about 12 ms. Run inline, every
third frame costs 30 ms against a 16.7 ms budget at 60 Hz - the loop stutters
and finger targets arrive late.

Pose inference therefore runs on its own thread. The main loop hands it a frame
and carries on; a result that lands a frame or two later costs nothing, because
the torso is the slow-changing part and the wrist comes from the hand path
every frame regardless.

| | inline | off-thread |
|---|---|---|
| every 3rd frame | 30 ms → 33 fps, stutters | 12 ms → 60 fps, steady |

If it still stutters, raise `pose_every_n`. The torso only needs re-fitting a
few times a second.

## Buttons

Calibration is driven by buttons in the UI, not only keystrokes. The bar under
the header shows the action for the current step - CAPTURE OPEN HAND, CAPTURE
FIST, CAPTURE BASELINE, MEASURE ARM - with RESTART beside it. Outside
calibration it shows RECALIBRATE, SAVE, CLUTCH and THEME.

Clicks and keys go through the same handler, so the shortcuts still work:
`o` `c` `b` `l` `k` `s`, space for the clutch, `t` for the theme.

Hit testing accounts for GLFW's top-left cursor origin against the UI's
bottom-left drawing origin, and for framebuffer-to-window scaling on HiDPI
displays. A click on a button does not also drag the camera.

## The limb step is required

`body_scale`, `upper_arm_m` and `forearm_m` are required keys in the
calibration file. A calibration saved before the limb step existed is rejected
with a message saying so, rather than loading and silently leaving the body fit
solving for scale on every frame.

## All 12 DOF in the panel

The joint panel shows both groups when the arm is present:

| ARM | HAND |
|---|---|
| J1 BASE, J2 SHLDR, J3 ELBOW | INDEX, MIDDLE, RING |
| J4 ROLL, J5 PITCH, J6 ROLL | PINKY, TH FLEX, TH ROT |

Arm values are degrees, and the bar fill shows how far each joint sits through
its own range from `lite6_robot_macro.xacro` - so a bar near either end means
the joint is approaching a limit, which matters far more than its absolute
angle.

### Both groups use the same biomechanical colour scale

Arm joints are coloured by Mean Squared Jerk exactly as the hand's are, using
the same calibrated reference. That works because the dimensionless form is
amplitude and unit free - the same smooth minimum-jerk profile scores the same
whether it is a fingertip moving 8 degrees or a shoulder moving 110:

| signal | amplitude | score |
|---|---|---|
| finger flexion | 90° | 13.9 |
| thumb rotation | 100° | 13.9 |
| arm J2 shoulder | 60° | 15.4 |
| arm J3 elbow | 110° | 15.4 |
| arm J1 base, small | 8° | 12.6 |

So one reference covers a finger and a shoulder alike, and green means the same
thing in both columns: motion as smooth as the operator's calibrated baseline.
Sensitivity is comparable too - 0.05° of noise on an arm joint takes its score
from 15 to 493.

The arm's MSJ is computed from its joint trajectories in the UI node, from the
same source that drives the 3D view: real robot feedback when available, the
commanded target otherwise.

The values follow the real robot when `/ufactory/joint_states` is available and
the commanded target otherwise, matching what the 3D view shows.

## Layout

Everything that shares a row is positioned in PIXELS, not in fractions of the
window. Normalised fractions put items at fixed proportions of the width, so
at a narrower window the strings run into each other - which is exactly what
happened to the header.

The font metrics were measured rather than guessed: a known string is rendered
and the pixel extent read back. At `mjFONTSCALE_150` that gives 13.9 px per
character and 17 px cap height for the normal font, 24.8 and 33 for the big
one. The guessed values were 30 % low, which made items that were calculated
to fit overlap by half a string.

With real metrics the layout can degrade sensibly instead of colliding:

* header items are packed right-to-left with measured widths, and the title is
  only drawn if there is room left for it
* joint columns size themselves from the longest label and value actually
  present, so a label can never run under a bar
* below about 1000 px the joint bars are dropped and the values take the MSJ
  colour instead, keeping the metric readable
* the gauge starts after the widest text in the quality panel and disappears
  if fewer than 80 px remain
* button labels are truncated to their button, and the calibration banner to
  the window

## Two fixes that only showed up in use

### The MSJ reference has to travel

The UI read the calibration file once at startup, so a calibration done DURING
a session never reached it - every bar stayed red against the stale default of
524. Teleop now publishes `[reference, spread]` on
`/ability_hand/<side>/quality/reference`, and the UI applies it live.

### Stationary is not a fault, for the arm

A motionless joint has MSJ near zero, which is a large deviation from a
reference derived from motion. Symmetric colouring therefore painted a
perfectly healthy idle arm red.

For the HAND that behaviour is wanted: a collapse to zero means tracking was
lost and the fingers have frozen, which should be visible. For the ARM, holding
still is normal.

So the arm uses one-sided colouring and the hand stays symmetric:

| arm state | MSJ | symmetric | one-sided |
|---|---|---|---|
| stationary | 0 | poor (orange) | **good (green)** |
| smooth motion | 19000 | good | good |
| slightly jerky | 26000 | fair | fair |
| very jerky | 60000 | bad | bad |

Jerky motion is still flagged either way - only the "too smooth" half of the
scale differs, and that half means different things for a tracked hand and a
commanded robot.

## Session analysis

Record a session, then turn it into a report.

```bash
# during the exercise, in its own terminal
ros2 run ah_mujoco session_recorder --ros-args -p out:=/ws/vendor/session1.npz
# Ctrl-C when finished; it writes on exit

ros2 run ah_mujoco session_report --ros-args \
    -p session:=/ws/vendor/session1.npz \
    -p out:=/ws/vendor/session1.html
```

One self-contained HTML file with Plotly figures - no server, no internet.

The recorder samples every channel on ONE timer rather than on each topic's
callback. Comparing an operator signal against a robot signal only means
something if both were sampled at the same instants, and topics arriving at 15,
30, 60 and 120 Hz do not give that for free.

### What it measures

**Biofidelity - did the robot reproduce the operator's motion?**

| metric | what it means |
|---|---|
| lag | delay between operator and robot, by cross-correlation |
| gain | fraction of the operator's motion that reached the robot |
| residual RMSE | what is left after removing lag and gain - the part the link genuinely failed to convey |
| bandwidth | the -3 dB point of the transfer function: the fastest motion still commandable |
| smoothness ratio | robot jerk over operator jerk. Above 1 the link added jerk; below 1 it filtered, which costs lag |

**Biomechanical stress - what did the operator pay?**

| metric | what it means |
|---|---|
| shoulder elevation, abduction, elbow flexion | over time, against ergonomic screening bands |
| exposure | time and fraction spent outside those bands |
| static loading | sustained holds, invisible in a range-of-motion summary because the angle is simply constant |
| jerk exposure | cumulative, as a repetitive-load proxy |
| posture score | a coarse RULA-style arm flag |

### Three things that had to be got right

**Correlate velocity, not position.** Teleoperation signals are smooth, and the
cross-correlation of two smooth position traces has a very broad peak -
measured 80-100 ms of error against a known lag. Differentiating first
sharpens it: lag is then recovered to within one sample across 0-500 ms.

**Coherence is the wrong bandwidth metric.** For a noiseless linear filter,
magnitude-squared coherence stays near 1 at every frequency however much the
filter attenuates. A 0.5 Hz low-pass measured a 0.94 Hz "bandwidth" that way,
and a 5 Hz one measured 12.9 Hz. The transfer-function magnitude gives the
right answer, and coherence is reported only as a confidence measure on the
gain estimate.

**Compare Cartesian speeds, not joint velocities.** The robot's joint angles
are run through forward kinematics first. Comparing a wrist speed in m/s
against the norm of six joint velocities in rad/s compares different
quantities - correlation 0.05 on a link that was in fact tracking at 0.81.

Validated against a synthetic session built through the real IK, with a known
180 ms lag, 0.30 position scale, joint-space rate limiting and a 20 s hold:

| | truth | measured |
|---|---|---|
| lag | 180 ms | 167 ms |
| gain | 0.30 | 0.256 |
| longest hold | 20 s | 17 s |

The gain reading below the commanded 0.30 is not an error - the rate limiter
is eating motion, which is the kind of thing the report exists to surface.

### Reading it

Lag and gain are both correctable: one by prediction, the other by a setting.
They are estimated and removed before the residual is computed, so the residual
is the part that is not correctable.

A link can score well on fidelity and badly on stress, or the reverse. A rigid
high-gain mapping transmits faithfully while forcing the operator into extreme
postures; a heavily filtered one is comfortable and transmits almost nothing.
The report keeps the two apart rather than averaging them into one number.

The posture bands are ergonomic screening ranges, not anatomical limits, and
the arm score is a coarse flag - a real RULA assessment needs wrist, neck,
trunk, load and muscle use scored by a trained observer.
