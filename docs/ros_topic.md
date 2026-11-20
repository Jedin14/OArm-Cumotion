# beast_arm ROS 2 interface

Everything another system needs to drive the OpenArm on `beast`: every topic, service and message the pick stack offers under `/beast_arm`, what each command does, and the ROS interfaces the stack itself depends on.

The interface is served by either of:

- `native/run_web_gui.sh`: the browser GUI at `http://beast:8080` plus this interface (`--click` = no box model).
- `native/run_pick_stack.sh`: this interface alone, with an optional status window.

Run only one of them at a time, and bring the robot up first with `native/run_launch_everything.sh`. Source the environment before using the commands below: `source native/setup.bash`.

---

## 1. Topics and services under `/beast_arm`

All names below are relative to `/beast_arm` (change it with `--ns`). "Latched" means transient-local, depth 1: a late subscriber gets the last value at once.

| Name | Kind | Type | Direction | QoS | Purpose |
| --- | --- | --- | --- | --- | --- |
| `arm_state` | topic | `std_msgs/String` | out | latched | Where the arms are. Sent on every change and every 1 s |
| `arm_state/cmd` | topic | `std_msgs/String` | in | depth 10 | Move both arms to a pose, bring a stranded arm back, or republish the state |
| `arm_state/get` | service | `std_srvs/Trigger` | request | - | The current state, on request |
| `arm_action` | topic | `std_msgs/String` | in | depth 10 | Start a pick, or stop all motion |
| `arm_action/status` | topic | `std_msgs/String` | out | latched | Progress and result of a pick |
| `arm_action/phase` | topic | `std_msgs/String` | out | latched | The step running right now |
| `stop` | topic | `std_msgs/Empty` | in | depth 10 | Cancel every motion at once |
| `held_object/size` | topic | `pick_interfaces/ObjectSize` | in | depth 10 | Size of the box to pick next |
| `held_object/state` | topic | `std_msgs/String` | out | latched | What each arm holds now |
| `held_object/release` | topic | `std_msgs/Empty` | in | depth 10 | Take the held box out of the planners now |
| `held_object/fixed_size` | topic | `pick_interfaces/ObjectSize` | out | latched | The user's fixed box size from the GUI (Held object card), mm, at every change -- whether or not Dummy size is on. `header.frame_id` = the saved size's name, or `dummy` when unnamed (2026-11-18) |
| `held_object/get_size` | service | `std_srvs/Trigger` | request | - | Pushes `held_object/fixed_size` again (and `held_object/size` while Dummy size is on) and answers JSON in `message`: `{"fixed": {name, height_mm, width_mm, depth_mm, dummy_on}, "measured": {height_mm, width_mm, depth_mm, stamp} or null}` -- `measured` is the camera's last box from `box_model/box_size` |
| `pick_target` | topic | `pick_interfaces/PickTarget` | out | depth 10 | The target of the current pick: coordinates, approach, tool, arm, the matched box |
| `pick_result` | topic | `std_msgs/String` (JSON) | out | latched | Each pick's outcome: picked true / false / null, from the camera's check |
| `base_adjust` | topic | `pick_interfaces/BaseAdjust` | out | depth 10, not latched | When the box cannot be picked from here (out of reach, only at the edge of reach, too near the camera), or after 2 failed picks in a row on it: how the wheels should move the robot |
| `vacuum` | topic | `std_msgs/Bool` | out | latched | What the vacuum should be: on just before the cup goes in, off at the drop pose. The ESP32 relay follows it: `app/vacuum_bridge.py`, started by terminal 1 (`run_launch_everything.sh`; `vacuum:=false` leaves it out), writes `1` / `0` over USB serial |

| `status` | topic | `std_msgs/String` (JSON) | out | latched | Everything above as one JSON object, on change and every 1 s |
| `log` | topic | `std_msgs/String` | out | depth 10 | Every human-readable message, as the GUI shows it |
| `box_model/detect` | service | `pick_interfaces/DetectTarget` | request | - | Served by the box model; called by the pick stack |
| `box_model/target` | topic | `pick_interfaces/PickTarget` | out | depth 10 | Every detection the model makes |
| `box_model/check` | service | `pick_interfaces/CheckPick` | request | - | After a pick: is the box gone? Called by the pick stack |
| `box_model/check_result` | topic | `std_msgs/String` (JSON) | out | latched | Every check's answer |
| `box_model/image` | topic | `sensor_msgs/Image` | out | latched | The frame of each answer and of each check, clean (nothing drawn) |
| `box_model/scene` | topic | `std_msgs/String` (JSON) | out | latched | What that frame shows, for a GUI to draw: every box's outline and face, the target (with camera coordinates and timings), or the check's outline, verdict numbers and frames |
| `box_model/picker_view` | topic | `sensor_msgs/Image` | out | latched | The picker's own drawing: of each answer (every box, its faces, grip points) and of each check (red gone, green unchanged, blue nearer, purple another box) |
| `box_model/close_up` | topic | `sensor_msgs/Image` | out | latched | The target box cut out, its pick face boxed in red |
| `box_model/settings` | topic | `std_msgs/String` (JSON) | out | latched | The picker's settings (`frames`, `check_frames`, `score`, `order`), `skipped` / `passed` counts, the `tilt` it uses (`source`: accelerometer / saved / tf, `deg` down) and the `last` command's answer |
| `box_model/box_size` | topic | `pick_interfaces/ObjectSize` | out | latched | Each detection's target box as measured, mm, no margin (`height_mm`, `width_mm`, `depth_mm`; depth 0 = not seen). Not `held_object/size` (that is the size attached to the tool) |
| `box_model/boxes` | topic | `std_msgs/String` (JSON) | out | latched | Each detection: the target and every box after it in the picker's order -- `{time, frame, box_mm {w, h, d}, num_boxes, boxes: [{order, target, xyz (world, m), approach, cell [layer, row, column], face_mm, box_mm {w, h, d}}]}` |
| `camera/up` | topic | `geometry_msgs/Vector3Stamped` | out | latched, 1 Hz | The D455's "up" (unit, the colour camera's optical frame: x right, y down, z forward): from the live accelerometer, else its saved reading (`calibration/camera_tilt.yaml`) |
| `camera/tilt` | topic | `std_msgs/String` (JSON) | out | latched, 1 Hz | `{deg, source, live, saved, time}`: the camera's tilt in degrees (negative = looking down), what it came from, whether the accelerometer is live, and the date of the saved reading |
| `box_model/cmd` | topic | `std_msgs/String` | in | - | The picker window's keys: `set frames N` (1-15), `set score X` (0.20-0.90), `set check_frames N` (1-10), `set order v1\|v2`, `pass` (next target), `skip`, `reset` (forget skipped), `save` (colour + depth snapshot), `flag` (photos of the target), `settings`. See 1.12 |

The D455's motion sensor (2026-11-18; the realsense node, started by terminal 1; not under `/beast_arm`). All best effort: echo them with `--qos-reliability best_effort`.

| Name | Kind | Type | Rate | What it is |
|---|---|---|---|---|
| `/camera/camera/imu` | topic | `sensor_msgs/Imu` | ~200 Hz | Gyro and accelerometer in one message (`unite_imu_method` 2: the accelerometer interpolated to each gyro sample). `launch_everything ... gyro:=false` turns it off |
| `/camera/camera/accel/sample` | topic | `sensor_msgs/Imu` | ~100 Hz | Accelerometer only (`linear_acceleration`, m/s^2). The box picker's tilt comes from it (`box_model_node`, `box_picker_bridge`) |
| `/camera/camera/gyro/sample` | topic | `sensor_msgs/Imu` | ~200 Hz | Gyro only (`angular_velocity`, rad/s) |
| `/camera/camera/accel/imu_info`, `/gyro/imu_info` | topic | `realsense2_camera_msgs/IMUInfo` | latched | The motion sensor's noise and bias figures |

**Checked live 2026-11-18 13:00: these three topics exist but carry NO data.** The camera's accelerometer has sent nothing since about 2026-11-12: the D455 is on USB 2 (480M) through two hubs (the same hub as the keyboard that drops out for "EMI?"), and the kernel's motion-sensor driver fails at every start (`hid-sensor-hub 0003:8086:0B5C: No report with id 0xffffffff found`). The box picker has been using the saved tilt of 2026-11-12 10:50 (see `camera/tilt`, `source`). Fix: the camera straight into a USB 3 (blue) port on beast with a USB 3 cable, no hub; then `camera/tilt` says `"live": true`.

```bash
ros2 topic hz /camera/camera/imu --qos-reliability best_effort        # ~200 Hz when it works
ros2 topic echo --once /beast_arm/camera/tilt --qos-durability transient_local
```

The Unitree L2 4D LiDAR (2026-11-11; started by terminal 1 with `lidar:=serial|udp`: the user's own driver, `4D_lidar`'s `l2_driver`, and `robot/lidar_bridge.py`; not under `/beast_arm`):

| Name | Kind | Type | Direction | QoS | What it is |
|---|---|---|---|---|---|
| `/l2/cloud` | topic | `sensor_msgs/PointCloud2` | in | best effort | The driver's cloud, 10 Hz, ~4000 points (frame `l2_lidar`) |
| `/l2/imu` | topic | `sensor_msgs/Imu` | in | depth 400 | The L2's IMU, ~250 Hz (checked live 2026-11-18: 250.6 Hz; frame `l2_imu`, which the driver hangs on `l2_lidar`). Its accelerometer's say is reported in `/lidar/status`; used for the tilt only with `lidar_tilt:=imu`. Its accelerometer reads ~1.9 g at rest (15.6 m/s^2 on x on 2026-11-18) -- the L2's own scale, not corrected |
| `/lidar/cloud` | topic | `sensor_msgs/PointCloud2` | out | default | The same points in `l2_lidar`, which hangs on `openarm_body_link0` by the mount (static TF): RViz |
| `/lidar/cloud_work` | topic | `sensor_msgs/PointCloud2` | out | default | Only the work area in front (x 0.15–1.5 m from the body): what goes to the octomap |
| `/lidar/cloud_octomap` | topic | `sensor_msgs/PointCloud2` | out | default | What MoveIt's `l2_lidar` updater reads: the work clouds of 0.8 s after each `/octomap_gater/refresh` (octomap:=static), or every one (live) |
| `/merged_cloud` | topic | `sensor_msgs/PointCloud2` | out | default | The D455's depth (coloured) and the last 1.5 s of LiDAR (coloured by height), in `openarm_body_link0`, 2 Hz while someone subscribes |
| `/lidar/status` | topic | `std_msgs/String` (JSON) | out | latched, 1 Hz | `ok`, `rate_hz`, `work_points`, `reach_points` (below the LiDAR, where the arms work: x 0.15–1.0, z up to 0.7), `mount_rpy_deg`, `mount_from` (`measured` / `imu`), `camera_in_view` / `camera_agree_pct` (its points in the D455's picture, and how many sit at the camera's depth: the mount's check), `imu_g` / `imu_lean_deg` (the accelerometer's say), `lean_deg` (with `lidar_tilt:=imu`), `why`. `ok` = 10 clouds/s, 200 reach points in 1.5 s, and the camera agreeing (30% of 50+ points within 4 cm). While `ok`, a capture removes the unseen-space walls; not `ok` (or no word for 3 s): the walls are back |

### 1.0 Tool and arm

Every pick says its tool, and the tool fixes the arm:

| Tool | Arm | The other arm |
| --- | --- | --- |
| `vacuum` (default) | `left` | The right arm is never moved for a vacuum pick. A target in the right half of the image, or one that names `right`, is still picked by the left arm, and the correction is logged |
| `finger` | `right` | The left arm is never moved for a finger pick |

Pose moves are for both arms, whatever the tool: `pre_pick` / `navigation` (`arm_state/cmd`, the GUI buttons) and the move to pre_pick that starts every pick (`ready_to_pick`, Detect box, Move). Only the pick itself (approach, contact, back out, drop, return) is the tool's arm alone; the other arm stays at pre_pick throughout. Arm calibration (Cal arms) also uses both arms, whatever the tool. Set the tool with `ready_to_pick ... vacuum|finger`, the GUI's Gripper switch, or `--pick-tool` / `pick_tool:=` at launch.

### 1.1 `arm_state` (out)

| Value | Meaning |
| --- | --- |
| `navigation` | Both arms at the navigation pose |
| `pre_pick` | Both arms at pre_pick: ready for a pick |
| `moving_to_navigation`, `moving_to_pre_pick` | Moving between the two |
| `checking` | After a pick: the camera's check |
| `detecting` | The model is finding the box; the camera map is being rebuilt |
| `picking` | An arm is out: approach, contact, drop, return |
| `returning` | Bringing a stranded arm back (`return`) |
| `arm_out` | A pick stopped half-way with an arm away from pre_pick. Send `return` |
| `unknown` | Neither pose: at boot, or moved by hand. Send `pre_pick` or `navigation` |

### 1.2 `arm_state/cmd` (in)

| Command | Does |
| --- | --- |
| `navigation` | Both arms to the navigation pose (right arm first) |
| `pre_pick` | Both arms to pre_pick |
| `return` | Brings back an arm a pick left out (`arm_out`): out along the tool, then the way it came |
| `status` | Publishes `arm_state` now |

`navigation` and `pre_pick` first bring back an arm that was left out. Any of these clears a STOP.

### 1.3 `arm_action` (in)

| Command | Does |
| --- | --- |
| `ready_to_pick` | Full pick with the current tool (default `vacuum`): both arms to pre_pick if needed, the model takes 5 frames and chooses the box and its face, the camera map is rebuilt, the tool's arm picks it (approach, corrections, contact; the box is attached in the planners), drops it at the drop pose, back to pre_pick, then the camera checks the box is gone |
| `ready_to_pick [top\|front\|auto] [vacuum\|finger]` | The same with the approach and/or the tool given, in any order, e.g. `ready_to_pick front vacuum`. The tool stays set for later picks. `left` / `right` words are ignored when they disagree with the tool |
| `stop` | Cancels every motion at once (the same as the `stop` topic) |

A command that arrives while another job runs is refused with `busy` on `arm_action/status`; nothing is queued. `stop` always works, even while busy. In click-only mode (`run_web_gui.sh --click`), `ready_to_pick` answers `pick_failed: click-only mode (--click), no box model`.

### 1.4 `arm_action/status` (out)

| Value | Meaning |
| --- | --- |
| `busy` | Refused: something else is running |
| `detecting` | The model is looking |
| `target_found` | The target is on `pick_target`; the reach check runs next |
| `picking` | The arm is moving to the box |
| `checking` | The arm is back; the camera compares the box's place with the frames from before the pick |
| `pick_complete` | The camera saw the box gone from its place (if the check is not served: contact felt, box dropped, arm back) |
| `pick_unverified: <outcome> - <why>` | The arm is back, but the camera could not tell (`blocked`, `camera_moved`, `unknown`); see `pick_result` |
| `stopped` | STOP was received |
| `no_target: <reason>` | Nothing to pick in the frame |
| `pick_failed: <reason>` | Unreachable, no contact, stopped short, did not get back, or the camera says the box was not taken (`missed`, `moved`, `dropped`, `wrong_box`). The reason says which |
| `retrying: <outcome>` | Not taken; the stack looks again and picks again by itself (not the end of the command) |

Every `ready_to_pick` ends in exactly one of `pick_complete`, `pick_unverified: ...`, `no_target: ...` or `pick_failed: ...` (or `stopped`). A pick that fails and will be tried again by itself reports the non-terminal `retrying: <outcome>` instead; only the last attempt's result is terminal. STOP is checked between every step (pre_pick, detection, reach check, touch) and ends the job with `stopped`. A command sent while a job runs is answered `busy` (also `arm_state/cmd`) and does not clear a STOP.

### 1.4b `base_adjust` (out)

Straight away when the reach check finds no way to pick the box from where the robot stands (`reason: unreachable`): every approach point from 10 cm down to 4 cm tried within `--plan-budget` (30 s; the search stops at the first workable path); or only at the edge of the arm's reach, where there is no room to correct the tool (corrections there diverged and knocked a neighbouring box over); or the box nearer the camera than `--min-depth` (0.35 m), where its depth is not reliable. And after `--adjust-after` (default 2) picks in a row that did not take the same box (`picked: false`; can't-tell results are not counted), the stack publishes one `pick_interfaces/BaseAdjust`. It says how far the robot should move **sideways only — left or right, never forward, back or a turn** — so the box lines up across the robot with the picking arm's sweet spot (`--sweet-spot` y, 0.10 m for the left arm): the move is the box's y minus that. The arms do nothing more on their own; after the base has moved, send `ready_to_pick` again (or Detect box in the GUI). The count resets after a request, a success, or a pick on another box.

| Field | Meaning |
| --- | --- |
| `header.frame_id` | `world` = the robot body: x forward, y left |
| `direction` | `left`, `right`, or `none`: the box is already lined up sideways (within 1 cm), and moving left or right cannot help — it is too near or too far |
| `distance_m` | How far to move that way, m (0 for `none`) |
| `left_m` | The same as one signed number: + left, − right |
| `target`, `goal` | The box now, and where the arm wants it (only `goal.y` matters for the move) |
| `arm`, `tool`, `approach` | `left` / `vacuum`, `front` / `top` |
| `failures`, `reason` | How many picks failed and their outcomes, e.g. `missed, moved` |

Drive straight sideways by `distance_m` towards `direction` (a base that cannot strafe: turn 90°, drive, turn back). The topic is not latched, so a restarted wheel node never acts on an old request. The same request is in `pick_result.base_adjust`, and the failing pick's status ends with `; base_adjust requested: ...`.

### 1.4a `pick_result` (out, JSON)

Latched. One object for each pick (also a GUI Move after Detect box):

```json
{"picked": true, "outcome": "success", "message": "PICK SUCCESS: the box is gone from its place",
 "contact": true, "held": true, "held_g": 96.0, "tool": "vacuum", "arm": "left", "approach": "front",
 "xyz": [0.512, 0.104, 0.231], "box_mm": [170, 100, 25], "time": "2026-10-28 14:02:11",
 "verdict": {"outcome": "success", "gone": 0.94, "same": 0.03, "closer": 0.0, "disturbed": [], "...": "..."}}
```

| `picked` | `outcome` | Meaning |
| --- | --- | --- |
| `true` | `success` | The box is gone from its place |
| `true` | `replaced` | Another box now stands where the target was (a neighbour slid or leaned into the gap). The box model counts the boxes around the spot to settle it, so this normally arrives as `success` (one box fewer) or `moved` (as many) |
| `false` | `missed` | The box is still there, unchanged |
| `false` | `moved` | Part of it changed: pushed, tipped, pulled part way |
| `false` | `dropped` | Gone, but the gripper says it is empty (only when the gripper can tell) |
| `false` | `wrong_box` | The target is still there, a neighbour is gone |
| `false` | `no_contact` | The arm never reached the box |
| `null` | `blocked` | Something was in front of the boxes (arm, hand), or not still; tried twice |
| `null` | `camera_moved` | The scene does not line up with before |
| `null` | `unknown` | Too little depth on the target to say |
| `null` | `arm_out` | The arm did not get back; send `return` |
| `false` | `empty` | No camera check available, and the joint torques show nothing on the cup |
| `false` | `unreachable` | Not pickable from where the robot stands (see `base_adjust`) |
| `null` | `not_started` | Refused before the arm moved (no plan now, a box still on the cup); not counted, not retried |
| `null` | `stopped` | STOP before the arm moved |
| `null` | `check_timeout` | The camera check did not answer: `pick_unverified` |
| `true`/`false` | (no `verdict`) | The camera check is not served or the picker is down: contact and weight decide |

The check needs the arm out of the view and the camera where it was, which holds after the arm returns to pre_pick.

`held` / `held_g`: whether the cup came away with the box, from the joint torques. The arm holds still at 3 cm short of the box before the press and again after it, arriving the same way both times, and the difference in the shoulder and elbow torques is the box's weight. `held` is null when the torques were too unsteady to tell. It is passed to the camera check, so a box that left its place but is not on the cup reads `dropped`. When nothing is on the cup, the arm presses again from there, 5 mm deeper and 0.5 s longer, up to `--press-retries` (2) times.

A pick that still fails is tried again automatically: the camera looks, then the same box is picked again, or it is detected again if it moved. This goes on until `--adjust-after` (2) failures on the box, which sends `base_adjust`. STOP ends it.

### 1.5 `arm_action/phase` (out)

The step running now, as `<arm> <step>` or `both <step>`; `idle` between jobs.

| Phase | Step |
| --- | --- |
| `both to_pre_pick`, `both to_navigation` | Both arms moving to a pose |
| `<arm> approach` | Flying to the stand-off point in front of (or above) the box |
| `<arm> turn_down` | Top pick: turning the tool down above the box |
| `<arm> correct` | Measuring and correcting the tip position at the stand-off |
| `<arm> in` | Straight line in towards the box |
| `<arm> contact` | The last, slow, guarded centimetres, until contact is felt |
| `<arm> holding` | At the surface (and the held box attached, if a size was given) |
| `<arm> off_surface` | Backing off the surface |
| `<arm> back_out` | Straight out along the tool |
| `<arm> box_clear` | Further out, until the held box has left its old spot |
| `<arm> to_pre_pick_top` | Top pick, after the press: to pre_pick_top (near pre_pick), the tool still pointing down so the box stays level; on to the drop from there |
| `<arm> turn_level` | Top pick without pre_pick_top: turning the tool level again |
| `<arm> return` | Back to pre_pick (replayed if clear, else planned): front picks |
| `<arm> to_drop`, `<arm> drop` | To the drop pose; releasing the box there |
| `<arm> to_pre_pick` | From the drop pose back to pre_pick |
| `<arm> done` | Back at pre_pick |
| `<arm> check` | The camera checks the box is gone |
| `idle` | Nothing running |

### 1.6 `stop` (in)

`std_msgs/Empty`. Cancels the goal running now; the arms hold where they are. Every new motion is refused until a command that moves is sent (`arm_state/cmd`, `ready_to_pick`, or a move in the GUI), which clears it.

### 1.7 `held_object/size` (in), `held_object/state` (out), `held_object/release` (in)

Publish the size of the box before the pick. At a felt contact the stack attaches a box of that size to the arm's tool in both planners (MoveIt: an exact box; cuMotion: up to 32 spheres that cover it completely), so the way back keeps it clear of everything.

1. At contact: the box is attached (`held_object/state` becomes `left 170x100x25 mm`).
2. It backs out along the tool until the whole box has left its old spot: the box's length along the tool plus 3 cm. Only on this straight line may the box overlap the camera map, which still shows the box where it stood.
3. The way back replays the approach only if that path is clear with the box on; otherwise a new path is planned around everything.
4. At the drop pose the box is released (`held_object/state` becomes `none`). Without a drop pose it is released back at pre_pick.

How the box sits on the tool:

- Front pick: height vertical, depth along the tool, width across.
- Top pick: height along the tool; which way width and depth lie is not known, so both take the larger of the two.

Rules:

- All zeros on `held_object/size` = no object: nothing is attached.
- The latest size is used for every pick until it changes.
- While an arm holds a box, a new pick with that arm is refused. Send `held_object/release` (or press Release in the GUI) once the box is gone some other way.
- The GUI's Dummy size publishes its H x W x D (`frame_id: dummy`) on this topic every second; turning it off publishes zeros. Since 2026-11-18 it is ON by default, keeps the last size across restarts, and sizes can be saved by name (Held object card: Save / the saved-sizes list / Delete) -- all in `calibration/held_sizes.json`. While it is on, it wins over the box model's measured size for what is attached.

### 1.8 `status` (out, JSON)

One JSON object with everything, on every change and every second:

```json
{
  "arm_state": "picking",
  "phase": "left contact",
  "mode": "pick",
  "busy": true,
  "halted": false,
  "result": "picking",
  "top_down": false,
  "pick": {"arm": "left", "xyz": [0.452, 0.061, 0.512], "ok": true, "why": null},
  "held": {"left": {"size_mm": [170, 100, 25], "top": false}},
  "object_size_mm": [170, 100, 25],
  "object_size_source": "dummy",
  "pick_tool": "vacuum",
  "tool_arm": "left",
  "pick_result": null,
  "target": {"approach": "front", "tool": "vacuum", "arm": "left", "confidence": 1.0,
             "xyz": [0.452, 0.061, 0.512], "face_mm": [168, 99], "box_mm": [170, 100, 25]},
  "tools": {"left": {"on": true, "length_mm": 20}, "right": {"on": false, "length_mm": 20}},
  "pieces_on": false,
  "message": "left arm: touching (felt the surface); holding 2 s"
}
```

`halted` is true after STOP. `pieces_on` is true while the calibration pieces (ArUco ext) are in the planners. Use `ros2 topic echo --full-length` to read it in a terminal; the default echo folds long strings.

### 1.9 `box_model/detect` (service)

Called by the pick stack on `ready_to_pick` and by the GUI's Detect box. Any node can serve it instead of `box_model_node.py` (start with `model:=false`). The box picker (`box_picker/`, SAM 3.1) looks at 5 frames. On each it finds every box's outline and its 3D faces, then chooses by its rules: boxes on top first, nothing in front, nearest, then leftmost. The frames must agree (at least half). The target is the centre of the face to grip. The picker's rules decide `front` or `top`; a forced approach is only reported.

The answer fills `tool` and `arm` (from the request's `tool`, else the node's `tool` parameter) and the measured face: `face_mm` (long, short), `face_normal`, `face_long` (world unit vectors).

### 1.10 Box size matching

When a box size is known (`held_object/size`, or the GUI's dummy size), the pick stack matches the measured face to the box's faces in any order. For a 170 × 100 × 25 box, a face seen as 168 × 99 or 99 × 168 is the 170 × 100 face, with 25 mm along the tool. A face seen as 171 × 24 is the 170 × 25 face, with 100 mm along the tool. Each edge may be off by 15 mm or 12 %, whichever is larger. The result is `box_mm` on `pick_target` (along `face_long`, across it, depth). At contact the box is attached to the tool in MoveIt and cuMotion with exactly that size, its long edge turned as the camera saw it. With no match (or no face size), the box is attached by the tool's direction as before and the log says why.

### 1.11 `box_model/check` (service)

`pick_interfaces/CheckPick`: the box picker's own pick check (`box_picker/src/pick_check.py`). It takes 3 new frames and compares them with the 5 frames of the last detection, inside the target's outline (depth, then colour), and checks the neighbours too. No SAM, a few milliseconds. The pick stack calls it after every pick. Each answer also goes on `box_model/check_result` (JSON) and its drawing on `box_model/image`.

---

### 1.12 `box_model/cmd` (in) and `box_model/settings` (out)

What the box picker's own window (`box_picker/src/pick_loop.py`) does, as one-line commands. Each is answered on `box_model/settings` (latched JSON): the settings after it, and `last` = `{text, ok, message, path, photos, time}`.

| Command | Does |
| --- | --- |
| `set frames N` | Frames per target, 1-15 (default 5), from the next detection |
| `set score X` | SAM's confidence needed to count as a box, 0.20-0.90 (default 0.4; thin boxes standing on end score 0.45-0.6) |
| `set check_frames N` | New frames the pick check compares with the detection's, 1-10 (default 3) |
| `set order v1` / `v2` | Which box first: v1 stack tops (highest first), then nearest, then leftmost; v2 the leftmost free box |
| `pass` | Next target: the last target, and every box under it, is passed over until the next successful pick. With no other box left, the first again |
| `skip` | The last target is never chosen again until `reset` (it still blocks the boxes behind and under it) |
| `reset` | Forget the skipped and passed boxes |
| `save` | The newest colour + depth, with the tilt, in `box_picker/src/outputs/gui/<run>/snapshots/` (`pick_pipeline.py --capture` replays it) |
| `flag` | Three photos of the last target for a person (the scene with it marked, close up, the last check) in `.../flags/` |

Skipped and passed boxes are remembered by their place in the picture, so they are only right while the base stays put. The pick stack sends `flag` when it gives up on a box (base_adjust), and `skip` when moving the base cannot help (`none`); when the base is asked to move it sends `reset`.

```bash
ros2 topic pub --once /beast_arm/box_model/cmd std_msgs/msg/String "{data: 'set order v2'}"
ros2 topic echo /beast_arm/box_model/settings
```

## 2. Messages (`pick_interfaces`)

The package is `src/pick_interfaces`. After a change: `native/build_ws.sh --packages-select pick_interfaces`.

### 2.1 `ObjectSize.msg`

| Field | Type | Meaning |
| --- | --- | --- |
| `header` | `std_msgs/Header` | `frame_id` = who says so (`dummy` from the GUI) |
| `height_mm` | `float32` | Vertical, as the box stands |
| `width_mm` | `float32` | Sideways, across the camera's view |
| `depth_mm` | `float32` | Away from the camera, along a front approach |

### 2.2 `PickTarget.msg`

| Field | Type | Meaning |
| --- | --- | --- |
| `header` | `std_msgs/Header` | `frame_id` = frame of `position` (`world`); `stamp` = the image's time |
| `position` | `geometry_msgs/Point` | Where the tool touches, m. Top: on the top, 2 cm in from its near edge. Front: on the face the camera sees, at the box's centre |
| `approach` | `string` | `top` or `front` (constants `APPROACH_TOP`, `APPROACH_FRONT`) |
| `arm` | `string` | `left` or `right`: always the tool's arm |
| `confidence` | `float32` | Share of the 5 frames that agree, 0 to 1 |
| `u`, `v` | `float32` | Face centre in the colour image, pixels |
| `bbox` | `float32[4]` | Zeros (the picker gives the face, not a box) |
| `image_width`, `image_height` | `uint32` | Colour image size, pixels |
| `num_boxes` | `uint32` | Every box the model saw in the frame |
| `tool` | `string` | `vacuum` (left arm) or `finger` (right arm) (constants `TOOL_VACUUM`, `TOOL_FINGER`) |
| `face_mm` | `float32[2]` | The gripped face as measured: long, short edge, mm |
| `face_normal` | `geometry_msgs/Vector3` | Unit, out of the face, world frame |
| `face_long` | `geometry_msgs/Vector3` | Unit, along the face's long edge, world frame |
| `box_mm` | `float32[3]` | The known box matched to the face: mm along `face_long`, across it, into the box. Zeros: no size known or no match |

### 2.3 `DetectTarget.srv`

```
string approach        # "auto", "top", "front"; "" = the model node's parameter
string tool            # "vacuum" (left arm) or "finger" (right arm); "" = the node's `tool` parameter
---
bool found
string message         # why not, when not found
PickTarget target
```

### 2.4 `CheckPick.srv`

```
string holding         # "held", "empty", or "" (the gripper can't tell; the vacuum has no sensor)
---
int8 picked            # 1 picked, 0 not picked, -1 can't tell (or nothing to check)
string outcome         # success, replaced, missed, moved, dropped, wrong_box, blocked, camera_moved, unknown; "" = nothing to check
string message
string verdict_json    # the check's whole verdict
```

---

## 3. Command examples

```bash
# a full pick, and following it
ros2 topic pub --once /beast_arm/arm_action std_msgs/msg/String "{data: ready_to_pick}"
ros2 topic echo /beast_arm/arm_action/status
ros2 topic echo /beast_arm/arm_action/phase

# force the approach / say the tool (vacuum = left arm, finger = right arm)
ros2 topic pub --once /beast_arm/arm_action std_msgs/msg/String "{data: ready_to_pick top}"
ros2 topic pub --once /beast_arm/arm_action std_msgs/msg/String "{data: ready_to_pick auto vacuum}"

# did it work? (the camera's check)
ros2 topic echo /beast_arm/pick_result --full-length
# what the wheels should do after 2 failed picks on one box
ros2 topic echo /beast_arm/base_adjust

# the size of the box to pick (mm), before the pick
ros2 topic pub --once /beast_arm/held_object/size pick_interfaces/msg/ObjectSize \
  "{header: {frame_id: wms}, height_mm: 170, width_mm: 100, depth_mm: 25}"
ros2 topic echo /beast_arm/held_object/state
ros2 topic pub --once /beast_arm/held_object/release std_msgs/msg/Empty
# the GUI's fixed size, and the camera's measured one (source native/setup.bash first)
ros2 topic echo /beast_arm/held_object/fixed_size --qos-durability transient_local
ros2 service call /beast_arm/held_object/get_size std_srvs/srv/Trigger
ros2 topic echo /beast_arm/box_model/box_size --qos-durability transient_local

# poses
ros2 topic pub --once /beast_arm/arm_state/cmd std_msgs/msg/String "{data: pre_pick}"
ros2 topic pub --once /beast_arm/arm_state/cmd std_msgs/msg/String "{data: navigation}"
ros2 topic pub --once /beast_arm/arm_state/cmd std_msgs/msg/String "{data: return}"
ros2 topic echo /beast_arm/arm_state
ros2 service call /beast_arm/arm_state/get std_srvs/srv/Trigger

# stop everything
ros2 topic pub --once /beast_arm/stop std_msgs/msg/Empty

# everything at once
ros2 topic echo /beast_arm/status --full-length

# ask the model directly (nothing moves)
ros2 service call /beast_arm/box_model/detect pick_interfaces/srv/DetectTarget "{approach: auto, tool: vacuum}"
# ... and, after a pick done some other way, with the arm out of view
ros2 service call /beast_arm/box_model/check pick_interfaces/srv/CheckPick "{holding: ''}"
```

A typical sequence from another system:

1. `arm_state` is `navigation`; send `pre_pick` on `arm_state/cmd` and wait for `arm_state` = `pre_pick`.
2. Publish the box size on `held_object/size`.
3. Send `ready_to_pick` on `arm_action`.
4. Follow `arm_action/phase`; wait for `arm_action/status` = `pick_complete` (or `pick_unverified: ...` / `no_target: ...` / `pick_failed: ...`); the details are on `pick_result`.
5. Repeat from 2, or send `navigation` on `arm_state/cmd` to drive away.

---

The obstacle box under the arms (2026-11-19; `robot/height_box.py`, started by terminal 1, `height_box:=off`: none; not under `/beast_arm`):

| Name | Kind | Type | Direction | QoS | What it is |
|---|---|---|---|---|---|
| `/c60000004/vertical_distance` | topic | **`sensor_msgs/Range`** (a range sensor: `range` in m, x = range × 100 cm; ±inf or outside `min_range`..`max_range` = no reading, REP 117). Also taken: a number (cm), a string like `"45 cm"` | in | best effort (takes reliable too) | x. The type is the publisher's, looked up when it appears and again when it changes |
| `/c60000004/vertical_distance` (pick stack) | topic | `sensor_msgs/Range` | in | best effort | Also read by the pick stack (`app/ctm/robot.py`): while x is 87–152 cm (`--no-drop-x`) a held box is let go at pre_pick instead of the drop pose, which would collide |
| `/height_box/status` | topic | `std_msgs/String` (JSON) | out | latched, 1 Hz | `source` (`live` / `default`), `x_cm`, `gap_cm`, `top_z` (m, `openarm_body_link0`), `clamped`, `held_by` (`camera_link` / `l2_lidar_body` / an arm base), `limit_top_z`, `pieces`, `stand_holes` ([z lo, z hi, x lo, y lo, x hi, y hi]), `arm_in_box` (links), `subscribed`, `age_s` |
| `height_box` | planning-scene object | box 0.20 x 0.60 x 0.28 m | out | | MoveIt and cuMotion plan round it. Its top is **152 - (x + 10) cm** below the arms' centre (0.0126, 0.0062, 0.7344), or **100 cm** when no value has come in for 3 s. It is centred under that point: 10 cm to the front and back, 30 cm to the left and right, 28 cm deep downward. It goes up round the stand (the aluminium column): it is sent in pieces round a hole that follows the stand's URDF shape layer by layer (~91% of the box kept at the foot), so the stand never counts as touching it. Only the camera, the LiDAR and the arm bases hold it down: its top stays 1.5 cm under the lowest of them over its footprint (the camera: top <= ~0.549 m, so x up to ~123.5 cm; above that `clamped`, `held_by`). Arm links inside it are named in `arm_in_box`: MoveIt and cuMotion will not plan from a start in collision. Until TF is up it stays under the stand's foot (z -0.01). See-through blue in RViz |

```
ros2 topic echo /height_box/status --qos-durability transient_local
ros2 topic pub /c60000004/vertical_distance sensor_msgs/msg/Range "{header: {frame_id: c60000004_range}, radiation_type: 1, min_range: 0.0, max_range: 1.52, range: 0.45}" -r 5   # test: gap 97 cm
ros2 launch robot/vertical_distance.launch.py value_cm:=45 mode:=fixed   # the stand-in publisher with a slider (robot/vertical_distance_pub.py): sensor_msgs/Range; fixed / sweep / off
ros2 param set /height_box default_gap_cm 100.0      # also total_cm, offset_cm, size, arm_centre, timeout_s, max_top_z, rgba
```

The guard round the D455 (2026-11-19; `robot/camera_guard.py`, started by terminal 1, `camera_guard:=off`: none; not under `/beast_arm`):

| Name | Kind | Type | Direction | QoS | What it is |
|---|---|---|---|---|---|
| `camera_guard` | planning-scene object | 6 boxes | out | | A see-through orange shell round the camera: 20 mm in front (7 mm off its face), 15 mm at each end, over its front lip and under its two ends. There are plates rather than one box because the camera sits half inside the body column (in the URDF), and a box over it would put the robot in collision with itself. It follows the camera's TF (sensor_align) and is checked against the body and the arm bases before every send: a plate within 3 mm of them is left out and logged. MoveIt and cuMotion plan round it (cuMotion did not know the camera at all before) |
| `/camera_guard/status` | topic | `std_msgs/String` (JSON) | out | latched | `plates_sent`, `left_out` ([name, mm]), `nearest_robot_mm`, `camera_xyz` |

## 4. Launch options

| Launch argument | Values | Default | Effect |
| --- | --- | --- | --- |
| `gui` (pick stack only) | `auto`, `on`, `off` | `auto` | Status window; `auto` only with a display |
| `approach` | `auto`, `top`, `front` | `auto` | Default approach for `ready_to_pick` |
| `pick_tool` (`--pick-tool`) | `vacuum`, `finger` | `vacuum` | The tool, and so the arm: vacuum = left only, finger = right only |
| `--adjust-after` (script) | number | `2` | Failed picks in a row on one box before `base_adjust` (0: never); the picks before it are retried automatically |
| `--press-retries` (script) | number | `2` | Presses again (deeper, longer) when the torques show nothing on the cup |
| `--plan-budget` (script) | s | `30` | How long the reach check may search (it stops at the first workable path); nothing found by then: not from here (`base_adjust`) |
| `--edge` (script) | `try`, `adjust` | `adjust` (pick stack, GUI) | A box reachable only at the edge of reach: try it anyway, or ask the wheels to move |
| `--prefer-top` / `--no-prefer-top` (script) | flag | on | A FRONT target on a face narrower than `--narrow-face-mm` (a flat box's 25 mm side), of a box with nothing on it whose top is wide (box size known), is picked from the TOP: its centre from the front face and the box size |
| `--narrow-face-mm` (script) | mm | `45` | Front faces narrower than this prefer the top |
| `--edge-face-mm` (script) | mm | `25` | With `adjust`, an edge-of-reach plan is still tried when the face's short side is at least this wide (25: the cup fits on it); the allowed miss before pressing then grows with the face, up to 25 mm |
| `--min-depth` (script) | m | `0.35` | Boxes nearer the camera than this are not picked from here (`base_adjust`) |
| `--held-min-g` (script) | g | `30` | Weight on the cup that counts as holding the box |
| `--sweet-spot X Y` (script) | m | `0.45 0.10` | Where the picking arm works best; `base_adjust` brings the box there |
| `model` | `true`, `false` | `true` | Start `box_model_node.py` |
| `camera_guard` (terminal 1) | `on`, `off` | `on` | The see-through guard round the D455 |
| `--no-drop-x MIN MAX` (script) | cm | `87 152` | x in this range: the box is let go at pre_pick, not at the drop pose (`0 0`: off) |
| `height_box` (terminal 1) | `on`, `off` | `on` | The obstacle box under the arms from `/c60000004/vertical_distance` |
| `click` (web GUI; `--click` on the script) | `true`, `false` | `false` | Click-only: no box model |
| `host`, `port` (web GUI) | address, number | `0.0.0.0`, `8080` | Where the page listens |

Pick stack options (`pick_stack.py`, also accepted by `web_gui.py`): `--ns` (default `/beast_arm`), `--state-period` (1.0 s), `--status-topic` (default `<ns>/arm_action/status`), `--model-service` (default `<ns>/box_model/detect`), `--drop-hold` (1.0 s at the drop pose; negative skips it), plus every option of `click_to_move.py`.

Model node parameters (`box_model_node.py --ros-args -p name:=value`): `ns` (`/beast_arm`), `approach` (`auto`, only reported), `frames` (5), `tool` (`vacuum`), `world_frame` (`world`), `picker_dir` (`box_picker/src`).

---

## 5. ROS interfaces the stack uses

These belong to the robot stack (`run_launch_everything.sh`); the pick stack is a client of them. Listed for debugging.

| Name | Kind | Type | Used for |
| --- | --- | --- | --- |
| `/camera/camera/color/image_raw` | topic, in | `sensor_msgs/Image` | Colour frames |
| `/camera/camera/aligned_depth_to_color/image_raw` | topic, in | `sensor_msgs/Image` | Depth aligned to colour |
| `/camera/camera/color/camera_info` | topic, in | `sensor_msgs/CameraInfo` | Intrinsics |
| `/joint_states` | topic, in | `sensor_msgs/JointState` | Joint positions and torques (contact detection) |
| `/tf`, `/tf_static` | topics, in | `tf2_msgs/TFMessage` | Camera and tool poses |
| `/move_action` | action | `moveit_msgs/MoveGroup` | Joint-goal planning (cuMotion, OMPL fallback) |
| `/execute_trajectory` | action | `moveit_msgs/ExecuteTrajectory` | Running planned paths; cancelled by STOP |
| `/compute_ik`, `/compute_fk` | services | `moveit_msgs/GetPositionIK`, `GetPositionFK` | Reach checks |
| `/compute_cartesian_path` | service | `moveit_msgs/GetCartesianPath` | Straight lines in and out |
| `/check_state_validity` | service | `moveit_msgs/GetStateValidity` | Checking the way back with a held box |
| `/get_planning_scene`, `/apply_planning_scene` | services | `moveit_msgs/GetPlanningScene`, `ApplyPlanningScene` | Collision exemptions; tool, pieces and held-box objects |
| `/clear_octomap` | service | `std_srvs/Empty` | Rebuilding the camera map |
| `/octomap_gater/refresh` | service | `std_srvs/Trigger` | Feeding fresh frames to the map |
| `planner_attach_object` | action | `isaac_ros_cumotion_interfaces/UpdateLinkSpheres` | cuMotion spheres: tool, calibration pieces, held box |
| `/{left,right}_gripper_controller/gripper_cmd` | action | `control_msgs/GripperCommand` | Closing the gripper (only with `--close-gripper`) |
| `/robot_state_publisher/get_parameters`, `set_parameters` | services | `rcl_interfaces` | Camera calibration (APPLY) |

Published for RViz:

| Name | Type | What |
| --- | --- | --- |
| `/click_to_move/marker` | `visualization_msgs/MarkerArray` (latched) | The picked point |
| `/click_to_move/target` | `geometry_msgs/PoseStamped` | The tool pose aimed for at contact |

cuMotion reserves sphere links for the optional collision shapes (in `openarm.yml`): `openarm_<arm>_pieces` (14 spheres, calibration pieces), `openarm_<arm>_tool` (4, the tool past the fingertips) and `openarm_<arm>_held` (32, the held box). After editing `openarm.yml`, restart `run_launch_everything.sh`.

---

## 6. Notes

- The pick stack's node is named `click_to_move` in `ros2 node list`.
- No login: anyone who can open the web GUI or publish on these topics can move the arms.
- The OpenCV tools (`click_to_move.py`, `calibrate.py`) do not offer this interface; only `run_web_gui.sh` and `run_pick_stack.sh` do.
