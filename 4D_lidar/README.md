# Unitree L2 – stationary, drift-free, time-synchronised point cloud for obstacle detection

ROS 2 Humble workspace for a **fixed-mounted** Unitree L2. It produces a levelled,
vibration-compensated point cloud in a fixed world frame and detects/tracks objects in it.
It cannot drift: no IMU integration into a pose exists anywhere in the pipeline.

```
                       hardware-timed                       per-point time + IMU
 Unitree L2 ──UDP/serial──► l2_driver ──► l2/cloud (lidar frame) ──► l2_stabilizer ──► l2/points (world frame)
                                └───────► l2/imu  ──────────────────────┘     │              │
                                                                     ~/.ros/l2/mount.yaml    ▼
                                                                                        l2_obstacles ──► l2/obstacles          (l2_msgs/ObstacleArray)
                                                                     ~/.ros/l2/background.npz      ├──► l2/obstacles/markers  (RViz wireframe boxes)
                                                                                                    └──► l2/obstacles/points   (foreground points)
                                                    l2/points + background + boxes ──► l2_dense ──► l2/points_image (scene), l2/points_live (people), l2/depth_image
```

| package | language | what it does |
|---|---|---|
| `l2_driver` | C++ | L2 driver with hardware-timestamp synchronisation (replaces `unitree_lidar_ros2`) |
| `l2_perception` | Python (numpy only) | mount calibration, vibration compensation, background model, clustering, tracking, simulator, timing checker |
| `l2_msgs` | msg | `Obstacle`, `ObstacleArray` |

---

## How the three failure modes are eliminated

| Problem | Root cause (found in the SDK / stock driver) | Fix | Proven by |
|---|---|---|---|
| **1. Bursty IMU (1000 msgs at once, then 100 ms silence)** | The SDK has **no receive thread**; the stock node calls `runParse()` once per **1 ms ROS timer tick**, so packets back up in the socket and come out in bursts. With the default `use_system_timestamp=true` the SDK stamps each packet with *host time at parse* (`unitree_lidar_utilities.h`: `getSystemTimeStamp() - scan_period`), so the bursts are baked into the stamps. | Dedicated reader thread in a tight loop. **Hardware stamps** (`use_system_timestamp=false`). IMU sample instants are rebuilt from the packet sequence counter (`SequenceTimeline`), so even firmware that stamps at *send* time yields evenly spaced IMU stamps. Delivery jitter no longer reaches the stamps at all. | gtest `Device1970ClockWithBurstyDelivery`, `BunchedStampsAreRegularised`; `l2_sync_check` |
| **2. Timestamp mismatch / "1970 bug" / "loop back, clear buffer"** | The lidar clock is unrelated to the host (1970, future, drifting); RTC-less computers boot at 1970 and step later. | `ClockMapper`: the device clock is mapped onto the host clock as the **lower envelope** of (arrival − device stamp), on `CLOCK_MONOTONIC_RAW`, which never steps and is not slewed by NTP. A second-order tracker follows oscillator drift (measured, in ppm). `TimelineGuard` uses the sequence counter to separate device clock steps or lidar reboots (re-initialise) from single corrupt stamps (dropped) and from host stalls (ignored). The driver waits for NTP/PTP sync before publishing (`wait_for_clock_sync`). Downstream nodes reset cleanly on stamp jumps. `scripts/setup_time_sync.sh` installs chrony. | gtest `DeviceFutureClock`, `OscillatorDriftOneHour`, `DetectsDeviceClockStep`, `DetectsLidarRebootSeqRestart`, `HostStallIsNotAClockStep`, `Pipeline.WorstCaseCombination`; simulated +5 s host clock step |
| **3. Weak initialisation → wrong IMU bias → drift** | Point-LIO / FAST-LIO estimate a moving pose and need a feature-rich scene in the first second. A stationary lidar does not need any of that. | **No odometry.** The pose is a constant mount calibration (gravity + RANSAC ground plane, works for any orientation including upside-down), estimated **once** and saved to `~/.ros/l2/mount.yaml`. Restarts load the file, so start-up no longer depends on the scene. Optional vibration compensation (a **leaky, self-centring gyro integral**, hard-clamped) can never drift either; it is **off by default** because on the real L2 it made the cloud worse (see below). | `test_cannot_drift_with_bias_and_no_vibration` (1 simulated hour), `test_levels_any_mount`, `test_ground_reference_cancels_accelerometer_bias` |

### What the real L2 does (measured on this unit: firmware 2.8.8.1, hardware 2.2.1.1, serial)

| behaviour | consequence | handled by |
|---|---|---|
| Every hardware time (IMU stamps, point-packet stamps, per-point `time_increment`) runs at **0.502× real time**. Only `scan_period` is in real seconds. | Device stamps drift away from real time by 0.5 s per second. This is probably why Unitree's driver defaults to host stamps. | `DeviceRate` measures the rate from the data and rescales all device times; nothing is hard-coded. Diagnostics: `device_clock_speed`. |
| The packet counter is **10-bit** and wraps at 1024 (every ~4 s). | A naive guard sees a "reboot" at every wrap. | `TimelineGuard` learns the counter width (`seq_counter_modulus`). |
| The IMU delivers **250 Hz** (not 500). The accelerometer reads **2 g** at rest. | Only the accel direction is used, so levelling is unaffected. | (no action needed) |
| The gyro shows a strong **4.33 Hz** signal (0.3° equivalent) at the head rotation rate (`com_rotation_period` 230.9 ms) that is **not** motion of the scan. | Vibration compensation driven by it blurred surfaces: 3.06 mm → 3.28 mm median thickness on recorded data. | `vibration.enabled: false` by default. |
| Spin-up from standby takes **~20 s** before point data flows. | | `l2.sh` / `command:=start` wait up to 25 s. |
| `dirty_percentage` reported ~40 % on this unit. | Fewer points; the SDK says the cover is dirty. | Clean the dome and watch the diagnostics value. |

Measured end to end on this L2:
* `l2_sync_check` **PASS**. IMU 250.5 Hz, stamps 3.992 ms apart with max deviation 0.010 ms.
  Clouds 10 Hz with ~3,950 points and a 100 ms point-time span. 100 % of clouds are covered by
  IMU stamps. No timeline resets, no lost IMU packets.
* Median surface thickness of the accumulated stable cloud: 3.06 mm.
* `l2.sh stop` is verified to put the lidar in standby (no point data). `l2.sh spin-up` restarts it.

Two more stock-driver issues are handled:
* The L2 orientation quaternion is `w,x,y,z` (checked against the gravity vector), but the stock node
  publishes it as `x,y,z,w`. It also drifts in yaw, so `l2_driver` does not publish it by default
  (`orientation_covariance[0] = -1`).
* SDK 2.0.9 `closeUDP()` recurses forever in `UDPHandler::~UDPHandler()` and segfaults. The driver
  never calls it.

### Results in the simulator

The simulated setup: inverted mount at 3 m with 2° mounting error; 0.15°/0.10°/0.05° vibration at 7/12/3 Hz;
0.3°/s gyro bias; 0.1 m/s² accelerometer bias; IMU delivered in bursts with gaps up to 100 ms.

| metric | result |
|---|---|
| IMU stamp spacing with 96 % of messages arriving in bursts | 2.500 ms, max deviation 0.000 ms |
| clouds fully covered by IMU stamps | 100 % (79 of 79 inside the recording window) |
| timing pipeline, 1970 clock + 60 ppm drift + bursts + bunched stamps, 5 min | absolute error < 1 ms, spacing error < 0.1 ms |
| mount calibration (true roll 182°, pitch −3°, height 3.000 m) | −178.02° (≡ 181.98°), −3.01°, 3.0005 m |
| floor flatness beyond 7 m with *simulated* mount vibration, compensation off → on | 1.57 cm → 0.30 cm (the real L2's gyro cannot be used for this; see above) |
| detection | walking person tracked at the right speed, box located within 0.13 m, 0 false positives |

---

## Hardware

**Connection.** Ethernet is the factory default: the lidar is `192.168.1.62:6101` and the host must be
`192.168.1.2:6201`. Use a dedicated port, such as a second NIC or a USB-Ethernet adapter:

```bash
sudo scripts/setup_network.sh <interface>          # persistent profile + bigger UDP buffers
```

Serial (USB, Unitree adapter board) needs work mode 8, which is set once over Ethernet. Then use
`connection:=serial` (4 Mbaud). Set `serial_port` in `config/l2.yaml` to the adapter's stable name
from `ls -l /dev/serial/by-id/`, because `/dev/ttyACM*` numbers change with plug order. The lidar
needs its own 12 V supply; USB only powers the adapter's serial chip. The driver never writes the lidar's persistent work mode unless you set
`work_mode` explicitly.

**Mounting.**
* The L2 sees a hemisphere around its +z axis (wide mode adds about 6° below its base plane). To see
  obstacles on the floor, hang it upside-down or tilt it. Any orientation works; the calibration
  handles it.
* Use a stiff bracket. Vibration compensation is off by default, because the L2's own gyro signal is
  dominated by its head rotation. A rigid mount is what keeps the cloud sharp. `mount_monitor` warns
  if the bracket tilts by more than 0.5° after calibration; recalibrate then.
* If the floor is not visible (for example the lidar sits upright on a desk), set
  `calibration.mount_height`. Otherwise world z = 0 is at the lidar, and `ground.z_max` in
  `l2_obstacles` will not mean "above the floor".
* Keep the floor visible during the 3 s calibration. That is what makes the levelling immune to
  accelerometer bias.

**Clock.** On computers without an RTC battery (Jetson), run `sudo scripts/setup_time_sync.sh [ntp-server]`.

---

## Build

```bash
cd "/home/mr/workspaces/4D lidar"
source /opt/ros/humble/setup.bash
colcon build
source install/setup.bash
```
The Unitree SDK (x86_64 and aarch64 static libraries, BSD-3) is vendored in
`src/l2_driver/third_party/unilidar_sdk2`. No internet connection is needed on the target.

## Run

Use `scripts/l2.sh`. Starting spins the lidar up and runs everything. Stopping it always spins
the lidar down: Ctrl+C, `l2.sh stop`, or a systemd stop.

```bash
scripts/l2.sh start                  # spin up + driver + stabilizer + obstacles + dense + RViz
scripts/l2.sh start rviz:=false      # any launch argument can be appended
scripts/l2.sh start sim:=true        # no hardware: simulator
scripts/l2.sh stop                   # from another terminal: stop everything + spin down
scripts/l2.sh status                 # pipeline running? lidar spinning?
scripts/l2.sh spin-up | spin-down    # motor only, while the pipeline is not running
```

The script sources ROS itself. The connection (`serial` or `udp`) and the port come from
`config/l2.yaml`. After editing that file, run `colcon build`, because the installed copy is the
one that's used. Spin-down is done twice: the driver sends "stop rotation" when it exits, then
the script sends it again and checks that point data has stopped. So the lidar also stops if the
driver crashed. `ros2 launch l2_perception l2.launch.py ...` works too, but only the driver's own
spin-down applies then.

**First run on a new installation:**
1. **Mount calibration (3 s).** Keep the scene still and the floor visible. The result goes to
   `~/.ros/l2/mount.yaml`. You may edit `yaw_deg` (world heading), `x` and `y` there.
2. **Background learning (30 s).** Keep moving objects out of view. The result goes to
   `~/.ros/l2/background.npz`. Things present the whole time become "scene"; anything appearing later
   is reported.

Later starts load both files and detect immediately. Re-learn after changing the scene:

```bash
ros2 service call /l2_obstacles/learn_background std_srvs/srv/Trigger
ros2 service call /l2_stabilizer/calibrate std_srvs/srv/Trigger   # after moving the lidar
```
(If the mount calibration changes, `l2_obstacles` notices and re-learns automatically.)

Start at boot, after the clock is synchronised: `sudo scripts/install_service.sh`. It runs
`l2.sh start rviz:=false`, and `sudo systemctl stop l2-perception` spins the lidar down.

### Viewing the 3D structure in RViz

The L2 is precise (surface noise ~3 mm measured on this unit) but sparse: one 0.1 s cloud
has ~4000 points. Its scan pattern also repeats: over UART it scans fixed elevation lines about
0.6° apart, so raw points show stripes and rings however long you collect them. Because the
lidar does not move, `l2_dense` builds the scene the way a depth camera sees it:

* **Scene image (`l2/points_image`)**: one point per pixel of a 0.2° grid around the lidar
  (3.5 mm apart at 1 m, 1 cm at 3 m), on black, coloured by height (low = red, then orange and
  yellow, high = green). Each pixel shows the nearest surface seen there, averaged over all its
  hits, so the range noise disappears. One-off hits (dust, mixed points at edges) are dropped.
  The rows between two scan lines are interpolated only where both lines lie on the same
  surface, so walls and objects become solid, gap-free surfaces with sharp outlines, while depth
  edges are never bridged. Each message is the whole picture (1 Hz, Decay Time 0). It fills in
  over the first ~30–60 s after start. Moving things never enter it.
  Finer grid: `l2_dense.image.resolution_deg` (smaller leaves more gaps); bigger dots: Size 3.
  `l2_dense.color_by: range` colours by distance instead (near red, far blue).
* **Scene raw points (`l2/points_scene`, off by default)**: the same static points as measured,
  kept by RViz for 60 s (Decay Time). Tick it in the Displays panel to compare.
* **People / moving / new objects (`l2/points_live`)**, white 3-pixel points: everything that is
  not part of the learned scene. Something standing still builds up to `live.hold` = 3 s of
  detail. When it moves, its old points vanish as soon as the tracked box has moved on, or the
  lidar has looked at that spot again and found it empty. In simulation a person walking at
  1 m/s 4–6 m away leaves a trail of about 0.2–0.3 m.
  Anything placed in the scene after learning stays here (white) until you re-learn the
  background: `ros2 service call /l2_obstacles/learn_background std_srvs/srv/Trigger` (area empty).
* **Depth image (`l2/depth_image`)** is a 360° panorama around the lidar (2048 px, 0.18°/px),
  red = near, blue = far. Its rows are cropped to the elevations the lidar sees
  (`auto_elevation`). The same data in metres is on `l2/depth_image/raw` (32FC1, NaN = no return).
* **Obstacle boxes** are wireframes, so they do not hide the shape inside. Set
  `l2_obstacles.markers.labels: true` for "#id range speed" text.
* Off by default, tick them in the Displays panel: the grid, the raw stabilised cloud with 5 s
  decay (`l2/points`) and the detector's foreground points (`l2/obstacles/points`).

## Verify timing on your hardware

```bash
ros2 run l2_perception l2_sync_check                                         # this driver
ros2 run l2_perception l2_sync_check --imu unilidar/imu --cloud unilidar/cloud   # stock driver, for comparison
ros2 topic echo /diagnostics          # or: rqt_runtime_monitor
```
`l2_sync_check` prints PASS/FAIL for:
* IMU stamp spacing vs. arrival bursts
* stamps vs. host wall time (catches 1970 or future clocks)
* IMU coverage of every cloud's point times
* vibration amplitude, which tells you whether compensation matters for your mount

The driver's diagnostics show:
* `published_dt_min/max_ms` (should be the IMU period)
* `arrival_max_gap_ms` (the raw delivery bursts it absorbed)
* `clock_drift_ppm`, `lost_total`, `timeline_resets_total`, `host_clock_synchronised`
* lidar temperatures and `dirty_percentage` (dirt on the cover)

---

## Interfaces

| topic | type | frame | notes |
|---|---|---|---|
| `l2/cloud` | PointCloud2 | `l2_lidar` | fields `x y z intensity time ring`; `time` [s] relative to `header.stamp` (first point), 10 Hz |
| `l2/imu` | Imu | `l2_imu` | 400 Hz, hardware-timed; gyro rad/s, accel m/s²; no orientation |
| `l2/points` | PointCloud2 | `world` | levelled, vibration-compensated, range-filtered |
| `l2/obstacles` | l2_msgs/ObstacleArray | `world` | id, confirmed, age, position, velocity, oriented box, closest point, range, num_points |
| `l2/obstacles/markers` | MarkerArray | `world` | RViz |
| `l2/obstacles/points` | PointCloud2 | `world` | foreground points used for detection |
| `l2/points_image` | PointCloud2 | `world` | depth-camera-style static scene, one point per 0.2° pixel, whole frame; fields `x y z range rgb`, 1 Hz (built only while subscribed) |
| `l2/points_scene` | PointCloud2 | `world` | raw static-scene points of the last 0.5 s (accumulate them, e.g. RViz Decay Time); fields `x y z intensity range rgb`, 2 Hz |
| `l2/points_live` | PointCloud2 | `world` | people / moving / new objects (not in the background); `x y z intensity range`, 10 Hz |
| `l2/points_dense` | PointCloud2 | `world` | persistent scene map, one point per voxel, 5 mm near … 8 cm far (built only while subscribed) |
| `l2/depth_image` | Image rgb8 | `world` | 360° panoramic depth, red = near, 2 Hz (computed only while subscribed) |
| `l2/depth_image/raw` | Image 32FC1 | `world` | same in metres, NaN = no return |
| `/diagnostics` | DiagnosticArray | | all nodes |

TF (static): `world → l2_lidar` (mount calibration), `l2_lidar → l2_imu` (from the Unitree datasheet).
The world frame has its origin on the floor below the lidar, +z up, and +x is the lidar's x axis
projected onto the floor (rotate it with `yaw_deg`).

## Tuning (`src/l2_perception/config/l2.yaml`)

| parameter | default | when to change |
|---|---|---|
| `l2_obstacles.mode` | `background` | use `geometric` to report every object above the floor, including static ones |
| `l2_obstacles.accumulate_time` | 0.3 s | longer gives denser clouds and better far-range detection, but smears fast movers |
| `l2_obstacles.roi.min/max` | ±30 m, z −1…3.5 | the area you monitor (world frame) |
| `l2_obstacles.ground.z_max` | 0.10 m | lowest obstacle height |
| `l2_obstacles.cluster.min_points` / `voxel_size` | 8 / 0.2 m | sensitivity vs. noise; voxel ≈ the largest gap inside one object |
| `l2_obstacles.background.min_span` | 0.5 | fraction of the learning period a voxel must be seen in to count as scene |
| `l2_obstacles.tracker.min_hits` | 3 | windows needed before an obstacle is reported (flicker suppression) |
| `l2_dense.image.resolution_deg` | 0.2° | pixel size of the scene image: smaller = finer but more gaps, larger = smoother |
| `l2_dense.image.max_gap` | 3 px | widest gap between scan lines that is bridged on a surface |
| `l2_dense.color_by` | `height` | `range` colours by distance from the lidar instead |
| `l2_dense.color_z_min` / `color_z_max` | auto | heights drawn red / green (both 0 = 1st..99th percentile of the scene) |
| `l2_dense.live.hold` | 3 s | how much detail builds up on a person or object standing still |
| `l2_dense.live.box_gate` | true | trail removal using the tracked boxes; without it the trail is ~1 s at 4–5 m range |
| `l2_obstacles.markers.labels` | false | text labels above obstacle boxes |
| `l2_stabilizer.vibration.enabled` | false | only for a genuinely vibrating mount, after checking with recorded data that it sharpens surfaces |
| `l2_stabilizer.calibration.level_reference` | `ground` | `gravity` if no floor is visible (set `mount_height`) |
| `l2_driver.cloud_period` | 0.1 s | cloud rate |
| `l2_driver.transport_latency` | −1 (SDK) | fixed value if you align with other sensors precisely |
| `l2_driver.realtime_priority` | 0 | e.g. 40 under heavy CPU load (systemd unit allows it) |

## Troubleshooting

* **No data:** `sudo tcpdump -i <iface> -c 5 udp port 6201`. If nothing arrives, check the cable,
  IP address and power. If packets arrive but the driver gets nothing, another process is bound to
  port 6201 (for example the stock driver).
* **"Waiting … for the host clock to be synchronised":** there is no NTP; run `setup_time_sync.sh`,
  or set `wait_for_clock_sync: 0` on an isolated system.
* **"mount tilted … since calibration":** the bracket moved. Fix it and call `/l2_stabilizer/calibrate`.
* **False detections at long range:** increase `accumulate_time`, `cluster.min_points` or
  `background.learn_time`, or shrink the ROI.
* **Static object you placed is reported forever:** that is intended in `background` mode. Re-learn
  the background to accept it as part of the scene.

## Tests

```bash
colcon test && colcon test-result --verbose      # 19 C++ timing tests + 53 Python tests
```
To check without hardware, run `scripts/l2.sh start sim:=true` together with
`ros2 run l2_perception l2_sync_check`.

## Layout

```
src/l2_driver/      include/l2_driver/{clock_mapper,timeline_guard,sequence_timeline,device_rate}.hpp
                    src/l2_driver_node.cpp, test/test_timing.cpp, third_party/unilidar_sdk2/
src/l2_perception/  l2_perception/{stabilizer_node,obstacle_node,dense_node,sim_node,sync_check}.py
                    l2_perception/{vibration,mount,background,detection,voxel,scene_map,geometry,pointcloud,stamp_guard}.py
                    config/l2.yaml, launch/l2.launch.py, rviz/l2.rviz, test/
src/l2_msgs/        msg/Obstacle.msg, msg/ObstacleArray.msg
scripts/            l2.sh (start/stop/status), setup_network.sh, setup_time_sync.sh, install_service.sh
```
