# 7DOF-OArm IK Environment

Runs the 7DOF-OArm bimanual exoskeleton with GPU-accelerated cuMotion kinematics,
MoveIt 2, and real hardware interfaces.

There are two ways to run it. **Use the native one** — see
[Why native](#why-native-and-not-the-container) below.

| Document | |
| --- | --- |
| **README.md** (this file) | running the robot, the VLM pick-and-place, tuning, design rationale |
| [INSTALL.md](INSTALL.md) | setting up a machine: requirements, the Python-stack split, GPU/CUDA, portability |
| [native/README.md](native/README.md) | how the native environment is assembled and exactly what it touches |

---

## 🚀 Quick Start (native — recommended)

One-time setup, if this machine has not been set up yet:

```bash
./install.sh
```

That builds both project environments and verifies they are isolated from the
system's Python — see [INSTALL.md](INSTALL.md) for requirements and what it does
and does not touch. Then, the one step that needs root:

```bash
sudo mkdir -p /workspaces && sudo ln -s /home/mr/workspaces/isaac_ros-dev /workspaces/isaac_ros-dev
```

### 1. Bring up the CAN interfaces

```bash
sudo ip link set can0 down && sudo ip link set can0 type can bitrate 1000000 dbitrate 5000000 fd on && sudo ip link set can0 up
sudo ip link set can1 down && sudo ip link set can1 type can bitrate 1000000 dbitrate 5000000 fd on && sudo ip link set can1 up
```

Check they came up (`ERROR-ACTIVE` is the normal healthy state):

```bash
ip -details link show can0
```

### 2. Launch the robot and planners

From a terminal **on the machine's own desktop session** — RViz and the MoveIt
depth-image octomap updater both need a real `DISPLAY`, so plain ssh will not
work:

```bash
cd /home/mr/workspaces/isaac_ros-dev && source native/setup.bash
```

```bash
native/run_launch_everything.sh
```

That is the whole thing — no container, no separate shell. It uses the same
launch arguments as before (`octomap:=static`, `use_fake_hardware:=false`,
`right_can_interface:=can0`, `left_can_interface:=can1`). Override any of them by
appending, e.g.:

```bash
native/run_launch_everything.sh use_fake_hardware:=true octomap:=live
```

You should see cuMotion warm up in about 3 seconds and print
`cuMotion is ready for planning queries!`, and `move_group` report
`MoveGroup context using planning plugin isaac_ros_cumotion_moveit/CumotionPlanner`.

### 3. Or start everything, including the pick-and-place panel

```bash
native/run_pick_place_demo.sh
```

Same robot bringup, plus the VLM detector, the pick-and-place orchestrator and a
panel to type into — see [VLM-guided pick and place](#-vlm-guided-pick-and-place).
Use `run_launch_everything.sh` above when you only want the robot.

Add `--fake` to rehearse with no arms and no CAN bus: the motion is simulated
and watched in RViz, the camera and the detector are real. See
[Dry-running on fake hardware](#dry-running-on-fake-hardware).

### 4. Rebuild after editing code

```bash
native/build_ws.sh
```

Or a single package:

```bash
native/build_ws.sh --packages-select openarm_description
```

### 5. Editing in VS Code

```bash
code ~/workspaces/isaac_ros-dev
```

---

## 📦 Installation

```bash
./install.sh          # everything; --native-only / --vlm-only / --check
./check_isolation.sh  # verify the venvs never overlap the system's Python
```

Requires Ubuntu 22.04, `ros-humble-desktop`, Python 3.10 and CUDA ≥ 12.8 already
on the host — it verifies the system, it does not provision it.

**Full details in [INSTALL.md](INSTALL.md)**: requirements, the three-Python-stack
split and how it is enforced, GPU/CUDA detection, what needs root, portability
limits, and install troubleshooting.

---

## 🤖 VLM-guided pick and place

A PaliGemma detector finds the object, cuMotion plans to it, and the same
detector checks whether each step actually worked — a failed grasp is retried
with an escalating strategy rather than repeated.

```
HOME ─► LOCATE ─► PREFLIGHT ─► PRE_PICK ─► TRANSIT ─► OPEN ─► DESCEND ─► CLOSE
  ▲                                                                       │
  └────── PRE_PICK ◄── CLEAR ◄── failed ──────────── VERIFY_GRASP ◄── LIFT
                                                           │
  HOME ◄── PRE_PICK ◄── VERIFY_PLACE ◄── RELEASE ◄── DROP ◄─┘
```

**HOME is the only pose the arm rests, observes and maps from.** There used to
be a separate READY as well, and having two was the whole problem: the octomap
must be captured with the arm out of the camera's frame, and the old READY
deliberately held it out over the table — *in* frame. One pose does both jobs.

The way out is the way in: LIFT, DROP, then back through PRE_PICK to HOME.
PRE_PICK is reachable from both ends, which is what makes it a safe waypoint
rather than a dash home across the workspace.

**A pick that fails leaves by the same door.** Every failure below TRANSIT —
the descent stopping short, the jaws closing on nothing, the grasp not
verifying, the cycle crashing — leaves the arm extended down at the object,
and PRE_PICK and HOME are *joint* goals: the short path in joint space from
reaching over the table to folded at the side goes through the table.
Measured, left arm, run `1788859355`: DESCEND fine, the grip closed on
nothing at 1.912 Nm, and the very next record is HOME as a joint goal from
tcp `z=0.3715`. It dragged across the surface on the way.

So the first thing any exit does is `CLEAR` — straight back up the descent
line, gripper left exactly as it is — and only then PRE_PICK and HOME.
`clear_the_surface()` is a no-op once the tool is above the pre-grasp height,
which on the successful path it already is.

Three named postures, all replayed as joint goals:

| | Where it comes from | |
| --- | --- | --- |
| **HOME** | `home_state` in the states file, else `home_joint_positions` | rest and observation pose. The object is located from here and the map is captured here, so the arm must be clear of the camera's view of the table |
| **PRE_PICK** | `pre_pick_state` in the states file | staging pose, entered on the way to the object and again on the way back |
| **DROP** | `drop_state` in the states file | where the object is released |

A recording that predates the merge calls HOME `ready_state`. That is **not**
reused as HOME — it was an observation pose held out over the table, and
replaying it would reintroduce the arm-in-the-map problem. The orchestrator says
so and falls back to `home_joint_positions`; record a proper one with
`python3 record_states.py --arm <arm> home_state`.

**PREGRASP** is 5 cm (`approach_height`) above the object's detected top
surface: the arm stops there, opens the gripper, descends onto the object,
closes, and lifts back to the same 5 cm.

### Why there is a TRANSIT above the object

**Nothing here plans a straight line.** cuMotion is handed a goal *pose* and
optimises a smooth trajectory to it. So a single move from the staging pose to a
point 5 cm above the object is free to arrive from the side and low — which is
how a gripper sweeps a screwdriver off the table on its way to sitting above it.

So the approach is split. **TRANSIT** ends the long free-space move
`transit_height` (default **20 cm**) above the grasp, well clear of anything on
the surface. Everything below that height travels down the *vertical line above
the object* in hops of at most `descend_step` (default **5 cm**): the waypoints
share x and y and differ only in z, so a short hop cannot bow far off that line.
LIFT comes back up the same line, because a retreat that bows drags whatever is
now in the gripper across the table it came off.

```
        TRANSIT      z = grasp + 0.20   ← the approach ends here,
           │                              and the gripper opens here
           │
           │  ONE straight line, all the way down
           │
        DESCEND      z = grasp          ← gripper closes to the torque cap
           │
           │  one straight line, back up
           │
        LIFT         z = grasp + 0.20   ← clear of the surface to carry
```

Set `transit_height:=0` to go straight to the pre-grasp in one move, which is
the old behaviour. Raising `descend_step` above the drop it has to cover
collapses that leg back into a single free plan.

#### The descent is a real straight line

**A goal pose says where to end up, not how to get there.** cuMotion optimises
a smooth trajectory to it, and over a 5 cm descent that can bow well away from
the vertical — observed here as the gripper taking a detour into the table
after opening above the object.

So the legs below `transit_height` ask `/compute_cartesian_path` for a genuine
Cartesian line. move_group interpolates it and runs IK at every
`cartesian_step` (5 mm), so the tool travels straight down by construction, and
the path is collision-checked. Three things follow:

- The service reports the **fraction** it managed. Anything under
  `cartesian_min_fraction` (0.98) is **refused, not executed** — a descent that
  stops at 60% leaves the gripper closing on air.
- It applies no speed scaling, so the returned trajectory is re-timed to
  `velocity_scaling` here. Without that the careful descent runs at full joint
  speed.
- Posture continuity comes free: IK at each interpolation step is seeded from
  the previous solution, so the arm cannot flip branch mid-descent.

`linear_descent:=false` reverts to the old behaviour.

**The final approach is exempt from collision checking**, and this is the part
that makes the descent actually work. On a top-down grasp **the target is
itself in the collision world**: the octomap holds the object being picked up
and the table under it, so the gripper is required to enter mapped voxels to
reach the thing it is grasping. A collision-checked descent onto an object can
therefore never complete. Measured on this robot:

```
PREGRASP: straight line to (0.348, -0.229, 0.411), 15 points     ← 20 cm, 100%
DESCEND:  only 25% of the straight line was solvable (need 98%)  ← last 5 cm
DESCEND:  no straight line available, stepping instead
DESCEND 1/3 ... 2/3 ... 3/3                                      ← free-space, curved
```

The arm could reach every point — the fallback goals got all the way down. It
was the checking that stopped the line about a centimetre in, right where the
gripper met the object's own voxels.

So `DESCEND` and `LIFT` now retry the same straight line with
`avoid_collisions` off (`approach_ignores_octomap`, default true) *before*
falling back to free-space goals. What keeps that safe is that the leg is
short, straight, vertical, between two points whose reach was already checked,
with the gripper open and `min_grasp_z` as a hard floor — and the alternative is
demonstrably worse: a free-space plan for the same 5 cm is what drove the
gripper into the table.

**All three legs get the exemption, not only the last 5 cm.** Restricting it to
the final descent was not enough — the 15 cm transit-to-pre-grasp leg then
stalled at 61% checked and went as eight curved free-space hops instead:

```
PREGRASP: only 61% of the straight line was solvable (need 98%, collision-checked)
PREGRASP: falling back to pose goals, so the planner may change posture on the way
PREGRASP 1/8 ... 8/8                                    ← the curve
DESCEND:  straight line to (0.385, -0.300, 0.359), 14 points   ← the fixed leg
```

Same cause: the gripper's own geometry meets the mapped object and table as it
comes down the column, whatever height the leg starts from. The whole vertical
column below `transit_height` is now exempt; the free-space move *up to*
`transit_height` keeps its checking, because there the obstacles are real.

### The retreat has to clear the table

`LIFT` used to return only to the pre-grasp, 5 cm above the object. The next
move carries the object to the drop pose as a **free-space plan**, and starting
that 5 cm off the surface dragged the gripper across the table.

`retreat_height` (default **0.20**, the same as `transit_height`) is how high
the object is lifted before it is carried anywhere, so it leaves at the altitude
the approach arrived at. `retreat_height:=0` restores the old
back-to-the-pre-grasp behaviour.

#### A curved descent is refused, not substituted

The fallbacks below are **not** used for `DESCEND` or `LIFT`
(`descend_linear_only`, default true). Free-space hops are not a milder version
of the same motion, and the motion log showed why:

```
DESCEND  cartesian  short   fraction=0.375 checked=True    ← 1.9 cm of 5 cm
DESCEND  cartesian  short   fraction=0.375 checked=False   ← same unchecked
DESCEND 1/3  pose  ok   tcp[+0.332 -0.166 +0.381]
DESCEND 2/3  pose  ok   tcp[+0.369 -0.186 +0.369]   ← +3.7 cm in x, −2 cm in y
DESCEND 3/3  pose  failed  code=-4                  ← CONTROL_FAILED
```

An identical fraction with checking **on and off** is the tell: nothing is in
the way, the arm simply **runs out of reach** along the line. The hops then
swung the tool 3.7 cm sideways while descending 1.2 cm and aborted against the
table.

So the descent refuses, the attempt fails, and the message says what a matching
checked/unchecked fraction means — bring the object closer. That is also the
signature to look for in the log: `fraction` short and equal in both records.

#### When no straight line is available (other legs)

Two fallbacks, in order — a pick beats no pick:

1. **Joint goals from seeded IK** at intermediate heights (below).
2. **Pose goals**, the original behaviour, which can wander.

Note `descend_step` is **0.02**, not 0.05. At 0.05 it was never subdivided at
all: the descent from the pre-grasp *is* `approach_height` = 0.05, so
`span <= step` held and the whole leg went as one free-space goal. That is the
bug that put the gripper into the table, and the straight line above is the
real fix; this is the safety net behind it.

#### The fallback column goes as joint goals, from seeded IK

A vertical line of *pose* goals is still not enough, and this is the part that
turned the arm inside out to lower the tool 5 cm.

The arm has **seven joints for a six-DOF pose**, so a tool pose does not pick a
posture — there is a whole null space of solutions, and elbow-up and
elbow-flipped are both correct answers to "the same pose, 5 cm lower". Each pose
goal is solved independently, so the planner is free to choose a different
branch for the next waypoint, and it does.

So the column is solved before it is flown:

1. Ask `/compute_ik` for each waypoint, **seeded with the joints of the waypoint
   above it** — KDL searches from the seed, so the answer stays in the branch
   the arm is already in. The first waypoint is seeded from the arm's measured
   position.
2. Reject any solution that moves a joint more than `max_joint_jump` (default
   **0.5 rad**) from its seed. That is a reconfiguration, not a descent.
3. Send **joint** goals, which cuMotion plans with `plan_single_js`. The tool
   still tracks the vertical line, because the waypoints are collinear and
   close together — but the posture is now pinned.

It is all-or-nothing per column: a column that were half joint goals and half
pose goals could still flip at the seam. If IK cannot supply the whole column —
no `/compute_ik`, no solution, or only a flip — it logs that and falls back to
pose goals, which can flip but at least still picks. `seeded_descent:=false`
forces the old behaviour.

The cost is more goals — and so more chances to hit cuMotion's 14.5%
`TRAJOPT_FAIL` rate, which is why `plan_attempts` exists.

### Everything in one command

```bash
native/run_pick_place_demo.sh
```

That starts the robot, MoveIt, cuMotion, the camera, the detector, the
orchestrator and a small panel you type into — it sets cuMotion's `tool_frame`
to match the arm itself, so the two cannot disagree. Type what to pick and press
**Pick**:

```
┌─ OpenArm pick and place ──────────────────┐
│ What should the arm pick up?              │
│ ┌───────────────────────────────────────┐ │
│ │ pick up the screwdriver               │ │
│ └───────────────────────────────────────┘ │
│ detector prompt:  detect screwdriver      │
│ Grip torque cap [ 2.0 ] Nm                │
│  [ Pick ]  [ Abort ]                      │
│ Gripper only:  [ Open ]  [ Grip ]         │
│ PRE_PICK                                  │
└───────────────────────────────────────────┘
```

The panel strips the conversational wrapper before the prompt reaches the model —
`paligemma-3b-pt-224` is a pretrained checkpoint, so `detect screwdriver` is a
well-formed prompt for it and `pick up the screwdriver` is not. It shows the
translation live so there is no guessing which of the two the model saw.

PaliGemma takes tens of seconds and several GB of VRAM to load; it loads
alongside robot bringup, and until it is ready a pick sits in LOCATE. That load
is why the launch files are otherwise kept separate — when the robot is already
up, use `pick_place.launch.py` on its own instead.

**Before the first run**, record the two poses the cycle needs (below). Without
them a pick refuses to start rather than moving somewhere nobody chose.

### Clearance: if it clips things it planned around

`collision_activation_distance` is the distance cuMotion keeps from everything
in the world, and it is the knob to reach for when the arm brushes obstacles the
plan was supposed to avoid. cuRobo's own default is 0.01 m; this workspace now
launches with **0.03 m**:

```bash
native/run_pick_place_demo.sh collision_activation_distance:=0.05
```

It inflates *every* obstacle, octomap voxels included, so it trades clearance
against reachability — push it too far and tight approaches stop planning at
all. The planner prints the value it is using at startup.

### The octomap is captured at the ready pose, and nowhere else

With `refresh_octomap_at_ready` (the default), the arm asks
`/octomap_gater/refresh` for frames each time it arrives at READY, and never
anywhere else. That is deliberate in both directions:

- A depth frame taken with the arm **out over the table** bakes the arm and the
  target object into the map, and the very next plan then has to avoid the thing
  it is trying to reach.
- A frame taken **while carrying** captures the payload as a permanent obstacle
  that travels with the tool. So the refresh is skipped whenever the gripper is
  holding something — the map you plan the carry against is the one taken before
  the pick, with the object still on the table.

At READY the arm is folded up out of the camera's view, so what the camera sees
is the workspace. `octomap_settle_time` (1.5 s) is the pause that lets the
updater integrate those frames before the next plan goes out; planning against a
half-built map is worse than planning against the previous one.

**"At READY" means measured, not assumed.** The check compares live
`/joint_states` against `ready_joint_positions` within `ready_pose_tolerance`
(0.05 rad) and refuses the capture otherwise — because a move goal can come back
SUCCESS without the arm having actually got there, and a map captured on that
assumption has the arm in it. Every path that could refresh the map goes through
that one guard, so there is no route to a capture from anywhere else.

The return trip after a grasp is planned with the payload attached
(`attach_object()` runs before the move), so the carry is collision-checked
against the object on the tool rather than against an invisible one.

### Grip torque cap

`gripper_torque_cap` is a **torque at the gripper motor, in Nm** — the same
units as `DEFAULT_GRIPPER_TORQUE_CAP_NM` in the exoskeleton bridge, defaulting
to **2.5**. Over the 42.0 mm/rad transmission that is about 59.5 N at the
finger, held by 5.25 mm of finger overshoot. The panel's **Grip torque cap** box sets it at runtime;
`close_gripper_to_cap` re-reads it on every close, so a new value applies to the
next grasp with no restart.

#### Testing the cap by hand

Worth doing before trusting the cap on a real pick, and it needs no arm motion:
the panel's **Gripper only** row drives the gripper on its own.

1. **Open** — the fingers go to `gripper_open`, 0.044 m, which is
   `finger_joint1`'s upper limit in `openarm_hand.xacro`. That is as wide as
   they mechanically go; there is no more travel to ask for.
2. Put the object between the fingers by hand.
3. **Grip** — applies whatever is in the torque box, then closes to it. The log
   line reports where the fingers stopped and the torque they stopped at, which
   is the number to compare against the cap.

Same thing without the panel:

```bash
ros2 service call /pick_place/open_gripper std_srvs/srv/Trigger
ros2 service call /pick_place/grip std_srvs/srv/Trigger
```

Both refuse while a cycle is running, and both are refused by the panel's
greyed-out buttons for the same reason — opening the fingers mid-descent would
drop whatever is in them. **Abort** works during a **Grip**: the stepped close
checks the abort flag between steps.

Nothing in ros2_control can enforce this, which is why it is enforced here.
`gripper_action_controller`'s position adapter writes *only* position to the
hardware and returns `max_effort` for its own stall bookkeeping
(`hardware_interface_adapter.hpp`, `updateCommand`), so the value never reaches a
command interface. The gripper is a position command whose closing force is
position error times the hardware's fixed `GRIPPER_DEFAULT_KP` (20 Nm/rad).

So the close is **stepped**, and it stops advancing when measured torque reaches
the cap — the same enforcement the bridge does at 100 Hz:

- coarse steps (`gripper_close_step`, 2 mm) until torque appears, then quarter
  steps near the cap, so a single step cannot overshoot it far. One 2 mm step is
  0.95 Nm at this gain, which is why the fine phase exists.
- on reaching the cap it **backs off to the last command that was under it**.
  It must not command the fingers' *measured* position as a "hold": that would
  zero the position error and therefore the grip force, and the object would
  drop. What holds the object is precisely that last under-cap command.

**This needs a rebuild.** The cap is unenforceable without torque feedback, and
`v10_simple_hardware.cpp` used to hardcode it:

```cpp
tau_states_[ARM_DOF] = 0;  // gripper_motors[0].get_torque();
```

That now reads the motor, like the arm joints already did. It is a read-only
change — it populates the effort state interface that was already exported for
the joint, and commands nothing — but it is C++:

```bash
native/build_ws.sh --packages-select openarm_hardware
```

Until that lands, `close_gripper_to_cap` finds no torque on `/joint_states`,
says so, and closes uncapped rather than silently pretending to limit anything.

### Both arms

Poses live in one file per arm — `pick_place_states_left.yaml` and
`pick_place_states_right.yaml` — because the arms are mirrored, so a pose
recorded on one is a different posture on the other. `states_file` defaults to
`auto`, which resolves to the file for whichever arm is running.

```bash
python3 record_states.py --arm left        # writes pick_place_states_left.yaml
python3 record_states.py --arm right       # writes pick_place_states_right.yaml
python3 record_states.py --arm left --list
```

The walkthrough records three poses: `ready_state`, `pre_pick_state` and
`drop_state`. `ready_state` is optional for a single-arm setup — it falls back to
the `ready_joint_positions` parameter — but required by `arm_selection:=by_side`,
where one parameter cannot describe both mirrored arms.

`load_states()` refuses a file recorded for the other arm rather than replaying
it as a mirrored posture, which it is not.

To run the left arm, launch for it — that sets cuMotion's `tool_frame` to the
left hand at the same time:

```bash
native/run_pick_place_demo.sh arm:=left
```

#### Switching arms automatically

**`by_side` is the default.** With `arm_selection:=fixed` the camera-half rule
never runs and the launch arm moves whatever half the object is in, which is
not what anyone means by a two-armed robot; `fixed` is still there for
single-arm work.

The arm is chosen **before anything moves**, on the detection taken when the
prompt arrives:

1. The half of the frame the object is in gives the preferred arm — left half →
   left arm, right half → right arm, split at the middle column
   (`arm_split_px` nudges it).
2. `/compute_ik` is asked whether that arm can reach the grasp, the pre-grasp
   and the transit height. If it can, it gets the job.
3. If it cannot, the other arm is asked the same question and takes over.
4. If neither can, the cycle stops with `OUT_OF_REACH` having **sent no motion
   goals at all**.

`arm_order` sets the order: `camera_half` (default) prefers the near arm and
falls back to the other; `right_then_left` and `left_then_right` are fixed.
The orders only disagree when *both* arms can reach, and then the near arm is
the right answer — a cross-body reach is slower, closer to the joint limits,
and more likely to clip the other arm.

Reachability is a kinematics question, so asking it costs a service call rather
than a trajectory. Driving an arm to a staging pose and only *then* discovering
the object is outside its envelope wastes the motion and reports the wrong
cause.

The camera needs a clear view to detect from, so if the arm is at neither HOME
nor READY it goes to READY first — one move, not the whole approach.

That is checkable by looking — the dot on `/vlm/debug_image`, or
`vlm_detect.py`, tells you which half the object was in and therefore which arm
should move. The detector publishes `image_size` alongside the detections so
the orchestrator can make that call; without it (an older detector) it falls
back to the object's world y and `arm_split_y`.

The two criteria agree anyway, which is worth knowing: the camera is pitched
about world Y with **no yaw**, so optical +x — image right — maps to world −y,
and world −y is the right arm's side.

The decision runs once from the ready pose, before the retry ladder, and
everything arm-dependent is re-pointed at the chosen arm: the **planning group**,
the tool and hand frames, the seven joints, the attached-object touch links, the
gripper action, and the states file. Switching mid-ladder is deliberately not
done — the attempts would not be comparable.

`ready_joint_positions` is one list measured on one arm, and the arms are
mirrored, so driving the other arm to it is a different posture entirely. So
the **other** arm needs its own recorded `ready_state` — the one named by
`ready_joint_positions_arm` (default `right`) is covered by the parameter.
Requiring it from both was over-strict and refused to start on a setup that
was already complete.

**One caveat that is not ours to fix.** The installed
`libisaac_ros_cumotion_moveit.so` is version 3.2.5 and advertises a single,
group-independent action (`cumotion/move_group`), so both planning groups funnel
through one cuMotion node — and that node has exactly one `ee_link`, fixed at
construction from `tool_frame`. Switching the planning group switches everything
on this side, but a *Cartesian* goal for the arm cuMotion was not launched for
still comes back `INVALID_LINK_NAME`; the preflight catches it and says which
arm to relaunch for. Joint-space goals (READY, PRE_PICK, DROP) are unaffected and
work for either arm.

The source tree already carries `isaac_ros_cumotion_moveit` **4.4.0**, whose
client uses per-group action names (`cumotion_<group>/move_group`, see
`src/isaac_ros_cumotion/isaac_ros_cumotion_moveit/src/cumotion_move_group_client.cpp`).
Building that instead of the prebuilt 3.2.5 allows one cuMotion node per arm,
each with its own `tool_frame`, which is what makes the Cartesian half work too.
Costs to weigh first: two nodes means double the GPU memory and two cuRobo
warmups, the Python planner here is the 3.2.5-era override and would need its
action name changed to match, and a 3.2.5 → 4.4.0 jump across that boundary is
exactly where API drift bites.

### Recording the poses

With the robot up, jog the arm in RViz to the pose you want — interactive marker
or the Joints tab, then Plan & Execute — and capture where it actually landed:

```bash
python3 record_states.py --arm right
```

It walks through `ready_state`, `pre_pick_state` and `drop_state` in that order,
waiting for Enter at each, and writes `pick_place_states_right.yaml`. Every
capture is written immediately, so a Ctrl-C halfway through keeps what you
already recorded — pick the rest up later without jogging back to them:

```bash
python3 record_states.py --arm left --missing    # only what is not on file yet
python3 record_states.py --arm left drop_state   # re-record just one
python3 record_states.py --arm left --list       # show what is on file
```

A run that ends with poses still absent says which ones and prints the command
to finish them, rather than leaving the gap to surface at pick time.

#### Mirroring one arm's poses onto the other

The arms are mirror images, so a pose recorded on one can be reflected onto the
other instead of jogged to again:

```bash
python3 record_states.py --arm left --mirror            # from the right arm's file
python3 record_states.py --arm left --mirror --missing   # only fill the gaps
```

The reflection **negates every joint except joint4**. That is measured rather
than assumed: all 128 sign patterns were scored by forward kinematics off the
generated URDF against 15 postures, asking which puts the other arm's tool at
the y-mirror of this arm's tool with the whole rotation mirrored too. Flipping
all but joint4 is exact and unique — joint4 is the elbow, the one axis the
reflection maps to itself. `native/tests/test_mirror_states.py` re-checks it.

Two plausible rules are wrong, and **both fail only on poses that use the middle
of the arm**:

| rule | `ready_state` | `drop_state` |
| --- | --- | --- |
| negate joint1 only | 138 mm out, tool tipped 45° wrong | worse |
| negate joint1/3/5/7 | exact to 0.1 mm | **107 mm out** |
| negate all but joint4 | exact | exact |

`ready_state` and `pre_pick_state` hold joint2 and joint6 within 0.0003 rad of
zero, so they cannot tell those rules apart; `drop_state` has joint2 at 0.185
and joint6 at 0.536, and it can. This is why the check uses random postures.

Mirrored values are checked against the **target** arm's limits, which are not
the mirror of the source arm's — the left joint2 runs `[-3.316, +0.175]` against
the right's `[-0.175, +3.316]`. A pose that is out of range once reflected is
reported and *not* written. Limits come from `/robot_description` when the robot
is up, and from xacro otherwise, so mirroring works before bringing it up.

Mirrored entries are marked `mirrored_from` in the YAML and in `--list`, because
they were computed rather than measured.

#### Playing the poses back

To see a recording rather than read it:

```bash
python3 record_states.py --arm left --play
```

It captures the current posture, drives through each pose on file, then returns
the arm to that captured posture — so it leaves things as it found them, even
if the arm started somewhere no pose describes. At each stop it prints the tool
position and the worst joint's error against the recording.

It asks before moving anything (`--yes` skips that), runs at `--speed 0.15` by
default — slower than a cycle on purpose — and plans through `move_group` like
everything else, so the octomap and self-collision checks still apply. A Ctrl-C
**stops** and leaves the arm where it is rather than driving it home, on the
grounds that a Ctrl-C usually means it is going somewhere it should not.

This is the one thing in `record_states.py` that commands motion. Especially
worth running after `--mirror`, since those values were computed and the
reflection knows nothing about the trunk or the other arm.

### cuMotion fails about one goal in seven, and it is not the pose

Measured on this robot: **20 of 138** joint-goal queries came back
`MotionGenStatus.TRAJOPT_FAIL`, on poses that planned fine on other tries.
14.5%. The optimiser is stochastic and the node reseeds between queries, so
**resending the identical goal normally works** — in that sample failures came
in runs of one (16 times) and two (twice) and never three.

It reaches MoveIt as `PLANNING_FAILED` (−1), which looks exactly like "that
pose is unreachable". It is not. Two things follow:

- `plan_attempts` (default **3**) resends a goal that failed for a retryable
  reason. Without it a cycle inherits 14.5% eight times over — only
  `0.855^8` = **29%** of cycles would get through without one spurious failure.
- Codes that describe the *goal* — in collision, bad link name, outside limits
  — are never retried, because asking again cannot change them. See
  `RETRYABLE_MOVEIT_CODES`.

`num_planning_attempts` on the request does **not** help: the plugin forwards
one request to the action server and ignores it. Measured — 5 attempts gave 11
failures in 60 where 1 attempt gave 8. cuMotion's own `max_attempts` is already
10 internally and this is the residue after it.

### Out of reach is not a planning failure

An object outside the arm's envelope fails every goal identically, and **no
ladder strategy recovers it** — a different yaw, or 8 mm lower, is still out of
reach. Running the ladder there produces six identical failures and a summary
that blames the planner.

So before anything moves, `/compute_ik` is asked whether the arm can put its
tool at the grasp, the pre-grasp, and the transit height. If it cannot, the
cycle stops:

```
out of reach: the right arm cannot put its tool at the grasp
(+0.700, -0.180, +0.010). Not moving. The left arm can reach it -- run with
arm_selection:=by_side, or arm:=left.
```

The check is **pure kinematics**: no collision checking and no
posture-continuity limit, because the question is whether the arm can reach the
point at all — not whether it can reach it from where it happens to be
standing, nor whether something is currently in the way. Those are different
failures and deserve different messages. It also asks the *other* arm, since
"the other one can" is the useful half of the answer.

**Two solvers get asked, and only one of them is an authority.**

`/compute_ik` (KDL) is quick, and its *yes* is conclusive. Its **no is not**.
KDL is a randomly-seeded iterative solver and this config shipped it a
`kinematics_solver_timeout` of **5 ms**, which is nowhere near enough for a
7-DOF arm on a full 6-DOF pose. Measured on this robot: **0 solutions in 10
tries, for both arms**, at a point the arm can actually pick from — unchanged
at a 1 s and a 5 s request timeout. Gating the cycle on that refused objects
the robot could pick, reporting `out of reach` for both arms at once.

So the timeout is now **0.05 s**, KDL gets `ik_attempts` (3) tries because it
is stochastic, and on a KDL *no* **cuMotion is asked with a `plan_only`
request**. cuMotion is the authority because cuMotion is what executes:
cuRobo's IK is far stronger, and if it can plan there the arm can go there.
`plan_only` means nothing moves. Only when both say no does the cycle refuse.

`reachable()` returns three values, not two: True, False, and **None for
"could not determine"**. None never refuses — reporting `out of reach` because
a service did not answer is how a stack that is merely not up looks like a
workspace problem. `check_reach:=false` turns the whole check off.

One thing the refusal message now says, because it is usually the real
constraint: the tool is asked for a **straight-down** grip, which is the
hardest orientation at long reach. A point that refuses at 50 cm out is often
reachable at a tilt.

### Re-planning a move the arm has already made

A retry re-enters `PRE_PICK` on every attempt. Re-planning a posture the arm is
already in costs several seconds and a roll against cuMotion's ~14.5% failure
rate, and the arm visibly steps back and forth between `PRE_PICK` and `TRANSIT`
for nothing. The named-state moves — HOME, READY, PRE_PICK, DROP — now return
success without sending a goal when the arm is already there, within
`at_goal_tolerance` (0.02 rad).

**That decision has to be made on a fresh reading**, and this is subtle enough
to have been wrong twice. `/joint_states` is cached by a callback on another
thread, so the cached value can still describe the posture from *before* the
last goal. Deciding not to move on that means skipping a move because the arm
was at the target a moment ago — exactly backwards. Anything that decides not
to act now waits for a reading that arrived after the question was asked.

The same applied to `at_home_pose`, which gates the octomap: a stale "I am at
home" is how the arm ends up inside its own map.

### An unreachable target looks exactly like a planner problem

The failure that reads worst in the logs is not a planning failure at all.
Measured here, with the arm asked for "red battery":

```
grasp z -0.755 below min_grasp_z 0.010; clamping
target at [3.0799, 2.3206, -0.7416] axis_yaw=-1.5711 depth=2.982 m from 40 px
...
MotionGenStatus.IK_FAIL      × every attempt
TRANSIT: still failing after 3 attempts (last code -1)
every pick strategy was exhausted
```

The object was **3.8 m from the base and 0.74 m below the floor**. `IK_FAIL`
was the correct answer. Two things made it look like a planner fault:

- `min_grasp_z` clamped the *height* and left x and y alone, so a nonsense
  detection became a plausible-looking grasp at `z = 0.01` that the arm then
  spent all six ladder attempts failing to reach. Clamping is right for a
  centimetre of noise and wrong for this — a z far below the surface means the
  depth is wrong, so x and y are not to be trusted either.
- The cuMotion MoveIt plugin reports "No trajectory" and the pipeline turns
  **every** cuMotion failure into `PLANNING_FAILED` (−1), whatever the
  planner's own status was. So `IK_FAIL` (hopeless, do not retry) and
  `TRAJOPT_FAIL` (stochastic, a resend fixes it) arrive as the same code, and
  the retry dutifully resent an unreachable goal three times while reporting
  that the planner had not converged.

So detections are now checked *before* anything moves — `workspace_radius`
(1.0 m in x-y), `max_z_clamp` (0.05 m of slack below `min_grasp_z` before the
detection is refused rather than clamped), `max_grasp_z`, and finite values.
A refused detection costs one log line naming the depth stream, instead of the
whole ladder. When a goal does fail with −1, the message now says that the code
covers both causes and where to find which:

```bash
grep MotionGenStatus ~/.ros/log/python3_*.log | tail
```

**The root cause is upstream of all of it.** In the same run `move_group`
reported:

```
[image_transport] Topics '.../depth/image_rect_raw' and '.../depth/camera_info'
do not appear to be synchronized. In the last 10s:
  Image messages received:      0
  CameraInfo messages received: 15
  Synchronized pairs:           0
```

Zero depth images. That is why the detection was metres out **and** why no
octomap was built. Camera info without images usually means the depth profile
failed to start — see the dead colour stream note in troubleshooting, and check
first:

```bash
ros2 topic hz /camera/camera/depth/image_rect_raw
```

### When cuMotion dies, everything looks unreachable

cuMotion plans in a **separate node**; `move_group` only holds an action client
for it. So when that node dies, `move_group` stays up and cheerfully accepts
goals, and every one comes back `PLANNING_FAILED` or `TIMED_OUT`. Nothing says
"the planner is gone".

It does die. Observed here: it planned five goals across both arms, then took
**SIGFPE (exit −8)** mid-run, and the two goals after that failed −1 then −6.
Being native (cuRobo/CUDA), it leaves no Python traceback.

How to tell in one command:

```bash
ros2 action info /cumotion/move_group
```

`Action servers: 0` means it is gone — relaunch the robot. `--play` now checks
this before it moves anything, and if the planner dies part-way it says so
rather than reporting that it "could not return to the starting posture", which
invites blaming the pose.

`move_group` itself can die too, and independently: observed here at exit
**−11 (SIGSEGV)** about a second after an action client was killed with goals
still in flight. `ros2 launch` does not respawn, so the stack ends up
half-alive — controllers, RViz and the camera still up, nothing planning. Both
deaths are recorded in the launch log, which is the first place to look:

```bash
grep "process has died" ~/.ros/log/latest/launch.log
```

Note that a clean Ctrl-C used to look like a crash there: the goal-set planner
called `rclpy.shutdown()` unguarded, and since rclpy's signal handler has
already shut the context down by the time `spin()` returns, it raised
`RCLError: rcl_shutdown already called` and exited 1 with a traceback. It is
guarded now, so a death in that log is a real one.

### The arm sags, and the planner was not told

Feedback is fine — the motion log shows positions, velocities and **motor
efforts** all updating, and `finger_effort` swinging −0.70 → +0.14 as the
gripper closes. Gravity compensation is implemented too: KDL `JntToGravity`
feedforward plus an integral term, `Kp` 100, `Kd` 5.5.

What the log showed is that the tool still does not arrive where it is sent:

```
TRANSIT   target[+0.336 -0.168 +0.555]  tcp[+0.303 -0.163 +0.521]  dz -33.5 mm
PREGRASP  target[+0.336 -0.168 +0.405]  tcp[+0.321 -0.164 +0.391]  dz -13.7 mm
```

Always **downward**, repeatable, and **growing with extension**. Joint by
joint it lands on the elbow: `joint4` sits 14–28 mrad from its target while
holding 4.4–5.0 Nm, with the same sign every run.

**The cause was in the gravity model.** The KDL chain ran
`base_link → link7`, so everything past `link7` was left out:

| beyond `link7` | mass |
| --- | --- |
| `hand` | 0.3500 kg |
| `right_finger` | 0.0360 kg |
| `left_finger` | 0.0360 kg |
| **unmodelled total** | **0.4221 kg** |

That is comparable to `link7`'s own 0.466 kg, and it hangs at the longest
moment arm on the arm — precisely the load that sags the pitch joints, and
precisely why the error grows as the arm extends. `tip_link` is now the hand.
Both joints to it are **fixed**, so the chain still has exactly `ARM_DOF`
movable joints and the existing sanity check still holds; only the fingers'
0.072 kg stays out. **This is C++ and needs a rebuild:**

```bash
native/build_ws.sh --packages-select openarm_hardware
```

### Arriving is not the same as being told you arrived

Independently of the servo work, the orchestrator used to trust that a move
which returned SUCCESS had landed on the commanded pose. It had not: SUCCESS
means the *joint trajectory controller's* tolerance was met, which is a
different and looser thing than the tool being in the right place. A grasp
planned on that assumption closes above the object.

So the achieved pose is **measured and corrected** — but not by re-commanding,
which was my first attempt and was simply wrong. The log shows why:

```
PREGRASP cartesian ok  19.3 mm      target z=0.412  landed z=0.398  dz -13.8
PREGRASP cartesian ok  19.0 mm      target z=0.412  landed z=0.398  dz -14.0
PREGRASP cartesian ok  18.8 mm      target z=0.412  landed z=0.398  dz -14.0
PREGRASP cartesian ok  18.6 mm      target z=0.412  landed z=0.399  dz -13.0
PREGRASP did-not-converge 18.6
```

Ten re-issued passes went 19.3 → 18.6 mm and gave up, then eight stepped hops
followed — **eighteen moves on one leg**, arriving no closer. The intermediate
samples say it all: during a re-issued pass the tool does not move, all eight
samples reading `tcp[+0.3426 -0.3062 +0.4094]`. The joints are already at their
solution; the error is between that solution and reality, so commanding it
again changes nothing.

Two things do work, and both are in the log:

- **Waiting.** Held at one target the error went
  `19.3 → 19.0 → 18.8 → 18.6 → 15.5 → 13.4 → 11.5 → 10.7 mm` over about eleven
  seconds — the servo's integral term pulling in slowly. `settle_at` waits for
  that (`pose_settle_time`, 4 s), and stops early once it stops improving
  rather than burning the whole timeout on a 0.2 mm/s creep.
- **Aiming past it.** The residual is repeatable — `dz -13.8, -14.0, -14.0,
  -14.0` across four passes at the same target — so the target is commanded
  once more, offset by the measured error (`pose_offset_correction`).

At most **two moves** per leg instead of eighteen, and a leg that lands
accurately costs exactly one.

Reading the achieved pose needs the same care as the joint states:
`tcp_position(newer_than=…)` insists on a transform stamped after the move,
because `rclpy.time.Time()` returns the latest *available*, and immediately
after a move that is still the pose from before it.

### Closing on nothing

A close that reaches the shut aperture without the torque ever passing the cap
had nothing between the fingers, and the cycle now **fails at the close** and
says so. Straight from the log:

```
commanded 0.040884  finger 0.042018  effort +0.501 Nm
...  22 steps ...
commanded 0.000000  finger 0.002492  effort +1.160 Nm
```

Flat at about 0.5 Nm — friction — against a 2.5 Nm cap, fingers shut to
2.5 mm. Meanwhile the descent that preceded it reported `error_mm=3.6`, so the
**tool did reach its commanded grasp**: the object was not where it was
detected. Worth separating, because those two need completely different fixes,
and the cycle used to carry on and attempt the lift, the carry and the drop
with an empty gripper.

### A retreat that has no line settles for less

`LIFT` asked for 20 cm straight up and got `fraction=0.125` checked, `0.0`
unchecked — there is no line that long from down at the grasp. Refusing to lift
at all is worse than lifting less, so the retreat now falls back to the
pre-grasp height, which the approach already proved reachable, and warns that
the carry starts lower.

### A residual is not distance left to fly

A leg that flies 95.8% and then returns `fraction=0.0` for the last 12 mm is
not blocked — that gap **is** the standing offset, and asking for a line to a
pose the arm cannot hold returns 0% for ever. That used to fail the leg and
hand it to eight curved hops. It is now passed to `settle_at` and the offset
correction, which are the only things that can close it
(`pose_residual_limit`, 25 mm).

The shortcut applies **only when there is no line at all**. While segments are
still gaining ground they are flying real distance, and stopping at 20 mm
because that happens to be offset-sized would abandon a leg that was working.

### A goal on a joint limit is never reached

Symptom: the arm is sitting at home, and the cycle refuses to start —
`could not reach home`, or the panel's Pick doing nothing.

What the robot said, with the arm at rest:

```
$ ros2 topic echo --once /right_joint_trajectory_controller/controller_state
reference.positions[3]: 0.0          # joint4, commanded
feedback.positions[3]:  0.15583      # joint4, measured
error.positions[3]:    -0.15583
```

`0.15583` rad, constant to five decimals across 301 samples over 3 s, at
0.02 Nm of effort — and joint4 tracks 0.70 to 2.15 rad perfectly in the same
motion log. So it is not a servo fault and not sag. It is a floor:

```
$ python3 -c "..."   # from openarm.urdf
openarm_right_joint4  lower=0.0  upper=2.443461
```

The SRDF `home` group state asks joint4 for **0.0, which is exactly its lower
limit**, and the hardware stops 8.9° short of it. So the elbow can never
arrive, the move reports success or exhausts its retries depending on the run,
and every later "is the arm home?" check sees 0.156 rad of error and fails the
cycle — with the arm parked at home.

Three changes, because there are three separate mistakes there:

* **HOME no longer sits on the limit.** `home_joint_positions` puts joint4 at
  `0.20` rad. At a folded-down posture that is 11° at the elbow: the arm still
  hangs by the base, still out of the camera's frame, and the goal is one the
  joint can hold.
* **No commanded posture is allowed onto a limit.** `clamp_to_limits` pulls
  every joint goal `joint_limit_margin` (0.02 rad) inside the bounds read from
  `/robot_description`, and says so once per joint. It also stops handing
  cuRobo's trajopt a joint with no room on one side.
* **A settled standing offset counts as arrived.** `at_home_pose` accepts an
  error up to `home_settle_tolerance` (0.20 rad) when the arm has *stopped*
  (`joint_still_speed`), naming the joint in the log. Still moving, or further
  out than that, still fails — a move cut short is still caught. Being a few
  degrees off at a folded posture does not put the arm back in the camera's
  view, which is what the check is actually guarding.

The failure message now names the joint rather than saying "could not reach
home": `joint4 is 0.156 rad off (+0.156 vs +0.020 commanded)`.

### Why the descent was not a straight line

Symptom, over several runs: the arm goes above the object, the straight-line
descent covers half or four-fifths of the way and then reports
`fraction=0.0`, and the fallback flies eight curved hops instead — including
one that swings the whole arm through 157 degrees.

The motion log says exactly what happened. From run `1788771153`, the joint
values at each end of the planned Cartesian path:

```
PREGRASP  fraction=0.5652
  plan start [-1.219, 2.982,  0.196, 0.736, -0.263, -0.026, 1.364]
  plan end   [-1.396, 3.013,  0.079, 0.659, -0.148, -0.068, 1.157]
              ^^^^^^ joint1, and its lower limit is -1.396263
```

**The line does not stop early. It stops when joint1 runs out.** The right
arm's joint1 travels `[-1.396, +3.491]` — 80 degrees one way, 200 the other —
and the transit pose goal had put the arm on the *short* side. A Cartesian
path cannot continue past a joint's stop, so the remaining 44% returns 0%
whether collision checking is on or off.

Then the pose fallback flips to the other branch, between two waypoints 20 mm
apart:

```
PREGRASP 5/8  [-1.366, 3.022,  0.041, 0.706, -0.126, -0.073, 1.052]
PREGRASP 6/8  [+1.372, 0.415, -1.569, 0.575, -1.583,  0.200, 1.364]
               ^^^^^^ joint1 swings 157 degrees
```

That is the "circular path" — a redundant 7-DOF arm has many solutions for
one tool pose, and a pose goal is free to pick a different one each time. The
branch it lands in reaches the object but sits **on** two stops: joint3 0.013
rad and joint5 0.018 rad from their limits. From there the descent managed
`fraction=0.8182` and then 0.0, the 12.9 mm offset correction got no line at
all, and the lift needed three segments and another flip.

#### It is not the planner's fault: there is no comfortable posture

Solving the tool pose the cycle asks for — position, gripper down, jaws at the
object's yaw — at that object position, over 48 random seeds:

| what is asked for | solved | roomiest solution's worst joint |
|---|---|---|
| tool position + down + jaw yaw (what the code asked) | 25/48 | **0.000 rad** — on the stop |
| the same grasp with the jaws turned 180° | 17/48 | 0.172 rad |
| joint7's centre + link6 able to tilt (wrist partition) | 17/48 | **0.381 rad**, 0.175 after the tilt |

The first row is the important one. **Every solution is against a stop**, and
the roomiest of the 25 is the posture the robot actually used. cuMotion and
`/compute_ik` both return the first solution they find, so nine times in ten
that is a posture with nowhere left to go.

#### So the posture is chosen here, not by the planner

`arm_kinematics.py` reads `/robot_description`, walks the joint tree, and
solves for postures directly — with the joint-limit headroom as the thing
being maximised. It sends the answer as a **joint goal**, which cuMotion
accepts for either arm whatever its `ee_link` is, and which gives the same
posture every time.

Two changes to what is asked for, both of which cost nothing:

* **The jaws are symmetric.** A parallel gripper closes on the same two faces
  whichever way round it arrives, so the grasp yaw and the yaw turned by 180°
  are the same grasp — and very different postures. Trying both takes the
  roomiest solution from 0.000 to 0.172 rad. `grasp_jaw_flip`.
* **joint7 is a pitch carrying the whole hand.** The tool sits **180.1 mm**
  from joint7's centre, so that one joint swings it through a 180 mm arc.
  Approaching with the hand already pointing down means dragging that 180 mm
  of gripper below the wrist the whole way, *and* it makes joints 1–6 solve a
  harder problem. So the approach is planned for **joint7's centre**, with
  joint7 left out of it, and the hand is tilted down once the arm is over the
  object — which is `approach_frame: wrist`, the default.

The wrist objective is four constraints instead of six: the wrist point, plus
"link6's y-axis is horizontal", which is what has to hold for joint7 alone to
be able to bring the hand to vertical. joint7 appears in none of them — the
wrist point lies on its axis, and link6 is upstream of it. Verified against
finite differences to 3e-7, and the resulting posture, once tilted, puts the
tool **0.3 mm** from the target pointing straight down.

```
$ python3 native/tests/test_arm_kinematics.py
headroom: robot 0.013 rad, tool-pose solve 0.000, jaws flipped 0.172,
          wrist solve 0.175
```

Falling back is always allowed: with no `/robot_description`, no numpy, or no
solution, the approach reverts to a pose goal and says so in the log.
`approach_frame: planner` restores the old behaviour outright.

The reach check tries both grasp directions too. It ends the cycle outright,
so refusing a point that is reachable with the jaws the other way round would
be the worst kind of false negative.

One thing worth knowing about the retry ladder while reading this: at the
measured object position, asking for the grasp yaw **rotated 60 or 90 degrees**
has *no solution at all* — 0 from 400 seeds. The `yaw+90` and `yaw+90-lower`
rungs cannot succeed there however many times they are tried. They are still
worth having for an object elsewhere in the envelope; they are simply not the
recovery for this failure.

### The descent does not consult the octomap

On a top-down grasp the target **is** an obstacle. The octomap holds the object
being picked up and the table under it, so the gripper is required to enter
occupied voxels to reach the thing it is grasping — measured, a
collision-checked line solved 100% of the 20 cm drop to the pre-grasp and 25%
of the last 5 cm, stalling about a centimetre above the object.

The checked attempt was therefore never going to succeed, and asking for it
first cost a planning round trip and risked flying a *partial* checked line
that left the tool somewhere the remainder did not solve from. So the column
legs now go straight to the unchecked line (`descend_ignores_octomap`).

The exemption stays **per leg**. It applies only where the caller already
marked the leg exempt — the short straight vertical ones, between ends whose
reach was checked, with `min_grasp_z` as a hard floor and the gripper open.
Everything else is planned against the collision world as before, and the
tests check that a leg without the exemption is still asked the checked way.

The pre-flight asks the same way the descent will. Probing checked when the
descent flies unchecked would have it refusing postures that work.

**The trade-off, stated plainly.** The 15 cm transit-to-pre-grasp leg solved
100% *checked* in one measured run — so on that leg the map check was working,
and it is now skipped along with the rest. The exemption is granted per leg by
`approach_ignores_octomap`, and both legs of the column have it, so both go
unchecked. If the map check is worth keeping on the upper leg, that parameter
needs splitting into "upper" and "final" rather than one flag for the column;
the ends of both legs are reach-checked and `min_grasp_z` still floors the
descent either way.

### The jaws have to close *across* the object

joint7 turns about link6's y-axis. The fingers slide along the hand's y — and
the hand's y **is** link6's y, because a rotation about an axis cannot move
that axis. Measured: swinging joint7 over its whole range leaves link6's y and
the hand's y identical to four decimals while the tool's z goes from `+0.765`
to `-0.988`.

So the direction the jaws close in is fixed by joints 1–6 alone, and the
original wrist objective — "the wrist is here, and link6's y is horizontal" —
never said *which way* horizontal. The closing angle was whatever the solver
landed on. In the reproducible case that is **17.1° off** the object's axis,
and in general it is unbounded.

The consequence, from run `1788776708`:

```
DESCEND  cartesian  ok  fraction=1.0  points=49  error_mm=6.3
CLOSE    ...  22 steps  ...  finger 0.0 mm  torque 1.14 Nm  cap 2.50 Nm
CLOSE    closed-on-nothing
```

A complete straight-line descent onto a point 6.3 mm from target, and then
the gripper shutting on air.

Pinning the azimuth costs the partition **nothing** — joint7 was never in
those equations, and the Jacobian's joint7 column is still identically zero.
The objective is now five constraints on six joints instead of four: wrist
position, and link6's y along ±the object's axis (± because a parallel gripper
closes on the same two faces either way round). Measured after the change:
jaws `0.000°` off, tool still 0.4 mm from target, still 0.44 rad of headroom.

`grasp_yaw_free` restores the loose behaviour, and should stay off for
anything that is not round. Note this is a different knob from
`grasp_tilt_max`: the yaw is *which way the jaws face*, the tilt is *how far
off vertical the gripper comes down*.

#### Ranking by the wrong posture threw the usable ones away

Pinning the jaws exposed a second mistake, in the *selection* rather than the
objective. Candidates were ranked by the untilted solve's headroom and only
then filtered for whether joint7 could tilt them into line — so the ranking
discarded the good answers before anything looked at them. At the transit
point:

```
rank  1: untilted +0.497  tilt -90.0  off 20.8 deg  ->  unusable
rank  2: untilted +0.485  tilt -90.0  off 14.9 deg  ->  unusable
...
rank  7: untilted +0.324  tilt -90.0  off 25.4 deg  ->  unusable
rank  8: untilted +0.263  tilt +76.2  off  0.1 deg  ->  0.240 rad, usable
```

The seven roomiest postures all needed joint7 **beyond its own ±90° limit**
and landed 15–25° short of vertical. Keeping the top six kept six unusable
postures. The score is now the headroom of the posture that will actually be
commanded — tilt applied, infeasible ones dropped first — and every candidate
that comes back puts the hand exactly down with the tool inside a millimetre
of target.

The lesson generalises: score the thing you are going to do, not an
intermediate step towards it.

Note this does **not** on its own prove the grip will now succeed. An
arbitrary closing angle is one way to close on nothing; a detection whose x/y
is off by more than half the 44 mm jaw opening, or whose depth read the table
rather than the object, is another. `VLM/run_in_vlm_env.sh vlm_detect.py
"<noun>" --once` on the actual scene is what separates them.

### The tilt is not a separate move

The tool sits 180.1 mm from joint7's centre, so tilting it *after* arriving
over the object swings the tool through an arc — measured at **116 mm**, from
`[0.4668, -0.2293, 0.5754]` to `[0.3521, -0.2350, 0.5569]`, immediately before
the descent. That is exactly the curved motion all the straight-line work
exists to remove, and it cost a second plan (which needed a retry).

The partition belongs to the *solve*, which still leaves joint7 out
entirely. The execution does not have to copy it: the tilt angle is known
before anything moves, so it goes out as part of the same joint goal and the
approach ends with the hand already pointing down. `approach_tilt_stage`
restores the two-move version for a case where carrying the gripper 180 mm
below the wrist through the approach would be the greater risk.

### Voxel size

`octomap_resolution` in
`src/openarm_ros2/openarm_bimanual_moveit_config/config/sensors_3d.yaml` is
**0.02** m, down from 0.05. At 50 mm a screwdriver and the table under it
merged into a single block — coarser than the object being picked. The cost is
leaf count, which for an octree scales with observed surface rather than
volume (about 6x here), so `max_range` and the updater's
`far_clipping_plane_distance` come down from 5.0 m to 1.5 m to match: the work
surface is well inside a metre of the base, and voxelising the rest of the room
at 20 mm is pure cost.

**This needs an install to take effect** — the native tree holds a real copy of
the config, not a symlink to `src/`:

```bash
native/build_ws.sh --packages-select openarm_bimanual_moveit_config
```

Still at its original value, and worth a look if fine voxels do not help as
much as expected: `padding_offset: 0.20` in the same file. That is the depth
padding used to mask the robot out of its own camera view, ten times MoveIt's
usual 0.02, and it erases real geometry near the arm along with the arm. It
was presumably set to stop the arm being captured — which is now handled by
capturing the map only at HOME, with the arm out of frame.

### Letting go of strictly top-down

A vertical approach is six constraints on seven joints at a fixed point, and
near the edge of the envelope there is frequently no solution clear of the
joint stops at all. `grasp_tilt_max` lets the gripper come down off vertical --
0.35 rad, 20 degrees -- which on most objects is the same grasp and a much
larger set to choose from.

Vertical is offered first, both ways round, then tilts in increasing
magnitude across four directions, and the search stops as soon as it has a
posture with adequate headroom. So a tilt is only ever used because vertical
could not be.

Whether it helps depends entirely on where the object is. Measured, best
worst-joint headroom over 32 seeds:

| point | best vertical | best tilted |
|---|---|---|
| pre-grasp `x=0.406 z=0.409` | **+0.159** | +0.039 |
| transit `x=0.406 z=0.559` | +0.242 | **+0.490** |
| further out `x=0.460 z=0.409` | **no solution at all** | +0.000 |
| nearer `x=0.320 z=0.409` | **+0.183** | +0.178 |

It doubles the headroom at the transit point and is the only thing that
reaches at all 5 cm further out. At the object position that had been failing,
though, vertical with the jaws flipped is still the best option — so the tilt
is worth having, and was not what was blocking that pick.

Each orientation costs a posture solve of about a second, and there are up to
eighteen of them, which is why the search exits early and why the reach probe
takes only `reach_orientations` of them.

### One descent, and the gripper opens before it

There used to be a stop at the pre-grasp: down to 5 cm above the object, open
the gripper there, then down again. Two lines to solve, two settles, two
offset corrections, and the arm visibly pausing in mid-air — for a stop
nothing needed. The gripper opens just as well at the transit height, 20 cm
up, where there is nothing for open fingers to catch on, and a gripper that
fails to open there has cost no motion at all.

So the grasp is now:

```
above the object   →   OPEN   →   one line straight down   →   CLOSE   →   one line up
```

The pre-flight proves that single line before the arm leaves home, so it is
validated rather than attempted. `single_descent: false` brings the pre-grasp
stop back, and there is a test asserting the column really does become two
legs again — the option is real, not a dead parameter.

### cuMotion is restarted when it dies

It dies. SIGFPE, exit code −8, in most launches on 2026-09-07, and the cell is
useless without it: move_group's `cumotion` pipeline forwards every goal to
that node, so a dead planner turns even a joint goal to a recorded posture
into `TIMED_OUT` — and the failure reads as though the recorded pose were at
fault.

`respawn=True` with a 4 second delay, in `launch_everything.launch.py`. The
delay is for the CUDA kernels it loads at startup, during which it logs
nothing at all; `check_planner_ready` covers the window where it is back but
not yet planning, by asking it to plan a goal to the posture the arm is
already in. No motion, and a failure there cannot be about the target.

That last point is what the check is for. Measured: the node was started in
the same second a cycle began, three attempts at a plain joint goal to a
recorded posture came back `TIMED_OUT`, and the report told the operator to
re-record the pose. Nothing was wrong with the pose.

### Straight lines the whole way down

The descent, the pre-grasp and the lift were real Cartesian lines. The long
move from the staging pose to above the object was not — it was a joint goal
to a chosen posture, and it is the leg you watch the arm make.

It is now **pre-flight candidate zero**, tried before any posture:

1. probe a straight line from the staging posture to above the object,
2. probe the descent legs *from where that line ends*,
3. if the whole chain solves, fly the approach as a line and send no posture
   goal at all.

```
PREFLIGHT: pre-flight passed on a straight-line approach --
           approach 100%, pre-grasp 100%, grasp 100%. No posture goal needed.
```

So `PRE_PICK → above object → pre-grasp → grasp → lift` is one continuous set
of straight lines: the motion you get dragging the end-effector arrow in RViz.
The posture path stays underneath for when no line exists, and the log says
which was used — `ok-linear`, or a candidate number.

`HOME ↔ PRE_PICK` stay joint-space. Those are recorded postures, and a
straight line between two postures is not a meaningful request.

### Headroom is a threshold, travel is the tie-break

A posture can be excellent by every criterion above and still be the wrong
one. From run `1788777730`:

```
PREFLIGHT  ok  candidate 1 of 6  legs pre-grasp 100% grasp 100%  margin 0.4931
PRE_PICK   ok
TRANSIT    heartbeat running  tcp=[0.1784, -0.1737, 0.4983]
TRANSIT    heartbeat running  tcp=[0.1784, -0.1737, 0.4982]
...  20 seconds, the tool not moving ...
```

The best headroom of the session, both descent legs proved, and the
free-space plan to *take the posture up* never came back. With
`motion_timeout` at 60 s and three attempts it would have stood there for
three minutes.

That is the criterion that was missing: nothing scored how far the arm has to
travel to reach the chosen posture. Past `posture_margin` more headroom buys
nothing — a joint with 0.3 rad to spare is no freer in practice than one with
0.15 — while a posture on the other side of the workspace is a far harder
plan and a far longer move. So the ranking is now:

1. postures joint7 can actually tilt into line (wrist mode only),
2. then those with **at least** `posture_margin` of headroom,
3. ordered by **least travel from the staging pose** — `max |Δjoint|` against
   `pre_pick_state`, which is where the approach is actually made from, not
   where the arm stands while the pre-flight runs.

If nothing clears the threshold the order falls back to roomiest-first rather
than refusing: the pre-flight is the real gate, and it checks the descent from
whichever posture is chosen. Both numbers are logged, so a run says what was
picked and what it was picked over:

```
PREFLIGHT: pre-flight passed on candidate 1 of 6 -- pre-grasp 100%,
           grasp 100%, 0.135 rad of joint headroom, 2.91 rad of travel
           from the staging pose
```

### The tool frame, after all

`approach_frame` defaults to `tool`: solve the full `hand_tcp` pose here, take
the roomiest solution, send it as one joint goal.

The wrist partition stays available and is off. Measured at the same target,
once the tilt joint7 needs is accounted for, the two are within a few
thousandths of a radian — **0.172 against 0.175** at the pre-grasp, **0.247
against 0.240** at the transit point. It bought no measurable headroom, and it
cost three things: the grasp yaw is not part of a wrist point so it had to be
pinned separately, the tilt had to be solved and range-checked, and the
postures the objective liked best were ones joint7 could not tilt into line at
all.

What actually did the work was never the frame. It was **choosing** the
solution rather than taking the planner's first one — and that applies to
either frame.

### Nothing is discovered mid-descent

Symptom: `HOME → PRE_PICK → TRANSIT → PREGRASP → OPEN_GRIPPER → DESCEND`, and
then the whole thing starts again from pre-pick and home.

From run `1788775217`, the joints on arrival at the pre-grasp — **before** the
gripper opened:

```
PREGRASP  ok    [-1.363, 2.936, 0.044, 1.084, -0.218, -0.073, 0.741]
                joint1 -1.363, and its limit is -1.396263: 0.033 rad left
DESCEND  short  fraction=0.1818 checked, 0.1818 unchecked
DESCEND  refused-no-line
PRE_PICK ...    back to the staging pose, and round again
```

The descent needed joint1 to keep going negative. It had 33 milliradians. So
18% of the line solved, `descend_linear_only` correctly refused to substitute
a curve, and the attempt died — standing over the object with the gripper
open.

**None of that needed the arm to move to find out.** The geometry was fixed
from the moment the object was detected, and `/compute_cartesian_path` takes
an explicit `start_state`, so it will answer questions about a posture the arm
is not in yet. So it is asked first:

1. Solve several *distinct* candidate postures for the approach point
   (`arm_kinematics.approach_candidates`, ranked by joint headroom, near
   duplicates merged — eight genuinely different arm configurations at the
   position above).
2. For each, from that posture as the start state, probe the line down to the
   pre-grasp and on to the grasp — checked, then unchecked, the same two ways
   the real descent asks.
3. Take the first candidate whose whole column solves. Nothing has moved yet.
4. If none does, **refuse before moving**, and say what the best one managed.

```
PREFLIGHT: pre-flight passed on candidate 1 of 6 -- pre-grasp 100%, grasp
           100%, 0.381 rad of joint headroom
```

or

```
PREFLIGHT: no straight-line descent exists from any of the 6 postures that
           reach (0.406, -0.218, 0.559). The best managed pre-grasp 18%,
           grasp 0%. Nothing moved. This is the arm running out of travel
           along the descent, not an obstacle -- bring the object closer to
           the base, or try the other arm.
```

Joint headroom alone would not have caught this: the posture that failed was
*comfortable* by that measure until the descent asked it to travel in one
particular direction. Only planning the line says so.

#### And a proved cycle does not retry by going home

The six-rung ladder exists for cuMotion's stochastic `TRAJOPT_FAIL` and for
yaw/height variants. Once the column has been checked in advance, the next
rung will find the same geometry — so the only visible effect of retrying is
the arm travelling back to pre-pick and home for another identical go. After
a passed pre-flight a failure therefore ends the cycle with its reason
(`retry_after_preflight` restores the old behaviour). Stochastic planner
failures are unaffected: `_send_move_goal` already resends in place,
`plan_attempts` times, without moving anywhere.

### The motion log

Every commanded motion is appended to `motion_log.jsonl`, one JSON object per
line. It exists because the interesting failures here are not reproducible on
demand and the interesting numbers are gone by the time anyone looks: what the
arm was asked for, which mechanism ran, what came back, and where the joints
and **motor efforts** actually were before and after.

```
state          label              method     outcome    target
HOME           HOME               joint      ok         7 joints
PRE_PICK       PRE_PICK_STATE     joint      ok         7 joints
TRANSIT        TRANSIT            pose       ok         [+0.350 -0.180 +0.245]
PREGRASP       PREGRASP           cartesian  ok         [+0.350 -0.180 +0.095]  fraction=1.0
OPEN_GRIPPER   OPEN               gripper    ok         0.044
DESCEND        DESCEND            cartesian  ok         [+0.350 -0.180 +0.045]  fraction=1.0
CLOSE_GRIPPER  CLOSE              gripper    ok         0.042 ... 0.016
```

Each record carries `measured` (joint positions, velocities and efforts, plus
the finger and its effort), `before` for the same taken ahead of the move,
`tcp`, and `run`/`cycle` ids — the file is append-only across runs, and without
those a reader cannot tell one run's long idle gap from a robot that sat still. Cartesian records add `fraction` and `checked`, so a refused straight
line is distinguishable from a blocked one; failed goals add `code`.

**Where the time went.** Cartesian records also carry `plan_s` and `exec_s`,
the detector's `LOCATE` record carries `secs`, and `PREFLIGHT` carries the
`secs` its probes cost. Those exist because a cycle's wall clock is nothing
like where it looks: in run `1788859355`, `TRANSIT` wrote its first record
**18.59 s** after the state began — all of it inside a single `cartesian_move`
call — while `LOCATE` spanned 10 s and then 12.6 s, windows that also hold the
reachability probes and the pre-flight. Nothing in the log separated the
detector's own latency from the probes, or planning from flying, so every
answer about "why is it slow" was a guess. These four numbers end that.

Moves that actually ran also carry **intermediate samples**: `path` is the
measured joints and TCP taken every `motion_sample_period` (0.1 s) *while the
arm moves*, and Cartesian records add `planned`, the path move_group
interpolated. Endpoints alone cannot tell a straight descent from one that
swings 3.7 cm sideways on the way — which is exactly the difference that breaks
things here. Samples are capped at `motion_sample_limit` (40) per motion, ends
preserved and the middle thinned, so one slow descent cannot fill the file. A
record that never commanded motion — a refused line — carries no `path`.

The file is opened, written and flushed per record, so a run that ends in a
segfault — which has happened here more than once — still leaves everything up
to that point on disk. `motion_log:=` (empty) disables it, and it is
`.gitignore`d.

```bash
# what the last cycle actually did
python3 -c "
import json
for line in open('motion_log.jsonl'):
    d = json.loads(line)
    print(f\"{d['state']:14s} {d['label']:18s} {d['method']:10s} {d['outcome']}\")
"
```

### Retries do not shuttle home

Only an attempt that needs a **fresh look** at the object travels back to HOME,
because that is the only thing requiring the camera's view of the table to be
clear. Changing the wrist yaw or dropping the grasp 8 mm changes the grasp, not
the object, so those retry from wherever the arm already is and reuse the last
detection — the arm was visibly going `pre_pick → home → pre_pick` between
attempts for nothing. `redetect` in `STRATEGIES` is the flag.

### Running the tests

```bash
source native/setup.bash
python3 native/tests/test_pick_cycle.py       # the cycle, against a fake robot
python3 native/tests/test_arm_kinematics.py   # FK against the robot, posture choice
python3 native/tests/test_mirror_states.py    # the mirror rule and --play
python3 native/tests/test_grasp_geometry.py   # grasp frames and the retry ladder
```

`test_arm_kinematics.py` needs no ROS and no robot: it checks the chain against
joint values the real arm reported in `motion_log.jsonl`, its Jacobians against
finite differences, and its limits against the URDF's own numbers.

`test_pick_cycle.py` uses `ROS_DOMAIN_ID` 77 plus its process id modulo 20
(override with `PICK_TEST_DOMAIN`) because it serves its own `/move_action`, `/joint_states`
and gripper action, and two copies of the suite must not find each other
either. On the default domain those collide with a running stack —
two action servers on one name, two joint-state publishers — and the checks
start reading the real arm instead of the modelled one. Worse, a goal with
`plan_only` false could reach the real `move_group` and move the robot.

Reading the values back off the robot beats typing them in: the Joints tab only
shows whole degrees, and an executed plan lands near its goal rather than exactly
on it. The orchestrator re-reads the file at the start of every cycle, so
re-recording a pose takes effect on the next pick — no restart, which matters
when a restart costs a PaliGemma load.

`drop_state` is where the object is *released*, so it falls from whatever height
that pose holds the tool at. `record_states.py --list` prints the tool position
of each pose, which is the number to look at before the first run.

### Driving it without the panel

The panel is a client of the ordinary interface, so anything it does can be done
from a terminal. With the robot already up:

```bash
cd /home/mr/workspaces/isaac_ros-dev && source native/setup.bash
ros2 launch pick_place.launch.py prompt:="detect screwdriver"
```

Wait for `model loaded`, then:

```bash
ros2 topic pub --once /pick_place/prompt std_msgs/String "{data: 'pick up the wrench'}"
ros2 service call /pick_place/start std_srvs/srv/Trigger
```

The prompt topic is normalised exactly as the panel does it, so publishing a
whole sentence works. `ros2 topic echo /pick_place/state` follows the state
machine and `ros2 service call /pick_place/abort std_srvs/srv/Trigger` stops it
after the current motion. `ros2 launch pick_place_demo.launch.py --show-args`
lists every tunable.

### Pick with the right arm, and why that needs a relaunch

**cuMotion accepts Cartesian goals for exactly one link.** It compares every
pose goal's `link_name` against its own `ee_link` and rejects a mismatch outright:

```
Link name for Target Pose "openarm_right_hand_tcp" and Planning frame
"openarm_left_hand_tcp" do not match, relaunch node with
tool_frame = openarm_right_hand_tcp
```

`openarm.yml` sets `ee_link` to the **left** hand, so with a default bringup every
right-arm pose goal fails — measured here, the whole right-hand work area came
back unreachable for that reason alone, not because the arm cannot get there. The
node reads `tool_frame` once at construction, so it cannot be fixed at runtime:

```bash
native/run_launch_everything.sh tool_frame:=openarm_right_hand_tcp
```

The orchestrator checks this before it moves anything — it reads the planner's
`tool_frame`, falls back to the `ee_link` in `openarm.yml` when that is unset, and
refuses to start on a mismatch rather than discovering it at PREGRASP with the
gripper already open. The *other* arm still takes joint-space goals normally; only
Cartesian goals are restricted to one link at a time. Both arms stay in cuMotion's
kinematic chain either way, which is why `openarm.yml` now lists both tool frames
in `link_names` — with only one listed, pointing `ee_link` at the right hand drops
the left arm to zero active joints, unmodelled and therefore not something the
right arm would plan around (measured: 14 active joints with both listed, 7 with
only the right).

### The ready pose

`ready_joint_positions` is `joint1..joint7` in radians. The default is the
elbow-up pose with the tool clear of the table, captured off the right arm rather
than read off the sliders. To use a different one, jog the arm there in RViz and
ask for the numbers:

```bash
ros2 service call /pick_place/capture_ready std_srvs/srv/Trigger
```

It returns a paste-ready list, which beats the Joints tab — that only shows whole
degrees.

**With the default `place_mode:=ready` the object is released at this pose**, so it
falls the distance between the tool and whatever is under it. Check what is
underneath before the first run. `place_mode:=position` restores the older
behaviour of moving over a separate `place_position` and releasing there.

### Which Python runs what

There are **two** virtualenvs, and they can never share a process:

| | Holds | Used by |
| --- | --- | --- |
| `native/venv` | torch 2.7+cu128, numpy 1.26.4 — the ABI cuMotion's prebuilt kernels need | the orchestrator, the panel, `record_states.py`, the `native/tests/` suites |
| `VLM/.venv` | torch 2.14+cu130, numpy 2.2.6, transformers, pyrealsense2 | the detector node and `vlm_detect.py` |

`native/setup.bash` activates the first and exports a `PYTHONPATH` for the
cuRobo overlay. `PYTHONPATH` beats a venv's own site-packages, so anything in
`VLM/` has to be launched through the wrapper that scrubs it:

```bash
VLM/run_in_vlm_env.sh vlm_detect.py "screwdriver" --once
VLM/run_in_vlm_env.sh vlm_detector_node.py --ros-args -p prompt:="detect cup"
```

The two stacks only ever meet over DDS. `./check_isolation.sh` verifies they
do not overlap.

Running a `VLM/` script with the system `python3` picks up
`~/.local/lib/python3.10/site-packages`, and its `transformers` fails against
the system PIL:

```
AttributeError: module 'PIL.Image' has no attribute 'Resampling'
ModuleNotFoundError: Could not import module 'AutoProcessor'.
```

That is the signature of the wrong interpreter, not a broken install. The one
exception is deliberate: [VLM/vlm_geometry.py](VLM/vlm_geometry.py) — the
parsing, depth geometry and overlay — imports nothing heavier than numpy and
cv2, so its test needs no venv and nothing sourced:

```bash
python3 VLM/test_vlm_geometry.py
```

### The pieces

| File | Runs in | Does |
| --- | --- | --- |
| [VLM/vlm_detector_node.py](VLM/vlm_detector_node.py) | `VLM/.venv` | PaliGemma → 3D points in `world` on `/vlm/detections` |
| [VLM/vlm_geometry.py](VLM/vlm_geometry.py) | anything with numpy + cv2 | parsing, depth geometry and the debug overlay, shared by the two above |
| [VLM/vlm_detect.py](VLM/vlm_detect.py) | `VLM/.venv` | standalone: prompt in, dot + coordinates + annotated PNG out; works with or without the stack |
| [VLM/run_in_vlm_env.sh](VLM/run_in_vlm_env.sh) | — | environment isolation for the above |
| [VLM/bootstrap.sh](VLM/bootstrap.sh) | — | builds `VLM/.venv` from `requirements.txt` |
| [pick_place_orchestrator.py](pick_place_orchestrator.py) | `native/venv` | the state machine, MoveIt + gripper + planning scene |
| [pick_place_ui.py](pick_place_ui.py) | `native/venv` | the panel: type an object, Pick, Abort, live state |
| [record_states.py](record_states.py) | `native/venv` | capture, mirror (`--mirror`) and replay (`--play`) the named poses, one file per arm |
| [vlm_prompt.py](vlm_prompt.py) | `native/venv` | "pick up the X" → "detect X", shared by the panel and the orchestrator |
| [pick_place_demo.launch.py](pick_place_demo.launch.py) | — | robot + detector + orchestrator + panel, one command |
| [native/run_pick_place_demo.sh](native/run_pick_place_demo.sh) | — | the above, with the preflight checks |
| [pick_place.launch.py](pick_place.launch.py) | — | detector + orchestrator only, for an already-running robot |
| [VLM/pixel_to_world.py](VLM/pixel_to_world.py) | `VLM/.venv` | click any point, get its world coordinate |
| [native/tests/check_reachability.py](native/tests/check_reachability.py) | `native/venv` | ask MoveIt if the arm can get there |
| [VLM/test_vlm_geometry.py](VLM/test_vlm_geometry.py) | system `python3` | pixel → world maths and the overlay, no camera, model or venv |
| [native/tests/test_grasp_geometry.py](native/tests/test_grasp_geometry.py) | `native/venv` | grasp pose maths, no robot |
| [native/tests/test_pick_cycle.py](native/tests/test_pick_cycle.py) | `native/venv` | a whole cycle against a fake robot, no hardware |

### Interfaces

| Name | Type | |
| --- | --- | --- |
| `/vlm/detections` | `std_msgs/String` | JSON, authoritative — see the node's docstring for the schema |
| `/vlm/detection_poses` | `geometry_msgs/PoseArray` | same points, for RViz |
| `/vlm/debug_image` | `sensor_msgs/Image` | annotated view |
| `/vlm/prompt` | `std_msgs/String` | retarget the detector at runtime |
| `/pick_place/prompt` | `std_msgs/String` | what to pick next; a whole sentence is fine |
| `/pick_place/state` | `std_msgs/String` | current state machine step |
| `/pick_place/start` | `std_srvs/Trigger` | run one cycle |
| `/pick_place/abort` | `std_srvs/Trigger` | stop after the current motion |
| `/pick_place/capture_ready` | `std_srvs/Trigger` | current arm joints as a `ready_joint_positions` list |
| `/octomap_gater/refresh` | `std_srvs/Trigger` | let 3 depth frames through |

`vision_msgs` would be the idiomatic type for the detections, but it is not
installed on this box and adding it needs root, while every type above ships
with `ros-humble-desktop`.

### Parameters worth tuning

Detector — pass with `--ros-args -p name:=value`:

| Parameter | Default | |
| --- | --- | --- |
| `prompt` | `detect screwdriver` | any PaliGemma detection prompt |
| `axis_depth_tolerance` | `0.03` | depth band counted as "the object", metres |
| `inference_period` | `0.4` | seconds between inferences |
| `show_window` | `false` | OpenCV window instead of just the topic |
| `debug_dashboard` | `false` | colour + depth side by side |

Orchestrator — all exposed as launch arguments, `--show-args` lists the rest:

| Parameter | Default | |
| --- | --- | --- |
| `arm` | `right` | must match the `tool_frame` the robot was launched with |
| `ready_joint_positions` | elbow-up pose | the observation pose, `joint1..joint7` in radians |
| `states_file` | `auto` | the poses `record_states.py` writes; `auto` is `pick_place_states_<arm>.yaml`, re-read every cycle |
| `place_mode` | `state` | `state` drops at `drop_state`; `ready` at the observation pose; `position` uses `place_position` |
| `place_position` | `[0.35, 0.30, 0.25]` | `place_mode:=position` only — **placeholder, measure yours** |
| `grasp_finger_min` | `0.003` | finger position above which the gripper counts as holding something — measure it on your object |
| `grasp_miss_warn` | `0.010` | how far the jaws may sit from the commanded grasp before `CLOSE_GRIPPER` logs `off-target`. Reporting only; nothing refuses on it |
| `grasp_z_offset` | `-0.005` | applied to the object's detected *top* surface |
| `workspace_radius` | `1.0` | m in x-y from the base; a detection beyond this is refused before anything moves |
| `check_reach` | `true` | check reach before moving, and refuse with "out of reach" only if both `/compute_ik` and a plan-only cuMotion query say no |
| `ik_attempts` | `3` | KDL tries per query — it is randomly seeded, so one failure means little |
| `ik_timeout` | `1.0` | s, passed to `/compute_ik` (the solver's own cap is `kinematics_solver_timeout`, raised from 5 ms to 50 ms) |
| `at_goal_tolerance` | `0.02` | rad; a named posture this close counts as reached, and the move is skipped rather than re-planned |
| `arm_selection` | `by_side` | `by_side` chooses the arm before moving; `fixed` always uses `arm` |
| `arm_order` | `camera_half` | `camera_half` prefers the half the object is in, then the other arm; or `right_then_left` / `left_then_right` |
| `arm_split_px` | `0.0` | pixels added to the middle column when splitting the camera view between the arms |
| `ready_joint_positions_arm` | `right` | which arm `ready_joint_positions` was measured on; the other one needs a recorded `ready_state` |
| `max_z_clamp` | `0.05` | m of slack below `min_grasp_z` that is still clamped; further down the depth is wrong, so the detection is refused |
| `max_grasp_z` | `0.80` | m; a detection above this is refused |
| `approach_height` | `0.05` | pre-grasp height above the grasp: where the gripper opens |
| `transit_height` | `0.20` | height above the grasp at which the free-space move ends; below it the tool descends the vertical line in hops. `0` disables it |
| `linear_descent` | `true` | ask `/compute_cartesian_path` for a straight line below `transit_height` instead of a goal pose |
| `cartesian_step` | `0.005` | m; interpolation step for that line — smaller is straighter |
| `cartesian_min_fraction` | `0.98` | refuse a partial line rather than execute it and close on air |
| `approach_ignores_octomap` | `true` | let every leg of the vertical column retry its straight line unchecked — the grasped object is itself in the octomap |
| `descend_ignores_octomap` | `true` | on those legs, do not ask for a checked line at all — go straight to the unchecked one |
| `retreat_height` | `0.20` | m above the grasp that the object is lifted to before being carried; `0` means back to the pre-grasp only |
| `descend_linear_only` | `true` | refuse `DESCEND`/`LIFT` when no straight line exists rather than substituting curved hops |
| `pose_tolerance` | `0.005` | m; how close the **tool** must end up, checked from TF rather than trusted |
| `pose_settle_time` | `4.0` | s to let the tool creep onto its target before judging; exits early once it stops improving |
| `pose_offset_correction` | `true` | after settling, aim past the target once by the error remaining |
| `pose_residual_limit` | `0.025` | m; a gap this small with no line available is treated as the standing offset, not path |
| `cartesian_partial_min` | `0.5` | smallest share of a line worth flying before continuing the rest |
| `cartesian_segments` | `4` | straight segments allowed per leg |
| `cartesian_min_gain` | `0.002` | m a segment must gain before another is tried |
| `motion_log_heartbeat` | `1.0` | s between heartbeat records while a cycle runs; `0` disables |
| `motion_log` | `motion_log.jsonl` | append-only record of every motion; empty disables |
| `motion_sample_period` | `0.1` | s between samples taken *during* a move; `0` keeps only the endpoints |
| `motion_sample_limit` | `40` | most samples kept per motion |
| `descend_step` | `0.02` | fallback hop below `transit_height`, m; `0` disables stepping |
| `seeded_descent` | `true` | solve the column with IK seeded from the posture above and send **joint** goals, so lowering the tool cannot reconfigure the arm |
| `max_joint_jump` | `0.5` | rad; reject an IK solution moving any joint further than this from its seed — that is a flip, not a descent |
| `plan_attempts` | `3` | resends of a goal that failed for a retryable reason — cuMotion misses ~14.5% of goals with `TRAJOPT_FAIL` |
| `velocity_scaling` / `acceleration_scaling` | `0.4` | a **time dilation** of the path already planned, so this changes speed, not geometry. A bigger lever on cycle time than it looks: `_retime` divides every timestamp by it, so `0.3` stretched a trajectory 3.33× and `0.4` stretches it 2.5× — a quarter off every leg |
| `detection_reuse_age` | `10.0` | how old a detection may be and still be picked from without asking the detector again. `choose_arm` detects to decide which arm can reach, and the first attempt then wanted one of its own — two inference waits at the same stationary object, 22.6 s of a measured 76 s cycle. A failed attempt takes nearer a minute, so this reuses the one and re-detects the other. `0` disables reuse |
| `gripper_torque_cap` | `2.5` | grip torque cap in **Nm at the motor** (≈59.5 N at the finger, 5.25 mm of finger overshoot); enforced by stepping the close and stopping at the cap |
| `gripper_close_step` | `0.002` | coarse close step, m; quarter steps are used near the cap |
| `refresh_octomap_at_home` | `true` | capture the octomap only at HOME |
| `home_joint_positions` | `[0, 0, 0, 0.20, 0, 0, 0]` | HOME, joint1..7 in rad. The SRDF `home` group state except for joint4 — `0.0` there is exactly that joint's URDF lower limit and the elbow stops 8.9° short of it |
| `home_pose_tolerance` | `0.05` | rad; how close the measured joints must be to HOME before the map may be captured |
| `home_settle_tolerance` | `0.20` | rad; how far a **stopped** joint may stand off HOME and still count as arrived |
| `joint_still_speed` | `0.05` | rad/s below which a joint counts as stopped — separates "as close as this hardware gets" from "still on its way" |
| `joint_limit_margin` | `0.02` | rad; keep commanded postures this far off the position limits. A goal on a limit cannot be held |
| `approach_frame` | `tool` | how the move above the object is aimed: `tool` solves the full tool pose here and sends the roomiest solution as a joint goal, `wrist` solves for joint7's centre and tilts joint7 afterwards, `planner` sends a pose goal and lets cuMotion choose (the old behaviour) |
| `grasp_jaw_flip` | `true` | also consider the grasp with the jaws turned 180° — the same grasp, often a much roomier posture |
| `grasp_yaw_free` | `false` | leave the closing angle to the solver. Off: the jaws close along link6's y, which joint7 cannot move, so an unconstrained solve picks the angle arbitrarily |
| `grasp_tilt_max` | `0.35` | rad the gripper may come down off vertical. Vertical is tried first and tilts in increasing order, so nothing tilts that need not. `0` restores strict top-down |
| `grasp_tilt_steps` / `grasp_tilt_azimuths` | `2` / `4` | tilt magnitudes between 0 and the max, and directions for each |
| `reach_orientations` | `3` | orientations the reach probe may try per point — it sends a planning request for each, so the full tilt set would make an unreachable object cost a minute |
| `approach_tilt_stage` | `false` | send the joint7 tilt as its own move after arriving. Off: tilting at the end swings the tool through a 116 mm arc right before the descent |
| `posture_margin` | `0.10` | rad of joint-limit headroom an approach posture should have. A **threshold**: past it the ranking prefers the posture nearest the staging pose instead |
| `posture_seeds` | `48` | random restarts per posture solve. 48 takes about a second and is run once per pick attempt |
| `single_descent` | `true` | one line from the approach height straight to the grasp, with the gripper opened before it. `false` restores the stop at the pre-grasp |
| `check_planner_ready` | `true` | before the cycle, ask the planner to plan a goal to the arm's *own current posture*. Plan-only, no motion, and it cannot fail for reasons about the target |
| `linear_transit` | `true` | fly the long move above the object as a straight line too, when one solves from the staging pose and the descent still flies from where it ends |
| `preflight_descent` | `true` | prove the descent from the intended posture, before the arm leaves its rest pose. A refusal here costs no motion at all |
| `preflight_candidates` | `6` | postures the pre-flight may probe. Each costs up to four `/compute_cartesian_path` calls and no motion |
| `retry_after_preflight` | `false` | run the retry ladder even after a pre-flight passed. Off, because the next rung finds the same geometry and the arm travels home for nothing |
| `arm_selection` | `fixed` | `by_side` decides the arm from the object's world y |
| `use_table_collision` / `table_z` | `false` / `0.0` | explicit work-surface box; worth enabling with `octomap:=static`, where the map can legitimately be empty |
| `velocity_scaling` | `0.15` | start lower on the first hardware run |

### Seeing where the VLM thinks the object is

The quickest answer, needing nothing but the prompt — a dot on the object, its
coordinates, and an annotated PNG on disk:

```bash
VLM/run_in_vlm_env.sh vlm_detect.py "screwdriver" --once
```

```
prompt: 'detect screwdriver'
source: ros (colour topic is publishing)
coordinates in world (tf2):
  #0  pixel (430, 250)  depth 0.612 m from 2601 px
        camera xyz  +0.0120 +0.0200 +0.6120
        target xyz  +0.3980 -0.1735 +0.2010
        axis yaw    -1.2 deg (from depth)
wrote /home/mr/workspaces/isaac_ros-dev/vlm_detection.png
```

Drop `--once` for a live window. [VLM/vlm_detect.py](VLM/vlm_detect.py) picks
its frame source automatically, because the D455 can only be claimed by one
process:

| source | when | coordinates |
| --- | --- | --- |
| `ros` | the robot is up — `realsense2_camera` owns the device | `world`, from tf2, exactly as the node gets them |
| `realsense` | the stack is down — it opens the camera itself | camera optical frame; add `--mount` for `world` from the URDF mount |

`--source` forces one. Other flags: `--save PATH`, `--json PATH`,
`--frame`, `--model`, `--period`, and the depth limits.

It shares `build_detections` with the node, so the dot you look at and the
coordinate the orchestrator drives to come out of the same function and cannot
drift apart. The prompt is normalised the same way too — `screwdriver`,
`the screwdriver` and `pick up the screwdriver` all become
`detect screwdriver`.

It is also the fastest way to catch the failure that produced an object four
metres away: with colour arriving and depth not, it says so and reports nothing
rather than inventing a coordinate.

**Several prompts per run.** The model load is the slow part, so finding a word
the checkpoint responds to costs one run rather than one per guess — every
prompt is tried on the same frame:

```bash
VLM/run_in_vlm_env.sh vlm_detect.py "tape" "spirit level" "yellow tool" --once
```

Measured on a table holding a roll of tape and a yellow spirit level:
`tape` and `spirit level` both landed tight boxes; **`yellow tool` and
`yellow object` returned nothing at all**. `paligemma-3b-pt-224` answers to
object nouns, not to colour-plus-category descriptions — worth knowing before
concluding the camera or the arm is at fault.

**A box covering more than a quarter of the frame is discarded**, because the
checkpoint always answers: asked for something absent it returns most of the
view rather than nothing. Observed here — `detect red battery`, on a table with
no battery on it, came back as the *entire frame*, 98527 depth pixels, and the
centre of that box became a grasp target in the middle of the table that was
out of reach of both arms. Three steps later that reads as a reach problem. On
the debug image such a box is the thin border right around the edge, easy to
miss because the eye goes to the axis contour drawn inside it.

**The camera can stop delivering.** A D455 will enumerate, report USB 3.2, and
accept a pipeline start while sending no frames at all — confirmed with a bare
`librealsense` pipeline, so it is the device and not this code. A hardware
reset clears it (0/10 frames before, 10/10 after), so `--source realsense` does
that automatically when the warm-up gets nothing, and says so. `--no-reset`
disables it.

### Watching it live inside the stack

`/vlm/debug_image` carries the same annotated view: the detection box, the
depth-derived object axis in green, and the numbers the orchestrator is
actually acting on — distance, world xyz, grasp yaw, which segmentation path
produced the axis, and how many valid depth pixels backed it. A box drawn in
**red** means the detection had no usable depth, so it was skipped.

```bash
ros2 run rqt_image_view rqt_image_view /vlm/debug_image
```

an `Image` display on `/vlm/debug_image` in RViz, or a plain OpenCV window
straight from the node:

```bash
VLM/run_vlm_detector.sh --ros-args -p show_window:=true
```

Add `-p debug_dashboard:=true` for the colour + depth side-by-side view, with the
boxes drawn on both panels — that is where you see *why* a detection had no depth.

The same numbers in text form:

```bash
ros2 topic echo /vlm/detections --once --full-length
```

`axis_source` is the field to watch: `depth` is the good path. `intensity` means
the object did not separate from the background in depth and the axis came from
image contrast instead, which is what produced a 90°-wrong grasp angle before —
raise `axis_depth_tolerance` if you see it. `bbox` means neither worked and the
axis is just the box's aspect ratio.

### Calibrate these two before trusting it

- **`pre_pick_state` and `drop_state`** — nothing ships with sensible defaults for
  these because they depend on your table. Record them with `record_states.py`,
  and check the tool height of `drop_state` with `--list` before the first run:
  that is how far the object falls.
- **The ready pose** — the default is a sensible elbow-up posture. Jog to the pose
  you want and capture it with `/pick_place/capture_ready`. Only under
  `place_mode:=position` does `place_position` matter, and its
  `[0.35, 0.30, 0.25]` default is a placeholder.
- **The detection itself** — before any motion, add a `PoseArray` display on
  `/vlm/detection_poses` in RViz. The arrow must land *on* the real object. If it
  is offset, the camera mount measurement in `cam_org.txt` is wrong, not the code.
  `/vlm/debug_image` shows what the model sees and the world coordinates it
  produced.

Both nodes' geometry can be checked without hardware — no camera, no robot, no
model load. Expected values in these are worked out by hand rather than taken
from the code, so they catch the failures that still produce plausible-looking
numbers:

```bash
VLM/run_in_vlm_env.sh test_vlm_geometry.py
```

Pixel → world: image decoding, the `<locNNNN>` parsing, deprojection, the tf
composition, median depth sampling, and the axis segmentation including the
Otsu trap described below.

```bash
source native/setup.bash && python3 native/tests/test_grasp_geometry.py
```

Grasp poses: that the tool approach axis lands on world −Z at every yaw, that
the fingers close perpendicular to the detected object axis, and that the retry
ladder actually escalates.

```bash
source native/setup.bash && python3 native/tests/test_pick_cycle.py
```

A whole cycle against a stubbed robot — `/move_action`, the gripper, TF,
`/joint_states`, the planning scene, cuMotion's parameters and the detector are
all faked, so the real orchestrator runs and every goal it sends is recorded.
It asserts the order of the eight goals, that PRE_PICK and DROP replay the
recorded joint values, that PREGRASP and LIFT sit `approach_height` above the
grasp, and that an unrecorded or wrong-arm states file makes the cycle refuse
without sending a single goal. This is the one that catches a reordered
sequence, which the geometry tests cannot see.

### Reach: ask cuMotion, never `/compute_ik`

**`/compute_ik` lies on this robot.** `config/kinematics.yaml` configures
`kdl_kinematics_plugin` with a **5 ms** solver timeout, and KDL is a local
numerical solver seeded from the current state. On a redundant 7-DOF arm it
returns `-31 NO_IK_SOLUTION` for poses cuMotion plans without trouble — measured
here, it rejected the entire work area including poses that plan fine. OMPL
inherits the same problem, because it samples pose goals through that plugin.

So check reachability against the planner you will actually use:

```bash
python3 native/tests/check_reachability.py --point 0.38 0.15 0.35
```

That sends a `plan_only` goal through cuMotion and sweeps approach azimuth and
tilt, so a failure tells you whether the *position* is out or only some approach
angles are.

What the geometry actually is, verified three ways — cuMotion planning, a
position-only damped-least-squares IK over the URDF, and 300k random FK samples
(the FK was checked against TF at all-zeros: `0.000, 0.1735, 0.0819` vs TF's
`0.000, 0.173, 0.082`):

- shoulders at `(0, ±0.051, 0.698)`; link offsets sum to 0.804 m but the usable
  straight-line reach is **≈0.738 m**, less in awkward directions
- `(0.380, +0.150, 0.350)` — table height — **plans top-down** ✓
- `(0.541, -0.023, 0.350)` is **93 mm beyond reach**; `(0.496, -0.249, 0.375)`
  is 222 mm beyond ✗
- both reachable solutions sit exactly on `joint2`'s upper limit of +0.175 rad
  (10°). Relaxing that limit recovers only ~3 cm and then plateaus, so it is
  the arm's geometry that runs out, not that one limit.

Practically, for the left arm keep pick targets around **x ≤ 0.40, y ≥ +0.10** at
table height. Reach does improve with height, so a riser buys a few centimetres —
but moving the object into the near-left area is simpler and enough.

The right arm mirrors it. Measured by planning a top-down pose straight through
cuRobo from the ready pose, empty world, so this is reach and not collision
(`# ` = plans):

```
              x=0.25  0.30  0.35  0.40  0.45  0.50
z=0.42          #     #     #     #     .     .      y = -0.05 .. -0.30, all rows
z=0.35          #     #     #     #     .     .      y = -0.10 .. -0.25
z=0.25          #     #     .     .     .     .      y = -0.05 .. -0.30
```

So **x ≤ 0.40, y ≤ −0.10** at z ≥ 0.35, and only **x ≤ 0.30** if the object sits
as low as z = 0.25. Nothing at x ≥ 0.45 at any height, exactly as on the left.
Note the grasp itself is the *low* pose in a cycle, so a pick off a low table is
the case that runs out of reach first.

Reproduce either arm's figures with `check_reachability.py --arm right`, but
remember it goes through cuMotion: the robot has to have been launched with a
matching `tool_frame`, or every cell comes back unreachable for that reason
instead.

`VLM/pixel_to_world.py` tells you where something *is*; this tells you whether
the arm can get there. A pick that dies on an opaque MoveIt error code is nearly
always one of these two.

### Dry-running on fake hardware

```bash
native/run_pick_place_demo.sh --fake
```

The arms are simulated by `mock_components` and the whole sequence plays out in
RViz, while **the camera, the depth stream and the detector are all real** — so
the object really is found where it is, and only the moving is pretend. No CAN
bus, no arms plugged in.

`--fake` also neutralises the two checks that cannot pass on simulated arms,
which is the difference between watching a cycle and watching the retry ladder
burn all six attempts: `mock_components` reports the finger exactly where it was
commanded, so the grip never registers, and the object never physically moves,
so the place is never confirmed. It expands to

```bash
use_fake_hardware:=true grasp_finger_min:=-1.0 object_moved_eps:=0.0
```

and it expands in the script rather than being passed through, so the CAN check
sees it too. Anything you write after `--fake` still wins — a repeated launch
argument takes its last value:

```bash
native/run_pick_place_demo.sh --fake arm:=left prompt:='detect mug'
```

Never pass those last two arguments on real hardware. They are exactly what
makes the retries work.

### Why it is built this way

- **The detector subscribes to the camera, it does not open it.**
  `launch_everything.launch.py` already gives the D455 to `realsense2_camera` for
  the octomap, and the device only allows one owner. Colour and `align_depth` are
  enabled there for this reason. Colour runs at `640x480x15`: this D455's RGB
  sensor offers only 1280x720, 1280x800x8, 640x480 and 424x240 — no 848x480, so
  matching the depth profile silently fell back to 640x480 anyway. Check yours
  with `ros2 param describe /camera/camera rgb_camera.color_profile`. Since the
  aligned depth follows the colour resolution,
  `/camera/camera/color/camera_info` is the one set of intrinsics that describes
  both. Expect ~6–10 Hz on the topics with alignment and RViz running, which is
  ample against a 0.4 s inference period.
- **No grasp-pose network — the VLM's axis plus a top-down approach is enough.**
  AnyGrasp, Contact-GraspNet and friends earn their keep on cluttered bins of
  unknown objects, where the question "where *can* I grip this at all" is
  genuinely open. That is not this problem: single objects on a flat table, a
  parallel-jaw gripper, and an arm whose verified envelope is a top-down
  approach over a small work area. Once the approach is fixed to straight down,
  the only free parameter left is the wrist angle — one number, which the
  detector already produces from the depth segmentation, and which the retry
  ladder already escalates when it is wrong. A grasp network would decide the
  same number at the cost of a third Python stack (the two here already conflict
  over torch and numpy), a licence key in AnyGrasp's case, several more GB of
  VRAM alongside PaliGemma, and 6-DOF poses that mostly fall outside an envelope
  that only plans top-down anyway. Worth revisiting the day the objects arrive
  in a pile.
- **The grasp axis is segmented from depth, not from image contrast.**
  The earlier prototype ran Otsu on the colour crop. On a narrow crop of a
  screwdriver that locks onto the band where the handle meets the shaft and
  reports an axis ~80° off — measured on this rig: a 25×119 px vertical
  detection came back as `image_angle_deg: -10.3`, which would have closed the
  fingers *along* the tool instead of across it. Depth has no such texture, so
  the axis now comes from the pixels within `axis_depth_tolerance` (3 cm) of the
  object's own median depth, with intensity and then the box aspect ratio as
  fallbacks. `axis_source` in each detection says which one ran.
- **Two Python stacks, two processes, DDS in between.** `native/venv` is pinned
  to torch 2.7.0+cu128 / numpy 1.26.4 for cuRobo's prebuilt kernels; the VLM
  needs torch 2.14+cu130 / numpy 2.2.6. They cannot coexist in one process, so
  `run_in_vlm_env.sh` scrubs `PYTHONPATH`/`LD_LIBRARY_PATH` before starting the
  detector.
- **Extrinsics come from tf2.** The hardcoded `T_CAM_TO_ROBOT` matrix the early
  prototypes carried disagreed with the calibrated URDF mount; the node now
  looks up `camera_color_optical_frame → world`, so re-measuring the mount needs
  no code change.
- **The gripper bypasses MoveIt** — cuMotion rejects 1-DOF groups (see below), so
  it goes straight to `/left_gripper_controller/gripper_cmd`.
- **Grasp success is judged geometrically, not by asking the VLM.**
  `paligemma-3b-pt-224` is a pretrained checkpoint: `detect X` is well-formed,
  yes/no VQA is not. So verification re-detects and checks *where the object
  went* — still at the pick point means the grasp failed — combined with the
  finger position from `/joint_states`. Note the gripper controller has
  `allow_stalling` unset, so a **successful** grasp returns `ABORTED`; the action
  result is deliberately ignored.

---

## Why native and not the container

**cuMotion cannot run in the Isaac ROS container on this machine's RTX 5070 Ti.**
The container's CUDA toolkit is 12.2 (inherited from its `tritonserver:23.10-py3`
base). cuRobo has to JIT-compile its kernels on this GPU, and 12.2's `nvcc` does
not know `compute_120`:

```
nvcc fatal : Unsupported gpu architecture 'compute_120'
```

The host has CUDA 12.8, so the native path works. The native environment is fully
self-contained in `native/` and does not modify the host's ROS, Python or apt
state — see [native/README.md](native/README.md) for exactly what it touches, how
it is built, and the two GPU-specific bugs it had to work around.

### Falling back to the container

The container setup is untouched and still works for everything that does not
need cuMotion:

```bash
cd ~/workspaces/isaac_ros-dev/src/isaac_ros_common && ./scripts/run_dev.sh
```

Then inside the container:

```bash
source install/setup.bash && ros2 launch /workspaces/isaac_ros-dev/launch_everything.launch.py
```

Note the container writes to the root `build/` and `install/` trees, while the
native setup uses `native/build` and `native/install`, so the two never collide.

---

## 🛠️ Important Notes & Fixes Applied

- **cuMotion Planner (GPU IK)**: cuMotion is the default MoveIt planner for the
  7-DOF arms (`left_joint_trajectory_controller`,
  `right_joint_trajectory_controller`).
- **One Cartesian link at a time**: cuMotion rejects any pose goal whose
  `link_name` differs from its `ee_link` (`INVALID_LINK_NAME`), so the arm you can
  give pose goals to is fixed at bringup by
  `tool_frame:=openarm_{left,right}_hand_tcp`. `launch_everything.launch.py`
  defaults it to the right hand. Joint-space goals are unaffected for either arm.
- **Gripper Planning**: cuMotion **does not support 1-DOF grippers**. Planning for
  them with cuMotion is safely rejected rather than crashing.
  **Always switch the Planning Pipeline to `ompl` in RViz** for `left_gripper` or
  `right_gripper`.
- **Gripper PID Tuning**: the DM4310 gripper motor gains (`Kp = 20.0`, `Kd = 0.5`)
  are tuned to remove high-frequency noise and vibration while keeping enough
  torque for accurate movement.
- **Blackwell register cap**: on `sm_120` only, `native/setup.bash` sets
  `NVCC_APPEND_FLAGS=-maxrregcount=160`. Without it, every cuMotion *trajectory*
  optimisation fails on this GPU with `too many resources requested for launch`
  (IK alone still works). It is applied conditionally because on `sm_75/86/89`
  the prebuilt cubins are used and the kernel fits in 118 registers anyway, so
  the cap would only force needless spilling. If you raise `num_steps` in
  `config/cumotion_planning.yaml` above 32, the cap must be recomputed —
  the formula is in [native/README.md](native/README.md).
- **Octomap gater**: in `octomap:=static`, `octomap_gater.py` sits between the
  camera and MoveIt. It now takes `/octomap_gater/refresh`
  (`std_srvs/srv/Trigger`) and `/octomap_gater/allow_frames`
  (`std_msgs/Int32`) as well as the Tk button, so an autonomous loop can update
  the map — the pick-and-place retry ladder calls it on its last attempt. It also
  no longer dies when there is no `DISPLAY`.
- **Camera mount**: the D455 pose lives in
  `src/openarm_description/urdf/robot/v10.urdf.xacro` as the `xacro:sensor_d455`
  origin, currently `xyz="0.10175 0.000 0.93272" rpy="0 1.0472 0"` — 101.75 mm
  forward, 932.72 mm high, tilted 60° forward. `cam_org.txt` tracks that
  measurement and its history. Note this positions
  `camera_bottom_screw_frame`; the D455's own geometry then puts `camera_link` at
  roughly `x=119.9, y=47.5, z=930.3` mm, which is expected.
- **Core Dump Cleanup**: `./scripts/run_dev.sh` deletes large `core.*` crash dumps
  from the workspace before the container starts, so the disk does not fill up.
  The native flow does not create them in the workspace root.

## Troubleshooting

Install-time problems — preflight failures, numpy ABI errors, an unexpected torch
version, the gated PaliGemma download — are tabulated in
[INSTALL.md](INSTALL.md#troubleshooting-the-install). Start there; the first check
is instant:

```bash
./check_isolation.sh
```

Check the native environment itself (works headless):

```bash
source native/setup.bash && native/verify_overlay.sh
```

Test the planning stack without a display — skips the octomap updater that
requires OpenGL:

```bash
source native/setup.bash && python3 native/tests/test_move_group_planners.py
```

If `move_group` dies immediately with `freeglut failed to open display`, you are
running without a display; use the headless test above, or start from a desktop
session.

### "every pick strategy was exhausted"

All six attempts failed. The state now names the cause, so read the rest of the
line before anything else:

```bash
ros2 topic echo /pick_place/state --once
```

| It says | What to do |
| --- | --- |
| `the planner is unusable for this arm` | see below — usually cuMotion has died |
| `nothing matched "detect X"` | the detector saw no such object. Check `/vlm/debug_image`, and that it has finished loading — a pick started during the PaliGemma load fails this way six times over |
| `could not plan to the pre-grasp above (x, y, z)` | almost always **out of reach**. The line includes a ready-made `check_reachability.py` command; for the right arm keep objects around `x ≤ 0.40, y ≤ −0.10` |
| `could not plan to pre_pick_state` | the recorded staging pose is not reachable from the observation pose; re-record it |
| `reached the pre-grasp but could not descend` | the octomap probably contains the object itself, or the grasp is under the table |
| `the gripper closed but nothing was held` | `grasp_finger_min` or the grasp height is wrong for this object — **read the `CLOSE_GRIPPER` line above it first**: if it says `off-target`, the jaws were not where they were sent and nothing about the detector or the fingers is wrong. Measured once at 29 mm high with the descent plan ending exactly on the point |

The out-of-reach case is the one that looks most like a software fault and is
not: the detector reports a perfectly good position, cuMotion simply cannot get
the tool there. `/vlm/detections` gives you the coordinate to check:

```bash
ros2 topic echo /vlm/detections --once --full-length
```

### cuMotion dies mid-session

It has crashed here with **SIGFPE**, six minutes into a session, leaving only
this in the launch output:

```
[ERROR] [cumotion_goal_set_planner_node-5]: process has died [pid ..., exit code -8, ...]
```

With no planner every pose goal fails, so a pick walks the whole retry ladder and
blames itself. The orchestrator now checks the planner is in the graph before it
moves anything and refuses with `the planner is unusable for this arm`, but if a
cycle is already running when it dies you still get the exhausted-ladder message.
Confirm with:

```bash
ros2 node list | grep cumotion || echo "cuMotion is not running"
```

The fix is to restart the robot; a dead planner cannot be revived in place, since
it reads `tool_frame` once at construction. Root cause unknown — the launch files
now run every Python node with `PYTHONUNBUFFERED=1` so that the next crash leaves
its traceback in `launch.log` instead of losing it in a stdout buffer, which is
what happened the first time.

### The detector never publishes, but the camera node is running

Check which streams are actually live — and note that **`ros2 topic echo` will
show nothing even on a healthy RealSense topic**, because it subscribes RELIABLE
while the camera publishes BEST_EFFORT:

```bash
ros2 topic hz /camera/camera/color/image_raw
```

If depth ticks and colour does not, look for a **second camera taking the video
device numbers**. librealsense pairs a colour stream with the adjacent
`/dev/videoN` for its metadata, so another UVC device landing in the middle of
the block breaks colour while leaving depth working:

```bash
for n in /sys/class/video4linux/video*; do
    echo "$(basename $n) $(cat $n/name) $(readlink -f $n/device | sed 's|.*/||')"
done
```

Every node of the RealSense colour interface (`:1.3` above) should be adjacent.
If they are split, unplug the other camera and **re-enumerate the RealSense** —
unplugging alone does not renumber what is already assigned:

```bash
sudo sh -c 'echo 0 > /sys/bus/usb/devices/2-4.1/authorized; sleep 2; echo 1 > /sys/bus/usb/devices/2-4.1/authorized'
```

### RViz never opens, but everything else starts

Look for this in the launch output — it scrolls past quickly, and every other
node comes up fine, so it reads like a launch-file problem when it is not:

```
rviz2: symbol lookup error: /snap/core20/current/lib/x86_64-linux-gnu/libpthread.so.0:
undefined symbol: __libc_pthread_init, version GLIBC_PRIVATE
[ERROR] [rviz2-4]: process has died [pid ..., exit code 127, ...]
```

That is a terminal opened **inside a snap** — VS Code's integrated terminal is
the usual one here, since `code` is installed as a snap. The snap exports
`GTK_PATH` and friends pointing back into itself; rviz2 loads a GTK module from
`/snap/code`, whose RPATH pulls in core20's glibc 2.31 `libpthread` next to the
host's 2.35, and it aborts before drawing anything.

`native/setup.bash` now clears those variables when they point into a snap, and
says so when it does:

```
note: cleared snap-provided GTK_PATH GTK_EXE_PREFIX ... (they crash rviz2)
```

So sourcing it is the fix. If RViz still will not start, check nothing re-added
them after sourcing:

```bash
source native/setup.bash && env | grep -E '^(GTK_PATH|GIO_MODULE_DIR)=' ; rviz2 --help | head -1
```

That should print no `GTK_PATH` line and then rviz2's usage. Launching from a
plain GNOME terminal instead of the VS Code one also avoids it entirely.
