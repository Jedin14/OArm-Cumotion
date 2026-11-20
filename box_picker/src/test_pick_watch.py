"""Checks for pick_watch.py on synthetic depth images (test_faces.py's scenes, no GPU needed).

A target box stands among others in front of the carton's back wall. The arm is a
bar in front of the target's pick face; the box is then taken away, left where it
was, or pushed sideways, and the watcher's verdict is compared with what happened.

Run: python test_pick_watch.py
"""
import numpy as np

from faces import analyse, choose_target, up_from_pitch
from pick_watch import (CLEAR_FRAMES, DROP_FRAMES, MOVED_FRAMES, SETTLE_S, STEADY_FRAMES, STILL_S, PickTracker,
                        PickWatcher)
from test_faces import INTR, PITCH, box_world, check, render

FRONT_TARGET = box_world(-0.06, 0.45, (0.10, 0.08, 0.04))     # upright: picked from the front
WALL = box_world(0.0, 0.62, (0.60, 0.40, 0.02))                # the carton's back wall
FLAT_TARGET = box_world(-0.06, 0.45, (0.10, 0.04, 0.08))      # lying flat: picked from above


def bar(x, y, z, size):
    """A box floating at (x, y, z), world metres: the arm."""
    return {"centre": np.array([x, y, z]), "axes": np.eye(3), "half": np.array(size) / 2}


ARM_FRONT = bar(-0.06, 0.10, 0.38, (0.04, 0.20, 0.04))         # in front of the upright target
ARM_ABOVE = bar(-0.06, 0.12, 0.45, (0.04, 0.12, 0.04))         # above the flat target


def beside(b):
    """The same box 12 cm to the right of b: the target's neighbour."""
    return {**b, "centre": b["centre"] + [0.12, 0, 0]}


def scene(boxes, noise, holes, seed):
    return render(boxes, noise, holes, seed=seed, floor=True)[0]


def run_frames(watcher, frames):
    """Feed each depth image MOVED_FRAMES + 1 times; the first verdict, or None."""
    for depth in frames:
        for _ in range(MOVED_FRAMES + 1):
            result = watcher.update(depth, INTR)
            if result:
                return result
    return None


def target_face(target_box, noise, holes, extra=(), with_masks=False):
    depth, masks = render([target_box, beside(target_box), WALL, *extra], noise, holes, seed=0, floor=True)
    boxes = analyse(depth, masks, INTR, up_from_pitch(PITCH))
    t = choose_target(boxes)
    return (boxes[t].pick, depth, t) + ((masks,) if with_masks else ())


CASES = []


def case(fn):
    CASES.append(fn)
    return fn


def verdict(name, target_box, arm, after, expect, noise, holes, arm_seen=True, extra=(), lifted=None, box_mm=None):
    """extra: more boxes, there all along. lifted: what is in view while the box is carried
    off (the arm with the box in the gripper), between the arm and `after`. box_mm: the box's
    size, given to the watcher."""
    face, depth0, t, masks = target_face(target_box, noise, holes, extra, with_masks=True)
    ok = check("target", t == 0 and face is not None, f"box {t}, {face.kind if face else None} face")
    if not ok:
        return ok
    w = PickWatcher(face, depth0, masks[0], box_mm=box_mm)
    rest = [beside(target_box), WALL, *extra]
    frames = [scene([target_box] + rest, noise, holes, 1)]
    frames += [scene([target_box, arm] + rest, noise, holes, 2)] if arm_seen else []
    frames += [scene(lifted + rest, noise, holes, 4)] if lifted else []
    frames += [scene(after + rest, noise, holes, 3)]
    result = run_frames(w, frames)
    return ok & check(name, result == expect, f"{result} (expected {expect})")


@case
def front_pick_taken(noise=0.0, holes=0.0):
    """Arm in front of the upright target, then the box is gone: picked."""
    return verdict("result", FRONT_TARGET, ARM_FRONT, [], "picked", noise, holes)


@case
def front_pick_missed(noise=0.0, holes=0.0):
    """Arm in front of the upright target, then the box is still there: missed."""
    return verdict("result", FRONT_TARGET, ARM_FRONT, [FRONT_TARGET], "missed", noise, holes)


@case
def front_pick_moved(noise=0.0, holes=0.0):
    """Arm in front, then the box is 5 cm to the left of where it was: moved."""
    pushed = box_world(-0.11, 0.45, (0.10, 0.08, 0.04))
    return verdict("result", FRONT_TARGET, ARM_FRONT, [pushed], "moved", noise, holes)


@case
def taken_with_same_box_behind(noise=0.0, holes=0.0):
    """Box taken, an identical box standing right behind it (4 cm further back): picked."""
    behind = box_world(-0.06, 0.49, (0.10, 0.08, 0.04))
    return verdict("result", FRONT_TARGET, ARM_FRONT, [], "picked", noise, holes, extra=[behind])


@case
def nudged_back_5mm(noise=0.0, holes=0.0):
    """Arm pushes the box 5 mm back and leaves it: missed, not picked."""
    nudged = box_world(-0.06, 0.455, (0.10, 0.08, 0.04))
    return verdict("result", FRONT_TARGET, ARM_FRONT, [nudged], "missed", noise, holes)


@case
def top_pick_taken(noise=0.0, holes=0.0):
    """Arm above the flat target, then the box is gone (floor under it): picked."""
    return verdict("result", FLAT_TARGET, ARM_ABOVE, [], "picked", noise, holes)


@case
def top_pick_missed(noise=0.0, holes=0.0):
    """Arm above the flat target, then the box is still there: missed."""
    return verdict("result", FLAT_TARGET, ARM_ABOVE, [FLAT_TARGET], "missed", noise, holes)


@case
def no_arm_box_stays(noise=0.0, holes=0.0):
    """Nothing comes near the target: no verdict, the watcher keeps waiting."""
    return verdict("result", FRONT_TARGET, None, [FRONT_TARGET], None, noise, holes, arm_seen=False)


def carried(b, lift=0.22):
    """The arm, reaching in from above out of the picture, with box b in its gripper `lift` m up."""
    held = {**b, "centre": b["centre"] + [0, lift, -0.03]}
    top = held["centre"][1] + held["half"][1]
    return [held, bar(held["centre"][0], top + 0.4, held["centre"][2], (0.04, 0.8, 0.04))]


DROPPED_LEFT = box_world(-0.22, 0.42, (0.10, 0.04, 0.08))     # the target fallen flat, left of where it stood


@case
def lifted_then_picked(noise=0.0, holes=0.0):
    """Arm, box carried up in the gripper (still in view), then out of view: picked."""
    return verdict("result", FRONT_TARGET, ARM_FRONT, [], "picked", noise, holes, lifted=carried(FRONT_TARGET))


@case
def lifted_then_dropped_elsewhere(noise=0.0, holes=0.0):
    """Box carried up, then it falls and lies flat elsewhere in the carton: dropped, not picked."""
    return verdict("result", FRONT_TARGET, ARM_FRONT, [DROPPED_LEFT], "dropped", noise, holes,
                   lifted=carried(FRONT_TARGET))


@case
def lifted_then_dropped_on_its_spot(noise=0.0, holes=0.0):
    """Box carried up, then it falls back onto its own spot: dropped, not picked."""
    return verdict("result", FRONT_TARGET, ARM_FRONT, [FRONT_TARGET], "dropped", noise, holes,
                   lifted=carried(FRONT_TARGET))


@case
def top_pick_lifted_then_dropped(noise=0.0, holes=0.0):
    """Flat box picked from above, carried up, then lands on the floor in front of the others: dropped."""
    landed = box_world(-0.03, 0.36, (0.10, 0.04, 0.08))
    return verdict("result", FLAT_TARGET, ARM_ABOVE, [landed], "dropped", noise, holes, lifted=carried(FLAT_TARGET))


@case
def neighbour_slides_into_gap(noise=0.0, holes=0.0):
    """Box carried off, then its neighbour slides 6 cm into the gap: picked (a box moved, none added)."""
    face, depth0, _, masks = target_face(FRONT_TARGET, noise, holes, with_masks=True)
    w = PickWatcher(face, depth0, masks[0])
    slid = {**beside(FRONT_TARGET), "centre": beside(FRONT_TARGET)["centre"] - [0.06, 0, 0]}
    frames = [scene([FRONT_TARGET, beside(FRONT_TARGET), WALL], noise, holes, 1),
              scene([FRONT_TARGET, ARM_FRONT, beside(FRONT_TARGET), WALL], noise, holes, 2),
              scene(carried(FRONT_TARGET) + [beside(FRONT_TARGET), WALL], noise, holes, 4),
              scene([slid, WALL], noise, holes, 3)]
    result = run_frames(w, frames)
    return check("result", result == "picked", f"{result} (expected picked)")


@case
def arm_stays_in_view(noise=0.0, holes=0.0):
    """Box carried up and the arm stays in view: no verdict while it is there."""
    return verdict("result", FRONT_TARGET, ARM_FRONT, carried(FRONT_TARGET), None, noise, holes,
                   lifted=carried(FRONT_TARGET))


@case
def tracker_dropped_then_retry(noise=0.0, holes=0.0):
    """Dropped: success false; the box where it fell is the next target, try 2."""
    face, depth0, _ = target_face(FRONT_TARGET, noise, holes)
    rest = [beside(FRONT_TARGET), WALL]
    tr, now, events = PickTracker(), 0.0, []
    tr.offer({}, face, depth0, now)
    tr.offer({}, face, depth0, now + STILL_S)
    now += STILL_S
    for boxes, n in [([FRONT_TARGET], STEADY_FRAMES + 1), ([FRONT_TARGET, ARM_FRONT], STEADY_FRAMES + 1),
                     (carried(FRONT_TARGET), STEADY_FRAMES + 5), ([DROPPED_LEFT], STEADY_FRAMES + DROP_FRAMES + 2)]:
        depth = scene(boxes + rest, noise, holes, 5)
        for _ in range(n):
            now += 1 / 30
            e = tr.update(depth, INTR, now)
            if e:
                events.append(e)
    depth, masks = render([DROPPED_LEFT] + rest, noise, holes, seed=6, floor=True)
    fallen = analyse(depth, masks, INTR, up_from_pitch(PITCH))[0].pick
    tr.offer({}, fallen, depth, now + 0.01)
    again = tr.offer({}, fallen, depth, now + 0.01 + STILL_S)
    got = [(e["result"], e["success"]) for e in events]
    ok = check("result", got == [("dropped", False)], f"{got}")
    return ok & check("retry where it fell", again is not None and again["attempt"] == 2, f"{again}")


@case
def tracker_moving_box_waits(noise=0.0, holes=0.0):
    """A box found 3 cm further each half second (carried away): never handed out."""
    face, depth0, _ = target_face(FRONT_TARGET, noise, holes)
    tr, got = PickTracker(), []
    for i in range(6):
        moved = type(face)(**{**vars(face), "centroid": face.centroid + [0.03 * i, 0, 0]})
        got.append(tr.offer({"x_m": float(moved.centroid[0])}, moved, depth0, i * STILL_S))
    return check("not handed out", got == [None] * 6, f"{got}")


@case
def tracker_retry_then_pick(noise=0.0, holes=0.0):
    """Not handed out until still; miss; a stale result is refused; the same box is try 2; picked."""
    face, depth0, _ = target_face(FRONT_TARGET, noise, holes)
    rest = [beside(FRONT_TARGET), WALL]
    there = scene([FRONT_TARGET] + rest, noise, holes, 1)
    arm = scene([FRONT_TARGET, ARM_FRONT] + rest, noise, holes, 2)
    gone = scene(rest, noise, holes, 3)
    tr, now, events = PickTracker(), 0.0, []

    def frames(depth, n=STEADY_FRAMES + 1):
        nonlocal now
        for _ in range(n):
            now += 1 / 30
            e = tr.update(depth, INTR, now)
            if e:
                events.append(e)

    target = {"x_m": float(face.centroid[0])}
    early = tr.offer(target, face, depth0, now)
    first = tr.offer(target, face, depth0, now + STILL_S)
    now += STILL_S
    frames(there), frames(arm), frames(there)
    stale = tr.offer(target, face, depth0, now - 1.0)     # from a frame taken before the miss
    tr.offer(target, face, depth0, now + 0.01)
    second = tr.offer(target, face, depth0, now + 0.01 + STILL_S)
    frames(arm), frames(gone, STEADY_FRAMES + CLEAR_FRAMES)
    ok = check("wait until still", early is None, f"{early}")
    ok &= check("first target", first is not None and first["attempt"] == 1, f"{first}")
    ok &= check("stale refused", stale is None, f"{stale}")
    ok &= check("retry", second is not None and second["pick"] == 2 and second["attempt"] == 2, f"{second}")
    got = [(e["pick"], e["attempt"], e["result"]) for e in events]
    return ok & check("results", got == [(1, 1, "missed"), (2, 2, "picked")], f"{got}")


# -- 19 Nov 2026: what went wrong on the robot ---------------------------------------------------

def feed(watcher, frames, fps=30.0, arm_out_at=None, edge_flicker=False):
    """Feed [(depth, n), ...] at fps, frame by frame, until a verdict -> (verdict or None, frames fed).
    arm_out_at: from this frame on (1 = the first) the robot says its arm is out. edge_flicker: each
    frame's left edge loses depth as the D455's does (flicker())."""
    k = 0
    for depth, n in frames:
        for _ in range(n):
            k += 1
            d = flicker(depth, 1000 + k) if edge_flicker else depth
            result = watcher.update(d, INTR, k / fps, arm_out=arm_out_at is not None and k >= arm_out_at)
            if result:
                return result, k
    return None, k


def flicker(depth, seed):
    """The D455's strip with no depth down the picture's left side, its border flickering from frame
    to frame: each row loses 50-90 px."""
    d = depth.copy()
    w = 50 + np.random.default_rng(seed).integers(0, 40, d.shape[0])
    d[np.arange(d.shape[1])[None, :] < w[:, None]] = 0
    return d


LIP = bar(-0.06, 0.04, 0.415, (0.10, 0.012, 0.01))   # a thin strip 1.5 cm in front of the upright target's face


@case
def pushed_back_15mm(noise=0.0, holes=0.0):
    """The arm pushes the box 1.5 cm back and leaves it (size not known): moved, not picked (was picked)."""
    pushed = box_world(-0.06, 0.465, (0.10, 0.08, 0.04))
    return verdict("result", FRONT_TARGET, ARM_FRONT, [pushed], "moved", noise, holes)


@case
def pushed_back_2cm_size_known(noise=0.0, holes=0.0):
    """Box 100 x 80 x 40 mm pushed 2 cm back, its size given: moved, not picked (gone needs 24 mm)."""
    pushed = box_world(-0.06, 0.47, (0.10, 0.08, 0.04))
    return verdict("result", FRONT_TARGET, ARM_FRONT, [pushed], "moved", noise, holes, box_mm=(80, 100, 40))


@case
def taken_with_same_box_behind_size_known(noise=0.0, holes=0.0):
    """Box taken, an identical box right behind it, the size given: picked (4 cm back > 24 mm)."""
    behind = box_world(-0.06, 0.49, (0.10, 0.08, 0.04))
    return verdict("result", FRONT_TARGET, ARM_FRONT, [], "picked", noise, holes, extra=[behind], box_mm=(80, 100, 40))


@case
def flickering_edge_picked_quickly(noise=0.0, holes=0.0):
    """The left edge's depth flickers in every frame; reference from 5 frames before the pick: picked
    within a second of the box going (it waited for the 10 s limit: 'can't tell' on the robot)."""
    face, depth0, _, masks = target_face(FRONT_TARGET, noise, holes, with_masks=True)
    rest = [beside(FRONT_TARGET), WALL]
    there = scene([FRONT_TARGET] + rest, noise, holes, 1)
    refs = [flicker(scene([FRONT_TARGET] + rest, noise, holes, 10 + i), 10 + i) for i in range(4)]
    w = PickWatcher(face, flicker(depth0, 9), masks[0], refs=refs)
    frames = [(there, 6), (scene([FRONT_TARGET, ARM_FRONT] + rest, noise, holes, 2), 10),
              (scene(rest, noise, holes, 3), 400)]
    result, k = feed(w, frames, edge_flicker=True)
    return check("result", result == "picked" and k - 16 <= 30, f"{result} {k - 16} frames after the box went "
                 f"(expected picked within 30)")


@case
def hand_with_box_stays_past_time_limit(noise=0.0, holes=0.0):
    """Box carried up, the arm with it stays in view past SETTLE_S: still no verdict (was picked)."""
    face, depth0, _, masks = target_face(FRONT_TARGET, noise, holes, with_masks=True)
    rest = [beside(FRONT_TARGET), WALL]
    w = PickWatcher(face, depth0, masks[0])
    frames = [(scene([FRONT_TARGET] + rest, noise, holes, 1), 6),
              (scene([FRONT_TARGET, ARM_FRONT] + rest, noise, holes, 2), 10),
              (scene(carried(FRONT_TARGET) + rest, noise, holes, 4), int(1.5 * SETTLE_S * 30))]
    result, _ = feed(w, frames)
    return check("result", result is None, f"{result} (expected None)")


@case
def outline_with_lip_missed(noise=0.0, holes=0.0):
    """Part of the face's outline is not on its plane before the pick (a lip in front): arm comes and
    goes, box still there: missed (it read 'arm in view' for ever: can't tell)."""
    face, _, _, masks = target_face(FRONT_TARGET, noise, holes, with_masks=True)
    rest = [beside(FRONT_TARGET), WALL, LIP]
    depth0 = scene([FRONT_TARGET] + rest, noise, holes, 0)
    w = PickWatcher(face, depth0, masks[0])
    frames = [(scene([FRONT_TARGET] + rest, noise, holes, 1), 6),
              (scene([FRONT_TARGET, ARM_FRONT] + rest, noise, holes, 2), 10),
              (scene([FRONT_TARGET] + rest, noise, holes, 3), 60)]
    result, _ = feed(w, frames)
    return check("result", result == "missed", f"{result} (expected missed; {w.off_plane} face pixels off its plane)")


@case
def arm_out_missed_arm_never_seen(noise=0.0, holes=0.0):
    """The arm never covered the face; the robot says its arm is out: missed at once (it waited for
    the arm for ever: can't tell)."""
    face, depth0, _, masks = target_face(FRONT_TARGET, noise, holes, with_masks=True)
    w = PickWatcher(face, depth0, masks[0])
    there = scene([FRONT_TARGET, beside(FRONT_TARGET), WALL], noise, holes, 1)
    result, k = feed(w, [(there, 60)], arm_out_at=31)
    return check("result", result == "missed" and k <= 31 + STEADY_FRAMES, f"{result} at frame {k} (arm out at 31)")


@case
def arm_out_flap_in_front_moved(noise=0.0, holes=0.0):
    """After the arm, something stays in front of the face, not reaching in from the edge (a flap
    pulled open); the robot says its arm is out: moved (it read 'arm in view' for ever)."""
    face, depth0, _, masks = target_face(FRONT_TARGET, noise, holes, with_masks=True)
    rest = [beside(FRONT_TARGET), WALL]
    w = PickWatcher(face, depth0, masks[0])
    frames = [(scene([FRONT_TARGET] + rest, noise, holes, 1), 6),
              (scene([FRONT_TARGET, ARM_FRONT] + rest, noise, holes, 2), 10),
              (scene([FRONT_TARGET, LIP] + rest, noise, holes, 3), 60)]
    no_robot, _ = feed(PickWatcher(face, depth0, masks[0]), frames)
    result, k = feed(w, frames, arm_out_at=40)
    ok = check("without the robot's word", no_robot is None, f"{no_robot} (expected None: something in front)")
    return ok & check("result", result == "moved" and k <= 40 + 2 * STEADY_FRAMES, f"{result} at frame {k}")


@case
def arm_out_picked_despite_lost_depth_at_edge(noise=0.0, holes=0.0):
    """Box gone, a patch at the picture's left edge loses its depth for good (a cable moved): without
    the robot's word picked only after SETTLE_S; with it (arm out) within a second."""
    face, depth0, _, masks = target_face(FRONT_TARGET, noise, holes, with_masks=True)
    rest = [beside(FRONT_TARGET), WALL]
    gone = scene(rest, noise, holes, 3)
    gone[200:400, :120] = 0
    frames = [(scene([FRONT_TARGET] + rest, noise, holes, 1), 6),
              (scene([FRONT_TARGET, ARM_FRONT] + rest, noise, holes, 2), 10), (gone, 400)]
    slow, k_slow = feed(PickWatcher(face, depth0, masks[0]), frames)
    fast, k_fast = feed(PickWatcher(face, depth0, masks[0]), frames, arm_out_at=20)
    ok = check("without the robot's word", slow == "picked" and k_slow >= 16 + SETTLE_S * 30,
               f"{slow} at frame {k_slow}")
    return ok & check("arm out", fast == "picked" and k_fast <= 20 + 30, f"{fast} at frame {k_fast} (arm out at 20)")


@case
def moved_after_one_second_at_15_fps(noise=0.0, holes=0.0):
    """At 15 frames a second, 'moved' comes 1 s after the arm (MOVED_S), not after 30 frames (2 s)."""
    face, depth0, _, masks = target_face(FRONT_TARGET, noise, holes, with_masks=True)
    rest = [beside(FRONT_TARGET), WALL]
    pushed = box_world(-0.11, 0.45, (0.10, 0.08, 0.04))
    frames = [(scene([FRONT_TARGET] + rest, noise, holes, 1), 6),
              (scene([FRONT_TARGET, ARM_FRONT] + rest, noise, holes, 2), 8), (scene([pushed] + rest, noise, holes, 3), 60)]
    result, k = feed(PickWatcher(face, depth0, masks[0]), frames, fps=15)
    return check("result", result == "moved" and k - 14 <= 16, f"{result} {k - 14} frames after the arm left (15 fps)")


@case
def reopened_miss_then_picked(noise=0.0, holes=0.0):
    """A miss before the robot is done is reopened (it presses again); then the box goes: picked."""
    face, depth0, _, masks = target_face(FRONT_TARGET, noise, holes, with_masks=True)
    rest = [beside(FRONT_TARGET), WALL]
    there = scene([FRONT_TARGET] + rest, noise, holes, 1)
    arm = scene([FRONT_TARGET, ARM_FRONT] + rest, noise, holes, 2)
    w = PickWatcher(face, depth0, masks[0])
    first, _ = feed(w, [(there, 6), (arm, 10), (there, 30)])
    w.reopen(first)
    again, _ = feed(w, [(there, 20)])
    second, _ = feed(w, [(arm, 10), (scene(rest, noise, holes, 3), 60)])
    ok = check("first", first == "missed", f"{first}")
    ok &= check("no second miss without the arm", again is None, f"{again}")
    return ok & check("then", second == "picked", f"{second}")


@case
def arm_out_camera_moved(noise=0.0, holes=0.0):
    """The camera tilts 6 degrees between the pick and the robot's word: camera_moved (can't tell)."""
    face, depth0, _, masks = target_face(FRONT_TARGET, noise, holes, with_masks=True)
    boxes = [FRONT_TARGET, beside(FRONT_TARGET), WALL]
    tilted = render(boxes, noise, holes, seed=3, floor=True, pitch=PITCH + 6)[0]
    result, _ = feed(PickWatcher(face, depth0, masks[0]), [(tilted, 30)], arm_out_at=1)
    return check("result", result == "camera_moved", f"{result}")


def main():
    results = {}
    for label, kw in [("clean", {}), ("noise 1.5 mm + 10% holes", {"noise": 1.5, "holes": 0.10})]:
        print(f"\n== {label} ==")
        for fn in CASES:
            print(f"{fn.__name__}: {fn.__doc__}")
            results[(label, fn.__name__)] = fn(**kw)
    failed = [k for k, ok in results.items() if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} passed" + (f"; failed: {failed}" if failed else ""))
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
