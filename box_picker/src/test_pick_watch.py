"""Checks for pick_watch.py on synthetic depth images (test_faces.py's scenes, no GPU needed).

A target box stands among others in front of the carton's back wall. The arm is a
bar in front of the target's pick face; the box is then taken away, left where it
was, or pushed sideways, and the watcher's verdict is compared with what happened.

Run: python test_pick_watch.py
"""
import numpy as np

from faces import analyse, choose_target, up_from_pitch
from pick_watch import CLEAR_FRAMES, DROP_FRAMES, MOVED_FRAMES, STEADY_FRAMES, STILL_S, PickTracker, PickWatcher
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


def verdict(name, target_box, arm, after, expect, noise, holes, arm_seen=True, extra=(), lifted=None):
    """extra: more boxes, there all along. lifted: what is in view while the box is carried
    off (the arm with the box in the gripper), between the arm and `after`."""
    face, depth0, t, masks = target_face(target_box, noise, holes, extra, with_masks=True)
    ok = check("target", t == 0 and face is not None, f"box {t}, {face.kind if face else None} face")
    if not ok:
        return ok
    w = PickWatcher(face, depth0, masks[0])
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
