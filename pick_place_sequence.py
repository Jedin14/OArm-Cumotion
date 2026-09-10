"""The pick-and-place cycle as data, so the order can be changed without code.

The sequence from the gripper opening above the object to the arm back at home
is a list of independent steps. It lives in a JSON file, the web UI edits it,
and pick_place_orchestrator walks it. Changing the order there changes what the
arm does on the next cycle -- which is the point: that order was edited three
times in one afternoon and each edit was a code change.

What is *not* in here, deliberately:

  HOME, LOCATE, PREFLIGHT, PRE_PICK, TRANSIT

Those are the approach, and they are one unit rather than a list. The pre-flight
proves the whole vertical column -- the line down to the pre-grasp and on to the
grasp -- from the staging posture, before anything moves, and it picks the arm
configuration that makes that column flyable. Reordering the things it depends
on would not produce a different cycle, it would produce an unchecked one. The
UI shows them, greyed, with that reason attached.

Everything from `open_gripper` on is genuinely a list, and is genuinely
reorderable.
"""

import copy
import json
import os


# Every step the sequence may contain.
#
#   needs      what must already be true, checked when the sequence is
#              validated rather than discovered halfway through a cycle
#   gives      what running it establishes
#   required   cannot be removed: the cycle has no meaning without it
#   repeatable may appear more than once (pre_pick is a waypoint, not an event)
#
# `needs`/`gives` are what stop an edited sequence from being nonsense -- a
# `lift` before a `descend`, or a `drop` while holding nothing -- without
# hard-coding the one true order.
STEPS = {
    'open_gripper': {
        'label': 'Open gripper',
        'about': 'Open the jaws. Above the object, so nothing catches.',
        'needs': [], 'gives': ['open'],
        'required': False, 'repeatable': True,
    },
    'descend': {
        'label': 'Descend',
        'about': 'One straight line down the column to the grasp height.',
        'needs': [], 'gives': ['at_grasp'],
        'required': True, 'repeatable': False,
    },
    'grasp': {
        'label': 'Close on the object',
        'about': 'Step the jaws shut until the torque cap, and report '
                 'whether anything ended up between them.',
        'needs': ['at_grasp'], 'gives': ['holding'],
        'required': True, 'repeatable': False,
    },
    'lift': {
        'label': 'Lift',
        'about': 'Straight back up the same column, clear of the surface.',
        'needs': ['at_grasp'], 'gives': ['clear'],
        'required': False, 'repeatable': False,
    },
    'verify_grasp': {
        'label': 'Verify grasp',
        'about': 'Fingers plus a fresh look: is the object held, and is it '
                 'gone from where it was?',
        'needs': ['holding'], 'gives': [],
        'required': False, 'repeatable': False,
    },
    'pre_pick': {
        'label': 'Pre-pick pose',
        'about': 'The recorded staging posture. Above the work surface and '
                 'reachable from both ends, which is what makes it the safe '
                 'waypoint.',
        'needs': [], 'gives': ['clear'],
        'required': False, 'repeatable': True,
    },
    'drop': {
        'label': 'Drop pose',
        'about': 'The recorded release posture.',
        'needs': [], 'gives': ['at_drop'],
        'required': False, 'repeatable': True,
    },
    'release': {
        'label': 'Release',
        'about': 'Open the jaws and let the object go.',
        'needs': ['holding'], 'gives': ['released', 'open'],
        'required': False, 'repeatable': False,
    },
    'verify_place': {
        'label': 'Verify place',
        'about': 'Look for the object where it was let go.',
        'needs': ['released'], 'gives': [],
        'required': False, 'repeatable': False,
    },
    'shut_jaws': {
        'label': 'Shut the jaws',
        'about': 'Close the empty gripper so the trip back is made with a '
                 'narrow profile rather than 44 mm of open fingers.',
        'needs': [], 'gives': [],
        'required': False, 'repeatable': True,
    },
    'home': {
        'label': 'Home',
        'about': 'The rest and observation pose. A joint goal to a folded '
                 'posture, so it is only safe from the staging pose -- see '
                 'home_requires_pre_pick.',
        'needs': [], 'gives': [],
        'required': True, 'repeatable': True,
    },
}

# The approach, shown but not edited. See the module docstring.
FIXED_PREFIX = [
    {'step': 'home', 'label': 'Home',
     'about': 'Rest, observe and capture the octomap, out of the camera frame.'},
    {'step': 'locate', 'label': 'Locate',
     'about': 'Ask the detector where the object is.'},
    {'step': 'choose_arm', 'label': 'Choose arm',
     'about': 'Which arm can reach it, settled before anything moves.'},
    {'step': 'preflight', 'label': 'Pre-flight',
     'about': 'Prove the whole descent column from the staging posture, and '
              'choose the arm configuration that makes it flyable. This is '
              'why the approach is not reorderable: the steps below it are '
              'checked as one unit.'},
    {'step': 'pre_pick', 'label': 'Pre-pick pose',
     'about': 'The staging posture the pre-flight measured from.'},
    {'step': 'transit', 'label': 'Transit',
     'about': 'A straight line to a point above the object.'},
]

DEFAULT_SEQUENCE = [
    'open_gripper',
    'descend',
    'grasp',
    'lift',
    'verify_grasp',
    'pre_pick',
    'drop',
    'release',
    'verify_place',
    'shut_jaws',
    'pre_pick',
    'home',
]

# Orchestrator parameters the UI is allowed to set, with the bounds it offers.
# Anything not here stays a launch argument: these are the ones worth reaching
# for between runs, and every one of them has been reached for this week.
SETTINGS = [
    {'name': 'gripper_torque_cap', 'label': 'Grip torque cap', 'unit': 'Nm',
     'type': 'double', 'min': 0.5, 'max': 5.0, 'step': 0.1,
     'about': 'What bounds the grip. The hardware drives the gripper as a '
              'position command with a fixed gain, so nothing else does.'},
    {'name': 'grasp_z_offset', 'label': 'Grasp height offset', 'unit': 'm',
     'type': 'double', 'min': -0.02, 'max': 0.05, 'step': 0.001,
     'about': 'Added to the detected top of the object. Positive stops the '
              'tool above it. Raise if the arm presses into the surface, '
              'lower if the jaws close above the object.'},
    {'name': 'grasp_finger_min', 'label': 'Holding threshold', 'unit': 'm',
     'type': 'double', 'min': -1.0, 'max': 0.02, 'step': 0.001,
     'about': 'Measured finger opening above which the jaws count as '
              'holding something. -1 disables the check, for fake hardware.'},
    {'name': 'velocity_scaling', 'label': 'Speed', 'unit': '',
     'type': 'double', 'min': 0.05, 'max': 1.0, 'step': 0.05,
     'about': 'A time dilation of the planned path, so it changes speed and '
              'not the route. Every leg scales with it.'},
    {'name': 'acceleration_scaling', 'label': 'Acceleration', 'unit': '',
     'type': 'double', 'min': 0.05, 'max': 1.0, 'step': 0.05,
     'about': 'The planner applies the lower of this and the speed.'},
    {'name': 'approach_height', 'label': 'Pre-grasp height', 'unit': 'm',
     'type': 'double', 'min': 0.01, 'max': 0.20, 'step': 0.005,
     'about': 'How far above the grasp the gripper opens.'},
    {'name': 'transit_height', 'label': 'Transit height', 'unit': 'm',
     'type': 'double', 'min': 0.05, 'max': 0.40, 'step': 0.01,
     'about': 'Height above the grasp at which the free-space move ends and '
              'the straight-line descent begins.'},
    {'name': 'retreat_height', 'label': 'Lift height', 'unit': 'm',
     'type': 'double', 'min': 0.02, 'max': 0.40, 'step': 0.01,
     'about': 'How far the lift tries to rise before settling for less.'},
    {'name': 'column_max_joint_travel', 'label': 'Column travel budget',
     'unit': 'rad', 'type': 'double', 'min': 0.2, 'max': 6.0, 'step': 0.1,
     'about': 'Refuse a column leg that costs more than this on one joint. '
              'A line can solve completely and still sweep the arm across '
              'the workspace; legs that work cost about 0.5 rad.'},
    {'name': 'home_requires_pre_pick', 'label': 'Home only via pre-pick',
     'unit': '', 'type': 'bool',
     'about': 'Refuse HOME when the staging pose cannot be reached, rather '
              'than sweeping the arm across the work surface to get there.'},
    {'name': 'contact_torque_margin', 'label': 'Contact torque margin',
     'unit': 'Nm', 'type': 'double', 'min': 0.0, 'max': 12.0, 'step': 0.25,
     'about': 'Stop a move when a joint pulls this much harder than it was '
              'pulling a moment ago -- and is also falling behind its own '
              'trajectory, which is what separates an obstacle from the '
              "arm's own weight. Then back off to where it was two seconds "
              'earlier. 0 disables it, and fake hardware reports no torques '
              'so it never fires there.'},
    {'name': 'contact_lag_rad', 'label': 'Contact lag threshold',
     'unit': 'rad', 'type': 'double', 'min': 0.0, 'max': 1.0, 'step': 0.05,
     'about': 'How far behind its trajectory the arm has to fall before '
              'high torque counts as contact. The worst honest tracking '
              'error measured at full speed was 0.126 rad.'},
    {'name': 'regrasp_attempts', 'label': 'Re-grasp attempts', 'unit': '',
     'type': 'double', 'min': 1, 'max': 6, 'step': 1,
     'about': 'How many times to go back down and try the same grasp when '
              'the detector says the object never moved.'},
    {'name': 'disengage_on_failure', 'label': 'Motors off after a failure',
     'unit': '', 'type': 'bool',
     'about': 'Once a failed cycle has parked the arm somewhere the collision '
              'world says is free, take the motors off. These are '
              'direct-drive with no brakes, so the arm is held up by nothing '
              'afterwards -- it is never done with the arm stranded, where '
              'letting go would drop it onto whatever it is over.'},
    {'name': 'descend_close_gap', 'label': 'Close the descent gap',
     'unit': '', 'type': 'bool',
     'about': 'Fly the remainder when the descent stops short of the grasp. '
              'Off by default -- no measured grasp has needed it, and it '
              'drives the tool toward the table.'},
]

SETTINGS_BY_NAME = {entry['name']: entry for entry in SETTINGS}

# Values that switch a check off rather than tune it. These belong to a
# rehearsal on fake hardware and nowhere else, and they reach the config file
# by accident: --fake sets them as launch arguments, the UI's sliders are
# initialised from the live parameters, and pressing Save writes what the
# sliders show. So a rehearsal leaves a saved config that disables the grasp
# check, and the next real run starts with it off and says nothing.
#
# Saying something is the whole of the fix. Refusing to load them would be
# wrong -- somebody rehearsing wants them -- but they are never allowed to be
# quiet.
DISABLES_A_CHECK = {
    'grasp_finger_min': (
        lambda v: v is not None and v < 0.0,
        'the grasp check is off: every close will report holding, whatever '
        'is between the fingers. This is the fake-hardware rehearsal value '
        '(--fake sets it) and it must not be used on the arms.'),
    'object_moved_eps': (
        lambda v: v is not None and v <= 0.0,
        'the place check is off: a place will be confirmed even with the '
        'object still sitting where it was picked up. Another rehearsal '
        'value.'),
}


def check_disabling_settings(settings):
    """Which saved settings switch a check off. One string each, or []."""
    found = []
    for name, (predicate, why) in DISABLES_A_CHECK.items():
        if name in (settings or {}) and predicate(settings[name]):
            found.append(f'{name}={settings[name]!r}: {why}')
    return found


def default_config():
    """A whole config: what the first run gets, and what a reset returns to."""
    return {
        'prompt': 'tape',
        'arm': 'auto',
        'sequence': list(DEFAULT_SEQUENCE),
        'settings': {},
    }


def validate_sequence(steps):
    """Clean a sequence and say what is wrong with it.

    Returns (cleaned, problems). Problems are for the operator to read; the
    cleaned list is always runnable, because an editor that can save a
    sequence the arm then refuses halfway through is worse than one that
    corrects it and says so.
    """
    problems = []
    cleaned, seen = [], set()
    for name in steps or []:
        if name not in STEPS:
            problems.append(f'{name!r} is not a step; dropped')
            continue
        if name in seen and not STEPS[name]['repeatable']:
            problems.append(f'{STEPS[name]["label"]} cannot run twice; '
                            f'the later one was dropped')
            continue
        cleaned.append(name)
        seen.add(name)

    for name, meta in STEPS.items():
        if meta['required'] and name not in seen:
            problems.append(f'{meta["label"]} is required; restored at the end')
            cleaned.append(name)

    # Ordering, by what each step needs rather than by a fixed list.
    #
    # A step whose needs were not met is not credited with what it gives: a
    # grasp before the descent has not grasped anything, so the release after
    # it is a second fault and not a consequence of the first. Crediting it
    # anyway reported one problem where there were two, and the second is the
    # one that opens the jaws over the table.
    have, ordered = set(), []
    for name in cleaned:
        missing = [n for n in STEPS[name]['needs'] if n not in have]
        if missing:
            problems.append(
                f'{STEPS[name]["label"]} needs {", ".join(missing)} first, '
                f'which nothing before it provides')
        else:
            have.update(STEPS[name]['gives'])
        ordered.append(name)
    return ordered, problems


def sequence_blockers(steps):
    """The problems that make a sequence incoherent rather than merely odd.

    Two kinds of problem come out of validate_sequence and they deserve
    different answers. A dropped unknown name, a restored required step, a
    de-duplicated one -- those are corrections, and a sequence with them
    applied still runs. An unmet dependency is not a correction: `grasp`
    before `descend` closes the jaws in mid-air, and no amount of cleaning
    makes that the order somebody meant.

    So the UI refuses to save these and says which, while leaving every
    *unusual but coherent* order available. The point of an editable cycle is
    to be able to try a different one, not to be stopped from it.
    """
    have, blocking = set(), []
    for name in steps or []:
        meta = STEPS.get(name)
        if meta is None:
            continue
        missing = [n for n in meta['needs'] if n not in have]
        if missing:
            blocking.append(
                f'{meta["label"]} needs {", ".join(missing)} first, and '
                f'nothing before it provides that')
        else:
            have.update(meta['gives'])
    return blocking


def load_config(path):
    """Read the saved config, falling back to the defaults for anything absent.

    Never raises. A settings file that cannot be read is a reason to start
    from the defaults and say so, not a reason for the robot not to come up.
    """
    config = default_config()
    problems = []
    if not path or not os.path.exists(path):
        return config, problems
    try:
        with open(path) as handle:
            saved = json.load(handle)
    except (OSError, ValueError) as exc:
        return config, [f'could not read {path}: {exc}']
    if not isinstance(saved, dict):
        return config, [f'{path} does not hold an object; ignored']

    if isinstance(saved.get('prompt'), str):
        config['prompt'] = saved['prompt']
    if saved.get('arm') in ('auto', 'left', 'right'):
        config['arm'] = saved['arm']
    if 'sequence' in saved:
        config['sequence'], problems = validate_sequence(saved['sequence'])
    settings = saved.get('settings')
    if isinstance(settings, dict):
        for name, value in settings.items():
            entry = SETTINGS_BY_NAME.get(name)
            if entry is None:
                problems.append(f'{name!r} is not a settable parameter; ignored')
                continue
            coerced, why = coerce_setting(entry, value)
            if why:
                problems.append(why)
            if coerced is not None:
                config['settings'][name] = coerced
    return config, problems


def coerce_setting(entry, value):
    """One setting, forced into range. Returns (value, complaint-or-None)."""
    if entry['type'] == 'bool':
        if isinstance(value, bool):
            return value, None
        return None, f'{entry["name"]} wants true or false, got {value!r}'
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None, f'{entry["name"]} wants a number, got {value!r}'
    if number != number:                       # NaN
        return None, f'{entry["name"]} cannot be NaN'
    low, high = entry['min'], entry['max']
    if number < low or number > high:
        clamped = min(max(number, low), high)
        return clamped, (f'{entry["name"]} {number:g} is outside '
                         f'{low:g}..{high:g}; using {clamped:g}')
    return number, None


def savable_settings(settings):
    """The settings minus any that switch a check off. (kept, dropped).

    A rehearsal value must never reach this file, because the file is read
    by every run afterwards -- including the ones with arms on. Measured,
    run 1789014831: a --fake session saved grasp_finger_min=-1.0, the next
    real run loaded it, and the jaws then stalled at 1.08 mm with 0.50 Nm --
    empty -- and reported a hold. The cycle carried nothing to the drop
    point and finished DONE.

    Dropped here rather than refused, so saving a sequence still works from
    a rehearsal; what is dropped is named for the operator.
    """
    kept, dropped = {}, []
    for name, value in (settings or {}).items():
        rule = DISABLES_A_CHECK.get(name)
        if rule is not None and rule[0](value):
            dropped.append(f'{name}={value!r} was not saved: {rule[1]}')
            continue
        kept[name] = value
    return kept, dropped


def save_config(path, config):
    """Write the config, atomically. Returns None or the reason it failed.

    Atomically because this file is read at every bringup: a half-written one
    would mean the robot starts with default settings and no explanation.
    """
    settings, _dropped = savable_settings(config.get('settings') or {})
    body = {
        'prompt': config.get('prompt', ''),
        'arm': config.get('arm', 'auto'),
        'sequence': list(config.get('sequence') or DEFAULT_SEQUENCE),
        'settings': settings,
    }
    temporary = f'{path}.writing'
    try:
        with open(temporary, 'w') as handle:
            json.dump(body, handle, indent=2, sort_keys=True)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except OSError as exc:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        return f'could not write {path}: {exc}'
    return None


def describe(sequence):
    """The sequence as the UI wants it: the fixed approach, then the steps."""
    return {
        'fixed': copy.deepcopy(FIXED_PREFIX),
        'steps': [dict(STEPS[name], step=name) for name in sequence
                  if name in STEPS],
        'available': [dict(meta, step=name) for name, meta in STEPS.items()],
    }
