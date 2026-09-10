#!/usr/bin/env python3
"""The editable sequence, the saved config, and the web server's routes.

No ROS and no robot: this is about whether a sequence typed into a browser can
reach the orchestrator intact, and what happens to a bad one on the way. The
cycle suite covers the running of it.

    python3 native/tests/test_sequence.py
"""

import asyncio
import json
import os
import sys
import tempfile

WS = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, WS)

import pick_place_sequence as seq            # noqa: E402

import re                                    # noqa: E402


def _included_launch_dirs():
    """Launch directories of the packages our launch files include."""
    roots = []
    for base in (os.path.join(WS, 'native', 'install'),
                 '/opt/ros/humble/share'):
        if not os.path.isdir(base):
            continue
        for package in ('realsense2_camera', 'openarm_bimanual_moveit_config',
                        'isaac_ros_cumotion'):
            folder = os.path.join(base, package, 'share', package, 'launch')
            if not os.path.isdir(folder):
                folder = os.path.join(base, package, 'launch')
            if os.path.isdir(folder):
                roots.append(folder)
    return roots


failures = []


def check(what, got, want):
    try:
        ok = got == want
    except Exception as exc:                          # noqa: BLE001
        ok = False
        got = f'comparison raised: {exc}'
    if ok:
        print(f'pass  {what}')
    else:
        failures.append(what)
        print(f'FAIL  {what}\n        got  {got!r}\n        want {want!r}')


def section(title):
    print(f'\n-- {title} ' + '-' * max(0, 66 - len(title)))


def main():
    section('the default order')
    config = seq.default_config()
    cleaned, problems = seq.validate_sequence(config['sequence'])
    check('the shipped sequence validates without complaint', problems, [])
    check('and survives validation unchanged', cleaned, config['sequence'])
    check('it descends before it grasps',
          cleaned.index('descend') < cleaned.index('grasp'), True)
    check('lifts before it goes to the drop',
          cleaned.index('lift') < cleaned.index('drop'), True)
    check('releases before it verifies the place',
          cleaned.index('release') < cleaned.index('verify_place'), True)
    check('shuts the jaws after releasing, not before',
          cleaned.index('shut_jaws') > cleaned.index('release'), True)
    check('and ends at home', cleaned[-1], 'home')
    # The order asked for, in the words it was asked in: lift with the gripper
    # closed, then pre-pick, then the drop, open, close there, pre-pick, home.
    tail = cleaned[cleaned.index('lift'):]
    check('the requested tail is exactly what ships',
          [s for s in tail if s != 'verify_grasp' and s != 'verify_place'],
          ['lift', 'pre_pick', 'drop', 'release', 'shut_jaws', 'pre_pick',
           'home'])

    section('an edited order')
    check('a step that does not exist is dropped, not run',
          seq.validate_sequence(['descend', 'fly', 'grasp', 'home'])[0],
          ['descend', 'grasp', 'home'])
    check('and the operator is told which',
          any('fly' in p for p in
              seq.validate_sequence(['descend', 'fly', 'grasp', 'home'])[1]),
          True)
    cleaned, problems = seq.validate_sequence(['descend', 'grasp'])
    check('a required step left out is put back', 'home' in cleaned, True)
    check('with a reason', any('required' in p for p in problems), True)
    cleaned, problems = seq.validate_sequence(
        ['descend', 'grasp', 'grasp', 'home'])
    check('a step that cannot repeat is not repeated',
          cleaned.count('grasp'), 1)
    check('but pre_pick can, because it is a waypoint and not an event',
          seq.validate_sequence(
              ['pre_pick', 'descend', 'grasp', 'pre_pick', 'home']
          )[0].count('pre_pick'), 2)

    # The check that makes an editable sequence safe rather than merely
    # possible: a grasp before the descent, or a release holding nothing, is
    # named rather than discovered by the arm.
    _cleaned, problems = seq.validate_sequence(
        ['grasp', 'descend', 'release', 'home'])
    check('grasping before descending is reported',
          any('at_grasp' in p for p in problems), True)
    check('and releasing without holding anything',
          any('holding' in p for p in problems), True)
    _cleaned, problems = seq.validate_sequence(
        ['descend', 'grasp', 'lift', 'release', 'verify_place', 'home'])
    check('a coherent order raises nothing', problems, [])

    # Two kinds of problem, two answers. A correction still runs; an unmet
    # dependency does not, so the UI refuses to save it rather than recording
    # a decision nobody made. An *unusual* order is nobody's business but the
    # operator's -- the point of an editable cycle is to try a different one.
    check('an unmet dependency blocks the save',
          seq.sequence_blockers(['grasp', 'descend', 'release', 'home']),
          ['Close on the object needs at_grasp first, and nothing before it '
           'provides that',
           'Release needs holding first, and nothing before it provides that'])
    check('a dropped unknown name does not block it',
          seq.sequence_blockers(
              seq.validate_sequence(['descend', 'fly', 'grasp', 'home'])[0]),
          [])
    check('nor does a restored required step',
          seq.sequence_blockers(
              seq.validate_sequence(['descend', 'grasp'])[0]), [])
    check('and an unusual but coherent order is allowed through',
          seq.sequence_blockers(
              ['descend', 'grasp', 'shut_jaws', 'lift', 'pre_pick', 'drop',
               'release', 'home']), [])
    check('the shipped order blocks nothing',
          seq.sequence_blockers(seq.DEFAULT_SEQUENCE), [])

    section('the saved config')
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, 'config.json')
        loaded, problems = seq.load_config(path)
        check('a config that is not there is the defaults, not an error',
              loaded['sequence'], list(seq.DEFAULT_SEQUENCE))
        check('and no complaint about it', problems, [])

        why = seq.save_config(path, {
            'prompt': 'detect tape', 'arm': 'left',
            'sequence': ['descend', 'grasp', 'home'],
            'settings': {'gripper_torque_cap': 2.0},
        })
        check('saving reports nothing when it worked', why, None)
        check('and the file is there', os.path.exists(path), True)
        loaded, problems = seq.load_config(path)
        check('what was saved is what comes back',
              loaded['sequence'], ['descend', 'grasp', 'home'])
        check('including the prompt', loaded['prompt'], 'detect tape')
        check('and the settings', loaded['settings'],
              {'gripper_torque_cap': 2.0})

        # A half-written config would mean a bringup with default settings and
        # no explanation, so the write goes through a temporary and a rename.
        check('nothing is left behind from the write',
              sorted(os.listdir(folder)), ['config.json'])

        with open(path, 'w') as handle:
            handle.write('{ this is not json')
        loaded, problems = seq.load_config(path)
        check('a corrupt config falls back to the defaults',
              loaded['sequence'], list(seq.DEFAULT_SEQUENCE))
        check('rather than raising', any('could not read' in p
                                         for p in problems), True)

        with open(path, 'w') as handle:
            json.dump({'settings': {'gripper_torque_cap': 99.0,
                                    'nonsense': 1}}, handle)
        loaded, problems = seq.load_config(path)
        check('a setting past its bound is clamped, not taken',
              loaded['settings']['gripper_torque_cap'],
              seq.SETTINGS_BY_NAME['gripper_torque_cap']['max'])
        check('and one that is not a parameter is refused',
              'nonsense' in loaded['settings'], False)
        check('both said out loud', len(problems), 2)

    section('settings')
    for entry in seq.SETTINGS:
        check(f'{entry["name"]} has something to explain it',
              bool(entry.get('about')), True)
        if entry['type'] != 'bool':
            check(f'{entry["name"]} has a usable range',
                  entry['min'] < entry['max'], True)
    good, why = seq.coerce_setting(
        seq.SETTINGS_BY_NAME['home_requires_pre_pick'], True)
    check('a bool setting takes a bool', (good, why), (True, None))
    bad, why = seq.coerce_setting(
        seq.SETTINGS_BY_NAME['home_requires_pre_pick'], 'yes')
    check('and refuses a string', bad, None)
    check('saying why', 'true or false' in (why or ''), True)
    bad, why = seq.coerce_setting(
        seq.SETTINGS_BY_NAME['velocity_scaling'], float('nan'))
    check('NaN is refused rather than clamped', bad, None)

    section('what the UI is handed')
    described = seq.describe(seq.DEFAULT_SEQUENCE)
    check('the fixed approach is described too',
          [s['step'] for s in described['fixed']],
          ['home', 'locate', 'choose_arm', 'preflight', 'pre_pick', 'transit'])
    check('every fixed entry says why it is fixed',
          all(s.get('about') for s in described['fixed']), True)
    check('the editable steps come back in order',
          [s['step'] for s in described['steps']], list(seq.DEFAULT_SEQUENCE))
    check('and every step the UI can offer has a label',
          all(s.get('label') for s in described['available']), True)
    check('the catalogue offers every implemented step',
          sorted(s['step'] for s in described['available']),
          sorted(seq.STEPS))

    section('the orchestrator implements every step')
    source = open(os.path.join(WS, 'pick_place_orchestrator.py')).read()
    handlers = source.split('STEP_HANDLERS = {')[1].split('}')[0]
    for name in seq.STEPS:
        check(f'{name} is wired to a handler', f"'{name}':" in handlers, True)
        check(f'{name} has a handler defined',
              f'def _step_{name}(' in source, True)
    # And nothing is wired that the UI cannot show, which would be a step the
    # sequence could contain and the editor could never remove.
    wired = [line.split("'")[1] for line in handlers.splitlines()
             if "':" in line]
    check('nothing is wired that the catalogue does not know about',
          sorted(set(wired) - set(seq.STEPS)), [])

    section('a rehearsal setting must not be saved quietly')
    # Measured: a --fake run put grasp_finger_min=-1.0 into the live
    # parameters, the UI's sliders were initialised from the live parameters,
    # and pressing Save wrote it to the config. The next real run would have
    # started with the grasp check off and said nothing -- every close
    # reporting a hold, whatever was between the fingers.
    check('a saved -1.0 grasp threshold is called out',
          len(seq.check_disabling_settings({'grasp_finger_min': -1.0})), 1)
    check('and it says which switch it is',
          'fake-hardware' in
          seq.check_disabling_settings({'grasp_finger_min': -1.0})[0], True)
    check('a place check of zero too',
          len(seq.check_disabling_settings({'object_moved_eps': 0.0})), 1)
    check('ordinary values are not nagged about',
          seq.check_disabling_settings(
              {'grasp_finger_min': 0.003, 'object_moved_eps': 0.05,
               'velocity_scaling': 0.8}), [])
    check('and neither is a config that does not mention them',
          seq.check_disabling_settings({'velocity_scaling': 0.4}), [])
    source = open(os.path.join(WS, 'pick_place_orchestrator.py')).read()
    check('the orchestrator shouts about it at load, before applying it',
          source.index('check_disabling_settings')
          < source.index('applied = self.apply_settings'), True)

    section('the 3D view')
    page = open(os.path.join(WS, 'pick_place_web.html')).read()
    # three.js's example modules import "three" by bare name. Without an
    # import map the browser has nothing to resolve it to, and the failure is
    # a console error rather than a fallback -- measured in the browser as
    # 'Failed to resolve module specifier "three"'.
    check('the page carries an import map', 'type="importmap"' in page, True)
    for bare in ('"three":', '"three/addons/":'):
        check(f'mapping {bare}', bare in page, True)
    check('and the loaders import by the mapped names',
          "await import('three/addons/loaders/STLLoader.js')" in page, True)
    check('with nothing importing a raw CDN URL, which the map cannot help',
          "await import('https://" in page, False)

    section('the detector gives the GPU back')
    node = open(os.path.join(WS, 'VLM', 'vlm_detector_node.py')).read()
    # It used to run a full model.generate every inference_period for ever,
    # whether or not anybody had asked for anything -- measured on the robot
    # as 6.4 GB of VRAM held permanently and the GPU busy between cycles, on a
    # 16 GB card that cuMotion and the octomap also live on. Measured after:
    # 817 MiB idle, 6493 MiB while working, 0.5 s to come back.
    check('on demand is the default',
          "declare_parameter('inference_mode', 'on_demand')" in node, True)
    check('the loop asks whether anything is wanted before inferring',
          'if self._wanted():' in node, True)
    check('and a repeated prompt still counts as a request -- the '
          'orchestrator republishes the same text for every look',
          '_wanted_until = time.time() + self.active_window'
          in node.split('def _on_prompt(')[1].split('\n    def ')[0], True)
    unload = node.split('def _maybe_unload(')[1].split('\n    def ')[0]
    check('idling moves the weights off the GPU',
          "self.model.to('cpu')" in unload, True)
    check('and empties the cache, or the memory is not actually returned',
          'empty_cache' in unload, True)
    check('with the transfer skipped when there is no GPU to give back',
          "!= 'cuda'" in unload, True)
    reload_ = node.split('def _ensure_loaded(')[1].split('\n    def ')[0]
    check('and a prompt brings them back',
          'self.model.to(self.device)' in reload_, True)
    check('the launch layer exposes the mode',
          "'inference_mode', default_value='on_demand'"
          in open(os.path.join(WS, 'pick_place.launch.py')).read(), True)

    section('the model proposes, the pre-flight disposes')
    source = open(os.path.join(WS, 'pick_place_orchestrator.py')).read()
    check('the orchestrator listens for candidates',
          "'/grasp/candidates'" in source, True)
    proposals = source.split('def model_proposals(')[1].split('\n    def ')[0]
    # Stale or misaddressed candidates are worse than none: the server
    # publishes for whatever it was last asked about, and using yesterday's
    # tape to pick today's screwdriver would put the jaws somewhere confident
    # and wrong.
    check('a stale candidate list is refused',
          'grasp_model_max_age' in proposals, True)
    check('and one about a different object too',
          'grasp_model_max_offset' in proposals, True)
    check('candidates come back in the same shape the synthesised grasp has, '
          'so the pre-flight cannot tell them apart',
          all(k in proposals for k in
              ("'grasp':", "'pregrasp':", "'quat':")), True)

    attempt = source.split('def _attempt_pick(')[1].split('\n    def ')[0]
    check('the model is tried before the synthesised pose',
          attempt.index('model_proposals') < attempt.index("'top-down'"),
          True)
    # And the synthesised pose stays on the end rather than being replaced.
    # GraspNet was trained for a 100 mm gripper against this one's 44 mm, so
    # on a wide object every proposal can be one the robot cannot close on --
    # measured on a roll of tape: three candidates needing 55, 66 and 98 mm.
    check('the synthesised pose is the last resort, not the discarded one',
          "'source': 'top-down'" in attempt, True)
    check('and the pre-flight is what chooses between them',
          attempt.count('self.preflight_column(') >= 1
          and 'for index, proposal in enumerate(proposals)' in attempt, True)

    node = open(os.path.join(WS, 'grasp', 'grasp_node.py')).read()
    check('the server refuses grasps this gripper cannot span',
          'gripper_open' in node and 'too_wide' in node, True)
    check('saying so rather than returning an empty list, because "found '
          'grasps you cannot make" is a different problem from "found '
          'nothing"', 'this gripper can span' in node, True)
    # Which part of the object gets grasped follows the VLM's point, so
    # naming a part in the prompt -- "screwdriver handle" -- steers the
    # grasp without any per-object rules.
    check('candidates are filtered to the point the detector located',
          'object_radius' in node, True)

    section('checking the descent that is actually flown')
    source = open(os.path.join(WS, 'pick_place_orchestrator.py')).read()
    recheck = source.split('def descend_orientation(')[1].split('\n    def ')[0]
    # The pre-flight proves the column from the posture it predicts the
    # approach will end in. The arm does not land there. Measured, run
    # 1788948709: TRANSIT arrived 32.6 mm off and the descent the pre-flight
    # had just passed then cost 6.42 rad against a 1.5 rad budget -- a whole
    # approach flown for a descent that was never the one checked.
    check('the re-probe starts from the measured joints, not a prediction',
          '_arm_positions' in recheck, True)
    check('and holds the descent to the same travel budget',
          'column_max_joint_travel' in recheck, True)
    check('the pre-flight\'s own choice is tried first, so a good landing '
          'changes nothing', 'options = [quat]' in recheck, True)
    check('an unanswerable probe leaves the choice alone rather than '
          'guessing', 'nothing could answer' in recheck, True)
    descend = source.split('def _step_descend(')[1].split('\n    def ')[0]
    check('the descent step uses it before committing',
          descend.index('descend_orientation')
          < descend.index('self.descend_column('), True)

    # And the escape holds the octomap exemption for its whole duration.
    # Measured, same run: the descent was refused, the tool was still at
    # transit height with the jaws open above the object, clear_the_surface
    # saw it as clear and released the column -- handing the gripper back to
    # collision checking against the voxels it was standing in. Both refuges
    # then returned -2 in under a second, not because they were unreachable
    # but because the arm's own start state had been declared invalid.
    shutdown = source.split('def safe_shutdown(')[1].split('\n    def ')[0]
    check('the escape keeps the gripper out of the octomap while escaping it',
          'allow_gripper_in_octomap(True)' in shutdown, True)
    check('and gives it back afterwards, whatever happened',
          'finally:' in shutdown, True)
    check('with the refuge search in its own method, inside that guard',
          '_reach_a_refuge' in shutdown, True)

    section('where the time went')
    source = open(os.path.join(WS, 'pick_place_orchestrator.py')).read()
    # Measured on the robot, run 1788949762: an 85-second gap between the
    # first look and the second, and a 45-second one on the next cycle --
    # both of them choose_arm probing reachability. The local solver runs 48
    # seeded damped-least-squares solves and takes 2.9 s whether it finds
    # anything or not (timed offline against the real description), and
    # choose_arm called it three orientations x three heights x two arms.
    check('the arm choice can stop at KDL',
          "def reachable(self, position, quat, arm=None, orientations=None,"
          in source and 'cheap=False' in source, True)
    reach = source.split('def reachable(')[1].split('\n    def ')[0]
    check('and cheap means unknown rather than unreachable -- "cannot tell" '
          'is not "no"',
          'if cheap:' in reach and 'return None' in reach.split('if cheap:')[1]
          .split('\n\n')[0], True)
    check('the expensive solver sits after that gate',
          reach.index('if cheap:') < reach.index('self.kinematics(arm)'), True)
    check('choose_arm uses it', 'arm_choice_cheap' in source, True)
    # Cheap alone would lose the case this method exists for -- an object
    # only the *far* arm can reach, where KDL's "cannot tell" leaves the near
    # arm selected and the pick refused. So the slow solvers are still asked,
    # but only when the quick answer settles nothing.
    choose = source.split('def choose_arm(')[1].split('\n    def ')[0]
    check('and falls through to the slow solvers when nothing is confirmed',
          'asking the slow solvers' in choose, True)
    quick = source.split('def _arm_that_reaches(')[1].split('\n    def ')[0]
    check('the quick pass reports "not confirmed" apart from "fatal", so a '
          'missing states file does not go round the slow path',
          'return False, states' in quick and 'return True, None' in quick,
          True)
    check('and it is on by default',
          "declare_parameter('arm_choice_cheap', True)" in source, True)
    # The other half: an answer nobody acts on is not worth paying for. The
    # per-attempt reach check is skipped outright when the pre-flight will
    # run, because the pre-flight is the authority and the check was made
    # advisory -- it was still costing a planning request per orientation per
    # target to produce something that got logged and ignored.
    attempt = source.split('def _attempt_pick(')[1].split('\n    def ')[0]
    check('the advisory reach check is not run when the pre-flight will be',
          "not self.get_parameter('preflight_descent').value" in attempt, True)

    section('the whole path, before the first move')
    source = open(os.path.join(WS, 'pick_place_orchestrator.py')).read()
    preflight = source.split('def preflight_column(')[1].split('\n    def ')[0]
    # Down and back up. The way out is part of the path, and it was not
    # checked at all: measured, a cycle whose descent and grasp both passed
    # and whose LIFT was then refused twice, at 2.97 and 3.01 rad against a
    # 1.5 rad budget -- with the object already in the jaws at the bottom of
    # the column, which is the worst place to find out.
    for leg in ("'grasp'", "'lift'"):
        check(f'the pre-flight proves the {leg} leg', leg in preflight, True)
    check('the lift is chained after the grasp, the way it is flown',
          preflight.index("wanted_legs.append(('grasp'")
          < preflight.index("wanted_legs.append(('lift'"), True)
    check('to the pre-grasp height, which is what lift_column falls back to',
          "pregrasp[2]))" in preflight, True)
    check('and the whole column is held to the travel budget',
          'column_max_joint_travel' in preflight, True)
    check('while the posture it proves them from is checked as reachable',
          'posture_is_gettable' in preflight, True)

    section('the failure path')
    source = open(os.path.join(WS, 'pick_place_orchestrator.py')).read()
    # A failed cycle does not leave the way a successful one does. Measured,
    # run 1788933180: the lift was refused for joint travel and settled 50 mm
    # up, PRE_PICK and DROP both came back -2 with the gripper in the
    # octomap, and the arm was then driven home anyway -- still holding the
    # object, as a joint goal across the workspace.
    check('every failure goes through the safe shutdown',
          source.count("self.safe_shutdown(states, 'after a failed pick')")
          + source.count("self.safe_shutdown(states, 'after a failed place')"),
          3)
    check('and a crash does too',
          "self.safe_shutdown(self.load_states()," in source, True)
    check('while a successful cycle still leaves by the ordinary retreat',
          "self._retreat_to_home(ctx['states']" in source, True)

    shutdown = source.split('def safe_shutdown(')[1].split('\n    def ')[0]
    # The refuge search moved into its own method so the octomap exemption
    # could wrap all of it, so the assertions about *choosing* a refuge
    # belong to that one and the ones about the escape as a whole stay here.
    refuge = source.split('def _reach_a_refuge(')[1].split('\n    def ')[0]
    check('it lifts clear of the surface first',
          'self.clear_the_surface' in shutdown, True)
    check('it asks the collision world before driving to a refuge',
          'self.posture_is_clear' in refuge, True)
    check('the check happens before the move, not after',
          'posture_is_clear' in refuge and 'move_to_state' in refuge
          and refuge.index('posture_is_clear')
          < refuge.index('move_to_state'), True)
    check('pre_pick is tried before home, being the nearer refuge',
          'PRE_PICK_STATE' in refuge and 'home_positions' in refuge
          and refuge.index('PRE_PICK_STATE')
          < refuge.index('home_positions'), True)
    # The rule that keeps "motors off" from being the dangerous option: an
    # arm that could not reach anywhere safe is stranded, and a stranded arm
    # going limp falls onto whatever it is over.
    check('nothing is disengaged when no refuge was reached',
          shutdown.index('no refuge could be reached')
          < shutdown.index('disengage_on_failure'), True)
    check('and the no-refuge branch returns before it',
          'return False' in shutdown.split('no refuge could be reached')[1]
          .split('disengage_on_failure')[0], True)

    validity = source.split('def posture_is_clear(')[1].split('\n    def ')[0]
    check('an unanswerable validity check is not read as "occupied"',
          'return None' in validity, True)
    check('and the contacts are named, not just the verdict',
          'contact_body_1' in validity, True)

    disengage = source.split('def disengage_motors(')[1].split('\n    def ')[0]
    check('disengaging deactivates the hardware component, which is what '
          'openarm_hardware disables the motors on',
          'PRIMARY_STATE_INACTIVE' in disengage, True)
    check('both arms are taken off, not just the working one',
          disengage.count('_hardware_interface'), 2)
    # Matched on the docstring: the warning itself is split across two
    # f-string fragments, so the sentence it prints does not appear
    # contiguously in the source.
    check('and it says the arm is then held up by nothing',
          'held up by nothing' in disengage, True)

    check('the toggle is settable from the browser',
          'disengage_on_failure' in seq.SETTINGS_BY_NAME, True)
    check('and it is a switch rather than a slider',
          seq.SETTINGS_BY_NAME['disengage_on_failure']['type'], 'bool')

    section('activation does not move the arm')
    source = open(os.path.join(WS, 'pick_place_orchestrator.py')).read()
    page = open(os.path.join(WS, 'pick_place_web.html')).read()
    # There was an Engage button, and it is gone on purpose. Putting the
    # motors back on is a power operation; hanging a trip home off it made
    # a button that moved the arm, and the useful half -- activation not
    # jumping -- belongs in the hardware, where it now is. Asserted so it
    # does not quietly come back.
    check('no engage service',
          "'/pick_place/engage'" in source, False)
    check('no engage_motors', 'def engage_motors(' in source, False)
    check('no button on the page', 'btn-engage' in page, False)
    check('and the panel cannot ask for it',
          'engage' in open(os.path.join(WS, 'pick_place_web.py')).read()
          .split('ACTIONS = {')[1].split('}')[0], False)

    # What remains is the part that was always right: activating a hardware
    # component must not drive the arm anywhere. on_activate used to call
    # return_to_zero(), which interpolates every joint to zero over 2.4
    # seconds -- unplanned, from wherever the arm had sagged to, because
    # these are direct-drive motors with no brakes. That happens at every
    # bringup, not just at the press of a button, so it matters more now
    # rather than less.
    hardware = open(os.path.join(
        WS, 'src', 'openarm_ros2', 'openarm_hardware', 'src',
        'v10_simple_hardware.cpp')).read()
    activate = hardware.split('OpenArm_v10HW::on_activate(')[1].split(
        '\nhardware_interface::CallbackReturn')[0]
    check('activating holds the posture the arm is already in',
          'hold_current_position();' in activate, True)
    check('and does not drive it to zero',
          'return_to_zero();' in activate, False)
    hold = hardware.split('void OpenArm_v10HW::hold_current_position()')[1] \
        .split('\nvoid OpenArm_v10HW::')[0]
    check('by seeding the command buffers from the measured state -- a '
          'command left at its resize() default of 0.0 is the jump',
          'pos_commands_[i] = pos_states_[i];' in hold, True)
    check('the gripper too, or the jaws snap shut on activation',
          'pos_commands_[ARM_DOF]' in hold, True)

    # And the reason a finished cycle left the Pick button dead: every status
    # the browser has comes from _set_state, and the terminal state is set
    # before the busy flag is cleared -- so the last thing it heard was
    # busy=true.
    # Sliced by the method, not by the first 'finally:' in the file -- there
    # are several, and the first one belongs to something else entirely.
    runner = source.split('def _run_cycle(')[1].split('\n    def ')[0]
    check('the busy flag is published after it is cleared, not before',
          'self._busy = False' in runner and 'self.publish_status()' in runner
          and runner.index('self._busy = False')
          < runner.index('self.publish_status()'), True)

    section('launch arguments do not collide with the ones we include')
    # This is not hypothetical. Adding a launch argument called `config_file`
    # took the entire bringup down with
    #
    #   FileNotFoundError: [Errno 2] No such file or directory:
    #                      'pick_place_config.json'
    #
    # because realsense2_camera's rs_launch.py declares an argument of that
    # name and opens it as YAML -- and a LaunchConfiguration set at the top
    # reaches every included launch file, whether or not it was passed to it.
    # Any argument we add can do this to any package we include, and the
    # failure names our file rather than their argument, so it reads as our
    # bug in a way that points nowhere useful.
    ours = set()
    for name in ('pick_place_demo.launch.py', 'pick_place.launch.py',
                 'launch_everything.launch.py'):
        text = open(os.path.join(WS, name)).read()
        ours.update(re.findall(
            r"DeclareLaunchArgument\(\s*'([a-z0-9_]+)'", text))
        ours.update(m[0] for m in re.findall(
            r"^\s*\('([a-z0-9_]+)', '", text, re.M))

    theirs = {}
    for launch_dir in _included_launch_dirs():
        for entry in sorted(os.listdir(launch_dir)):
            if not entry.endswith('.py'):
                continue
            try:
                text = open(os.path.join(launch_dir, entry)).read()
            except OSError:
                continue
            for found in re.findall(
                    r"[\'\"]name[\'\"]:\s*[\'\"]([a-z0-9_]+)[\'\"]", text):
                theirs.setdefault(found, entry)
            for found in re.findall(
                    r"DeclareLaunchArgument\(\s*[\'\"]([a-z0-9_]+)", text):
                theirs.setdefault(found, entry)

    if not theirs:
        print('      skipped: no included launch files found to check against')
    else:
        # The ones we pass deliberately are not collisions -- they are the
        # interface. Everything else sharing a name is an accident.
        intended = {'use_fake_hardware', 'right_can_interface',
                    'left_can_interface', 'octomap', 'tool_frame',
                    'collision_activation_distance', 'robot_description',
                    'use_sim_time', 'arm', 'prompt', 'model_id', '4d'}
        clashes = sorted((name, theirs[name]) for name in ours & set(theirs)
                         if name not in intended)
        check('no launch argument of ours shadows one we include', clashes, [])

    section('the web server')
    try:
        import pick_place_web
    except ImportError as exc:
        print(f'      skipped: {exc}')
    else:
        check('every action the page can press has a service behind it',
              sorted(pick_place_web.ACTIONS),
              ['abort', 'grip', 'open_gripper', 'reload_config',
               'save_config', 'start'])
        # The browser asks for whatever the URDF names, so the mesh route is a
        # path traversal unless the package root is checked. It is checked.
        for package, tail, want in (
                ('openarm_description', '../../../etc/passwd', 403),
                ('openarm_description', '..%2f..%2fetc/passwd', 404),
                ('nope', 'meshes/a.stl', 404)):
            request = _FakeRequest({'package': package, 'path': tail})
            response = asyncio.run(pick_place_web.mesh(request))
            check(f'{package}/{tail} is refused', response.status, want)
        real = os.path.join('meshes', 'ee', 'openarm_hand', 'visual',
                            'hand.dae')
        if os.path.exists(os.path.join(
                pick_place_web.MESH_ROOTS['openarm_description'], real)):
            request = _FakeRequest({'package': 'openarm_description',
                                    'path': real})
            response = asyncio.run(pick_place_web.mesh(request))
            check('but a real mesh is served', response.status, 200)

        page = open(os.path.join(WS, 'pick_place_web.html')).read()
        check('the page asks for the word rather than a sentence',
              'just the word' in page, True)
        check('and no longer suggests "pick up the"',
              'pick up the' in page, False)
        check('the flow chart highlights on the published step, not the '
              'state string', "LIVE.step" in page, True)
        # A still, not a stream. The interesting frame is the one the
        # detector decided from, and it stays interesting for the minute the
        # cycle then takes -- pushing it thirty times a second showed the
        # same tabletop over and over.
        check('the camera is a still, taken when the detection lands',
              "'/camera.jpg'" in page, True)
        check('and nothing streams it', 'camera.mjpg' in page, False)
        check('the settings live under their own tab with the workflow',
              "data-page=\"setup\"" in page and 'page-setup' in page, True)
        check('and the running cycle is still visible on the run tab',
              "id=\"flow-live\"" in page, True)

    print()
    if failures:
        print(f'{len(failures)} check(s) failed: ' + ', '.join(failures))
        return 1
    print('the sequence survives being edited, and a bad edit says why')
    return 0


class _FakeRequest:
    """Just enough of aiohttp's request for the route under test."""

    def __init__(self, match_info):
        self.match_info = match_info


if __name__ == '__main__':
    sys.exit(main())
