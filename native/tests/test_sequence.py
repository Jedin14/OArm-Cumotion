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
import shutil
import subprocess
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
    # A cycle ends where the next one starts, so that a Pick pressed
    # afterwards begins with the transit rather than by folding out of
    # HOME and back.
    check('and ends at the staging pose', cleaned[-1], 'pre_pick')
    # The order asked for, in the words it was asked in: lift with the gripper
    # closed, then pre-pick, then the drop, open, close there, pre-pick.
    tail = cleaned[cleaned.index('lift'):]
    check('the requested tail is exactly what ships',
          [s for s in tail if s != 'verify_grasp' and s != 'verify_place'],
          ['lift', 'pre_pick', 'drop', 'release', 'shut_jaws', 'pre_pick'])

    section('an edited order')
    check('a step that does not exist is dropped, not run',
          seq.validate_sequence(['descend', 'fly', 'grasp', 'home'])[0],
          ['descend', 'grasp', 'home'])
    check('and the operator is told which',
          any('fly' in p for p in
              seq.validate_sequence(['descend', 'fly', 'grasp', 'home'])[1]),
          True)
    cleaned, problems = seq.validate_sequence(['open_gripper', 'lift'])
    check('a required step left out is put back', 'grasp' in cleaned, True)
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

    section('where Pick stops and Place starts')
    # One rule, two readers. sequence_split in the orchestrator and
    # place_from here have to agree, or the panel labels steps the Place
    # button will not run.
    check('the split is after verify_grasp when there is one',
          seq.place_from(list(seq.DEFAULT_SEQUENCE)),
          seq.DEFAULT_SEQUENCE.index('verify_grasp') + 1)
    check('after the grasp when there is not',
          seq.place_from(['open_gripper', 'grasp', 'lift', 'drop']), 2)
    check('and past the end when nothing follows the grasp',
          seq.place_from(['open_gripper', 'descend', 'grasp']), 3)
    orchestrator_source = open(
        os.path.join(WS, 'pick_place_orchestrator.py')).read()
    split = orchestrator_source.split(
        'def sequence_split(')[1].split('\n    def ')[0]
    check('the orchestrator splits on the same anchors',
          [a for a in seq.PLACE_ANCHORS if repr(a) in split
           or f"'{a}'" in split], list(seq.PLACE_ANCHORS))
    check('and the panel is told, rather than guessing',
          'place_from' in seq.describe(list(seq.DEFAULT_SEQUENCE)), True)

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
    # In the order it actually happens: the arm waits at the staging pose,
    # so a cycle begins by looking rather than by folding down to HOME.
    # HOME is still there because a look sometimes has to be taken from out
    # of the camera frame, but it is a detour now, not the first step.
    check('the fixed approach is described too',
          [s['step'] for s in described['fixed']],
          ['pre_pick', 'locate', 'home', 'choose_arm', 'preflight',
           'pre_pick', 'transit'])
    check('and it starts where the arm is actually waiting',
          described['fixed'][0]['step'], 'pre_pick')
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
          'nothing" and from "they were all somewhere else"',
          all(part in node for part in
              ('below score', 'further than', 'these jaws span')), True)
    # And "cannot span" is judged on what is actually between the jaws, not
    # on the opening GraspNet recommends: that number is binned over a
    # 100 mm gripper's range and this one spans 44 mm, so the
    # recommendation is coarse exactly where it has to be fine.
    check('a too-wide candidate is measured against the cloud first',
          '_span_between_jaws' in node, True)
    check('and published as what the jaws actually need',
          "entry['width_source'] = 'measured'" in node, True)
    # Which part of the object gets grasped follows the VLM's point, so
    # naming a part in the prompt -- "screwdriver handle" -- steers the
    # grasp without any per-object rules.
    check('candidates are filtered to the point the detector located',
          'object_radius' in node, True)

    section('where on the object to grab')
    import grasp_recipes
    # The detector answers referring expressions, so the part worth
    # gripping is a prompt and not a new model. What matters is that the
    # substitution is right and that it is always only a first try.
    check('a bare object name is turned into its part',
          grasp_recipes.part_prompt('detect tape'), 'detect edge of the tape')
    check('the object keeps whatever else was said about it',
          grasp_recipes.part_prompt('detect blue bottle'),
          'detect middle of the blue bottle')
    check('an operator who already named a part is left alone',
          grasp_recipes.part_prompt('detect handle of the screwdriver'), None)
    check('and so is an object with no recipe',
          grasp_recipes.part_prompt('detect spirit level'), None)
    # Whole words, or a longer name inherits a shorter one's recipe and a
    # roll of cellotape gets grasped as if it were a ring.
    check('a longer word containing a shorter one does not match',
          grasp_recipes.part_prompt('detect cellotape'), None)
    check('every recipe says why, because the log prints it',
          [r for r in grasp_recipes.RECIPES if not r.get('why')], [])
    check('and every part template names the object',
          [r for r in grasp_recipes.RECIPES
           if '{object}' not in r.get('part', '')], [])

    recipes, problem = grasp_recipes.load_recipes('/no/such/file.json')
    check('a missing recipe file is not a problem', problem, None)
    check('and leaves the built-ins in place', len(recipes),
          len(grasp_recipes.RECIPES))

    source = open(os.path.join(WS, 'pick_place_orchestrator.py')).read()
    locate = source.split('def locate(')[1].split('\n    def ')[0]
    check('the part is asked for first',
          locate.index('self.part_prompt()') < locate.index('self.detect('),
          True)
    check('and the plain object is the fallback, not an error',
          'falling back to' in locate, True)
    # A part the detector cannot find has to cost one inference, not the
    # whole detect_timeout: min_count=0 returns on the first fresh answer
    # whatever is in it, so the fallback is immediate rather than 15 s of
    # waiting for a second answer that was never coming.
    check('and a part it cannot find costs one look, not a timeout',
          'min_count=0, prompt=part' in locate, True)
    check('with which question found it remembered',
          '_pick_prompt' in locate, True)
    verify = source.split('def verify_grasp(')[1].split('\n    def ')[0]
    # Deliberately the object and not the part. Where a recipe aimed the
    # grasp at a part, that part is the bit now inside the jaws -- run
    # 1789022566 asked for "handle of the screwdriver" while holding the
    # handle, saw nothing twice, and dropped it.
    check('but the grasp check asks for the object, not the hidden part',
          'prompt=self.prompt' in verify, True)
    check('and an empty frame falls to the fingers rather than failing',
          'grasp_by_feel' in verify, True)

    # The gripper is asked three separate questions, because each of them
    # goes blind somewhere the others do not: where the jaws stopped, what
    # they are pressing with, and whether either has changed since the
    # close that was known good.
    feel = source.split('def finger_verdict(')[1].split('\n    def ')[0]
    for name, reading in (('grasp_finger_min', 'the width band'),
                          ('grasp_hold_torque_min', 'the grip force'),
                          ('grasp_width_drop', 'the drift since the close')):
        check(f'{reading} is one of the readings', name in feel, True)
    # Zero has to mean off for both of the new ones. A gripper whose driver
    # reports no torque, or an object that deforms under the jaws, needs a
    # way back to the width alone.
    check('the torque reading can be switched off', 'weak > 0.0' in feel, True)
    check('and so can the drift reading', 'drop > 0.0' in feel, True)
    # Only the width ends the check on its own. Believing the fingers over
    # a clear sighting is exactly how run 1789022566 threw away a good pick.
    check('a slack or drifted grip goes to the detector before it decides',
          "feel['source'] == 'width'" in verify, True)

    for layer in ('pick_place.launch.py', 'pick_place_demo.launch.py'):
        text = open(os.path.join(WS, layer)).read()
        for name in ('grasp_hold_torque_min', 'grasp_width_drop'):
            check(f'{name} is reachable from {layer}', name in text, True)
    check('and the grip force is on the panel, beside the width',
          'grasp_hold_torque_min' in seq.SETTINGS_BY_NAME, True)

    part = source.split('def part_prompt(')[1].split('\n    def ')[0]
    # A recipe is a pair -- where to grip and how to come at it -- and
    # half of it is worse than none: a top-down grasp aimed at the middle
    # of a standing bottle is worse than one aimed at the object's centre.
    check('a recipe whose approach cannot be flown is not half-applied',
          "entry.get('approach') not in (None, 'top')" in part, True)
    check('and says so rather than quietly aiming somewhere else',
          'vertical column' in part, True)

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
    # engage_motors exists again, and deliberately: the recovery after a
    # torque trip takes the motors off, lets the arm settle off whatever it
    # hit, and takes hold again. What must not come back is an operator
    # *pressing* it -- a button that puts power back on and then drives the
    # arm somewhere is the thing that was removed.
    callers = [line.strip() for line in source.split('\n')
               if 'self.engage_motors(' in line]
    check('engage_motors has exactly one caller', len(callers), 1)
    recovery = source.split(
        'def recover_from_contact(')[1].split('\n    def ')[0]
    check('and it is the recovery after a contact',
          'self.engage_motors(' in recovery, True)
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
              ['abort', 'grip', 'open_gripper', 'place', 'place_left',
               'place_right', 'reload_config', 'save_config', 'start',
               'stop_arm'])

        # What the field shows has to be what the cycle asks. It asks for
        # the part first where there is a recipe for one, so a panel
        # showing only "detect tape" would disagree with its own log.
        def translated(text):
            return json.loads(asyncio.run(pick_place_web.api_translate(
                _FakeRequest({}, query={'text': text}))).text)

        answer = translated('tape')
        check('the panel shows the part that will actually be asked for',
              answer.get('part'), 'detect edge of the tape')
        check('with the plain object still there as the fallback',
              answer.get('prompt'), 'detect tape')
        check('and why, since the reason is the useful half',
              bool(answer.get('why')), True)
        # A recipe the cycle will not apply must not be advertised either.
        check('a side-approach recipe is not shown, because it is not used',
              'part' in translated('bottle'), False)
        check('and an object with no recipe just shows the object',
              translated('spirit level'),
              {'prompt': 'detect spirit level'})
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

        # Pick and place are two presses, so the page has two buttons --
        # and the second is dead until something is actually held. A place
        # with empty jaws flies the whole approach and opens on nothing.
        check('the page has a Place button of its own',
              'id="btn-place"' in page, True)
        # One button per hand once both are full: a single Place cannot
        # say which object it means.
        check('disabled until a gripper is holding something',
              "const canPlace = !status.busy && full.length === 1;" in page
              and "$('btn-place').disabled = !canPlace;" in page, True)
        check('and two buttons appear when both hands are full',
              'id="btn-place-left"' in page and 'id="btn-place-right"' in page,
              True)
        check('each naming what that hand is holding',
              "`Place ${held[side] || 'object'} (${side})`" in page, True)
        check('with the pair hidden while only one hand is full',
              "$('place-pair').hidden = !both;" in page, True)
        check('and it says why it is disabled',
              'pick something first' in page, True)
        # Grey in a row of grey buttons reads as "not implemented". Once
        # there is something to put down it is the only thing left to
        # press, so it looks like it.
        check('and becomes the primary action once something is held',
              "canPlace ? 'primary grow'" in page, True)
        # Stopping the arm cannot be undone from the panel -- the motors
        # come back with a bringup restart -- so it takes two presses.
        check('the page has a Stop button', 'id="btn-stop"' in page, True)
        check('that takes two presses', 'Press again to stop' in page, True)
        check('and says what it costs', 'needs a restart' in page, True)

        # A browser that keeps serving the copy it already has is a panel
        # missing a button that is right there in the file.
        server = open(os.path.join(WS, 'pick_place_web.py')).read()
        check('the page is never served from the browser cache unchecked',
              "'Cache-Control': 'no-cache'" in server, True)
        # "gripper holding" answers a question nobody asked. Which object is
        # the thing an operator across the room cannot see for themselves.
        check('the pill names which hand holds what',
              '${side} <b>${escapeHtml(' in page, True)
        # Checked as "the object goes through escapeHtml" rather than
        # against one spelling of the expression: it is the operator's own
        # text coming back through a status topic into innerHTML, and which
        # local it is read out of is not the point.
        check('escaped, because it is the operator\'s own text coming back',
              "held[side] || 'something'" in page
              and '${escapeHtml(held[side]' in page, True)

        # The page is one file with its script inline, so nothing compiles
        # it until a browser does -- and a syntax error there is a blank
        # panel with the robot still running. Ask node, which is on this
        # machine for exactly this.
        #
        # Only the plain blocks: the import map is JSON in a <script> tag
        # and a module src is not here to read.
        blocks = re.findall(r'<script([^>]*)>(.*?)</script>', page, re.S)
        inline = [body for attrs, body in blocks
                  if 'src=' not in attrs and 'type=' not in attrs]
        check('the page has inline script to check', bool(inline), True)
        # The editor draws the same line the orchestrator runs. A panel
        # drawing its own guess would show one thing while the robot did
        # another, which is the whole reason the split is computed once.
        check('the flow chart shows where Place takes over',
              'place_from' in page, True)
        check('and it comes from the server, not a second guess',
              'CONFIG.catalogue.place_from' in page, True)
        if shutil.which('node') is None:
            print('      skipped: node is not installed')
        else:
            handle, path = tempfile.mkstemp(suffix='.mjs')
            with os.fdopen(handle, 'w') as script:
                script.write('\n'.join(inline))
            checked = subprocess.run(['node', '--check', path],
                                     capture_output=True, text=True)
            os.unlink(path)
            check('and it parses', (checked.returncode,
                                    checked.stderr.strip().splitlines()[:3]),
                  (0, []))

    # -- the split itself, in the orchestrator --------------------------
    source = open(os.path.join(WS, 'pick_place_orchestrator.py')).read()
    cycle = source.split('def _cycle(')[1].split('\n    def ')[0]
    check('a pick stops after the grasp unless told otherwise',
          "get_parameter('place_after_pick')" in cycle, True)
    check('and ends by standing at the staging pose, holding',
          '_hold_at_staging' in cycle, True)
    hold = source.split('def _hold_at_staging(')[1].split('\n    def ')[0]
    check('which is a state of its own, not DONE', "'HOLDING'" in hold, True)
    check('and it is terminal, so the buttons come back',
          "'HOLDING'" in source.split('TERMINAL_STATES = ')[1].split(')')[0],
          True)

    # The motors come off for two reasons and no others: the operator
    # asked, or the arm ran into something and is leaning on it. An
    # ordinary failure leaves a healthy arm, and taking the motors off it
    # costs a restart of the whole stack to get going again.
    check('an ordinary failure leaves the motors on',
          "declare_parameter('disengage_on_failure', False)" in source, True)
    check('a contact is the exception',
          "declare_parameter('disengage_on_contact', True)" in source, True)
    contact = source.split(
        'def retreat_from_contact(')[1].split('\n    def ')[0]
    check('and the contact guard records that it fired',
          '_contact_fired = True' in contact, True)
    shutdown = source.split('def safe_shutdown(')[1].split('\n    def ')[0]
    check('which is what the shutdown acts on',
          "_contact_fired and self.get_parameter(" in shutdown, True)

    stop = source.split('def _run_stop_arm(')[1].split('\n    def ')[0]
    check('Stop parks the arm before letting go of it',
          stop.index('_retreat_to_home') < stop.index('disengage_motors'),
          True)
    check('and refuses to let go if it could not park',
          'letting go here would drop it' in stop, True)

    # An empty frame is an answer. Driving home to look again finds the
    # same nothing thirty seconds later.
    view = source.split('def take_up_the_view(')[1].split('\n    def ')[0]
    check('nothing detected is reported, not re-looked from home',
          'NOTHING_SEEN' in view, True)
    # A detection that is the robot is a different answer from an empty
    # frame: the view is blocked rather than the object absent. What to do
    # about it depends on whether there is anywhere else to look from --
    # with use_home off there is not, and saying so beats driving to the
    # pose the arm is already standing in and finding the same gripper.
    check('a detection that is the robot still folds out of frame when '
          'HOME is available', 'arrive_at_home' in view, True)
    check('and says there is nowhere to look from when it is not',
          'other pose to look from' in view, True)
    # And a detector that never answered is neither of those.
    check('a silent detector is reported as its own thing',
          'NO_DETECTOR' in view, True)

    # The grasp floor is the surface the object stands on, measured per
    # detection. Flooring on the object's own top face means the jaws can
    # only ever close on air above it.
    grasp = source.split(
        'def grasp_from_detection(')[1].split('\n    def ')[0]
    check('the grasp floor is the measured surface',
          "detection.get('surface')" in grasp, True)
    check('kept clear of it by a configured margin',
          "get_parameter('grasp_table_clearance')" in grasp, True)
    check('with the old rule as the fallback when there is none',
          "get_parameter('grasp_max_depth')" in grasp, True)
    # Both arms wait at the staging pose.
    check('the idle arm waits there too',
          "declare_parameter('boot_other_arm', 'pre_pick')" in source, True)
    # "The map was refreshed" has never meant "the map has anything in
    # it". The updater sensors_3d.yaml names is a separate apt package,
    # and without it move_group plans against an empty world while every
    # log line says the map was captured.
    capture = source.split('def capture_octomap(')[1].split('\n    def ')[0]
    check('a capture says how much actually landed in the map',
          'octomap_voxels' in capture, True)
    # A map that caught the robot is worse than no map: the arm's own
    # start state is then invalid and nothing plans from it. Measured at
    # 13:37 -- all four fingers in the octomap, both arms called invalid,
    # every Pick reporting "the planner is not planning".
    check('a capture away from home checks it did not catch the robot',
          'robot_in_map' in capture, True)
    check('and drops the map rather than keeping it',
          'self.clear_octomap()' in capture, True)
    caught = source.split('def robot_in_map(')[1].split('\n    def ')[0]
    check('asking about both arms, since both are parked in frame',
          'other_arm(self.arm)' in caught, True)
    check('and only counting contacts with the map itself',
          "'octomap' in name" in caught, True)
    # The gripper stands in the camera view at the staging pose, so it
    # lands in any map taken from there -- and an arm inside its own map
    # cannot plan at all. Handled where it belongs: the gripper links are
    # exempt from the octomap for the session, so the arms need not be
    # moved out of the way to take a map.
    check('the grippers are exempt from the map for the whole session',
          "declare_parameter('gripper_never_in_map', True)" in source, True)
    exempt = source.split(
        'def allow_gripper_in_octomap(')[1].split('\n    def ')[0]
    check('and a request to withdraw it is ignored',
          "if not allow and self.get_parameter('gripper_never_in_map')"
          in exempt, True)
    check('so the boot walk need not fold down to home to map',
          "declare_parameter('boot_via_home', False)" in source, True)
    walking = source.split(
        'def _walk_to_boot_pose(')[1].split('\n    def ')[0]
    check('the exemption is in force before the map is captured',
          walking.index('gripper_never_in_map')
          < walking.index('_map_on_the_way'), True)
    # Only the gripper. The forearm and upper arm in the map would still
    # stop everything, and that is still caught.
    check('while anything else in the map still drops it',
          'robot_in_map' in capture, True)

    check('and an empty one is an error with the causes in it',
          'padding_offset' in capture and 'updater' in capture, True)
    # The self-filter is what keeps the robot out of its own map, and the
    # margin it uses is the number that decides whether it works. Too
    # little leaves slivers of gripper as obstacles; too much erases the
    # table and the object with the arm.
    sensors = open(os.path.join(
        WS, 'src', 'openarm_ros2', 'openarm_bimanual_moveit_config',
        'config', 'sensors_3d.yaml')).read()
    check('the octomap updater is the one with a self-filter',
          'DepthImageOctomapUpdater' in sensors, True)
    margin = float(re.search(r'padding_offset:\s*([0-9.]+)', sensors).group(1))
    # 0.20 erased the table under the gripper and the object being reached
    # for along with the arm; 0.05 left whole voxels sitting on the
    # gripper. The usable range is between, and a margin near the top of it
    # is a symptom of the camera mount's TF rather than a setting.
    check('the self-filter margin is inside the measured usable range',
          0.02 <= margin <= 0.15, True)
    # An unreachable pose goal does not come back refused, it comes back
    # not at all -- 60 s, and move_group wedged afterwards. So the drop
    # point is chosen from among points the planner says it can reach.
    drop = source.split('def _drop_on_detected(')[1].split('\n    def ')[0]
    check('the drop point is chosen before it is commanded',
          drop.index('self.drop_point(') < drop.index("'OVER_DROP'"), True)
    where = source.split('def drop_point(')[1].split('\n    def ')[0]
    check('by asking the planner, which is what executes',
          'planner_can_reach' in where, True)
    # Getting the tool *above* a spot proves nothing about getting it
    # down: measured at 13:31, the approach solved, the descent then made
    # 58% of a 92 mm vertical line and the last 36 mm had no solution at
    # all -- checked and unchecked alike. The sheet is wide enough to have
    # other spots, so the candidate is judged at the release height.
    check('a candidate is judged at the release height, not above it',
          'above = (x, y, release_z)' in where, True)
    spots = source.split('def sheet_candidates(')[1].split('\n    def ')[0]
    check('the middle of the sheet first, then outwards',
          'points.sort(key=lambda p: math.hypot' in spots, True)
    check('across a grid of it, not a handful of corners',
          "get_parameter('place_grid')" in spots, True)
    check('pulled in from the edge, because the object has width',
          "get_parameter('place_edge_margin')" in spots, True)
    # cuMotion serves Cartesian goals for one ee_link per bringup, so for
    # the other arm its answer is about the link name and not the point.
    # Reading it as "out of reach" refused a drop the arm could make: run
    # at 12:16, the pick worked on the left arm, the sheet was found, and
    # the place was refused before it moved.
    # PaliGemma always answers something, so "nothing detected" is nearly
    # always "boxes were found and thrown away" -- and which reason it was
    # decides what to do about it. Without it the failure is unactionable.
    node = open(os.path.join(WS, 'VLM', 'vlm_detector_node.py')).read()
    check('the detector says what it threw away', "'rejected': rejected"
          in node, True)
    check('and how many boxes the model returned at all',
          "'raw': len(raw)" in node, True)
    reasons = source.split('def why_nothing(')[1].split('\n    def ')[0]
    check('which the place failure reports back',
          'rejected' in reasons and 'shrugging' in reasons, True)

    check('a verdict the planner cannot give is not a refusal',
          '_pose_goals_ok is False' in where, True)
    check('the kinematics are asked instead for that arm',
          'reachable(above, quat, self.arm, cheap=True)' in where, True)
    check('and a planner that cannot be asked does not refuse the drop',
          'verdict is None' in where, True)

    # The release height is the one the pick measured, not a guess about
    # how far the object hangs below the tool.
    check('the pick records how the object is held',
          '_pick_hold_offset = float(grasp_z - surface[2])' in source, True)
    check('and the jaws closing replaces it with the measured height',
          '_pick_hold_offset = float(' in source,
          True)
    check('and the place uses it', 'offset is not None' in drop, True)

    # A cycle ends where the next one starts. Ending at HOME meant every
    # Pick began by folding out of it again, which is the round trip the
    # staging start exists to avoid.
    check('the cycle ends at the staging pose, not at home',
          seq.DEFAULT_SEQUENCE[-1], 'pre_pick')
    check('and home is no longer compulsory',
          seq.STEPS['home']['required'], False)
    check('but it is still available to add back',
          'home' in seq.STEPS, True)

    # The model's grasps are requested at the staging pose and the
    # pre-flight that uses them runs on the next attempt, so an age limit
    # measured on the clock expired them before anything could act on
    # them. The bound is the cycle instead: the list is dropped when one
    # starts, so no limit means no limit within this cycle.
    check('the grasp model has no age limit by default',
          "declare_parameter('grasp_model_max_age', 0.0)" in source, True)
    proposals = source.split(
        'def model_proposals(')[1].split('\n    def ')[0]
    check('and zero really means no limit',
          'limit > 0.0 and age > limit' in proposals, True)
    starting = source.split('def _cycle(')[1].split('\n    def ')[0]
    check('bounded by the cycle, which drops them when it starts',
          'self._grasps = None' in starting, True)
    check('while distance from the object still rejects them',
          "get_parameter('grasp_model_max_offset')" in proposals, True)

    # The jaws shutting on nothing is usually the grasp being computed
    # above a thin object. Stepping down and closing again costs one short
    # line; a ladder rung costs the whole approach.
    lower = source.split('def close_lower(')[1].split('\n    def ')[0]
    check('an empty grip steps the tool down and tries again first',
          "get_parameter('regrasp_lower_step')" in lower, True)
    check('bounded by the surface the object stands on',
          '_pick_surface_z' in lower and 'grasp_table_clearance' in lower,
          True)
    check('and it gives up to the ladder rather than pressing down',
          'going to the ladder' in lower, True)
    grasping = source.split('def _step_grasp(')[1].split('\n    def ')[0]
    check('tried before the ladder is involved at all',
          'close_lower(ctx)' in grasping, True)

    # cuMotion serves Cartesian goals for one ee_link per bringup, and
    # which arm a cycle uses is decided per cycle, by which half of the
    # frame the object is in. Nothing about the detected place needs a
    # pose goal -- the descent onto the sheet and the retreat off it go
    # through /compute_cartesian_path, which takes a link name and serves
    # either arm -- so getting above the sheet must not need one either.
    # Run at 11:29: the object was in the left half, the cycle correctly
    # switched right -> left, and then refused to run at all.
    check('a detected place does not force a relaunch for the other arm',
          "place_mode in ('position', 'detected')" in source, False)
    lands = source.split('def move_above_drop(')[1].split('\n    def ')[0]
    check('it gets above the drop on a joint goal from IK',
          lands.index('choose_approach_posture')
          < lands.index('_move_to_joints'), True)
    check('and refuses rather than sending a pose goal it cannot',
          '_pose_goals_ok is False' in lands, True)

    # An arm switch costs no motion when the new arm is already standing
    # at its staging pose, which is where both of them wait.
    cycle_body = source.split('def _cycle(')[1].split('\n    def ')[0]
    check('an arm switch does not fold down to home from the staging pose',
          'PRE_PICK_STATE, states)' in cycle_body, True)

    start_place = source.split('def _start_place(')[1].split('\n    def ')[0]
    check('Place refuses when nothing has been picked',
          'self.busy_arms()' in start_place, True)
    # Two questions, and the second is the one that catches an object
    # dropped between the pick and the press.
    check('and asks the gripper again rather than trusting the flag',
          'finger_verdict' in start_place, True)
    check('the stale claim is dropped when the jaws disagree',
          'self._held.pop(arm, None)' in start_place, True)

    release = source.split('def _step_release(')[1].split('\n    def ')[0]
    # The object is down the moment the jaws open. A retreat that then
    # fails is an arm short of home, and calling that "place failed" is
    # what once took the motors off a cycle that had just placed a roll of
    # tape.
    check('the release is recorded before anything that can still fail',
          release.index("ctx['released'] = True")
          < release.index('RETREAT'), True)
    half = source.split('def run_place_half(')[1].split('\n    def ')[0]
    check('and the failure path reads it',
          "ctx.get('released')" in half, True)
    check('a sequence with nothing after the grasp is refused, not "done"',
          'if not place_steps:' in half, True)

    # A joint goal lands where it lands -- OVER_DROP put the tool 47 mm
    # from the point it aimed at and 63 mm low, and coming down to the
    # commanded point from there is a diagonal, not a descent: 14% of the
    # line solved, and the retry was refused for 3.285 rad of travel.
    # Anywhere on the paper will do, so if the tool is already over it,
    # straight down from there.
    lowering = source.split('def _drop_on_detected(')[1].split('\n    def ')[0]
    check('the descent asks where the tool actually is',
          'here = self.fresh_tcp()' in lowering, True)
    check('and comes straight down when it is already over the sheet',
          'on_the_sheet' in lowering, True)
    check('closing the gap at height only when it is not',
          'align_over_column' in lowering, True)
    check('from the height the tool actually reached, not the commanded one',
          'from_z = here[2] if here is not None' in lowering, True)
    # A descent that flew most of the way has the object within
    # millimetres of the sheet. Carrying it back for that is worse than
    # setting it down from slightly higher -- bounded, and said out loud.
    # The cap is on the height of the *object*, not on how far the descent
    # fell short: the tool is hold_offset above the object's base, so a
    # shortfall adds directly to the drop. Whatever the descent misses
    # comes out of the same budget as place_clearance.
    check('the drop is capped on the height of the object above the sheet',
          "get_parameter('place_max_drop')" in lowering, True)
    check('measured from where the tool actually ended up',
          'base_above_sheet' in lowering, True)
    check('with one more short line tried before giving up',
          "'PLACE_GAP'" in lowering, True)
    check('and a drop over the cap is a refusal, not a release',
          'may be dropped (place_max_drop)' in lowering, True)
    sheet = source.split('def on_the_sheet(')[1].split('\n    def ')[0]
    check('which is decided by the sheet\'s own footprint',
          "target.get('footprint')" in sheet, True)
    check('pulled in from its edge', "place_edge_margin" in sheet, True)

    drop = source.split('def _drop_on_detected(')[1].split('\n    def ')[0]
    check('the detected drop goes over the target before descending',
          drop.index("'OVER_DROP'") < drop.index('descend_column'), True)
    check('and lowers on a straight vertical line, not one pose goal',
          'descend_column' in drop, True)
    check('gently -- it is the leg that lowers something held',
          "at_speed(self.get_parameter('place_speed')" in drop, True)
    check('letting go a clearance above the surface rather than on it',
          "get_parameter('place_clearance')" in drop, True)
    walk = source.split('def _walk_to_boot_pose(')[1].split('\n    def ')[0]
    where = source.split('def boot_joints(')[1].split('\n    def ')[0]
    check('the boot walk is slow', "at_speed(speed" in walk, True)
    check('goes to a recorded posture rather than a computed one',
          'PRE_PICK_STATE' in where, True)
    check('as a joint goal, the one kind that is repeatable',
          '_move_to_joints' in walk and 'move_to_pose' not in walk, True)
    check('and skips an arm that has no recording',
          'skipped.append' in walk, True)
    # Waiting at HOME is what makes a Pick begin by looking rather than by
    # folding down to HOME and coming back. The map is captured on arrival
    # for the same reason -- it is the one moment the arm is out of frame.
    # They wait where the approach starts, and the walk goes via HOME so
    # the map can be taken with the robot out of the camera frame.
    check('the staging pose is the default place to wait',
          "declare_parameter('boot_pose', 'pre_pick')" in source, True)
    check('and the walk maps the work area on its way through',
          '_map_on_the_way' in walk, True)
    through = source.split('def _map_on_the_way(')[1].split('\n    def ')[0]
    check('which means going to home for it',
          'home_positions' in through and 'capture_octomap' in through, True)

    # A Pick pressed at the staging pose starts there rather than folding
    # down to HOME and back.
    view = source.split('def take_up_the_view(')[1].split('\n    def ')[0]
    check('a cycle at the staging pose looks from there',
          "get_parameter('locate_from_staging')" in view, True)
    check('only when it is actually standing there',
          'at_state(PRE_PICK_STATE' in view, True)
    check('and falls back to home the moment the look is unusable',
          view.rstrip().endswith('return self.arrive_at_home()'), True)
    check('the answer is kept, so the attempt pays for one look',
          '_last_detection' in view, True)
    # The arm is in the picture from there, and the detector always
    # answers something.
    plausible = source.split(
        'def implausible_detection(')[1].split('\n    def ')[0]
    check('a detection on the gripper is refused as the robot',
          "get_parameter('self_detect_radius')" in plausible, True)
    check('measured against where the tool actually is',
          'tcp_position()' in plausible, True)

    print()
    if failures:
        print(f'{len(failures)} check(s) failed: ' + ', '.join(failures))
        return 1
    print('the sequence survives being edited, and a bad edit says why')
    return 0


class _FakeRequest:
    """Just enough of aiohttp's request for the routes under test."""

    def __init__(self, match_info, query=None):
        self.match_info = match_info
        self.query = query or {}


if __name__ == '__main__':
    sys.exit(main())
