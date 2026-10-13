"""The tool fitted past each gripper's fingertips: on or off, and how long.

The working tip -- what touches the clicked point -- is the closed fingertips
(FINGERTIP_BEYOND_TCP past hand_tcp) plus the tool, if one is on. Kept in
tools.yaml so it survives restarts; changed in the window (tool buttons,
[ ] and , . for the length) or with --tool-extension-left/right.
"""

import yaml

from ctm.common import FINGERTIP_BEYOND_TCP, TOOL_EXTENSION, WS

TOOLS_FILE = f'{WS}/tools.yaml'
STEP = 0.001                    # m: one key press
MAX_LENGTH = 0.15               # m


def load():
    """{arm: {'on': bool, 'length': m}}, from tools.yaml or the defaults."""
    tools = {arm: {'on': TOOL_EXTENSION[arm] > 0, 'length': TOOL_EXTENSION[arm] or 0.020}
             for arm in ('left', 'right')}
    try:
        with open(TOOLS_FILE) as handle:
            saved = yaml.safe_load(handle) or {}
    except (OSError, yaml.YAMLError):
        saved = {}
    for arm in tools:
        if isinstance(saved.get(arm), dict):
            tools[arm]['on'] = bool(saved[arm].get('on', tools[arm]['on']))
            tools[arm]['length'] = float(saved[arm].get('length', tools[arm]['length']))
    return tools


def save(tools):
    with open(TOOLS_FILE, 'w') as handle:
        handle.write('# Tools past the fingertips, per arm (click_to_move tool buttons).\n'
                     + yaml.safe_dump({arm: {'on': t['on'], 'length': round(t['length'], 4)}
                                       for arm, t in tools.items()}, sort_keys=False))


def apply(args, tools):
    """Set args.tip_offsets (the dict every touch reads) from the tools."""
    if not isinstance(getattr(args, 'tip_offsets', None), dict):
        args.tip_offsets = {}
    for arm, t in tools.items():
        args.tip_offsets[arm] = FINGERTIP_BEYOND_TCP + (t['length'] if t['on'] else 0.0)


def label(arm, t):
    return f'{arm[0].upper()} TOOL {t["length"] * 1000:.0f}mm' if t['on'] else \
        f'{arm[0].upper()} TOOL OFF'
