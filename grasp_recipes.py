"""Where to grab a named object, and how to come at it.

The detector answers referring expressions, not just nouns. PaliGemma is
trained on them: "detect handle of the screwdriver" localises the handle,
not the whole tool, and costs exactly the same as "detect screwdriver".
Nothing here is a new model or a new capability -- it is the prompt, which
has always gone through to the detector as written.

What it adds is not having to remember. Typing "tape" should not silently
aim at the middle of a ring, and it does: the detector reports the centre
of what it sees, the centre of a roll is the hole, and the jaws close on
nothing. A recipe turns the bare word into the phrase that finds the part
worth gripping, and says why in a sentence the log prints.

    tape         -> detect edge of the tape
    screwdriver  -> detect handle of the screwdriver
    bottle       -> detect middle of the bottle

Everything here is advisory. A part prompt that finds nothing falls back to
the plain object immediately -- a recipe that makes an object unpickable
would be worse than no recipe at all -- and the fallback is logged, because
"the part was not found" is the thing worth knowing when a pick gets worse
rather than better.

    approach     'top' or 'side'. Only 'top' can be flown today: the
                 approach and retreat are a vertical column, so a side
                 grasp is reported and refused rather than driven. See
                 grasp_model_max_tilt.

The table is a starting point, not a truth. Override or extend it with a
JSON file of the same shape -- see load_recipes -- and check what the
detector actually does with a phrase before trusting it:

    VLM/run_vlm_detect.sh --prompt "detect handle of the screwdriver"
"""

import json
import os


# Ordered: the first whose `match` appears in the object name wins, so a
# more specific entry has to come first. Matching is on words, not
# substrings -- "cellotape" must not match "tape" and read as a ring.
RECIPES = [
    {
        'match': ['tape', 'roll of tape', 'masking tape', 'duct tape',
                  'sellotape', 'reel'],
        'part': 'edge of the {object}',
        'approach': 'top',
        'why': 'a roll is a ring, and the centre of a ring is the hole',
    },
    {
        'match': ['screwdriver', 'wrench', 'spanner', 'hammer', 'pliers',
                  'knife', 'brush', 'file', 'chisel'],
        'part': 'handle of the {object}',
        'approach': 'top',
        'why': 'the handle is the part shaped to be held, and it is the '
               'part whose width the jaws can span',
    },
    {
        'match': ['mug', 'cup'],
        'part': 'rim of the {object}',
        'approach': 'top',
        'why': 'the rim is a wall the jaws can close on; the body is wider '
               'than they open and the handle is easy to miss',
    },
    {
        'match': ['bottle', 'can', 'jar', 'flask', 'tin'],
        'part': 'middle of the {object}',
        'approach': 'side',
        'why': 'a standing bottle is gripped around the barrel, which means '
               'coming in horizontally rather than down onto the cap',
    },
]

# Where an override lives, if there is one. Beside the workspace, like the
# saved config.
DEFAULT_RECIPE_FILE = 'grasp_recipes.json'


def object_words(prompt):
    """The object out of a detection prompt: "detect tape" -> ["tape"]."""
    text = (prompt or '').strip().lower()
    if text.startswith('detect '):
        text = text[len('detect '):]
    return [word.strip(' .,') for word in text.split() if word.strip(' .,')]


def recipe_for(prompt, recipes=None):
    """The recipe for this prompt, or None.

    Matched on whole words so that a longer name containing a shorter one
    does not inherit its recipe. A prompt that already names a part --
    "handle of the screwdriver" -- is left alone: the operator has said
    what they want and a recipe second-guessing them would be worse than
    no recipe.
    """
    words = object_words(prompt)
    if not words:
        return None
    joined = ' '.join(words)
    for entry in (RECIPES if recipes is None else recipes):
        for name in entry.get('match') or []:
            if name in joined and _names_a_part(joined, entry):
                return None
            if _matches(words, joined, name):
                return entry
    return None


def _matches(words, joined, name):
    """Whole-word match, so "cellotape" is not "tape"."""
    parts = name.split()
    if len(parts) == 1:
        return parts[0] in words
    return f' {name} ' in f' {joined} '


def _names_a_part(joined, entry):
    """Has the operator already said which part they mean?"""
    template = entry.get('part') or ''
    lead = template.split('{')[0].strip()
    return bool(lead) and lead in joined


def part_prompt(prompt, recipes=None):
    """The prompt that finds the part worth gripping, or None.

    None whenever the plain prompt is already the right thing to ask, which
    is the common case: there is no recipe for most objects, and none is
    needed.
    """
    entry = recipe_for(prompt, recipes)
    if entry is None:
        return None
    words = object_words(prompt)
    if not words:
        return None
    part = entry['part'].format(object=' '.join(words))
    return f'detect {part}'


def load_recipes(path=None):
    """The table, with a JSON file's entries in front of the built-ins.

    Returns (recipes, problem). The problem is a string for the operator
    and never an exception: a malformed recipe file must not stop the robot
    coming up, it must stop *being used* and say so.
    """
    if not path:
        return list(RECIPES), None
    if not os.path.exists(path):
        return list(RECIPES), None
    try:
        with open(path) as handle:
            loaded = json.load(handle)
    except (OSError, ValueError) as exc:
        return list(RECIPES), f'{path} could not be read ({exc}); using the built-in recipes'
    if not isinstance(loaded, list):
        return list(RECIPES), f'{path} should hold a list of recipes; using the built-in ones'
    clean, dropped = [], 0
    for entry in loaded:
        if (isinstance(entry, dict) and entry.get('match')
                and isinstance(entry.get('match'), list)
                and isinstance(entry.get('part'), str)
                and '{object}' in entry['part']):
            clean.append(entry)
        else:
            dropped += 1
    problem = (None if not dropped else
               f'{dropped} recipe(s) in {path} were dropped: each needs a '
               f'"match" list and a "part" containing {{object}}')
    return clean + list(RECIPES), problem
