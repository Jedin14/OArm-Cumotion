"""Turn what a person types into a prompt PaliGemma actually answers.

paligemma-3b-pt-224 is a pretrained checkpoint, not an instruction-tuned one:
"detect screwdriver" is a well-formed prompt for it and "pick up the
screwdriver" is not -- the latter is a request, and the model has no notion of
answering one. So the conversational wrapper comes off before the prompt reaches
the detector.

Shared by pick_place_ui.py, which shows the translation live as you type, and
pick_place_orchestrator.py, which applies it to anything arriving on
/pick_place/prompt so the topic behaves the same whoever publishes to it --
including a bare `ros2 topic pub`.
"""

import re

# Longest first, so "pick up the" is matched before "pick". Only the leading
# verb phrase is removed: the rest goes to the detector as written, because
# "red screwdriver on the left" is a better detection prompt than any single
# word of it.
LEAD_INS = [
    'pick up the', 'pick up a', 'pick up an', 'pick up',
    'pick the', 'pick a', 'pick an', 'pick',
    'grab the', 'grab a', 'grab an', 'grab',
    'grasp the', 'grasp a', 'grasp an', 'grasp',
    'get the', 'get a', 'get an', 'get',
    'take the', 'take a', 'take an', 'take',
    'find the', 'find a', 'find an', 'find',
    'the', 'a', 'an',
]


def to_detection_prompt(text):
    """"pick up the screwdriver" -> "detect screwdriver".

    Returns '' when nothing but a lead-in was given, so callers can tell "no
    object named" from a usable prompt rather than sending "detect the".
    """
    cleaned = re.sub(r'\s+', ' ', text.strip().lower()).strip(' .!?,')
    if not cleaned:
        return ''
    if cleaned.startswith('detect '):
        return cleaned
    for lead in LEAD_INS:
        if cleaned == lead:
            return ''
        if cleaned.startswith(lead + ' '):
            cleaned = cleaned[len(lead) + 1:]
            break
    return f'detect {cleaned}'
