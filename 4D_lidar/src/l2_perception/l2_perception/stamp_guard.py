"""Detects discontinuities in a message stream's header stamps."""


class StampGuard:
    """Classifies each new stamp relative to the previous one.

    'ok'        normal forward progress
    'duplicate' same stamp again (ignore the message)
    'backward'  time went backwards (host clock stepped back, bag looped, driver restarted)
    'jump'      forward jump larger than max_jump (host clock stepped forward / long outage)
    """

    def __init__(self, name: str, max_jump: float = 1.0):
        self.name = name
        self.max_jump = max_jump
        self.last = None
        self.last_delta = 0.0
        self.counts = {"ok": 0, "duplicate": 0, "backward": 0, "jump": 0}

    def check(self, t: float) -> str:
        if self.last is None:
            self.last = t
            self.counts["ok"] += 1
            return "ok"
        d = t - self.last
        self.last_delta = d
        if d == 0.0:
            res = "duplicate"
        elif d < 0.0:
            res = "backward"
            self.last = t
        elif d > self.max_jump:
            res = "jump"
            self.last = t
        else:
            res = "ok"
            self.last = t
        self.counts[res] += 1
        return res
