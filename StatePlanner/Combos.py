"""
Combos.py — F8 deterrence combinations (DR2 plan X3).

Each chase plays one combination of
    approach shape  x  speed profile  x  opening pose  x  opening sound
so a turkey doesn't see the same thing twice in a row. COMBOS below is the
full list the planner picks from; edit it freely (keep at least 10 rows and
use only the names in APPROACHES, SPEED_PROFILES, POSES and SOUNDS).

Picking is a shuffle bag: all combos in random order, then a fresh shuffle,
never starting a new round with the combo that ended the last one. So there is
never a back-to-back repeat, and the first len(COMBOS) chases are all different.

The opening is the first CHASE reaction (pose + sound for ~3 s, like W26's
GrowlAndChase). After it, W26's distance-based AggressivePursuit / BurstAttack
loop runs as before. Speed profiles use only the two existing chase limits
(C7, no new speed values):
    "slow" = StatePlannerReceiver.SPEED_CHASE_GROWL,  "full" = SPEED_CHASE
"""

from __future__ import annotations

import random
from typing import List, NamedTuple, Optional

APPROACHES = ("zigzag", "straight")
POSES      = ("AggressivePose", "ExcitedGait", "AlertStep")          # PoseDriver poses (C6)
SOUNDS     = ("Growl", "AggressiveBark", "BarkBurst", "BarkShort_3x", "BarkShort_2x2")   # C6

PULSE_PERIOD_S = 3.0

# (seconds after the chase starts, "slow" | "full"); the last entry holds.
SPEED_PROFILES = {
    "growl_then_rush": [(0.0, "slow"), (3.0, "full")],                     # W26's
    "steady":          [(0.0, "full")],
    "creep":           [(0.0, "slow")],
    "creep_then_rush": [(0.0, "slow"), (6.0, "full")],
    "pulse":           [(i * PULSE_PERIOD_S, "slow" if i % 2 == 0 else "full") for i in range(20)],
}


class Combo(NamedTuple):
    name:     str
    approach: str
    speed:    str
    pose:     str
    sound:    str

    def describe(self) -> str:
        return f"'{self.name}': {self.approach}, {self.speed}, {self.pose}, {self.sound}"


COMBOS: List[Combo] = [Combo(*row) for row in [
    # name                approach    speed              pose              sound
    ("W26 classic",       "straight", "growl_then_rush", "AggressivePose", "Growl"),
    ("Stalker",           "zigzag",   "creep",           "AlertStep",      "Growl"),
    ("Charger",           "straight", "steady",          "AggressivePose", "AggressiveBark"),
    ("Weaver",            "zigzag",   "steady",          "ExcitedGait",    "BarkShort_3x"),
    ("Pouncer",           "zigzag",   "creep_then_rush", "AlertStep",      "BarkBurst"),
    ("Pulser",            "zigzag",   "pulse",           "ExcitedGait",    "BarkShort_2x2"),
    ("Bluffer",           "straight", "pulse",           "AlertStep",      "BarkBurst"),
    ("Sneak and shout",   "straight", "creep_then_rush", "ExcitedGait",    "AggressiveBark"),
    ("Slow growl",        "straight", "creep",           "AggressivePose", "Growl"),
    ("Zig rush",          "zigzag",   "creep_then_rush", "AggressivePose", "AggressiveBark"),
    ("Yapper",            "straight", "steady",          "ExcitedGait",    "BarkShort_3x"),
    ("Weaving growl",     "zigzag",   "pulse",           "AggressivePose", "Growl"),
]]


def fixed_combo(approach: str) -> Combo:
    """W26's chase with a chosen approach shape, for CHASE_RANDOM_COMBOS = False."""
    return Combo("fixed", approach, "growl_then_rush", "AggressivePose", "Growl")


def validate(combos: List[Combo]) -> None:
    for c in combos:
        if (c.approach not in APPROACHES or c.speed not in SPEED_PROFILES
                or c.pose not in POSES or c.sound not in SOUNDS):
            raise ValueError(f"Combos.py: bad combo {c}")
    if len({c.name for c in combos}) != len(combos):
        raise ValueError("Combos.py: combo names must be unique")


validate(COMBOS)


class ComboPicker:
    """Shuffle bag with no back-to-back repeat across rounds."""

    def __init__(self, combos: Optional[List[Combo]] = None, rng: Optional[random.Random] = None):
        self.combos = list(combos or COMBOS)
        if len(self.combos) < 2:
            raise ValueError("ComboPicker needs at least 2 combos to avoid repeats")
        validate(self.combos)
        self.rng   = rng or random.Random()
        self._bag: List[Combo] = []
        self.last: Optional[Combo] = None
        self.count = 0

    def pick(self) -> Combo:
        if not self._bag:
            self._bag = self.combos[:]
            self.rng.shuffle(self._bag)
            if self._bag[-1] == self.last:          # popped from the end: don't repeat
                self._bag[0], self._bag[-1] = self._bag[-1], self._bag[0]
        self.last = self._bag.pop()
        self.count += 1
        return self.last
