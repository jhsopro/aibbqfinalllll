# rules.py
from collections import defaultdict, deque

# Multi-doneness foods
MULTI_DONENESS = {"beef1", "beef2"}

# Doneness score
DONENESS_SCORE = {
    "raw": 0,
    "mediumrare": 1,
    "medium": 2,
    "cooked": 3,
}

# History buffer
history = defaultdict(lambda: deque(maxlen=5))


def parse_label(label: str):
    food, state = label.rsplit("_", 1)
    return food, state


def is_done(label, confidence, mode):
    food, state = parse_label(label)
    score = DONENESS_SCORE[state]

    history[food].append(score)

    # MODE A – trust color
    if mode == "A":
        return score == 3

    # MODE B – normal
    if mode == "B":
        return score == 3 and confidence >= 0.6

    # MODE C – high heat, strict
    if mode == "C":
        if food in MULTI_DONENESS:
            return (
                confidence >= 0.75 and
                sum(1 for s in history[food] if s >= 2) >= 3
            )
        else:
            return (
                confidence >= 0.8 and
                history[food].count(3) >= 2
            )

    return False
