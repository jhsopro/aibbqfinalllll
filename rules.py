# rules.py
# Beef: done based on flips (mode adjusts target)
# Others: done based on YOLO label + confidence thresholds (mode adjusts strictness)

def parse_label(label: str):
    """
    label example: 'chicken_cooked', 'beef1_mediumrare', etc.
    returns: (food, state)
    """
    s = (label or "").lower().strip()
    food, state = s.rsplit("_", 1)
    return food, state

def is_beef_label(label: str) -> bool:
    food, _ = parse_label(label)
    return food in ("beef1", "beef2")

def beef_stage_ok(state: str, preference: str = "medium") -> bool:
    """
    Returns True if the predicted state is at least the preference target.
    Your model states: raw, mediumrare, medium, cooked
    """
    order = ["raw", "mediumrare", "medium", "cooked"]
    s = (state or "").lower().strip()

    # Treat rare as mediumrare (your dataset doesn't include "rare")
    if s == "rare":
        s = "mediumrare"

    # preference normalization
    pref = (preference or "medium").lower().strip()
    if pref == "welldone":
        pref = "cooked"

    if s not in order or pref not in order:
        return False

    return order.index(s) >= order.index(pref)

def beef_target_flips(preference="medium", mode="B"):
    """
    Your beef has multiple doneness targets.
    This is your “flip-count model”.
    You MUST tune these numbers based on real tests.
    """
    # base targets for Mode B (normal)
    base = {
        "rare": 3,
        "mediumrare": 4,
        "medium": 5,
        "cooked": 6,      # or "welldone"
        "welldone": 6,
    }
    target = base.get(preference, 5)

    # Mode A: lower heat -> might require MORE time -> more flips
    if mode == "A":
        target += 1

    # Mode C: high heat -> outside browns fast, but inside may lag
    # Your design says Mode C should be stricter → require “more cooked look”
    # With flip logic, we increase flips by +1 to reduce undercooked risk.
    if mode == "C":
        target += 1

    return target

def beef_is_done(current_flips: int, target_flips: int) -> bool:
    return current_flips >= target_flips

def food_is_done_color(label: str, conf: float, mode: str) -> bool:
    food, state = parse_label(label)
    state = (state or "").lower().strip()

    if state not in ("cooked", "done"):
        return False


    # Confidence thresholds by mode
    if mode == "A":
        return conf >= 0.45
    if mode == "B":
        return conf >= 0.60
    # mode C strict
    return conf >= 0.75
