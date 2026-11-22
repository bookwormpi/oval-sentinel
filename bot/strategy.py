"""Upgrade valuation and conservative energy policy."""
from __future__ import annotations
import os

# Change this value and submit the resulting commit to test a different policy.
SELECTED_STRATEGY = "neural"
STRATEGIES = ("baseline", "survivor", "scavenger", "skirmisher", "vanguard", "adaptive", "neural")


def selected_strategy(mode=None):
    if mode is None:
        mode = os.environ.get("OVAL_STRATEGY", SELECTED_STRATEGY)
    if mode not in STRATEGIES:
        raise ValueError(f"Unknown strategy {mode!r}; choose from {', '.join(STRATEGIES)}")
    return mode

# Relative importance of each stat for overall effectiveness. This is the only
# remaining judgment call: the *size* of each upgrade's effect is computed from
# the mech's real current stats (see _effect_value), not guessed per upgrade.
STAT_WEIGHT = {
    "hp": 1.00, "armor": 1.00, "regen_per_s": 0.45, "regen_delay": 0.22,
    "walk_speed": 0.22, "boost_speed": 0.26, "energy": 0.45, "energy_regen": 0.32,
    "boost_drain": 0.28, "jet_drain": 0.16, "turn_rate": 0.40, "pitch_rate": 0.26,
    "near_range_bonus": 0.14, "vision_bonus": 0.14, "hearing": 0.18,
    "damage": 1.00, "cooldown": 0.95, "speed": 0.20, "range": 0.18,
    "mag": 0.45, "reload": 0.26, "charge": 0.85, "pierce": 0.30,
    "chain": 0.32, "chain_range": 0.18, "cone_deg": 0.16, "splash_radius": 0.30,
}
# These stats are consumed as a pool/rate (uses per energy, actions per second),
# so a fractional drop in the raw number buys more than that fraction of extra
# throughput: a 25% cooldown cut yields 1/0.75 - 1 = 33% more shots, not 25%.
RECIPROCAL_STATS = {"cooldown", "reload", "charge", "boost_drain", "jet_drain"}


def _effect_value(effect, mech_stats, weapon_stats_by_kind, vision):
    """Real marginal value of one more copy of a single upgrade effect."""
    parts = effect["path"].split(".")
    leaf = parts[-1]
    if parts[0] == "mech":
        current = mech_stats.get(leaf)
    elif parts[0] == "weapons" and len(parts) == 3:
        current = weapon_stats_by_kind.get(parts[1], {}).get(leaf)
    else:
        current = None
    if not isinstance(current, (int, float)):
        return 0.0
    new = current * effect.get("mul", 1.0) + effect.get("add", 0.0)
    if parts[0] == "mech" and leaf == "armor":
        # Armor is consumed as (1 - armor) multiplying incoming damage; value it
        # by the resulting change in effective HP, not the raw armor number.
        old_effective_hp = 1.0 / max(1e-6, 1.0 - current)
        new_effective_hp = 1.0 / max(1e-6, 1.0 - min(0.95, new))
        change = abs(new_effective_hp - old_effective_hp) / old_effective_hp
    elif leaf in ("near_range_bonus", "vision_bonus"):
        # These are meter bonuses added on top of a separate base vision range,
        # so they start at (or near) zero; a raw-number ratio is meaningless.
        # Value the added meters relative to the range they extend.
        base_range = vision.get("near_range" if leaf == "near_range_bonus" else "range", 1.0)
        change = abs(effect.get("add", 0.0)) / max(1.0, float(base_range))
    elif leaf in RECIPROCAL_STATS and current > 1e-9 and new > 1e-9:
        change = abs(1.0 / new - 1.0 / current) / (1.0 / current)
    elif abs(current) > 1e-6:
        change = abs(new - current) / abs(current)
    else:
        # A handful of remaining stats start at zero (e.g. laser pierce); there
        # is no meaningful ratio, so value the raw (typically small, integer)
        # amount gained directly rather than as a fraction of zero.
        change = abs(effect.get("add", 0.0)) * 0.5
    return STAT_WEIGHT.get(leaf, 0.3) * change


def choose_upgrade(obs, hello):
    offer = obs.get("offer")
    if not offer:
        return None
    config = hello.get("config", {})
    definitions = {u["id"]: u for u in config.get("upgrades", [])}
    vision = config.get("vision", {})
    me = obs["self"]
    owned = {}
    for upgrade in me.get("upgrades", []):
        owned[upgrade] = owned.get(upgrade, 0) + 1
    weapons = {w["kind"] for w in me.get("weapons", []) if w}
    weapon_stats = {w["kind"]: w.get("stats", {}) for w in me.get("weapons", []) if w}
    mech_stats = me.get("stats", {})
    hp_frac = me["hp"] / max(1.0, me["max_hp"])
    best = (-1e9, 0)
    for i, choice in enumerate(offer.get("choices", [])):
        uid = choice["id"]
        definition = definitions.get(uid, {})
        maximum = int(definition.get("max", 1))
        if owned.get(uid, 0) >= maximum:
            continue
        req = definition.get("requires")
        if req and req not in weapons:
            continue
        score = sum(_effect_value(effect, mech_stats, weapon_stats, vision)
                    for effect in definition.get("effects", []))
        # Stack defense while it remains available; extra value when damaged or late.
        if uid in ("armor", "plating", "nanites"):
            score += (1.0 - hp_frac) * 0.4 + max(0, 10 - int(obs.get("alive", 16))) * 0.03
        if score > best[0]:
            best = (score, i)
    return best[1] if best[0] > -1e8 else None


# Relative secondary-weapon priority, shared with the goal-seeking logic in
# policies.py so pursuing and actually taking a weapon agree on which is better.
WEAPON_VALUE = {"laser": 1.0, "lightning": 0.95, "mortar": 0.70}
# Minimum value gap to justify a swap: equip_s briefly disables firing, so a
# sideways or marginal trade is not worth it.
SWAP_MARGIN = 0.1


def choose_take(obs):
    """Return (take_id, drop_slot); both are None if nothing worth taking is near."""
    me = obs["self"]
    slots = me.get("weapons", [None, None])
    if len(slots) < 2:
        return None, None
    x, y, z = me["pos"]
    current = slots[1]
    current_value = WEAPON_VALUE.get(current["kind"], 0) if current else -1.0
    options = []
    for p in obs.get("pickups", []):
        w = WEAPON_VALUE.get(p["kind"], 0)
        if w <= 0:
            continue
        dxy = math_hypot(x-p["pos"][0], y-p["pos"][1])
        dz = abs(z-p["pos"][2])
        if dxy <= 4.0 and dz <= 4.0:
            options.append((w, p["id"]))
    if not options:
        return None, None
    best_weight, best_id = max(options)
    if current is None:
        return best_id, None
    # A laser mid-charge cannot be dropped (the engine rejects it); don't bother.
    if best_weight > current_value + SWAP_MARGIN and not (
            current["kind"] == "laser" and float(current.get("charge", 0)) > 0):
        return best_id, 1
    return None, None


def math_hypot(x, y):
    return (x*x + y*y) ** 0.5


def survival_boost(obs, goal_distance, danger=0.0):
    me = obs["self"]
    energy = float(me.get("energy", 0))
    if not me.get("grounded", True):
        return energy > 18 and (danger > 0.25 or not me.get("in_zone", True))
    if not me.get("in_zone", True):
        return energy > 12 and goal_distance > 12
    if danger > 0.55 and energy > 28:
        return True
    if me["hp"] < 0.32*me["max_hp"] and obs.get("alive", 16) <= 5:
        return energy > 35 and goal_distance > 18
    # Preserve energy; don't spend it on ordinary patrols.
    return energy > 72 and goal_distance > 55


def should_jump(obs, urgency=0.0):
    me = obs["self"]
    return bool(me.get("grounded") and me.get("energy", 0) > 20 and urgency > 0.72)
