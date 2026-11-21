"""Upgrade valuation and conservative energy policy."""
from __future__ import annotations
from collections import Counter

# Higher values indicate better long-term survival / expected combat value.
UPGRADE_BASE = {
    "armor": 132, "plating": 128, "nanites": 120, "reactor": 105,
    "capacitor": 98, "jets": 94, "servos": 88, "gyros": 84,
    "sensors": 76, "gun_rate": 91, "gun_damage": 89,
    "gun_velocity": 78, "gun_mag": 74,
    "laser_charge": 88, "laser_damage": 86, "laser_pierce": 76,
    "lightning_efficiency": 83, "lightning_range": 77, "lightning_chain": 75,
    "mortar_reload": 70, "mortar_velocity": 65, "mortar_splash": 65,
}


def choose_upgrade(obs, hello):
    offer = obs.get("offer")
    if not offer:
        return None
    definitions = {u["id"]: u for u in hello.get("config", {}).get("upgrades", [])}
    me = obs["self"]
    owned = Counter(me.get("upgrades", []))
    weapons = {w["kind"] for w in me.get("weapons", []) if w}
    hp_frac = me["hp"] / max(1.0, me["max_hp"])
    best = (-1e9, 0)
    for i, choice in enumerate(offer.get("choices", [])):
        uid = choice["id"]
        definition = definitions.get(uid, {})
        maximum = int(definition.get("max", 1))
        if owned[uid] >= maximum:
            continue
        req = definition.get("requires")
        if req and req not in weapons:
            continue
        score = float(UPGRADE_BASE.get(uid, 55))
        # Stack defense while it remains available; extra value when damaged or late.
        if uid in ("armor", "plating", "nanites"):
            score += (1.0 - hp_frac) * 24 + max(0, 10-int(obs.get("alive", 16))) * 2
        if uid in ("gun_rate", "gun_damage", "gun_velocity", "gun_mag") and "gun" in weapons:
            score += 10
        if uid.startswith("laser_") and "laser" in weapons:
            score += 9
        if uid.startswith("lightning_") and "lightning" in weapons:
            score += 8
        # Diminishing returns from stacks, but do not prohibit stacking.
        score -= owned[uid] * 13
        if score > best[0]:
            best = (score, i)
    return best[1] if best[0] > -1e8 else None


def choose_take(obs):
    me = obs["self"]
    slots = me.get("weapons", [None, None])
    if len(slots) < 2 or slots[1] is not None:
        return None
    x, y, z = me["pos"]
    weights = {"laser": 1.0, "lightning": 0.95, "mortar": 0.70}
    options = []
    for p in obs.get("pickups", []):
        w = weights.get(p["kind"], 0)
        if w <= 0:
            continue
        dxy = math_hypot(x-p["pos"][0], y-p["pos"][1])
        dz = abs(z-p["pos"][2])
        if dxy <= 4.0 and dz <= 4.0:
            options.append((w, p["id"]))
    return max(options)[1] if options else None


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
