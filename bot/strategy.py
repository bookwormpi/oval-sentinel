"""Long-horizon survival, upgrade selection, weapon pickup, and threat policy."""
from __future__ import annotations
import math
from geometry import dist2

# Scores are intentionally explainable and can be tuned from tournament results.
UPGRADE_BASE = {
    "armor": 120, "plating": 115, "regen": 105, "repair": 100,
    "servos": 88, "capacitor": 82, "reactor": 80, "thrusters": 78,
    "sensors": 65, "gun_rate": 75, "gun_damage": 72, "damage": 72,
    "laser": 70, "mortar": 68, "lightning": 66,
}

def choose_upgrade(obs, hello):
    offer = obs.get("offer")
    if not offer:
        return None
    upgrades = {u["id"]: u for u in hello["config"].get("upgrades", [])}
    me = obs["self"]
    owned = set(me.get("upgrades", []))
    best = (float("-inf"), 0)
    for i, c in enumerate(offer.get("choices", [])):
        uid = c["id"]
        definition = upgrades.get(uid, {})
        if uid in owned:
            score = -100
        else:
            score = UPGRADE_BASE.get(uid, 45)
        # Value armor/health especially when already damaged or late-game.
        if uid in ("armor", "plating", "regen", "repair"):
            score += (1-me["hp"]/max(1,me["max_hp"]))*22
            score += max(0, 12-obs["alive"])*1.5
        if uid in ("gun_rate", "gun_damage") and me["weapons"][0]:
            score += 12
        req = definition.get("requires")
        if req and not any(w and w["kind"] == req for w in me["weapons"]):
            score -= 80
        # Upgrade stacks are multiplicative; first survivability stacks remain valuable.
        if score > best[0]:
            best = (score, i)
    return best[1]

def choose_take(obs):
    me = obs["self"]
    slots = me.get("weapons", [None, None])
    if len(slots) < 2 or slots[1] is not None:
        return None
    x,y,z = me["pos"]
    # Weapon quality is matchup-aware, but the gun remains in slot zero.
    weights = {"laser": 1.0, "mortar": 0.96, "lightning": 0.90}
    options = []
    for p in obs.get("pickups", []):
        w = weights.get(p["kind"], 0)
        if not w:
            continue
        d = math.dist((x,y,z), p["pos"])
        if d <= 4.0:
            options.append((w, p["id"]))
    return max(options)[1] if options else None

def survival_boost(obs, goal_distance):
    me = obs["self"]
    # Preserve energy for emergency escape/jetting; boost when travel is useful.
    if not me["grounded"]:
        return me["energy"] > 12
    if not me["in_zone"]:
        return me["energy"] > 4
    if me["hp"] < 0.35*me["max_hp"] and obs["alive"] <= 5:
        return me["energy"] > 35 and goal_distance > 18
    return me["energy"] > 55 and goal_distance > 45

def should_jump(obs, goal, waypoint):
    me = obs["self"]
    if not me["grounded"] or me["energy"] < 25:
        return False
    # Jump occasionally over small terrain irregularities, not on a fixed cadence.
    dz = waypoint[2] - me["pos"][2]
    return dz > 1.0 and math.dist(me["pos"][:2], waypoint[:2]) < 18
