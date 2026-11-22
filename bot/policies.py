"""Alternative strategic objectives and combat movement."""
from __future__ import annotations

import math
from geometry import dist2, unit2
from navigation import zone_state


def _interior_point(pos, center, radius, margin):
    dx, dy = pos[0] - center[0], pos[1] - center[1]
    distance = math.hypot(dx, dy)
    safe_radius = max(0.0, radius - margin)
    if distance <= safe_radius:
        return pos[0], pos[1]
    if distance < 1e-9:
        return center[0], center[1]
    return center[0] + dx * safe_radius / distance, center[1] + dy * safe_radius / distance


def _zone_goal(obs, world):
    me = obs["self"]
    pos = me["pos"]
    zone = obs["zone"]
    center, radius = zone_state(zone, obs["t"])
    margin = min(20.0, radius * 0.1)
    point = _interior_point(pos, center, radius, margin)
    if dist2(pos, point) > 0.5 or not me["in_zone"]:
        return (point[0], point[1], world.ground(*point)), "zone"

    next_center = zone["next_center"]
    next_radius = float(zone["next_radius"])
    point = _interior_point(pos, next_center, next_radius, min(20.0, next_radius * 0.1))
    distance = dist2(pos, point)
    speed = max(1.0, float(me.get("stats", {}).get("walk_speed", 11.7)))
    time_left = max(0.0, float(zone["shrink_end"]) - float(obs["t"]))
    if distance > 1.0 and time_left <= distance / speed + 12.0:
        return (point[0], point[1], world.ground(*point)), "zone"
    return None


def choose_variant_goal(obs, world, mode):
    """Choose objectives without requiring information outside the observation."""
    zone_goal = _zone_goal(obs, world)
    if zone_goal is not None:
        return zone_goal

    me = obs["self"]
    pos = me["pos"]
    center, radius = zone_state(obs["zone"], obs["t"])
    enemies = [e for e in obs.get("mechs", [])
               if e.get("kind") == "mech" and e.get("id") != me["id"]]

    def safe(point, threat_radius):
        if dist2(point, center) > max(0.0, radius - min(12.0, radius * 0.05)):
            return False
        if abs(point[2] - pos[2]) > 4.0:
            return False
        return all(dist2(point, enemy["pos"]) >= threat_radius for enemy in enemies)

    # A full repair and three cubes can justify a detour, including early in a match.
    for drop in obs.get("drops", []):
        point = drop["pos"]
        radius_drop = float(drop.get("radius", 8.0))
        distance = dist2(pos, point)
        if (drop.get("lands_in", 0) <= 0
                and drop.get("holder") in (None, me["id"])
                and distance < (100.0 if mode == "scavenger" else 60.0)
                and safe(point, 45.0)):
            if distance <= radius_drop * 0.6:
                return tuple(pos), "capture"
            return tuple(point), "drop"

    if mode == "scavenger":
        choices = []
        if me.get("cubes", 0) < 10:
            for cube in obs.get("cubes", []):
                point = cube["pos"]
                distance = dist2(pos, point)
                if distance < 100 and safe(point, 45):
                    choices.append((distance, point, "cube"))
        slots = me.get("weapons", [])
        if len(slots) > 1 and slots[1] is None:
            weights = {"laser": 1.0, "lightning": 0.9, "mortar": 0.7}
            for pickup in obs.get("pickups", []):
                weight = weights.get(pickup["kind"], 0)
                point = pickup["pos"]
                distance = dist2(pos, point)
                if weight and distance < 85 and safe(point, 45):
                    choices.append((distance / weight + 8, point, "weapon"))
        if choices:
            _, point, kind = min(choices, key=lambda item: item[0])
            return tuple(point), kind
    elif mode == "skirmisher":
        # Pick up immediate resources without chasing them across the map.
        for cube in obs.get("cubes", []):
            if me.get("cubes", 0) < 10 and dist2(pos, cube["pos"]) < 30 and safe(cube["pos"], 35):
                return tuple(cube["pos"]), "cube"
        slots = me.get("weapons", [])
        if len(slots) > 1 and slots[1] is None:
            for pickup in obs.get("pickups", []):
                if dist2(pos, pickup["pos"]) < 35 and safe(pickup["pos"], 35):
                    return tuple(pickup["pos"]), "weapon"

    return tuple(pos), "hold"


def choose_crate(obs, last_damage_t):
    """Farm a nearby crate only when no enemy is close and damage is not recent."""
    me = obs["self"]
    if obs["t"] - last_damage_t < 8.0:
        return None
    if any(e.get("kind") == "mech" and dist2(me["pos"], e["pos"]) < 120
           for e in obs.get("mechs", [])):
        return None
    crates = [c for c in obs.get("crates", [])
              if dist2(me["pos"], c["pos"]) < 70
              and abs(me["pos"][2] - c["pos"][2]) <= 4]
    if not crates:
        return None
    crate = min(crates, key=lambda c: dist2(me["pos"], c["pos"]))
    return {"id": crate["id"], "kind": "crate", "pos": crate["pos"], "vel": [0, 0, 0]}


def variant_movement(mode, obs, target, goal_kind, base, evasion, dodge, danger, projectile_urgency):
    """Return a direction and speed fraction for an alternative policy."""
    me = obs["self"]
    urgency = max(danger, projectile_urgency)
    if not me["in_zone"] or goal_kind == "zone":
        direction = unit2(base[0] * 2 + dodge[0] * projectile_urgency,
                          base[1] * 2 + dodge[1] * projectile_urgency)
        return direction, 1.0
    if mode == "skirmisher" and target and target.get("kind") == "mech":
        dx = target["pos"][0] - me["pos"][0]
        dy = target["pos"][1] - me["pos"][1]
        distance = math.hypot(dx, dy)
        if 1.0 < distance < 100.0:
            toward = (dx / distance, dy / distance)
            side = 1.0 if int(me["id"]) % 2 == 0 else -1.0
            radial = 0.55 if distance > 70 else (-0.8 if distance < 45 else 0.0)
            direction = unit2(
                radial * toward[0] - side * toward[1] + evasion[0] * danger + dodge[0] * projectile_urgency * 2,
                radial * toward[1] + side * toward[0] + evasion[1] * danger + dodge[1] * projectile_urgency * 2,
            )
            return direction, 0.9
    if urgency > 0.12:
        direction = unit2(base[0] * 0.35 + evasion[0] * (1.0 + danger) + dodge[0] * projectile_urgency * 2,
                          base[1] * 0.35 + evasion[1] * (1.0 + danger) + dodge[1] * projectile_urgency * 2)
        return direction, 0.9
    if goal_kind in ("hold", "capture"):
        return (0.0, 0.0), 0.0
    return base, 1.0
