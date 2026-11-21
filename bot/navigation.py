"""Zone-aware goal selection and map-aware steering."""
from __future__ import annotations
import math
from geometry import clamp, dist2, unit2

def zone_center(zone, t):
    c = zone["center"]
    n = zone["next_center"]
    start, end = float(zone.get("shrink_start", 1e9)), float(zone.get("shrink_end", 1e9))
    if end > start and start <= t <= end:
        f = clamp((t-start)/(end-start), 0, 1)
        return (c[0]+(n[0]-c[0])*f, c[1]+(n[1]-c[1])*f)
    if t > end:
        return tuple(n)
    return tuple(c)

def choose_goal(obs, world, history):
    me = obs["self"]
    x, y, z = me["pos"]
    zone = obs["zone"]
    center = zone_center(zone, obs["t"])
    radius = float(zone["radius"])
    if dist2((x,y), center) > radius - 22:
        return (center[0], center[1], world.ground(*center)), "zone"

    # Drops: pursue only after landing and only if there is no other known holder.
    for d in obs.get("drops", []):
        if d.get("lands_in", 999) <= 0 and d.get("holder") in (None, me["id"]):
            p = d["pos"]
            # A drop is a high-value objective only if it can plausibly be reached.
            if dist2((x,y), p[:2]) < 110 and obs["alive"] <= 9:
                return tuple(p), "drop"

    # Cubes are consistently valuable: +4% HP and damage, up to stat cap.
    if me.get("cubes", 0) < 10 and obs.get("cubes"):
        c = min(obs["cubes"], key=lambda q: dist2((x,y), q["pos"][:2]))
        cp = c["pos"]
        if dist2((x,y), cp[:2]) < 100:
            return tuple(cp), "cube"

    # Early weapon pickup: keep the gun and fill slot 1 with a weapon.
    slots = me.get("weapons", [])
    if len(slots) > 1 and slots[1] is None:
        weights = {"laser": 1.0, "mortar": 0.94, "lightning": 0.82, "gun": 0.0}
        choices = [p for p in obs.get("pickups", []) if weights.get(p["kind"], 0) > 0]
        if choices:
            p = min(choices, key=lambda q: dist2((x,y), q["pos"][:2]) / weights[q["kind"]])
            if dist2((x,y), p["pos"][:2]) < 95:
                return tuple(p["pos"]), "weapon"

    # Stay near the safer interior, not exactly at the circle center if that is contested.
    return (center[0], center[1], world.ground(*center)), "center"

def steer(obs, world, goal):
    me = obs["self"]
    p = me["pos"]
    waypoint = world.route(tuple(p), tuple(goal))
    dx, dy = waypoint[0]-p[0], waypoint[1]-p[1]
    direct = unit2(dx, dy)
    # Small obstacle repulsion keeps the mech from scraping building edges.
    rx = ry = 0.0
    if world.ready:
        for prism in world.prisms:
            ring = prism.get("ring", [])
            if not ring:
                continue
            cx = sum(ring[0::2])/(len(ring)//2)
            cy = sum(ring[1::2])/(len(ring)//2)
            d = math.hypot(p[0]-cx, p[1]-cy)
            if d < 22 and d > 1e-3:
                strength = (22-d)/22
                rx += (p[0]-cx)/d*strength
                ry += (p[1]-cy)/d*strength
    return unit2(direct[0]+0.8*rx, direct[1]+0.8*ry), waypoint
