"""Zone-aware objectives and low-signature movement."""
from __future__ import annotations
import math
from geometry import clamp, dist2, unit2


def zone_state(zone, t):
    c0, c1 = zone["center"], zone["next_center"]
    r0, r1 = float(zone["radius"]), float(zone.get("next_radius", zone["radius"]))
    start, end = float(zone.get("shrink_start", 1e9)), float(zone.get("shrink_end", 1e9))
    if end > start and start <= t <= end:
        f = clamp((t - start) / (end - start), 0.0, 1.0)
        return ((c0[0] + (c1[0] - c0[0]) * f, c0[1] + (c1[1] - c0[1]) * f), r0 + (r1-r0)*f)
    if t > end:
        return (tuple(c1), r1)
    return (tuple(c0), r0)


def choose_goal(obs, world, history=None):
    me = obs["self"]
    x, y, z = me["pos"]
    center, radius = zone_state(obs["zone"], obs["t"])
    radial = dist2((x, y), center)
    # Project into the live safe circle with a substantial buffer; never target the boundary.
    safe_radius = max(0.0, radius - max(28.0, radius * 0.08))
    if radial > safe_radius:
        return (center[0], center[1], world.ground(*center)), "zone"

    nearby_enemies = [e for e in obs.get("mechs", [])
                      if e.get("kind") == "mech" and dist2((x, y), e["pos"][:2]) < 42]
    contested = bool(nearby_enemies)

    # Cubes improve both maximum HP and damage. Don't detour into a close fight for one.
    if not contested and me.get("cubes", 0) < 10 and obs.get("cubes"):
        c = min(obs["cubes"], key=lambda q: dist2((x, y), q["pos"][:2]))
        d = dist2((x, y), c["pos"][:2])
        if d < 72 and dist2(c["pos"][:2], center) < safe_radius - 8:
            return tuple(c["pos"]), "cube"

    # Take a useful secondary weapon only when it is reasonably close and safe.
    slots = me.get("weapons", [None, None])
    if not contested and len(slots) > 1 and slots[1] is None:
        weights = {"laser": 1.0, "mortar": 0.72, "lightning": 0.92}
        choices = [p for p in obs.get("pickups", []) if weights.get(p["kind"], 0) > 0]
        if choices:
            p = min(choices, key=lambda q: dist2((x, y), q["pos"][:2]) / weights[q["kind"]])
            if dist2((x, y), p["pos"][:2]) < 65:
                return tuple(p["pos"]), "weapon"

    # Supply drops can be decisive, but only approach an uncontested landed drop.
    if obs.get("alive", 16) <= 7 and not contested:
        for d in obs.get("drops", []):
            if d.get("lands_in", 999) <= 0 and d.get("holder") in (None, me["id"]):
                p = d["pos"]
                if dist2((x, y), p[:2]) < 85:
                    rivals = [e for e in obs.get("mechs", [])
                              if e.get("id") != me["id"] and dist2(e["pos"][:2], p[:2]) < 24]
                    if not rivals:
                        return tuple(p), "drop"

    # Quiet, slow patrol inside the zone: stay difficult to hear and avoid crossing the map.
    phase = float(obs.get("t", 0)) * 0.19 + float(me.get("id", 0)) * 2.399963
    patrol_r = min(28.0, max(8.0, safe_radius * 0.12))
    gx = x + math.cos(phase) * patrol_r
    gy = y + math.sin(phase) * patrol_r
    if dist2((gx, gy), center) > safe_radius:
        gx, gy = center
    return (gx, gy, world.ground(gx, gy)), "patrol"


def steer(obs, world, goal):
    me = obs["self"]
    p = me["pos"]
    waypoint = world.route(tuple(p), tuple(goal))
    dx, dy = waypoint[0] - p[0], waypoint[1] - p[1]
    direct = unit2(dx, dy)
    rx = ry = 0.0
    if world.ready:
        # Repel only from nearby solid footprints; distant prisms are irrelevant.
        for prism, bounds in zip(world.prisms, world.prism_bounds):
            if bounds[2] < p[0]-24 or bounds[0] > p[0]+24 or bounds[3] < p[1]-24 or bounds[1] > p[1]+24:
                continue
            ring = prism.get("ring", [])
            if not ring:
                continue
            cx = sum(ring[0::2]) / (len(ring)//2)
            cy = sum(ring[1::2]) / (len(ring)//2)
            d = math.hypot(p[0]-cx, p[1]-cy)
            if 1e-3 < d < 24:
                strength = (24-d)/24
                rx += (p[0]-cx)/d*strength
                ry += (p[1]-cy)/d*strength
    return unit2(direct[0] + 0.65*rx, direct[1] + 0.65*ry), waypoint
