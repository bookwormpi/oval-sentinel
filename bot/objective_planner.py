"""Risk-aware objective selection inspired by budgeted orienteering.

Selects the best reachable nearby objective by reward / (travel + exposure),
rather than greedily selecting the closest pickup.
"""
from __future__ import annotations
import math
from navigation import zone_state


def _d(a, b):
    return math.hypot(float(a[0])-float(b[0]), float(a[1])-float(b[1]))


def _enemy_exposure(point, enemies):
    risk = 0.0
    for e in enemies:
        if not isinstance(e, dict) or not isinstance(e.get("pos"), list):
            continue
        d = _d(point, e["pos"])
        # Smooth short-range risk; nearby enemies make detours expensive.
        if d < 100.0:
            risk += ((100.0-d)/100.0) ** 2
            aim = e.get("aim", [0.0, 0.0])
            bearing = math.atan2(point[1]-e["pos"][1], point[0]-e["pos"][0])
            delta = (float(aim[0])-bearing+math.pi)%(2*math.pi)-math.pi
            if abs(delta) < 0.6:
                risk += 0.35 * (1.0-abs(delta)/0.6)
    return risk


def choose_risk_aware_goal(obs, world):
    me = obs["self"]
    p = me["pos"]
    now = float(obs.get("t", 0.0))
    center, radius = zone_state(obs["zone"], now)
    safe_r = max(0.0, radius - max(28.0, radius * 0.08))
    radial = _d(p, center)
    if radial > safe_r:
        return (center[0], center[1], world.ground(*center)), "zone"

    enemies = [e for e in obs.get("mechs", []) if e.get("id") != me.get("id")]
    candidates = []
    cubes = int(me.get("cubes", 0))
    if cubes < 10:
        for c in obs.get("cubes", []):
            q = c.get("pos")
            if not q or _d(q, center) > safe_r - 5:
                continue
            d = _d(p, q)
            if d > 95:
                continue
            value = 1.0 + 0.08 * (10-cubes)
            candidates.append((q, "cube", value, d))

    slots = me.get("weapons", [None, None])
    if len(slots) > 1 and slots[1] is None:
        wvalue = {"laser": 1.2, "lightning": 1.0, "mortar": 0.72}
        for item in obs.get("pickups", []):
            q = item.get("pos")
            value = wvalue.get(item.get("kind"), 0.0)
            if q and value and _d(p, q) < 75 and _d(q, center) < safe_r - 5:
                candidates.append((q, "weapon", value, _d(p, q)))

    if int(obs.get("alive", 16)) <= 7:
        for drop in obs.get("drops", []):
            q = drop.get("pos")
            if q and float(drop.get("lands_in", 999)) <= 0 and drop.get("holder") in (None, me.get("id")) and _d(p, q) < 90:
                if min((_d(q, e["pos"]) for e in enemies if e.get("pos")), default=999) > 26:
                    candidates.append((q, "drop", 1.55, _d(p, q)))

    # The objective must remain inside the projected safe area. Score includes
    # expected combat exposure and a modest penalty for long detours.
    best = None
    for q, kind, value, distance in candidates:
        exposure = _enemy_exposure(q, enemies)
        if exposure > 1.25:
            continue
        score = value / (1.0 + distance / 35.0 + 2.4 * exposure)
        if best is None or score > best[0]:
            best = (score, q, kind)
    if best:
        q = best[1]
        return (float(q[0]), float(q[1]), float(q[2]) if len(q) > 2 else world.ground(q[0], q[1])), best[2]

    # Quiet patrol point: retain the existing low-signature behavior.
    phase = now * 0.19 + float(me.get("id", 0)) * 2.399963
    patrol_r = min(28.0, max(8.0, safe_r * 0.12))
    gx, gy = p[0] + math.cos(phase)*patrol_r, p[1] + math.sin(phase)*patrol_r
    if _d((gx, gy), center) > safe_r:
        gx, gy = center
    return (gx, gy, world.ground(gx, gy)), "patrol"
