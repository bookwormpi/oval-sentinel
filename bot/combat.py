"""Weapon-aware targeting, lead, firing gates, and projectile threat response."""
from __future__ import annotations
import math
from geometry import (clamp, dist2, dist3, wrap_angle, angle_delta,
                      intercept_time, ballistic_pitch)

def choose_target(obs, me):
    candidates = []
    mp = me["pos"]
    for e in obs.get("mechs", []):
        d = dist3(mp, e["pos"])
        if d > 280:
            continue
        # Prefer vulnerable real players; NPCs are worthwhile cube sources but secondary.
        hp = float(e.get("hp", 900))
        kind = 1.0 if e["kind"] == "mech" else 0.70
        threat = 1.20 if e.get("charging") else 1.0
        score = kind * threat * (0.65 + 0.35*clamp(1-hp/1400, 0, 1)) / (18+d)
        candidates.append((score, e))
    return max(candidates, key=lambda x:x[0])[1] if candidates else None

def predict_target(me, target, weapon, config):
    muzzle_h = float(me.get("stats", {}).get("muzzle_height", 7))
    origin = (me["pos"][0], me["pos"][1], me["pos"][2]+muzzle_h)
    tp = target["pos"]
    target_h = 4.5 if target["kind"] == "mech" else 1.75
    aimpoint = (tp[0], tp[1], tp[2]+target_h)
    r = tuple(aimpoint[i]-origin[i] for i in range(3))
    vel = tuple(target.get("vel", [0,0,0]))
    kind = weapon["kind"]
    stats = weapon.get("stats", {})
    if kind == "gun":
        t = min(1.2, intercept_time(r, vel, float(stats.get("speed", 252))))
        q = tuple(r[i]+vel[i]*t for i in range(3))
        return math.atan2(q[1], q[0]), math.atan2(q[2], math.hypot(q[0],q[1])), dist3(origin, aimpoint), q
    if kind == "mortar":
        horizontal = math.hypot(r[0], r[1])
        speed = float(stats.get("speed", 72))
        gravity = float(stats.get("gravity", 22))
        # Iteratively account for target horizontal movement using approximate flight time.
        yaw = math.atan2(r[1], r[0])
        pitch = ballistic_pitch(horizontal, r[2], speed, gravity)
        if pitch is None:
            return yaw, 0.0, dist3(origin, aimpoint), r
        for _ in range(2):
            ct = math.cos(pitch)
            flight = horizontal / max(1e-3, speed*ct)
            qx, qy = r[0]+vel[0]*flight, r[1]+vel[1]*flight
            qz = r[2]+vel[2]*flight + 0.5*gravity*flight*flight
            yaw = math.atan2(qy, qx)
            pitch = ballistic_pitch(math.hypot(qx,qy), qz, speed, gravity)
            if pitch is None:
                break
        return yaw, pitch or 0.0, dist3(origin, aimpoint), r
    # Laser and lightning are instant/hitscan.
    return math.atan2(r[1],r[0]), math.atan2(r[2],math.hypot(r[0],r[1])), dist3(origin,aimpoint), r

def projectile_dodge(obs, me):
    """Return a small lateral impulse if a visible projectile is on a near collision course."""
    p = me["pos"]
    vx, vy = me.get("vel", [0,0,0])[:2]
    best = 0.0
    dodge = (0.0, 0.0)
    for shot in obs.get("projectiles", []):
        q, v = shot["pos"], shot["vel"]
        rx, ry = p[0]-q[0], p[1]-q[1]
        vv = v[0]*v[0]+v[1]*v[1]
        if vv < 1:
            continue
        t = clamp(-(rx*v[0]+ry*v[1])/vv, 0, 0.8)
        cx, cy = q[0]+v[0]*t, q[1]+v[1]*t
        sep = math.hypot(p[0]+vx*t-cx, p[1]+vy*t-cy)
        if sep < 8 and t < 0.65:
            strength = (8-sep)/8 * (1-t/0.8)
            side = 1 if (rx*v[1]-ry*v[0]) >= 0 else -1
            n = math.hypot(v[0],v[1]) or 1
            candidate = (-v[1]/n*side, v[0]/n*side)
            if strength > best:
                best, dodge = strength, candidate
    return dodge, best

def combat_action(obs, me, target, config):
    aim = list(me.get("aim", [0,0]))
    fire = [False, False]
    weapons = me.get("weapons", [None, None])
    target_info = None
    if target:
        for slot, w in enumerate(weapons[:2]):
            if not w:
                continue
            yaw, pitch, distance, rel = predict_target(me, target, w, config)
            pitch = clamp(pitch, config["mech"]["min_pitch"], config["mech"]["max_pitch"])
            # Command the target solution; fire only when the actual slewed torso is aligned.
            if slot == 0 or target_info is None:
                aim = [yaw, pitch]
            yaw_err = abs(angle_delta(yaw, me["aim"][0]))
            pitch_err = abs(pitch-me["aim"][1])
            aligned = yaw_err < 0.075 and pitch_err < 0.065
            stats = w.get("stats", {})
            cooldown = float(w.get("cooldown", 0))
            if w["kind"] == "gun":
                ammo = int(w.get("ammo", 0))
                reloading = float(w.get("reload", 0))
                in_range = distance < float(stats.get("range", 230))*0.97
                fire[slot] = aligned and in_range and cooldown <= 0.035 and ammo > 0 and reloading <= 0
            elif w["kind"] == "laser":
                charge = float(w.get("charge", 0))
                max_charge = float(stats.get("charge", 0.75))
                min_charge = float(stats.get("min_charge", 0.15))
                in_range = distance < float(stats.get("range", 280))*0.98
                # Charge while tracking, then release on full charge or if target is about to escape.
                fire[slot] = in_range and aligned and charge < max_charge - 0.015
                if charge >= min_charge and (charge >= max_charge-0.015 or not in_range):
                    fire[slot] = False
            elif w["kind"] == "lightning":
                in_range = distance < float(stats.get("range", 45))*0.95
                fire[slot] = in_range and aligned and cooldown <= 0.035 and me["energy"] >= float(stats.get("energy",3))
            elif w["kind"] == "mortar":
                # Avoid self-splash. Require a clear ballistic solution and modest target range.
                horizontal = math.hypot(rel[0], rel[1])
                pitch_ok = -0.05 <= pitch <= 1.15
                safe = horizontal > 18
                fire[slot] = aligned and safe and pitch_ok and distance < 225 and cooldown <= 0.035
            if target_info is None:
                target_info = (target["id"], distance, yaw_err, pitch_err)
    # Gun reload when the magazine is low and no immediate shot is ready.
    reload_now = False
    if weapons and weapons[0] and weapons[0]["kind"] == "gun":
        w = weapons[0]
        reload_now = (w.get("ammo", 12) <= 3 and w.get("reload", 0) <= 0 and
                      (not target or dist3(me["pos"], target["pos"]) > 55))
    return aim, fire, reload_now, target_info
