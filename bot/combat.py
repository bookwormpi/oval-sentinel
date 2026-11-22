"""Threat-weighted targeting and weapon-aware combat for Mech Royale."""
from __future__ import annotations
import math
from geometry import clamp, dist2, dist3, wrap_angle, angle_delta, intercept_time, ballistic_pitch, slew_angle


def _bearing_to(a, b):
    return math.atan2(b[1] - a[1], b[0] - a[0])


def choose_target(obs, me, recent_attacker=None, current_target_id=None):
    """Prefer enemies who are close, aiming at us, charging, or recently damaging us."""
    candidates = []
    mp = me["pos"]
    for e in obs.get("mechs", []):
        if not isinstance(e, dict) or not isinstance(e.get("pos"), list):
            continue
        d = dist3(mp, e["pos"])
        if d > 280:
            continue
        kind = 1.0 if e.get("kind") == "mech" else 0.62
        hp = float(e.get("hp", 900) or 900)
        low_hp = 1.0 + 0.32 * clamp(1.0 - hp / 1400.0, 0.0, 1.0)
        bearing = _bearing_to(e["pos"], mp)
        enemy_aim = e.get("aim", [0.0, 0.0])
        aim_error = abs(angle_delta(float(enemy_aim[0]), bearing))
        aiming_at_us = aim_error < 0.48
        threat = 1.0 + (0.75 if aiming_at_us else 0.0)
        if e.get("charging"):
            threat += 0.55
        if recent_attacker is not None and e.get("id") == recent_attacker:
            threat += 1.4
        # Close enemies are dangerous; the distance term also prevents chasing distant low-HP targets.
        score = kind * low_hp * threat / (22.0 + d)
        if d < 32:
            score *= 1.25
        if current_target_id is not None and e.get("id") == current_target_id:
            score *= 1.12
        candidates.append((score, e))
    return max(candidates, key=lambda x: x[0])[1] if candidates else None


def predict_target(me, target, weapon, config):
    stats_me = me.get("stats", {})
    muzzle_h = float(stats_me.get("muzzle_height", 7.0))
    origin = (me["pos"][0], me["pos"][1], me["pos"][2] + muzzle_h)
    tp = target["pos"]
    target_h = 4.5 if target.get("kind") == "mech" else 1.75
    aimpoint = (tp[0], tp[1], tp[2] + target_h)
    r = tuple(aimpoint[i] - origin[i] for i in range(3))
    vel = tuple(target.get("vel", [0, 0, 0]))
    own_vel = tuple(me.get("vel", [0, 0, 0]))
    relative_vel = tuple(vel[i] - own_vel[i] for i in range(3))
    kind = weapon["kind"]
    stats = weapon.get("stats", {})
    if kind == "gun":
        speed = float(stats.get("speed", 252))
        t = min(1.5, intercept_time(r, relative_vel, speed))
        q = tuple(r[i] + relative_vel[i] * t for i in range(3))
        return math.atan2(q[1], q[0]), math.atan2(q[2], max(1e-6, math.hypot(q[0], q[1]))), dist3(origin, aimpoint), q
    if kind == "mortar":
        horizontal = math.hypot(r[0], r[1])
        speed = float(stats.get("speed", 72))
        gravity = float(stats.get("gravity", 22))
        yaw = math.atan2(r[1], r[0])
        pitch = ballistic_pitch(horizontal, r[2], speed, gravity)
        if pitch is None:
            return yaw, 0.0, dist3(origin, aimpoint), r
        for _ in range(3):
            flight = horizontal / max(1e-3, speed * math.cos(pitch))
            qx = r[0] + relative_vel[0] * flight
            qy = r[1] + relative_vel[1] * flight
            qz = r[2] + relative_vel[2] * flight + 0.5 * gravity * flight * flight
            yaw = math.atan2(qy, qx)
            next_pitch = ballistic_pitch(math.hypot(qx, qy), qz, speed, gravity)
            if next_pitch is None:
                break
            pitch = next_pitch
        return yaw, pitch, dist3(origin, aimpoint), r
    # Laser and lightning are hitscan; lightning only uses horizontal yaw for its cone.
    return math.atan2(r[1], r[0]), math.atan2(r[2], max(1e-6, math.hypot(r[0], r[1]))), dist3(origin, aimpoint), r


def _weapon_aim_priority(w, distance):
    kind = w["kind"]
    stats = w.get("stats", {})
    if kind == "lightning":
        return 180 if distance <= float(stats.get("range", 45)) * 0.98 else -1
    if kind == "laser":
        charge = float(w.get("charge", 0))
        if charge > 0.01:
            return 240
        return 125 if distance >= 65 and distance <= float(stats.get("range", 280)) * 0.98 else 75
    if kind == "gun":
        if int(w.get("ammo", 1)) <= 0 or float(w.get("reload", 0)) > 0:
            return 35
        return 110 if distance <= float(stats.get("range", 230)) else 95
    if kind == "mortar":
        # Use the mortar as a fallback when the gun is empty/reloading; avoid
        # trading reliable gun damage for a slow, self-splashing shell.
        return 72 if 28 < distance < min(215, float(stats.get("range", 235))) else -1
    return -1


def combat_action(obs, me, target, config, world=None):
    """Choose the best weapon's aim; independently gate each trigger on actual alignment."""
    actual = me.get("aim", [0.0, 0.0])
    aim = [float(actual[0]), float(actual[1])]
    fire = [False, False]
    weapons = me.get("weapons", [None, None])
    candidates = []
    solutions = {}
    if target:
        target_pos = target["pos"]
        aimpoint = (target_pos[0], target_pos[1],
                    target_pos[2] + (4.5 if target.get("kind") == "mech" else 1.75))
        origin = (me["pos"][0], me["pos"][1],
                  me["pos"][2] + float(me.get("stats", {}).get("muzzle_height", 7.0)))
        has_los = world is None or world.line_of_sight(origin, aimpoint)
        for slot, w in enumerate(weapons[:2]):
            if not w:
                continue
            yaw, pitch, distance, rel = predict_target(me, target, w, config)
            pitch = clamp(pitch, float(config["mech"].get("min_pitch", -0.7)), float(config["mech"].get("max_pitch", 1.3)))
            solutions[slot] = (yaw, pitch, distance, rel)
            priority = _weapon_aim_priority(w, distance)
            if priority >= 0:
                candidates.append((priority, slot, yaw, pitch))
        if candidates:
            _, _, yaw, pitch = max(candidates, key=lambda x: x[0])
            aim = [yaw, pitch]

        for slot, w in enumerate(weapons[:2]):
            if not w or slot not in solutions:
                continue
            yaw, pitch, distance, rel = solutions[slot]
            stats = w.get("stats", {})
            dt = 0.1
            stats_me = me.get("stats", {})
            projected_yaw = slew_angle(float(actual[0]), yaw,
                                       float(stats_me.get("turn_rate", 4.0)), dt)
            projected_pitch = clamp(
                float(actual[1]) + clamp(pitch - float(actual[1]),
                                         -float(stats_me.get("pitch_rate", 3.0))*dt,
                                         float(stats_me.get("pitch_rate", 3.0))*dt),
                float(config["mech"].get("min_pitch", -0.7)),
                float(config["mech"].get("max_pitch", 1.3)))
            yaw_err = abs(angle_delta(yaw, projected_yaw))
            pitch_err = abs(pitch - projected_pitch)
            aligned = yaw_err < 0.075 and pitch_err < 0.075
            yaw_aligned = yaw_err < math.radians(float(stats.get("cone_deg", 30)) * 0.5)
            cooldown = float(w.get("cooldown", 0))
            kind = w["kind"]
            direct_weapon_has_los = has_los or kind == "mortar"
            if kind == "gun":
                fire[slot] = (direct_weapon_has_los and aligned and distance < float(stats.get("range", 230)) * 0.985
                              and cooldown <= 0.04 and int(w.get("ammo", 0)) > 0
                              and float(w.get("reload", 0)) <= 0)
            elif kind == "laser":
                charge = float(w.get("charge", 0))
                max_charge = float(stats.get("charge", 0.75))
                min_charge = float(stats.get("min_charge", 0.15))
                in_range = distance < float(stats.get("range", 280)) * 0.985
                # Holding charges; releasing at full charge fires along the actual aim.
                if charge >= max_charge - 0.015:
                    fire[slot] = False
                else:
                    # Once charging starts, keep holding while tracking the target;
                    # requiring perfect alignment every frame causes premature release.
                    fire[slot] = bool(direct_weapon_has_los and in_range and (aligned or charge > 0.01))
                # If target disappears, retain the charge until full rather than wasting it.
                if charge >= min_charge and not target:
                    fire[slot] = False
            elif kind == "lightning":
                in_range = distance < float(stats.get("range", 45)) * 0.98
                fire[slot] = (direct_weapon_has_los and in_range and yaw_aligned and cooldown <= 0.04
                              and float(me.get("energy", 0)) >= float(stats.get("energy", 3)))
            elif kind == "mortar":
                horizontal = math.hypot(rel[0], rel[1])
                pitch_ok = -0.05 <= pitch <= 1.15
                # Self-splash is severe; keep a generous minimum separation.
                safe = horizontal > 22
                fire[slot] = bool(aligned and safe and pitch_ok and distance < 215 and cooldown <= 0.04)

    reload_now = False
    if weapons and weapons[0] and weapons[0]["kind"] == "gun":
        w = weapons[0]
        ammo = int(w.get("ammo", 12))
        reloading = float(w.get("reload", 0))
        close = bool(target and dist3(me["pos"], target["pos"]) < 65)
        reload_now = ammo <= 3 and reloading <= 0 and (ammo == 0 or not close)
    return aim, fire, reload_now, target


def projectile_dodge(obs, me):
    """Predict 3-D projectile proximity and return a lateral escape vector + urgency."""
    p = me["pos"]
    mv = me.get("vel", [0, 0, 0])
    height = float(me.get("stats", {}).get("height", 9.0))
    best, dodge = 0.0, (0.0, 0.0)
    for shot in obs.get("projectiles", []):
        q, v = shot["pos"], shot["vel"]
        rx, ry = p[0] - q[0], p[1] - q[1]
        vx, vy = v[0] - mv[0], v[1] - mv[1]
        vv = vx * vx + vy * vy
        if vv < 1e-6:
            continue
        raw_t = (rx * vx + ry * vy) / vv
        # Ignore projectiles already moving away (including our own shots).
        if raw_t <= 0.02:
            continue
        t = clamp(raw_t, 0.0, 0.75)
        sx = q[0] + v[0] * t
        sy = q[1] + v[1] * t
        mx = p[0] + mv[0] * t
        my = p[1] + mv[1] * t
        horizontal = math.hypot(mx - sx, my - sy)
        shot_z = q[2] + v[2] * t
        if shot.get("kind") == "mortar":
            shot_z -= 0.5 * 22.0 * t * t
        feet_z = p[2] + mv[2] * t
        vertical_gap = max(feet_z - shot_z, shot_z - (feet_z + height), 0.0)
        splash = shot.get("kind") == "mortar"
        radius = 9.0 if splash else 5.2
        if vertical_gap > (7.0 if splash else 1.8):
            continue
        if horizontal < radius and t < 0.7:
            strength = (radius - horizontal) / radius * (1.0 - t / 0.85)
            # Move perpendicular to the projectile's horizontal path, away from its approach side.
            side = 1.0 if (rx * vy - ry * vx) >= 0 else -1.0
            n = math.hypot(vx, vy) or 1.0
            candidate = (-vy / n * side, vx / n * side)
            if strength > best:
                best, dodge = strength, candidate
    return dodge, clamp(best, 0.0, 1.0)
