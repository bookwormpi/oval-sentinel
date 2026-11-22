"""Threat-weighted targeting and weapon-aware combat for Mech Royale."""
from __future__ import annotations
import math
from geometry import clamp, dist2, dist3, wrap_angle, angle_delta, intercept_time, ballistic_pitch, slew_angle


def _bearing_to(a, b):
    return math.atan2(b[1] - a[1], b[0] - a[0])


def _target_height(target, config):
    kind = target.get("kind")
    if kind == "mech":
        size = 1.0 + float(config.get("cubes", {}).get("size_per_cube", 0.05)) * float(target.get("cubes", 0))
        return float(config.get("mech", {}).get("height", 9.0)) * size
    if kind == "npc":
        return float(config.get("npc", {}).get("height", 3.5))
    return float(config.get("crates", {}).get("height", 2.4))


def choose_target(obs, me, recent_attacker=None, current_target_id=None):
    """Prefer enemies who are close, aiming at us, charging, recently damaging us,
    or near death (finishing them denies their HP/energy regen for good)."""
    candidates = []
    mp = me["pos"]
    for e in obs.get("mechs", []):
        if not isinstance(e, dict) or not isinstance(e.get("pos"), list):
            continue
        d = dist3(mp, e["pos"])
        if d > 280:
            continue
        kind = 1.0 if e.get("kind") == "mech" else 0.62
        # Cube stacking raises max HP, so a fixed baseline under- or over-states
        # how close a buffed enemy actually is to death; use its real max HP.
        max_hp = float(e.get("max_hp", 1400) or 1400)
        hp = float(e.get("hp", 0.64 * max_hp))
        hp_frac = clamp(hp / max(1.0, max_hp), 0.0, 1.0)
        low_hp = 1.0 + 0.32 * (1.0 - hp_frac)
        if hp_frac < 0.15:
            # One more solid hit likely kills; a target left alive this low fully
            # heals back within regen_delay + a few seconds of disengagement, so
            # committing to the kill is usually worth more than switching targets.
            low_hp *= 2.0
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


def predict_target(me, target, weapon, config, correct_mortar=False):
    stats_me = me.get("stats", {})
    muzzle_h = (float(stats_me.get("height", config.get("mech", {}).get("height", 9.0))) + 0.5
                if correct_mortar and weapon["kind"] == "mortar" else
                float(stats_me.get("muzzle_height", 7.0)))
    origin = (me["pos"][0], me["pos"][1], me["pos"][2] + muzzle_h)
    tp = target["pos"]
    target_h = (_target_height(target, config) * 0.5 if correct_mortar else
                4.5 if target.get("kind") == "mech" else 1.75)
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
            return yaw, None if correct_mortar else 0.0, dist3(origin, aimpoint), r
        for _ in range(6 if correct_mortar else 3):
            flight = horizontal / max(1e-3, speed * math.cos(pitch))
            qx = r[0] + relative_vel[0] * flight
            qy = r[1] + relative_vel[1] * flight
            qz = r[2] + relative_vel[2] * flight
            if not correct_mortar:
                qz -= 0.5 * gravity * flight * flight
            yaw = math.atan2(qy, qx)
            predicted_horizontal = math.hypot(qx, qy)
            next_pitch = ballistic_pitch(predicted_horizontal, qz, speed, gravity)
            if next_pitch is None:
                if correct_mortar:
                    return yaw, None, dist3(origin, aimpoint), r
                break
            pitch = next_pitch
            if correct_mortar:
                horizontal = predicted_horizontal
        return yaw, pitch, dist3(origin, aimpoint), r
    # Laser and lightning are hitscan; lightning only uses horizontal yaw for its cone.
    return math.atan2(r[1], r[0]), math.atan2(r[2], max(1e-6, math.hypot(r[0], r[1]))), dist3(origin, aimpoint), r


def _weapon_aim_priority(w, distance, precision=False):
    kind = w["kind"]
    stats = w.get("stats", {})
    if kind == "lightning":
        return 180 if distance <= float(stats.get("range", 45)) * 0.98 else -1
    if kind == "laser":
        charge = float(w.get("charge", 0))
        if charge > 0.01:
            return 240
        if precision and (float(w.get("cooldown", 0)) > 0
                          or distance >= float(stats.get("range", 280)) * 0.98):
            return -1
        return 125 if distance >= 65 and distance <= float(stats.get("range", 280)) * 0.98 else 75
    if kind == "gun":
        if int(w.get("ammo", 1)) <= 0 or float(w.get("reload", 0)) > 0:
            return 35
        if precision and distance >= float(stats.get("range", 230)) * 0.985:
            return -1
        return 110 if distance <= float(stats.get("range", 230)) else 95
    if kind == "mortar":
        # Use the mortar as a fallback when the gun is empty/reloading; avoid
        # trading reliable gun damage for a slow, self-splashing shell.
        speed = float(stats.get("speed", 72))
        gravity = float(stats.get("gravity", 22))
        maximum = float(stats.get("range", speed * speed / max(1e-6, gravity))) if precision else 215
        return 72 if 28 < distance < maximum else -1
    return -1


def combat_action(obs, me, target, config, world=None, correct_mortar=False, precision=False, commanded_aim=None):
    """Choose the best weapon's aim; independently gate each trigger on actual alignment."""
    actual = me.get("aim", [0.0, 0.0])
    aim = [float(actual[0]), float(actual[1])]
    fire = [False, False]
    weapons = me.get("weapons", [None, None])
    candidates = []
    solutions = {}
    if target:
        target_pos = target["pos"]
        aimpoint = (target_pos[0], target_pos[1], target_pos[2] +
                    (_target_height(target, config) * 0.5 if correct_mortar else
                     4.5 if target.get("kind") == "mech" else 1.75))
        origin = (me["pos"][0], me["pos"][1],
                  me["pos"][2] + float(me.get("stats", {}).get("muzzle_height", 7.0)))
        has_los = world is None or world.line_of_sight(origin, aimpoint)
        for slot, w in enumerate(weapons[:2]):
            if not w:
                continue
            yaw, pitch, distance, rel = predict_target(me, target, w, config, correct_mortar)
            if pitch is None:
                continue
            commanded_pitch = clamp(pitch, float(config["mech"].get("min_pitch", -0.7)),
                                    float(config["mech"].get("max_pitch", 1.3)))
            solutions[slot] = (yaw, pitch, distance, rel)
            priority = _weapon_aim_priority(w, distance, precision)
            if priority >= 0:
                candidates.append((priority, slot, yaw, commanded_pitch))
        if candidates:
            _, _, yaw, pitch = max(candidates, key=lambda x: x[0])
            aim = [yaw, pitch]
        if commanded_aim is not None:
            aim = commanded_aim

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
            if precision:
                # Actions are held for two simulation ticks; use the observed
                # torso and the actual command, not this weapon's hypothetical aim.
                yaw_err = max(abs(angle_delta(yaw, float(actual[0]))),
                              abs(angle_delta(yaw, aim[0])))
                pitch_err = max(abs(pitch - float(actual[1])), abs(pitch - aim[1]))
            yaw_limit = pitch_limit = 0.075
            if precision and w["kind"] in ("gun", "laser"):
                target_kind = target.get("kind")
                if target_kind == "mech":
                    size = 1.0 + float(config.get("cubes", {}).get("size_per_cube", 0.05)) * float(target.get("cubes", 0))
                    radius = float(config.get("mech", {}).get("radius", 2.5)) * size
                    height = _target_height(target, config)
                elif target_kind == "npc":
                    radius = float(config.get("npc", {}).get("radius", 1.8))
                    height = float(config.get("npc", {}).get("height", 3.5))
                else:
                    radius = float(config.get("crates", {}).get("radius", 1.4))
                    height = float(config.get("crates", {}).get("height", 2.4))
                horizontal = math.hypot(target_pos[0] - origin[0], target_pos[1] - origin[1])
                yaw_limit = min(yaw_limit, math.atan2(radius, max(1.0, horizontal)))
                pitch_limit = min(pitch_limit, math.atan2(height * 0.5, max(1.0, horizontal)))
            aligned = yaw_err < yaw_limit and pitch_err < pitch_limit
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
                    fire[slot] = bool(precision and in_range and direct_weapon_has_los and not aligned)
                else:
                    # Once charging starts, keep holding while tracking the target;
                    # requiring perfect alignment every frame causes premature release.
                    fire[slot] = bool(direct_weapon_has_los and in_range and (aligned or charge > 0.01))
                # The neural policy's commanded aim can diverge from the solved aim this
                # tick; keep holding rather than releasing off-target, but only while the
                # shot would still be viable (in range and not blocked), so a charge isn't
                # held forever (and eventually force-released by hold_s) while unusable.
                if commanded_aim is not None and charge > 0 and not aligned and in_range and direct_weapon_has_los:
                    fire[slot] = True
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
                max_range = float(stats.get("range", float(stats.get("speed", 72)) ** 2 /
                                            max(1e-6, float(stats.get("gravity", 22))))) if precision else 215
                fire[slot] = bool(aligned and safe and pitch_ok and distance < max_range and cooldown <= 0.04)

    reload_now = False
    gun = next((w for w in weapons if w and w["kind"] == "gun"), None)
    if gun:
        w = gun
        ammo = int(w.get("ammo", 12))
        reloading = float(w.get("reload", 0))
        close = bool(target and dist3(me["pos"], target["pos"]) < 65)
        reload_now = ammo <= 3 and reloading <= 0 and (ammo == 0 or not close)
    return aim, fire, reload_now, target


def projectile_dodge(obs, me, correct_side=False, config=None):
    """Predict 3-D projectile proximity and return a lateral escape vector + urgency."""
    p = me["pos"]
    mv = me.get("vel", [0, 0, 0])
    stats = me.get("stats", {})
    height = float(stats.get("height", 9.0))
    body_radius = float(stats.get("radius", 2.5))
    config = config or {}
    mortar = config.get("weapons", {}).get("mortar", {})
    gravity = float(mortar.get("gravity", 22.0))
    splash_radius = float(mortar.get("splash_radius", 7.0))
    best, dodge = 0.0, (0.0, 0.0)
    for shot in obs.get("projectiles", []):
        q, v = shot["pos"], shot["vel"]
        rx, ry = p[0] - q[0], p[1] - q[1]
        vx, vy = v[0] - mv[0], v[1] - mv[1]
        vv = vx * vx + vy * vy
        splash = shot.get("kind") == "mortar"
        projectile_radius = float(config.get("projectile", {}).get(
            "npc_radius" if shot.get("kind") == "npc" else "gun_radius", 0.3))
        radius = (max(9.0, splash_radius + body_radius) if splash else
                  max(5.2, body_radius + projectile_radius + 1.0))
        vertical_limit = splash_radius if splash else projectile_radius + 1.5

        def vertical_gap_at(t):
            shot_z = q[2] + v[2] * t - (0.5 * gravity * t * t if splash else 0.0)
            feet_z = p[2] + mv[2] * t
            return max(feet_z - shot_z, shot_z - (feet_z + height), 0.0)

        if vv < 1e-6:
            if math.hypot(rx, ry) >= radius:
                continue
            t = next((i * 0.05 for i in range(14) if vertical_gap_at(i * 0.05) <= vertical_limit), None)
            if t is None:
                continue
        else:
            raw_t = (rx * vx + ry * vy) / vv
            if raw_t < 0:
                continue
            t = clamp(raw_t, 0.0, 0.75)
        sx = q[0] + v[0] * t
        sy = q[1] + v[1] * t
        mx = p[0] + mv[0] * t
        my = p[1] + mv[1] * t
        horizontal = math.hypot(mx - sx, my - sy)
        if vertical_gap_at(t) > vertical_limit:
            continue
        if horizontal < radius and t < 0.7:
            strength = (radius - horizontal) / radius * (1.0 - t / 0.85)
            if vv < 1e-6:
                candidate = (rx / horizontal, ry / horizontal) if horizontal > 1e-6 else (1.0, 0.0)
            else:
                side = 1.0 if (rx * vy - ry * vx) >= 0 else -1.0
                if correct_side:
                    side = -side
                n = math.sqrt(vv)
                candidate = (-vy / n * side, vx / n * side)
            if strength > best:
                best, dodge = strength, candidate
    return dodge, clamp(best, 0.0, 1.0)
