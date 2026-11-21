"""Lightweight opponent tracking and predictive threat estimation.

Uses an alpha-beta constant-velocity filter. Standard-library only and bounded
by the number of visible mechs (normally <= 15).
"""
from __future__ import annotations
import math


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _unit(x, y):
    n = math.hypot(x, y)
    return (x / n, y / n) if n > 1e-9 else (0.0, 0.0)


def _angle_delta(a, b):
    return (a - b + math.pi) % (2.0 * math.pi) - math.pi


class OpponentTracker:
    """Tracks visible opponents, smoothing noisy velocity and predicting intent."""
    def __init__(self, alpha=0.68, beta=0.22, horizon=0.65):
        self.alpha = alpha
        self.beta = beta
        self.horizon = horizon
        self.tracks = {}
        self.last_t = None

    def update(self, obs):
        now = float(obs.get("t", 0.0))
        dt_global = 0.0 if self.last_t is None else _clamp(now - self.last_t, 0.01, 0.5)
        self.last_t = now
        seen = set()
        me = obs.get("self", {})
        my_id = me.get("id")
        my_pos = me.get("pos", [0.0, 0.0, 0.0])
        for enemy in obs.get("mechs", []):
            if not isinstance(enemy, dict) or enemy.get("id") == my_id:
                continue
            pos = enemy.get("pos")
            if not isinstance(pos, list) or len(pos) < 3:
                continue
            eid = enemy.get("id")
            if eid is None:
                continue
            seen.add(eid)
            observed_v = enemy.get("vel", [0.0, 0.0, 0.0])
            old = self.tracks.get(eid)
            if old is None:
                filt_p = [float(v) for v in pos[:3]]
                filt_v = [float(v) for v in observed_v[:3]]
                dt = dt_global
            else:
                dt = _clamp(now - old["t"], 0.01, 0.75)
                predicted = [old["p"][i] + old["v"][i] * dt for i in range(3)]
                residual = [float(pos[i]) - predicted[i] for i in range(3)]
                filt_p = [predicted[i] + self.alpha * residual[i] for i in range(3)]
                measured_v = [residual[i] / dt for i in range(3)]
                # Blend server velocity with the alpha-beta residual estimate.
                filt_v = [0.55 * float(observed_v[i]) + 0.45 * (old["v"][i] + self.beta * measured_v[i]) for i in range(3)]
            speed = math.sqrt(sum(v*v for v in filt_v))
            if speed > 40.0:
                filt_v = [v * 40.0 / speed for v in filt_v]
            bearing = math.atan2(float(my_pos[1]) - filt_p[1], float(my_pos[0]) - filt_p[0])
            aim = enemy.get("aim", [0.0, 0.0])
            aim_error = abs(_angle_delta(float(aim[0]), bearing))
            distance = math.dist(my_pos[:3], filt_p)
            closing = 0.0
            if distance > 1e-5:
                closing = -sum((filt_p[i] - float(my_pos[i])) * (filt_v[i] - float(me.get("vel", [0,0,0])[i])) for i in range(3)) / distance
            self.tracks[eid] = {
                "p": filt_p, "v": filt_v, "t": now,
                "aiming": aim_error < 0.52,
                "charging": bool(enemy.get("charging")),
                "distance": distance, "closing": closing,
                "kind": enemy.get("kind", "mech"),
            }
        # Retain briefly through visibility gaps; discard stale records quickly.
        for eid in list(self.tracks):
            if eid not in seen and now - self.tracks[eid]["t"] > 1.8:
                del self.tracks[eid]
        return self.tracks

    def filtered_target(self, target):
        """Return a shallow copy with filtered velocity for projectile leading."""
        if not target:
            return None
        tr = self.tracks.get(target.get("id"))
        if not tr:
            return target
        result = dict(target)
        result["vel"] = list(tr["v"])
        return result

    def threat_vector(self, me, now=None):
        """Predict enemy positions and produce a smooth repulsive potential field."""
        p = me.get("pos", [0.0, 0.0, 0.0])
        rx = ry = 0.0
        danger = 0.0
        nearby = 0
        for tr in self.tracks.values():
            age = 0.0 if now is None else _clamp(float(now) - tr["t"], 0.0, 1.8)
            horizon = self.horizon + age * 0.35
            ep = [tr["p"][i] + tr["v"][i] * horizon for i in range(3)]
            dx, dy = float(p[0]) - ep[0], float(p[1]) - ep[1]
            d = math.hypot(dx, dy)
            if d > 85.0 or d < 1e-3:
                continue
            nearby += 1
            proximity = _clamp((85.0 - d) / 85.0, 0.0, 1.0)
            weight = proximity * (1.0 + 0.55 * max(0.0, tr["closing"]) / 12.0)
            if tr["aiming"]:
                weight *= 1.65
            if tr["charging"]:
                weight *= 1.55
            weight *= 0.72 if tr["kind"] == "npc" else 1.0
            rx += dx / d * weight
            ry += dy / d * weight
            danger = max(danger, min(1.0, weight))
        n = math.hypot(rx, ry)
        return ((rx / n, ry / n) if n > 1e-9 else (0.0, 0.0), danger, nearby)
