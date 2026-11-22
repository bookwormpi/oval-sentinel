"""Hierarchical controller: survive first, then take favorable combat opportunities."""
from __future__ import annotations
import math
import mechfight as mf
from world import World
from navigation import choose_goal, steer
from combat import choose_target, combat_action, projectile_dodge
from strategy import choose_upgrade, choose_take, survival_boost, should_jump
from geometry import dist2, unit2, angle_delta


class MechRoyaleBot:
    name = "Oval Sentinel II"

    def __init__(self, hello):
        self.hello = hello
        self.config = hello["config"]
        self.world = World(hello.get("map", {}).get("dir"))
        self.last_attacker = None
        self.last_damage_t = -999.0
        self.tick_count = 0
        self.current_target_id = None

    def _threat_vector(self, obs, me):
        """Repulsion from nearby attackers and recent weapon sounds."""
        x, y = me["pos"][:2]
        rx = ry = 0.0
        danger = 0.0
        nearby = 0
        for e in obs.get("mechs", []):
            if e.get("id") == me.get("id"):
                continue
            p = e.get("pos", [0, 0, 0])
            dx, dy = x-p[0], y-p[1]
            d = math.hypot(dx, dy)
            if d > 72 or d < 1e-3:
                continue
            nearby += 1
            bearing_to_us = math.atan2(y-p[1], x-p[0])
            enemy_aim = e.get("aim", [0.0, 0.0])
            aiming = abs(angle_delta(float(enemy_aim[0]), bearing_to_us)) < 0.55
            kind_weight = 0.65 if e.get("kind") == "npc" else 1.0
            weight = kind_weight * (72-d)/72
            if aiming:
                weight *= 1.75
            if e.get("charging"):
                weight *= 1.45
            if e.get("id") == self.last_attacker and obs.get("t", 0)-self.last_damage_t < 2.5:
                weight *= 1.8
            rx += dx/d*weight
            ry += dy/d*weight
            danger = max(danger, weight)

        # Sounds have no source ID, but a recent nearby weapon sound is actionable.
        weapon_sounds = {"gun", "mortar", "explosion", "laser", "laser_charge", "lightning", "npc_gun"}
        for s in obs.get("sounds", []):
            if s.get("kind") not in weapon_sounds or float(s.get("age", 9)) > 0.8:
                continue
            dist_class = s.get("dist")
            strength = 0.55 if dist_class == "near" else (0.22 if dist_class == "mid" else 0.08)
            if s.get("kind") in ("mortar", "explosion", "laser"):
                strength *= 1.35
            b = float(s.get("bearing", 0))
            rx -= math.cos(b)*strength
            ry -= math.sin(b)*strength
            danger = max(danger, strength)

        # A recent damage event is much stronger evidence than a sound bearing.
        for ev in obs.get("events", []):
            if ev.get("type") == "damaged":
                b = ev.get("bearing")
                if b is not None and ev.get("weapon") != "zone":
                    strength = min(2.2, 0.65 + float(ev.get("amount", 0))/65.0)
                    rx -= math.cos(float(b))*strength
                    ry -= math.sin(float(b))*strength
                    danger = max(danger, min(1.0, strength/1.5))
        return unit2(rx, ry), min(1.0, danger), nearby

    def act(self, obs):
        self.tick_count += 1
        me = obs["self"]
        now = float(obs.get("t", 0))
        for event in obs.get("events", []):
            if event.get("type") == "damaged" and event.get("by") is not None:
                self.last_attacker = event["by"]
                self.last_damage_t = now

        recent_attacker = self.last_attacker if now-self.last_damage_t < 2.5 else None
        target = choose_target(obs, me, recent_attacker, self.current_target_id)
        self.current_target_id = target.get("id") if target else None
        goal, goal_kind = choose_goal(obs, self.world)
        base_direction, waypoint = steer(obs, self.world, goal)
        goal_distance = dist2(me["pos"][:2], goal[:2])

        evasion, danger, nearby = self._threat_vector(obs, me)
        if nearby >= 2:
            danger = max(danger, 0.60)
        dodge, projectile_urgency = projectile_dodge(obs, me)
        urgency = max(danger, projectile_urgency)

        # Zone recovery outranks combat evasion; otherwise break away from focused attackers.
        if not me.get("in_zone", True):
            mx, my = unit2(base_direction[0]*1.8 + evasion[0]*0.25,
                           base_direction[1]*1.8 + evasion[1]*0.25)
            move_scale = 1.0
        elif urgency > 0.12:
            evade_weight = 0.75 + 0.65*urgency + (0.25 if nearby >= 2 else 0.0)
            mx, my = unit2(base_direction[0]*0.55 + evasion[0]*evade_weight + dodge[0]*projectile_urgency*1.8,
                           base_direction[1]*0.55 + evasion[1]*evade_weight + dodge[1]*projectile_urgency*1.8)
            move_scale = 0.95 if nearby >= 2 or projectile_urgency > 0.45 else 0.72
        elif goal_kind == "patrol":
            # Slow patrol remains below the sound-producing step threshold.
            mx, my = base_direction
            move_scale = 0.22
        else:
            mx, my = base_direction
            move_scale = 1.0

        # If a recent attacker is behind us, face it to eliminate the 1.5x rear-hit penalty.
        aim, fire, reload_now, target_info = combat_action(obs, me, target, self.config, self.world)
        for event in obs.get("events", []):
            if event.get("type") == "damaged" and event.get("bearing") is not None:
                if event.get("rear") or target is None:
                    aim = [float(event["bearing"]), aim[1]]

        # With no visible target, scan recent sound bearings; otherwise sweep the horizon.
        if target is None:
            recent_sounds = [s for s in obs.get("sounds", []) if float(s.get("age", 9)) < 1.2]
            if recent_sounds:
                s = min(recent_sounds, key=lambda q: float(q.get("age", 9)))
                aim = [float(s["bearing"]), 0.0]
            elif not any(e.get("type") == "damaged" for e in obs.get("events", [])):
                phase = now*0.42 + float(me.get("id", 0))*2.399963
                aim = [(phase + math.pi) % (2*math.pi) - math.pi, 0.0]

        # Keep commanded movement magnitude intentional (the protocol uses it as speed fraction).
        move = [round(mx*move_scale, 4), round(my*move_scale, 4)]
        action = {
            "tick": obs["tick"],
            "move": move,
            "boost": survival_boost(obs, goal_distance, urgency),
            "jump": should_jump(obs, max(urgency, projectile_urgency)),
            "aim": [round(float(aim[0]), 5), round(float(aim[1]), 5)],
            "fire": fire,
            "reload": reload_now,
        }
        pick = choose_upgrade(obs, self.hello)
        if pick is not None:
            action["pick"] = pick
        take = choose_take(obs)
        if take is not None:
            action["take"] = take

        # Debug drawings are sparse and small; they do not influence the match.
        if self.tick_count % 10 == 0:
            mf.draw.label(f"{goal_kind} | hp {int(me['hp'])} | threat {urgency:.1f}")
            mf.draw.point([goal[0], goal[1], goal[2]], "#40c0ff", 4)
            if target:
                tp = target["pos"]
                mf.draw.line([me["pos"][0], me["pos"][1], me["pos"][2]+7],
                             [tp[0], tp[1], tp[2]+4.5], "#ff5040")
        return action
