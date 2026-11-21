"""Top-level hierarchical controller."""
from __future__ import annotations
import math
import mechfight as mf
from world import World
from navigation import choose_goal, steer
from combat import choose_target, combat_action, projectile_dodge
from strategy import choose_upgrade, choose_take, survival_boost, should_jump
from geometry import dist2, unit2

class MechRoyaleBot:
    name = "Oval Sentinel"

    def __init__(self, hello):
        self.hello = hello
        self.config = hello["config"]
        self.world = World(hello.get("map", {}).get("dir"))
        self.last_goal = None
        self.last_goal_kind = "spawn"
        self.last_replan = -999
        self.last_seen = {}
        self.tick_count = 0

    def act(self, obs):
        self.tick_count += 1
        me = obs["self"]
        target = choose_target(obs, me)
        goal, goal_kind = choose_goal(obs, self.world, self.last_seen)
        direction, waypoint = steer(obs, self.world, goal)

        # Reuse strategic objective for short intervals to avoid oscillating between loot.
        self.last_goal, self.last_goal_kind = goal, goal_kind
        mx, my = direction

        aim, fire, reload_now, target_info = combat_action(obs, me, target, self.config)
        dodge, urgency = projectile_dodge(obs, me)
        if urgency > 0.2:
            mx, my = unit2(mx + dodge[0]*urgency*1.5, my + dodge[1]*urgency*1.5)

        # If damaged from behind, face the attack bearing even if target is temporarily unseen.
        for event in obs.get("events", []):
            if event.get("type") == "damaged" and event.get("bearing") is not None:
                if event.get("rear") or not target:
                    aim = [event["bearing"], aim[1]]

        goal_distance = dist2(me["pos"][:2], goal[:2])
        action = {
            "tick": obs["tick"],
            "move": [round(mx, 4), round(my, 4)],
            "boost": survival_boost(obs, goal_distance),
            "jump": should_jump(obs, goal, waypoint),
            "aim": [round(aim[0], 5), round(aim[1], 5)],
            "fire": fire,
            "reload": reload_now,
        }
        pick = choose_upgrade(obs, self.hello)
        if pick is not None:
            action["pick"] = pick
        take = choose_take(obs)
        if take is not None:
            action["take"] = take

        # Do not hold a laser trigger after its full charge; release to fire.
        if target:
            for i,w in enumerate(me.get("weapons", [])[:2]):
                if w and w["kind"] == "laser":
                    if w.get("charge", 0) >= w.get("stats", {}).get("charge", .75)-.015:
                        action["fire"][i] = False

        # Debug is compact and within the kit's caps.
        if self.tick_count % 5 == 0:
            mf.draw.label(f"{goal_kind} | hp {int(me['hp'])} | alive {int(obs['alive'])}")
            mf.draw.point([goal[0], goal[1], goal[2]], "#40c0ff", 5)
            if target:
                tp = target["pos"]
                mf.draw.line([me["pos"][0],me["pos"][1],me["pos"][2]+7],
                             [tp[0],tp[1],tp[2]+4.5], "#ff5040")
            mf.draw.line([me["pos"][0],me["pos"][1],me["pos"][2]+1],
                         [waypoint[0],waypoint[1],waypoint[2]+1], "#ffd060")
        return action
