"""Approximate autonomous NPC drones."""
from dataclasses import dataclass, field
import math

from geometry import dist2


@dataclass
class NPC:
    id: int
    pos: list
    home: list
    hp: float
    vel: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    aim: list = field(default_factory=lambda: [0.0, 0.0])
    target: int | None = None
    goal: list | None = None
    next_wander: float = 0.0
    cooldown: float = 0.0
    last_combat: float = 0.0
    events: list = field(default_factory=list)
    hits: int = 0
    shots: int = 0
    kills: int = 0
    damage: float = 0.0


class NPCs:
    def __init__(self, sim):
        self.sim = sim
        cfg = sim.config["npc"]
        self.actors = []
        for i in range(int(cfg["count"])):
            pos = sim.loot.spawn()
            self.actors.append(NPC(100 + i, pos, pos[:], float(cfg["hp"])))

    def step(self):
        sim, cfg = self.sim, self.sim.config["npc"]
        for npc in self.actors:
            if npc.hp <= 0:
                continue
            npc.cooldown = max(0.0, npc.cooldown - sim.dt)
            target = next((m for m in sim.mechs if m.id == npc.target and m.place is None and m.hp > 0), None)
            if target and (dist2(npc.pos, target.pos) > cfg["deaggro_range"]
                           or dist2(npc.pos, npc.home) > cfg["chase_leash"]):
                target = None
                npc.target = None
            if target is None and dist2(npc.pos, npc.home) <= cfg["leash"]:
                visible = [m for m in sim.mechs if m.place is None and m.hp > 0
                           and dist2(npc.pos, m.pos) <= cfg["aggro_range"]
                           and sim.clear([npc.pos[0], npc.pos[1], npc.pos[2] + cfg["height"] / 2],
                                         [m.pos[0], m.pos[1], m.pos[2] + sim.mech_stats(m)["height"] / 2],
                                         sight=True)]
                if visible:
                    target = min(visible, key=lambda m: (dist2(npc.pos, m.pos), m.id))
                    npc.target = target.id
            if target:
                goal = target.pos
            elif dist2(npc.pos, npc.home) > cfg["leash"]:
                goal = npc.home
            else:
                if npc.goal is None or sim.time >= npc.next_wander:
                    angle = sim.rng.uniform(-math.pi, math.pi)
                    distance = math.sqrt(sim.rng.random()) * cfg["leash"]
                    npc.goal = [npc.home[0] + distance * math.cos(angle),
                                npc.home[1] + distance * math.sin(angle), npc.home[2]]
                    npc.next_wander = sim.time + 5.0
                goal = npc.goal
            dx, dy = goal[0] - npc.pos[0], goal[1] - npc.pos[1]
            distance = math.hypot(dx, dy)
            if distance > 1:
                step = min(distance, cfg["speed"] * sim.dt)
                nxt = [npc.pos[0] + dx / distance * step, npc.pos[1] + dy / distance * step, 0.0]
                nxt[2] = sim.world.ground(nxt[0], nxt[1])
                if not sim.world.blocked(npc.pos, nxt, cfg["radius"]):
                    npc.vel = [(nxt[i] - npc.pos[i]) / sim.dt for i in range(3)]
                    npc.pos = nxt
                else:
                    npc.vel = [0.0, 0.0, 0.0]
                    npc.next_wander = sim.time
            else:
                npc.vel = [0.0, 0.0, 0.0]
            if target:
                own = [npc.pos[0], npc.pos[1], npc.pos[2] + cfg["height"] / 2]
                center = [target.pos[0], target.pos[1], target.pos[2] + sim.mech_stats(target)["height"] / 2]
                dx, dy, dz = (center[i] - own[i] for i in range(3))
                npc.aim = [math.atan2(dy, dx), math.atan2(dz, math.hypot(dx, dy))]
                if npc.cooldown <= 0 and sim.clear(own, center):
                    sim.fire_npc(npc, own)
                    npc.cooldown = cfg["cooldown"]

    def die(self, npc):
        for _ in range(int(self.sim.config["npc"]["cubes"])):
            self.sim.loot.add_cube(npc.pos)
        self.sim.sounds.append(("death", npc.id, npc.pos[:], self.sim.time))
