"""Approximate, deterministic mechfight self-play; not the tournament engine."""
from __future__ import annotations

from dataclasses import dataclass, field
import gzip
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bot"))

import mechfight
from agent import MechRoyaleBot
from check import Schema
from geometry import angle_delta, clamp, dist2, point_in_ring, segment_hits_prism, slew_angle
from navigation import zone_state
from world import World
from .loot import Loot
from .npc import NPC, NPCs
from .environment import Environment


def load_config(stream=ROOT / "stream.jsonl.gz"):
    with gzip.open(stream, "rt", encoding="utf-8") as source:
        for line in source:
            message = json.loads(line)
            if message.get("type") == "hello":
                return message["config"]
    raise ValueError(f"No hello message in {stream}")


def load_world(map_dir=ROOT / "arena"):
    root = Path(map_dir).resolve()
    for name in ("world.json", "terrain.i16"):
        if not (root / "v2" / name).is_file():
            raise FileNotFoundError(root / "v2" / name)
    world = World(str(root))
    if (not world.ready or world.root != root or world.terrain is None
            or len(world.terrain) != world.nx * world.ny):
        raise ValueError(f"Unable to load arena from {root}")
    return world


def cylinder_hit(start, end, feet, radius, height):
    """First segment fraction inside a vertical cylinder, or None."""
    dx, dy, dz = (end[i] - start[i] for i in range(3))
    x, y = start[0] - feet[0], start[1] - feet[1]
    a = dx * dx + dy * dy
    c = x * x + y * y - radius * radius
    if a < 1e-12:
        if c > 0:
            return None
        lo, hi = 0.0, 1.0
    else:
        b = 2 * (x * dx + y * dy)
        disc = b * b - 4 * a * c
        if disc < 0:
            return None
        root = math.sqrt(disc)
        lo, hi = max(0.0, (-b - root) / (2 * a)), min(1.0, (-b + root) / (2 * a))
    if abs(dz) < 1e-12:
        if not feet[2] <= start[2] <= feet[2] + height:
            return None
    else:
        z1 = (feet[2] - start[2]) / dz
        z2 = (feet[2] + height - start[2]) / dz
        lo, hi = max(lo, min(z1, z2)), min(hi, max(z1, z2))
    return lo if lo <= hi else None


@dataclass
class Weapon:
    kind: str
    cooldown: float = 0.0
    ammo: int = 0
    reload: float = 0.0
    reload_mag: int = 0
    charge: float = 0.0
    hold: float = 0.0
    release_pending: bool = False


@dataclass
class Mech:
    id: int
    strategy: str
    bot: object
    pos: list
    aim: list
    hp: float
    energy: float
    laser_short: float = 0.0
    vel: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    grounded: bool = True
    weapons: list = field(default_factory=lambda: [None, None])
    cubes: int = 0
    upgrades: list = field(default_factory=list)
    offer: dict | None = None
    pending_offers: int = 0
    last_combat: float = -1e9
    last_energy_use: float = -1e9
    last_move_sound: float = -1e9
    events: list = field(default_factory=list)
    memory: dict = field(default_factory=dict)
    action: dict = field(default_factory=dict)
    desired_aim: list = field(default_factory=list)
    place: int | None = None
    kills: int = 0
    shots: int = 0
    hits: int = 0
    damage: float = 0.0

    @property
    def ammo(self):
        return next(w.ammo for w in self.weapons if w and w.kind == "gun")

    @property
    def reload(self):
        return next(w.reload for w in self.weapons if w and w.kind == "gun")


@dataclass
class Projectile:
    id: int
    by: int
    pos: list
    vel: list
    remaining: float
    kind: str = "gun"
    damage: float = 0.0
    stats: dict = field(default_factory=dict)


class Simulator:
    def __init__(self, strategies, seed, config, world, bot_factory=None):
        if not 2 <= len(strategies) <= int(config["match"]["mechs"]):
            raise ValueError("A match needs between 2 and the configured maximum number of bots")
        self.strategies = list(strategies)
        self.seed = seed
        self.config = config
        self.world = world
        self.environment = Environment(self)
        self.rng = random.Random(seed)
        self.tick = 0
        self.dt = 1.0 / float(config["match"]["tick_hz"])
        self.decide_every = int(config["match"]["decide_every"])
        self.projectiles = []
        self.beams = []
        self.next_projectile_id = 1
        self.sounds = []
        self.action_schema = Schema(ROOT / "mechfight.schema.json")
        self.zone_phases = self._make_zones()
        self.mechs = []
        for slot, strategy in enumerate(strategies):
            pos = self._spawn()
            hello = {"type": "hello", "protocol": "mechfight/1", "slot": slot,
                     "seed": 1000 + slot, "map": {"dir": str(getattr(world, "root", ROOT / "arena")),
                     "files": ["v2/world.json", "v2/terrain.i16"]}, "config": config}
            bot = (bot_factory(hello, strategy, world) if bot_factory else
                   MechRoyaleBot(hello, strategy=strategy, world=world))
            mech = config["mech"]
            self.mechs.append(Mech(slot, strategy, bot, pos, [self.rng.uniform(-math.pi, math.pi), 0.0],
                                   float(mech["hp"]), float(mech["energy"])))
            self.mechs[-1].weapons[0] = self.new_weapon("gun")
            self.mechs[-1].desired_aim = self.mechs[-1].aim[:]
        self.loot = Loot(self)
        self.npcs = NPCs(self)

    def new_weapon(self, kind):
        return Weapon(kind, ammo=int(self.config["weapons"]["gun"]["mag"]) if kind == "gun" else 0)

    def modified_stats(self, mech, prefix, base):
        stats = dict(base)
        definitions = {u["id"]: u for u in self.config["upgrades"]}
        for upgrade in mech.upgrades:
            for effect in definitions[upgrade]["effects"]:
                path = effect["path"].split(".")
                if path[:-1] == prefix:
                    key = path[-1]
                    stats[key] = stats[key] * effect.get("mul", 1) + effect.get("add", 0)
        return stats

    def mech_stats(self, mech):
        stats = self.modified_stats(mech, ["mech"], self.config["mech"])
        cubes = self.config["cubes"]
        stats["hp"] *= 1 + cubes["hp_per_cube"] * min(mech.cubes, cubes["stat_cap"])
        size = 1 + cubes["size_per_cube"] * mech.cubes
        for key in ("radius", "height", "eye_height", "muzzle_height"):
            stats[key] *= size
        return stats

    def weapon_stats(self, mech, kind):
        stats = self.modified_stats(mech, ["weapons", kind], self.config["weapons"][kind])
        cubes = self.config["cubes"]
        multiplier = 1 + cubes["damage_per_cube"] * min(mech.cubes, cubes["stat_cap"])
        for key in ("damage", "splash"):
            if key in stats:
                stats[key] *= multiplier
        return stats

    @property
    def time(self):
        return self.tick * self.dt

    def clear(self, start, end, sight=False, canopy=True):
        return self.environment.clear(start, end, sight=sight, canopy=canopy)

    def entity(self, entity_id):
        if entity_id < 100:
            return self.mechs[entity_id]
        return next(n for n in self.npcs.actors if n.id == entity_id)

    def _make_zones(self):
        zone = self.config["zone"]
        center = list(zone["start_center"])
        radius = float(zone["start_radius"])
        start = 0.0
        phases = []
        for phase in zone["phases"]:
            next_radius = float(phase["radius"])
            angle = self.rng.uniform(-math.pi, math.pi)
            offset = math.sqrt(self.rng.random()) * max(0.0, radius - next_radius)
            next_center = [center[0] + math.cos(angle) * offset, center[1] + math.sin(angle) * offset]
            shrink_start = start + float(phase["hold_s"])
            end = shrink_start + float(phase["shrink_s"])
            phases.append({"center": center, "radius": radius, "next_center": next_center,
                           "next_radius": next_radius, "shrink_start": shrink_start,
                           "shrink_end": end, "dps": float(phase["dps"])})
            center, radius, start = next_center, next_radius, end
        if not phases:
            raise ValueError("At least one zone phase is required")
        return phases

    def zone(self):
        return next((z for z in self.zone_phases if self.time < z["shrink_end"]), self.zone_phases[-1])

    def _spawn(self):
        zone = self.config["zone"]
        fraction = float(self.config["match"]["spawn_zone_fraction"])
        separation = float(self.config["match"]["spawn_min_separation"])
        radius = float(self.config["mech"]["radius"])
        for _ in range(5000):
            x = self.rng.uniform(self.world.min_x + 30, self.world.min_x + self.world.nx * self.world.step - 30)
            y = self.rng.uniform(self.world.min_y + 30, self.world.min_y + self.world.ny * self.world.step - 30)
            if dist2((x, y), zone["start_center"]) > float(zone["start_radius"]) * fraction:
                continue
            if any(dist2((x, y), m.pos) < separation for m in self.mechs):
                continue
            pos = [x, y, self.world.ground(x, y)]
            if not self.world.blocked(pos, pos, radius):
                return pos
        raise RuntimeError("Could not find a valid spawn; check the map and spawn configuration")

    def _visible(self, viewer, pos):
        vision = self.config["vision"]
        stats = self.mech_stats(viewer)
        d = dist2(viewer.pos, pos)
        if d > float(vision["range"]) + stats["vision_bonus"]:
            return False
        bearing = math.atan2(pos[1] - viewer.pos[1], pos[0] - viewer.pos[0])
        if d > float(vision["near_range"]) + stats["near_range_bonus"] and abs(angle_delta(bearing, viewer.aim[0])) > math.radians(float(vision["fov_deg"]) / 2):
            return False
        eye = (viewer.pos[0], viewer.pos[1], viewer.pos[2] + stats["eye_height"])
        return self.clear(eye, pos, sight=True, canopy=d > float(vision["near_range"]) + stats["near_range_bonus"])

    def _visible_actor(self, viewer, pos, height):
        return (self._visible(viewer, [pos[0], pos[1], pos[2] + height / 2])
                or self._visible(viewer, [pos[0], pos[1], pos[2] + height]))

    def observation(self, viewer):
        config = self.config
        stats = self.mech_stats(viewer)
        zone = self.zone()
        center, radius = zone_state(zone, self.time)
        mechs = []
        for other in self.mechs:
            if other.id == viewer.id or other.place is not None:
                continue
            other_stats = self.mech_stats(other)
            if self._visible_actor(viewer, other.pos, other_stats["height"]):
                viewer.memory[other.id] = self.time
            if self.time - viewer.memory.get(other.id, -1e9) <= float(config["vision"].get("memory_s", 0)):
                actor = {"id": other.id, "kind": "mech", "pos": other.pos[:], "vel": other.vel[:],
                         "aim": other.aim[:], "cubes": other.cubes,
                         "weapons": [w.kind for w in other.weapons if w],
                         "charging": any(w and w.kind == "laser" and w.charge > 0 for w in other.weapons)}
                if config["vision"].get("show_enemy_hp", False):
                    actor.update({"hp": max(0.0, other.hp), "max_hp": other_stats["hp"]})
                mechs.append(actor)
        for npc in self.npcs.actors:
            if npc.hp <= 0:
                continue
            if self._visible_actor(viewer, npc.pos, config["npc"]["height"]):
                viewer.memory[npc.id] = self.time
            if self.time - viewer.memory.get(npc.id, -1e9) <= float(config["vision"].get("memory_s", 0)):
                actor = {"id": npc.id, "kind": "npc", "pos": npc.pos[:], "vel": npc.vel[:], "aim": npc.aim[:]}
                if config["vision"].get("show_enemy_hp", False):
                    actor.update({"hp": npc.hp, "max_hp": config["npc"]["hp"]})
                mechs.append(actor)
        projectiles = [{"id": p.id, "kind": p.kind, "pos": p.pos[:], "vel": p.vel[:]}
                       for p in self.projectiles if p.by != viewer.id and self._visible(viewer, p.pos)]
        sounds = []
        sound_config = config["sound"]
        for kind, by, pos, t in self.sounds:
            d = dist2(viewer.pos, pos)
            if by == viewer.id or d > sound_config["range"][kind]:
                continue
            bearing = math.atan2(pos[1] - viewer.pos[1], pos[0] - viewer.pos[0])
            bearing += self.rng.gauss(0, math.radians(float(sound_config["bearing_noise_deg"])) * stats["hearing"])
            sounds.append({"kind": kind, "bearing": bearing, "age": max(0.0, self.time - t),
                           "dist": "near" if d < sound_config["near"] else
                                   ("mid" if d < sound_config["mid"] else "far")})
        weapons = []
        for weapon in viewer.weapons:
            if weapon is None:
                weapons.append(None)
                continue
            item = {"kind": weapon.kind, "cooldown": max(0.0, weapon.cooldown),
                    "stats": self.weapon_stats(viewer, weapon.kind)}
            if weapon.kind == "gun":
                item.update({"reload": weapon.reload, "ammo": weapon.ammo})
            if weapon.kind == "laser":
                item["charge"] = weapon.charge
            weapons.append(item)
        own = {"id": viewer.id, "pos": viewer.pos[:], "vel": viewer.vel[:], "aim": viewer.aim[:],
               "hp": max(0.0, viewer.hp), "max_hp": stats["hp"], "energy": viewer.energy,
               "max_energy": stats["energy"], "grounded": viewer.grounded, "cubes": viewer.cubes,
               "upgrades": viewer.upgrades[:], "weapons": weapons, "stats": stats,
               "in_zone": dist2(viewer.pos, center) <= radius}
        crates, cubes, pickups, drops = self.loot.observations(viewer)
        return {"type": "obs", "tick": self.tick, "t": self.time,
                "alive": sum(m.place is None for m in self.mechs), "self": own,
                "mechs": mechs, "projectiles": projectiles, "sounds": sounds,
                "crates": crates, "cubes": cubes, "pickups": pickups, "drops": drops,
                "beams": [{"by": b["by"], "kind": b["kind"], "path": b["path"],
                           "age": max(0.0, self.time - b["time"])} for b in self.beams
                          if b["by"] != viewer.id and any(self._visible(viewer, point) for point in b["path"])],
                "events": list(viewer.events), "zone": dict(zone), "offer": viewer.offer}

    def _decide(self):
        active = [m for m in self.mechs if m.place is None]
        observations = [self.observation(m) for m in active]
        self.sounds.clear()
        self.beams.clear()
        for mech, obs in zip(active, observations):
            mech.events.clear()
            mechfight.draw.clear()
            action = mech.bot.act(obs)
            errors = self.action_schema.errors(action, "Action")
            if not errors and action.get("tick") != self.tick:
                errors.append("wrong tick")
            if not errors and any(not math.isfinite(value) for key in ("move", "aim")
                                  for value in action.get(key, [])):
                errors.append("move and aim must contain finite numbers")
            if not errors and any(not math.isfinite(action[key]) for key in ("tick", "pick", "take", "drop")
                                  if key in action):
                errors.append("tick, pick, take and drop must be finite numbers")
            if errors:
                raise ValueError(f"Bot {mech.id} returned an invalid action at tick {self.tick}: {errors}")
            mech.action = action
            if "aim" in action:
                mech.desired_aim = action["aim"][:]

    def _move(self, mech):
        stats = self.mech_stats(mech)
        action = mech.action
        move = action.get("move", [0.0, 0.0])
        mx, my = float(move[0]), float(move[1])
        length = math.hypot(mx, my)
        if length > 1:
            mx, my = mx / length, my / length
        boosting = bool(action.get("boost")) and mech.energy > 0 and (not mech.grounded or length > 0)
        speed = float(stats["boost_speed"] if boosting and mech.grounded else stats["walk_speed"])
        if any(w and w.kind == "laser" and w.charge > 0 for w in mech.weapons):
            speed *= stats["charge_move_mult"]
        accel = float(stats["ground_accel"] if mech.grounded else stats["air_accel"])
        dvx, dvy = mx * speed - mech.vel[0], my * speed - mech.vel[1]
        change = math.hypot(dvx, dvy)
        scale = min(1.0, accel * self.dt / change) if change else 0.0
        mech.vel[0] += dvx * scale
        mech.vel[1] += dvy * scale
        if action.get("jump") and mech.grounded:
            mech.vel[2] = float(stats["jump_speed"])
            mech.grounded = False
        if boosting:
            drain = float(stats["boost_drain"] if mech.grounded else stats["jet_drain"])
            mech.energy = max(0.0, mech.energy - drain * self.dt)
            mech.last_energy_use = self.time
            if not mech.grounded:
                mech.vel[2] = min(float(stats["max_rise_speed"]), mech.vel[2] + float(stats["jet_accel"]) * self.dt)
        elif self.time - mech.last_energy_use >= float(stats["energy_regen_delay"]):
            mech.energy = min(float(stats["energy"]), mech.energy + float(stats["energy_regen"]) * self.dt)
        if not mech.grounded:
            mech.vel[2] -= float(stats["gravity"]) * self.dt
        old = mech.pos[:]
        radius = float(stats["radius"])
        x = clamp(old[0] + mech.vel[0] * self.dt, self.world.min_x + radius,
                  self.world.min_x + (self.world.nx - 1) * self.world.step - radius)
        y = clamp(old[1] + mech.vel[1] * self.dt, self.world.min_y + radius,
                  self.world.min_y + (self.world.ny - 1) * self.world.step - radius)
        z = old[2] + mech.vel[2] * self.dt
        if z > old[2]:
            head = [old[0], old[1], old[2] + stats["height"]]
            head_end = [old[0], old[1], z + stats["height"]]
            ceiling = self.environment.impact(head, head_end)
            if ceiling < 1:
                z = old[2] + (z - old[2]) * ceiling
                mech.vel[2] = 0.0
        ground = self._surface(x, y, max(old[2], z) + (stats["step_up"] if mech.grounded else 0))
        if not self._walkable(mech, old, (x, y, max(z, ground)), ground, stats):
            options = []
            for nx, ny in ((x, old[1]), (old[0], y)):
                surface = self._surface(nx, ny, max(old[2], z) + (stats["step_up"] if mech.grounded else 0))
                if self._walkable(mech, old, (nx, ny, max(z, surface)), surface, stats):
                    options.append((math.hypot(nx - old[0], ny - old[1]), nx, ny, surface))
            if options:
                _, x, y, ground = max(options)
                if x == old[0]:
                    mech.vel[0] = 0.0
                if y == old[1]:
                    mech.vel[1] = 0.0
            else:
                x, y = old[0], old[1]
                ground = self._surface(x, y, max(old[2], z) + (stats["step_up"] if mech.grounded else 0))
                mech.vel[0] = mech.vel[1] = 0.0
        if z <= ground:
            z, mech.vel[2], mech.grounded = ground, 0.0, True
        else:
            mech.grounded = False
        mech.pos = [x, y, z]
        sound_cfg = self.config["sound"]
        if self.time - mech.last_move_sound >= sound_cfg["step_every"]:
            kind = None
            if mech.grounded and math.hypot(mech.vel[0], mech.vel[1]) > sound_cfg["step_speed"]:
                kind = "boost" if boosting else "step"
            elif not mech.grounded and boosting:
                kind = "jet"
            if kind:
                self.sounds.append((kind, mech.id, mech.pos[:], self.time))
                mech.last_move_sound = self.time
        aim = mech.desired_aim
        mech.aim[0] = slew_angle(mech.aim[0], float(aim[0]), float(stats["turn_rate"]), self.dt)
        mech.aim[1] += clamp(float(aim[1]) - mech.aim[1],
                             -float(stats["pitch_rate"]) * self.dt, float(stats["pitch_rate"]) * self.dt)
        mech.aim[1] = clamp(mech.aim[1], float(stats["min_pitch"]), float(stats["max_pitch"]))

    def _walkable(self, mech, start, end, ground, stats):
        rise = ground - start[2]
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        if mech.grounded and (rise > stats["step_up"] or
                              (distance > 1e-9 and rise > math.tan(math.radians(stats["max_slope_deg"])) * distance)):
            return False
        return not self._body_blocked(start, end, stats)

    def _body_blocked(self, start, end, stats):
        prisms = getattr(self.world, "prisms", None)
        if prisms is None:
            return self.world.blocked(start, end, stats["radius"])
        radius = stats["radius"]
        min_x, max_x = min(start[0], end[0]) - radius, max(start[0], end[0]) + radius
        min_y, max_y = min(start[1], end[1]) - radius, max(start[1], end[1]) + radius
        feet, head = min(start[2], end[2]), max(start[2], end[2]) + stats["height"]
        for prism, bounds in zip(prisms, self.world.prism_bounds):
            low, high = max(feet, prism["base"]), min(head, prism["top"])
            if (low >= high - 1e-6 or bounds[2] < min_x or bounds[0] > max_x
                    or bounds[3] < min_y or bounds[1] > max_y):
                continue
            z = (low + high) / 2 - 4.5
            if segment_hits_prism((start[0], start[1], z), (end[0], end[1], z), prism, radius):
                return True
        return False

    def _surface(self, x, y, max_height):
        ground = self.world.ground(x, y)
        for prism, bounds in zip(getattr(self.world, "prisms", []), getattr(self.world, "prism_bounds", [])):
            top = float(prism["top"])
            if (ground < top <= max_height and bounds[0] <= x <= bounds[2] and bounds[1] <= y <= bounds[3]
                    and point_in_ring(x, y, prism["ring"])
                    and not any(point_in_ring(x, y, hole) for hole in prism.get("holes", []))):
                ground = top
        return ground

    def _fire(self, mech):
        fire = mech.action.get("fire", [False, False])
        mech.laser_short = max(0.0, mech.laser_short - self.dt)
        for slot, weapon in enumerate(mech.weapons):
            if weapon is None:
                continue
            stats = self.weapon_stats(mech, weapon.kind)
            weapon.cooldown = weapon.cooldown - self.dt if weapon.cooldown > 0 else 0.0
            if weapon.kind == "gun":
                if weapon.reload > 0:
                    weapon.reload = max(0.0, weapon.reload - self.dt)
                    if weapon.reload <= 1e-9:
                        weapon.reload = 0.0
                        weapon.ammo = int(stats["mag"])
                if mech.action.get("reload") and weapon.reload <= 0 and weapon.ammo < int(stats["mag"]):
                    weapon.reload = float(stats["reload"])
                    weapon.reload_mag = int(stats["mag"])
                    self.sounds.append(("reload", mech.id, mech.pos[:], self.time))
                if not fire[slot] or weapon.cooldown > 0 or weapon.reload > 0 or weapon.ammo <= 0:
                    continue
                weapon.ammo -= 1
                if weapon.ammo == 0:
                    weapon.reload = float(stats["reload"])
                    weapon.reload_mag = int(stats["mag"])
                    self.sounds.append(("reload", mech.id, mech.pos[:], self.time))
                self._launch(mech, weapon, stats)
            elif weapon.kind == "laser":
                if weapon.cooldown > 0 or mech.laser_short > 0:
                    continue
                if fire[slot] and not weapon.release_pending:
                    if weapon.charge == 0:
                        self.sounds.append(("laser_charge", mech.id, mech.pos[:], self.time))
                    weapon.charge = min(stats["charge"], weapon.charge + self.dt)
                    if weapon.charge >= stats["charge"]:
                        weapon.hold += self.dt
                elif weapon.charge > 0:
                    weapon.release_pending = True
                    weapon.charge = min(stats["charge"], weapon.charge + self.dt)
                if weapon.charge > 0 and ((weapon.release_pending and weapon.charge >= stats["min_charge"])
                                           or weapon.hold >= stats["hold_s"]):
                    self._laser(mech, weapon, stats)
            elif fire[slot] and weapon.cooldown <= 0:
                if weapon.kind == "mortar":
                    self._launch(mech, weapon, stats)
                elif weapon.kind == "lightning" and mech.energy >= stats["energy"]:
                    self._lightning(mech, weapon, stats)

    def _origin(self, mech, kind, spread=0):
        yaw = mech.aim[0] + self.rng.gauss(0, spread) if spread else mech.aim[0]
        pitch = mech.aim[1] + self.rng.gauss(0, spread) if spread else mech.aim[1]
        direction = [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)]
        stats = self.mech_stats(mech)
        height = stats["height"] + 0.5 if kind == "mortar" else stats["muzzle_height"]
        muzzle = [mech.pos[0], mech.pos[1], mech.pos[2] + height]
        origin = [muzzle[i] + (stats["radius"] + 0.5) * direction[i] for i in range(3)]
        return muzzle, origin, direction

    def _launch(self, mech, weapon, stats):
        spread = math.radians(stats["spread_deg"]) if weapon.kind == "gun" else 0
        muzzle, origin, direction = self._origin(mech, weapon.kind, spread)
        mech.shots += 1
        weapon.cooldown += float(stats["cooldown"])
        self.sounds.append((weapon.kind, mech.id, mech.pos[:], self.time))
        if self.clear(muzzle, origin):
            remaining = stats["range"] if weapon.kind == "gun" else stats["ttl"]
            self.projectiles.append(Projectile(self.next_projectile_id, mech.id, origin,
                                               [d * stats["speed"] for d in direction],
                                               remaining, weapon.kind, stats["damage"], dict(stats)))
            self.next_projectile_id += 1

    def fire_npc(self, npc, origin):
        stats = self.config["npc"]
        spread = math.radians(stats["spread_deg"])
        yaw, pitch = (a + self.rng.gauss(0, spread) for a in npc.aim)
        direction = [math.cos(pitch) * math.cos(yaw), math.cos(pitch) * math.sin(yaw), math.sin(pitch)]
        self.projectiles.append(Projectile(self.next_projectile_id, npc.id, origin[:],
                                           [v * stats["projectile_speed"] for v in direction],
                                           float(stats["deaggro_range"]), "npc", stats["damage"]))
        self.next_projectile_id += 1
        self.sounds.append(("npc_gun", npc.id, npc.pos[:], self.time))

    def _targets(self, shooter, start, end, radius=0.0):
        targets = []
        for victim in self.mechs:
            if victim.id == shooter.id or victim.place is not None or victim.hp <= 0:
                continue
            stats = self.mech_stats(victim)
            feet = [victim.pos[0], victim.pos[1], victim.pos[2] - radius]
            fraction = cylinder_hit(start, end, feet, stats["radius"] + radius, stats["height"] + 2 * radius)
            if fraction is not None:
                targets.append((fraction, victim))
        for npc in self.npcs.actors:
            if npc.id == shooter.id or npc.hp <= 0:
                continue
            cfg = self.config["npc"]
            feet = [npc.pos[0], npc.pos[1], npc.pos[2] - radius]
            fraction = cylinder_hit(start, end, feet, cfg["radius"] + radius, cfg["height"] + 2 * radius)
            if fraction is not None:
                targets.append((fraction, npc))
        stats = self.config["crates"]
        for crate in self.loot.crates:
            feet = [crate["pos"][0], crate["pos"][1], crate["pos"][2] - radius]
            fraction = cylinder_hit(start, end, feet, stats["radius"] + radius, stats["height"] + 2 * radius)
            if fraction is not None:
                targets.append((fraction, crate))
        return sorted(targets, key=lambda x: x[0])

    def _hit(self, shooter, victim, kind, damage, source, effect=None):
        if isinstance(victim, dict):
            if victim not in self.loot.crates:
                return
            victim["hp"] -= damage
            shooter.last_combat = self.time
            shooter.hits += 1
            shooter.damage += damage
            shooter.events.append({"type": "hit", "target": victim["id"], "amount": damage, "weapon": kind})
            if victim["hp"] <= 0:
                self.loot.destroy_crate(victim)
            return
        if isinstance(victim, NPC):
            if victim.hp <= 0:
                return
            victim.hp -= damage
            if isinstance(shooter, Mech):
                victim.target = shooter.id
            shooter.last_combat = self.time
            shooter.hits += 1
            shooter.damage += damage
            shooter.events.append({"type": "hit", "target": victim.id, "amount": damage, "weapon": kind})
            if victim.hp <= 0:
                self.npcs.die(victim)
            return
        if victim.hp <= 0 or victim.place is not None:
            return
        stats = self.mech_stats(victim)
        bearing = math.atan2(source[1] - victim.pos[1], source[0] - victim.pos[0])
        rear = abs(angle_delta(bearing, victim.aim[0])) > math.radians(stats["front_arc_deg"] / 2)
        damage *= (stats["rear_mult"] if rear else 1.0) * max(0.0, 1.0 - stats["armor"])
        if shooter is victim:
            damage *= stats["self_damage_mult"]
        victim.hp -= damage
        victim.last_combat = shooter.last_combat = self.time
        shooter.hits += 1
        shooter.damage += damage
        event = {"type": "damaged", "amount": damage, "weapon": kind, "bearing": bearing, "by": shooter.id, "rear": rear}
        hit = {"type": "hit", "target": victim.id, "amount": damage, "weapon": kind, "rear": rear}
        if effect:
            hit["effect"] = effect
            victim.events.append({"type": "effect", "effect": effect, "by": shooter.id})
        victim.events.append(event)
        shooter.events.append(hit)
        if victim.hp <= 0 and shooter is not victim:
            shooter.kills += 1
            shooter.events.append({"type": "kill", "target": victim.id})

    def _laser(self, mech, weapon, stats):
        muzzle, origin, direction = self._origin(mech, "laser")
        damage = stats["damage"] * (stats["min_frac"] + (1 - stats["min_frac"]) * weapon.charge / stats["charge"])
        end = [origin[i] + direction[i] * stats["range"] for i in range(3)]
        path_end = muzzle[:]
        if self.clear(muzzle, origin):
            wall = self.environment.impact(origin, end)
            path_end = [origin[i] + wall * (end[i] - origin[i]) for i in range(3)]
            hits = 0
            for fraction, target in self._targets(mech, origin, end):
                if fraction > wall:
                    break
                impact = [origin[i] + fraction * (end[i] - origin[i]) for i in range(3)]
                effect = None
                if isinstance(target, Mech) and any(w and w.kind == "mortar" for w in target.weapons):
                    effect = "cookoff"
                    actual_damage = damage * (1 + stats["cookoff"])
                else:
                    actual_damage = damage
                self._hit(mech, target, "laser", actual_damage, origin, effect)
                hits += 1
                if hits > int(stats["pierce"]):
                    path_end = impact
                    break
        mech.shots += 1
        self.sounds.append(("laser", mech.id, mech.pos[:], self.time))
        self.beams.append({"by": mech.id, "kind": "laser", "path": [muzzle, path_end], "time": self.time})
        weapon.charge = weapon.hold = 0.0
        weapon.release_pending = False
        weapon.cooldown = stats["cooldown"]

    def _lightning(self, mech, weapon, stats):
        muzzle, origin, direction = self._origin(mech, "lightning")
        path = [muzzle]
        hit_ids = set()
        point = origin
        blocked_muzzle = not self.clear(muzzle, origin)
        if blocked_muzzle:
            path.append(muzzle[:])
        for jump in range(0 if blocked_muzzle else int(stats["chain"]) + 1):
            options = []
            candidates = ([m for m in self.mechs if m is not mech and m.place is None and m.hp > 0]
                          + [n for n in self.npcs.actors if n.hp > 0] + self.loot.crates)
            for target in candidates:
                tid = target["id"] if isinstance(target, dict) else target.id
                if tid in hit_ids:
                    continue
                pos = target["pos"] if isinstance(target, dict) else target.pos
                height = (self.config["crates"]["height"] if isinstance(target, dict) else
                          self.config["npc"]["height"] if isinstance(target, NPC) else self.mech_stats(target)["height"])
                center = [pos[0], pos[1], pos[2] + height / 2]
                distance = math.dist(point, center)
                if distance > (stats["range"] if jump == 0 else stats["chain_range"]):
                    continue
                if jump == 0 and abs(angle_delta(math.atan2(center[1] - point[1], center[0] - point[0]),
                                                  mech.aim[0])) > math.radians(stats["cone_deg"] / 2):
                    continue
                if self.clear(point, center):
                    options.append((distance, tid, target, center))
            if not options:
                if jump == 0:
                    end = [origin[i] + direction[i] * stats["range"] for i in range(3)]
                    wall = self.environment.impact(origin, end)
                    path.append([origin[i] + wall * (end[i] - origin[i]) for i in range(3)])
                break
            _, tid, target, center = min(options, key=lambda x: (x[0], x[1]))
            effect = None
            if isinstance(target, Mech):
                for other in target.weapons:
                    if other and other.kind == "laser":
                        other.charge = other.hold = 0.0
                        other.release_pending = False
                        target.laser_short = stats["short_s"]
                        effect = "short"
            self._hit(mech, target, "lightning", stats["damage"] * stats["chain_falloff"] ** jump, point, effect)
            path.append(center)
            hit_ids.add(tid)
            point = center
        mech.energy -= stats["energy"]
        mech.last_energy_use = self.time
        mech.shots += 1
        weapon.cooldown += stats["cooldown"]
        self.sounds.append(("lightning", mech.id, mech.pos[:], self.time))
        self.beams.append({"by": mech.id, "kind": "lightning", "path": path, "time": self.time})

    def _explode(self, shot, point, direct=None):
        shooter = self.entity(shot.by)
        stats = shot.stats
        if direct is not None:
            self._hit(shooter, direct, "mortar", shot.damage, shot.pos)
        for target in ([m for m in self.mechs if m.place is None and m.hp > 0]
                       + [n for n in self.npcs.actors if n.hp > 0] + self.loot.crates[:]):
            pos = target["pos"] if isinstance(target, dict) else target.pos
            height = (self.config["crates"]["height"] if isinstance(target, dict) else
                      self.config["npc"]["height"] if isinstance(target, NPC) else self.mech_stats(target)["height"])
            center = [pos[0], pos[1], pos[2] + height / 2]
            distance = math.dist(point, center)
            if distance > stats["splash_radius"] or not self.clear(point, center):
                continue
            falloff = 1 - (1 - stats["splash_edge"]) * distance / stats["splash_radius"]
            self._hit(shooter, target, "mortar", stats["splash"] * falloff, point,
                      "concuss" if isinstance(target, Mech) else None)
            if isinstance(target, Mech):
                dx, dy = target.pos[0] - point[0], target.pos[1] - point[1]
                length = math.hypot(dx, dy)
                if length:
                    target.vel[0] += dx / length * stats["knock"] * falloff
                    target.vel[1] += dy / length * stats["knock"] * falloff
                target.vel[2] += stats["knock_up"] * falloff
                target.grounded = False
                target.energy = max(0.0, target.energy - stats["drain"] * falloff)
        self.sounds.append(("explosion", shooter.id, point[:], self.time))

    def _advance_projectiles(self):
        projectile_radius = float(self.config["projectile"]["gun_radius"])
        remaining = []
        for shot in self.projectiles:
            speed = math.sqrt(sum(v * v for v in shot.vel))
            distance = min(shot.remaining, speed * self.dt) if shot.kind != "mortar" else speed * self.dt
            end = [shot.pos[i] + shot.vel[i] * distance / speed for i in range(3)] if speed else shot.pos[:]
            if shot.kind == "mortar":
                end[2] -= 0.5 * shot.stats["gravity"] * self.dt ** 2
                shot.vel[2] -= shot.stats["gravity"] * self.dt
            radius = float(self.config["projectile"]["npc_radius"]) if shot.kind == "npc" else projectile_radius
            candidates = self._targets(self.entity(shot.by), shot.pos, end, radius)
            wall = self.environment.impact(shot.pos, end)
            if candidates and candidates[0][0] <= wall:
                fraction, victim = candidates[0]
                impact = [shot.pos[i] + (end[i] - shot.pos[i]) * fraction for i in range(3)]
                if shot.kind == "mortar":
                    self._explode(shot, impact, victim)
                else:
                    source = [impact[i] - shot.vel[i] for i in range(3)]
                    self._hit(self.entity(shot.by), victim, shot.kind, shot.damage, source)
                continue
            shot.remaining -= self.dt if shot.kind == "mortar" else distance
            if shot.kind == "mortar" and (wall < 1 or shot.remaining <= 0):
                impact = [shot.pos[i] + wall * (end[i] - shot.pos[i]) for i in range(3)]
                self._explode(shot, impact)
            elif shot.remaining > 1e-6 and wall == 1:
                shot.pos = end
                remaining.append(shot)
        self.projectiles = remaining

    def step(self):
        if sum(m.place is None for m in self.mechs) <= 1:
            return
        self.loot.announce_drops()
        for mech in self.mechs:
            if mech.place is None and mech.offer and self.time >= mech.offer["expires"]:
                self.loot.resolve_offer(mech, 0)
        if self.tick % self.decide_every == 0:
            self._decide()
            self.loot.process_actions([m for m in self.mechs if m.place is None])
        active = [m for m in self.mechs if m.place is None]
        for mech in active:
            self._move(mech)
        self._separate(active)
        for mech in active:
            self._fire(mech)
        self.npcs.step()
        self._advance_projectiles()
        center, radius = zone_state(self.zone(), self.time)
        for mech in active:
            stats = self.mech_stats(mech)
            if mech.hp <= 0:
                continue
            if dist2(mech.pos, center) > radius:
                damage = float(self.zone()["dps"]) * self.dt
                mech.hp -= damage
                mech.last_combat = self.time
                mech.events.append({"type": "damaged", "amount": damage, "weapon": "zone",
                                    "bearing": None, "by": None})
            elif self.time - mech.last_combat >= float(stats["regen_delay"]):
                mech.hp = min(float(stats["hp"]), mech.hp + float(stats["regen_per_s"]) * self.dt)
        for mech in active:
            if mech.hp <= 0:
                self.loot.death(mech)
                self.sounds.append(("death", mech.id, mech.pos[:], self.time))
                mech.place = len(active)
        self.loot.step([m for m in active if m.place is None])
        self.tick += 1
        survivors = [m for m in self.mechs if m.place is None]
        if len(survivors) == 1:
            survivors[0].place = 1

    def _separate(self, active):
        for i, first in enumerate(active):
            for second in active[i + 1:]:
                a, b = self.mech_stats(first), self.mech_stats(second)
                if first.pos[2] >= second.pos[2] + b["height"] or second.pos[2] >= first.pos[2] + a["height"]:
                    continue
                dx, dy = second.pos[0] - first.pos[0], second.pos[1] - first.pos[1]
                distance = math.hypot(dx, dy)
                overlap = a["radius"] + b["radius"] - distance
                if overlap <= 0:
                    continue
                ux, uy = (dx / distance, dy / distance) if distance else (1.0, 0.0)
                for mech, stats, sign in ((first, a, -1), (second, b, 1)):
                    x = clamp(mech.pos[0] + sign * ux * overlap / 2,
                              self.world.min_x + stats["radius"],
                              self.world.min_x + (self.world.nx - 1) * self.world.step - stats["radius"])
                    y = clamp(mech.pos[1] + sign * uy * overlap / 2,
                              self.world.min_y + stats["radius"],
                              self.world.min_y + (self.world.ny - 1) * self.world.step - stats["radius"])
                    end = [x, y, mech.pos[2]]
                    if not self._body_blocked(mech.pos, end, stats):
                        mech.pos = end

    def run(self, max_seconds=None):
        duration = float(self.config["match"]["duration_s"])
        if max_seconds is not None and (not math.isfinite(float(max_seconds)) or float(max_seconds) <= 0):
            raise ValueError("max_seconds must be finite and positive")
        limit = duration if max_seconds is None else min(duration, float(max_seconds))
        while self.time < limit and sum(m.place is None for m in self.mechs) > 1:
            self.step()
        survivors = [m for m in self.mechs if m.place is None]
        if len(survivors) > 1 and self.time >= duration:
            for mech in survivors:
                mech.place = 1 + sum(other.hp > mech.hp for other in survivors)
        completed = all(m.place is not None for m in self.mechs)
        return {"seed": self.seed, "status": "completed" if completed else "truncated",
                "seconds": round(self.time, 3), "results": [
                    {"slot": m.id, "strategy": m.strategy, "place": m.place,
                     "score": len(self.mechs) - m.place if m.place is not None else None,
                     "hp": round(max(0.0, m.hp), 2), "kills": m.kills,
                     "shots": m.shots, "hits": m.hits, "damage": round(m.damage, 2),
                     "cubes": m.cubes, "upgrades": m.upgrades[:]}
                    for m in self.mechs]}
