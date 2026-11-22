"""Loot and upgrade rules for the approximate self-play simulator."""
import math

from geometry import dist2, point_in_ring


class Loot:
    def __init__(self, sim):
        self.sim = sim
        self.crates = []
        self.cubes = []
        self.pickups = []
        self.drops = []
        self.next_crate = 1000
        self.next_cube = 5000
        self.next_pickup = 8000
        self.next_offer = 1
        self.next_drop = 0
        cfg = sim.config
        for i in range(int(cfg["crates"]["count"])):
            roof = i < round(cfg["crates"]["count"] * cfg["crates"]["roof_fraction"])
            self.crates.append({"id": self.next_crate, "pos": self.spawn(roof), "hp": float(cfg["crates"]["hp"])})
            self.next_crate += 1
        for i in range(int(cfg["loot"]["floor_count"])):
            roof = i < round(cfg["loot"]["floor_count"] * cfg["loot"]["floor_roof_fraction"])
            self.add_weapon(self.random_weapon(), self.spawn(roof))

    def spawn(self, roof=False, center=None, radius=None):
        sim, world = self.sim, self.sim.world
        prisms = getattr(world, "prisms", [])
        if roof and prisms:
            for _ in range(500):
                prism = sim.rng.choice(prisms)
                ring = prism.get("ring", [])
                if len(ring) < 6:
                    continue
                x = sim.rng.uniform(min(ring[0::2]), max(ring[0::2]))
                y = sim.rng.uniform(min(ring[1::2]), max(ring[1::2]))
                if point_in_ring(x, y, ring) and not any(point_in_ring(x, y, h) for h in prism.get("holes", [])):
                    z = float(prism["top"])
                    if z >= world.ground(x, y):
                        return [x, y, z]
            raise RuntimeError("Unable to find a roof loot spawn")
        for _ in range(5000):
            if center is None:
                x = sim.rng.uniform(world.min_x + 30, world.min_x + (world.nx - 1) * world.step - 30)
                y = sim.rng.uniform(world.min_y + 30, world.min_y + (world.ny - 1) * world.step - 30)
            else:
                a = sim.rng.uniform(-math.pi, math.pi)
                r = math.sqrt(sim.rng.random()) * radius
                x, y = center[0] + r * math.cos(a), center[1] + r * math.sin(a)
                if not (world.min_x <= x < world.min_x + (world.nx - 1) * world.step
                        and world.min_y <= y < world.min_y + (world.ny - 1) * world.step):
                    continue
            pos = [x, y, world.ground(x, y)]
            if not world.blocked(pos, pos, 1.4):
                return pos
        raise RuntimeError("Unable to find an open loot spawn")

    def random_weapon(self):
        weights = self.sim.config["loot"]["weights"]
        kinds = [kind for kind, weight in weights.items() if weight > 0]
        if not kinds:
            raise ValueError("loot.weights must have a positive weapon weight")
        return self.sim.rng.choices(kinds, weights=[weights[k] for k in kinds])[0]

    def add_cube(self, pos):
        self.cubes.append({"id": self.next_cube, "pos": pos[:]})
        self.next_cube += 1

    def add_weapon(self, kind, pos, weapon=None):
        item = {"id": self.next_pickup, "kind": kind, "pos": pos[:], "weapon": weapon}
        self.next_pickup += 1
        self.pickups.append(item)
        return item

    def destroy_crate(self, crate):
        self.crates.remove(crate)
        cfg = self.sim.config
        for _ in range(int(cfg["crates"]["cubes"])):
            self.add_cube(crate["pos"])
        if self.sim.rng.random() < cfg["loot"]["crate_weapon_chance"]:
            self.add_weapon(self.random_weapon(), crate["pos"])
        self.sim.sounds.append(("crate", None, crate["pos"][:], self.sim.time))

    def award_cubes(self, mech, count, event=True):
        sim = self.sim
        old_hp = sim.mech_stats(mech)["hp"]
        before = mech.cubes
        mech.cubes += count
        mech.hp += sim.mech_stats(mech)["hp"] - old_hp
        mech.pending_offers += sum(before < n <= mech.cubes for n in sim.config["cubes"]["upgrade_thresholds"])
        if event:
            mech.events.append({"type": "pickup", "cubes": mech.cubes})
        self.offer_next(mech)

    def offer_next(self, mech):
        sim = self.sim
        if mech.offer is not None or not mech.pending_offers:
            return
        held = {w.kind for w in mech.weapons if w}
        eligible = [u for u in sim.config["upgrades"]
                    if mech.upgrades.count(u["id"]) < u["max"]
                    and ("requires" not in u or u["requires"] in held)]
        if not eligible:
            mech.pending_offers = 0
            return
        choices = sim.rng.sample(eligible, min(3, len(eligible)))
        mech.offer = {"id": self.next_offer, "expires": sim.time + sim.config["bot"]["offer_timeout_s"],
                      "choices": [{k: u[k] for k in ("id", "name", "desc")} for u in choices]}
        self.next_offer += 1
        mech.pending_offers -= 1

    def resolve_offer(self, mech, pick):
        if (mech.offer is None or not isinstance(pick, (int, float)) or not math.isfinite(pick)
                or pick != int(pick) or not 0 <= pick < len(mech.offer["choices"])):
            return
        pick = int(pick)
        old_hp = self.sim.mech_stats(mech)["hp"]
        old_energy = self.sim.mech_stats(mech)["energy"]
        choice = mech.offer["choices"][pick]["id"]
        mech.upgrades.append(choice)
        new_stats = self.sim.mech_stats(mech)
        mech.hp += new_stats["hp"] - old_hp
        mech.energy += new_stats["energy"] - old_energy
        mech.events.append({"type": "upgrade", "id": choice})
        mech.offer = None
        self.offer_next(mech)

    def process_actions(self, active):
        sim = self.sim
        for mech in active:
            if "pick" in mech.action:
                self.resolve_offer(mech, mech.action["pick"])
        requests = {}
        for mech in active:
            if "take" in mech.action:
                requests.setdefault(mech.action["take"], []).append(mech)
            elif "drop" in mech.action:
                self.drop_weapon(mech, mech.action["drop"])
        for item_id, contenders in requests.items():
            item = next((p for p in self.pickups if p["id"] == item_id), None)
            contenders.sort(key=lambda m: (dist2(m.pos, item["pos"]) if item else math.inf, m.id))
            for mech in contenders:
                self.take_weapon(mech, item if item in self.pickups else None, item_id)

    def drop_weapon(self, mech, slot):
        if not isinstance(slot, (int, float)) or not math.isfinite(slot) or slot not in (0, 1):
            return False
        slot = int(slot)
        weapon = mech.weapons[slot]
        if weapon is None or (weapon.kind == "laser" and weapon.charge > 0):
            return False
        item = self.add_weapon(weapon.kind, mech.pos, weapon)
        mech.weapons[slot] = None
        mech.events.append({"type": "dropped", "id": item["id"], "kind": weapon.kind, "slot": slot})
        return True

    def take_weapon(self, mech, item, item_id):
        sim = self.sim
        reason = None
        if item is None:
            reason = "gone"
        elif dist2(mech.pos, item["pos"]) > sim.config["loot"]["pickup_radius"] or abs(mech.pos[2] - item["pos"][2]) > 4:
            reason = "range"
        elif any(w and w.kind == item["kind"] for w in mech.weapons):
            reason = "have"
        elif None in mech.weapons:
            slot = mech.weapons.index(None)
        elif "drop" not in mech.action or mech.action["drop"] not in (0, 1):
            reason = "full"
        else:
            slot = int(mech.action["drop"])
            if mech.weapons[slot].kind == "laser" and mech.weapons[slot].charge > 0:
                reason = "busy"
            else:
                self.drop_weapon(mech, slot)
        if reason:
            mech.events.append({"type": "take_failed", "id": item_id, "reason": reason})
            return
        weapon = item["weapon"] or sim.new_weapon(item["kind"])
        weapon.cooldown = max(weapon.cooldown, sim.config["loot"]["equip_s"],
                              sim.weapon_stats(mech, weapon.kind)["cooldown"])
        mech.weapons[slot] = weapon
        self.pickups.remove(item)
        mech.events.append({"type": "took", "id": item_id, "kind": weapon.kind, "slot": slot})

    def announce_drops(self):
        sim, cfg = self.sim, self.sim.config
        drop_cfg = cfg["drops"]
        while self.next_drop < len(drop_cfg["times"]) and sim.time >= drop_cfg["times"][self.next_drop]:
            zone = sim.zone()
            pos = self.spawn(center=zone["next_center"], radius=zone["next_radius"] * drop_cfg["spread"])
            self.drops.append({"id": 20000 + self.next_drop, "pos": pos,
                               "land_time": drop_cfg["times"][self.next_drop] + drop_cfg["fall_s"],
                               "holder": None, "progress": 0.0})
            self.next_drop += 1

    def step(self, active):
        sim, cfg = self.sim, self.sim.config
        for item in self.pickups:
            weapon = item["weapon"]
            if weapon:
                weapon.cooldown = max(0.0, weapon.cooldown - sim.dt)
                if weapon.reload > 0:
                    weapon.reload = max(0.0, weapon.reload - sim.dt)
                    if weapon.reload <= 1e-9:
                        weapon.reload = 0.0
                        weapon.ammo = weapon.reload_mag
        for cube in self.cubes[:]:
            candidates = [m for m in active if m.hp > 0
                          and dist2(m.pos, cube["pos"]) <= cfg["cubes"]["pickup_radius"]
                          and abs(m.pos[2] - cube["pos"][2]) <= 4]
            if candidates:
                winner = min(candidates, key=lambda m: (dist2(m.pos, cube["pos"]), m.id))
                self.cubes.remove(cube)
                self.award_cubes(winner, 1)
        drop_cfg = cfg["drops"]
        for drop in self.drops[:]:
            if sim.time < drop["land_time"]:
                continue
            if sim.time - sim.dt < drop["land_time"]:
                sim.sounds.append(("drop", None, drop["pos"][:], sim.time))
            occupants = [m for m in active if m.hp > 0 and dist2(m.pos, drop["pos"]) <= drop_cfg["radius"]
                         and abs(m.pos[2] - drop["pos"][2]) <= 4]
            if not occupants:
                drop["holder"], drop["progress"] = None, 0.0
            elif len(occupants) == 1:
                mech = occupants[0]
                if drop["holder"] != mech.id:
                    drop["holder"], drop["progress"] = mech.id, 0.0
                drop["progress"] += sim.dt
                if drop["progress"] + 1e-9 >= drop_cfg["capture_s"]:
                    self.award_cubes(mech, int(drop_cfg["cubes"]), event=False)
                    stats = sim.mech_stats(mech)
                    mech.hp, mech.energy = stats["hp"], stats["energy"]
                    mech.events.append({"type": "drop", "id": drop["id"], "cubes": mech.cubes})
                    self.drops.remove(drop)

    def death(self, mech):
        for _ in range(mech.cubes + int(self.sim.config["cubes"]["death_drop_extra"])):
            self.add_cube(mech.pos)
        for weapon in mech.weapons:
            if weapon and weapon.kind != "gun":
                weapon.charge = weapon.hold = 0.0
                weapon.release_pending = False
                self.add_weapon(weapon.kind, mech.pos, weapon)

    def observations(self, viewer):
        sim, cfg = self.sim, self.sim.config
        crates = [dict(c) for c in self.crates if sim._visible_actor(viewer, c["pos"], cfg["crates"]["height"])]
        cubes = [dict(c) for c in self.cubes if sim._visible(viewer, c["pos"])]
        pickups = [{k: p[k] for k in ("id", "kind", "pos")} for p in self.pickups if sim._visible(viewer, p["pos"])]
        drops = [{"id": d["id"], "pos": d["pos"][:], "radius": cfg["drops"]["radius"],
                  "lands_in": max(0.0, d["land_time"] - sim.time), "capture_s": cfg["drops"]["capture_s"],
                  "progress": d["progress"], "holder": d["holder"]} for d in self.drops]
        return crates, cubes, pickups, drops
