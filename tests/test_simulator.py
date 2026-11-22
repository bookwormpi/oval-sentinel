import copy
import math
from pathlib import Path
import tempfile
import unittest

from check import Schema
from simulator.engine import ROOT, Projectile, Simulator, cylinder_hit, load_config, load_world
from agent import MechRoyaleBot
from combat import predict_target
from policies import adaptive_zone_guard


class FlatWorld:
    min_x = min_y = -500.0
    nx = ny = 1001
    step = 1.0

    def ground(self, x, y):
        return 0.0

    def blocked(self, start, end, radius=2.5):
        return False

    def line_of_sight(self, start, end):
        return True


class IdleBot:
    def act(self, obs):
        return {"tick": obs["tick"], "move": [0.0, 0.0], "aim": [0.0, 0.0],
                "fire": [False, False]}


def factory(hello, strategy, world):
    return IdleBot()


class SimulatorTests(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(load_config())
        self.config["weapons"]["gun"]["spread_deg"] = 0
        self.config["crates"]["count"] = 0
        self.config["loot"]["floor_count"] = 0
        self.config["npc"]["count"] = 0
        self.sim = Simulator(["baseline", "vanguard"], 42, self.config, FlatWorld(), factory)

    def test_segment_cylinder_collision_checks_height_and_tunneling(self):
        self.assertAlmostEqual(cylinder_hit([-10, 0, 4], [10, 0, 4], [0, 0, 0], 2, 9), 0.4)
        self.assertIsNone(cylinder_hit([-10, 0, 12], [10, 0, 12], [0, 0, 0], 2, 9))
        self.assertIsNone(cylinder_hit([-10, 3, 4], [10, 3, 4], [0, 0, 0], 2, 9))

    def test_zone_guard_reduces_outward_drift_under_simulated_air_acceleration(self):
        zone = {"center": [0, 0], "radius": 10, "next_center": [0, 0],
                "next_radius": 10, "shrink_start": 100, "shrink_end": 200}

        def travel(use_guard):
            sim = Simulator(["baseline", "vanguard"], 42, self.config, FlatWorld(), factory)
            mech = sim.mechs[0]
            mech.pos, mech.vel, mech.grounded = [9.0, 0.0, 10.0], [10.0, 0.0, 0.0], False
            obs = {"t": 0.0, "zone": zone, "self": {
                "pos": mech.pos, "vel": mech.vel, "grounded": False, "in_zone": True,
                "stats": sim.mech_stats(mech), "weapons": [],
            }}
            direction, fraction = ((0.0, 1.0), 0.9)
            if use_guard:
                direction, fraction = adaptive_zone_guard(obs, sim.world, direction, fraction, False)
            mech.action = {"move": [fraction * direction[0], fraction * direction[1]]}
            for _ in range(12):
                sim._move(mech)
            return mech.pos[0]

        self.assertLess(travel(True), travel(False))

    def test_observation_conforms_to_schema(self):
        obs = self.sim.observation(self.sim.mechs[0])
        self.assertEqual(Schema(ROOT / "mechfight.schema.json").errors(obs, "Observation"), [])

    def test_idle_scan_discovers_enemy_behind_torso(self):
        scanner, enemy = self.sim.mechs
        scanner.pos, enemy.pos = [0.0, 0.0, 0.0], [-100.0, 0.0, 0.0]
        scanner.aim = [0.0, 0.0]
        scanner.bot = MechRoyaleBot({"config": self.config}, strategy="vanguard", world=self.sim.world)
        self.assertEqual(self.sim.observation(scanner)["mechs"], [])
        found = False
        for _ in range(60):
            self.sim.step()
            if self.sim.observation(scanner)["mechs"]:
                found = True
                break
        self.assertTrue(found)

    def test_gun_projectile_damages_target_and_emits_events(self):
        shooter, victim = self.sim.mechs
        shooter.pos = [0.0, 0.0, 0.0]
        victim.pos = [40.0, 0.0, 0.0]
        shooter.aim = [0.0, math.atan2(-2.5, 40)]
        victim.aim = [math.pi, 0.0]
        shooter.action = {"fire": [True, False]}
        self.sim._fire(shooter)
        for _ in range(4):
            self.sim._advance_projectiles()
        self.assertEqual(shooter.shots, 1)
        self.assertEqual(shooter.hits, 1)
        self.assertAlmostEqual(victim.hp, 1400 - 45)
        self.assertEqual(victim.events[0]["type"], "damaged")
        self.assertEqual(shooter.events[0]["type"], "hit")

    def test_building_blocks_projectile(self):
        class WallWorld(FlatWorld):
            def line_of_sight(self, start, end):
                return not (start[0] < 15 <= end[0])

        self.sim.world = WallWorld()
        shooter, victim = self.sim.mechs
        shooter.pos, victim.pos = [0.0, 0.0, 0.0], [40.0, 0.0, 0.0]
        shooter.aim = [0.0, math.atan2(-2.5, 40)]
        shooter.action = {"fire": [True, False]}
        self.sim._fire(shooter)
        for _ in range(4):
            self.sim._advance_projectiles()
        self.assertEqual(victim.hp, 1400)
        self.assertEqual(shooter.hits, 0)

    def test_corrected_mortar_aim_hits_stationary_target(self):
        shooter, victim = self.sim.mechs
        shooter.pos, victim.pos = [0.0, 0.0, 0.0], [100.0, 0.0, 0.0]
        shooter.weapons[1] = self.sim.new_weapon("mortar")
        weapon = {"kind": "mortar", "stats": self.sim.weapon_stats(shooter, "mortar")}
        me = {"pos": shooter.pos, "vel": [0, 0, 0], "stats": self.sim.mech_stats(shooter)}
        target = {"kind": "mech", "pos": victim.pos, "vel": [0, 0, 0], "cubes": 0}
        yaw, pitch, _, _ = predict_target(me, target, weapon, self.config, correct_mortar=True)
        shooter.aim = [yaw, pitch]
        shooter.action = {"fire": [False, True]}
        self.sim._fire(shooter)
        for _ in range(50):
            self.sim._advance_projectiles()
        self.assertGreater(shooter.hits, 0)
        self.assertLess(victim.hp, self.config["mech"]["hp"])

    def test_gun_cadence_and_automatic_reload(self):
        shooter = self.sim.mechs[0]
        shooter.action = {"fire": [True, False]}
        for _ in range(50):
            self.sim._fire(shooter)
        self.assertEqual(shooter.shots, 11)
        for _ in range(10):
            self.sim._fire(shooter)
        self.assertEqual(shooter.ammo, 0)
        shooter.action = {"fire": [False, False]}
        for _ in range(50):
            self.sim._fire(shooter)
        self.assertEqual(shooter.ammo, self.config["weapons"]["gun"]["mag"])
        self.assertEqual(shooter.reload, 0)

    def test_omitted_aim_keeps_previous_command(self):
        class AimBot:
            def act(self, obs):
                return {"tick": obs["tick"], **({"aim": [1.5, 0]} if obs["tick"] == 0 else {})}

        mech = self.sim.mechs[0]
        mech.bot = AimBot()
        mech.aim = [0.0, 0.0]
        for _ in range(3):
            self.sim.step()
        self.assertAlmostEqual(mech.aim[0], 0.6)

    def test_invalid_action_is_rejected(self):
        class InvalidBot:
            def act(self, obs):
                return {"tick": obs["tick"], "move": [float("nan"), 0]}

        self.sim.mechs[0].bot = InvalidBot()
        with self.assertRaisesRegex(ValueError, "invalid action"):
            self.sim.step()

    def test_invalid_custom_map_does_not_fall_back_to_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "v2"
            root.mkdir()
            (root / "world.json").write_text("not json", encoding="utf-8")
            (root / "terrain.i16").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "Unable to load arena"):
                load_world(directory)

    def test_same_tick_deaths_share_place(self):
        self.sim.zone_phases = [{"center": [0, 0], "radius": 5, "next_center": [0, 0],
                                 "next_radius": 5, "shrink_start": 100, "shrink_end": 200, "dps": 20}]
        for i, mech in enumerate(self.sim.mechs):
            mech.pos = [10.0 if i == 0 else -10.0, 0.0, 0.0]
            mech.hp = 0.1
        self.sim.step()
        result = self.sim.run()
        self.assertEqual(result["status"], "completed")
        self.assertEqual([m["place"] for m in result["results"]], [2, 2])
        self.assertEqual([m["score"] for m in result["results"]], [0, 0])
        self.assertEqual([m["survival_seconds"] for m in result["results"]], [self.sim.dt] * 2)
        self.assertFalse(any(m["survival_censored"] for m in result["results"]))

    def test_truncation_does_not_assign_fake_placements(self):
        result = self.sim.run(0.5)
        self.assertEqual(result["status"], "truncated")
        self.assertTrue(all(m["place"] is None and m["score"] is None for m in result["results"]))
        self.assertTrue(all(m["survival_seconds"] is None and m["survival_censored"]
                            and m["observed_survival_seconds"] == result["seconds"] for m in result["results"]))

    def test_survival_records_death_winner_and_timeout(self):
        self.sim.mechs[0].hp = 0
        self.sim.step()
        result = self.sim.run()
        self.assertEqual([m["survival_seconds"] for m in result["results"]], [self.sim.dt] * 2)
        self.assertEqual([m["place"] for m in result["results"]], [2, 1])
        config = copy.deepcopy(self.config)
        config["match"]["duration_s"] = 0.1
        sim = Simulator(["baseline", "vanguard"], 42, config, FlatWorld(), factory)
        result = sim.run()
        self.assertEqual([m["survival_seconds"] for m in result["results"]], [0.1, 0.1])
        self.assertFalse(any(m["survival_censored"] for m in result["results"]))

    def test_truncation_retains_completed_survival_for_eliminated_bots(self):
        sim = Simulator(["baseline", "vanguard", "baseline"], 42, self.config, FlatWorld(), factory)
        sim.mechs[0].hp = 0
        sim.step()
        result = sim.run(0.2)
        self.assertEqual(result["status"], "truncated")
        self.assertEqual(result["results"][0]["survival_seconds"], sim.dt)
        self.assertFalse(result["results"][0]["survival_censored"])
        self.assertTrue(all(m["survival_seconds"] is None and m["survival_censored"]
                            for m in result["results"][1:]))

    def test_seeded_match_is_reproducible(self):
        config = copy.deepcopy(self.config)
        config["npc"]["count"] = 2
        config["crates"]["count"] = 3
        config["loot"]["floor_count"] = 4
        first = Simulator(["baseline", "vanguard"], 42, config, FlatWorld(), factory)
        other = Simulator(["baseline", "vanguard"], 42, config, FlatWorld(), factory)
        self.assertEqual(first.run(5), other.run(5))
        self.assertEqual([(n.pos, n.hp) for n in first.npcs.actors],
                         [(n.pos, n.hp) for n in other.npcs.actors])
        self.assertEqual(first.loot.crates, other.loot.crates)
        self.assertEqual(first.loot.pickups, other.loot.pickups)

    def test_loot_spawns_are_seeded_and_visible(self):
        config = copy.deepcopy(self.config)
        config["crates"]["count"], config["loot"]["floor_count"] = 3, 4
        a = Simulator(["baseline", "vanguard"], 8, config, FlatWorld(), factory)
        b = Simulator(["baseline", "vanguard"], 8, config, FlatWorld(), factory)
        self.assertEqual(a.loot.crates, b.loot.crates)
        self.assertEqual(a.loot.pickups, b.loot.pickups)
        self.assertEqual(len(a.loot.crates), 3)
        self.assertEqual(len(a.loot.pickups), 4)
        a.mechs[0].pos = a.loot.crates[0]["pos"][:]
        obs = a.observation(a.mechs[0])
        self.assertIn(a.loot.crates[0]["id"], [c["id"] for c in obs["crates"]])
        self.assertEqual(Schema(ROOT / "mechfight.schema.json").errors(obs, "Observation"), [])

    def test_shooting_crate_drops_cube_and_weapon(self):
        shooter, victim = self.sim.mechs
        shooter.pos, victim.pos = [0.0, 0.0, 0.0], [300.0, 300.0, 0.0]
        shooter.aim = [0.0, math.atan2(-5.8, 40)]
        shooter.action = {"fire": [True, False]}
        crate = {"id": 1000, "pos": [40.0, 0.0, 0.0], "hp": 45.0}
        self.sim.loot.crates.append(crate)
        self.config["loot"]["crate_weapon_chance"] = 1
        self.sim._fire(shooter)
        for _ in range(4):
            self.sim._advance_projectiles()
        self.assertFalse(self.sim.loot.crates)
        self.assertEqual(len(self.sim.loot.cubes), 1)
        self.assertEqual(len(self.sim.loot.pickups), 1)
        self.assertEqual(shooter.events[0]["target"], 1000)

    def test_cube_pickup_ties_scaling_and_queued_offers(self):
        a, b = self.sim.mechs
        a.pos, b.pos = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        self.sim.loot.add_cube([0, 0, 0])
        self.sim.loot.step([a, b])
        self.assertEqual((a.cubes, b.cubes), (1, 0))
        self.assertAlmostEqual(a.hp, 1456)
        self.assertAlmostEqual(self.sim.mech_stats(a)["radius"], 2.625)
        self.assertAlmostEqual(self.sim.weapon_stats(a, "gun")["damage"], 46.8)
        self.sim.loot.award_cubes(a, 5)
        self.assertEqual(a.pending_offers, 2)
        first = a.offer["choices"][0]["id"]
        self.sim.loot.resolve_offer(a, 0)
        self.assertEqual(a.upgrades, [first])
        self.assertIsNotNone(a.offer)
        self.assertEqual(a.pending_offers, 1)
        self.assertEqual(Schema(ROOT / "mechfight.schema.json").errors(self.sim.observation(a), "Observation"), [])

    def test_cube_stat_cap_does_not_cap_size(self):
        mech = self.sim.mechs[0]
        self.sim.loot.award_cubes(mech, 15)
        self.assertAlmostEqual(self.sim.mech_stats(mech)["hp"], 1400 * 1.4)
        self.assertAlmostEqual(self.sim.weapon_stats(mech, "gun")["damage"], 45 * 1.4)
        self.assertAlmostEqual(self.sim.mech_stats(mech)["height"], 9 * 1.75)

    def test_upgrade_expiry_and_stats(self):
        mech = self.sim.mechs[0]
        mech.offer = {"id": 1, "expires": 0, "choices": [{"id": "armor", "name": "x", "desc": "x"}]}
        self.sim.step()
        self.assertEqual(mech.upgrades, ["armor"])
        self.assertAlmostEqual(self.sim.mech_stats(mech)["armor"], 0.08)

    def test_pickups_resolve_by_distance_and_preserve_gun(self):
        a, b = self.sim.mechs
        a.pos, b.pos = [2.0, 0, 0], [1.0, 0, 0]
        item = self.sim.loot.add_weapon("laser", [0, 0, 0])
        a.action = b.action = {"take": item["id"]}
        self.sim.loot.process_actions([a, b])
        self.assertIsNone(a.weapons[1])
        self.assertEqual(a.events[-1]["reason"], "gone")
        self.assertEqual(b.weapons[1].kind, "laser")
        self.assertGreaterEqual(b.weapons[1].cooldown, self.config["weapons"]["laser"]["cooldown"])
        b.weapons[0].ammo = 4
        other = self.sim.loot.add_weapon("mortar", b.pos)
        b.action = {"take": other["id"], "drop": 0}
        self.sim.loot.process_actions([b])
        dropped = next(p for p in self.sim.loot.pickups if p["kind"] == "gun")
        self.assertEqual(dropped["weapon"].ammo, 4)
        self.assertEqual(b.weapons[0].kind, "mortar")
        self.assertEqual(Schema(ROOT / "mechfight.schema.json").errors(self.sim.observation(b), "Observation"), [])
        b.weapons[1].charge = 0.2
        self.assertFalse(self.sim.loot.drop_weapon(b, 1))

    def test_supply_drop_contest_freezes_and_capture_repairs(self):
        a, b = self.sim.mechs
        a.pos, b.pos = [0, 0, 0], [100, 0, 0]
        a.hp, a.energy = 100, 1
        self.config["drops"]["capture_s"] = 0.1
        self.sim.loot.drops.append({"id": 20000, "pos": [0, 0, 0], "land_time": 0,
                                    "holder": None, "progress": 0})
        self.sim.loot.step([a, b])
        b.pos = [0, 0, 0]
        self.sim.loot.step([a, b])
        self.assertAlmostEqual(self.sim.loot.drops[0]["progress"], 0.05)
        b.pos = [100, 0, 0]
        self.sim.loot.step([a, b])
        self.assertFalse(self.sim.loot.drops)
        self.assertEqual(a.cubes, 3)
        self.assertEqual(a.hp, self.sim.mech_stats(a)["hp"])
        self.assertEqual(a.energy, self.sim.mech_stats(a)["energy"])
        self.assertEqual(a.events[-1]["type"], "drop")

    def test_supply_drop_announced_before_decision(self):
        class CaptureObservation:
            def __init__(self):
                self.drops = None

            def act(self, obs):
                self.drops = obs["drops"]
                return {"tick": obs["tick"]}

        self.config["drops"]["times"] = [0]
        bot = CaptureObservation()
        self.sim.mechs[0].bot = bot
        self.sim.step()
        self.assertEqual(len(bot.drops), 1)
        self.assertEqual(bot.drops[0]["lands_in"], self.config["drops"]["fall_s"])

    def test_death_drops_cubes_and_secondary_weapon(self):
        mech = self.sim.mechs[0]
        mech.cubes = 3
        mech.weapons[1] = self.sim.new_weapon("laser")
        self.sim.loot.death(mech)
        self.assertEqual(len(self.sim.loot.cubes), 5)
        self.assertEqual([p["kind"] for p in self.sim.loot.pickups], ["laser"])

    def test_laser_and_lightning_signature_effects(self):
        shooter, victim = self.sim.mechs
        shooter.pos, victim.pos = [0, 0, 0], [30, 0, 0]
        shooter.aim = [0, math.atan2(-2.5, 30)]
        victim.aim = [math.pi, 0]
        shooter.weapons[1] = self.sim.new_weapon("laser")
        victim.weapons[1] = self.sim.new_weapon("mortar")
        shooter.action = {"fire": [False, True]}
        for _ in range(16):
            self.sim._fire(shooter)
        shooter.action = {"fire": [False, False]}
        self.sim._fire(shooter)
        self.assertAlmostEqual(victim.hp, 1400 - 315 * 1.6)
        self.assertTrue(any(e.get("effect") == "cookoff" for e in victim.events))
        shooter.weapons[1] = self.sim.new_weapon("lightning")
        victim.weapons[1] = self.sim.new_weapon("laser")
        victim.weapons[1].charge = 0.4
        shooter.action = {"fire": [False, True]}
        self.sim._fire(shooter)
        self.assertEqual(victim.weapons[1].charge, 0)
        self.assertEqual(victim.laser_short, self.config["weapons"]["lightning"]["short_s"])
        self.assertEqual(shooter.energy, 97)

    def test_mortar_impact_concusses_and_drains(self):
        shooter, victim = self.sim.mechs
        shooter.pos, victim.pos = [0, 0, 0], [30, 0, 0]
        victim.aim = [math.pi, 0]
        stats = self.sim.weapon_stats(shooter, "mortar")
        shot = Projectile(1, shooter.id, [29, 0, 4.5], [1, 0, 0], 1, "mortar", stats["damage"], stats)
        self.sim._explode(shot, [29.9, 0, 4.5], victim)
        falloff = 1 - (1 - stats["splash_edge"]) * 0.1 / stats["splash_radius"]
        self.assertAlmostEqual(victim.hp, 1400 - 200 - 160 * falloff)
        self.assertAlmostEqual(victim.energy, 100 - 45 * falloff)
        self.assertGreater(victim.vel[2], 0)
        self.assertTrue(any(e.get("effect") == "concuss" for e in victim.events))

    def test_npc_acquires_attacks_and_drops_cubes(self):
        config = copy.deepcopy(self.config)
        config["npc"]["count"] = 1
        config["npc"]["spread_deg"] = 0
        sim = Simulator(["baseline", "vanguard"], 42, config, FlatWorld(), factory)
        npc = sim.npcs.actors[0]
        npc.pos, npc.home = [0, 0, 0], [0, 0, 0]
        mech = sim.mechs[0]
        mech.pos, mech.aim = [20, 0, 0], [math.pi, 0]
        sim.mechs[1].pos = [300, 300, 0]
        sim.npcs.step()
        self.assertEqual(npc.target, mech.id)
        for _ in range(10):
            sim._advance_projectiles()
        self.assertLess(mech.hp, 1400)
        self.assertTrue(any(e["weapon"] == "npc" for e in mech.events if e["type"] == "damaged"))
        self.assertEqual(Schema(ROOT / "mechfight.schema.json").errors(sim.observation(mech), "Observation"), [])
        sim._hit(mech, npc, "gun", 1000, mech.pos)
        self.assertEqual(len(sim.loot.cubes), 2)
        self.assertFalse(any(m["id"] == npc.id for m in sim.observation(mech)["mechs"]))

    def test_trees_block_sight_but_not_projectiles(self):
        tree = {"x": 5, "y": 0, "z": 0, "height": 10, "crownR": 2,
                "crownBase": 4, "trunkR": 0.5, "crownX": 5, "crownY": 0}
        self.sim.environment.add_tree(tree)
        start, end = [0, 0, 2], [10, 0, 2]
        self.assertFalse(self.sim.clear(start, end, sight=True))
        self.assertTrue(self.sim.clear(start, end))
        self.sim.environment.density = 0.5
        self.assertFalse(self.sim.clear([0, 0.75, 7], [10, 0.75, 7], sight=True))
        self.assertTrue(self.sim.clear([0, 0.75, 7], [10, 0.75, 7], sight=True, canopy=False))

    def test_terrain_blocks_projectiles(self):
        class Ridge(FlatWorld):
            def ground(self, x, y):
                return 8 if 4 <= x <= 6 else 0

        self.sim.world = Ridge()
        self.assertFalse(self.sim.clear([0, 0, 3], [10, 0, 3]))
        self.assertTrue(self.sim.clear([0, 0, 10], [10, 0, 10]))

    def test_bot_line_of_sight_accounts_for_terrain_ridges(self):
        from world import World
        world = World.__new__(World)
        world.ready, world.terrain, world.prisms, world.los_cache = True, [0], [], {}
        world.prism_bounds, world.prism_zbounds = [], []
        world.ground = lambda x, y: 8 if 4 <= x <= 6 else 0
        self.assertFalse(world.line_of_sight([0, 0, 3], [10, 0, 3]))
        self.assertTrue(world.line_of_sight([0, 0, 10], [10, 0, 10]))

    def test_roof_is_a_landing_surface(self):
        class Roof(FlatWorld):
            prisms = [{"ring": [-5, -5, 5, -5, 5, 5, -5, 5], "holes": [], "top": 10, "base": 0}]
            prism_bounds = [(-5, -5, 5, 5)]

        self.sim.world = Roof()
        mech = self.sim.mechs[0]
        mech.pos, mech.vel, mech.grounded = [0, 0, 10.1], [0, 0, -4], False
        mech.action = {"move": [0, 0]}
        self.sim._move(mech)
        self.assertEqual(mech.pos[2], 10)
        self.assertTrue(mech.grounded)

    def test_vanguard_flies_to_collect_otherwise_unreachable_elevated_loot(self):
        from agent import MechRoyaleBot

        class Roof(FlatWorld):
            prisms = [{"ring": [40, -5, 50, -5, 50, 5, 40, 5], "holes": [], "top": 10, "base": 0}]
            prism_bounds = [(40, -5, 50, 5)]
            ready = False

            def route(self, start, goal):
                return goal

        def vanguard_factory(hello, strategy, world):
            return MechRoyaleBot(hello, strategy="vanguard", world=world)

        config = copy.deepcopy(self.config)
        sim = Simulator(["vanguard", "vanguard"], 7, config, Roof(), vanguard_factory)
        mech, other = sim.mechs
        mech.pos = [0.0, 0.0, 0.0]
        other.pos = [-400.0, -400.0, 0.0]  # far away: not a threat, no interference
        sim.loot.add_cube([45.0, 0.0, 10.0])  # on the 10 m rooftop
        for _ in range(400):  # 20 s of match time at 20 Hz
            sim.step()
            if mech.cubes > 0:
                break
        self.assertGreater(mech.cubes, 0, "never climbed onto the roof and collected the cube")
        self.assertGreaterEqual(mech.pos[2], 9.0)

    def test_overlapping_mechs_are_separated(self):
        a, b = self.sim.mechs
        a.pos, b.pos = [0, 0, 0], [0, 0, 0]
        self.sim._separate([a, b])
        self.assertAlmostEqual(math.dist(a.pos, b.pos), 5)

    def test_slides_along_wall(self):
        class Wall(FlatWorld):
            def blocked(self, start, end, radius=2.5):
                return end[0] > 0

        self.sim.world = Wall()
        mech = self.sim.mechs[0]
        mech.pos = [-0.05, 0, 0]
        mech.action = {"move": [1, 1]}
        self.sim._move(mech)
        self.assertAlmostEqual(mech.pos[0], -0.05)
        self.assertGreater(mech.pos[1], 0)

    def test_overhang_stops_upward_motion(self):
        class Ceiling(FlatWorld):
            prisms = [{"ring": [-10, -10, 10, -10, 10, 10, -10, 10],
                       "holes": [], "top": 11, "base": 10}]
            prism_bounds = [(-10, -10, 10, 10)]

        self.sim.world = Ceiling()
        mech = self.sim.mechs[0]
        mech.pos, mech.vel, mech.grounded = [0, 0, 0], [0, 0, 20], False
        mech.action = {"move": [0, 0]}
        self.sim._move(mech)
        self.sim._move(mech)
        self.assertLessEqual(mech.pos[2], 1)
        self.assertEqual(mech.vel[2], 0)


if __name__ == "__main__":
    unittest.main()
