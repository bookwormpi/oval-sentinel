import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, "bot")
from agent import MechRoyaleBot
from policies import choose_crate, choose_variant_goal, variant_movement, vanguard_movement, vanguard_target
from strategy import selected_strategy


class FlatWorld:
    ready = False

    def __init__(self, map_dir=None):
        pass

    def ground(self, x, y):
        return 0.0

    def route(self, start, goal):
        return goal

    def line_of_sight(self, start, goal):
        return True

    def blocked(self, start, goal):
        return False


class PolicyTests(unittest.TestCase):
    def setUp(self):
        self.world = FlatWorld()
        self.obs = {
            "t": 0.0, "alive": 16,
            "self": {"id": 2, "pos": [0.0, 0.0, 0.0], "in_zone": True,
                     "stats": {"walk_speed": 10.0}, "cubes": 0,
                     "weapons": [{"kind": "gun"}, None]},
            "zone": {"center": [0.0, 0.0], "radius": 200.0,
                     "next_center": [0.0, 0.0], "next_radius": 100.0,
                     "shrink_start": 60.0, "shrink_end": 110.0},
            "mechs": [], "cubes": [], "pickups": [], "drops": [], "crates": [],
        }

    def test_strategy_selection_rejects_unknown_mode(self):
        with patch.dict(os.environ, {"OVAL_STRATEGY": "typo"}):
            with self.assertRaises(ValueError):
                selected_strategy()

    def test_zone_recovery_goes_to_nearest_safe_point(self):
        self.obs["self"]["pos"] = [210.0, 0.0, 0.0]
        self.obs["self"]["in_zone"] = False
        goal, kind = choose_variant_goal(self.obs, self.world, "survivor")
        self.assertEqual(kind, "zone")
        self.assertEqual(goal[:2], (180.0, 0.0))

    def test_next_zone_is_reached_before_shrink_finishes(self):
        self.obs["t"] = 95.5
        self.obs["self"]["pos"] = [116.0, 0.0, 0.0]
        goal, kind = choose_variant_goal(self.obs, self.world, "survivor")
        self.assertEqual(kind, "zone")
        self.assertEqual(goal[:2], (90.0, 0.0))

    def test_scavenger_pursues_cube_but_survivor_holds(self):
        self.obs["cubes"] = [{"id": 5000, "pos": [40.0, 0.0, 0.0]}]
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger")[1], "cube")
        self.assertEqual(choose_variant_goal(self.obs, self.world, "survivor")[1], "hold")

    def test_scavenger_avoids_contested_and_rooftop_loot(self):
        self.obs["cubes"] = [{"id": 5000, "pos": [40.0, 0.0, 0.0]},
                             {"id": 5001, "pos": [10.0, 0.0, 10.0]}]
        self.obs["mechs"] = [{"id": 3, "kind": "mech", "pos": [50.0, 0.0, 0.0]}]
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger")[1], "hold")

    def test_drop_capture_stays_in_ring(self):
        self.obs["drops"] = [{"pos": [2.0, 0.0, 0.0], "radius": 8.0,
                              "lands_in": 0, "holder": 2}]
        goal, kind = choose_variant_goal(self.obs, self.world, "scavenger")
        self.assertEqual(kind, "capture")
        direction, speed = variant_movement("scavenger", self.obs, None, kind,
                                             (0, 0), (0, 0), (0, 0), 0, 0)
        self.assertEqual((direction, speed), ((0.0, 0.0), 0.0))

    def test_scavenger_farms_crates_only_when_safe(self):
        self.obs["crates"] = [{"id": 1000, "pos": [30.0, 0.0, 0.0]}]
        self.assertEqual(choose_crate(self.obs, -100)["id"], 1000)
        self.obs["mechs"] = [{"id": 3, "kind": "mech", "pos": [100.0, 0.0, 0.0]}]
        self.assertIsNone(choose_crate(self.obs, -100))
        self.obs["mechs"] = []
        self.assertIsNone(choose_crate(self.obs, -1))

    def test_vanguard_prioritizes_near_crate_over_distant_drone(self):
        self.obs["crates"] = [{"id": 1000, "pos": [30.0, 0.0, 0.0]}]
        drone = {"id": 100, "kind": "npc", "pos": [0.0, 90.0, 0.0]}
        self.assertEqual(vanguard_target(self.obs, drone, -100)["id"], 1000)
        self.obs["self"].update({"hp": 70, "max_hp": 100})
        self.assertIsNone(vanguard_target(self.obs, drone, -100))

    def test_vanguard_steers_around_blocked_direction(self):
        class WallWorld(FlatWorld):
            def blocked(self, start, goal):
                return goal[0] > start[0] + 1

        enemy = {"id": 3, "kind": "mech", "pos": [-50.0, 0.0, 0.0]}
        direction, speed = vanguard_movement(self.obs, WallWorld(), enemy, "hold",
                                              (0, 0), (0, 0), (0, 0), 0, 0)
        self.assertGreater(speed, 0)
        self.assertLessEqual(direction[0], 0)

    def test_skirmisher_strafes_a_nearby_opponent(self):
        enemy = {"id": 3, "kind": "mech", "pos": [60.0, 0.0, 0.0]}
        direction, speed = variant_movement("skirmisher", self.obs, enemy, "hold",
                                             (0, 0), (0, 0), (0, 0), 0, 0)
        self.assertAlmostEqual(direction[0], 0.0)
        self.assertGreater(abs(direction[1]), 0.9)
        self.assertGreater(speed, 0.0)

    def test_zone_priority_overrides_skirmishing(self):
        self.obs["self"]["in_zone"] = False
        enemy = {"id": 3, "kind": "mech", "pos": [60.0, 0.0, 0.0]}
        direction, speed = variant_movement("skirmisher", self.obs, enemy, "zone",
                                             (1, 0), (0, 1), (0, 0), 1, 0)
        self.assertEqual(direction, (1.0, 0.0))
        self.assertEqual(speed, 1.0)

    def test_scavenger_fires_on_safe_crate_through_agent(self):
        self.obs.update({"tick": 2, "sounds": [], "events": [], "projectiles": []})
        self.obs["self"].update({
            "vel": [0.0, 0.0, 0.0], "aim": [0.0, 0.0],
            "hp": 1400, "max_hp": 1400, "energy": 100, "grounded": True,
            "weapons": [{"kind": "gun", "ammo": 12, "cooldown": 0, "reload": 0,
                         "stats": {"speed": 252, "range": 230}}, None],
        })
        self.obs["crates"] = [{"id": 1000, "pos": [30.0, 0.0, 0.0]}]
        hello = {"config": {"mech": {"min_pitch": -0.7, "max_pitch": 1.3}}}
        with patch.dict(os.environ, {"OVAL_STRATEGY": "scavenger"}), patch("agent.World", FlatWorld):
            action = MechRoyaleBot(hello).act(self.obs)
        self.assertTrue(action["fire"][0])

    def test_vanguard_fires_on_aligned_safe_crate_through_agent(self):
        self.obs.update({"tick": 2, "sounds": [], "events": [], "projectiles": []})
        self.obs["self"].update({
            "vel": [0.0, 0.0, 0.0], "aim": [0.0, -0.173245],
            "hp": 1400, "max_hp": 1400, "energy": 100, "grounded": True,
            "weapons": [{"kind": "gun", "ammo": 12, "cooldown": 0, "reload": 0,
                         "stats": {"speed": 252, "range": 230}}, None],
        })
        self.obs["crates"] = [{"id": 1000, "pos": [30.0, 0.0, 0.0]}]
        hello = {"config": {"mech": {"min_pitch": -0.7, "max_pitch": 1.3}}}
        with patch.dict(os.environ, {"OVAL_STRATEGY": "vanguard"}), patch("agent.World", FlatWorld):
            action = MechRoyaleBot(hello).act(self.obs)
        self.assertTrue(action["fire"][0])


if __name__ == "__main__":
    unittest.main()
