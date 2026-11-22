import os
import math
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, "bot")
from agent import MechRoyaleBot
from policies import (choose_crate, choose_variant_goal, variant_movement, vanguard_movement,
                      vanguard_target, tactician_goal, tactician_movement, adaptive_zone_guard)
from strategy import selected_strategy
from geometry import angle_delta, slew_angle


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

    def test_zone_departure_accounts_for_slow_charging_laser(self):
        self.obs["t"] = 92.0
        self.obs["self"]["pos"] = [120.0, 0.0, 0.0]
        self.assertEqual(choose_variant_goal(self.obs, self.world, "survivor")[1], "hold")
        self.obs["self"]["weapons"][1] = {"kind": "laser", "charge": 0.5}
        self.assertEqual(choose_variant_goal(self.obs, self.world, "survivor")[1], "zone")

    def test_tactician_repositions_early_only_when_no_loot_objective(self):
        self.obs["t"] = 78
        self.obs["self"]["pos"] = [120.0, 0.0, 0.0]
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger")[1], "hold")
        self.assertEqual(tactician_goal(self.obs, self.world, -100)[1], "zone")
        self.obs["cubes"] = [{"id": 5000, "pos": [110.0, 0.0, 0.0]}]
        self.assertEqual(tactician_goal(self.obs, self.world, -100)[1], "cube")

    def test_tactician_interrupts_loot_detour_to_recover_after_damage(self):
        self.obs["self"].update({"hp": 600, "max_hp": 1400})
        self.obs["cubes"] = [{"id": 5000, "pos": [30.0, 0.0, 0.0]}]
        self.assertEqual(tactician_goal(self.obs, self.world, -1)[1], "hold")
        self.assertEqual(tactician_goal(self.obs, self.world, -9)[1], "cube")

    def test_outside_zone_never_targets_current_position_when_center_is_known(self):
        self.obs["self"]["pos"] = [20.0, 0.0, 0.0]
        self.obs["self"]["in_zone"] = False
        goal, kind = choose_variant_goal(self.obs, self.world, "survivor")
        self.assertEqual(kind, "zone")
        self.assertEqual(goal[:2], (0.0, 0.0))

    def test_loot_detour_reserves_time_to_reach_next_zone(self):
        self.obs["self"]["pos"] = [60.0, 0.0, 0.0]
        self.obs["cubes"] = [{"id": 5000, "pos": [130.0, 0.0, 0.0]}]
        self.obs["t"] = 80.0
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger")[1], "cube")
        self.obs["t"] = 90.0
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger")[1], "hold")
        self.obs["cubes"][0]["pos"] = [70.0, 0.0, 0.0]
        self.obs["t"] = 109.0
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger")[1], "cube")

    def test_laser_charge_reduces_time_available_for_a_detour(self):
        self.obs["t"] = 80.0
        self.obs["self"]["pos"] = [60.0, 0.0, 0.0]
        self.obs["self"]["weapons"][1] = {"kind": "laser", "charge": 0.5}
        self.obs["cubes"] = [{"id": 5000, "pos": [130.0, 0.0, 0.0]}]
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger")[1], "hold")

    def test_weapon_and_flight_detours_respect_zone_deadline(self):
        self.obs["self"]["pos"] = [60.0, 0.0, 0.0]
        self.obs["pickups"] = [{"id": 8000, "kind": "laser", "pos": [130.0, 0.0, 0.0]}]
        self.obs["t"] = 80.0
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger")[1], "weapon")
        self.obs["t"] = 90.0
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger")[1], "hold")
        self.obs["pickups"] = []
        self.obs["self"].update({"hp": 1400, "max_hp": 1400, "energy": 100})
        self.obs["cubes"] = [{"id": 5000, "pos": [130.0, 0.0, 20.0]}]
        self.obs["t"] = 80.0
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger")[1], "cube")
        self.obs["t"] = 86.0
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger")[1], "hold")

    def test_drop_detour_accounts_for_remaining_capture_time(self):
        self.obs["t"] = 94.4
        self.obs["self"]["pos"] = [110.0, 0.0, 0.0]
        self.obs["drops"] = [{"pos": [110.0, 0.0, 0.0], "radius": 8.0,
                              "lands_in": 0, "holder": 2, "capture_s": 4.0, "progress": 0.0}]
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger")[1], "hold")
        self.obs["drops"][0]["progress"] = 3.0
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger")[1], "capture")

    def test_scavenger_pursues_cube_but_survivor_holds(self):
        self.obs["cubes"] = [{"id": 5000, "pos": [40.0, 0.0, 0.0]}]
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger")[1], "cube")
        self.assertEqual(choose_variant_goal(self.obs, self.world, "survivor")[1], "hold")

    def test_scavenger_does_not_chase_weapon_already_held_in_other_slot(self):
        self.obs["self"]["weapons"] = [{"kind": "laser"}, None]
        self.obs["pickups"] = [{"id": 8000, "kind": "laser", "pos": [20.0, 0.0, 0.0]}]
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger")[1], "hold")

    def test_scavenger_avoids_contested_and_rooftop_loot(self):
        self.obs["cubes"] = [{"id": 5000, "pos": [40.0, 0.0, 0.0]},
                             {"id": 5001, "pos": [10.0, 0.0, 10.0]}]
        self.obs["mechs"] = [{"id": 3, "kind": "mech", "pos": [50.0, 0.0, 0.0]}]
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger")[1], "hold")

    def test_risk_aware_loot_rejects_a_route_past_an_enemy(self):
        self.obs["cubes"] = [{"id": 5000, "pos": [90.0, 0.0, 0.0]}]
        self.obs["mechs"] = [{"id": 3, "kind": "mech", "pos": [45.0, 35.0, 0.0]}]
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger")[1], "cube")
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger", risk_aware=True)[1], "hold")
        self.obs["mechs"][0]["pos"] = [-60.0, 0.0, 0.0]
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger", risk_aware=True)[1], "cube")

    def test_risk_aware_flight_checks_the_approach_not_only_the_landing(self):
        self.obs["self"].update({"hp": 1400, "max_hp": 1400, "energy": 100})
        self.obs["cubes"] = [{"id": 5000, "pos": [90.0, 0.0, 20.0]}]
        self.obs["mechs"] = [{"id": 3, "kind": "mech", "pos": [45.0, 80.0, 0.0]}]
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger")[1], "cube")
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger", risk_aware=True)[1], "hold")

    def test_scavenger_flies_to_an_otherwise_unreachable_rooftop_cube(self):
        # 20 m up is far beyond the 4 m walkable-height check, but well within
        # reach of a sustained boost-climb given plenty of energy and no threats.
        self.obs["self"].update({"hp": 1000.0, "max_hp": 1400.0, "energy": 100.0,
                                 "stats": {"walk_speed": 10.0, "jet_drain": 15.0,
                                          "max_rise_speed": 10.0}})
        self.obs["cubes"] = [{"id": 5000, "pos": [5.0, 0.0, 20.0]}]
        goal, kind = choose_variant_goal(self.obs, self.world, "scavenger")
        self.assertEqual(kind, "cube")
        self.assertEqual(tuple(goal), (5.0, 0.0, 20.0))

    def test_rooftop_loot_is_skipped_without_enough_energy_or_hp(self):
        base = {"hp": 1000.0, "max_hp": 1400.0, "energy": 100.0,
               "stats": {"walk_speed": 10.0, "jet_drain": 15.0, "max_rise_speed": 10.0}}
        self.obs["cubes"] = [{"id": 5000, "pos": [5.0, 0.0, 20.0]}]
        self.obs["self"].update({**base, "energy": 30.0})  # climb budget ~3 m, not 20 m
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger")[1], "hold")
        self.obs["self"].update({**base, "hp": 600.0})  # hp_frac ~0.43, below the 0.6 floor
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger")[1], "hold")

    def test_rooftop_loot_is_skipped_if_any_enemy_is_within_the_wider_radius(self):
        self.obs["self"].update({"hp": 1000.0, "max_hp": 1400.0, "energy": 100.0,
                                 "stats": {"walk_speed": 10.0, "jet_drain": 15.0,
                                          "max_rise_speed": 10.0}})
        self.obs["cubes"] = [{"id": 5000, "pos": [5.0, 0.0, 20.0]}]
        self.obs["mechs"] = [{"id": 3, "kind": "mech", "pos": [60.0, 0.0, 0.0]}]
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger")[1], "hold")

    def test_scavenger_prefers_a_closer_ground_pickup_over_flying_for_a_rooftop_one(self):
        self.obs["self"].update({"hp": 1400.0, "max_hp": 1400.0, "energy": 100.0,
                                 "stats": {"walk_speed": 10.0, "jet_drain": 15.0,
                                          "max_rise_speed": 10.0},
                                 "weapons": [{"kind": "gun"}, {"kind": "mortar", "charge": 0}]})
        self.obs["pickups"] = [{"id": 1, "kind": "laser", "pos": [30.0, 0.0, 0.0]},
                               {"id": 2, "kind": "laser", "pos": [5.0, 0.0, 20.0]}]
        goal, kind = choose_variant_goal(self.obs, self.world, "scavenger")
        self.assertEqual(kind, "weapon")
        self.assertEqual(tuple(goal), (30.0, 0.0, 0.0))

    def test_drop_capture_stays_in_ring(self):
        self.obs["drops"] = [{"pos": [2.0, 0.0, 0.0], "radius": 8.0,
                              "lands_in": 0, "holder": 2}]
        goal, kind = choose_variant_goal(self.obs, self.world, "scavenger")
        self.assertEqual(kind, "capture")
        direction, speed = variant_movement("scavenger", self.obs, None, kind,
                                             (0, 0), (0, 0), (0, 0), 0, 0)
        self.assertEqual((direction, speed), ((0.0, 0.0), 0.0))

    def test_closest_eligible_drop_is_preferred_without_abandoning_capture(self):
        self.obs["drops"] = [
            {"pos": [50.0, 0.0, 0.0], "radius": 8.0, "lands_in": 0, "holder": None},
            {"pos": [10.0, 0.0, 0.0], "radius": 8.0, "lands_in": 0, "holder": None},
        ]
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger")[0][:2], (10.0, 0.0))
        self.obs["drops"] = [
            {"pos": [6.0, 0.0, 0.0], "radius": 8.0, "lands_in": 0, "holder": 2, "progress": 3},
            {"pos": [1.0, 0.0, 0.0], "radius": 8.0, "lands_in": 0, "holder": None},
        ]
        self.assertEqual(choose_variant_goal(self.obs, self.world, "scavenger")[0][:2], (6.0, 0.0))

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

    def test_tactician_avoids_moving_toward_a_second_attacker(self):
        self.obs["mechs"] = [
            {"id": 3, "kind": "mech", "pos": [-40.0, 0.0, 0.0], "aim": [0.0, 0.0]},
            {"id": 4, "kind": "mech", "pos": [35.0, 0.0, 0.0], "aim": [math.pi, 0.0]},
        ]
        direction, speed = tactician_movement(self.obs, self.world, self.obs["mechs"][0], "hold",
                                               (0, 0), (0, 0), (0, 0), 0.6, 0)
        self.assertLess(direction[0], 1e-8)
        self.assertGreater(abs(direction[1]), 0.9)
        self.assertGreater(speed, 0)

    def test_tactician_stays_inside_capture_ring_when_threat_is_distant(self):
        self.obs["mechs"] = [{"id": 3, "kind": "mech", "pos": [100.0, 0.0, 0.0]}]
        self.assertEqual(tactician_movement(self.obs, self.world, None, "capture",
                                            (0, 0), (0, 0), (0, 0), 0, 0), ((0.0, 0.0), 0.0))

    def test_tactician_does_not_replace_mandatory_zone_escape(self):
        self.obs["self"]["in_zone"] = False
        self.obs["mechs"] = [{"id": 3, "kind": "mech", "pos": [-40.0, 0.0, 0.0]}]
        direction, speed = tactician_movement(self.obs, self.world, self.obs["mechs"][0], "zone",
                                               (-1, 0), (1, 0), (0, 0), 0.8, 0)
        self.assertEqual(direction, (-1.0, 0.0))
        self.assertEqual(speed, 1.0)

    def test_vanguard_moves_inward_when_every_escape_points_outside_small_zone(self):
        self.obs["zone"]["radius"] = 10.0
        self.obs["zone"]["next_radius"] = 0.0
        self.obs["self"]["pos"] = [9.0, 0.0, 0.0]
        direction, speed = vanguard_movement(self.obs, self.world, None, "cube",
                                              (1, 0), (0, 0), (0, 0), 0, 0)
        self.assertEqual(direction, (-1.0, 0.0))
        self.assertGreater(speed, 0)

    def test_adaptive_zone_guard_compensates_for_outward_airborne_momentum(self):
        self.obs["zone"].update({"radius": 10.0, "next_radius": 10.0})
        self.obs["self"].update({"pos": [9.0, 0.0, 0.0], "vel": [10.0, 0.0, 0.0],
                                 "grounded": False, "stats": {"walk_speed": 10.0, "air_accel": 12.0}})
        direction, speed = adaptive_zone_guard(self.obs, self.world, (0.0, 1.0), 0.9, False)
        self.assertEqual(direction, (-1.0, 0.0))
        self.assertEqual(speed, 1.0)

    def test_adaptive_zone_guard_accounts_for_shrinking_boundary(self):
        self.obs["zone"].update({"radius": 10.0, "next_radius": 0.0,
                                 "shrink_start": 0.0, "shrink_end": 10.0})
        self.obs["self"].update({"pos": [9.8, 0.0, 0.0], "vel": [0.0, 0.0, 0.0],
                                 "grounded": True})
        direction, speed = adaptive_zone_guard(self.obs, self.world, (0.0, 0.0), 0.0, False)
        self.assertEqual(direction, (-1.0, 0.0))
        self.assertEqual(speed, 1.0)

    def test_adaptive_zone_guard_does_not_override_safe_or_outside_motion(self):
        self.obs["self"].update({"pos": [0.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0],
                                 "grounded": True})
        self.assertEqual(adaptive_zone_guard(self.obs, self.world, (1.0, 0.0), 0.9, False),
                         ((1.0, 0.0), 0.9))
        self.obs["self"].update({"pos": [210.0, 0.0, 0.0], "in_zone": False})
        self.assertEqual(adaptive_zone_guard(self.obs, self.world, (-1.0, 0.0), 1.0, False),
                         ((-1.0, 0.0), 1.0))

    def test_adaptive_zone_guard_does_not_select_blocked_escape(self):
        class WallWorld(FlatWorld):
            def blocked(self, start, end):
                return end[0] < start[0]

        self.obs["zone"].update({"radius": 10.0, "next_radius": 10.0})
        self.obs["self"].update({"pos": [9.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0],
                                 "grounded": True})
        direction, speed = adaptive_zone_guard(self.obs, WallWorld(), (1.0, 0.0), 1.0, False)
        self.assertNotEqual(direction, (-1.0, 0.0))
        self.assertLessEqual(direction[0] * speed, 0.0)

    def test_vanguard_stops_if_all_escape_directions_are_blocked(self):
        class BlockedWorld(FlatWorld):
            def blocked(self, start, goal):
                return True

        direction, speed = vanguard_movement(self.obs, BlockedWorld(), None, "cube",
                                              (1, 0), (0, 0), (0, 0), 0, 0)
        self.assertEqual((direction, speed), ((0.0, 0.0), 0.0))

    def test_badly_hurt_vanguard_prefers_a_walkable_direction_that_breaks_cover(self):
        class CoverWorld(FlatWorld):
            def blocked(self, start, goal):
                return False

            def line_of_sight(self, start, goal):
                return start[0] >= -2  # only the -x side has anything blocking sight

        enemy = {"id": 3, "kind": "mech", "pos": [-50.0, 0.0, 0.0]}
        # Healthy: takes the first walkable candidate regardless of visibility.
        direction, _ = vanguard_movement(self.obs, CoverWorld(), enemy, "hold",
                                         (0, 0), (0, 0), (0, 0), 0, 0)
        self.assertGreater(direction[0], 0)
        # Badly hurt: among the walkable candidates, prefers the one that would
        # also break the engaged target's line of sight.
        self.obs["self"].update({"hp": 50.0, "max_hp": 200.0})
        direction, speed = vanguard_movement(self.obs, CoverWorld(), enemy, "hold",
                                             (0, 0), (0, 0), (0, 0), 0, 0)
        self.assertGreater(speed, 0)
        self.assertLess(direction[0], 0)

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

    def radar_bot(self):
        self.obs.update({"tick": 0, "sounds": [], "events": [], "projectiles": []})
        self.obs["self"].update({
            "vel": [0.0, 0.0, 0.0], "aim": [0.0, 0.0],
            "hp": 1400, "max_hp": 1400, "energy": 100, "grounded": True,
            "stats": {"walk_speed": 10.0, "turn_rate": 4.0},
            "weapons": [{"kind": "gun", "ammo": 12, "cooldown": 0, "reload": 0,
                         "stats": {"speed": 252, "range": 230}}, None],
        })
        hello = {"config": {"mech": {"min_pitch": -0.7, "max_pitch": 1.3},
                            "match": {"tick_hz": 20, "decide_every": 2},
                            "vision": {"fov_deg": 120}}}
        return MechRoyaleBot(hello, strategy="vanguard", world=self.world)

    def test_tactician_preserves_adaptive_targeting_and_fire(self):
        baseline = self.radar_bot()
        from adaptive import DEFAULT_PARAMETERS
        baseline = MechRoyaleBot(baseline.hello, strategy="adaptive", world=self.world,
                                 policy_parameters=DEFAULT_PARAMETERS)
        candidate = MechRoyaleBot(baseline.hello, strategy="tactician", world=self.world,
                                  policy_parameters=DEFAULT_PARAMETERS)
        self.obs["mechs"] = [{"id": 3, "kind": "mech", "pos": [100.0, 0.0, 0.0],
                               "vel": [0.0, 0.0, 0.0], "aim": [math.pi, 0.0], "hp": 1400}]
        first, second = baseline.act(self.obs), candidate.act(self.obs)
        self.assertEqual((first["aim"], first["fire"], first["reload"]),
                         (second["aim"], second["fire"], second["reload"]))

    def test_idle_scan_covers_full_circle_from_actual_torso_angle(self):
        bot = self.radar_bot()
        self.obs["self"]["aim"][0] = 2.9
        travel = 0.0
        for i in range(34):
            self.obs["tick"], self.obs["t"] = i * 2, i * 0.1
            actual = self.obs["self"]["aim"][0]
            command = bot.act(self.obs)["aim"][0]
            updated = slew_angle(actual, command, 4.0, 0.1)
            travel += angle_delta(updated, actual)
            self.obs["self"]["aim"][0] = updated
        self.assertGreater(travel, 2 * math.pi)

    def test_repeated_sounds_do_not_stop_scan_and_damage_takes_priority(self):
        bot = self.radar_bot()
        self.obs["sounds"] = [{"kind": "gun", "bearing": 1.5, "age": 0.0, "dist": "far"}]
        self.obs["events"] = [{"type": "damaged", "bearing": -1.0, "weapon": "gun"}]
        self.assertAlmostEqual(bot.act(self.obs)["aim"][0], -1.0)
        self.obs["events"] = []
        self.assertAlmostEqual(bot.act(self.obs)["aim"][0], 1.5)
        travel = 0.0
        for i in range(1, 41):
            self.obs["tick"], self.obs["t"] = i * 2, i * 0.1
            actual = self.obs["self"]["aim"][0]
            command = bot.act(self.obs)["aim"][0]
            updated = slew_angle(actual, command, 4.0, 0.1)
            travel += angle_delta(updated, actual)
            self.obs["self"]["aim"][0] = updated
        self.assertGreater(travel, 2 * math.pi)

    def test_target_aim_interrupts_scan_and_zone_damage_does_not_stop_it(self):
        bot = self.radar_bot()
        self.obs["events"] = [{"type": "damaged", "bearing": None, "weapon": "zone"}]
        self.assertGreater(bot.act(self.obs)["aim"][0], 0)
        self.obs["mechs"] = [{"id": 3, "kind": "mech", "pos": [0.0, 100.0, 0.0],
                               "vel": [0.0, 0.0, 0.0], "aim": [0.0, 0.0], "hp": 1400}]
        self.assertAlmostEqual(bot.act(self.obs)["aim"][0], math.pi / 2, places=5)

    def test_rear_defense_does_not_fire_using_the_previous_aim(self):
        bot = self.radar_bot()
        self.obs["self"]["aim"] = [0.0, math.atan2(-2.5, 100)]
        self.obs["mechs"] = [{"id": 3, "kind": "mech", "pos": [100.0, 0.0, 0.0],
                               "vel": [0.0, 0.0, 0.0], "aim": [math.pi, 0.0], "hp": 1400}]
        self.obs["events"] = [{"type": "damaged", "weapon": "gun", "by": 4,
                                "amount": 45, "bearing": math.pi / 2, "rear": True}]
        action = bot.act(self.obs)
        self.assertAlmostEqual(action["aim"][0], math.pi / 2, places=5)
        self.assertFalse(action["fire"][0])

    def test_rear_defense_holds_charged_laser_instead_of_firing_off_target(self):
        bot = self.radar_bot()
        self.obs["self"]["aim"] = [0.0, math.atan2(-2.5, 100)]
        self.obs["self"]["weapons"][1] = {
            "kind": "laser", "charge": 0.75, "cooldown": 0,
            "stats": {"charge": 0.75, "range": 280},
        }
        self.obs["mechs"] = [{"id": 3, "kind": "mech", "pos": [100.0, 0.0, 0.0],
                               "vel": [0.0, 0.0, 0.0], "aim": [math.pi, 0.0], "hp": 1400}]
        self.obs["events"] = [{"type": "damaged", "weapon": "gun", "by": 4,
                                "amount": 45, "bearing": math.pi / 2, "rear": True}]
        action = bot.act(self.obs)
        self.assertAlmostEqual(action["aim"][0], math.pi / 2, places=5)
        self.assertTrue(action["fire"][1])

    def test_weapon_swap_waits_until_immediate_threat_passes(self):
        bot = self.radar_bot()
        self.obs["self"]["weapons"][1] = {"kind": "mortar", "cooldown": 0, "stats": {}}
        self.obs["pickups"] = [{"id": 8000, "kind": "laser", "pos": [2, 0, 0]}]
        self.obs["mechs"] = [{"id": 3, "kind": "mech", "pos": [20, 0, 0],
                               "vel": [0, 0, 0], "aim": [math.pi, 0], "hp": 1400}]
        action = bot.act(self.obs)
        self.assertNotIn("take", action)
        self.obs["mechs"] = []
        action = bot.act(self.obs)
        self.assertEqual((action["take"], action["drop"]), (8000, 1))

    def test_unknown_attacker_damage_suppresses_crate_farming(self):
        bot = self.radar_bot()
        self.obs["crates"] = [{"id": 1000, "pos": [30.0, 0.0, 0.0]}]
        self.obs["events"] = [{"type": "damaged", "weapon": "gun", "by": None,
                                "amount": 45, "bearing": 0.0}]
        bot.act(self.obs)
        self.assertIsNone(bot.current_combat_target)
        self.assertEqual(bot.last_damage_t, 0.0)

    def test_largest_rear_hit_controls_defensive_facing_and_attacker(self):
        bot = self.radar_bot()
        self.obs["events"] = [
            {"type": "damaged", "weapon": "mortar", "by": 3, "amount": 200,
             "bearing": 1.0, "rear": True},
            {"type": "damaged", "weapon": "gun", "by": 4, "amount": 45,
             "bearing": -1.0, "rear": True},
        ]
        self.assertEqual(bot.act(self.obs)["aim"][0], 1.0)
        self.assertEqual(bot.last_attacker, 3)

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
