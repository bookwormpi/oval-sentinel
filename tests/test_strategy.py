import sys, unittest
from unittest.mock import patch
sys.path.insert(0, "bot")
from strategy import _effect_value, choose_take, choose_upgrade, should_jump, survival_boost

class StrategyTests(unittest.TestCase):
    def test_take_only_nearby_weapon(self):
        obs = {"self": {"pos": [0,0,0], "weapons": [{"kind":"gun"}, None]},
               "pickups": [{"id": 1, "kind":"laser", "pos":[3,0,0]},
                           {"id": 2, "kind":"mortar", "pos":[10,0,0]}]}
        self.assertEqual(choose_take(obs), (1, None))

    def test_take_swaps_a_clearly_worse_held_secondary(self):
        obs = {"self": {"pos": [0, 0, 0],
                        "weapons": [{"kind": "gun"}, {"kind": "mortar", "charge": 0}]},
               "pickups": [{"id": 5, "kind": "laser", "pos": [2, 0, 0]}]}
        self.assertEqual(choose_take(obs), (5, 1))

    def test_take_does_not_swap_for_a_marginal_or_worse_weapon(self):
        obs = {"self": {"pos": [0, 0, 0],
                        "weapons": [{"kind": "gun"}, {"kind": "laser", "charge": 0}]},
               "pickups": [{"id": 5, "kind": "lightning", "pos": [2, 0, 0]}]}
        self.assertEqual(choose_take(obs), (None, None))

    def test_take_does_not_drop_a_charging_laser(self):
        # Laser is already the top-valued secondary, so nothing currently beats
        # it; patch in a hypothetical better kind to exercise the charge guard
        # regardless of today's weapon ranking.
        obs = {"self": {"pos": [0, 0, 0],
                        "weapons": [{"kind": "gun"}, {"kind": "laser", "charge": 0.4}]},
               "pickups": [{"id": 5, "kind": "lightning", "pos": [2, 0, 0]}]}
        with patch("strategy.WEAPON_VALUE", {"laser": 1.0, "lightning": 1.5}):
            self.assertEqual(choose_take(obs), (None, None))
            obs["self"]["weapons"][1]["charge"] = 0.0
            self.assertEqual(choose_take(obs), (5, 1))

    def test_take_avoids_weapons_already_held_in_either_slot(self):
        obs = {"self": {"pos": [0, 0, 0], "weapons": [{"kind": "laser"}, None]},
               "pickups": [{"id": 5, "kind": "laser", "pos": [2, 0, 0]},
                           {"id": 6, "kind": "mortar", "pos": [3, 0, 0]}]}
        self.assertEqual(choose_take(obs), (6, None))

    def test_take_uses_configured_horizontal_radius_and_fixed_vertical_limit(self):
        obs = {"self": {"pos": [0, 0, 0], "weapons": [{"kind": "gun"}, None]},
               "pickups": [{"id": 5, "kind": "laser", "pos": [5, 0, 0]}]}
        self.assertEqual(choose_take(obs, {"loot": {"pickup_radius": 6}}), (5, None))
        obs["pickups"][0]["pos"][2] = 4.1
        self.assertEqual(choose_take(obs, {"loot": {"pickup_radius": 6}}), (None, None))

    def test_lethal_zone_escape_uses_remaining_energy_for_a_short_boost(self):
        obs = {"self": {"in_zone": False, "grounded": True, "energy": 5,
                        "hp": 10, "max_hp": 1400, "stats": {"walk_speed": 10}, "weapons": []},
               "zone": {"dps": 160}}
        self.assertTrue(survival_boost(obs, 4))
        obs["self"]["hp"] = 1400
        self.assertFalse(survival_boost(obs, 4))

    def test_airborne_zone_escape_does_not_waste_energy_on_vertical_jets(self):
        obs = {"self": {"in_zone": False, "grounded": False, "energy": 100}}
        self.assertFalse(survival_boost(obs, 50, 0))
        self.assertTrue(survival_boost(obs, 50, 0.8))

    def test_imminent_projectile_can_trigger_free_jump_at_low_energy(self):
        obs = {"self": {"grounded": True, "energy": 0}}
        self.assertTrue(should_jump(obs, 0.9, 0.9))
        self.assertFalse(should_jump(obs, 0.9, 0.0))

    def test_upgrade_armor(self):
        hello = {"config": {"upgrades": [{"id":"armor", "max":3, "effects":[]}]}}
        obs = {"offer": {"choices": [{"id":"armor"}]},
               "self": {"upgrades": [], "weapons": [{"kind":"gun"}, None],
                        "hp": 500, "max_hp": 1400}, "alive": 16}
        self.assertEqual(choose_upgrade(obs, hello), 0)

    def test_upgrade_can_stack_until_configured_max(self):
        hello = {"config": {"upgrades": [{"id":"armor", "max":3, "effects":[]}]}}
        obs = {"offer": {"choices": [{"id":"armor"}]},
               "self": {"upgrades": ["armor", "armor"],
                        "weapons": [{"kind":"gun"}, None],
                        "hp": 1400, "max_hp": 1400}, "alive": 10}
        self.assertEqual(choose_upgrade(obs, hello), 0)

    def test_upgrade_respects_stack_cap(self):
        hello = {"config": {"upgrades": [{"id":"armor", "max":2, "effects":[]}]}}
        obs = {"offer": {"choices": [{"id":"armor"}]},
               "self": {"upgrades": ["armor", "armor"],
                        "weapons": [{"kind":"gun"}, None],
                        "hp": 1400, "max_hp": 1400}, "alive": 10}
        self.assertIsNone(choose_upgrade(obs, hello))

    def test_multiplicative_hp_upgrade_beats_additive_armor(self):
        # +15% HP multiplicatively compounds across stacks; -8% damage taken is
        # only worth ~8.7% effective HP on the first stack, so a fresh mech
        # should value plating higher than armor (the old fixed scores did not).
        mech_stats = {"hp": 1400.0, "armor": 0.0}
        plating = _effect_value({"path": "mech.hp", "mul": 1.15}, mech_stats, {}, {})
        armor = _effect_value({"path": "mech.armor", "add": 0.08}, mech_stats, {}, {})
        self.assertGreater(plating, armor)

    def test_cost_stat_valued_by_reciprocal_throughput_not_raw_percentage(self):
        # A 25% cooldown cut buys 1/0.75 - 1 = 33% more shots, not 25%: the
        # reciprocal transform should value it above a naive percentage would.
        mech_stats = {}
        cooldown_effect = {"path": "weapons.gun.cooldown", "mul": 0.75}
        value = _effect_value(cooldown_effect, mech_stats, {"gun": {"cooldown": 1.0}}, {})
        self.assertGreater(value, 0.25 * 0.95)  # 0.95 is cooldown's STAT_WEIGHT

    def test_zero_baseline_bonus_stats_do_not_blow_up(self):
        # Regression test: near_range_bonus/vision_bonus start at 0 and are meter
        # bonuses on a separate base range, not a self-relative ratio; a naive
        # ratio against ~0 previously made "sensors" absurdly overvalued.
        mech_stats = {"near_range_bonus": 0.0, "vision_bonus": 0.0, "hearing": 1.0}
        vision = {"near_range": 45.0, "range": 200.0}
        sensors_effects = [
            {"path": "mech.near_range_bonus", "add": 15.0},
            {"path": "mech.vision_bonus", "add": 25.0},
            {"path": "mech.hearing", "mul": 0.5},
        ]
        total = sum(_effect_value(e, mech_stats, {}, vision) for e in sensors_effects)
        self.assertLess(total, 0.5)

    def test_full_magazine_upgrade_reflects_its_large_ammo_increase(self):
        mech_stats = {}
        value = _effect_value({"path": "weapons.gun.mag", "add": 8}, mech_stats,
                              {"gun": {"mag": 12}}, {})
        self.assertAlmostEqual(value, 0.45 * (8 / 12))

if __name__ == "__main__":
    unittest.main()

