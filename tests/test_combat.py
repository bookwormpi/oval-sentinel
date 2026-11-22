import math
import sys, unittest
sys.path.insert(0, "bot")
from combat import choose_target, combat_action, projectile_dodge, predict_target


class CombatTests(unittest.TestCase):
    def setUp(self):
        self.config = {"mech": {"min_pitch": -0.7, "max_pitch": 1.3}}
        self.me = {
            "id": 1, "pos": [0.0, 0.0, 0.0], "vel": [0.0, 0.0, 0.0],
            "aim": [0.0, 0.0], "energy": 100.0,
            "stats": {"muzzle_height": 7.0, "height": 9.0},
            "weapons": [
                {"kind": "gun", "cooldown": 0.0, "ammo": 12, "reload": 0.0,
                 "stats": {"speed": 252, "range": 230}},
                {"kind": "laser", "cooldown": 0.0, "charge": 0.0,
                 "stats": {"charge": 0.75, "min_charge": 0.15, "range": 280}},
            ],
        }
        self.target = {"id": 2, "kind": "mech", "pos": [0.0, 120.0, 0.0],
                       "vel": [0.0, 0.0, 0.0], "aim": [0.0, 0.0], "hp": 900}

    def test_target_selection_uses_real_max_hp_not_a_fixed_baseline(self):
        # A cube-buffed enemy's max HP can exceed the base 1400. Scoring its
        # remaining fraction against a fixed 1400 baseline would rank it as
        # healthier (50%) than a non-buffed enemy at 43% real HP, when its real
        # fraction (36%) is actually lower: using the real max HP flips that.
        me = {"pos": [0.0, 0.0, 0.0]}
        buffed = {"id": 2, "kind": "mech", "pos": [50.0, 0.0, 0.0],
                 "aim": [0.0, 0.0], "hp": 700.0, "max_hp": 1960.0}
        baseline = {"id": 3, "kind": "mech", "pos": [50.0, 0.0, 0.0],
                   "aim": [0.0, 0.0], "hp": 600.0, "max_hp": 1400.0}
        self.assertEqual(choose_target({"mechs": [buffed, baseline]}, me)["id"], 2)

    def test_near_death_target_is_preferred_to_deny_the_kill(self):
        me = {"pos": [0.0, 0.0, 0.0]}
        near_death = {"id": 2, "kind": "mech", "pos": [50.0, 0.0, 0.0],
                     "aim": [math.pi, 0.0], "hp": 100.0, "max_hp": 1400.0}
        full_health_closer = {"id": 3, "kind": "mech", "pos": [40.0, 0.0, 0.0],
                              "aim": [math.pi, 0.0], "hp": 1400.0, "max_hp": 1400.0}
        self.assertEqual(choose_target({"mechs": [near_death, full_health_closer]}, me)["id"], 2)

    def test_zero_reported_hp_is_not_treated_as_unknown(self):
        empty = dict(self.target, id=2, hp=0.0, pos=[50.0, 0.0, 0.0])
        healthy = dict(self.target, id=3, hp=500.0, pos=[50.0, 0.0, 0.0])
        self.assertEqual(choose_target({"mechs": [empty, healthy]}, self.me)["id"], 2)

    def test_precision_targets_actual_scaled_mech_and_crate_centers(self):
        gun = self.me["weapons"][0]
        target = dict(self.target, pos=[100, 0, 0], cubes=10)
        _, pitch, _, _ = predict_target(self.me, target, gun, self.config, True)
        self.assertAlmostEqual(pitch, math.atan2(6.75 - 7, 100))
        target = {"kind": "crate", "pos": [30, 0, 0], "vel": [0, 0, 0]}
        _, pitch, _, _ = predict_target(self.me, target, gun, self.config, True)
        self.assertAlmostEqual(pitch, math.atan2(1.2 - 7, 30))

    def test_precision_targets_configured_npc_height(self):
        target = {"kind": "npc", "pos": [50, 0, 0], "vel": [0, 0, 0]}
        config = {**self.config, "npc": {"height": 8}}
        _, pitch, _, _ = predict_target(self.me, target, self.me["weapons"][0], config, True)
        self.assertAlmostEqual(pitch, math.atan2(4 - 7, 50))

    def test_pitch_limit_does_not_make_an_unreachable_target_appear_aligned(self):
        self.me["weapons"][1] = None
        self.config["mech"]["max_pitch"] = 0.3
        self.me["aim"] = [0.0, 0.3]
        self.target["pos"] = [30.0, 0.0, 100.0]
        _, fire, _, _ = combat_action({}, self.me, self.target, self.config,
                                       correct_mortar=True, precision=True)
        self.assertFalse(fire[0])

    def test_unreachable_ballistic_solution_does_not_fire_mortar(self):
        self.me["weapons"] = [None, {"kind": "mortar", "cooldown": 0.0,
                                     "stats": {"speed": 72, "gravity": 22}}]
        self.target["pos"] = [100.0, 0.0, 300.0]
        _, pitch, _, _ = predict_target(self.me, self.target, self.me["weapons"][1], self.config, True)
        self.assertIsNone(pitch)
        _, fire, _, _ = combat_action({}, self.me, self.target, self.config,
                                       correct_mortar=True, precision=True)
        self.assertFalse(fire[1])

    def test_upgraded_mortar_can_fire_beyond_legacy_215_m_cutoff(self):
        mortar = {"kind": "mortar", "cooldown": 0.0, "stats": {"speed": 100, "gravity": 22}}
        self.me["weapons"] = [None, mortar]
        self.target["pos"] = [250.0, 0.0, 0.0]
        yaw, pitch, _, _ = predict_target(self.me, self.target, mortar, self.config, True)
        self.me["aim"] = [yaw, pitch]
        _, fire, _, _ = combat_action({}, self.me, self.target, self.config,
                                       correct_mortar=True, precision=True)
        self.assertTrue(fire[1])

    def test_out_of_range_gun_does_not_override_usable_mortar(self):
        self.me["weapons"][0]["stats"]["range"] = 50
        self.me["weapons"][1] = {"kind": "mortar", "cooldown": 0,
                                  "stats": {"speed": 72, "gravity": 22}}
        self.target["pos"] = [100.0, 0.0, 0.0]
        aim, _, _, _ = combat_action({}, self.me, self.target, self.config,
                                      correct_mortar=True, precision=True)
        self.assertGreater(aim[1], 0.1)

    def test_out_of_range_laser_does_not_override_usable_gun(self):
        self.me["weapons"][1]["stats"]["range"] = 50
        self.target["pos"], self.target["vel"] = [100.0, 0.0, 0.0], [0.0, 20.0, 0.0]
        aim, _, _, _ = combat_action({}, self.me, self.target, self.config,
                                      correct_mortar=True, precision=True)
        self.assertGreater(aim[0], 0.05)

    def test_reload_finds_gun_in_secondary_slot(self):
        self.me["weapons"] = [{"kind": "mortar"}, {"kind": "gun", "ammo": 0, "reload": 0}]
        _, _, reload_now, _ = combat_action({}, self.me, None, self.config)
        self.assertTrue(reload_now)

    def test_secondary_laser_controls_aim_at_range(self):
        aim, fire, _, _ = combat_action({"mechs": []}, self.me, self.target, self.config)
        self.assertAlmostEqual(aim[0], math.pi/2, places=3)
        self.assertFalse(fire[1])  # actual torso aim has not slewed to the commanded aim yet

    def test_charging_laser_keeps_holding_while_tracking(self):
        self.me["weapons"][1]["charge"] = 0.3
        # Simulate a target whose lead has shifted slightly from the current torso aim.
        self.target["vel"] = [2.0, 0.0, 0.0]
        aim, fire, _, _ = combat_action({"mechs": []}, self.me, self.target, self.config)
        self.assertTrue(fire[1])

    def test_commanded_aim_holds_charge_only_while_in_range_and_visible(self):
        self.me["weapons"][1]["charge"] = 0.3
        # commanded_aim disagrees with the solved aim (yaw ~pi/2): a diverging neural
        # command should keep holding the charge while the shot is still viable...
        _, fire, _, _ = combat_action({"mechs": []}, self.me, self.target, self.config,
                                      commanded_aim=[0.0, 0.0])
        self.assertTrue(fire[1])
        # ...but not force a hold (and the eventual wasted hold_s release) once the
        # target is out of the laser's range.
        self.target["pos"] = [0.0, 10000.0, 0.0]
        _, fire, _, _ = combat_action({"mechs": []}, self.me, self.target, self.config,
                                      commanded_aim=[0.0, 0.0])
        self.assertFalse(fire[1])

    def test_full_laser_charge_releases(self):
        self.me["weapons"][1]["charge"] = 0.75
        _, fire, _, _ = combat_action({"mechs": []}, self.me, self.target, self.config)
        self.assertFalse(fire[1])

    def test_projectile_above_mech_is_not_false_positive(self):
        obs = {"projectiles": [{"kind": "gun", "pos": [0, -10, 30], "vel": [0, 20, 0]}]}
        _, urgency = projectile_dodge(obs, self.me)
        self.assertEqual(urgency, 0.0)

    def test_projectile_on_collision_course_triggers_dodge(self):
        obs = {"projectiles": [{"kind": "gun", "pos": [0, -10, 4], "vel": [0, 20, 0]}]}
        _, urgency = projectile_dodge(obs, self.me)
        self.assertGreater(urgency, 0.0)

    def test_projectile_moving_away_is_ignored(self):
        obs = {"projectiles": [{"kind": "gun", "pos": [0, -3, 4], "vel": [0, -20, 0]}]}
        _, urgency = projectile_dodge(obs, self.me)
        self.assertEqual(urgency, 0.0)

    def test_projectile_about_to_hit_is_not_discarded(self):
        obs = {"projectiles": [{"kind": "gun", "pos": [0, -0.2, 4], "vel": [0, 20, 0]}]}
        _, urgency = projectile_dodge(obs, self.me, correct_side=True)
        self.assertGreater(urgency, 0.0)

    def test_vertical_projectile_on_collision_course_triggers_lateral_dodge(self):
        obs = {"projectiles": [{"kind": "gun", "pos": [0, 0, 20], "vel": [0, 0, -40]}]}
        direction, urgency = projectile_dodge(obs, self.me, correct_side=True)
        self.assertGreater(urgency, 0.0)
        self.assertGreater(math.hypot(*direction), 0.9)
        obs["projectiles"][0]["vel"][2] = 40
        self.assertEqual(projectile_dodge(obs, self.me, correct_side=True)[1], 0.0)

    def test_dodge_accounts_for_increased_mech_radius(self):
        self.me["stats"]["radius"] = 10.0
        obs = {"projectiles": [{"kind": "gun", "pos": [8, -10, 4], "vel": [0, 20, 0]}]}
        self.assertGreater(projectile_dodge(obs, self.me, correct_side=True)[1], 0)

    def test_mortar_dodge_uses_configured_gravity(self):
        obs = {"projectiles": [{"kind": "mortar", "pos": [0, -10, 20], "vel": [0, 20, 0]}]}
        self.assertEqual(projectile_dodge(obs, self.me)[1], 0)
        config = {"weapons": {"mortar": {"gravity": 100}}}
        self.assertGreater(projectile_dodge(obs, self.me, config=config)[1], 0)

    def test_mortar_dodge_uses_configured_splash_radius(self):
        obs = {"projectiles": [{"kind": "mortar", "pos": [18, -10, 4], "vel": [0, 20, 0]}]}
        self.assertEqual(projectile_dodge(obs, self.me)[1], 0)
        config = {"weapons": {"mortar": {"splash_radius": 20}}}
        self.assertGreater(projectile_dodge(obs, self.me, config=config)[1], 0)

    def test_corrected_dodge_moves_away_from_projectile_path(self):
        for shot_x in (-1.0, 1.0):
            with self.subTest(shot_x=shot_x):
                obs = {"projectiles": [{"kind": "gun", "pos": [shot_x, -10, 4], "vel": [0, 20, 0]}]}
                direction, urgency = projectile_dodge(obs, self.me, correct_side=True)
                self.assertGreater(urgency, 0)
                self.assertLess(direction[0] * shot_x, 0)

    def test_precision_does_not_fire_before_actual_torso_alignment(self):
        self.me["weapons"][1] = None
        self.target["pos"] = [100.0, 10.0, 0.0]
        _, legacy_fire, _, _ = combat_action({}, self.me, self.target, self.config)
        _, precision_fire, _, _ = combat_action({}, self.me, self.target, self.config, precision=True)
        self.assertTrue(legacy_fire[0])
        self.assertFalse(precision_fire[0])

    def test_precision_uses_target_size_at_range(self):
        self.me["weapons"][1] = None
        self.me["aim"] = [0.05, 0.0]
        self.target["pos"] = [100.0, 0.0, 0.0]
        _, legacy_fire, _, _ = combat_action({}, self.me, self.target, self.config)
        _, precision_fire, _, _ = combat_action({}, self.me, self.target, self.config, precision=True)
        self.assertTrue(legacy_fire[0])
        self.assertFalse(precision_fire[0])
        self.me["aim"] = [0.0, math.atan2(-2.5, 100)]
        _, precision_fire, _, _ = combat_action({}, self.me, self.target, self.config, precision=True)
        self.assertTrue(precision_fire[0])

    def test_precision_holds_full_laser_until_aligned(self):
        self.me["weapons"][1]["charge"] = 0.75
        _, fire, _, _ = combat_action({}, self.me, self.target, self.config, precision=True)
        self.assertTrue(fire[1])
        self.me["aim"] = [math.pi / 2, math.atan2(-2.5, 120)]
        _, fire, _, _ = combat_action({}, self.me, self.target, self.config, precision=True)
        self.assertFalse(fire[1])

    def test_precision_does_not_aim_unavailable_laser_over_gun(self):
        self.me["weapons"][1]["cooldown"] = 1.0
        self.target["pos"] = [100.0, 0.0, 0.0]
        self.target["vel"] = [0.0, 20.0, 0.0]
        aim, _, _, _ = combat_action({}, self.me, self.target, self.config, precision=True)
        self.assertGreater(aim[0], 0.05)

    def test_shooter_velocity_is_used_for_gun_interception(self):
        self.me["vel"] = [100.0, 0.0, 0.0]
        self.target["pos"] = [0.0, 100.0, 0.0]
        self.target["vel"] = [100.0, 0.0, 0.0]
        aim, _, _, _ = combat_action({"mechs": []}, self.me, self.target, self.config)
        self.assertAlmostEqual(aim[0], math.pi/2, places=3)

    def test_mortar_aim_accounts_for_gravity_during_target_prediction(self):
        mortar = {"kind": "mortar", "cooldown": 0.0,
                  "stats": {"speed": 72, "gravity": 22, "range": 235}}
        _, pitch, _, _ = predict_target(
            self.me,
            {"id": 2, "kind": "mech", "pos": [100.0, 0.0, 0.0],
             "vel": [0.0, 0.0, 10.0], "aim": [0.0, 0.0]},
            mortar,
            self.config,
        )
        self.assertGreater(pitch, 0.0)

    def test_corrected_mortar_intercepts_stationary_and_moving_targets(self):
        mortar = {"kind": "mortar", "stats": {"speed": 72, "gravity": 22}}
        for velocity in ([0.0, 0.0, 0.0], [5.0, 7.0, 0.0]):
            with self.subTest(velocity=velocity):
                target = {"kind": "mech", "pos": [100.0, 0.0, 0.0], "vel": velocity}
                yaw, pitch, _, _ = predict_target(self.me, target, mortar, self.config, True)
                vx = 72 * math.cos(pitch) * math.cos(yaw)
                vy = 72 * math.cos(pitch) * math.sin(yaw)
                flight = 100.0 / (vx - velocity[0])
                self.assertAlmostEqual(vy * flight, velocity[1] * flight, delta=0.1)
                shell_z = 9.5 + 72 * math.sin(pitch) * flight - 11 * flight * flight
                self.assertAlmostEqual(shell_z, 4.5, delta=0.1)


if __name__ == "__main__":
    unittest.main()
