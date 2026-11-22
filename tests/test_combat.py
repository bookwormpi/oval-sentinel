import math
import sys, unittest
sys.path.insert(0, "bot")
from combat import combat_action, projectile_dodge, predict_target


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
                shell_z = 7.0 + 72 * math.sin(pitch) * flight - 11 * flight * flight
                self.assertAlmostEqual(shell_z, 4.5, delta=0.1)


if __name__ == "__main__":
    unittest.main()
