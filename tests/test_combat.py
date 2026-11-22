import math
import sys, unittest
sys.path.insert(0, "bot")
from combat import combat_action, projectile_dodge


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

    def test_shooter_velocity_is_used_for_gun_interception(self):
        self.me["vel"] = [100.0, 0.0, 0.0]
        self.target["pos"] = [0.0, 100.0, 0.0]
        self.target["vel"] = [100.0, 0.0, 0.0]
        aim, _, _, _ = combat_action({"mechs": []}, self.me, self.target, self.config)
        self.assertAlmostEqual(aim[0], math.pi/2, places=3)


if __name__ == "__main__":
    unittest.main()
