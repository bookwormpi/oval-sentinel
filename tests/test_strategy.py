import sys, unittest
sys.path.insert(0, "bot")
from strategy import choose_take, choose_upgrade

class StrategyTests(unittest.TestCase):
    def test_take_only_nearby_weapon(self):
        obs = {"self": {"pos": [0,0,0], "weapons": [{"kind":"gun"}, None]},
               "pickups": [{"id": 1, "kind":"laser", "pos":[3,0,0]},
                           {"id": 2, "kind":"mortar", "pos":[10,0,0]}]}
        self.assertEqual(choose_take(obs), 1)

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

if __name__ == "__main__":
    unittest.main()
