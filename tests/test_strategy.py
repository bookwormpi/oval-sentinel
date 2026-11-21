import sys, unittest
sys.path.insert(0, "bot")
from strategy import choose_upgrade, choose_take

class StrategyTests(unittest.TestCase):
    def test_upgrade_armor(self):
        hello = {"config":{"upgrades":[{"id":"armor","max":3,"effects":[]},{"id":"gun_rate","max":3,"effects":[]}]}}
        obs = {"offer":{"choices":[{"id":"gun_rate"},{"id":"armor"}]},
               "self":{"hp":500,"max_hp":1400,"upgrades":[],"weapons":[{"kind":"gun"},None]},
               "alive":8}
        self.assertEqual(choose_upgrade(obs,hello),1)
    def test_take_only_nearby_weapon(self):
        obs={"self":{"pos":[0,0,0],"weapons":[{"kind":"gun"},None]},
             "pickups":[{"id":8001,"kind":"laser","pos":[2,0,0]},
                        {"id":8002,"kind":"mortar","pos":[8,0,0]}]}
        self.assertEqual(choose_take(obs),8001)

if __name__ == "__main__":
    unittest.main()
