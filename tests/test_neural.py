import copy
import gzip
import json
import math
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simulator.engine import ROOT, Simulator, load_config
from simulator.neural_train import evaluate_population, evaluate_validation, reproduce
from neural import (CONFIG_BUCKETS, GLOBAL_SIZE, INPUT_SIZE, NeuralBot, _compressed, _map_distances,
                    encode_observation, load_model, random_model)
from test_simulator import FlatWorld, factory


class NeuralTests(unittest.TestCase):
    def setUp(self):
        self.config = copy.deepcopy(load_config())
        self.config["crates"]["count"] = self.config["npc"]["count"] = self.config["loot"]["floor_count"] = 0
        sim = Simulator(["vanguard", "vanguard"], 42, self.config, FlatWorld(), factory)
        self.obs = sim.observation(sim.mechs[0])
        self.hello = {"config": self.config, "seed": 1000}
        self.model = random_model(random.Random(4), 100)

    def test_normalized_inputs_include_all_observation_fields_and_nearby_distances(self):
        obs = self.obs
        x, y, z = obs["self"]["pos"]
        obs["cubes"] = [{"id": 5000 + i, "pos": [x + 10 * i, y, z]} for i in range(1, 7)]
        config_vector = _compressed(self.config, "config", CONFIG_BUCKETS, self.config)
        inputs = encode_observation(obs, self.config, config_vector, 50)
        self.assertEqual(len(inputs), INPUT_SIZE)
        self.assertTrue(all(-1 <= v <= 1 for v in inputs))
        cube_start = GLOBAL_SIZE + 3 * 5
        self.assertEqual(inputs[cube_start:cube_start + 4], [0.8, 0.6, 0.4, 0.19999999999999996])
        self.assertAlmostEqual(inputs[cube_start + 4], 5 / 9)
        altered = copy.deepcopy(obs)
        altered["events"] = [{"type": "pickup", "cubes": 9}]
        self.assertNotEqual(inputs, encode_observation(altered, self.config, config_vector, 50))
        narrow = encode_observation(obs, self.config, config_vector, 15)
        self.assertAlmostEqual(narrow[cube_start + 4], 1 / 5)

    def test_move_rotates_with_actual_facing_and_aim_respects_pitch_limits(self):
        bot = NeuralBot(self.hello, world=FlatWorld(), model=self.model)
        self.obs["self"]["aim"] = [math.pi / 2, 1.2]
        with patch("neural.forward", return_value=[1.0, 0.0, 0.0, 1.0]), \
                patch.object(bot.automatic, "act", return_value={"aim": [0.0, 0.0], "fire": [True, False]}):
            action = bot.act(self.obs)
        self.assertAlmostEqual(action["move"][0], 0, places=5)
        self.assertAlmostEqual(action["move"][1], 1, places=5)
        self.assertAlmostEqual(action["aim"][0], math.pi / 2, places=5)
        self.assertEqual(action["aim"][1], self.config["mech"]["max_pitch"])
        self.assertFalse(action["fire"][0])

    def test_static_map_distances_use_radius_and_height(self):
        index = ([("prisms", (10, -2, 20, 2, 0, 5)), ("trees", (60, 0, 65, 5, 0, 10))],
                 {(0, -1): [0], (0, 0): [0], (1, 0): [1]})
        distances = _map_distances(index, 0, 0, 0, 20)
        self.assertEqual(distances["prisms"], [10])
        self.assertEqual(distances["trees"], [])
        distances = _map_distances(index, 15, 0, 15, 20)
        self.assertEqual(distances["prisms"], [10])

    def test_charging_laser_is_held_when_neural_aim_diverges(self):
        bot = NeuralBot(self.hello, world=FlatWorld(), model=self.model)
        self.obs["self"]["weapons"][0] = {"kind": "laser", "charge": 0.3, "stats": {}}
        with patch("neural.forward", return_value=[0, 0, 1, 0]), \
                patch.object(bot.automatic, "act", return_value={"aim": [0, 0], "fire": [False, False]}):
            self.assertTrue(bot.act(self.obs)["fire"][0])

    def test_firing_uses_precise_target_gates_with_neural_command(self):
        bot = NeuralBot(self.hello, world=FlatWorld(), model=self.model)
        x, y, z = self.obs["self"]["pos"]
        self.obs["self"]["aim"] = [0, 0]
        bot.automatic.current_combat_target = {"id": 1, "kind": "mech", "pos": [x + 100, y, z],
                                               "vel": [0, 0, 0], "cubes": 0}
        action = {"aim": [0, 0], "fire": [True, False]}
        with patch("neural.forward", return_value=[0, 0, 0, 0]), \
                patch.object(bot.automatic, "act", return_value=copy.deepcopy(action)):
            self.assertTrue(bot.act(self.obs)["fire"][0])
        with patch("neural.forward", return_value=[0, 0, 0.04 / math.pi, 0]), \
                patch.object(bot.automatic, "act", return_value=copy.deepcopy(action)):
            self.assertFalse(bot.act(self.obs)["fire"][0])

    def test_population_competes_with_rotated_seats_and_rejects_truncation(self):
        records = []

        class FakeSimulator:
            def __init__(self, strategies, seed, config, world, factory):
                self.strategies = strategies

            def run(self):
                return {"status": "completed", "seconds": 10, "results": [
                    {"slot": i, "strategy": strategy, "survival_seconds": i + 1,
                     "survival_censored": False} for i, strategy in enumerate(self.strategies)]}

        population = [self.model] * 4
        scores = evaluate_population(population, [1], 4, 4, self.config, FlatWorld(),
                                     records.append, FakeSimulator)
        self.assertEqual(scores, [2.5] * 4)
        self.assertEqual(len(records), 4)
        self.assertEqual(len({record["members_by_slot"][0] for record in records}), 4)
        validation = evaluate_validation(self.model, [10], 4, 4, self.config, FlatWorld(),
                                         simulator=FakeSimulator)
        self.assertEqual(validation["candidate_mean_survival_seconds"], 2.5)

        class Truncated(FakeSimulator):
            def run(self):
                return {"status": "truncated", "results": []}

        with self.assertRaisesRegex(RuntimeError, "did not complete"):
            evaluate_population(population, [1], 4, 1, self.config, FlatWorld(), simulator=Truncated)

    def test_reproduction_retains_elite_and_changes_children(self):
        population = [random_model(random.Random(i), 50) for i in range(4)]
        children = reproduce(population, [1, 2, 3, 4], random.Random(3), 1.0, 0.1)
        self.assertIs(children[0], population[3])
        self.assertNotEqual(children[1]["weights"], population[3]["weights"])

    def test_training_cli_and_simulator_accept_saved_model(self):
        self.config["match"]["duration_s"] = 0.1
        with tempfile.TemporaryDirectory() as directory:
            stream = Path(directory) / "config.jsonl.gz"
            model = Path(directory) / "model.json"
            with gzip.open(stream, "wt", encoding="utf-8") as file:
                file.write(json.dumps({"type": "hello", "config": self.config}) + "\n")
            result = subprocess.run(
                [sys.executable, "-m", "simulator.neural_train", "--population", "2", "--group-size", "2",
                 "--rotations", "2", "--generations", "2", "--config-stream", str(stream),
                 "--output", str(model)], cwd=ROOT, capture_output=True, text=True, check=True)
            events = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(sum(event["type"] == "training_match" for event in events), 4)
            self.assertEqual(sum(event["type"] == "validation_match" for event in events), 2)
            self.assertEqual(events[-1]["type"], "summary")
            self.assertEqual(load_model(model)["object_radius"], 200)
            result = subprocess.run(
                [sys.executable, "-m", "simulator", "--strategies", "neural", "vanguard",
                 "--model", str(model), "--seconds", "0.1"], cwd=ROOT, capture_output=True, text=True, check=True)
            self.assertEqual(json.loads(result.stdout.splitlines()[0])["status"], "truncated")

    def test_model_validation_and_disjoint_seeds(self):
        broken = dict(self.model, weights=[float("nan")])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.json"
            path.write_text(json.dumps(broken), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_model(path)
            path.unlink()
            result = subprocess.run(
                [sys.executable, "-m", "simulator.neural_train", "--seed", "42",
                 "--validation-seed", "42", "--output", str(path)], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must not overlap", result.stderr)


if __name__ == "__main__":
    unittest.main()
