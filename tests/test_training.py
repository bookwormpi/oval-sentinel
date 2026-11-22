import copy
import gzip
import json
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from simulator.engine import ROOT, load_config
from simulator.train import denormalized, evaluate, normalized, optimize
from adaptive import DEFAULT_PARAMETERS, PARAMETERS, load_policy, validate_parameters
from agent import MechRoyaleBot
import test_policies as policy_fixtures


class TrainingTests(unittest.TestCase):
    def test_adaptive_defaults_match_vanguard_action(self):
        fixture = policy_fixtures.PolicyTests()
        fixture.setUp()
        obs = fixture.obs
        obs.update({"tick": 2, "sounds": [], "events": [], "projectiles": []})
        obs["self"].update({"vel": [0, 0, 0], "aim": [0, -0.173245],
                            "hp": 1400, "max_hp": 1400, "energy": 100, "grounded": True,
                            "weapons": [{"kind": "gun", "ammo": 12, "cooldown": 0, "reload": 0,
                                         "stats": {"speed": 252, "range": 230}}, None]})
        obs["crates"] = [{"id": 1000, "pos": [30, 0, 0]}]
        hello = {"config": {"mech": {"min_pitch": -0.7, "max_pitch": 1.3}}}
        vanguard = MechRoyaleBot(hello, strategy="vanguard", world=policy_fixtures.FlatWorld())
        adaptive = MechRoyaleBot(hello, strategy="adaptive", world=policy_fixtures.FlatWorld(),
                                policy_parameters=DEFAULT_PARAMETERS)
        self.assertEqual(vanguard.act(copy.deepcopy(obs)), adaptive.act(copy.deepcopy(obs)))

    def test_parameters_change_crate_and_loot_decisions(self):
        fixture = policy_fixtures.PolicyTests()
        fixture.setUp()
        from policies import choose_variant_goal, vanguard_target
        obs = fixture.obs
        obs["cubes"] = [{"id": 5000, "pos": [60, 0, 0]}]
        params = dict(DEFAULT_PARAMETERS, cube_seek_distance=45.0)
        self.assertEqual(choose_variant_goal(obs, fixture.world, "scavenger", params)[1], "hold")
        obs["crates"] = [{"id": 1000, "pos": [50, 0, 0]}]
        params["crate_attack_distance"] = 40.0
        self.assertIsNone(vanguard_target(obs, None, -100, params))
        params["crate_attack_distance"] = 70.0
        self.assertEqual(vanguard_target(obs, None, -100, params)["id"], 1000)

    def test_policy_file_is_strictly_validated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            policy = {"version": 1, "base_strategy": "vanguard", "parameters": DEFAULT_PARAMETERS}
            path.write_text(json.dumps(policy), encoding="utf-8")
            self.assertEqual(load_policy(path), DEFAULT_PARAMETERS)
            policy["parameters"] = dict(DEFAULT_PARAMETERS, cube_seek_distance=float("nan"))
            path.write_text(json.dumps(policy), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_policy(path)
        with self.assertRaises(ValueError):
            validate_parameters(dict(DEFAULT_PARAMETERS, extra=1))

    def test_simulator_can_use_external_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            path.write_text(json.dumps({"version": 1, "base_strategy": "vanguard",
                                        "parameters": DEFAULT_PARAMETERS}), encoding="utf-8")
            result = subprocess.run(
                [sys.executable, "-m", "simulator", "--strategies", "adaptive", "vanguard",
                 "--policy", str(path), "--seconds", "0.1"],
                cwd=ROOT, capture_output=True, text=True, check=True)
            match, summary = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual([r["strategy"] for r in match["results"]], ["adaptive", "vanguard"])
            self.assertEqual(match["status"], "truncated")
            self.assertIsNone(summary["strategies"]["adaptive"]["mean_local_score"])

    def test_missing_submission_policy_fails_before_starting_bot(self):
        import main
        with patch("main.selected_strategy", return_value="adaptive"), \
                patch("main.load_policy", side_effect=FileNotFoundError("missing policy")), \
                patch("main.run") as run:
            with self.assertRaises(FileNotFoundError):
                main.main()
            run.assert_not_called()

    def test_mirrored_evaluation_and_normalized_rewards(self):
        calls = []

        class FakeSimulator:
            def __init__(self, strategies, seed, config, world, factory):
                self.strategies, self.seed = strategies, seed
                calls.append((seed, strategies[:]))

            def run(self):
                return {"status": "completed", "seconds": 1, "results": [
                    {"slot": i, "strategy": strategy, "score": 3 if strategy == "adaptive" else 0}
                    for i, strategy in enumerate(self.strategies)]}

        result = evaluate(DEFAULT_PARAMETERS, [1, 2], 4, load_config(), policy_fixtures.FlatWorld(), simulator=FakeSimulator)
        self.assertEqual(result["margin"], 1.0)
        self.assertEqual(len(result["matches"]), 4)
        self.assertEqual(calls[0], (1, ["adaptive", "vanguard", "adaptive", "vanguard"]))
        self.assertEqual(calls[1], (1, ["vanguard", "adaptive", "vanguard", "adaptive"]))
        self.assertEqual(calls[2][0], 2)

    def test_incomplete_matches_are_rejected(self):
        class FakeSimulator:
            def __init__(self, *args):
                pass

            def run(self):
                return {"status": "truncated", "results": []}

        with self.assertRaisesRegex(RuntimeError, "did not complete"):
            evaluate(DEFAULT_PARAMETERS, [1], 4, load_config(), policy_fixtures.FlatWorld(), simulator=FakeSimulator)

    def test_training_rejects_overlapping_validation_seeds(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.json"
            result = subprocess.run(
                [sys.executable, "-m", "simulator.train", "--seed", "42",
                 "--validation-seed", "42", "--output", str(path)],
                cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("must not overlap", result.stderr)
            self.assertFalse(path.exists())

    def test_training_cli_writes_valid_policy_after_completed_matches(self):
        config = load_config()
        config["match"]["duration_s"] = 0.1
        config["crates"]["count"] = config["npc"]["count"] = config["loot"]["floor_count"] = 0
        with tempfile.TemporaryDirectory() as directory:
            stream = Path(directory) / "config.jsonl.gz"
            policy = Path(directory) / "policy.json"
            with gzip.open(stream, "wt", encoding="utf-8") as file:
                file.write(json.dumps({"type": "hello", "config": config}) + "\n")
            result = subprocess.run(
                [sys.executable, "-m", "simulator.train", "--slots", "2",
                 "--config-stream", str(stream), "--output", str(policy)],
                cwd=ROOT, capture_output=True, text=True, check=True)
            events = [json.loads(line) for line in result.stdout.splitlines()]
            self.assertEqual(len([e for e in events if e["type"].endswith("_match")]), 8)
            self.assertEqual(events[-1]["type"], "summary")
            self.assertEqual(len(events[-1]["source_sha256"]["config_stream"]), 64)
            self.assertEqual(set(load_policy(policy)), set(DEFAULT_PARAMETERS))

    def test_antithetic_update_improves_synthetic_objective(self):
        score = lambda p: p["cube_seek_distance"] / 150.0
        result = optimize(DEFAULT_PARAMETERS, 10, 0.1, 0.2, score, random.Random(12))
        self.assertGreater(result["cube_seek_distance"], DEFAULT_PARAMETERS["cube_seek_distance"])
        self.assertEqual(validate_parameters(result), result)
        for key in PARAMETERS:
            self.assertAlmostEqual(denormalized(normalized(DEFAULT_PARAMETERS))[key], DEFAULT_PARAMETERS[key])


if __name__ == "__main__":
    unittest.main()
