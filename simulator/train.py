"""Fit an experimental policy with episodic evolutionary policy search."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

from .engine import ROOT, Simulator, load_config, load_world
from adaptive import DEFAULT_PARAMETERS, PARAMETERS, validate_parameters
from agent import MechRoyaleBot


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized(parameters):
    parameters = validate_parameters(parameters)
    return {name: 2 * (parameters[name] - lo) / (hi - lo) - 1
            for name, (_, lo, hi) in PARAMETERS.items()}


def denormalized(values):
    return {name: lo + (max(-1.0, min(1.0, values[name])) + 1) * (hi - lo) / 2
            for name, (_, lo, hi) in PARAMETERS.items()}


def evaluate(parameters, seeds, slots, config, world, report=None, simulator=Simulator):
    """Evaluate mirrored seats; margins are normalized to the local points range."""
    parameters = validate_parameters(parameters)
    if slots < 2 or slots % 2:
        raise ValueError("Evaluation requires an even number of at least two slots")
    if not seeds:
        raise ValueError("At least one evaluation seed is required")
    records = []
    for seed in seeds:
        for candidate_parity in (0, 1):
            strategies = ["adaptive" if slot % 2 == candidate_parity else "vanguard" for slot in range(slots)]

            def factory(hello, strategy, shared_world):
                return MechRoyaleBot(hello, strategy=strategy, world=shared_world,
                                     policy_parameters=parameters if strategy == "adaptive" else None)

            if hasattr(world, "los_cache"):
                world.los_cache.clear()
            result = simulator(strategies, seed, config, world, factory).run()
            if result["status"] != "completed" or len(result["results"]) != slots:
                raise RuntimeError(f"Seed {seed} did not complete with {slots} scored bots")
            groups = {"adaptive": [], "vanguard": []}
            seen = set()
            for entry in result["results"]:
                slot = entry["slot"]
                if (not isinstance(slot, int) or not 0 <= slot < slots or entry["strategy"] != strategies[slot]
                        or slot in seen
                        or not isinstance(entry["score"], (int, float))
                        or not math.isfinite(entry["score"]) or not 0 <= entry["score"] < slots):
                    raise RuntimeError(f"Seed {seed} returned an invalid score or strategy")
                seen.add(slot)
                groups[entry["strategy"]].append(entry["score"] / (slots - 1))
            if any(len(group) != slots // 2 for group in groups.values()):
                raise RuntimeError(f"Seed {seed} returned an unbalanced match")
            candidate = statistics.mean(groups["adaptive"])
            control = statistics.mean(groups["vanguard"])
            record = {"seed": seed, "candidate_parity": candidate_parity,
                      "candidate_score": candidate, "control_score": control,
                      "margin": candidate - control, "seconds": result["seconds"]}
            records.append(record)
            if report:
                report(record)
    return {"margin": statistics.mean(r["margin"] for r in records), "matches": records}


def optimize(parameters, directions, sigma, step_size, evaluate_candidate, rng):
    """One antithetic evolution-strategy update in normalized parameter space."""
    center = normalized(parameters)
    gradients = {name: 0.0 for name in PARAMETERS}
    for _ in range(directions):
        noise = {name: rng.gauss(0.0, 1.0) for name in PARAMETERS}
        plus = denormalized({name: center[name] + sigma * noise[name] for name in PARAMETERS})
        minus = denormalized({name: center[name] - sigma * noise[name] for name in PARAMETERS})
        difference = evaluate_candidate(plus) - evaluate_candidate(minus)
        for name in PARAMETERS:
            gradients[name] += difference * noise[name] / (2 * sigma * directions)
    return validate_parameters(denormalized({name: center[name] + step_size * gradients[name] for name in PARAMETERS}))


def emit(value):
    print(json.dumps(value, separators=(",", ":"), allow_nan=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="new policy JSON file; existing files are not overwritten")
    parser.add_argument("--seed", type=int, default=42, help="first training seed and search RNG seed")
    parser.add_argument("--validation-seed", type=int, default=None, help="first held-out seed (default: seed + 1000000)")
    parser.add_argument("--train-seeds", type=int, default=1)
    parser.add_argument("--validation-seeds", type=int, default=1)
    parser.add_argument("--slots", type=int, default=8)
    parser.add_argument("--generations", type=int, default=1)
    parser.add_argument("--directions", type=int, default=1)
    parser.add_argument("--sigma", type=float, default=0.25)
    parser.add_argument("--step-size", type=float, default=0.2)
    parser.add_argument("--map-dir", default=str(ROOT / "arena"))
    parser.add_argument("--config-stream", default=str(ROOT / "stream.jsonl.gz"))
    args = parser.parse_args()
    config = load_config(args.config_stream)
    if args.slots < 2 or args.slots % 2 or args.slots > int(config["match"]["mechs"]):
        parser.error("--slots must be even and between 2 and the configured mech count")
    if min(args.train_seeds, args.validation_seeds, args.generations, args.directions) < 1:
        parser.error("seed counts, generations, and directions must be positive")
    if not (math.isfinite(args.sigma) and 0 < args.sigma <= 1
            and math.isfinite(args.step_size) and 0 < args.step_size <= 1):
        parser.error("--sigma and --step-size must be finite and in (0, 1]")
    output = Path(args.output)
    if output.exists():
        parser.error(f"output already exists: {output}")
    train_seeds = list(range(args.seed, args.seed + args.train_seeds))
    validation_start = args.validation_seed if args.validation_seed is not None else args.seed + 1000000
    validation_seeds = list(range(validation_start, validation_start + args.validation_seeds))
    if set(train_seeds) & set(validation_seeds):
        parser.error("training and validation seeds must not overlap")
    world = load_world(args.map_dir)
    params = dict(DEFAULT_PARAMETERS)
    rng = random.Random(args.seed)
    for generation in range(args.generations):
        evaluation = 0

        def score(candidate):
            nonlocal evaluation
            current = evaluation
            evaluation += 1
            emit({"type": "candidate", "generation": generation, "evaluation": current, "parameters": candidate})
            return evaluate(candidate, train_seeds, args.slots, config, world,
                            lambda r: emit({"type": "training_match", "generation": generation,
                                            "evaluation": current, **r}))["margin"]

        params = optimize(params, args.directions, args.sigma, args.step_size, score, rng)
        emit({"type": "generation", "generation": generation, "parameters": params})
    candidate = evaluate(params, validation_seeds, args.slots, config, world,
                         lambda r: emit({"type": "validation_match", "policy": "candidate", **r}))
    control = evaluate(DEFAULT_PARAMETERS, validation_seeds, args.slots, config, world,
                       lambda r: emit({"type": "validation_match", "policy": "default", **r}))
    differences = []
    for seed in validation_seeds:
        candidate_margin = statistics.mean(r["margin"] for r in candidate["matches"] if r["seed"] == seed)
        control_margin = statistics.mean(r["margin"] for r in control["matches"] if r["seed"] == seed)
        differences.append(candidate_margin - control_margin)
    summary = {"method": "antithetic_es_v1", "train_seeds": train_seeds, "validation_seeds": validation_seeds,
               "slots": args.slots, "generations": args.generations, "directions": args.directions,
               "sigma": args.sigma, "step_size": args.step_size,
               "source_sha256": {"world": file_hash(Path(args.map_dir) / "v2" / "world.json"),
                                 "terrain": file_hash(Path(args.map_dir) / "v2" / "terrain.i16"),
                                 "config_stream": file_hash(args.config_stream)},
               "candidate_margin": candidate["margin"], "default_margin": control["margin"],
               "paired_difference": statistics.mean(differences),
               "paired_differences": differences,
               "standard_error": statistics.stdev(differences) / math.sqrt(len(differences))
               if len(differences) > 1 else None}
    policy = {"version": 1, "base_strategy": "vanguard", "parameters": params, "training": summary}
    with output.open("x", encoding="utf-8") as destination:
        json.dump(policy, destination, indent=2, allow_nan=False)
        destination.write("\n")
    emit({"type": "summary", "policy_file": str(output), **summary})


if __name__ == "__main__":
    main()
