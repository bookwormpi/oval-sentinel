"""Train neural movement and aim by evolutionary self-play."""
import argparse
import json
import math
import os
from pathlib import Path
import random
import signal
import statistics
import tempfile

from .engine import ROOT, Simulator, load_config, load_world
from .train import emit, file_hash
from agent import MechRoyaleBot
from neural import NeuralBot, load_model, random_model, validate_model


def _match(strategies, models, seed, config, world, simulator=Simulator):
    def factory(hello, strategy, shared_world):
        if strategy == "neural":
            return NeuralBot(hello, world=shared_world, model=models[hello["slot"]])
        return MechRoyaleBot(hello, strategy="vanguard", world=shared_world)

    if hasattr(world, "los_cache"):
        world.los_cache.clear()
    result = simulator(strategies, seed, config, world, factory).run()
    if result["status"] != "completed" or len(result["results"]) != len(strategies):
        raise RuntimeError(f"Seed {seed} did not complete with all bots scored")
    seen = set()
    times = [None] * len(strategies)
    for entry in result["results"]:
        slot = entry["slot"]
        survival = entry["survival_seconds"]
        if (not isinstance(slot, int) or not 0 <= slot < len(strategies) or slot in seen
                or entry["strategy"] != strategies[slot] or entry["survival_censored"]
                or not isinstance(survival, (int, float)) or not math.isfinite(survival)
                or not 0 <= survival <= result["seconds"] + 0.001):
            raise RuntimeError(f"Seed {seed} returned invalid or censored survival data")
        seen.add(slot)
        times[slot] = survival
    return times, result["seconds"]


def evaluate_population(population, seeds, group_size, rotations, config, world, report=None, simulator=Simulator):
    if not seeds or not 2 <= group_size <= int(config["match"]["mechs"]):
        raise ValueError("Self-play needs seeds and a valid group size")
    if len(population) < group_size or len(population) % group_size:
        raise ValueError("Population size must be a positive multiple of group size")
    if not 1 <= rotations <= group_size:
        raise ValueError("Rotations must be between 1 and group size")
    scores = [[] for _ in population]
    for seed in seeds:
        order = list(range(len(population)))
        random.Random(seed).shuffle(order)
        for group_start in range(0, len(order), group_size):
            group = order[group_start:group_start + group_size]
            for rotation in range(rotations):
                seats = group[rotation:] + group[:rotation]
                times, seconds = _match(["neural"] * group_size, [population[i] for i in seats],
                                        seed, config, world, simulator)
                for slot, member in enumerate(seats):
                    scores[member].append(times[slot])
                if report:
                    report({"seed": seed, "group": group_start // group_size,
                            "rotation": rotation, "members_by_slot": seats,
                            "survival_seconds": times, "match_seconds": seconds})
    return [statistics.mean(values) for values in scores]


def evaluate_validation(model, seeds, group_size, rotations, config, world, report=None, simulator=Simulator):
    if not seeds or not 2 <= group_size <= int(config["match"]["mechs"]) or not 1 <= rotations <= group_size:
        raise ValueError("Validation needs seeds, a valid group size and seat rotations")
    candidate, controls = [], []
    for seed in seeds:
        for slot in range(rotations):
            strategies = ["vanguard"] * group_size
            strategies[slot] = "neural"
            models = [None] * group_size
            models[slot] = model
            times, seconds = _match(strategies, models, seed, config, world, simulator)
            candidate.append(times[slot])
            controls.append(statistics.mean(value for i, value in enumerate(times) if i != slot))
            if report:
                report({"seed": seed, "candidate_slot": slot, "survival_seconds": times,
                        "match_seconds": seconds})
    return {"candidate_mean_survival_seconds": statistics.mean(candidate),
            "opponents_mean_survival_seconds": statistics.mean(controls),
            "matches": len(candidate)}


def reproduce(population, fitness, rng, mutation_rate, mutation_sigma):
    if len(population) != len(fitness) or len(population) < 2:
        raise ValueError("Fitness must cover at least two individuals")
    ranking = sorted(range(len(population)), key=lambda i: (-fitness[i], i))
    parents = ranking[:max(2, len(ranking) // 2)]
    elite = population[ranking[0]]
    offspring = [elite]
    while len(offspring) < len(population):
        first, second = rng.sample(parents, 2)
        child = []
        for a, b in zip(population[first]["weights"], population[second]["weights"]):
            value = a if rng.random() < 0.5 else b
            if rng.random() < mutation_rate:
                value += rng.gauss(0, mutation_sigma)
            child.append(max(-50.0, min(50.0, value)))
        offspring.append(validate_model({"version": elite["version"], "architecture": elite["architecture"],
                                         "object_radius": elite["object_radius"], "weights": child}))
    return offspring


def save_model(output, model, overwrite):
    """Write a complete model before making it visible at the output path."""
    mode = output.stat().st_mode & 0o777 if output.exists() else 0o644
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=f".{output.name}.", delete=False) as destination:
            temp_path = Path(destination.name)
            json.dump(model, destination, separators=(",", ":"), allow_nan=False)
            destination.write("\n")
            destination.flush()
            os.fsync(destination.fileno())
        os.chmod(temp_path, mode)
        if overwrite:
            os.replace(temp_path, output)
        else:
            os.link(temp_path, output)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, help="model JSON output path")
    parser.add_argument("--resume", help="initialize the population from this existing model")
    parser.add_argument("--overwrite", action="store_true", help="allow replacing an existing output model")
    parser.add_argument("--select-best", action="store_true",
                        help="select generation winners on the fixed validation seeds")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-seed", type=int)
    parser.add_argument("--population", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=4)
    parser.add_argument("--rotations", type=int, default=2)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--train-seeds", type=int, default=1, help="fresh seeds per generation")
    parser.add_argument("--validation-seeds", type=int, default=1)
    parser.add_argument("--object-radius", type=float, default=200.0)
    parser.add_argument("--mutation-rate", type=float, default=0.1)
    parser.add_argument("--mutation-sigma", type=float, default=0.05)
    parser.add_argument("--map-dir", default=str(ROOT / "arena"))
    parser.add_argument("--config-stream", default=str(ROOT / "stream.jsonl.gz"))
    args = parser.parse_args()
    config = load_config(args.config_stream)
    if (args.group_size < 2 or args.group_size > int(config["match"]["mechs"])
            or args.population < args.group_size or args.population % args.group_size):
        parser.error("population must be a multiple of group size, between 2 and the configured mech count")
    if not 1 <= args.rotations <= args.group_size:
        parser.error("--rotations must be between 1 and group size")
    if min(args.generations, args.train_seeds, args.validation_seeds) < 1:
        parser.error("generations and seed counts must be positive")
    if not math.isfinite(args.object_radius) or not 0 < args.object_radius <= 5000:
        parser.error("--object-radius must be finite and in (0, 5000]")
    if not math.isfinite(args.mutation_rate) or not 0 <= args.mutation_rate <= 1:
        parser.error("--mutation-rate must be finite and in [0, 1]")
    if not math.isfinite(args.mutation_sigma) or not 0 < args.mutation_sigma <= 10:
        parser.error("--mutation-sigma must be finite and in (0, 10]")
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        parser.error(f"output already exists: {output}")
    training_seeds = list(range(args.seed, args.seed + args.generations * args.train_seeds))
    validation_start = args.validation_seed if args.validation_seed is not None else args.seed + 1000000
    validation_seeds = list(range(validation_start, validation_start + args.validation_seeds))
    if set(training_seeds) & set(validation_seeds):
        parser.error("training and validation seeds must not overlap")
    world = load_world(args.map_dir)
    rng = random.Random(args.seed)
    initial = load_model(args.resume) if args.resume else None
    if initial is not None:
        population = reproduce([initial] * args.population, [0.0] * args.population,
                               rng, args.mutation_rate, args.mutation_sigma)
    else:
        population = [random_model(rng, args.object_radius) for _ in range(args.population)]
    winner = initial if initial is not None else population[0]
    validation = None
    best_generation = None
    history = []
    interrupted = False
    # Detached shells can inherit SIGINT as ignored; make kill -INT equivalent to Ctrl-C.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    try:
        if args.select_best and initial is not None:
            validation = evaluate_validation(
                winner, validation_seeds, args.group_size, args.rotations, config, world,
                lambda record: emit({"type": "validation_match", "generation": None, **record}))
            emit({"type": "selection", "generation": None, **validation})
        for generation in range(args.generations):
            seeds = training_seeds[generation * args.train_seeds:(generation + 1) * args.train_seeds]
            fitness = evaluate_population(
                population, seeds, args.group_size, args.rotations, config, world,
                lambda record: emit({"type": "training_match", "generation": generation, **record}))
            history.append({"generation": generation, "seeds": seeds, "fitness_seconds": fitness})
            candidate = population[max(range(len(population)), key=lambda i: fitness[i])]
            emit({"type": "generation", **history[-1]})
            if args.select_best:
                candidate_validation = evaluate_validation(
                    candidate, validation_seeds, args.group_size, args.rotations, config, world,
                    lambda record: emit({"type": "validation_match", "generation": generation, **record}))
                if (validation is None or candidate_validation["candidate_mean_survival_seconds"]
                        > validation["candidate_mean_survival_seconds"]):
                    winner, validation, best_generation = candidate, candidate_validation, generation
                emit({"type": "selection", "generation": generation, "selected_generation": best_generation,
                      **candidate_validation})
            else:
                winner, best_generation = candidate, generation
            if generation + 1 < args.generations:
                population = reproduce(population, fitness, rng, args.mutation_rate, args.mutation_sigma)
        if not args.select_best:
            validation = evaluate_validation(
                winner, validation_seeds, args.group_size, args.rotations, config, world,
                lambda record: emit({"type": "validation_match", **record}))
    except KeyboardInterrupt:
        interrupted = True
    summary = {"method": "population_elitism_uniform_crossover_gaussian_mutation_v1",
               "seed": args.seed, "training_seeds": training_seeds, "validation_seeds": validation_seeds,
               "population": args.population, "group_size": args.group_size, "rotations": args.rotations,
               "generations": args.generations, "mutation_rate": args.mutation_rate,
               "mutation_sigma": args.mutation_sigma, "history": history, "validation": validation,
               "interrupted": interrupted, "selection": "validation" if args.select_best else "last_generation",
               "best_generation": best_generation, "resumed_from": args.resume,
               "source_sha256": {"world": file_hash(Path(args.map_dir) / "v2" / "world.json"),
                                 "terrain": file_hash(Path(args.map_dir) / "v2" / "terrain.i16"),
                                 "config_stream": file_hash(args.config_stream)}}
    save_model(output, {**winner, "training": summary}, args.overwrite)
    emit({"type": "summary", "model_file": str(output), **summary})


if __name__ == "__main__":
    main()
