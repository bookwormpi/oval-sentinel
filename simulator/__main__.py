"""Run approximate self-play matches and emit JSON lines."""
import argparse
import json
import math

from .engine import ROOT, Simulator, load_config, load_world
from adaptive import load_policy
from agent import MechRoyaleBot
from strategy import STRATEGIES


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES,
                        default=["scavenger", "vanguard"])
    parser.add_argument("--slots", type=int, help="number of bots; cycles through the strategies")
    parser.add_argument("--matches", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--seconds", type=float, help="truncate each match after this many seconds")
    parser.add_argument("--map-dir", default=str(ROOT / "arena"))
    parser.add_argument("--config-stream", default=str(ROOT / "stream.jsonl.gz"))
    parser.add_argument("--policy", help="policy JSON for adaptive bots (default: bot/adaptive_policy.json)")
    args = parser.parse_args()
    config = load_config(args.config_stream)
    slots = args.slots if args.slots is not None else len(args.strategies)
    if not 2 <= slots <= int(config["match"]["mechs"]):
        parser.error("--slots must be between 2 and the configured mech count")
    if args.matches < 1:
        parser.error("--matches must be positive")
    if args.seconds is not None and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("--seconds must be finite and positive")
    if args.policy and "adaptive" not in args.strategies:
        parser.error("--policy requires the adaptive strategy")

    world = load_world(args.map_dir)
    parameters = load_policy(args.policy) if args.policy else None

    def factory(hello, strategy, shared_world):
        return MechRoyaleBot(hello, strategy=strategy, world=shared_world,
                             policy_parameters=parameters if strategy == "adaptive" else None)

    totals = {name: {"matches": 0, "points": 0, "wins": 0} for name in args.strategies}
    for index in range(args.matches):
        strategies = [args.strategies[(slot + index) % len(args.strategies)] for slot in range(slots)]
        result = Simulator(strategies, args.seed + index, config, world, factory).run(args.seconds)
        print(json.dumps({"type": "match", **result}, separators=(",", ":")), flush=True)
        if result["status"] == "completed":
            for entry in result["results"]:
                totals[entry["strategy"]]["matches"] += 1
                totals[entry["strategy"]]["points"] += entry["score"]
                totals[entry["strategy"]]["wins"] += int(entry["place"] == 1)
    summary = {}
    for name, total in totals.items():
        count = total["matches"]
        summary[name] = {"completed_entries": count, "wins": total["wins"],
                         "mean_local_score": total["points"] / count if count else None}
    print(json.dumps({"type": "summary", "strategies": summary}, separators=(",", ":")), flush=True)


if __name__ == "__main__":
    main()
