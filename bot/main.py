#!/usr/bin/env python3
"""Entry point for the Mech Royale heuristic bot."""
from mechfight import run
from agent import MechRoyaleBot
from strategy import selected_strategy
from adaptive import load_policy


def main():
    strategy = selected_strategy()
    if strategy == "adaptive":
        load_policy()
    if strategy == "neural":
        from neural import NeuralBot, load_model
        load_model()
        run(NeuralBot)
    else:
        run(MechRoyaleBot)


if __name__ == "__main__":
    main()
