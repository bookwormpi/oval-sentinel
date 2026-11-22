#!/usr/bin/env python3
"""Entry point for the Mech Royale heuristic bot."""
from mechfight import run
from agent import MechRoyaleBot
from strategy import selected_strategy
from adaptive import load_policy


def main():
    if selected_strategy() == "adaptive":
        load_policy()
    run(MechRoyaleBot)


if __name__ == "__main__":
    main()
