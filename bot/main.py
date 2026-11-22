#!/usr/bin/env python3
"""Entry point for the Mech Royale heuristic bot."""
from mechfight import run
from agent import MechRoyaleBot
from strategy import selected_strategy

if __name__ == "__main__":
    selected_strategy()
    run(MechRoyaleBot)
