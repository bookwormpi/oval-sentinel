# Oval Sentinel — Mech Royale Bot

A modular, standard-library-only Python bot for the official `mechfight/1` protocol.

## Architecture

- `bot/mechfight.py`: official kit transport wrapper (unchanged).
- `bot/agent.py`: hierarchical controller and action integration.
- `bot/world.py`: loads official static campus geometry and terrain.
- `bot/navigation.py`: zone/objective selection and local obstacle-aware steering.
- `bot/combat.py`: target scoring, projectile lead, ballistic mortar solution, weapon firing gates, projectile dodge.
- `bot/strategy.py`: upgrade valuation, weapon pickup, boost/jump policy.
- `bot/geometry.py`: numerical primitives.
- `check.py`, `stream.jsonl`, `protocol.md`, `mechfight.schema.json`, `arena/`: official kit assets.

## Local conformance

From this directory:

```bash
python3 check.py --schema mechfight.schema.json --stream stream.jsonl.gz --map-dir arena -- python3 -u bot/main.py
```

If the checker expects the original directory layout, run from the original kit root:

```bash
python3 check/check.py -- python3 -u /path/to/oval-sentinel/bot/main.py
```

## Docker conformance

Build from this directory:

```bash
docker build -t oval-sentinel .
python3 check.py --schema mechfight.schema.json --stream stream.jsonl.gz --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/arena:/arena:ro" oval-sentinel
```

The match runtime is 1 CPU, 512 MB RAM, no network, read-only filesystem except `/tmp`, and uid 65534.

## Strategy

The bot optimizes survival and placement, not kill count:

1. Return inside the current safe zone when needed.
2. Pursue nearby cubes and uncontested late-game supply drops.
3. Fill the empty second weapon slot while retaining the starting gun.
4. Prefer defensive upgrades and then weapon-specific improvements.
5. Select a visible target, lead projectiles, respect weapon range/cooldown, and avoid mortar self-splash.
6. Dodge visible projectiles and preserve energy for emergencies.

The map planner loads `arena/v2/world.json` and `terrain.i16` when available. It uses prism geometry to choose a waypoint around a directly blocking building. This is a pragmatic local planner, not a globally optimal campus-wide A* solver.

## Limitations to test in actual matches

- `check.py` validates protocol and latency only; its observations do not respond to actions.
- Map collision and visibility are approximations; the authoritative server remains the source of truth.
- Mortar aiming is approximate and should be tuned using match replays.
- No learned policy is included; this keeps startup, memory, and decision latency low.
- Use the 48-match leaderboard to tune the scoring weights and compare variants.

## Development

Run the unit tests:

```bash
python3 -m unittest discover -s tests -v
```
