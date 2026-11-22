# Oval Sentinel II — Mech Royale Bot

A modular, standard-library-only Python bot for the official `mechfight/1` protocol. This is a targeted revision of the first Oval Sentinel submission, with emphasis on correcting combat control and improving survival behavior.

## What changed from the first submission

- Fixed secondary-weapon aim selection: the bot now selects its commanded aim from the most useful available weapon instead of always using slot 0.
- Fixed projectile-dodge prediction: the first version used the wrong sign for closest-approach time, so incoming projectiles were generally not identified as incoming. The revised predictor uses relative 3-D geometry and ignores shots already moving away.
- Improved laser trigger handling: once charging begins, it continues tracking instead of prematurely releasing whenever the torso aim briefly lags.
- Improved target selection: enemies aiming at the bot, charging lasers, or recently damaging it receive higher priority.
- Added evasive movement from nearby attackers and recent weapon sounds, while preserving a stronger return-to-zone vector.
- Replaced long cross-map default movement with quiet, low-speed patrols inside the safe zone. This is intended to reduce exposure and avoid generating step sounds.
- Fixed upgrade stacking: the first version treated any owned upgrade as unavailable, even when the official definition allowed multiple stacks. The new selector counts stacks and respects each upgrade's configured maximum.
- Reworked upgrade values around the actual sample-kit upgrade IDs (`plating`, `armor`, `nanites`, `reactor`, `capacitor`, etc.).
- Bounded local obstacle-route candidate testing. The earlier all-vertices search could exceed the 50 ms decision budget on some map positions.
- Added shooter-relative gun and mortar lead so interception accounts for the bot's own velocity.
- Added bounded 100 ms yaw/pitch slew prediction before firing direct weapons.
- Suppressed direct-fire shots on blocked lines of sight while retaining mortar fire over cover.
- Added modest hysteresis for the current combat target to reduce aim and target-selection churn.

## Architecture

- `bot/mechfight.py`: official kit transport wrapper (unchanged).
- `bot/agent.py`: survival-first controller, threat estimation, evasive movement, scanning, and action integration.
- `bot/world.py`: loads official static campus geometry and terrain; bounded local obstacle waypoint search.
- `bot/navigation.py`: interpolated safe-zone tracking, conservative resource goals, and quiet patrol behavior.
- `bot/combat.py`: threat-weighted targeting, gun/laser/lightning/mortar aiming, firing gates, and 3-D projectile dodge.
- `bot/strategy.py`: stack-aware upgrade valuation, weapon pickup, and conservative boost/jump policy.
- `bot/geometry.py`: numerical primitives.

## Local validation

From this directory:

```bash
python3 -m unittest discover -s tests -v
python3 check.py --schema mechfight.schema.json --stream stream.jsonl.gz --map-dir arena -- python3 -u bot/main.py
```

The included recorded stream is a protocol/conformance fixture. Its observations do not react to bot actions, so it cannot measure placement or combat quality.

## Docker validation

Build from this directory:

```bash
docker build -t oval-sentinel-ii .
python3 check.py --schema mechfight.schema.json --stream stream.jsonl.gz --map-dir /arena -- docker run --rm -i --network none --cpus 1 --memory 512m --read-only --tmpfs /tmp:rw,size=64m --user 65534:65534 -v "$PWD/arena:/arena:ro" oval-sentinel-ii
```

The match runtime is 1 CPU, 512 MB RAM, no network, read-only filesystem except `/tmp`, and uid 65534. Docker was not installed in the environment used to prepare this revision, so the image build/runtime command has not been verified here.

## Important limitations

- This revision has **not** been run through the 48 scored practice matches. No new score or improvement over 1.15 is claimed.
- The conformance checker verifies protocol correctness and decision latency only; it does not simulate responsive opponents.
- Navigation is a local waypoint planner, not globally optimal A*; collision and visibility calculations remain approximations.
- Mortar aiming is approximate and should be tuned from actual match replays.
- The strategy is deterministic and heuristic; no reinforcement learning or match-outcome optimization has been performed.

The next meaningful evaluation is a scored 48-match run. Compare placement, survival time, damage dealt/received, and cube collection against both the original 0.29 result and the 1.15 starter baseline.
