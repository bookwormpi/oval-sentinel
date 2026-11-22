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
- Added shooter-relative gun and mortar lead; this assumes projectiles inherit the shooter's velocity, which the protocol does not specify.
- Added bounded 100 ms yaw/pitch slew prediction before firing direct weapons.
- Suppressed direct-fire shots on blocked lines of sight while retaining mortar fire over cover.
- Added modest hysteresis for the current combat target to reduce aim and target-selection churn.
- Added bounded LOS caching and prism-bounds filtering to keep direct-fire LOS checks under the 50 ms deadline.
- Retained the original mortar iteration in the baseline for score comparisons; experimental strategies correct its double-counting of gravity.

## Strategy experiments

Set `SELECTED_STRATEGY` in `bot/strategy.py` before committing and submitting.
The tournament builds your commit itself, so a local environment variable or
Docker build argument alone will not select a strategy for a submission.

| Strategy | Behavior |
| --- | --- |
| `baseline` | Existing Oval Sentinel II controller. |
| `survivor` | Hold position when safe, avoid threats, recover to the nearest safe zone point, and take nearby safe supply drops. |
| `scavenger` | Survivor-style movement plus pursuit of nearby safe cubes, weapons and drops; shoots nearby crates when there is no tracked target or recent damage. |
| `skirmisher` | Collects nearby loot, strafes nearby mechs and maintains distance while fighting; zone recovery takes priority. |
| `vanguard` | Scavenger-based experiment with corrected dodge direction, actual-aim firing gates, safe crate prioritization and short-range obstacle-aware evasion. The default for the next submission. |

The alternatives also correct a mortar-aiming defect that is deliberately
retained in `baseline`, so comparisons with the baseline include this combat
change as well as strategic changes. These are experimental heuristics, not
measured improvements. Practice matches
use cooperating house bots, while final matches are free-for-alls; compare
strategies in the actual competition rather than assuming results transfer.
For local checks only, `OVAL_STRATEGY` overrides the committed selection:

```bash
OVAL_STRATEGY=scavenger python3 check.py --schema mechfight.schema.json --stream stream.jsonl.gz --map-dir arena -- python3 -u bot/main.py
```

For Docker testing, pass `-e OVAL_STRATEGY=scavenger` to `docker run`.
To compare practice scores, submit separate commits with different values of
`SELECTED_STRATEGY`; record each commit and score. Submissions are limited to
four per hour. The recorded stream checks conformance, not placement.

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

The match runtime is 1 CPU, 512 MB RAM, no network, read-only filesystem except `/tmp`, and uid 65534.

## Important limitations

- Reported practice scores: scavenger 1.88, skirmisher 1.21, survivor 1.19.
  Vanguard has not been evaluated in scored matches.
- The conformance checker verifies protocol correctness and decision latency only; it does not simulate responsive opponents.
- Navigation is a local waypoint planner, not globally optimal A*; collision and visibility calculations remain approximations.
- Mortar aiming is approximate and should be tuned from actual match replays;
  the baseline has a known gravity double-counting defect.
- The strategy is deterministic and heuristic; no reinforcement learning or match-outcome optimization has been performed.

The next meaningful evaluation is a scored 48-match run. Compare placement, survival time, damage dealt/received, and cube collection against both the original 0.29 result and the 1.15 starter baseline.
