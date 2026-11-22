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
| `adaptive` | Opt-in Vanguard-derived policy whose loot, crate and zone thresholds come from `bot/adaptive_policy.json`. This file must exist and be included in the submission. |
| `neural` | Experimental neural movement and aim, with automatic Vanguard-derived actions. Requires `bot/neural_policy.json`. |

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
- `bot/neural.py`: experimental observation encoder and neural controller.

## Local validation

From this directory:

```bash
python3 -m unittest discover -s tests -v
python3 check.py --schema mechfight.schema.json --stream stream.jsonl.gz --map-dir arena -- python3 -u bot/main.py
```

The included recorded stream is a protocol/conformance fixture. Its observations do not react to bot actions, so it cannot measure placement or combat quality.

## Approximate self-play

The offline simulator runs the actual bot strategies against one another on the
included map. From this directory:

```bash
# Alternate eight bots of each strategy, across four seeded matches.
python3 -m simulator --strategies scavenger vanguard --slots 16 --matches 4 --seed 42 > selfplay.jsonl

# Short smoke test; the survivors remain unranked.
python3 -m simulator --strategies scavenger vanguard --seconds 20
```

`--slots` cycles through the specified strategies (default: one bot per
strategy); their order rotates between matches. Seeds increase by one per match.
Use `--map-dir` to choose another arena containing `v2/world.json` and
`v2/terrain.i16`, and `--config-stream` to choose a gzipped recorded stream
containing a hello/config message. Output is one JSON object per match followed
by a summary. Local placement points are `number of bots - place`; equal-tick
eliminations share a place, and timeout survivors are ranked by remaining HP.
These are not official scores. Truncated matches are excluded from the summary,
and their surviving bots have null placement, score and `survival_seconds`.
`observed_survival_seconds` gives the time observed, and `survival_censored`
identifies survivors whose match was truncated. Winners' survival is measured
until the match ends; timeout survivors are measured until the timeout.

**This is not the tournament engine or a training algorithm.** The simulator
approximates movement and aiming, all four weapons and their signature effects,
NPC drones, crates, cubes, upgrades, weapon pickups, supply drops, vision,
sounds, terrain, the zone and placement. Per-bot results include collected
`cubes` and chosen `upgrades`.

The official engine is unavailable. Important remaining differences include
sampled collisions and terrain, approximate rooftop/ledge and overhang physics,
ellipsoidal tree canopies and attenuation, NPC AI and spawn locations, projectile
trajectories and impacts, and zone/drop randomization. Projectiles do not inherit
shooter velocity; the protocol does not specify whether they should. Bots run
in-process, so timeouts, crashes, transport limits and official cooperative
house-bot behavior are not simulated. Practice matches use 15 cooperating
opponents rather than this free-for-all. Results are useful for debugging and
experiments, not predicting the leaderboard; calibrate against real match
observations before optimizing against them.

## Experimental policy search

The trainer uses episodic **evolutionary policy search**, not a neural network:
it perturbs six bounded Vanguard thresholds and updates them using completed
match placement rewards. Opponents remain the fixed Vanguard policy. Each
candidate plays both alternating seat assignments for every training seed.
Validation uses separate seeds and repeats those mirrored matches for both the
candidate and unmodified Vanguard parameters. Its `paired_difference` is the
candidate's mean normalized placement margin minus the default's margin; one
point is one entire local placement-score range, not an official point.
Identical seeds reproduce initial conditions, but later randomness can diverge
as actions differ. Two or three validation seeds are not enough to establish a
reliable improvement.

```bash
# Requires a new output path; the trainer never overwrites an existing policy.
python3 -m simulator.train --seed 42 --train-seeds 2 --validation-seeds 2 \
  --slots 8 --generations 1 --directions 1 --output /tmp/adaptive-policy.json

# Test the resulting policy against Vanguard on other seeds.
python3 -m simulator --strategies adaptive vanguard --policy /tmp/adaptive-policy.json \
  --slots 8 --matches 4 --seed 2000042
```

Training emits JSON lines with the sampled policies, each match and a summary;
the output policy also contains its training and validation metadata. Only full
matches count. Training costs `4 * train_seeds * generations * directions +
4 * validation_seeds` full matches. The policy is an experiment, not an
automatically promoted improvement. To submit one, put the chosen JSON at
`bot/adaptive_policy.json`, set `SELECTED_STRATEGY = "adaptive"`, run the
conformance check and commit both files. Other strategies remain unchanged.

The included first candidate was trained on seeds 42 and 43 and validated on
1000042 and 1000043 with eight bots. Its mean paired validation difference was
approximately zero; this small experiment does not show an improvement.

## Experimental neural self-play

The opt-in neural controller uses a 16-unit hidden-layer network to choose
facing-relative forward/left movement and yaw/pitch offsets. Movement is
rotated by the **observed** torso yaw into the protocol's world frame; the
engine, not the network, slews the torso toward the commanded aim. Vanguard
still chooses firing, reload, boost, jump, pickups and upgrades. Firing uses
Vanguard's target and firing gates with the neural aim; a laser
already charging is held instead of deliberately released off-target, although
the game can force a release after its maximum hold time.

The encoder includes normalized scalar summaries of self state, all observation
fields, hello fields other than the map descriptor, and nearby distances. Variable-length data is
compressed into fixed-size signed hash buckets: this is **lossy**, collisions
and array ordering matter, and individual inputs cannot always be recovered.
For each visible entity category and each static map category (prisms, trees,
cars, barriers), it includes the four nearest distances, a nearby count, and
hashes the distances of all objects within `--object-radius` (meters; default
200). Map-object distances are approximate distances to 3-D bounding boxes,
not exact surfaces; terrain height samples and the full map are not network
inputs. The automatic Vanguard controller still uses map geometry. Observation
fields, including positions of reported objects outside the distance radius,
are also encoded; the radius limits the **derived distance features**, not the
protocol observations.

Individuals compete in groups, earn mean survival seconds from completed
matches, and evolve using elitism, uniform crossover and Gaussian mutation.
Seats rotate; use `--rotations` equal to `--group-size` for every individual to
occupy every seat. Each generation uses fresh training seeds. The final
generation's winner is evaluated against Vanguard on separate validation seeds.
It is **not automatically promoted**; the small initial run is only a smoke
test. Within a group, all survivors' clocks stop when the match ends; fitness
therefore also depends on the opponents and can favor groups that avoid combat.
The same initial seed does not guarantee identical later random events when
actions differ.

```bash
# New output path required. The number of full matches is
# generations * train_seeds * (population / group_size) * rotations
# plus validation_seeds * rotations.
python3 -m simulator.neural_train --population 4 --group-size 4 \
  --rotations 4 --generations 2 --train-seeds 2 --validation-seeds 2 \
  --object-radius 200 --output /tmp/neural-policy.json

python3 -m simulator --strategies neural vanguard --model /tmp/neural-policy.json \
  --slots 4 --matches 4 --seed 2000042
```

Training emits per-match survival times and stores its settings, map/config
hashes and held-out results in the model. To test a model in the tournament
container, place it at `bot/neural_policy.json`, set `SELECTED_STRATEGY` to
`neural`, and run the restricted conformance check before submitting. Do not
assume simulator survival predicts leaderboard performance.

The included initial model is a two-generation, four-individual smoke test.
On one held-out seed with two seat rotations, it averaged 46.7 seconds of
survival versus 267.8 seconds for its Vanguard opponents. This is too little
evaluation for a reliable comparison, but it does not justify replacing the
default.

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
- The default strategy is deterministic and heuristic; simulator policy search
  does not establish an improvement against the tournament engine.

The next meaningful evaluation is a scored 48-match run. Compare placement, survival time, damage dealt/received, and cube collection against both the original 0.29 result and the 1.15 starter baseline.
