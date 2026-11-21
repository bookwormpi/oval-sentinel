# Oval Sentinel II — change log

## Verified defects corrected

1. **Secondary weapon aiming:** the previous controller set the desired aim from slot 0 first and never replaced it with slot 1's solution. This prevented reliable use of secondary weapons. The revised controller scores weapon opportunities and chooses the appropriate aim solution.
2. **Projectile dodge sign error:** with `r = mech_position - projectile_position` and `v = projectile_velocity - mech_velocity`, closest approach is at `t = dot(r, v) / dot(v, v)`. The previous implementation used the negative of this expression, which selected time zero for a projectile approaching from ahead and failed to dodge it. The revised implementation uses the correct sign and ignores projectiles moving away.
3. **Upgrade stacking:** the previous selector rejected an upgrade as soon as its ID appeared in `self.upgrades`, even though the kit defines multi-stack maximums. The revised selector counts owned stacks and respects the configured cap.
4. **Laser charge tracking:** the previous firing gate could release a partially charged laser when a moving target caused a brief alignment error. The revised controller keeps charging while tracking a valid in-range target and releases at full charge.
5. **Navigation latency:** the previous route planner tested every polygon vertex and repeatedly ran map-wide collision checks. The revised planner considers only the six vertices nearest the direct route and tests two offsets per vertex.

## Strategic changes

- Targets aiming at the bot, charging a laser, or recently damaging the bot are prioritized.
- Nearby enemies and recent weapon sounds contribute to an evasive movement vector.
- When safely inside the zone and not pursuing a nearby resource, the bot uses low-magnitude patrol movement rather than walking at full speed toward the center.
- Boost energy is reserved for zone recovery and high-threat situations.
- Zone center and radius are interpolated during shrink phases when determining whether to recover toward safety.

## Validation performed

- Unit tests: 15 tests covering geometry, secondary weapon aiming, laser charge/release, projectile dodge (including vertical false-positive and moving-away cases), nearby weapon pickup, and upgrade stacking/caps.
- Official sample-stream conformance: 300/300 observations answered; 0 missed the 50 ms budget; 0 wrong ticks; 0 bad lines; 30 debug drawings within caps.
- Observed host timing: p50 0.2 ms, p90 0.5 ms, p99 7.9 ms, max 9.1 ms on this run. These are host measurements, not a guarantee under the one-CPU match container.
- Docker build not performed because Docker is unavailable in the environment.
- No scored match evaluation performed; competitive performance remains unverified.
