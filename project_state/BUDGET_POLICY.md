# Phase 2 weekly allowance policy

Latest telemetry refresh: `2026-09-11T22:59:45-04:00` (`2026-09-12T02:59:45Z`).

| Field | Current value |
| --- | --- |
| Telemetry source | Supported shared Codex weekly-allowance telemetry |
| Measurement timestamp | `2026-09-11T22:59:45-04:00` (`2026-09-12T02:59:45Z`) |
| Latest weekly remaining | `96%` (`4%` used) |
| Weekly remaining at Medium measurement start | `100%` |
| Weekly remaining at Medium measurement end | `97%` |
| Weekly reset | `2026-09-15T00:49:42-04:00` (`1789447782`) |
| Current reasoning model / effort | `gpt-5.6-terra` / `low` (duty-cycled) |
| Active reasoning-agent count | `0` while in the mandatory idle interval |
| Maximum allowed model / effort | `gpt-5.6-terra` / `high` |
| Maximum reasoning concurrency | `1` |
| Speed / prohibited routes | Standard speed; no Sol, Astra, xhigh, Max, Ultra, paid API, credits, OpenCode, or Muse |
| Corrective supervisor | Paused; runtime model/effort is not verifiable |
| Medium measurement duration | `0.501061 h` |
| Medium observed burn rate | `5.9873 percentage points/hour` (`3 / 0.501061`) |
| Sustainable target at ratchet | `0.46332 percentage points/hour` (`0.80 * 97 / hours_to_reset`) |
| Medium verdict | `UNSUSTAINABLE` |
| Selected maximum profile | One `gpt-5.6-terra` / `low` reasoning worker; zero reasoning subagents |
| Low measurement duration | `0.497894 h` |
| Low observed burn rate | `2.00846 percentage points/hour` (`1 / 0.497894`) |
| Sustainable target at Low reading | `0.45994 percentage points/hour` (`0.80 * 96 / hours_to_reset`) |
| Current sustainable target | `1.04020 percentage points/hour` (`0.80 * 96 / 73.8323 h to reset`) |
| Low verdict | `UNSUSTAINABLE` |
| Required reasoning duty cycle | `22.90%` (`target / burn`): at most `13.74` minutes of Terra Low work per wall-clock hour, followed by at least `46.26` minutes without a reasoning agent |
| Next measurement | After the next accumulated 30 minutes of Terra Low work, after a profile/concurrency/major-workload change, or by `2026-09-12T08:59:45Z`, whichever occurs first |

The Medium profile was ratcheted down after its valid second reading. The Low second reading also exceeded the safety-adjusted target, and Terra Low is the lowest supported practical effort on this runtime. The September 12 supported shared-account refresh observed 4 percent used / 96 percent remaining, but contains no new active-worker interval and therefore does not replace the measured Low burn rate. Retain at most one Terra Low reasoning worker, zero reasoning subagents, and the stricter existing duty cycle until the next accumulated 30-minute Low reading. During its mandatory idle interval, only deterministic local work may run. The paused corrective supervisor must remain paused because its runtime effort cannot be verified within this policy. Do not alter scientific acceptance criteria.

The two sampled workload windows consisted of bounded local Phase 2 implementation, validation, and receipt generation. They did not change model family, reasoning concurrency, source acquisition, model inference, sealed alpha/holdout evidence, or supervisor state. The next Low sample must be based on a new accumulated 30-minute active Low interval and must recalculate the duty cycle before any increase in reasoning time.

## Active-duty ledger

- `2026-09-12T03:01Z` to before `2026-09-12T03:04Z`: one explicitly
  configured `gpt-5.6-terra` / `low` worker, zero concurrent subagents, made
  the isolated cross-source lifecycle patch that was lead-integrated as
  `4cd9e9d`.  The worker exited after its targeted tests; no supervisor or
  network acquisition was used.
- Conservatively allow no further reasoning worker before `2026-09-12T03:50Z`
  (more than the required 46.26-minute idle period after the bounded worker).
  This partial interval does not constitute the required new 30-minute
  telemetry measurement.
