# Phase 2 weekly allowance policy

Latest ratchet decision: `2026-09-08T01:50:57-04:00`.

| Field | Current value |
| --- | --- |
| Telemetry source | Supported shared Codex weekly-allowance telemetry |
| Measurement timestamp | `2026-09-08T01:50:57-04:00` (`2026-09-08T05:50:57Z`) |
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
| Low verdict | `UNSUSTAINABLE` |
| Required reasoning duty cycle | `22.90%` (`target / burn`): at most `13.74` minutes of Terra Low work per wall-clock hour, followed by at least `46.26` minutes without a reasoning agent |
| Next measurement | After the next accumulated 30 minutes of Terra Low work, after a profile/concurrency/major-workload change, or by `2026-09-08T11:50:57Z`, whichever occurs first |

The Medium profile was ratcheted down after its valid second reading. The Low second reading also exceeded the safety-adjusted target, and Terra Low is the lowest supported practical effort on this runtime. Retain at most one Terra Low reasoning worker, zero reasoning subagents, and use the stated duty cycle. During its mandatory idle interval, only deterministic local work may run. The paused corrective supervisor must remain paused because its runtime effort cannot be verified within this policy. Do not alter scientific acceptance criteria.

The two sampled workload windows consisted of bounded local Phase 2 implementation, validation, and receipt generation. They did not change model family, reasoning concurrency, source acquisition, model inference, sealed alpha/holdout evidence, or supervisor state. The next Low sample must be based on a new accumulated 30-minute active Low interval and must recalculate the duty cycle before any increase in reasoning time.
