# Phase 2 weekly allowance policy

Latest ratchet decision: `2026-09-08T01:50:29-04:00`.

| Field | Current value |
| --- | --- |
| Telemetry source | Supported shared Codex weekly-allowance telemetry |
| Measurement timestamp | `2026-09-08T01:50:29-04:00` (`2026-09-08T05:50:29Z`) |
| Weekly remaining at Medium measurement start | `100%` |
| Weekly remaining at Medium measurement end | `97%` |
| Weekly reset | `2026-09-15T00:49:42-04:00` (`1789447782`) |
| Current reasoning model / effort | `gpt-5.6-terra` / `low` |
| Active reasoning-agent count | `1` |
| Maximum allowed model / effort | `gpt-5.6-terra` / `high` |
| Maximum reasoning concurrency | `1` |
| Speed / prohibited routes | Standard speed; no Sol, Astra, xhigh, Max, Ultra, paid API, credits, OpenCode, or Muse |
| Corrective supervisor | Paused; runtime model/effort is not verifiable |
| Medium measurement duration | `0.501061 h` |
| Medium observed burn rate | `5.9873 percentage points/hour` (`3 / 0.501061`) |
| Sustainable target at ratchet | `0.46332 percentage points/hour` (`0.80 * 97 / hours_to_reset`) |
| Medium verdict | `UNSUSTAINABLE` |
| Selected maximum profile | One `gpt-5.6-terra` / `low` reasoning worker; zero reasoning subagents |
| Low observed burn rate | `PENDING_SECOND_LOW_READING` |
| Required reasoning duty cycle | `PENDING_SECOND_LOW_READING` (derive only after the Low reading; if continuous Low burn exceeds target, use `target / burn`) |
| Next measurement | At or after `2026-09-08T01:50:29-04:00` (`2026-09-08T05:50:29Z`); also after a model, effort, concurrency, or major workload-profile change, and at most every six hours of continuous Phase 2 work |

The Medium profile was ratcheted down after its valid second reading. Until a valid Low second reading, retain exactly one Terra Low reasoning worker, zero reasoning subagents, and run deterministic jobs independently. Do not infer Low sustainability or alter scientific acceptance criteria. The Low second reading must calculate percentage-points consumed divided by elapsed wall-clock hours, compare it with the safety-adjusted sustainable target, and apply the current duty-cycle policy if needed.

Current workload since this policy was recorded includes bounded V2.4 calibration-readiness and V2.2 cross-modal contract tests and receipts, plus prior deterministic Phase 2 slices. It did not change model, effort, reasoning concurrency, source acquisition, model inference, or supervisor state; the second telemetry reading remains due on the stated schedule.
