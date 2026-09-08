# Phase 2 weekly allowance policy

Measurement started: `2026-09-08T00:51:00-04:00`.

| Field | Current value |
| --- | --- |
| Telemetry source | Supported shared Codex weekly-allowance telemetry |
| Weekly remaining at start | `100%` |
| Weekly reset | `2026-09-15T00:49:42-04:00` (`1789447782`) |
| Current reasoning model / effort | `gpt-5.6-terra` / `medium` |
| Active reasoning-agent count | `1` |
| Maximum allowed model / effort | `gpt-5.6-terra` / `high` |
| Maximum reasoning concurrency | `1` |
| Speed / prohibited routes | Standard speed; no Sol, Astra, xhigh, Max, Ultra, paid API, credits, OpenCode, or Muse |
| Corrective supervisor | Paused; runtime model/effort is not verifiable |
| Observed burn rate | `PENDING_SECOND_READING` |
| Sustainable target | `PENDING_SECOND_READING` (must be `0.80 * remaining / hours_to_reset`) |
| Verdict | `PENDING_SECOND_READING` |
| Required reasoning duty cycle | `PENDING_SECOND_READING` |
| Next measurement | At or after `2026-09-08T01:21:00-04:00`, preferably near `2026-09-08T01:51:00-04:00`; also after a model, effort, concurrency, or major workload-profile change, and at most every six hours of continuous Phase 2 work |

Until a valid second reading, retain exactly one Terra Medium reasoning worker, zero reasoning subagents, and run deterministic jobs independently. Do not infer sustainability from the earlier window or alter scientific acceptance criteria. The second reading must calculate percentage-points consumed divided by elapsed wall-clock hours, compare it with the safety-adjusted sustainable target, and ratchet only as the current user policy specifies.

Current workload since this policy was recorded: bounded local V2.3 evidence-compiler, V2.4 expert-review, V2.5 scale/recovery, and literature-integrity contract integration with deterministic test/receipt execution. It did not change model, effort, reasoning concurrency, source acquisition, model inference, or supervisor state; the second telemetry reading remains due on the stated schedule.
