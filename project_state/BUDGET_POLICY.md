# Current resource policy: continuous execution authorized

The project-owner override `OWNER_CONTINUOUS_OVERRIDE_20260913.md` revokes the
former 13.74/46.26-minute duty cycle, mandatory idle periods, safety-margin
throttling, and September 13 21:13:30 UTC delay effective immediately. Earlier
measurements and policies remain historical evidence in Git at `b863d86`.

| Field | Current policy |
| --- | --- |
| Mode | Continuous execution |
| Maximum model | gpt-5.6-terra |
| Allowed effort | low, medium, high |
| Reasoning concurrency | 1 verified execution |
| Concurrent subagents | 0 |
| Speed | Standard only |
| Mandatory idle | None |
| Next permitted time | Immediately when runtime is verified within the ceiling |
| Actual current lead | gpt-6-astra / max; outside current authorization |
| Supervisor | PAUSED; no restart or schedule creation |
| Paid routes | No paid API usage, credits, external billing or paid fallback |
| Throttling on projected use | Revoked |
| Telemetry cadence | Normal supported checks at major checkpoints |
| Real capacity stop | Actual account/platform exhaustion |

The last recorded shared-account read at 2026-09-13T20:17:10Z observed 97%
remaining, reset 2026-09-20T19:24:15Z. It is historical, not a refreshed reading.
No usage projection is used to block work. The present blocker is the verified
runtime model/effort mismatch, not allowance or an idle deadline.
