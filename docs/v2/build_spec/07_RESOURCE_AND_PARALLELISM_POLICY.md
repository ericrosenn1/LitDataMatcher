# Resource scheduling and useful parallelism

## Inspect, do not assume

Historical hardware context is a Ryzen 9 9950X3D, 96 GB RAM and RTX 5090. Verify the actual machine, logical/physical CPU counts, free RAM, GPU/VRAM, drivers, storage, active applications and other research jobs. Do not allocate resources as though other projects were idle. Do not stop, reprioritize, or change configuration for unrelated processes.

Codex cloud reasoning and local CPU/GPU use are separate resources. Giving the local machine more cores does not raise Codex's account limits. Allocate model-agent concurrency and computational job concurrency independently.

## Minimal governor

Implement a small optional local resource governor, not a mandatory large monitoring service. Prefer existing OS/process telemetry and already installed GPU tools. If an optional sensor is unavailable, use conservative documented limits and continue. Do not install drivers, alter firmware, overclock, change power limits, or require administrator services for ordinary scheduling.

Suggested starting policies, to tune against actual measurements:

| Mode | Trigger | Behavior |
|---|---|---|
| Interactive | Recent user input or observed interactive workload | Smaller background pools, below-normal priority for task-owned heavy jobs, modest GPU batches, responsive UI |
| Idle | Sustained absence of input plus available measured capacity | Expand independent CPU/I/O jobs and batch GPU work while preserving headroom |
| Pressure | RAM/VRAM pressure, paging, failed allocations, thermal throttling or sustained I/O congestion | Stop admitting heavy new jobs, shrink batches/pools, retain checkpoints, recover before expanding |
| User pause/maintenance | Explicit project pause or stop | Stop admitting work, checkpoint safely, no automatic supervisor takeover |

Use hysteresis so a mouse movement does not repeatedly restart jobs. If idle time is available, begin with roughly ten minutes to enter idle and immediate protective throttling on renewed activity; tune rather than freezing this as a scientific rule. Resource changes should primarily affect admission, batching and task-owned process priority, not interrupt valid long-running work.

Reserve approximately 15-20% system RAM at initial configuration and GPU headroom appropriate to display use and model workload, then adjust empirically. These are scheduling defaults, not acceptance thresholds. Avoid paging and redundant per-process copies of large models. One shared inference service or batched worker can be better than multiple GPU agents competing for VRAM. Idle mode is not permission to consume 100% of every resource or starve other applications.

CPU-only work, GPU inference, and network acquisition may overlap where bottlenecks differ. Network jobs still honor provider limits and `Retry-After`. A slower source must not block unrelated sources. Do not evade source rate limits with extra accounts or excessive retry.

## Process control

Every owned process/job has a task ID, command, cwd, input version, start time, PID plus creation identity, expected outputs, timeout/lease policy and log path. Drain stdout and stderr concurrently or redirect both safely to bounded logs. Preserve numeric exit status immediately. A terminal model message alone is not the process exit code.

Timeouts are specific to workload size and observed progress. On a real stall, terminate only the verified task-owned process tree after preserving diagnostic state; do not kill all Python, WSL, Node, or Codex processes. Handle parent/child cleanup and output flushing. Do not delete partial downloads or completed batches that can be safely resumed.

Do not assume Windows-to-WSL filesystem access or Git metadata is interchangeable. Keep the lead/Git controller native in one environment. WSL can host a qualified inference or data worker with an explicit file/IPC contract. Do not patch `.git` pointer files as a permanent interoperability trick.

## Concurrency decisions

Use an adaptive pool based on actual Codex capacity, task independence, shared-file conflict risk, measured CPU/GPU/RAM/disk limits, and whether another worker advances the critical path. Do not set 'ten roles' equal to 'ten active agents'. Do not impose an arbitrary tiny pool when more useful independent work fits.

A writer owns one worktree. The lead integrates serially. Shared databases and mutable caches follow their actual concurrency guarantees. Read-only snapshot work may fan out; Git operations and dependent schema changes need coordination. Materialize a single reusable immutable corpus/index instead of duplicating it in each worktree.

## Measurement and unattended prerequisites

Measure cold/warm acquisition, parsing and extraction throughput, catalog update time, local matching p50/p95, CPU/RAM/VRAM peaks, cache hit rates, failed/retried jobs and external requests. Compare quality and supervision overhead alongside speed. Performance results must describe dataset/model/hardware versions and whether inference was cached.

Keep the computer awake while a verified local unattended job needs it using an existing user-controlled setting or a reversible task-scoped mechanism. Do not silently change global power policy. App scheduling also needs the appropriate local app running and project available [S02]. Show any required user setting once, rather than repeatedly declaring a silent schedule active.
