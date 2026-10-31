# Benchmark handoff

The benchmark is complete. The original supplied source directory was not changed; all contestant work remained in disposable, isolated worktrees derived from a temporary hash-recorded snapshot.

Primary result: choose Codex subagents for the three tested task classes. Do not add Muse or OpenCode to the long-running build. The initially undiscoverable OpenCode native CLI was later independently validated and then benchmarked; no tested OpenCode model lowered net Codex supervision cost.

Review in this order:

1. `docs/v2/MUSE_DELEGATION_BENCHMARK.md` for the decision and limitations.
2. `benchmarks/muse_delegation/results.tsv` for task-level outcomes and claim issues.
3. `benchmarks/muse_delegation/task_specs/` for the frozen contracts.
4. `benchmarks/muse_delegation/evidence/` for candidate artifacts and raw Muse orchestration logs.
5. `docs/v2/AUXILIARY_AGENT_FINAL_DECISION.md` and `benchmarks/opencode_delegation/` for the Windows-native OpenCode correction, evidence, and final routing decision.

Do not integrate the preserved Muse Task C candidate without correcting its tuple-authorship acceptance and adding a regression test. Do not integrate Muse Task B tests: they fail under the required Windows command.

Do not integrate Big Pickle Task B/C candidates without the documented revisions. No OpenCode candidate is selected for production routing.
