# v0.4 periodic independent review rotation

The v0.4 controller uses fresh, bounded, read-only reviewers at meaningful
campaign checkpoints. This is an independence control, not a standing pool of
concurrent writers. The primary lead remains the sole production-code
integrator.

## Triggers

Run one new reviewer after a major stage, a material scientific or architectural
decision, repeated bounded failures, before held-out freeze/release, after a
major adversarial failure, or after roughly 60–90 minutes of substantive work.
Do not use a reasoning reviewer merely to observe a healthy deterministic job.

## Roles

- Supervisor/status: state, jobs, outputs, failures, and next action.
- Skeptical scientific: source grounding, requirements, abstention, and false
  candidate dispositions.
- Corpus/leakage: grouping, provenance, cross-corpus identity, and split
  contamination.
- Runtime/CUDA: only while isolated CUDA qualification is unresolved.
- Benchmark: only while constructing or revising the 30-case benchmark.

Each review receives the current state, SHA, relevant artifact paths, and an
explicit read-only scope. It returns a compact receipt with reviewer role,
timestamp, source commit, reviewed stage, findings/severity, recommended action,
blocking findings, and evidence paths. The lead persists the receipt under
`data/v0.4_compiler_20260913/controller/reviews/`, records an accept/defer/reject
disposition for every finding, and then stops the reviewer.

No review receipt is a scientific acceptance decision by itself.
