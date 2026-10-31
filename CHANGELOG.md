# Changelog

## 0.4.0 research preview — unreleased

### Research engine

- Configurable parallel source-query plans, independent clients, immutable
  cached responses, source receipts, and explicit partial failure.
- Serial multi-case analysis with shared local models, verified vectors,
  durable stage checkpoints, per-case journals, and resumable execution.
- Opt-in source-constrained generation with an independent source guard.
  Qualification binds the model, code, decoder, source, and settings.
- Escape-safe source-literal decoding, with preserved failed alternatives,
  exact source spans, and independently checked fresh/cache replay.
- Source-linked retrieval evaluation, fixed local contrastive training, paired
  development comparisons, and larger candidate-pool tooling.
- Annual PubMed integrity checks and additive revision/deletion ledgers.
  Original shards and earlier validated outputs remain unchanged.

### Correctness and recovery

- Reject nonfinite JSON, malformed provider responses, and corrupt caches.
- Preserve Unicode paragraph separators inside scientific JSONL text.
- Prevent cross-clause negation and unsupported relation/context extraction.
- Keep inferred or missing capabilities distinct from explicit source evidence.
- Preserve completed work across interruptions and close journal ownership races.
- Gate subprocess execution on durable admission, preserve pause requests,
  and refuse unsafe recovery of ambiguous abandoned workers.
- Validate complete model asset sets, tokenizer/index references, and native
  run manifests; reject unrelated ancestor Git identities in source exports.
- Package the training protocol and retain Python 3.10 model-hashing support.

### Evaluation and delivery

The [research workflow](docs/research_workflow.md),
[retrieval results](benchmarks/campaign_20261031/retrieval_development.json), and
[campaign state](project_state/campaign_20261031/TASK_STATE.json) distinguish
completed measurements from ongoing validation. Source-link recovery and tests
do not establish dataset suitability. No expert labels or calibrated scientific
validity claims were fabricated.

The browser workspace remains an optional review surface. Public release,
default-branch merge, and deployment are separate actions.
