# Research-engine campaign — 2026-10-31

The owner requested a substantial biomedical research-tool campaign, with a soft target of 24 hours: improve model quality and performance, complete and harden the multi-node workflow, test and repair repeatedly, and deliver a clean, documented, reproducible research application. The browser interface remains an optional review surface.

## Starting point and preservation

- Research incumbent: `068f32618754a822a8bbdb53244b515d11e2050d`, branch `codex/litdatamatcher-v2-build`.
- Browser interface: `07f74c86923612d456085ea25306c40fcbf42be6`.
- New integration branch: `codex/research-engine-campaign-20261031`.
- Prior v0.3 completion and v0.4 continuation files are historical evidence, not proof that the current task is finished.
- The original checkouts, shared source corpora, frozen alpha, sealed holdout, and existing validated outputs remain unchanged. New experiments use new output directories and explicit input/model fingerprints.
- The old supervisor remains paused. No schedules, paid APIs, forced pushes, main-branch merges, or public releases are authorized by this campaign.

## Work and evidence

| Workstream | Concrete output | Validation |
| --- | --- | --- |
| Integrate the actual research incumbent | One maintained Python tool plus optional browser review | Existing tests, artifact compatibility, clean installation |
| Extraction and requirements quality | Repaired source guard, conservative coreference, complete role parsing, retained valid inference partitions | Reproduced failures, adversarial negatives, source-span evidence, unchanged existing cases |
| Source/data integrity | Faithful capability evidence, cache acquisition lineage, strict source schema/error handling | Frozen-cache replay, missing/failed/empty separation, source contract tests |
| Multi-node execution | Durable checkpoints, bounded retries, strict artifact validation, recoverable stages | Forced interruption, corrupt artifacts, stale input/config invalidation, concurrent ownership tests |
| Corpus readiness | A new per-shard integrity receipt for the available corpus snapshot | Parquet/schema/count/source reconciliation and recorded file digests; no assumption from old handoff counts |
| Model and retrieval strength | Comparable lexical/pretrained/hybrid candidates and measured runtime behavior | Training/development separation, source-backed benchmark, coverage/abstention/error analysis, cold/warm cost |
| Research validation | At least 30 source-backed cases where data support them; actual data inspection for selected feasible cases | Original source identities and limitations, explicit no-fit cases, no fabricated expert labels |
| Delivery | Organized source, documented CLI/workflows, validation report, wheel/sdist and verified handoff ZIP | Fresh environment install, offline replay/fresh-runtime checks, archive/manifest integrity |

## Experiment rules

1. Establish the integrated incumbent before modifying scientific code. Preserve its output and software versions.
2. Use the existing training/development allocation only after input integrity is qualified. Do not inspect or tune on sealed or reserved final-evaluation content.
3. Separate engineering fixtures, source/model-assisted reference cases, and genuine externally annotated evaluation. Never call one a substitute for another.
4. Promote a candidate only after before/after comparison on the same inputs and regression checks. A failure or no-gain result remains in the report.
5. Source-explicit facts, inferred requirements, unknown metadata, incompatibility, and retrieval failure remain distinct. Scores remain uncalibrated unless justified by valid labels and evaluation.
6. Every production writer owns an isolated worktree. The lead integrates and pushes. No simultaneous writers to the same checkout.
7. Resource use is bounded and recorded. The soft time budget is a planning guide, not a reason for idle waiting or premature scientific acceptance.

Current state and evidence pointers are recorded separately in `project_state/campaign_20261031/`; large inputs, model weights, and run outputs remain outside Git.
