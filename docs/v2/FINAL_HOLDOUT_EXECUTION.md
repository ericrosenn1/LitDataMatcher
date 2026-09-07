# Sealed v3 final-holdout execution

`GSE264666` is reserved under `final_holdout_reservation_v3.json`. It has not
been scored. The command below is prepared only; it must not be run until the
lead gives the one-time execution authorization required by the reservation.

Before it opens the selected GEO snapshot, `evaluate_final_holdout.py` requires
the exact reservation status, the pinned audit hash, completed official
relation checks, zero exact identifier overlap, and all four pre-execution
states (`UNINSPECTED`, `UNINSPECTED`, `NOT_RUN`, `NOT_RUN`). It then creates
the exclusive consumption ledger. A ledger already present refuses every later
attempt, including after a failed attempt.

```powershell
& C:\Codex\LitDataMatcher-v2\runtime-env\Scripts\python.exe benchmarks\v2\evaluate_final_holdout.py `
  --reservation benchmarks\v2\final_holdout_reservation_v3.json `
  --audit benchmarks\v2\fallback_identifier_lineage_audit.json `
  --frozen-universe benchmarks\v2\E04_SOURCE_SNAPSHOT_MATCHING.json `
  --source-snapshot C:\Codex\LitDataMatcher-v2\data\snapshots\datasets\objects\10a1096e704bf668a409b9e7aef65d9b6d0453f62eae24c442741b64050b32a2 `
  --source-snapshot-retrieved-at 2026-09-07T11:24:42.941014+00:00 `
  --lead C:\Codex\LitDataMatcher-v2\lead `
  --model C:\Codex\LitDataMatcher-v2\data\models\all-MiniLM-L6-v2\1110a243fdf4706b3f48f1d95db1a4f5529b4d41 `
  --output C:\Codex\LitDataMatcher-v2\data\evaluation\final_holdout_v3\run `
  --consumption-ledger C:\Codex\LitDataMatcher-v2\data\evaluation\final_holdout_v3\CONSUMED.json
```

The evaluator writes lexical, MiniLM-hybrid, and compatibility-aware rankings
over the same bound universe, metric numerators and denominators, and the full
hard-negative accession list. Its `RUN_MANIFEST.json` is schema-validated with
`split_role: FINAL_HOLDOUT` and a
`PROVEN_SOURCE_DISJOINT` proof whose `unknown_overlap_count` is zero. It does
not tune models or modify source, reservation, or evaluator code from results.
