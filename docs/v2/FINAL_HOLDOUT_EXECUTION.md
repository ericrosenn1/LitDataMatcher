# Sealed v4 final-holdout execution

`GSE264666` is consumed contamination evidence after a failed authorized run;
its receipt is preserved and it must never be rerun. `GSE284624` is also
ineligible after accidental nested sample-title exposure. `GSE279879` is the
new v4 identifier-only reservation. It has not been scored. The command below
is prepared only and must not be run until the lead explicitly authorizes its
single execution.

Before it creates a consumption ledger or opens the selected source snapshot,
the evaluator validates the exact reservation/audit state and output path,
loads `jsonschema`, validates the manifest schema, verifies MiniLM file hashes,
and instantiates the local CPU MiniLM runtime. The command uses `runtime-env`,
which was checked on 2026-09-07 to provide `jsonschema`, `torch`,
`transformers`, `numpy`, and the locally verified MiniLM model.

```powershell
& C:\Codex\LitDataMatcher-v2\runtime-env\Scripts\python.exe benchmarks\v2\evaluate_final_holdout.py `
  --reservation benchmarks\v2\final_holdout_reservation_v4.json `
  --audit benchmarks\v2\fallback_identifier_lineage_audit_v4.json `
  --frozen-universe benchmarks\v2\E04_SOURCE_SNAPSHOT_MATCHING.json `
  --source-snapshot C:\Codex\LitDataMatcher-v2\data\snapshots\datasets\objects\10a1096e704bf668a409b9e7aef65d9b6d0453f62eae24c442741b64050b32a2 `
  --source-snapshot-retrieved-at 2026-09-07T11:24:42.941014+00:00 `
  --lead C:\Codex\LitDataMatcher-v2\lead `
  --model C:\Codex\LitDataMatcher-v2\data\models\all-MiniLM-L6-v2\1110a243fdf4706b3f48f1d95db1a4f5529b4d41 `
  --output C:\Codex\LitDataMatcher-v2\data\evaluation\final_holdout_v4\run `
  --consumption-ledger C:\Codex\LitDataMatcher-v2\data\evaluation\final_holdout_v4\CONSUMED.json
```

The evaluator writes lexical, MiniLM-hybrid, and compatibility-aware rankings
over one bound candidate universe, metric numerators and denominators, and the
full hard-negative accession list. Its `RUN_MANIFEST.json` is schema-validated
with `split_role: FINAL_HOLDOUT`,
`source_disjointness.status: PROVEN_SOURCE_DISJOINT`, and
`unknown_overlap_count: 0`. It does not tune models or modify source,
reservation, or evaluator code from results.
