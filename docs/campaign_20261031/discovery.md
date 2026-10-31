# Configurable biomedical discovery

```bash
python -m litdatamatcher.discovery --plan examples/discovery-plan.json --out runs/biomedical --workers 2
```

The example searches six broad biomedical topics. Change its queries and source
lists to match the study. Query plans require unique IDs, explicit source names,
and a bounded number of retained records per source. They do not change the
research question or imply exhaustive literature coverage.

Each source node owns a separate HTTP client and cache, avoiding cross-thread
response-provenance races. At most four network workers are allowed. The default
is two. Existing adapters apply their pagination and retry limits. A retained
result limit is not a guarantee that a provider returns that many records.

The output is a native analysis workspace: `catalog/literature.jsonl`,
`catalog/studies.jsonl`, immutable response snapshots, and per-query/source
checkpoints. `DISCOVERY_MANIFEST.json` reports success, empty, partial failure,
and failure separately. Duplicate dataset observations point back to every
checkpoint; the first observation in plan order supplies the catalog view.
Alternative source records remain intact in those checkpoints.

Repeat with `--resume` to verify and reuse all completed checkpoints. Add
`--retry-failed` to rerun only failed source jobs; their previous bytes move into
content-addressed attempt history. `--offline --resume` verifies the run without
new requests. Changed query plans or implementation fingerprints require a new
output directory. Checkpoint corruption fails closed.

Discovery retains inferred adapter annotations for inspection but initializes
scientific capabilities as unknown. A title, disease term, trial registration,
or public metadata page does not establish available patient-level data,
observed outcomes, independent participants, reuse rights, or statistical power.
Those facts require source-specific inspection and an explicit evidence record.

The CLI returns a nonzero exit code when any source fails, while preserving
successful nodes and the failure report. Such a run can still be inspected and
resumed; it must not be presented as complete source coverage.
