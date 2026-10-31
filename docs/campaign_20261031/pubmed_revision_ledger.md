# Additive PubMed revision ledger

The complete ledger and its independent audit passed for the supplied 2026
baseline/update snapshot. The ledger preserves **45,488,930 historical events**
in separate Parquet sidecars. It does not overwrite the original corpus,
retroactively change its snapshot's **FAIL** status, or materialize a global
current-version view.

The compact [public result record](https://github.com/ericrosenn1/LitDataMatcher/blob/9e2d2916752a23d2b6510ccf8f2d1cd574e5c6ae/benchmarks/campaign_20261031/pubmed_revision_ledger.json)
contains exact counts, implementation identities, and source-receipt hashes.

## Measured full-run results

| Measure | Observed |
| --- | ---: |
| Verified shards | 1,621 / 1,621 |
| Baseline / update shards | 1,334 / 287 |
| Citation upsert events | 45,477,637 |
| Deletion events | 11,293 |
| Total events | 45,488,930 |
| Failed shards / audit errors | 0 / 0 |
| Unknown versions / unresolved revision dates | 0 / 0 |
| Full builder elapsed | 6,342.674 seconds (105.71 minutes) |
| Independent audit elapsed | 105.204 seconds |

All observed deletions were version-specific. The audit found zero PMID-wide
or unresolved-version deletion events in this snapshot. These are measured
properties of these files, not assumptions about future inputs. Counts describe
**events**, not unique active citations.

The full build used two spawned process workers, with worker-pool startup
included in its elapsed timer and no reused partitions. It ran alongside other
local research work; this is a single-machine observation, not a throughput
promise. The earlier two-shard extrapolation is superseded by this measured
full-run duration.

The frozen full run used builder SHA-256
`a02fe627a8dbe8526f8405874320aaa727f46d3cdc9822c2867c6d7a3fbcdf70`.
Its manifest SHA-256 is
`d2b17324ebe48fdd577df3687e75e1c84d18553d78c49114a74a97946ddfdc3f`;
the matching immutable run receipt is `RUN_RECEIPT_1793454538150667800.json`.
The original corpus-snapshot receipt remains unchanged and reports FAIL.

## What is retained

Each upsert has its original zero-based `legacy_row_index`. Deletions have no
legacy citation row. Every event retains PMID, raw `Version`, parsed numeric
version, raw `DateRevised` fields, any valid ISO revision date, source shard,
source class, global shard order, and zero-based event index. Incomplete dates
or unknown versions remain explicit rather than being filled in.

NLM describes versioned citations using the PMID/version pair and permits
individual versions to be deleted. See [Versioning in PubMed](https://www.nlm.nih.gov/pubs/techbull/jf12/jf12_pm_versioning.html).
Its [archive-maintenance guide](https://www.nlm.nih.gov/dataguide/edirect/archive.html)
processes baseline files before updates. The [DeleteCitation definition](https://dtd.nlm.nih.gov/ncbi/pubmed/doc/out/240101/el-DeleteCitation.html)
also allows a deletion for a record not previously distributed.

The separate `resolve_pmid_events` function retains history, then selects the
highest active numeric version and the latest source event within that version.
An explicit-version deletion removes that version. Under the project's ingestion
policy, a genuinely unversioned deletion is a PMID-wide tombstone; malformed
version strings remain unresolved. Only later observed upserts can restore a
record after a tombstone. Missing-version upserts do not acquire an invented
version. Revision dates do not replace source-event ordering.

**No global current-view resolution was performed by this build or audit.** A
consumer must group events by PMID and apply an explicit resolution policy,
retaining the source/version lineage in the resulting derivative.

## Verification and preservation

The builder streamed the entire raw XML citation sequence and compared every
upsert PMID with the original normalized Parquet order, including repeated
identifiers. It checked acquisition MD5, retained raw and normalized SHA-256,
and atomically published each sidecar with its receipt. Streaming root clearing
and bounded Parquet batches avoid retaining the full XML tree in memory.

The independent auditor uses its own validation code, not the builder's
validator. It checked every sidecar and shard receipt, the final manifest and
snapshot binding, exact Arrow schemas/counts, event order, legacy-row continuity,
raw/parsed versions, revision dates, and deletion fields. It compared the current
normalized PMID column in full and rechecked **6,484** raw/legacy/sidecar/receipt
stat identities at completion. All 1,621 partitions share the single recorded
builder implementation hash.

A further review rechecked 1,631 control JSON files and aggregate arithmetic,
with status PASS_WITH_LIMITATIONS. This additional review did not repeat the
full Parquet scan or reparse raw XML.

The independent audit does **not** reparse or rehash raw XML, or independently
rehash non-PMID legacy columns. It verifies recorded acquisition bindings and
current source size/timestamps. The original snapshot FAIL, all earlier pilot
receipts, and the full-run artifacts remain preserved. Computational integrity
is not biomedical truth, source completeness, or downstream dataset suitability.

## Later path/format compatibility check

After the full build, a Windows path-normalization repair and formatting change
were checked in a **new two-shard pilot**. Its 60,000 upserts and 13 deletions were
byte-identical to the corresponding full-run sidecars. Fresh execution took
10.063 seconds; verified resume reused both partitions in 0.960 seconds. Its
independent audit took 0.412 seconds and correctly reported PARTIAL coverage.

That check has a different implementation fingerprint. It does not imply the
whole ledger was rerun under the new code. Full and pilot identities, timing
scopes, and the byte-equivalence receipt are separated in the public JSON.

## Reproduction commands

Use the matching checkout/environment with PyArrow; see
[reproducibility](../reproducibility.md). Replace the uppercase path tokens with
explicit local paths. Preserve validated outputs; use a new directory for a
new implementation or experiment.

```text
python scripts/campaign_20261031/build_revision_ledger.py --corpus-root CORPUS_ROOT --snapshot CORPUS_SNAPSHOT.json --out NEW_LEDGER_DIRECTORY --workers 2 --executor process
python scripts/campaign_20261031/audit_revision_ledger.py --corpus-root CORPUS_ROOT --snapshot CORPUS_SNAPSHOT.json --ledger LEDGER_DIRECTORY --report NEW_AUDIT.json
```

Resume construction only with matching source and implementation identities.
Verified partitions are reused; corrupt derivatives are quarantined with their
history retained. The builder defaults to two workers, allows at most four, and
does not silently switch executors. Any explicit shard filter or `--max-shards`
is PARTIAL. Audit a pilot only with `--allow-partial`; leave that flag off for a
full-ledger acceptance check. Reports must use new paths outside both protected
input roots.
