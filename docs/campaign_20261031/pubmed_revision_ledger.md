# Additive PubMed revision ledger

The original 2026 corpus and its failed full-snapshot receipt remain unchanged.
`pubmed_revision_ledger.py` creates separate Parquet sidecars that recover citation
version, revision-date, deletion, and event-order information from the raw XML.
It does not silently replace the old normalization or declare the full corpus
qualified after a small pilot.

## Identity and lifecycle rules

NLM describes a versioned citation by the combination of PMID and version, and
notes that an individual version can be deleted. See
[Versioning in PubMed](https://www.nlm.nih.gov/pubs/techbull/jf12/jf12_pm_versioning.html).
NLM's archive instructions apply baseline files before updates and handle
replacement and deletion events. See
[local archive maintenance](https://www.nlm.nih.gov/dataguide/edirect/archive.html)
and [DeleteCitation](https://dtd.nlm.nih.gov/ncbi/pubmed/doc/out/240101/el-DeleteCitation.html).

Every upsert carries its original zero-based `legacy_row_index`; deletion events
have no legacy row. The sidecar preserves the source `Version` string and its
numeric interpretation, `DateRevised` fields, source filename, global shard order,
and zero-based event index. Complete valid revision dates also have an ISO value;
incomplete dates remain null with the original fields retained.

The pure `resolve_pmid_events` function retains all input events and chooses the
highest numeric active version, then the latest source event within that version.
It does not sort versions lexically or use row order to prefer version 1 over 2.
An explicit-version deletion removes that version. An unversioned deletion is a
PMID-wide tombstone under the ingestion rule; no missing version is invented.
After it, only new upserts may restore an active citation. Missing-version
upserts and malformed deletion versions remain unresolved, with no asserted
selected record. Revision dates are preserved, not guessed or substituted for
the declared source-event ordering.

## Verification and recovery

- The parser compares the **entire ordered PMID sequence** from raw citation
  records with the existing Parquet PMID column, including repeated identifiers.
- Raw compressed MD5 must match the acquisition checksum; raw SHA-256 and current
  normalized-file SHA-256 are retained. Original file changes during processing
  fail validation.
- Completed XML root children are cleared after their data are emitted. Parquet
  writes and reads use 4,096-row batches; the full XML tree is not accumulated.
- A verified Parquet file and its receipt are published together by an atomic
  same-filesystem directory rename. Failed staging and corrupt prior derivatives
  remain in task-owned `failed/` and `quarantine/` directories.
- Resume rechecks raw and normalized hashes, implementation fingerprint, sidecar
  hash/schema/counts, and row/event continuity. It skips XML parsing only for
  already verified partitions. A process lease prevents two writers using the
  same output directory.
- Workers default to two and are limited to four. Interrupts cancel queued work
  and wait only for active partitions; verified partitions remain resumable.
- Any explicit shard filter or `--max-shards` produces `PARTIAL`, even if every
  selected shard passes. A full request covers baseline and update shards from
  the supplied snapshot. Every invocation retains an immutable run receipt.

## Bounded real-source pilot

Only `pubmed26n0001.xml.gz` and `pubmed26n1339.xml.gz` were parsed. The final pilot
directory is
`C:\Codex\LitDataMatcher-v2\data\campaign_20261031\revision-ledger-pilot-v2`.

| Result | Observed |
| --- | ---: |
| Full raw/Parquet sequences verified | 2 shards |
| Citation upserts preserved | 60,000 |
| Deletion events preserved | 13 |
| Fresh two-worker execution | 19.51 seconds |
| Verified resume, no XML reparse | 0.86 seconds |
| Synthetic contract tests | 17 passed |

The known repeated PMID `34621506` occurs at legacy rows 6,278 and 6,282 with
versions 2 and 1 respectively. Both were retained; the resolver selected version
2 at row 6,278. The original Parquet remains intact. The prior full snapshot
remains `FAIL`; this pilot's status is `PARTIAL` with no failed selected shards.

Fresh-run evidence: `RUN_RECEIPT_1793448004466102000.json`.
Resume evidence: `RUN_RECEIPT_1793448012771005900.json`.
Test evidence: `ledger_tests.xml`. Targeted Ruff checks also passed.

A naive row-count extrapolation from this two-shard mix is about four hours for
45.5 million input rows at two workers. This is only an order-of-magnitude
planning estimate: XML size, record complexity, storage load, and worker
contention vary. No full-corpus parser was started by this workstream, and a
larger worker count has not been benchmarked.

## Commands

Run from the changed checkout using the existing environment with PyArrow.
The output must be outside the source corpus. Repeating the same command resumes
only verified matching partitions.

```powershell
$env:PYTHONPATH = (Get-Location).Path
& C:\Codex\LitDataMatcher-v2\env\Scripts\python.exe scripts/campaign_20261031/build_revision_ledger.py `
  --corpus-root D:\LitDataMatcher-Corpora `
  --snapshot C:\Codex\LitDataMatcher-v2\data\campaign_20261031\corpus-snapshot-v1\CORPUS_SNAPSHOT.json `
  --out C:\Codex\LitDataMatcher-v2\data\campaign_20261031\revision-ledger-full `
  --workers 2
```

Add `--max-shards 2 --shard pubmed26n0001.xml.gz --shard pubmed26n1339.xml.gz`
and use a new pilot output path for the bounded check. Full sidecar construction
does not itself perform a global current-version materialization; consumers can
group sidecar events by PMID and apply the documented pure resolver. Accessions,
version lineage, and deletion events must remain available in that derivative.
