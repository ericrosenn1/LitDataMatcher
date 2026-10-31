# Current PubMed corpus and local search

Complete snapshots can receive an independent bounded acceptance check with
`python -m litdatamatcher.literature_validation --corpus-root CORPUS --ledger-manifest LEDGER_MANIFEST.json --database pubmed-current.sqlite3 --out NEW_VALIDATION_DIRECTORY`.
The validator replays hashed event sidecars into a separate NumPy projection,
compares every current PMID/version, checks evenly spaced winning source events
and 32 sampled normalized title/abstract rows with their retained source hashes,
and performs full SQLite plus external-content FTS integrity checks. Its default
bound is 50 million events and an estimated 6 GiB memory ceiling; no source files
are changed. This oracle explicitly requires positive known versions, including
version-specific deletion events, as present in the audited 2026 ledger. Other
histories fail its admission rather than receiving an incomplete comparison.
The materializer itself retains its broader unknown-version and deletion rules.
`VALIDATION.json` is written only after every check passes; `PROGRESS.json` records
the running phase or a failure. This validates computational/source fidelity,
not biomedical conclusions.

`litdatamatcher.literature_index` materializes the existing verified revision
ledger and normalized Parquet records into a SQLite database. The source corpus,
revision ledger, frozen research outputs, and evaluation holdouts remain unchanged.

Each applied shard and its recovery checkpoint commit in one SQLite transaction.
Continuation admission checks the exact SQLite schema, committed event counter,
and source-bound shard prefix without rescanning the complete payload. Full
SQLite/FTS validation is required separately for final acceptance and is explicitly
marked as pending in materialization receipts. Title metadata remains searchable,
but records with missing abstracts have empty evidence text; titles are never
silently substituted for abstract or full-paper evidence.
An interrupted shard rolls back; a resumed invocation verifies the existing
checkpoint prefix and continues at the next shard. Appended update manifests can
extend an index. Changed or removed previously applied ledger entries are rejected.
New inputs are SHA-256 checked, and existing inputs are stat-checked and rehashed
when their size or modification time changes. `--rehash-applied` requests full
input rehashing during recovery.

The current view follows `pubmed_revision_ledger.resolve_pmid_events`: select the
highest active numeric citation version, use the latest event within a version,
apply version-specific deletions to that version, and apply unversioned deletions
to the entire PMID. Missing/invalid versions and unresolved deletion versions
withhold that PMID from search. A later valid event does not silently erase an
earlier unresolved event. This preserves the existing scientific identity policy.

Start with a bounded run from the repository using an environment with the
existing `corpus` extra (PyArrow):

```powershell
python -m litdatamatcher.literature_index materialize `
  --corpus-root 'D:\LitDataMatcher-Corpora' `
  --ledger-manifest 'C:\Codex\LitDataMatcher-v2\data\campaign_20261031\revision-ledger-full-v1\LEDGER_MANIFEST.json' `
  --database 'C:\Codex\LitDataMatcher-v2\data\full_platform_20261031\literature\pubmed-current.sqlite3' `
  --max-shards 2 --max-seconds 180
if ($LASTEXITCODE -ne 0) { throw 'Literature materialization failed; committed shards remain resumable.' }
```

Repeat the command to apply the next two shards. `--max-shards` limits **new**
shards in that invocation. `--all-shards --max-seconds 600` allows a ten-minute
chunk, stopping between shards; one final shard can extend beyond that bound.
The database status is `PARTIAL` until all shards in a complete ledger are applied.
It is a current view of the acquired snapshot, not a claim of coverage through
the present calendar date. No background process or scheduler is created.

```powershell
python -m litdatamatcher.literature_index search --database 'C:\Codex\LitDataMatcher-v2\data\full_platform_20261031\literature\pubmed-current.sqlite3' 'cancer biomarkers' --limit 10
python -m litdatamatcher.literature_index status --database 'C:\Codex\LitDataMatcher-v2\data\full_platform_20261031\literature\pubmed-current.sqlite3'
python -m litdatamatcher.literature_index validate --database 'C:\Codex\LitDataMatcher-v2\data\full_platform_20261031\literature\pubmed-current.sqlite3'
```

The API supports `LiteratureIndex(path, read_only=True)`, `search(query, limit)`,
`get(pmid)`, `passages(pmid)`, `iter_records(batch_size)`, and `status()`.
For job fingerprints, `snapshot_identity()` hashes the committed shard-receipt
prefix (including source checksums) in a single read transaction. Its cost scales
with shard count, not the number of articles or database bytes. Pin that identity
when submitting a job; a changed prefix requires a new job or explicit policy.
`status(include_counts=False)` avoids record-table scans for routine UI polling.
Older active builders without incremental count metadata return `null` for that
fast count; `status()` still computes exact counts. Never fill missing counts
from an outdated run receipt.
Use a separate connection per thread. Search uses literal-token FTS5 BM25;
scores rank retrieval candidates and are not compatibility probabilities.
Search uses SQLite's [ordered FTS rank interface](https://www.sqlite.org/fts5.html#sorting_by_auxiliary_function_results),
then loads source payloads only for the retained results. Every score-boundary
tie is considered before the existing PMID tie-break, so the weighted BM25
scores and ordering remain unchanged. Ranking and payload reads share one SQLite
snapshot, including while a writer commits new shards.
Documents carry stable `PMID:` identifiers, source URLs, source-shard checksums,
event/version identity, and exact legacy row offsets. Passage results preserve
their exact character offsets in the title or abstract and content-bound IDs.
The returned `text` contains only the abstract. Title-only records remain
searchable metadata and are not promoted to question-extraction text.
Explicit retraction, correction, and expression-of-concern publication types
are mapped to the existing literature-integrity contract. A current citation
must not be confused with eligible evidence. Lifecycle coverage is limited to
the retained publication types: the legacy normalized corpus does not preserve
all comment/correction links, and no claim of complete lifecycle review is made.

`semantic_search(query, model_dir, cache_root, limit=20, candidate_limit=200)`
uses the existing fingerprinted `CachedSemanticIndex` to rerank a bounded lexical
shortlist. It is explicitly labeled `reranked_lexical_shortlist`; it is not an
exhaustive semantic index over all PubMed records. No model download is initiated.

The index stores each active-version payload once. FTS uses an external-content
view, avoiding a second stored abstract copy. Work is streamed in configurable
batches (default 4,096), SQLite's page cache is capped at approximately 64 MiB,
temporary storage is disk-backed, and one writer is protected by the existing
integration lease. SQLite WAL allows concurrent read-only search between commits.
Current-version refreshes use the touched batch as the outer SQL loop, so each
batch does indexed PMID lookups rather than rescanning the accumulated corpus.
The regression suite bounds SQLite virtual-machine work for a single-record
update against thousands of existing records. Each invocation preserves an
immutable receipt and refreshes the latest receipt; receipts bind the source
implementation hash captured at module import.
`validate()` checks SQLite and exact FTS external-content agreement; its full
FTS pass may take time on a multi-million-record index.

Validation initially covered eight focused tests: randomized equivalence to the
existing resolver, updates/deletions/uncertainty, source-span preservation,
transaction rollback, resume and append-only updates, changed inputs, safe query
handling, and semantic shortlist labeling. The first real pilot materialized
60,000 records from two unchanged baseline shards in 3.28 seconds and passed
SQLite/FTS integrity checks. Pilot timing is not an extrapolated full-corpus
performance result. The receipt and source-bound search demonstration are beside
the local database; any subsequent larger run records its actual coverage there.

## Bounded automatic continuation

The existing durable `Controller` can run successive twenty-minute chunks with
unique, immutable output receipts. This is a real foreground coordinator and
native child process, not a schedule or the old supervisor. The cumulative
allowance is capped at two hours, including a reserved two-minute termination
margin per attempt. Completed attempts refund unused reserved time. If the
coordinator is interrupted, its reservation remains charged conservatively;
resuming does not reset the allowance.

```powershell
& 'C:\Codex\LitDataMatcher-v2\env\Scripts\python.exe' -m litdatamatcher.literature_campaign `
  --corpus-root 'D:\LitDataMatcher-Corpora' `
  --ledger-manifest 'C:\Codex\LitDataMatcher-v2\data\campaign_20261031\revision-ledger-full-v1\LEDGER_MANIFEST.json' `
  --database 'C:\Codex\LitDataMatcher-v2\data\full_platform_20261031\literature\pubmed-current.sqlite3' `
  --root 'C:\Codex\LitDataMatcher-v2\data\full_platform_20261031\literature_campaign' `
  --chunk-seconds 1200 --max-total-seconds 7200
if ($LASTEXITCODE -ne 0) { throw 'Corpus continuation stopped on a failure; inspect CORPUS_CAMPAIGN.json and controller logs.' }
```

Add `--resume` to resume that campaign with unchanged code, interpreter, source
identity, and resource configuration. `CORPUS_CAMPAIGN.json` records coordinator
PID/creation time, current job, charged budget, completed chunks, and the latest
committed corpus snapshot. The existing controller database records native
worker ownership, admission, retries, logs, and output hashes. A busy corpus
writer returns exit code 75 and receives at most two retries. Schema/identity
failures stop immediately. Ambiguous surviving workers are preserved and reported
as `RECOVERY_REQUIRED`; the coordinator does not invent safe ownership.

`--max-chunks` and `--shards-per-chunk` provide additional bounded development
profiles. A budget-limited campaign can be followed by a new, explicitly launched
campaign root using the same database; it reuses the applied corpus prefix.
`COMPLETE` means every shard in the pinned ledger was materialized and chunk
source checks/output receipts passed. Run the separate `validate` command for
the full SQLite/FTS content integrity check before final corpus acceptance.

The lower-level materialize command accepts `--receipt-out` for a unique
controller output. Replaying a valid completed chunk verifies its immutable
receipt and historical source prefix, republishes identical receipt bytes, and
does not advance the database. `--expected-ledger-sha256` and
`--expected-implementation-sha256` fence admission against changed inputs/code.
