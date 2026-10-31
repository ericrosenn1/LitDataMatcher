# Current PubMed corpus and local search

`litdatamatcher.literature_index` materializes the existing verified revision
ledger and normalized Parquet records into a SQLite database. The source corpus,
revision ledger, frozen research outputs, and evaluation holdouts remain unchanged.

Each applied shard and its recovery checkpoint commit in one SQLite transaction.
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
Use a separate connection per thread. Search uses literal-token FTS5 BM25;
scores rank retrieval candidates and are not compatibility probabilities.
Documents carry stable `PMID:` identifiers, source URLs, source-shard checksums,
event/version identity, and exact legacy row offsets. Passage results preserve
their exact character offsets in the title or abstract and content-bound IDs.
The returned `text` is the abstract when present, otherwise the title.

`semantic_search(query, model_dir, cache_root, limit=20, candidate_limit=200)`
uses the existing fingerprinted `CachedSemanticIndex` to rerank a bounded lexical
shortlist. It is explicitly labeled `reranked_lexical_shortlist`; it is not an
exhaustive semantic index over all PubMed records. No model download is initiated.

The index stores each active-version payload once. FTS uses an external-content
view, avoiding a second stored abstract copy. Work is streamed in configurable
batches (default 4,096), SQLite's page cache is capped at approximately 64 MiB,
temporary storage is disk-backed, and one writer is protected by the existing
integration lease. SQLite WAL allows concurrent read-only search between commits.
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
