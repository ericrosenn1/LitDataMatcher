# Larger fixed retrieval pools

The expansion command creates 50,000- and 100,000-paper candidate pools for the
existing frozen source-link experiment. The original training and development
query files are copied byte for byte, and the original 1,508 seed papers remain
at the beginning of each pool. No model is trained or invoked by preparation.

```powershell
$env:PYTHONPATH = (Get-Location).Path
& 'C:\Codex\LitDataMatcher-v2\env\Scripts\python.exe' scripts/campaign_20261031/expand_retrieval_candidates.py --corpus-root 'D:\LitDataMatcher-Corpora' --snapshot 'C:\Codex\LitDataMatcher-v2\data\campaign_20261031\corpus-baseline-qualified-v1\CORPUS_SNAPSHOT.json' --edges 'D:\LitDataMatcher-Corpora\normalized\question_paper\question_paper.parquet' --benchmark 'C:\Codex\LitDataMatcher-v2\data\campaign_20261031\retrieval-data-v1' --out 'C:\Codex\LitDataMatcher-v2\data\campaign_20261031\retrieval-scale-v1' --sizes 50000 100000 --views full abstract_only
if ($LASTEXITCODE -ne 0) { throw 'Candidate preparation failed; inspect EXPANSION.json' }
```

Each `pool-50000/full`, `pool-50000/abstract_only`, and corresponding 100,000-paper
directory contains the original query files, candidate `documents.jsonl`, and a
hashed `BENCHMARK_MANIFEST.json`. The manifest records its seed, candidate view,
ordered ID digest, exclusion receipt, source hashes, and source row locations.

The input must be a qualified annual-baseline snapshot with raw checksum
verification. Update shards and failed qualifications are rejected. All
normalized question-paper identifier rows are read regardless of split, without
loading question text. PMID/PMCID types are explicit. EvidenceBench's
`pubmed_<digits>` and `pmc_<digits>` namespaces are explicitly mapped; bare
internal numbers and unsupported types fail closed.

Identifier-only passes across every qualified shard expand normalized linked
aliases until the observed graph reaches a fixed point. Repeated passes catch
backward links between shards. This derivative checkpoint is resumable. Sampled candidates
cannot intersect the resulting PMID/PMCID/DOI exclusion set, repeat a seed ID,
duplicate a selected full text or abstract, lack an abstract, or have a
retracted-publication/retraction-notice type. Sampled shards receive fresh hash
checks; the identifier-only pass checks qualified sizes and modification times.
Original corpus and benchmark files are never overwritten.

Shards are ordered by seeded path hashes; qualifying records inside each shard
are ordered by seeded canonical-PMID hashes. This is a reproducible shard-cluster
sample, not a globally uniform sample. The 50,000-paper ID sequence is a prefix
of the 100,000-paper sequence. Both views have identical IDs and ordering. The
abstract-only view changes only `text`; its retained title is provenance and
must not be fed separately to the encoder.

The larger pools and title-removal view are predeclared diagnostic stress tests,
not replacements for the frozen primary endpoint. Candidates remain unjudged;
the sampler does not assert scientific irrelevance or label them hard negatives.
Compare unchanged methods on the same 175 queries after inference is scheduled,
and report source strata and title-removal effects separately.

A completed expansion is reused only after all manifest/output hashes pass.
Changed configurations and damaged completed artifacts fail clearly. Interrupted
identifier resolution retains checked progress. Published smaller pools are
preserved; an interrupted sampling stage can be regenerated deterministically
from unchanged inputs. Exclusions are exact for recorded identifiers and their
observed aliases; incomplete or incorrect source metadata cannot prove absence
of a scientific relationship.

The October 31 prepared `retrieval-scale-v1` package retains its original pool
bytes and adds `CLOSURE_QUALIFICATION.json` plus `EXCLUSIONS_FIXED_POINT.json`.
Two complete passes over all 1,334 qualified shards closed 13,298 aliases; five
additional aliases were absent from every sampled distractor. The supplemental
qualification checks all four output sets against the completed closure.
`SEED_BENCHMARK_MANIFEST.json` preserves the original benchmark manifest exactly,
so consumers can verify copied train/development bytes against its frozen hashes.
The source implementation and qualification script are retained with the data
receipt. Existing source files and the first pool manifests were not rewritten.

The 1,508-paper pilot strongly favors BM25 and contains substantial exact-title
overlap in PubMedQA. Neither larger pools nor development improvements establish
scientific suitability or unseen-question generalization. The abstract-only view
specifically measures dependence on title availability; it does not replace the
predeclared full-text endpoint or justify selecting the best-looking view.

## Scheduled evaluation

After the GPU is allocated, run the separate evaluator in the existing runtime
environment. It uses local verified weights and never downloads or trains models.

```powershell
$env:PYTHONPATH = (Get-Location).Path
& 'C:\Codex\LitDataMatcher-v2\runtime-env\Scripts\python.exe' scripts/campaign_20261031/evaluate_retrieval_scale.py --root 'C:\Codex\LitDataMatcher-v2\data\campaign_20261031\retrieval-scale-v1' --pretrained 'C:\Codex\LitDataMatcher-v2\data\models\all-MiniLM-L6-v2\1110a243fdf4706b3f48f1d95db1a4f5529b4d41' --trained 'C:\Codex\LitDataMatcher-v2\data\campaign_20261031\retrieval-training-v1\epoch-02' --out 'C:\Codex\LitDataMatcher-v2\data\campaign_20261031\retrieval-scale-evaluation-v1' --device cuda
if ($LASTEXITCODE -ne 0) { throw 'Evaluation failed; inspect RUN.json and completed checkpoints' }
```

Use the identical command with `--resume` after an interruption. Changed code,
inputs, runtime versions, device or model fingerprints require a distinct output
directory. Completed encoding and method checkpoints are hash-verified, including
their concrete record and model identities. An exclusive lease covers execution
and failure receipts. Corrupt artifacts fail without silent replacement. Final
input and model hashes are rechecked before publishing a successful run.

The four conditions share the same 175 development queries. For each model and
view, the evaluator encodes the 100,000 documents once and slices those immutable
vectors for the 50,000 prefix. Query embeddings are cached independently, so the
identical query vectors can also be reused across views. BM25 is fit independently
at each pool size because its corpus statistics change. Its build time and query
ranking time are recorded separately. Semantic receipts distinguish fresh
encoding from verified cache replay, including an explicit warm-cache read.

Every condition reports BM25, pretrained dense retrieval, trained dense retrieval,
BM25 plus pretrained RRF, and BM25 plus trained RRF. Retrieval depth is 100, the
metric cutoff is 10, and the RRF rank constant is 60. Compressed `rankings.jsonl.gz`
files retain all top-100 IDs, scores and fusion components. Method metrics retain
per-query ranks and source strata; `COMPARISON.json` and `RUN.json` contain compact
aggregates and paired comparisons against BM25 and pretrained retrieval.
`DIAGNOSTIC_CONTRASTS.json` records paired title-removal and pool-expansion effects.
Model manifests and vector files are referenced by hashes rather than repeated in
every result. These development comparisons have no promotion rule.

The synthetic tests exercise prefix slicing, exact ranking ties, all five methods,
query-byte changes, missing titles, input drift, corrupted vectors/rankings and
interruption/resume without repeated successful encoding. Passing these tests is
software evidence. The prepared real package has passed CPU-only input validation;
the expanded retrieval results require the separately scheduled model run.
