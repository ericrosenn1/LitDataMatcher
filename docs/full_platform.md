# Integrated scientific application

**Scientific workflow** connects the React workspace to the Python question
compiler, source adapters, dataset inspectors, compatibility assessor and durable
controller. The original Discover view remains a lightweight lexical workflow.

With dependencies installed, build and launch from the repository:

```powershell
Push-Location web
npm ci
npm run build
if ($LASTEXITCODE -ne 0) { throw 'Web build failed' }
Pop-Location
python -m litdatamatcher.platform_server --root local/application --port 8765
```

Open `http://127.0.0.1:8765`. This is a single-user loopback service with
same-origin/Host checks and fixed worker commands. Multi-user hosting and
authentication require a separate deployment design. No scheduler is implied.

Optional arguments:

- `--catalog-root PATH`: existing `catalog/literature.jsonl` and
  `catalog/studies.jsonl`; small-catalog searches scan at most 10,000 rows.
- `--literature-index PATH`: materialized PubMed SQLite database. Searches pin
  a committed shard prefix in one read transaction. Jobs freeze acquired records;
  create a new job to refresh coverage.
- `--model-dir PATH --embedding-dir PATH --device cuda`: verified local Qwen and
  MiniLM assets. Model downloads are not automatic. Install the repository's
  `runtime,structured` optional dependencies; health reports missing libraries.
- `--offline`: require cached responses. A cache miss remains a source failure.
- `--job-timeout 1800`: configurable from 30 to 7,200 seconds.

The application supports source selection, exact passage extraction, duplicate
grouping, question editing, typed mandatory/preferred requirements, preview,
dataset discovery, bounded inspection, ranked compatibility, field-level evidence
and JSON/CSV/Markdown exports. Models and semantic ranking are explicit modes.
Proposed designs remain unassessed until justified. Unknown measurements never
become confirmed absences or positive matches.

Source adapters include GEO, ENA/SRA, BioProject, BioSample,
BioStudies/ArrayExpress, Expression Atlas, PRIDE, Metabolomics Workbench and
ClinicalTrials.gov. Discovery does not establish independent replication,
measured outcomes or statistical adequacy. Deeper inspectors currently cover GEO
samples/matrices and bounded processed tables.

## Recovery

The existing Controller owns `controller/jobs.sqlite3`, process admission,
bounded attempts and output receipts. Existing CheckpointRun journals own stage
artifacts. Pause stops the payload tree while preserving completed checkpoints.
Resume continues queued work. Failed/incomplete jobs can be retried up to three
attempts, reusing successful source nodes and dataset inspections. Ambiguous
abandoned-process ownership remains RECOVERY_REQUIRED. Application jobs share
fingerprinted inference/vector caches and request-keyed source caches.

Restart with the same arguments plus `--resume`. Software, dependency, parameter,
model or catalog changes invalidate run fingerprints: preserve the old job and
submit a new one. Exports require valid output receipts. Supplied source text is
untrusted evidence. Eligibility precedes task relevance, then coverage in ranking;
scores remain uncalibrated.

Local executable development plans use the same controller:

```powershell
python -m litdatamatcher.development_campaign --plan PLAN.json --state-root local/campaign --max-jobs 10 --timeout 1800
```

A plan contains `campaign_id`, optional immutable `inputs`, and dependency-ordered
`milestones` with `id`, literal `command` array, `cwd`, `outputs`, optional
`depends_on` and `required_commit`. Resume verifies commit ancestry, input hashes
and output receipts before requeuing unfinished/failed work. Only locally
authored configuration executes commands. The bounded invocation exits normally;
it does not claim an ongoing scheduler.

## Scientific limits

Software checks and development examples are not independent scientific
evaluation. Twelve variable-availability checks across two datasets do not
establish twelve scientifically qualified opportunities. Full contracts retain
unresolved timing, ascertainment and independence requirements. Scientific
criteria were not weakened to produce positives.

The GSE279879 final holdout remains excluded from development inspection. The
final protocol, independent scientific review and calibration remain pending.
There is no claim of global novelty, clinical utility or exhaustive coverage.
