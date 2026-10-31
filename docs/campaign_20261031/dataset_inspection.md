# Dataset inspection and qualification development

The public dataset registry now includes BioStudies and its ArrayExpress collection.
Search results retain repository accessions, response snapshots, total-hit metadata,
and bounded-page status. File counts and literature-associated BioStudies entries
are never interpreted as reusable measurements. GEO, ENA/SRA, PRIDE, Metabolomics
Workbench, clinical registries and DataCite continue using the existing adapters.

`litdatamatcher.dataset_inspection.inspect_dataset(record, cache_dir, *,
offline=False, max_samples=40, max_bytes=8_000_000)` returns an enriched record.
It acquires small GEO SOFT series/sample annotations through the primary GEO
endpoint. Its immutable hash-verified snapshots use the existing acquisition
client. A study can return `INSPECTED_METADATA`, `PARTIAL_METADATA`,
`SOURCE_FAILURE`, or `UNSUPPORTED_SOURCE`. Failures remain distinct from negative
compatibility. Complete sample coverage is required before generalizing sampled
organism/tissue/perturbation attributes to the study. Sample attribute evidence
is named explicitly and never relabeled as a measured outcome. GSM count is not
an independent-subject count. The reserved GSE279879 study and explicitly linked
series are excluded from iterative inspection.

```powershell
python -m litdatamatcher.dataset_inspection --accession GSE4107 --cache .cache/inspection --out local/GSE4107.json
python -m litdatamatcher.dataset_inspection --accession GSE4107 --cache .cache/inspection --out local/GSE4107-replay.json --offline
```

`inspect_geo_matrix(path, accession, source_url=...)` checks a local matrix's
identity, complete table, column alignment, numeric cells and unique features
using the native acquisition parser. It does not fetch arbitrary URLs.
`inspect_tabular_dataset(path, metadata, source_url=..., metadata_url=...)`
validates a processed CSV against its source variable dictionary, retaining
missingness, units, categorical values, observed variables and target/feature
co-observation counts. No normalization, imputation or biological analysis occurs.

The executed development example reused the hash-verified UCI WDBC and Cleveland
Heart Disease source files. It produced **12 positive variable-availability
matches across two datasets**, two incompatible missing-measurement controls,
and two correctly uncertain independence controls. The **full compiled scientific
contracts qualified zero matches**: temporal ordering and outcome ascertainment
remain unsupported. This is a functional availability demonstration, not twelve
novel unanswered questions, independent studies, expert labels or scientifically
validated research opportunities. The initial failed qualification attempt is
preserved separately; scientific criteria were not weakened to obtain positives.

```powershell
python -m scripts.campaign_20261031.demonstrate_dataset_qualification --cached-uci-root 'C:\Codex\LitDataMatcher-v2\data\campaign_20261031\processed-inspection-v1\uci-tabular' --out 'C:\Codex\LitDataMatcher-v2\data\full_platform_20261031\dataset-qualification-new'
```

Live source execution inspected all 22 GEO samples in GSE4107. BioStudies and
ArrayExpress each returned 25 bounded records for `type 2 diabetes`, and offline
replay retained their accession lists. These are transport/normalization tests,
not precision or recall estimates. Large matrices, other repositories' sample
inspection, systematic family closure, causal feasibility and independent
scientific reference evaluation remain incomplete.

The implementation follows the primary [GEO SOFT format](https://www.ncbi.nlm.nih.gov/geo/info/soft.html)
and [BioStudies API documentation](https://www.ebi.ac.uk/biostudies/help).
Local executed evidence is in
`C:\Codex\LitDataMatcher-v2\data\full_platform_20261031\dataset-qualification-v2`;
GEO inspection and the first attempt are in the adjacent `dataset-qualification-v1`.
