# Final holdout fallback audit

`GSE112372` remains retired because detailed metadata was exposed.
`GSE282859` remains `DISQUALIFIED_INCOMPLETE_LINEAGE` because its lineage could
not establish the required publication/version and ENA/SRA disjointness.

The authorized v3 `GSE264666` attempt created an exclusive consumption receipt,
opened and scored the snapshot, then failed before manifest creation with
`ModuleNotFoundError: No module named 'jsonschema'`. It is preserved as
`FAILED_CONSUMED_CONTAMINATED_DEVELOPMENT_EVIDENCE`; no result manifest exists,
and it must not be rerun. The receipt hash is recorded in
`final_holdout_reservation_v3.json`.

`GSE284624` was the next source-order candidate. During identifier extraction,
nested sample-title values were inadvertently exposed. It is retained as
`EXPOSED_METADATA_NOT_ELIGIBLE_FOR_UNTOUCHED_CLAIM`; no score, ranking,
prediction, outcome, or label was generated.

The v4 audit selected `GSE279879`, the next candidate in persisted source
order. It used only official identifier relations: GEO accession/sample and
BioProject identifiers; GDS-to-PubMed; PubMed-to-PMC; and ENA/SRA
study/secondary-study/sample/run fields. It found complete candidate relations
and zero exact intersections with the inherited base union plus consumed
`GSE264666`. A successful empty ENA GSE query is recorded as an explicit
no-indexed-link result; the BioProject query returned 14 rows, below the 1,000
row boundary. No GSE279879 title, summary, outcome, label, rank, prediction, or
score was accessed.

Raw identifier responses are outside Git at
`C:\Codex\LitDataMatcher-v2\data\evaluation\replacement_holdout_v4_identifier_lineage_20260907`.
The committed v4 audit retains filenames, sizes, hashes, counts, and the raw
manifest digest. `GSE279879` remains sealed until one-time lead authorization.
