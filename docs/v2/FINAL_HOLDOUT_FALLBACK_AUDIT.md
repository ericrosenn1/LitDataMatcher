# Final holdout fallback audit

GSE282859 is retained as `DISQUALIFIED_INCOMPLETE_LINEAGE`. Its zero known identifier intersections were insufficient because its publication/version and ENA/SRA lineage could not initially be established; the frozen protocol treats that unknown as disqualifying for a proven-independent holdout claim.

The versioned fallback rule in [fallback_identifier_lineage_audit.json](../../benchmarks/v2/fallback_identifier_lineage_audit.json) then considered the next source-order family, GSE264666, using only identifier fields. It found zero known overlap in series, BioProject, GEO samples, PubMed, PMC, ENA study, secondary-study, sample, and run identifiers. It did not read a title, summary, outcome field, label, rank, or prediction.

The v3 continuation closed those relation states using official identifier-only endpoints. A successful GDS-to-PubMed or PubMed-to-PMC response without the requested link is recorded as `EXPLICIT_NO_INDEXED_LINK`; a successful ENA JSON empty array is `EXPLICIT_NO_INDEXED_ENA_READ_RUN_LINK`. These are complete negative relation results, distinct from error or truncation. All 43 ENA/SRA queries returned fewer than 1,000 rows (maximum 234), so pagination was not needed.

GSE264666 now has complete linked PubMed, PMC, ENA, and SRA identifiers and zero exact intersections with the full comparison union across series, BioProject, GEO samples, PubMed, PMC, ENA study, secondary-study, sample, and run identifiers. It is prospectively reserved in [final_holdout_reservation_v3.json](../../benchmarks/v2/final_holdout_reservation_v3.json), pending one-time lead authorization. It has not been scored or executed.

The identifier-only source responses are preserved outside Git in `C:\Codex\LitDataMatcher-v2\data\evaluation\replacement_holdout_identifier_lineage_20260907`; the committed audit records their filenames and SHA256 values. Do not execute the holdout until the lead explicitly authorizes the one-time final run.
