# Final holdout fallback audit

GSE282859 is retained as `DISQUALIFIED_INCOMPLETE_LINEAGE`. Its zero known identifier intersections were insufficient because its publication/version and ENA/SRA lineage could not be established; the frozen protocol treats that unknown as disqualifying for a proven-independent holdout claim.

The versioned fallback rule in [fallback_identifier_lineage_audit.json](../../benchmarks/v2/fallback_identifier_lineage_audit.json) then considered the next source-order family, GSE264666, using only identifier fields. It found zero known overlap in series, BioProject, GEO samples, PubMed, PMC, ENA study, secondary-study, sample, and run identifiers. It did not read a title, summary, outcome field, label, rank, or prediction.

The candidate is still not reservable. The base comparison union contains 15 series without an official GDS-to-PubMed link and nine BioProjects whose bounded ENA identifier query is empty or reaches the response limit. Therefore no candidate can establish publication/version and SRA/ENA disjointness from every required component. The decision is `BLOCKED_INCOMPLETE_BASE_COMPONENT_LINEAGE`, not a weakened independence claim.

The identifier-only source responses are preserved outside Git in `C:\Codex\LitDataMatcher-v2\data\evaluation\replacement_holdout_identifier_lineage_20260907`; the committed audit records their filenames and SHA256 values. Do not execute a final holdout unless the base lineage gaps are closed and a fresh eligible reservation is made.
