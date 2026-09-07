# Replacement final holdout preparation

GSE112372 cannot be claimed as untouched because its detailed metadata was exposed during targeted acquisition. The replacement reservation is [replacement_final_holdout_reservation.json](../../benchmarks/v2/replacement_final_holdout_reservation.json).

The reservation selects GSE282859 using only the persisted catalog order and explicit identifier fields. It excludes every family used for development or transfer scoring, the five challenge-development families, the retired GSE112372 family, and both transfer reservations. No source title, summary, outcome field, generated label, rank, or prediction for GSE282859 was read by this evaluator.

Known series, BioProject, and GEO sample identifiers have zero exact intersections with the excluded-family union. The catalog has no declared publication identifier for this candidate and the acquired ENA rows cover only GSE278521, so publication/version and ENA/SRA disjointness remain unresolved. Exact identifier separation therefore supports only `KNOWN_IDENTIFIER_DISJOINT`, never proven cohort independence or a full final-holdout claim.

The replacement is `PREPARED_NOT_AUTHORIZED_FOR_EXECUTION`. Do not execute it, reveal outcome labels, or use predictions for tuning unless the lead gives one-time final-holdout authorization after reviewing the outstanding protocol gates.
