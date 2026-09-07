# Final holdout reservation v4

`GSE279879` is the first source-order primary family after consumed
`GSE264666` and metadata-exposed `GSE284624` that satisfies the complete
identifier-only rule. Its series, BioProject, GEO sample, PubMed, PMC, ENA
study, secondary-study, sample, and run identifiers have zero exact
intersections with the inherited comparison union, including GSE264666.

The candidate has linked PubMed `39489917`, PMC `11638867`, BioProject
`PRJNA1175029`, SRA `SRP539541`, and complete 14-row ENA relation evidence.
The GEO-keyed ENA query returned a successful explicit empty response. No
candidate relation errored or reached the 1,000-record boundary.

The v4 reservation is sealed. No GSE279879 title, summary, outcome, label,
rank, prediction, or score has been accessed. It remains
`RESERVED_PENDING_ONE_TIME_FINAL_HOLDOUT_AUTHORIZATION`; only an explicit lead
authorization may start its single final execution.
