# Controlled organism mapping repair

The first real expanded benchmark exposed six false exclusions: the exact names
Arabidopsis thaliana, Drosophila melanogaster and Rattus norvegicus were absent
from the small qualified local organism table. The source records and benchmark
queries are unchanged. Exact names and the explicitly listed parenthetical labels
are now qualified. Arbitrary strains, abbreviations and fuzzy names remain unknown.

Actual primary-source JSON was acquired on 2026-09-13 into the derivative
`data/final_campaign_20260913/taxonomy_qualification` snapshot store:

| Source | Scientific name | SHA-256 |
| --- | --- | --- |
| https://rest.uniprot.org/taxonomy/3702.json | Arabidopsis thaliana | cba691f2d3f5b82d203fd3654990ad8dce522507b9e4f31980fce7f205ca018d |
| https://rest.uniprot.org/taxonomy/7227.json | Drosophila melanogaster | bdc24f80235b293c3db0a15651aca094a41ff776783cb72f8be0eb8c1b32201f |
| https://rest.uniprot.org/taxonomy/10116.json | Rattus norvegicus | 144cb51d3345001436aa2a659e2b0eee1d83519d33d6256472954b7fd9a192e1 |

This qualifies identifiers, not biological equivalence across species or cohorts.
The initial failed benchmark remains preserved; the correction requires a new
derivative evaluation against the unchanged queries and acceptance rules.
