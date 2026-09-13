# Exact publication identifier reconciliation

`search_literature_sources` uses DOI exclusively when present, retaining the
existing DOI normalization and precedence. It does not join DOI-bearing and
DOI-absent records through secondary identifiers. Without DOI it indexes all
valid declared top-level `pmid` and `pmcid` values. PMID is the preferred display
key, but a PMCID-only record can match a record declaring the same PMCID and a
PMID. No identifiers are extracted from titles, abstracts or descriptive text.

PMID accepts positive integer values (not booleans) or ASCII decimal strings.
PMCID accepts strings containing `PMC` and a positive ASCII decimal accession.
Whitespace is trimmed, PMC case is normalized, and leading numeric zeroes are
removed. Blank, zero, negative, fractional, compound, list, dictionary and
non-ASCII-digit values do not form keys. Invalid fields do not negate another
valid explicitly shared identifier. With no valid publication identifier,
only an exact native `source_id` within the same `source` namespace can match;
records without that pair remain distinct, even with identical titles.

PubMed exposes PMID from its ESearch UID and PMCID from the existing EFetch
ArticleId or ESummary `articleids` entry of type `pmc`. Europe PMC exposes PMID
from the MED record ID and PMCID from its declared `pmcid` field. Other adapters
participate only through fields they explicitly supply.

Conflicting valid secondary identifiers are recorded in
`metadata.identifier_conflicts`. Without DOI, a conflict prevents merging;
a record connecting multiple existing groups remains separate with an explicit
ambiguity record. The check includes identifiers declared by earlier duplicates,
so a metadata-poor representative cannot bridge conflicting groups. Exact DOI
matches retain their existing merge behavior while disclosing secondary-ID
conflicts. This does not assert that conflicting source metadata is resolved.

The first record remains representative. Alternate native IDs, provenance,
complete alternate rows (`metadata.merged_source_records`), and source-scoped
version relations are retained. Source snapshots pair each identity with its
own provenance; missing provenance stays unknown. Merged records require source
review and cannot count as independent evidence. Primary and alternate lifecycle
relations both affect classification and derivation invalidation.

The output limit bounds representatives, while each requested source is still
queried within its existing per-source limit to retain later duplicate notices.
This bounded selection is not a source-completeness claim. Offline fixture tests
exercise the actual CLI, cache replay and adapters, including a one-record output
limit. The sealed scientific holdout is unrelated to these fixture tests.
