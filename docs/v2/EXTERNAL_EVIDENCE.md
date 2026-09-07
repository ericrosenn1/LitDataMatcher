# Bounded UniProt inflammatory reference evidence

The application imports three human UniProtKB cytokine records: TNF P01375, IL6 P05231, and IL1B P01584. This is a small structured reference panel, not comprehensive inflammatory knowledge. UniProt's official license endpoint identifies CC BY 4.0 for copyrightable database content; imported records retain attribution and the license URL. No upstream code is copied. Reactome's three attempted ContentService routes returned HTTP 403, so that optional source was deferred without bypass.

```python
from litdatamatcher.external_evidence import import_resource, query_resource
records = import_resource(data_root, offline=False)
context = query_resource(records, ["TNF", "IL6"], proposition_id=question["proposition_id"])
replayed = import_resource(data_root, offline=True)
```

Snapshots and manifests live under `<data_root>/external_evidence/uniprot`. Initial connected import captures the current license and three JSON entries; normal subsequent imports verify cached hashes. `refresh=True` requests a connected refresh. Offline mode raises a visible missing-cache/corruption error and never initiates a download. There are at most twenty explicit accession requests, two MB per response, three transient attempts, and bounded Retry-After handling. HTTP 404 becomes unavailable, never biological absence. A failed semantic refresh preserves the previously validated entry pointer. Atomic writes prevent partial JSON promotion; response checksums and accession/schema/license qualification are repeated on replay.

Each normalized record keeps source URI, content SHA256, retrieval date, UniProt release, entry revision, annotation date, exact gene names/synonyms, taxon, and function-comment JSON locators. Primary PubMed citations and ECO codes are retained separately from the UniProt aggregator. Comments can mix experimentally based statements and similarity-derived statements; the importer does not flatten those into experimental observations. Multiple cited PMIDs remain a list rather than being replaced with one convenient citation.

Queries use exact gene/identifier aliases, with case and hyphen normalization. They do not interpret arbitrary question prose or establish a gene-to-proposition relation. Results always have `proposition_id=None`, caller-linked `related_proposition_id`, `measurement_type=curation`, `scope_match=related`, `answers_question=False`, and `CONTEXT_ONLY_OR_UNRESOLVED`. Study/cohort lineage and between-experiment independence remain unknown. Compiler integration must connect all `primary_publication_ids` to corresponding paper lineage when present; unknown overlap cannot increase independent support. This resource cannot close a gap, establish controls or donor counts, or support numerical pooling.

Executed 2026-09-07: release 2026_03; TNF entry revision 283, IL1B 267, IL6 259. The actual import produced three records and six context items. Initial live elapsed time was 2.10 seconds; hash-verified offline replay was 0.033 seconds and identical. These are a single small workload observation, not a general performance estimate. Actual records, query evidence and validation JSON are retained under the external evaluation data root. Ten replay/failure tests passed, including HTTP404, bounded429 defer, corruption, missing cache, wrong species, context-only semantics, and preservation after schema drift. This evaluator authored those tests; they are self-validation, not independent approval of the module.

The broader independent scientific test suite was separately rerun against the lead's repaired implementation: 28/28 passed. The five first-round scientific defects are fixed on that tested working copy. Holdout and fresh-model extraction evaluation remain separate outstanding gates.

Sources: [UniProt license](https://www.uniprot.org/help/license), [machine-readable license](https://rest.uniprot.org/help/license), [UniProt API](https://www.uniprot.org/help/api_queries), [TNF record](https://rest.uniprot.org/uniprotkb/P01375.json).
