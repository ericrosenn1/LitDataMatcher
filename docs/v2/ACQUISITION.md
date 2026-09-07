# Real acquisition v2

Run from an installed environment (or the checkout):

```powershell
python -X utf8 scripts/v2/acquire_pilot.py --root C:/Codex/LitDataMatcher-v2/data --stage literature --expanded
python -X utf8 scripts/v2/acquire_pilot.py --root C:/Codex/LitDataMatcher-v2/data --stage datasets --expanded
python -X utf8 scripts/v2/acquire_pilot.py --root C:/Codex/LitDataMatcher-v2/data --stage all --expanded --offline
```

The two source stages can run concurrently because each owns its snapshot namespace and output files. Do not start two writers for the same stage/root. The lead supervisor owns that exclusion. Interrupted stages reuse verified snapshots; normalized JSONL is atomically replaced. `--refresh` deliberately revisits requests, retains previous immutable objects and updates only request indexes. `--offline` never constructs a network client and fails cache misses/corrupt snapshots explicitly. The CLI exits nonzero when requested coverage is not met.

The fixed primary query addresses human inflammatory-response transcriptomics; the reserved transfer query addresses gut/IBD transcriptomics. Both use publication dates2015–2024 to make bounded reproducible source selection. Queries, hit counts, selected rows and exclusions are saved before any quality assessment. Catalog discovery intentionally includes design mismatches: a mention of inflammation does not establish an intervention. The source-ordered sample profile budget is70% primary and30% transfer; studies above250 discovery samples are metadata-only for this bounded pass. Neither selection nor the acquisition coverage report declares scientific fit.

The initial live collection on2026-09-07 reached200 unique literature identities with50 JATS bodies parsed,127 GEO series,30 sample profiles, two processed numeric studies and37 ENA run metadata rows. Subsequent refinement/replay evidence is in the external catalog reports; those machine records supersede these initial counts. Literature DOI identity is case normalized with PMID/PMCID/source-ID fallbacks. JATS body paragraphs retain exact offsets into normalized body text and source XML snapshots. Table markup and figure artwork remain in raw XML; this parser does not claim scientific table extraction. Publication correction/version relationships are retained when provided.

Dataset records preserve GEO accession, series relations, BioProject/SRA aliases, source PMID links, sample attributes and processed matrices. Shared explicit aliases/related-series/sample IDs define dependence groups. ENA mirrors add zero studies. Unreported cohort reuse remains unknown, and the report distinguishes accession counts from known dependence groups. Source sample IDs are never interpreted as donors; only explicit donor/subject/patient ID attributes yield observed donor counts. Missing controls, pairing and independent units stay unknown. Group strings are taken from explicit characteristic keys, with column locators. Series summaries and titles do not silently establish usable contrasts.

Each parsed matrix is inspected for actual numeric cells, dimensions, finite values, unique feature IDs and exact equality of matrix sample columns to sample annotations. Empty RNA-seq series matrices count only as sample metadata, never as processed expression. Known missing numeric values are counted and retained. Numeric inspection does not establish biological validity or cross-study comparability.

All source data, permitted XML bodies, matrix snapshots and verbose acquisition events remain outside Git in the supplied root. A SHA256-addressed object is verified on every reuse. Sources may carry third-party rights; acquisition permission is distinct from redistribution permission. Do not copy these corpora into a source ZIP.

## Qualified reuse

The existing `literature_xml.py` parser was inspected and retained for legacy behavior. Its section traversal can duplicate nested paragraph content, so v2 adds a bounded JATS paragraph/span parser without rewriting legacy code. Existing adapters were reviewed; their cache contracts lack this worker's immutable byte snapshots and strict offline behavior, so v2 uses a narrow independent HTTP adapter. This is an in-project implementation, with no donor-platform code or prompts copied.

Adopted components: Python3.12 standard-library ElementTree/csv/gzip/hashlib for parsing and integrity; installed `requests` for HTTP. Exact environment versions are captured by the lead lock. Python code licensing and Requests Apache-2.0 licensing do not grant source-data redistribution rights. Cold mode retrieves bounded HTTP responses; warm mode verifies object hashes and performs local parsing. Live real-input tests executed EuropePMC search/fullTextXML, NCBI GDS search/summary, GEO SOFT/matrix and ENA file reports. The code needs no model/tokenizer downloads.

Official contracts checked during implementation:

- [Europe PMC REST API](https://europepmc.org/RestfulWebService): open-access fullTextXML and cursor paging.
- [NCBI GEO programmatic access](https://www.ncbi.nlm.nih.gov/geo/info/geo_paccess.html): series matrix layout and accession paths.
- [GEO SOFT specification](https://www.ncbi.nlm.nih.gov/geo/info/soft.html): series/sample characteristics.
- [ENA file report API](https://ena-docs.readthedocs.io/en/latest/retrieval/programmatic-access/file-reports.html): study/run metadata.

Cadmus and specialist platform adoption are deferred because these existing documented public routes satisfy the narrow acquisition contract without another framework or license surface. No OpenCode/Muse work or benchmark code is used.
