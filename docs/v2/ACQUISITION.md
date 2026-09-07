# Real acquisition v2

Run from an installed environment (or the checkout):

```powershell
python -X utf8 scripts/v2/acquire_pilot.py --root C:/Codex/LitDataMatcher-v2/data --stage literature --expanded
python -X utf8 scripts/v2/acquire_pilot.py --root C:/Codex/LitDataMatcher-v2/data --stage datasets --expanded
python -X utf8 scripts/v2/acquire_pilot.py --root C:/Codex/LitDataMatcher-v2/data --stage all --expanded --offline
python -X utf8 scripts/v2/acquire_pilot.py --root C:/Codex/LitDataMatcher-v2/data --stage integrate
python -X utf8 scripts/v2/acquire_pilot.py --root C:/Codex/LitDataMatcher-v2/data --stage audit-offline
python -X utf8 scripts/v2/acquire_pilot.py --root C:/Codex/LitDataMatcher-v2/data --stage targeted --accession GSE193336 --accession GSE128885
```

The two source stages can run concurrently because each owns its snapshot namespace and output files. An OS-backed per-stage lease rejects duplicate writers and automatically releases ownership if a process terminates. Interrupted stages reuse verified snapshots; normalized JSONL is atomically replaced through unique temporary files. `--refresh` deliberately revisits requests, retains previous immutable objects and updates only request indexes. `--offline` never constructs a network client and fails cache misses/corrupt snapshots explicitly. The CLI exits nonzero when requested coverage is not met. HTTP429/5xx retries honor numeric and HTTP-date Retry-After headers. A server-requested delay above the configured60-second worker wait produces `deferred_retry` with the next eligible timestamp, without making an early retry.

The fixed primary query addresses human inflammatory-response transcriptomics; the reserved transfer query addresses gut/IBD transcriptomics. Both use publication dates 2015–2024 to make bounded reproducible source selection. Queries, hit counts, selected rows and exclusions are saved before any quality assessment. Catalog discovery intentionally includes design mismatches: a mention of inflammation does not establish an intervention. The source-ordered sample profile budget is 70% primary and 30% transfer; studies above 250 discovery samples are metadata-only for this bounded pass. Neither selection nor the acquisition coverage report declares scientific fit.

The initial live collection on 2026-09-07 reached 200 unique literature identities with 50 JATS bodies parsed, 127 GEO series, 30 sample profiles, two processed numeric studies and 37 ENA run metadata rows. Refinement inspected all 127 series relationships, yielding 111 known dependence groups, 31 profiles (22 primary/9 transfer) and three numeric studies. The acquisition source floors pass; independent-cohort certainty and broader scientific quality are separate. Literature DOI identity is case normalized with PMID/PMCID/source-ID fallbacks. JATS body paragraphs retain exact offsets into normalized body text and source XML snapshots. Table markup and figure artwork remain in raw XML; this parser does not claim scientific table extraction. Publication correction/version relationships are retained when provided.

Dataset records preserve GEO accession, series relations, BioProject/SRA aliases, source PMID links, sample attributes and processed matrices. Shared explicit aliases/related-series/sample IDs define dependence groups. ENA mirrors add zero studies. Unreported cohort reuse remains unknown, and the report distinguishes accession counts from known dependence groups. Source sample IDs are never interpreted as donors; only explicit donor/subject/patient ID attributes yield observed donor counts. Missing controls, pairing and independent units stay unknown. Group strings are taken from explicit characteristic keys, with column locators. Series summaries and titles do not silently establish usable contrasts.

Each parsed matrix is inspected for actual numeric cells, dimensions, finite values, unique feature IDs and exact equality of matrix sample columns to sample annotations. Empty RNA-seq series matrices count only as sample metadata, never as processed expression. Known missing numeric values are counted and retained. Numeric inspection does not establish biological validity or cross-study comparability.

All source data, permitted XML bodies, matrix snapshots and verbose acquisition events remain outside Git in the supplied root. A SHA256-addressed object is verified on every reuse. Sources may carry third-party rights; acquisition permission is distinct from redistribution permission. Do not copy these corpora into a source ZIP.

## Qualified reuse

The existing `literature_xml.py` parser was inspected and retained for legacy behavior. Its section traversal can duplicate nested paragraph content, so v2 adds a bounded JATS paragraph/span parser without rewriting legacy code. Existing adapters were reviewed; their cache contracts lack this worker's immutable byte snapshots and strict offline behavior, so v2 uses a narrow independent HTTP adapter. This is an in-project implementation, with no donor-platform code or prompts copied.

Adopted components: Python 3.12 standard-library ElementTree/csv/gzip/hashlib for parsing and integrity; Requests 2.34.2 for HTTP and NumPy 2.5.3 for matrix alignment in the executed environment. Exact environment versions are captured by the lead lock. Python/Requests/NumPy code licensing does not grant source-data redistribution rights. Cold mode retrieves bounded HTTP responses; warm mode verifies object hashes and performs local parsing. Live real-input tests executed EuropePMC search/fullTextXML, NCBI GDS search/summary, GEO SOFT/matrix and ENA file reports. The code needs no model/tokenizer downloads.

Official contracts checked during implementation:

- [Europe PMC REST API](https://europepmc.org/RestfulWebService): open-access fullTextXML and cursor paging.
- [NCBI GEO programmatic access](https://www.ncbi.nlm.nih.gov/geo/info/geo_paccess.html): series matrix layout and accession paths.
- [GEO SOFT specification](https://www.ncbi.nlm.nih.gov/geo/info/soft.html): series/sample characteristics.
- [ENA file report API](https://ena-docs.readthedocs.io/en/latest/retrieval/programmatic-access/file-reports.html): study/run metadata.

Cadmus and specialist platform adoption are deferred because these existing documented public routes satisfy the narrow acquisition contract without another framework or license surface. No OpenCode/Muse work or benchmark code is used.

## Executed refinement evidence

The first worker pass established live acquisition and actual matrix inspection. The second fixed complete accession/series lineage, transfer-profile coverage, immutable replay, Retry-After handling, writer exclusion and real numerical alignment. Unit/fault tests cover source retry/defer, interrupted transfer, immutable refresh, corrupt/missing offline snapshots, size bounds, donor-vs-title safeguards, nonfinite/mismatched matrices, paragraph offsets, invalid joins and killed-writer lease recovery.

`--stage integrate` uses real GSE212865 values (27,189 features by137 samples). It creates engineered disjoint sample partitions, reverses feature order in one partition, aligns exact IDs and reconstructs all original values/missingness. Original and reconstructed numeric digests match. This is an executed same-study alignment demonstration, not independent replication, an effect-size analysis or a newly discovered biological finding. Duplicate samples, missing samples and a proposed cross-study GSE193185 combination without compatible source/units contracts are rejected. No observed measurements are imputed or renormalized.

`--stage audit-offline` blocks socket and Requests connections, deliberately interrupts both source stages after three verified inputs, resumes them and compares catalog bytes and identity counts. The executed audit made zero network attempts, produced byte-identical literature and study records and verified all271 immutable objects. Its measured wall time was25.23 seconds on the build host. Reports live in `catalog/acquisition_offline_recovery.json`, `catalog/numeric_integration.json`, and `catalog/source_object_manifest.json`; preserved connected acquisition reports live in `validation/acquisition_live_v2`. Later machine evidence supersedes these measured values.

Explicit accessions use `sync_targeted_studies` / `--stage targeted`, writing only `targeted_studies.jsonl` and `targeted_acquisition.json`. The default source-selected catalog and coverage denominator are unchanged. This route acquires sample profiles for seed accessions and follows linked series metadata within a 30-record family budget. Reserve labels propagate through explicit shared series, BioProject, sample and publication links. Known held-out families GSE112372, GSE214695 and GSE226875 remain reserved; acquisition is permitted but tuning on their linked sources is not. Missing undisclosed cohort/version relationships remain a limitation, not an assertion of complete independence. The five requested evaluator families were acquired live successfully; all five remained separate in the observed explicit lineage graph.
