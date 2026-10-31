# v2 prospective evaluation and independent challenges

`split_reservations.json` and `docs/v2/EVALUATION_PROTOCOL.md` were committed before tuning. Reserved sources are not yet proven disjoint: connected publication/cohort/version closure is mandatory before final holdout claims. The primary and transfer holdouts have not been evaluated.

`challenge_contracts.json` contains eight source-selected families of checks with snapshot hashes and locators. Explicit constructed perturbations are identified separately from real source facts. The pytest cases exercise the lead's scientific contract implementation; they do not measure fresh-model semantic accuracy or meet real corpus coverage floors. Their labels are source-determined, not expert gold.

`source_qualification_manifest.json` records five successful live GEO SOFT captures and six bounded official repository/license inspections. Raw responses, full article XML, and publication metadata are retained in the configured external evaluation data root, not Git. `qualify_sources.py --output <external-directory>` reproduces bounded live qualification with three concurrent network workers and a 4 MB response ceiling. It records unavailable responses explicitly; HTTP retrieval success requires downstream format/semantic checking. E01 checked the five returned GEO records for their correct accession and expected series fields. Three donor repositories had affirmative MIT license contents; three returned 404 from the license endpoint. No donor library/code/prompt/weights were adopted, installed, or benchmarked. The reuse matrix records these limits instead of claiming an unexecuted comparison.

`E01_ROUND1.json` records 23 passing and five failing contract tests. The failing cases were delivered to the lead for repair: Boolean/integer conflation, unsupported absence, informative negative results, future evidence, and order-dependent evidence identity. Initial tests ran through the lead package using `PYTHONPATH=<lead-root>` and the absolute evaluation test path. No lead source was edited by the evaluator. These results are an actual independent review of the lead implementation, not approval of this evaluator's own fixtures.

Additional external snapshots used for source interpretation:

| Record | SHA256 | Locator |
|---|---|---|
| PMID31291584 core metadata | 841640eb65d960b7121ac3e91b6fbf550141d64443b6141a8febfa83c0bded16 | Europe PMC REST core result |
| PMID35821263 core metadata | 176deef0a752a79eb6462c23bf2ae20c37240cab34e46ae5df74725d698a1aef | Europe PMC REST core result |
| PMC6635384 JATS | c62881fa6513bbe063b51ff21038e7df76f0694470d8e703b28d56a0a7145eac | zero-based p index 42, IRG1 qualification |

GSE193336's linked publication has a murine arthritis title but includes human macrophage experiments. This is a source-backed warning against propagating one paper-title species to every deposited dataset. GSE95435's series summary describes a larger UC project, while its own overall design describes three healthy donors; superseries context must not invent diseased participants for this subseries.

One local console read failed because the Windows default encoding could not print Greek characters after the XML had already downloaded successfully. A bounded read using JSON ASCII escaping succeeded; no acquisition rerun was necessary. This technical repair did not alter sources or labels.
