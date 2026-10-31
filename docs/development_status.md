# Development status

The full-platform campaign integrates the Python engine with the
[scientific browser workflow](full_platform.md): extract source questions,
preserve their citations when selected or edited, preview typed requirements,
discover and inspect datasets, assess compatibility, and export evidence.
This remains a research preview with independent scientific validation pending.

[Application acceptance at `2669cfa`](campaign_20261031/question_selection.md)
records 1,491 passing Python tests with two platform skips, 43 passing browser
checks with one intentional skip, 27 web-library tests, real source-selection
workflows and a fresh installed-package HTTP demonstration. Later corpus and
scale changes have separate targeted tests and revision-bound receipts.

The [100,000-abstract continuation](campaign_20261031/question_scale_100000.md)
retained 7,883 literal passages while reusing the completed first 10,000 inputs
and results. [Twelve descriptive comparisons](campaign_20261031/descriptive_comparisons.md)
pass full explicit contracts over two source-inspected tables; original
association cases remain unresolved. These observations do not establish
independent scientific accuracy, statistical adequacy or new research opportunities.

[Current PubMed materialization](literature_index.md) has its own explicit
coverage and integrity receipts; historical event totals are not counts of
unique current papers. `project_state/full_platform_20261031/STATE.json` records
the current campaign. The [scientific evaluation draft](campaign_20261031/scientific_evaluation_readiness.md)
preserves existing endpoints and identifies the independent judgments and final
protocol decisions still required.

## Earlier October 31 campaign evidence

That earlier integrated regression run passed 1,317 tests with two platform-specific
skips. The local fine-tuning experiment improved its pretrained parent on the
175-query development set, including 100,000-paper pools; BM25 remained the
strongest measured method. The source-literal 7B profile passed both positive
development documents, then completed all 30 development cases without a
retryable technical failure. All case outputs passed the artifact/source audit
and were reused unchanged on resume.

| Workstream | Earlier campaign evidence |
| --- | --- |
| Source discovery | 24 completed source jobs; 170 dataset and 56 literature metadata records; offline replay verified |
| Retrieval | Four large-pool conditions independently audited; source-dependent gains and negative findings retained |
| Extraction | 20 accepted surface-language claims; 11 PASS and 19 terminal PARTIAL cases; no expert accuracy estimate |
| Model size | The 1.5B literal configuration failed both positive documents and was not promoted |
| Dataset suitability | All 140 candidate rows require inspection; no scientifically qualified fit established |
| Recovery | Windows and Linux fault-injection checks cover worker admission, pause, checkpoints, and ownership |
| Corpus history | Full 1,621-shard event ledger and independent audit passed; final path repair reproduced both pilot Parquet files byte-for-byte |
| Distribution | Installable wheel and source archives; exact-revision build and installed-CLI evidence accompanies the final handoff |

See the [research workflow](research_workflow.md),
[retrieval protocol](campaign_20261031/retrieval_protocol.md),
[extraction protocol](campaign_20261031/extraction_protocol.md), and
[campaign specification](research_campaign_20261031.md).
`project_state/campaign_20261031/TASK_STATE.json` preserves that earlier campaign;
historical v0.3 and earlier v0.4 files retain their original scope.

Final validation is performed on the integrated revision. A passing software
check does not imply calibrated scientific validity. PyPI publication, DOI
registration, merging to the public default branch, and deployment are separate
release actions; none is implied by a local build or a draft pull request.
