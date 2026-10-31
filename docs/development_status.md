# Development status

The current research campaign integrates the local Python engine, configurable
source discovery, structured extraction, typed requirements, model experiments,
and recoverable batch execution. The browser workspace is an optional review
surface. This is an active research preview, not a claim of expert-validated
scientific accuracy.

The latest integrated regression run passed 1,317 tests with two platform-specific
skips. The local fine-tuning experiment improved its pretrained parent on the
175-query development set, including 100,000-paper pools; BM25 remained the
strongest measured method. The source-literal 7B profile passed both positive
development documents, then completed all 30 development cases without a
retryable technical failure. All case outputs passed the artifact/source audit
and were reused unchanged on resume.

| Workstream | Current evidence |
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
`project_state/campaign_20261031/TASK_STATE.json` records current machine-readable
status; historical v0.3 and earlier v0.4 files retain their original scope.

Final validation is performed on the integrated revision. A passing software
check does not imply calibrated scientific validity. PyPI publication, DOI
registration, merging to the public default branch, and deployment are separate
release actions; none is implied by a local build or a draft pull request.
