# Development status

The current research campaign integrates the local Python engine, configurable
source discovery, structured extraction, typed requirements, model experiments,
and recoverable batch execution. The browser workspace is an optional review
surface. This is an active research preview, not a claim of expert-validated
scientific accuracy.

The initial integrated regression run passed 814 tests. The local fine-tuning
experiment passed its predefined comparison against the pretrained encoder,
while BM25 remained the stronger overall method on the small source-link
benchmark. The first repaired 7B extraction profile passed two fresh offline
smoke documents and verified replay. Further integration, larger retrieval pools,
and the frozen case pack are recorded in the ongoing campaign.

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
