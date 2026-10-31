# Rules and model extraction on one preserved source snapshot

The actual completed Qwen job `041f39b700814d92aff0a51867dea689` processed two
saved abstracts and retained one literal unresolved passage. Running the rules
extractor on exactly those saved documents retained the same passage and stable
question ID. Both outputs passed exact character-span checks. Both compilations
were `QUESTION_UNDERDETERMINED`; the model job produced zero dataset candidates.

This example provides no measured downstream benefit from model extraction.
It does not establish equal scientific utility across other documents: there
are no independent utility labels, and two documents are not a representative
benchmark. Literal fidelity is distinct from validity or unansweredness. The
historical model manifest retains its version, configuration and inference
timing; that timing is not treated as directly comparable to a fresh full run.

`scripts/campaign_20261031/compare_extraction_snapshot.py` executes the rules
comparison without loading a model, acquiring sources, or repeating dataset
qualification. It binds the saved job bytes and executed script by SHA-256,
preserves both outputs, and verifies source spans and question identities.
The receipt is `docs/validation/extraction-mode-comparison-20261031.json`.
The complete source and result files are in the separate external
`full_platform_20261031/extraction-comparison-v2` directory. An earlier run with
identical conclusions remains preserved in `extraction-comparison-v1`.
