# V2.2 eligibility integration validation slice

Objective: prove at the `scientific_v2.rank_candidates` integration point that an explicit adapter modality mismatch remains disqualified even when its semantic relevance is maximal.

Change: added `test_maximum_semantic_score_cannot_rescue_adapter_contract_mismatch`. The synthetic ENA-shaped record declares `sequencing_genomics`; the requirement is `bulk_transcriptomics`; the supplied semantic score is `1.0`. The assessment is `INCOMPATIBLE` and `is_qualified` is false.

Validation: `python -m pytest tests/test_modality_contract.py tests/test_cross_source_adversarial.py -q` passed (`7 passed`). The cross-source deterministic receipt passed with input digest `2a24076aaad5bf442baeb8e0c427aac4c94ad089e3acee4ce66653d478a0a114`; it reports `wrong_modality=INCOMPATIBLE` and `technical_runs_not_donors=UNKNOWN`. Read-only alpha-baseline validation passed, including all six frozen artifact/manifest hashes; its derivative receipt SHA-256 is `512ba7471ca576ff88981bc4a7b12c2495b47dada8b974f34fadd9b125b4f4b4`.

Limitations: this is a local synthetic regression slice. It did not acquire data, run model inference, rerun the sealed holdout, or change the frozen hardened-alpha artifacts.

Resume: retain the current Terra Low single-worker policy until the required second Low telemetry reading. Any subsequent Phase 2 work must remeasure after material workload/profile changes and before exceeding the six-hour maximum interval.
