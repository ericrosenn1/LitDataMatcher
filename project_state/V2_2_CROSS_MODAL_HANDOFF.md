# V2.2 cross-modal contract

Local proteomics/metabolomics fixtures now retain feature type/unit, quantification, normalization, and metadata availability. Missing fields remain `UNKNOWN`; explicit transcript/protein/metabolite feature or normalization mismatch is ineligible; technical runs never become biological units. This is not repository qualification.

Validation: 39 focused modality/scientific tests passed. Cross-modal receipt PASS at `C:\Codex\LitDataMatcher-v2\data\phase2\v2_2_cross_modal\receipt.json`. Read-only frozen-alpha guard passed all six artifact hashes without rerunning holdout.
