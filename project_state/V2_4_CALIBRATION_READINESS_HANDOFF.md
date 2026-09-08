# V2.4 calibration readiness

`build_calibration_scorecard` is fail-closed. It retains only source-determined labels with provenance, a declared split family, finite score, and binary label. Pending expert labels emit `PENDING_EXPERT_REVIEW`; invalid denominators/provenance emit `NOT_CALIBRATED`. Metrics and ablation denominators are absent unless calibration is valid. Novelty, unresolvedness, and scientific significance are never calibrated.

Validation: six focused calibration/expert-review tests passed. Synthetic receipt PASS: `C:\Codex\LitDataMatcher-v2\data\phase2\v2_4_calibration_readiness\receipt.json`, SHA-256 `14ac6ece9b6e5b85fd91c7e4275143bbba01313863d01b9a7c9235411690c1ab`.
