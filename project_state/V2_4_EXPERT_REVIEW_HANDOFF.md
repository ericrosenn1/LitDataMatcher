# V2.4 expert-review machinery

Status: `PENDING_EXPERT_REVIEW`.

The project now has versioned blinded packet generation, strict categorical-label validation, descriptive inter-reviewer agreement, and adjudication-record generation. Packets retain question source spans, source identifiers, dataset provenance, organisms, assays, and access metadata. They exclude scores, ranks, predictions, heuristics, calibration fields, gold labels, and reviewer identities. Internal linkage maps blinded item IDs to match IDs and must not be distributed with packets.

Supported review dimensions are relevance, question validity, dataset compatibility, answerability, novelty, and evidence classification. The latter accepts the V2.3 relation vocabulary. Labels remain invalid if their reviewer/item/dimension identity duplicates an existing label, their packet item is unknown, or their categorical value is unsupported.

Run the deterministic infrastructure receipt with:

```powershell
C:\Codex\LitDataMatcher-v2\runtime-env\Scripts\python.exe scripts\v2\expert_review_packet_receipt.py --out C:\Codex\LitDataMatcher-v2\data\phase2\v2_4_expert_review\packet_receipt.json
```

This receipt uses a source-determined fixture only. It deliberately exports zero expert labels and therefore cannot establish calibration, agreement quality, adjudicated gold labels, or scientific validity.
