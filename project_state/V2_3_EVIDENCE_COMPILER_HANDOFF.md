# V2.3 evidence compiler tranche

Objective: attach auditable evidence-relation classification and exact lineage dependence grouping to the existing compiler without treating paper, repository, or knowledge-graph derivatives as independent confirmations.

Implemented scope: `evidence_relation_graph` records exact shared identifier and source-of-source lineage edges plus source-located declared classifications. Supported classifications are same underlying evidence, derivative evidence, duplicated cohort, replicated evidence, orthogonal evidence, direct perturbational evidence, associative evidence, mechanistic evidence, indirect evidence, contradictory evidence, incompatible evidence, and unknown dependence. Only same-underlying, derivative, and duplicated-cohort edges join known-dependence components. Distinct components retain `UNKNOWN` between-group independence.

Validation: targeted adversarial tests cover copied paper/repository/KG lineage, exact shared lineage, explicit classifications, and malformed relation assertions. The deterministic receipt command is:

```powershell
C:\Codex\LitDataMatcher-v2\runtime-env\Scripts\python.exe scripts\v2\evidence_compiler_contract_receipt.py --out C:\Codex\LitDataMatcher-v2\data\phase2\v2_3_evidence_compiler\contract_receipt.json
```

The executed receipt is `C:\Codex\LitDataMatcher-v2\data\phase2\v2_3_evidence_compiler\contract_receipt.json`; it reports `PASS`, one known-dependence component for the paper/repository/KG fixture, and no independent-support count.

Limitation: a graph can preserve only declared or exact identifier lineage. It never upgrades disconnected evidence to independent evidence or infers a scientific relation from semantic similarity.
