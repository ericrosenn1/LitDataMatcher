# Source-backed descriptive comparison development

Twelve new descriptive question/dataset pairs pass the full existing comparative
compiler contract, including essential `joint_observation` and
`outcome_ascertainment`. They reuse two actual public tables. These are development
data-feasibility positives, not independent studies, source-stated novel questions,
causal effects, statistically adequate analyses or independent validation.

The original twelve association questions remain frozen. Their compiled
`exposure_before_outcome` requirement has no supporting timing evidence. They
also lack explicit source-column bindings, so new pair-scoped evidence is not
silently applied. Replay leaves all twelve unresolved. No rule, threshold,
original question, source table, review packet or sealed holdout was changed.

## Sources

- [UCI Breast Cancer Wisconsin Diagnostic](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic): 569 records, with 357 B and 212 M diagnosis codes. Six questions compare literal `radius1`, `texture1`, `perimeter1`, `area1`, `smoothness1` and `compactness1` columns. The dictionary and nucleus-feature documentation are retained. Suffix `1` is not expanded into a statistic absent from the cached documentation.
- [UCI Heart Disease](https://archive.ics.uci.edu/dataset/45/heart+disease): 303 processed Cleveland records. Six questions compare `age`, `trestbps`, `chol`, `thalach`, `oldpeak` and `exang` across literal `num` codes 0–4. Counts are 164, 55, 36, 35 and 13. Codes are not recoded or interpreted as severity. Selected columns have complete joint observations; six missing cells in other columns remain recorded.

Original acquisition hashes of both CSVs and dictionaries are verified before
reuse. Each comparison retains source and metadata hashes, exact definition text
and JSON pointers, row indices, per-group co-observation counts, missing/nonfinite
cells and excluded-group counts. Definition-to-column mappings are explicit
source curation, visible for scientific review. Recorded ascertainment does not
imply clinical gold-standard adjudication.

## Runtime integration

`inspect_tabular_comparisons` adds `inspection.comparison_evidence` to an ordinary
inspected profile. It adds no global joint, ascertainment or temporal capability.
`comparison_constraints(outcome, grouping, values)` creates ordinary public
compiler user constraints naming columns and literal codes. The existing compiler
adds both comparative method necessities. The generator refuses contracts missing
either essential method requirement.

Before ranking, call `scope_tabular_comparison(dataset, requirements)` for each
candidate. Only exact stored scopes project method facts. Extra measurements,
alternate outcome/group codes, changed hashes, unbound requests and altered
evidence fail closed. Existing contradictions are preserved; reused projections
restore all exact original capabilities. Never cache a projected profile globally.

`requests.jsonl` contains ordinary question/constraint inputs;
`inspected_datasets.jsonl` is a local catalog. The browser need not accept source
claims or local inspection paths. Parent platform/API acceptance is a separate
stage from this generator's direct compiler/assessor execution.

## Validation and limits

Execution: 12/12 full-contract matches and 8/8 expected controls. Missing IL6 stays
`NOT_QUALIFIED`; timing, independence and unscoped method claims remain
`REQUIRES_INSPECTION`. Original association cases stay 0/12 qualified with unchanged
report hash. Sixteen scoped evidence tests plus inspection/compiler tests pass
(46 total). Row matches do not establish independence, power, confounder control
or novelty. No expert labels, model training, endpoint estimates or final evaluation
were created; the earlier scientific review packet remains frozen.

## Reproduce

Run from the repository root using a fresh output directory:

```powershell
& 'C:\Codex\LitDataMatcher-v2\env\Scripts\python.exe' -m scripts.campaign_20261031.demonstrate_tabular_comparisons `
  --cached-uci-root 'C:\Codex\LitDataMatcher-v2\data\campaign_20261031\processed-inspection-v1\uci-tabular' `
  --original-report 'C:\Codex\LitDataMatcher-v2\data\full_platform_20261031\dataset-qualification-v2\QUALIFICATION_DEMONSTRATION.json' `
  --out 'C:\Codex\LitDataMatcher-v2\data\full_platform_20261031\comparative-development-new'
```

The generator writes report, catalog, requests, state, handoff, essential code,
SHA-256 manifest and deterministic verified ZIP. The committed small receipt
identifies the validated external run. Next: exercise identical contracts through
the integrated platform and separately review scientific validity/unresolvedness.
