# Consumed v3 holdout failure

The authorized `GSE264666` final-holdout attempt is recorded in
`C:\Codex\LitDataMatcher-v2\data\evaluation\final_holdout_v3\CONSUMED.json`.
Its SHA-256 is
`9c1245f7938b489ed148a018d2973c96fecc69ed823a18ba753108760c7885d9`.

The receipt records `FAILED_CONSUMED`, attempt
`c1496f5d-ef56-46bc-a303-27581b953fc0`, and
`ModuleNotFoundError: No module named 'jsonschema'`. The evaluator had opened
and scored the source snapshot before that dependency failure; no result
manifest was created. `GSE264666` is now contamination evidence for development
only and permanently ineligible for any untouched final-holdout claim.
