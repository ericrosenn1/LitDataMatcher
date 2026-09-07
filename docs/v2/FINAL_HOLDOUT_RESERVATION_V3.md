# Final holdout reservation v3 (historical, consumed)

GSE264666 originally satisfied the v3 identifier-only rule. Its authorized
one-time execution then opened and scored the snapshot, but failed before a
result manifest was created because the runtime lacked `jsonschema`.

The consumption receipt is retained with SHA-256
`9c1245f7938b489ed148a018d2973c96fecc69ed823a18ba753108760c7885d9`.
The reservation is now `FAILED_CONSUMED_CONTAMINATED_DEVELOPMENT_EVIDENCE`.
It must never be rerun or claimed as an untouched final holdout.
