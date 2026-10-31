# Native manifest contract alignment

The optional `question_compilation` field now declares the three counts already
written by native analysis: compiled, assessable, and underdetermined questions.
The field is a closed object and each count is a nonnegative integer. Existing
manifests without this optional field remain valid. Scientific classification
and acceptance criteria are unchanged.

Both native terminal-manifest writes validate against the packaged JSON schema
before atomic publication. The first write includes the already-created output
artifacts, satisfying the existing `PASS` contract before report rendering. The
second write inventories the complete publication. A stale or incompatible
schema fails the run instead of publishing an invalid success manifest.

The runtime schema uses LF line endings so its packaged byte count and SHA-256
remain stable across Windows and POSIX checkouts. The frozen build specification
is not modified.

All 30 retained baseline manifests validate against the aligned schema without
rewriting their bytes; their existing 9 `PASS` and 21 `PARTIAL` statuses remain.
This is a structural contract check, not a change to scientific interpretation.
