# Blank descriptive development review addendum

This additive packet covers the 12 new descriptive comparisons without changing
the frozen 42-case review packet. It contains a CSV worksheet, JSONL context,
separate machine-decision sidecar, protocol, existing label vocabulary, source
manifest, validation/state, essential generator and a verified ZIP.

Each row preserves the exact question, dataset, outcome/group columns and literal
group codes. Context contains the full compilation, source definitions, source
hashes, observed group counts and pointers to exact row indices in the existing
development report. Raw CSVs are not copied. All 15 reviewer fields remain empty
on every row (180 blank fields); no reviewer identity, label or adjudication is
invented.

The protocol distinguishes descriptive data feasibility from unansweredness and
novelty. It requires actual reviewers, source-definition/access interpretation,
study-specific independence and precision judgments when relevant, and dated,
coverage-bounded literature evidence for unansweredness/novelty. Development
positives do not imply expert acceptance. Independent scientific validation remains
a separate activity, and this worksheet is not untouched final evaluation.

The execution verified seven external source/evidence hashes and the unchanged
prior review ZIP hash. The generated archive was independently reopened to check
all manifest sizes/hashes, counts, exact case scopes and blank reviewer fields.
The small committed receipt records the versioned external output.

Run from the repository root with a fresh output directory:

```powershell
& 'C:\Codex\LitDataMatcher-v2\env\Scripts\python.exe' -m scripts.campaign_20261031.prepare_comparative_review `
  --comparison-root 'C:\Codex\LitDataMatcher-v2\data\full_platform_20261031\comparative-development-v4' `
  --frozen-review-zip 'C:\Codex\LitDataMatcher-v2\data\full_platform_20261031\scientific-review-v3\LitDataMatcher-development-scientific-review.zip' `
  --out 'C:\Codex\LitDataMatcher-v2\data\full_platform_20261031\comparative-review-new'
```
