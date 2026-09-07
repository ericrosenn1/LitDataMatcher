# Current continuation

Status: `RUNNING` / `HARDENED_ALPHA_COMPLETE`, current phase `PHASE_2`. Canonical lead: `C:\Codex\LitDataMatcher-v2\lead`; data/state: `C:\Codex\LitDataMatcher-v2\data`; branch: `codex/litdatamatcher-v2-build`.

The one-time v4 final holdout at `data\evaluation\final_holdout_v4\run` passed and is sealed: do not rerun it. Preserve completed acquisition, local model qualification, final integrated run, release archives, and deterministic test evidence. The corrective supervisor is paused and model/effort runtime-unverified, so it remains unavailable. All Codex continuation must be `gpt-5.6-terra` at `high` or below; the current lead is Terra Medium. Supported shared-account telemetry projects weekly exhaustion before reset at the observed rate, so use one Terra Medium reasoning worker only, with no concurrent reasoning workers; prefer deterministic local work and pause reasoning while long deterministic jobs run.

Frozen V2.0-HARDENED-ALPHA baseline: source commit `5747cbea2ae65c8570280d0e53f77bfabc968712`; final3 wheel/sdist and acceptance evidence are retained under `C:\Codex\LitDataMatcher-v2\data\releases\0.2.0-hardened-alpha-final3` and `C:\Codex\LitDataMatcher-v2\data\acceptance`. The first bounded V2.1-MULTISOURCE tranche is complete at source commit `3a3d734a5a6f2dd1580bcc7fa51d21515b08aa41`: optional Europe PMC and Crossref metadata adapters retain stable identifiers, provenance timestamps, cache snapshots, DOI-based cross-source relations, and fail-closed offline cache replay. Qualification is limited to one metadata record per source under `C:\Codex\LitDataMatcher-v2\data\phase2\v2_1_multisource`; it did not invoke acquisition or a full-text download.

For the next V2.1 continuation, first verify the frozen baseline and replay the bounded source cache:

```powershell
C:\Codex\LitDataMatcher-v2\runtime-env\Scripts\python.exe scripts\v2\validate_alpha_baseline.py --source C:\Codex\LitDataMatcher-v2\lead --data C:\Codex\LitDataMatcher-v2\data --out C:\Codex\LitDataMatcher-v2\data\phase2\v2_1_multisource\alpha_baseline_non_regression.json
C:\Codex\LitDataMatcher-v2\runtime-env\Scripts\python.exe -m litdatamatcher.cli literature-search --query microbiome --source europepmc --limit 1 --cache-dir C:\Codex\LitDataMatcher-v2\data\phase2\v2_1_multisource\http_cache --offline --out C:\Codex\LitDataMatcher-v2\data\phase2\v2_1_multisource\continuation_europepmc_offline.jsonl
C:\Codex\LitDataMatcher-v2\runtime-env\Scripts\python.exe -m litdatamatcher.cli literature-search --query microbiome --source crossref --limit 1 --cache-dir C:\Codex\LitDataMatcher-v2\data\phase2\v2_1_multisource\http_cache --offline --out C:\Codex\LitDataMatcher-v2\data\phase2\v2_1_multisource\continuation_crossref_offline.jsonl
```

Do not restart acquisition, download application models, rebuild validated artifacts, rerun holdout scoring, or weaken scientific acceptance criteria.
