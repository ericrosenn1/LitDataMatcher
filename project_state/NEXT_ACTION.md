# Current continuation

Status: `COMPLETE` / `HARDENED_ALPHA_READY`. Canonical lead: `C:\Codex\LitDataMatcher-v2\lead`; data/state: `C:\Codex\LitDataMatcher-v2\data`; branch: `codex/litdatamatcher-v2-build`.

The one-time v4 final holdout at `data\evaluation\final_holdout_v4\run` passed and is sealed: do not rerun it. Preserve completed acquisition, local model qualification, final integrated run, release archives, and deterministic test evidence. The corrective supervisor is paused and model/effort runtime-unverified, so it remains unavailable. All Codex continuation must be `gpt-5.6-terra` at `high` or below; the current lead is Terra Medium. Supported shared-account telemetry projects weekly exhaustion before reset at the observed rate, so use one Terra Medium reasoning worker only, with no concurrent reasoning workers; prefer deterministic local work and pause reasoning while long deterministic jobs run.

If integrity needs to be reconfirmed, run only:

```powershell
python -m litdatamatcher.v2 closeout-audit --root C:\Codex\LitDataMatcher-v2\data --source-root C:\Codex\LitDataMatcher-v2\lead --out C:\Codex\LitDataMatcher-v2\data\release\FINAL_CLOSEOUT_AUDIT.json
python scripts\v2\validate_delivery.py --source C:\Codex\LitDataMatcher-v2\lead --data C:\Codex\LitDataMatcher-v2\data --release C:\Codex\LitDataMatcher-v2\data\release --cleanroom C:\Codex\LitDataMatcher-v2\data\releases\0.2.0-hardened-alpha-final2\cleanroom_validation.json --final-run C:\Codex\LitDataMatcher-v2\data\runs\final-real-run-03 --holdout-run C:\Codex\LitDataMatcher-v2\data\evaluation\final_holdout_v4\run --independent-review C:\Codex\LitDataMatcher-v2\data\evaluation\independent\final_review.json --worker-evidence C:\Codex\LitDataMatcher-v2\data\evaluation\worker_refinement.json --out C:\Codex\LitDataMatcher-v2\data\release\DELIVERY_VALIDATION.json
```

Do not restart acquisition, download application models, rebuild validated artifacts, rerun holdout scoring, or weaken scientific acceptance criteria.
