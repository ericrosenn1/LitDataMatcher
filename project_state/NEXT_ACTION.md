# Current continuation

Status: `RUNNING` / `HARDENED_ALPHA_COMPLETE`, current phase `PHASE_2`. Canonical lead: `C:\Codex\LitDataMatcher-v2\lead`; data/state: `C:\Codex\LitDataMatcher-v2\data`; branch: `codex/litdatamatcher-v2-build`.

The one-time v4 final holdout at `data\evaluation\final_holdout_v4\run` passed and is sealed: do not rerun it. Preserve completed acquisition, local model qualification, final integrated run, release archives, and deterministic test evidence. The corrective supervisor is paused and model/effort runtime-unverified, so it remains unavailable. All Codex continuation must be `gpt-5.6-terra` at `high` or below; the current lead is Terra Medium. Supported shared-account telemetry projects weekly exhaustion before reset at the observed rate, so use one Terra Medium reasoning worker only, with no concurrent reasoning workers; prefer deterministic local work and pause reasoning while long deterministic jobs run.

Frozen V2.0-HARDENED-ALPHA baseline: source commit `5747cbea2ae65c8570280d0e53f77bfabc968712`; final3 wheel/sdist and acceptance evidence are retained under `C:\Codex\LitDataMatcher-v2\data\releases\0.2.0-hardened-alpha-final3` and `C:\Codex\LitDataMatcher-v2\data\acceptance`. The Phase 2 objective is planning a bounded next milestone backlog while preserving this baseline and its scientific acceptance criteria. Do not implement broad Phase 2 work in this transition.

For Phase 2 planning, run only:

```powershell
python -m litdatamatcher.v2 doctor --root C:\Codex\LitDataMatcher-v2\data
git -C C:\Codex\LitDataMatcher-v2\lead status --short
```

Do not restart acquisition, download application models, rebuild validated artifacts, rerun holdout scoring, or weaken scientific acceptance criteria.
