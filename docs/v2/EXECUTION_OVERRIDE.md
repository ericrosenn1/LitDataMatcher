# Execution override, 2026-09-07
This later explicit user correction supersedes every earlier development-model selection, automatic effort escalation, and agent-concurrency instruction. The original verified launch package is unchanged. Scientific scope, acceptance criteria and safe checkpoint/push policy are unchanged.

- The model selected by the user in the main task is the lead: Sol High for the current development profile; Terra High when the user selects sustained lower usage. Do not override a later user selection.
- New implementation, acquisition, test and supporting-review subagents use `gpt-5.6-terra`, reasoning `high`. Initially at most two actively reasoning subagents plus the lead. Use compact scoped contexts, not full-conversation forks.
- With Sol lead, a bounded Sol High worker is allowed for a difficult scientific decision or persistent defect. With Terra lead, Sol requires an explicit assignment-specific user authorization.
- No automatic Astra, xhigh/Extra High, Max, Ultra, Fast, paid API, purchased credit, or alternate billing route. No global/unrelated configuration changes.
- Existing Astra workers must preserve state and stop at safe boundaries; verify stopped ownership before replacement uses their worktree. Healthy deterministic CPU/GPU/download/test jobs continue with recorded process identity. Do not restart acquisition or redownload valid models.
- This policy affects Codex development models only. Keep qualified application semantic models and their scientific validation requirements.
- Preserve meaningful tests/independent review; targeted tests during development, full integration at useful milestones; no unchanged repeated reviews or token-consuming polling. Use supported usage checks sparingly.
- Resume actual unfinished work and finish only at existing hardened-alpha and closeout gates. No restart, new build archive, architecture rewrite or auxiliary-agent benchmark.

Verified transition: current in-flight lead turn and old runtime/evaluation workers were gpt-6-astra/high in their turn contexts. Both old active workers saved and stopped. Native tools support explicit gpt-5.6-sol/high and gpt-5.6-terra/high for replacement turns. Saved global default reads Terra/high but does not prove the effective main-turn model; it was not changed. No matching project automation or custom project agent configuration existed. Supervisor remains PREPARED_NOT_ENABLED, no installed schedule.
