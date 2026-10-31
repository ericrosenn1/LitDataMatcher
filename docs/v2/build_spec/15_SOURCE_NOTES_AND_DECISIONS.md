# Source notes, verification and final decisions

Prepared 2026-09-07. This file distinguishes supplied evidence, current documentation, historical local clues, and the design decisions finalized for this build. External documentation must still be checked when a specific component is installed.

## Supplied project evidence

**U01. Earlier build task:** `LitDataMatcher_Multi_Agent_Build_Task_v1.md`, supplied in this conversation. The task's scientific pipeline and real-runtime requirements were reviewed. Its local-only/no-commit restriction is superseded by the user's later authorization for routine commits and pushes. It is deliberately not copied here as a competing active prompt.

**U02. Final auxiliary benchmark:** `LitDataMatcher_Auxiliary_Agent_Final_Benchmark_20260907_v2.zip`. The included decision and handoff say not to add Muse or OpenCode to the long-running build. The report notes net-supervision estimates, not directly comparable provider-token measurements. Selected verbatim source files are retained under `evidence/` with original-member hashes in `evidence/SOURCE_MANIFEST.json`.

The benchmark also preserves controller-observability limitations and optional disposable-drafting language. These do not reopen production routing. The user's finalized instruction is Codex-only development agents. A residual `muse_optional_for` field in the benchmark state is superseded; that state is not copied into this package's live-state template.

**U03. Local project clues:** earlier project context identified two candidate paths and the cleanup branch in file 12. These are not independently inspected local files in this chat. Codex must resolve the real current checkout and preserve newer changes. Historical hardware descriptions similarly require live inspection.

## Official sources checked while preparing this package

**S01. OpenAI, Subagents.**
https://learn.chatgpt.com/docs/agent-configuration/subagents
Documents native delegation and configurable local agents. Each agent consumes model/tool work. The package requests real delegation and capability discovery, not simulated reviewers. Exact setting names must be checked in the installed client.

**S02. OpenAI, Scheduled tasks.**
https://learn.chatgpt.com/docs/automations?surface=app
Documents local desktop project/worktree access with the computer and app running; web tasks cannot directly use a PC folder. CLI/IDE can prepare work but do not provide the Scheduled management interface. Hence preparation, test and confirmed activation are separate states here.

**S03. OpenAI, Long-running work.**
https://learn.chatgpt.com/docs/long-running-work
Documents Goal mode and explicit outcomes/completion criteria in supported local clients, plus worktree separation. This specification uses those principles without promising unlimited runtime or broader permissions.

**S04. cadmus license.**
https://github.com/biomedicalinformaticsgroup/cadmus/blob/main/LICENSE
The inspected file contains MIT licensing. File blob observed: `6c8675d0c0c09c1556805840890b26a2fc729ca0`. This is not a license audit of every dependency or retrieved article.

**S05. OptimusKG documentation.**
https://github.com/mims-harvard/OptimusKG/blob/main/README.md
The inspected README describes cached local Parquet access. File blob observed: `bfc7cf26de2a37ef84dad3e658db8b9e0890e952`. Qualify source data terms independently; the package does not assert graph superiority or transfer published speed to this computer.

**S06. Current remote LitDataMatcher.**
https://github.com/ericrosenn1/LitDataMatcher
https://api.github.com/repos/ericrosenn1/LitDataMatcher/commits/main
The inspected main commit was `48ef6580efccf55578dd865cb7154cfa34c5a872`. README blob: `be7ca5f674f119c5bd7b2a0e73ca0596971135ea`. Its documented reproducible package and older worker paths support a migration approach, but local newer work may supersede this remote snapshot. No remote modifications were made to prepare this package.

**S07. NCBI GEO download and format documentation.**
https://www.ncbi.nlm.nih.gov/geo/info/download.html
https://www.ncbi.nlm.nih.gov/geo/info/MINiML.html
https://www.ncbi.nlm.nih.gov/geo/info/soft.html
Official source material for accession/sample/series retrieval and machine-readable metadata. Qualify current endpoints and rate limits at runtime. These docs are integration references, not evidence that a particular study satisfies a question.

**S08. Biolink source provenance and association model.**
https://biolink.github.io/biolink-model/knowledge-source-retrieval/
https://biolink.github.io/biolink-model/Association/
Resource retrieval provenance and experiment/publication lineage are distinct. The proposed compiler should retain both. Biolink compatibility is a useful interoperability target; copying an ontology field is not a complete evidence-dependence model.

**S09. OpenAI pricing/model controls.**
https://learn.chatgpt.com/docs/pricing
https://developers.openai.com/api/docs/models/gpt-6-astra
Use supported client controls and observed account capacity, not guessed token multipliers. The API model's existence is not permission to spend on it. This package contains no new paid model/API budget.

## Design choices, not externally established findings

The ten-workstream decomposition; two alpha milestones; 50/20 and 200/50 literature coverage floors; 50/20 and 100/30 study/profile floors; two worker passes, three integration rounds; a local-first modular Python application; and an hourly corrective supervisor are finalized project requirements/design defaults. They are not literature-derived performance guarantees.

Initial pilot subjects and resource thresholds are configurable implementation defaults. Do not turn them into universal biological eligibility rules. Numeric release floors must not be reduced merely to make a failed release pass; any necessary scientific-scope revision is explicit and versioned.

Earlier unverified claims about interaction-finder licensing, every source's capabilities, automatic cross-chat control, or exact performance are not authority. Do not assume a donor is installed, licensed, current or beneficial without direct qualification.
