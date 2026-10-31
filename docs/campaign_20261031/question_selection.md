# Extraction-first workflow and verified question selection

The scientific browser now follows literature search/source selection → **Find
questions** → source-question selection or explicit editing → typed requirement
preview → **Find datasets** → evidence and exports. Users can still enter their
own question directly. Existing API requests default to the full workflow.

`POST /api/jobs` accepts `workflow_mode: "questions"` for literature acquisition
and extraction only. This mode requires no dataset sources or inspection budget;
it does not read the dataset catalog, load a retrieval model, inspect datasets,
or run qualification. Dataset stages are absent from its checkpoint journal.
Metadata-only, failed, excluded, and empty source views retain their established
meanings. Model extraction remains explicit when requested.

The browser sends selected questions as
`question_reference: {job_id, question_id}` to both `/api/requirements` and
`/api/jobs`. It sends no citation, source claim, or trusted binding. The server:

- requires unique membership in a successful saved job and a unique source;
- reads result bytes matching the controller receipt and freezes that exact
  file identity, source document, source hash, and extracted question;
- validates literal offsets, text, native question ID, source locator, citation,
  provenance, and existing source lifecycle eligibility;
- preserves the original source version and checks any currently observed
  restrictive lifecycle metadata before the next workflow;
- reuses saved extraction before constructing a model runtime;
- compiles preview and execution with the same pinned source context. Context
  evidence points to the original document's exact offsets and text hash.

Unedited selections preserve the original question identity, evidence,
extraction method, model manifest, citation, and lifecycle metadata. Edited
wording is `USER_MODIFIED_SOURCE_QUESTION` / `USER_MODIFIED`, with
`source_stated: false` and `derived_from` containing the original question.
The edited wording has no fabricated source quote. Pasted text remains
`user_submitted_unverified`; verified receipt integrity does not authenticate
the scientific content. A user can clear the reference with **Use as a new
question**. Existing jobs and their successful artifacts are preserved.

Result reads, reference resolution, exports, and retry classification share
the same verified-byte reader. A file replacement between receipt validation
and reading is rejected. The frozen request does not reread its parent job
while executing or resuming.

The UI and dossier expose bounded addressing evidence as related result
passages needing review, with no definitive answer or global novelty claim.
**Pause jobs** describes the controller's existing behavior: active payloads
stop, queued execution pauses, and completed stages are retained.

## Targeted validation

- Ten Python selection regressions cover original and edited lineage, typed
  false preferences, contextual resolution and exact context locators,
  absent/tampered receipts and post-check byte replacement, wrong membership,
  ambiguous/missing source context, forged provenance, current retraction,
  frozen reuse without model inference, and extraction-only stage isolation.
- The existing 23 platform regressions pass; an independent read-only reviewer
  also ran the combined 33 tests and separately checked cross-job mismatches
  and retention of unverified pasted-source status.
- Current-bundle browser tests: eight passed across desktop and mobile, with
  two explicit live-computation tests skipped. They cover extract-first
  transitions, selected source references, edited wording, typed requirements,
  original-source links, addressing-evidence display, source selection,
  accessible queue controls, and an absent engine. Twenty-seven web library
  tests and the production build pass.

These checks establish software behavior, not scientific accuracy, novelty,
statistical adequacy, or clinical utility. Live reuse of the preserved Qwen job
is recorded separately below; no new model extraction was required.

## Executed integrated acceptance

The live API reused the real saved Qwen question
`rq_322307cc16134d272d375ce8` from job `041f39b700814d92aff0a51867dea689`.
Its original ID, citation, exact passage and model manifest were preserved;
preview and execution used the same source context. An edited-wording control
retained derivation without presenting the new wording as a source quotation.
The parent output hash and model-cache hashes and modification times remained
unchanged. A questions-only job produced no dataset outputs.

A separate actual browser walkthrough submitted **Find questions**, selected
that literal passage, previewed requirements, then submitted **Find datasets**.
The final result retained its verified source reference. Desktop and mobile
checks found no page errors or horizontal overflow. Saved extraction was reused
even with model mode selected; the underdetermined question retained zero fits.

Integrated validation at `2669cfa`: 1,491 Python tests passed with two platform
skips, 43 browser checks passed with one intentional skip, and 27 web-library
tests passed. A freshly installed wheel also served the frontend and completed
a real HTTP descriptive comparison with essential method evidence and UNKNOWN
statistical adequacy. The source-bound receipt is
[question-selection-live-20261031.json](../validation/question-selection-live-20261031.json).
