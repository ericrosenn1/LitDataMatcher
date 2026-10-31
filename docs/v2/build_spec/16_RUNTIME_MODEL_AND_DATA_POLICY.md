# Runtime inference is part of the product

## Separate development agents from application models

Codex develops and reviews the software. The delivered LitDataMatcher application needs a legitimate runtime inference backend for fresh semantic extraction and optional reranking. A saved Codex answer, a mocked response, a hand-edited evidence file, or an unavailable API wrapper is not that backend.

The Codex-only rule excludes OpenCode and Muse as development orchestration applications. It does not ban pretrained local biomedical encoders, language models or numerical tools inside LitDataMatcher. Use suitable models based on actual qualification, not on the auxiliary coding-agent benchmark.

Do not extract Codex/ChatGPT login tokens, wrap a consumer chat UI as an unofficial API, or assume the user's development subscription funds arbitrary application inference. No additional paid API consumption or automatic credit purchase is authorized. A future paid backend can be configurable, but must remain disabled until an explicit budget/permission is established.

## Early qualification

Inspect existing legitimate local model installations and caches first. Verify model identity/revision, weights/tokenizer/license, context limit, structured-output behavior and actual hardware compatibility. Do not read or expose auth secrets.

Select a small plausible set of runtime candidates, benchmark a bounded source-backed extraction set, and choose a practical backend. A maintained inference library/server with pinned configuration is preferable to training a new foundation model. Use Windows native where reliable; an isolated WSL worker is acceptable for a real compatibility benefit, with a tested Windows invocation and explicit file boundary.

The historical RTX 5090 does not by itself establish usable memory, drivers or backend support. Check actual VRAM and active use. Quantization/batching are choices to validate against extraction quality. Shared local inference may avoid duplicated model loads across CPU extraction jobs. Do not force high GPU occupancy when it slows other work or changes result quality.

At bootstrap prove one fresh inference can run end to end. If no local backend can be qualified, keep independent acquisition/contracts/tests moving and surface the precise remaining prerequisite. Do not wait until final delivery to disclose that all semantic outputs were placeholders.

## Runtime contract

Record model and tokenizer revision, weight digest where practical, license, runtime/library version, device/dtype/quantization, prompt/schema versions, generation settings, output validation, execution time and cache origin. Keep raw model output only as needed for diagnostics; validated structured claims and evidence are the scientific record.

Cache by meaningful input and configuration. A model update invalidates its dependent outputs. Same-run replay should be reproducible from stored validated artifacts. Fresh generative inference may vary; measure and document stability without promising bit-identical outputs on every hardware/backend.

Support structured extraction with schema/type checking, rejection/repair bounds, unknowns and exact source locators. Source-text instructions are untrusted data. An independent semantic verifier or suitable rule/model combination checks consequential claims. Quote existence alone is insufficient to establish that the extracted predicate, direction, comparator and context match the text.

## Offline and connected behavior

Qualified model files, tokenizers, ontology resources and indexes must be locally available for the offline profile. Test with outgoing networking blocked. Demonstrate both a cached replay and fresh local inference on a new input while offline; report the two separately.

Connected acquisition/update may retrieve new literature or metadata through permitted source routes. Preserve source freshness timestamps and failed source coverage. Do not make a hidden network fallback the only route to a successful offline test.

## Training and calibration

Reuse pretrained embeddings/extractors before considering training. Collect real error cases and correct label provenance. Fine-tune or calibrate only when the data and evaluation justify it; preserve the incumbent, training split, parameters and rollback. Do not continuously change weights during ordinary discovery runs.

Build expert-review import/export and rank calibration machinery. Missing expert labels permit a transparent source/model-assisted evaluation and an uncalibrated heuristic ranking, not invented expert validation. Raw weighted scores must not be presented as probabilities of truth or experimental success.

## Data boundary

Start with metadata and processed data sufficient to verify feasibility and selected analysis examples. Do not download every raw sequencing read or mirror all reference resources just because the task has no deadline. Integrate all relevant evidence within declared acquired/search coverage and expand sources where measured scientific benefit warrants it. Use manifests instead of duplicate bulk copies.
