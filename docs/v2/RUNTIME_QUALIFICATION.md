# Local runtime contract and qualification

The application uses local pretrained models through Transformers. Runtime calls
require an explicit local model directory with a verified `MODEL_MANIFEST.json`;
they cannot download missing weights or call a paid service. Model setup is an
explicit connected operation in `scripts/v2/runtime_download_models.py`.

Selected candidates are [Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct)
(Apache 2.0, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, model context 32,768 tokens)
and [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
(Apache 2.0, revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`).
Every downloaded model/tokenizer/config file has a recorded SHA-256 checked at
load. No model weights or downloaded source corpora belong in Git.

Qualified installation candidate: Windows native Python 3.12, `torch==2.11.0+cu128`,
`transformers==4.57.6`, RTX 5090, CUDA driver 610.74. CUDA tensor execution passed.
The private environment is `C:/Codex/LitDataMatcher-v2/runtime-env`; the lead's
shared environment was not modified. Dependency consistency passed `pip check`.
Suggested optional product dependency: `transformers==4.57.6` plus an explicitly
selected compatible PyTorch wheel. The core package remains importable without them.

```python
from litdatamatcher.semantic_runtime import (
    LocalSemanticRuntime, RuntimeConfig, PretrainedSemanticIndex,
)
runtime = LocalSemanticRuntime(model_dir, RuntimeConfig(device="cuda", dtype="bfloat16"))
result = runtime.extract(document, cache_root=cache_dir, force_fresh=True)
proposals = runtime.interpret_question(user_question)
index = PretrainedSemanticIndex(embedding_dir).fit(records)  # records: id, text
candidates = index.search(user_question, k=10)
```

`document` requires `document_id` and `text`; `title`, `sections` and
`source_provenance` are retained as input context. Extraction returns `claims`,
`questions`, `rejected`, and `inference_manifest`. Every accepted claim retains
its complete verbatim statement, exact character start/end, original provenance,
subject/predicate/object, direction, negation, source role, context and comparator.
Unknown context/comparator are null. Future-work text is a question origin, never
proof of novelty. Question gap state is initially `insufficient_coverage`.

The independent deterministic source guard rejects unsupported spans, omitted
sentence context, reversed relation order, paraphrased fields lacking exact source
support, negation mismatch, mixed or borrowed direction, unsupported comparators,
background/objectives promoted to current results, and instruction-like evidence.
This conservative bound deliberately abstains on complex/passive relations.
It does not establish expert-calibrated extraction quality. Full source text and
accepted statements remain the authority over downstream typed interpretations.

Generation is greedy, seeded, and bounded to 4,096 input and 1,024 output tokens,
with at most two attempts by default (three maximum). Oversized input fails clearly;
the caller must explicitly chunk it and preserve offsets. Valid partitions survive
repair attempts. Empty accepted output has `abstained_no_valid_records` status.
Diagnostics retain bounded actual generated text outside Git. Cache keys include
input, model manifest, config, prompt, schema, implementation and runtime-library
digests. Checksummed atomic cache writes distinguish replay from fresh inference.
Changing code while a process is qualifying correctly invalidates replay: freeze
the implementation for the qualification run.

Requirement interpretation returns eight source-explicit field proposals with
exact source locators. Unspecified fields remain null/unknown. `essential` is null
for every proposal: scientific indispensability is owned by explicit user contracts,
not inferred silently by a language model. The proposal interface does not validate
correct semantic field assignment and requires review before use as a hard filter.

MiniLM implements the model card's masked mean pooling and vector normalization,
using 16-item batches and a 256-token limit per retrieval text. Cosine scores are
not truth probabilities. Long-text truncation is declared in every search result;
the caller should use bounded title/metadata summaries or explicit chunks. This is
candidate retrieval, never a compatibility decision.

Run qualification from an installed package or set `PYTHONPATH` to the worktree:

```powershell
python scripts/v2/runtime_qualify.py --model-dir <extractor-directory> --embedding-dir <encoder-directory> --document <new-document.json> --output <qualification.json> --device cuda --question "Does rapamycin reduce IL6 in human macrophages?"
python -m pytest tests/test_semantic_runtime.py
```

The qualifier runs fresh model extraction, checks nonempty accepted records,
verifies a separate cached replay, and executes actual pretrained retrieval. It
blocks Python socket connection/DNS/send operations with an interpreter audit hook
before model initialization and verifies denial with an attempted connection.
This control does not claim OS firewall enforcement against arbitrary native
libraries; `local_files_only=True` additionally prevents model hub fallbacks.

Initial source-backed execution used Europe PMC PMID 42074327 / PMC13115588,
independent of the reserved evaluation families. Prompt v2.1 failed correctly:
the model generated unsupported paraphrases and no claims passed. A quote-first
prompt with an extractive example was developed using that development source.
The first implementation review added relation-specific direction verification,
explicit abstention, source-role/injection checks, cache corruption detection and
successful-partition preservation. All failure artifacts are retained outside Git.

Final executed qualification status and evidence paths are appended after the
unchanging-code qualification completes; installation/hardware alone is not G05 PASS.

## Continuation at execution-policy stop, 2026-09-07 07:43 EDT

The parent requested an immediate safe stop under the inherited Astra configuration;
the replacement is to use Terra High. No new qualification/refinement was started
after that request. Commit `edf4e43` is already integrated by the lead. The additional
downloader change selects Qwen2.5-7B-Instruct and downloads three distinct files in
parallel; it does not change runtime scientific interpretation.

Executed evidence: 28 focused tests and Ruff passed. The 1.5B candidate produced two
accepted real-source claims on PMID42074327 in 44.34 seconds in an actual process;
its cached replay and pretrained retrieval completed, but optional question proposal
validation caused the then-running qualifier to fail before writing an overall PASS.
The hardened proposal path preserves unsupported fields as explicit unknowns.
On source-selected PMID42455795, 1.5B reversed a negated predicate and omitted explicit
future work; the guard rejected all output (`abstained_no_valid_records`, 42.57 seconds).
This measured failure justified the second candidate. No final G05 qualification
report has passed, and the 7B model has not yet been executed.

All immutable source snapshots, normalized input documents, actual raw generated
diagnostics and checksummed extraction caches remain in
`C:/Codex/LitDataMatcher-v2/data/runtime-qualification`. Strong positive cached
artifact: `runtime-cache/7438c139bba60a5fe20810ecf6f9ff057f8534755b1be3dbf736f1255035a712.json`.
The two input documents are `qualification_document.json` (PMID42074327) and
`qualification_document2.json` (PMID42455795). These are development sources, not
held-out evaluation families.

Healthy deterministic download is intentionally left running in exec session `84269`.
The Windows venv launcher is PID 17832, created `2026-09-07T07:41:10.505189-04:00`;
its actual Python child is PID 63448, created `2026-09-07T07:41:10.579539-04:00`.
Working directory: `C:/Codex/LitDataMatcher-v2/worktrees/runtime`. Exact command:

```powershell
C:/Codex/LitDataMatcher-v2/runtime-env/Scripts/python.exe scripts/v2/runtime_download_models.py --root C:/Codex/LitDataMatcher-v2/data/models --kind extractor7b --revision a09a35458c702b33eeacc393d103063234e8bc28
```

Do not kill, restart or redownload while this process is healthy. Shards 2 and 3
were complete at the stop boundary (~3.864 GB each); shards 1 and 4 were still in
their bounded network requests. Success will write `MODEL_MANIFEST.json` under
`C:/Codex/LitDataMatcher-v2/data/models/Qwen2.5-7B-Instruct/a09a35458c702b33eeacc393d103063234e8bc28`.
Output is attached to exec session 84269; no separate downloader log file was created.

First continuation: collect downloader completion, verify its manifest, then run
the documented qualifier unchanged with the 7B directory, existing MiniLM directory
`C:/Codex/LitDataMatcher-v2/data/models/all-MiniLM-L6-v2/1110a243fdf4706b3f48f1d95db1a4f5529b4d41`,
and `qualification_document2.json` to a new `qualified_7b_pass1.json` output. Include
`--device cuda --question "Does rapamycin reduce IL6 in human macrophages compared with untreated controls?"`.
Freeze the module during qualification so source-code cache invalidation does not
turn the replay step into a second fresh run. Preserve every rejected attempt.
Only after actual fresh source-supported inference, replay, retrieval and proposal
checks pass should the new candidate be declared qualified.
