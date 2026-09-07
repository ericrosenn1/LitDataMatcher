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
