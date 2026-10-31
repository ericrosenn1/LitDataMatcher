# Reproducibility

The primary research workflow is the native `litdatamatcher-v2` engine:
source acquisition, local-model extraction, source verification, question
compilation, and dataset assessment. Reproduction requires the same source
snapshots, model files, code, configuration, and resolved dependencies—not just
the same query or random seed. See the [workflow and evidence contracts](research_workflow.md).

## Environment and execution paths

Use Python 3.10 or newer in a dedicated environment. After creating it with
`python -m venv .venv`, run `. .venv/bin/activate` on POSIX or
`.\.venv\Scripts\Activate.ps1` in PowerShell. Subsequent commands are the same:

```text
python -m pip install --upgrade pip
python -m pip install -e ".[dev,corpus,runtime,structured]"
python -m pip freeze > environment.txt
```

`.[dev]` is sufficient for core development. `corpus` adds Parquet support;
`runtime` adds PyTorch, Transformers, and safetensors; `structured` adds the
pinned LM-format-enforcer integration. `report` is optional for plotting.
Record the resolved versions and the platform-appropriate PyTorch build.
Optional-dependency test skips must be reported; a core-only test run does not
validate the native model path. The legacy `nlp`/`ml` extras are not substitutes
for native runtime qualification, and native inference does not silently fall
back to lexical extraction when a model is unavailable.

The core smoke test remains useful:

```text
litdatamatcher demo --out runs/illustrative-demo
```

Its synthetic papers and curated example metadata are illustrative. A successful
demo is an installation check, not research validation or proof of model quality.

## Freeze sources before inference

Edit a [bounded query plan](campaign_20261031/discovery.md), then acquire into a
new workspace:

```text
litdatamatcher-discover --plan examples/discovery-plan.json --out runs/study-workspace --workers 2
litdatamatcher-discover --plan examples/discovery-plan.json --out runs/study-workspace --offline --resume
```

Retain the exact plan, raw response snapshots, normalized catalogs, source
locators, checksums, timestamps, and `DISCOVERY_MANIFEST.json`. An empty result
differs from a failed request; neither establishes exhaustive coverage. A live
query can change later, so replay the recorded snapshots for comparison. Preserve
local document bytes, parent offsets, and the frozen batch input JSONL as well.
Source metadata alone does not establish available measurements or reuse rights.

## Prepare, qualify, and run the same backend

Model acquisition is an explicit connected step, separate from inference:

```text
python scripts/v2/runtime_download_models.py --root models --kind extractor7b --revision MODEL_COMMIT
python scripts/v2/runtime_download_models.py --root models --kind embedding --revision ENCODER_COMMIT
```

Replace the commit tokens with immutable 40-character revisions. Retain the
reported directories and `MODEL_MANIFEST.json` files, including every weight and
tokenizer checksum. Runtime loading uses verified local files; it does not fetch
missing weights. Follow the [model guide](campaign_20261031/model_source_schema.md)
and [literal decoder guide](campaign_20261031/model_source_literal.md).

The following template uses the reported `MODEL_DIR` and `ENCODER_DIR` paths.
It shows CUDA/bfloat16; CPU/float32 requires its own matching qualification.
Run the qualification for each declared positive development document, with a
separate new output path for each:

```text
python scripts/v2/runtime_qualify.py --model-dir MODEL_DIR --embedding-dir ENCODER_DIR --document inputs/positive-1.json --output runs/qualification-v1/doc1/QUALIFICATION.json --device cuda --decoding-mode source_literal --max-new-tokens 384 --require-claims
```

Under the [extraction protocol](campaign_20261031/extraction_protocol.md), both
positive documents must pass fresh inference and verified cache replay with at
least one accepted claim each. The current 7B source-literal profile passed those
two development checks and completed the 30-case follow-up with no retryable
technical failures. This is runtime qualification, not general scientific
accuracy. Retained baseline and failed
source-index results remain in the [extraction evidence](campaign_20261031/extraction_results.md).

Use the identical backend settings and an explicit matching receipt for a batch:

```text
litdatamatcher-v2 batch --root runs/study-workspace --inputs inputs/pipeline_inputs.jsonl --out runs/analysis-v1 --model MODEL_DIR --embeddings ENCODER_DIR --device cuda --decoding-mode source_literal --max-new-tokens 384 --runtime-qualification runs/qualification-v1/doc1/QUALIFICATION.json --fresh
```

For recovery, repeat that command with `--resume`. Verified terminal cases skip;
`--fresh` does not force them to rerun. Use a new output directory for an
independent fresh comparison or changed identities. See the [batch guide](campaign_20261031/native_batch.md)
for input limits, checkpoints, cache replay, and failure semantics.

The qualification wrapper denies listed Python socket/DNS operations and tests
its denial hook. This is not an OS firewall or proof about arbitrary native
libraries. Record any separate network isolation policy; discovery replay and
local model loading alone are not an operating-system network sandbox.

## Preserve the evidence and comparison boundary

Keep model revisions/checksums, source and input hashes, Git commit and dirty-tree
identity, packaged protocols, runtime configuration, dependency versions, Python,
device/dtype, and hardware details. Qualification and cache identities include
these relevant inputs. A changed backend requires a new matching receipt; do
not relabel an old receipt to accept a different environment.

Archive `RUN_MANIFEST.json`, `BATCH_STATE.json`, `BATCH_RESULT.json`, stage receipts,
accepted records, rejected/raw diagnostics, no-fit outcomes, and output checksums.
Preserve unsuccessful candidates and partial/empty results alongside successes.
Keep immutable inputs and validated outputs separate from new attempts; retain
quarantined derivatives and history instead of overwriting the original evidence.

Fixed weights and seeds do **not** guarantee bitwise GPU reproducibility across
hardware, kernels, drivers, precision modes, or library versions. Distinguish
verified cached replay from fresh generation and report observed differences.
Keep training, development, reserved evaluation, and sealed holdout data separate.
Once a case motivates a repair, its later result is development/regression
evidence, not an untouched test. See the [retrieval protocol](campaign_20261031/retrieval_protocol.md).

## Build and test the installable artifacts

From the checkout, use a new distribution directory:

```text
python -m pip install build
python -m pytest
python -m build --outdir dist/repro-v1
python scripts/validate_package.py --dist dist/repro-v1
```

Retain the wheel and sdist checksums, build log, environment record, and test
counts/skips. The package check expects one wheel and one sdist and verifies
required packaged resources; it does not establish scientific correctness.

Create a separate environment with `python -m venv .venv-wheel`, activate it,
leave the checkout, and clear any source-checkout `PYTHONPATH`. Install the exact
wheel path emitted by the build, replacing `WHEEL_PATH` below:

```text
python -m pip install WHEEL_PATH
python -m pip check
python -c "import litdatamatcher; print(litdatamatcher.__file__)"
litdatamatcher demo --out illustrative-installed-check
```

The printed module must come from the new environment, not the checkout. Repeat
in another new environment for the exact sdist path. For native installed tests,
install the corresponding runtime/structured extras from the same artifact,
then qualify and replay under that installed environment's identity. Model
weights and research data remain separately retained inputs, not package content.
Check the import origin in each test process: repository helper scripts
deliberately prepend the checkout and are not, by themselves, wheel-import tests.
