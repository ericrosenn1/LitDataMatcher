# Installation and CLI

The primary research path is the local `litdatamatcher-v2` engine. Follow the
[research workflow](research_workflow.md) for source discovery, model inference,
typed question compilation, and recoverable batch runs. The commands below
also retain the deterministic reference pipeline for lightweight checks.

LitDataMatcher requires Python 3.10 or newer. The deterministic core does not
require a GPU, a private corpus, model weights, or live network access.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,corpus]"
```

Run the built-in offline demo:

```bash
litdatamatcher demo --out run/demo
litdatamatcher report --run-dir run/demo
```

To run a JSONL literature corpus, supply records containing some combination of
`title`, `abstract`, `text`, and `doi`:

```bash
litdatamatcher run --input test_input.jsonl --out run/full --top-n 100
```

Optional NLP and machine-learning extras are intentionally separate from the
core install. Their use does not change the requirement for provenance,
inspection, and expert review of ranked outputs.

For the local Transformers runtime and source-constrained decoding:

```bash
python -m pip install -e ".[runtime,structured]"
```

Model preparation is explicit, and runtime inference uses only verified local
files. Consult the [model guide](campaign_20261031/model_source_schema.md) for
identity, hardware, offline, and qualification details. Install `.[report]` to
recreate the research figures with Matplotlib.
