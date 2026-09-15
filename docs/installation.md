# Installation and CLI

LitDataMatcher requires Python 3.10 or newer. The deterministic core does not
require a GPU, a private corpus, model weights, or live network access.

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
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
