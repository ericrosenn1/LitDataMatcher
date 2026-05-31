# Reproducibility

LitDataMatcher is intended to support auditable computational research.

## Environment

Recommended setup:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional NLP and ML dependencies:

```bash
python -m pip install -e ".[dev,nlp,ml]"
```

## Deterministic Core

The default package pipeline is deterministic and offline:

- Stable IDs are content-derived.
- The default dataset catalog is bundled in code.
- JSONL input order is preserved where possible.
- Scores are deterministic functions of input text and normalized metadata.

## Recommended Run Record

For a publication run, archive:

- Git commit SHA
- Python version
- `pyproject.toml`
- Input literature JSONL
- Optional dataset catalog JSONL
- Output directory
- `litdatamatcher.sqlite`
- `metrics.jsonl`

## Validation

Run:

```bash
python -m pytest
python -m compileall litdatamatcher data_worker.py lit_gpu_worker.py matcher.py orchestrator.py
python -m litdatamatcher.cli demo --out run/demo
python -m litdatamatcher.cli report --run-dir run/demo
```

If optional NLP dependencies are not installed, the legacy analyzer falls back
to deterministic lexical matching for basic semantic checks.

## Data Governance

Before using a ranked pair in downstream analysis, review:

- Consent and license terms
- Protected or controlled-access fields
- Population compatibility
- Variable definitions and measurement units
- Missingness and batch effects
- Confounding and study design
- Statistical power
