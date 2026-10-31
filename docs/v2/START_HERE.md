# LitDataMatcher v2 local alpha

LitDataMatcher 0.2.0 is a local, auditable question-to-data matching alpha. It
stores raw source snapshots outside Git, preserves exact claim spans, separates
unknown metadata from incompatibility, and labels ranking scores as uncalibrated
review priorities. Expert scientific calibration remains a separate status.

## Install

From a PowerShell prompt, create an environment and install the delivered wheel:

```powershell
py -3.12 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install .\litdatamatcher-0.2.0-py3-none-any.whl
& .\.venv\Scripts\litdatamatcher-v2.exe doctor --root C:\path\to\litdatamatcher-data
```

Local semantic analysis additionally needs PyTorch and Transformers. Use the
versions in `requirements-v2.lock`; model directories are external to the ZIP.
The tested extractor is Qwen2.5-7B-Instruct revision
`a09a35458c702b33eeacc393d103063234e8bc28`. The tested encoder is
all-MiniLM-L6-v2 revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.

## Run

The acquisition stages are independent and update one shared catalog:

```powershell
& .\.venv\Scripts\litdatamatcher-v2.exe sync --root C:\path\to\litdatamatcher-data --stage literature --expanded
& .\.venv\Scripts\litdatamatcher-v2.exe sync --root C:\path\to\litdatamatcher-data --stage datasets --expanded
```

Analyze a topic, local document, or explicit question. Every run needs a new
output directory. Requirements are JSON objects with `field`, `expected`,
Boolean `essential`, and `source_locator` fields.

```powershell
& .\.venv\Scripts\litdatamatcher-v2.exe analyze --root C:\path\to\litdatamatcher-data --out C:\path\to\runs\run-001 --model C:\path\to\Qwen2.5-7B-Instruct --embeddings C:\path\to\all-MiniLM-L6-v2 --topic primary --question "Which human studies have the required assay?" --requirements .\requirements.json --fresh --device cuda
```

Use `--document C:\path\to\paper.txt` for a new local document. Use `report
--run C:\path\to\runs\run-001` to rebuild its HTML view. The report escapes
source text and uses a restrictive content security policy.

## Validate and recover

Run the machine acceptance validator on a versioned evidence ledger:

```powershell
& .\.venv\Scripts\litdatamatcher-v2.exe acceptance --evidence C:\path\to\ACCEPTANCE_EVIDENCE.json --out C:\path\to\ACCEPTANCE_REPORT.json
```

The validator checks run schemas, current file sizes and SHA-256 digests,
successful retained commands, timestamps, required evidence kinds, coverage,
refinement, and readiness. Missing or stale evidence stays `NOT_RUN` or `FAIL`.

The controller provides checked resume and stop controls:

```powershell
& .\.venv\Scripts\litdatamatcher-controller.exe --root C:\path\to\controller preflight
& .\.venv\Scripts\litdatamatcher-controller.exe --root C:\path\to\controller run-next
& .\.venv\Scripts\litdatamatcher-controller.exe --root C:\path\to\controller pause
& .\.venv\Scripts\litdatamatcher-controller.exe --root C:\path\to\controller resume
& .\.venv\Scripts\litdatamatcher-controller.exe --root C:\path\to\controller stop
```

Never interpret semantic similarity as a probability or dataset metadata as a
claim of statistical power. Inspect exact requirements, source locators,
independent-unit status, dependence groups, and unresolved gaps in each dossier.
