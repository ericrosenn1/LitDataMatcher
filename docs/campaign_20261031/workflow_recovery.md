# Native workflow recovery

The native analysis now records durable extraction, question compilation, and
matching checkpoints. The scientific functions and their scoring rules are unchanged.
Run outputs and checkpoint metadata remain in the chosen output directory;
source data and model directories are read only.

## Run and resume

```powershell
python -m litdatamatcher.v2 analyze --root C:\research\data --out C:\research\runs\experiment-01 --model C:\models\extractor --embeddings C:\models\retriever --limit 3 --chunks 2
```

After interruption, repeat the exact command with `--resume`:

```powershell
python -m litdatamatcher.v2 analyze --root C:\research\data --out C:\research\runs\experiment-01 --model C:\models\extractor --embeddings C:\models\retriever --limit 3 --chunks 2 --resume
```

Preserve any original `--question`, `--requirements`, `--document`, `--topic`,
`--reference-accessions`, `--question-source-id`, `--device`, or `--fresh` options.
Changing inputs or parameters requires a new output directory. Legacy runs without
checkpoints are preserved and are not silently converted.

The first run writes a `RUNNING` manifest and `.workflow/state.json` before model
initialization. The journal records input/configuration, model-file, package-code,
rules/schema, and runtime-library fingerprints. Fingerprints are checked again
before accepting a completed execution. Hashing model files incurs real I/O;
this is deliberately stronger than relying on file timestamps.

Each successful source view is saved before processing the next view. If an
interruption occurs halfway through extraction, the next invocation reuses the
successful views. Complete extraction, compilation, and matching stages have
separate receipts with parent digests. A failed inference view remains retryable
on explicit resume; source-guard rejections remain recorded scientific outcomes.

Completed outputs are hash-checked before reuse. Publication damage is regenerated
from valid stage checkpoints. A corrupt stage is moved into `.workflow/quarantine`
and that stage and any changed dependent stages are recomputed. Nothing in a
source or model directory is removed. The run lock prevents simultaneous writers
to the same output directory and is released even if writing the journal fails.

## Bounded native job retries

The controller still executes one registered job while holding its integration
lease. Different scientific nodes can be registered as explicit commands; this
is a local controller, not a distributed scheduler. No background supervisor or
schedule is started by these changes.

Retry is opt-in and restricted to declared transient exit codes:

```python
controller.register(
    "catalog-partition-01",
    [python_executable, "partition_worker.py", "--partition", "01"],
    working_directory,
    ["partition-01.jsonl"],
    retryable_exit_codes=[75],
    max_retries=2,
    retry_backoff=10,
)
```

Call `run_next()` again after a `RETRY_PENDING` result. An invocation before the
retry time returns `WAITING_RETRY` with `next_eligible_at`. There are at most three
total attempts. Undeclared nonzero exits, timeouts, stale artifacts, and malformed
outputs remain failures. Pause/capacity handling retains its previous semantics.
Changing retry policy changes job identity and cannot silently alter a registered job.

JSON/JSONL outputs now reject malformed content, duplicate JSON keys, nonfinite
constants, numerical overflow, and non-object JSONL rows. CSV outputs must have
consistent row widths. Empty JSONL remains valid for scientific stages that
legitimately emit zero records, including controller-declared JSONL outputs.
Other controller-declared files must be nonempty. Missing files always fail.

## Validation and limits

The added tests use synthetic metadata and stub runtimes, with no downloaded
models or external APIs. They exercise actual local subprocesses for controller
retries and invalid-output failures. Native analysis tests compare the scientific
JSONL from interrupted/resumed and uninterrupted execution, inspect call counts,
and verify that successful inference views and matching results are reused.

Covered failures include interruption during extraction/compilation, corruption
of publication and matching checkpoints, model/input/configuration drift,
mid-run input mutation, duplicate ownership, journal-write failure, retry
exhaustion, and retry delay. These are engineering tests and establish no claim
about scientific accuracy, model quality, or real-model throughput.

Recovery deliberately fails closed when the fingerprint or checkpoint journal
cannot establish identity. Use a new output directory after an intentional
source/code/model change. Keep the old run for comparison; do not delete it to
make a stale run appear resumable.
