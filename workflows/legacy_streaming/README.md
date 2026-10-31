# Legacy Streaming Workflow

This folder preserves the original subprocess-based streaming demo. It is kept
for provenance and possible future distributed-worker development, but it is not
the canonical production workflow.

Use `python -m litdatamatcher.cli ...` for reproducible runs. If this legacy
workflow is needed, run it from the repository root with:

```bash
python workflows/legacy_streaming/orchestrator.py --out run/orchestrator_matches.jsonl
```

The worker scripts were moved intact; no scientific behavior was intentionally
changed during the layout cleanup.
