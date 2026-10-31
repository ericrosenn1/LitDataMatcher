# V2.5 bounded scale/recovery instrumentation

The V2.5 runner measures a declared synthetic derivative fixture only. It records SQLite FTS catalog ingestion and query timing, deterministic matching and evidence-compilation timing, peak Python allocation, written disk bytes, cache replay, and reopen/resume after an interrupted half-index. It records host platform, Python version, CPU count, free disk, backends, limits, input digest, and a machine-readable PASS/FAIL receipt.

Run:

```powershell
C:\Codex\LitDataMatcher-v2\runtime-env\Scripts\python.exe scripts\v2\run_local_scale_benchmark.py --root C:\Codex\LitDataMatcher-v2\data\phase2\v2_5_scale\benchmark_workspace --out C:\Codex\LitDataMatcher-v2\data\phase2\v2_5_scale\benchmark_receipt.json --count 32
```

The executed receipt is `C:\Codex\LitDataMatcher-v2\data\phase2\v2_5_scale\benchmark_receipt.json`. It is bounded to 32 synthetic records, does no network or model inference, and makes no production-scale or scientific performance claim. Malformed receipts and counts outside 1..1000 are rejected.
