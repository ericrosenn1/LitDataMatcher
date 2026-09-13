# Offline cache repair

Status: **PASS for this bounded repair; campaign acceptance remains lead-owned.**
Base: `98e6e95ad924caafdd1b7ec5c373e27aef57b923`.

`CachedHttpClient.post_file_text(offline=True)` previously reached
`requests.post` on a cache miss or when `use_cache=False`. It now raises
`FileNotFoundError` before any request. GET methods retain their existing
offline miss/bypass behavior, and JSON refresh remains unavailable offline.

The audit also found premature/stale response metadata and non-atomic cache
writes. Every public request method now clears metadata at entry. Cache hits
publish provenance only after decoding succeeds. Successful POSTs expose the
same cache provenance fields as GETs. Malformed JSON, invalid UTF-8, and file
read errors preserve the cache and raise their existing exceptions; they do
not implicitly trigger retrieval or publish a successful cache hit.

All response writes now use a same-directory temporary file. The complete
body, timestamp, and digest must be readable before atomic replacement. Only
then is response metadata published. Partial writes, metadata failures, and
replacement failures leave prior valid entries intact and do not leave a
partial response at a new cache key. Temporary files are removed on handled
failures. Explicit online JSON refresh continues to record the replaced digest.

| Request method | Offline hit | Offline miss or bypass | Corrupt cache | Failed cache write |
| --- | --- | --- | --- | --- |
| `get_json` | Replay JSON and provenance | Raise; refresh also raises | Preserve; raise without success metadata | Prior refresh entry survives |
| `get_text` | Replay text and provenance | Raise | Preserve; raise without success metadata | No partial cache entry |
| `post_file_text` | Replay text and provenance | Raise | Preserve; raise without success metadata | No partial cache entry |

## Validation

All execution used `C:\Codex\LitDataMatcher-v2\env\Scripts\python.exe`
(Python 3.12.13, pytest 9.1.1, requests 2.34.2), synthetic temporary files, and
mocked responses. No live service was contacted.

- Before the repair: **45 failed, 4 passed** in the 49 new regression cases.
- After the repair: **49 passed, 0 failed, 0 skipped** with identical test source.
- Existing adapter, GROBID, and identifier regressions: **65 passed, 0 failed,
  0 skipped**, under process-wide request/socket guards; guard calls: **0**.
- **114 distinct affected tests passed**. Focused Ruff and `git diff --check` pass.
- Failure injection covers partial body writes, prospective metadata hashing,
  and atomic replacement. The prior JSON entry remains replayable offline
  after each injected refresh failure.

Test commands:

```powershell
& 'C:\Codex\LitDataMatcher-v2\env\Scripts\python.exe' -B -m pytest tests/test_http_cache.py
& 'C:\Codex\LitDataMatcher-v2\env\Scripts\python.exe' -B 'C:\Codex\LitDataMatcher-v2\data\final_campaign_20260913\offline_repair\run_related_regression.py'
```

The second command is the preserved one-shot regression receipt runner; it
refuses to overwrite existing evidence. Its exact pytest arguments and result
are in `related_result.json`.

## Preservation and limits

Public signatures, request parameters, JSON serialization, and cache key
construction are unchanged. Regression fixtures replay literal pre-repair
filenames: GET key `e7137d05aeefd369d614ff629cc8e94a12384cf2` (`.json`/`.txt`)
and file POST key `ec5b64fb95bd6a9a4849a862effa42500de1cb17` (`.txt`). No existing
cache migration or scientific-output rebuild is required.

No acquisition, shared-corpus writes, frozen-alpha changes, sealed-holdout
reruns, scientific criteria changes, schedules, or pushes occurred in this
repair. Tests establish local request/cache behavior, not live API availability
or scientific validity. The legacy cache format has no independent persisted
expected digest: syntactically valid content changes require higher-level
validation. Atomic replacement is not a power-loss durability or multi-writer
locking guarantee; interruption before cleanup can leave an unused temporary
file. Existing request-key semantics, including multipart field-name omission,
remain unchanged as required.

External evidence is retained in:
`C:\Codex\LitDataMatcher-v2\data\final_campaign_20260913\offline_repair`.
It includes red/green JUnit and logs, related regression receipts, structured
task state, self-audit, source snapshots, and a verified compact handoff ZIP.
Lead integration, final independent review, full-campaign regression, and push
remain separate campaign steps.
