# V2.1 adapter cache-refresh contract slice

Objective: enable a safe incremental refresh of one optional-adapter JSON cache entry, preserving deterministic offline replay and exact payload lineage.

Implementation: `CachedHttpClient.get_json(..., refresh=True)` skips a cache hit, fetches normally, and replaces the cache file only after a successful response. Provenance metadata now carries the current `cache_content_sha256`; a replacement also carries `replaced_cache_content_sha256` and status `live_refreshed`. Offline refresh fails closed, and a failed live refresh leaves the existing entry intact.

Validation: `python -m pytest tests/test_adapters_and_capabilities.py -q` passed (`21 passed`). It includes successful refresh/new offline replay and failed-refresh/prior offline replay cases. `python -m compileall -q litdatamatcher/http_cache.py scripts/v2/adapter_cache_refresh_receipt.py` passed. The deterministic receipt is PASS at `C:\Codex\LitDataMatcher-v2\data\phase2\v2_1_cache_refresh\receipt.json`, with old SHA-256 `cc9e53ce877e17a6cf97e4d9f37d48c7ea3b9e057ccfd49f527524f6722a58a5` and new SHA-256 `3214343a42cffe880dd19ece61d2a9539cb596ce36db6d26b8e0f32afc76fb6d`.

Limitation: this is a synthetic local cache contract. It does not fetch a page, enumerate a source, imply completeness, change pagination, or perform bulk acquisition. Frozen alpha artifacts and the sealed holdout were not read or modified.
