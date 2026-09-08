"""Write a deterministic receipt for successful cache refresh lineage."""
from __future__ import annotations

import argparse
import sys
import tempfile
from hashlib import sha256
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import litdatamatcher.http_cache as http_cache
from litdatamatcher.data_plane import atomic_json
from litdatamatcher.http_cache import CachedHttpClient


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        client = CachedHttpClient(cache_dir=directory)
        url, params = "https://fixture.invalid/records", {"page": 1}
        path = client._cache_path(url, params)
        path.write_text('{"revision":"old"}', encoding="utf-8")
        old_digest = sha256(path.read_bytes()).hexdigest()

        class Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {"revision": "new"}

        original_get = http_cache.requests.get
        http_cache.requests.get = lambda *unused_args, **unused_kwargs: Response()
        try:
            refreshed = client.get_json(url, params, refresh=True)
        finally:
            http_cache.requests.get = original_get
        replay = CachedHttpClient(cache_dir=directory, offline=True).get_json(url, params)
        metadata = client.last_response_metadata
    passed = (
        refreshed == replay == {"revision": "new"}
        and metadata.get("cache_status") == "live_refreshed"
        and metadata.get("replaced_cache_content_sha256") == old_digest
        and metadata.get("cache_content_sha256") != old_digest
    )
    atomic_json(
        args.out,
        {
            "schema_version": "v2_adapter_cache_refresh_receipt_v1",
            "fixture_scope": "synthetic/local cache refresh only",
            "old_content_sha256": old_digest,
            "new_content_sha256": metadata.get("cache_content_sha256"),
            "cache_status": metadata.get("cache_status"),
            "offline_replay": replay,
            "validation_status": "PASS" if passed else "FAIL",
        },
    )


if __name__ == "__main__":
    main()
