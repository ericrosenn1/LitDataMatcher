"""Small cached HTTP client for optional live adapters.

Cached responses improve reproducibility for local workflows; they do not prove
that a live API is currently reachable or unchanged.
"""

from __future__ import annotations

import json
import mimetypes
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha1, sha256
from pathlib import Path
from urllib.parse import urlencode

import requests


@dataclass(slots=True)
class CachedHttpClient:
    """Requests-based client with disk cache, retry, and polite user agent."""

    cache_dir: Path | str = Path(".cache/litdatamatcher/http")
    timeout: float = 20.0
    min_interval_seconds: float = 0.2
    user_agent: str = "LitDataMatcher/0.1 research pipeline"
    offline: bool = False
    _last_request: float = field(default=0.0, init=False, repr=False)
    last_response_metadata: dict[str, str] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_path(self, url: str, params: dict | None, suffix: str = ".json") -> Path:
        """Return a stable cache path for a request."""

        query = urlencode(sorted((params or {}).items()), doseq=True)
        key = sha1(f"{url}?{query}".encode()).hexdigest()
        return self.cache_dir / f"{key}{suffix}"

    def _wait_for_rate_limit(self) -> None:
        """Pause just enough to respect the configured request interval."""

        wait = max(0.0, self.min_interval_seconds - (time.time() - self._last_request))
        if wait:
            time.sleep(wait)

    def get_json(
        self,
        url: str,
        params: dict | None = None,
        *,
        use_cache: bool = True,
        retries: int = 2,
    ) -> dict:
        """Fetch JSON with retry and cache support."""

        path = self._cache_path(url, params, suffix=".json")
        if use_cache and path.exists():
            # Cached/mock payloads are reproducible fixtures, not live validation.
            self._record_cache_response(path, "cache_hit")
            return json.loads(path.read_text(encoding="utf-8"))
        if self.offline:
            raise FileNotFoundError(f"offline cache missing: {url}")

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                self._wait_for_rate_limit()
                response = requests.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                    headers={"User-Agent": self.user_agent},
                )
                self._last_request = time.time()
                response.raise_for_status()
                data = response.json()
                if use_cache:
                    path.write_text(json.dumps(data, sort_keys=True), encoding="utf-8")
                    self._record_cache_response(path, "live_cached")
                return data
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"HTTP JSON request failed for {url}: {last_error}")

    def get_text(
        self,
        url: str,
        params: dict | None = None,
        *,
        use_cache: bool = True,
        retries: int = 2,
    ) -> str:
        """Fetch text/XML with retry and cache support."""

        path = self._cache_path(url, params, suffix=".txt")
        if use_cache and path.exists():
            # Text cache covers XML and TEI intermediates as well as plain text responses.
            self._record_cache_response(path, "cache_hit")
            return path.read_text(encoding="utf-8")
        if self.offline:
            raise FileNotFoundError(f"offline cache missing: {url}")

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
                self._wait_for_rate_limit()
                response = requests.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                    headers={"User-Agent": self.user_agent},
                )
                self._last_request = time.time()
                response.raise_for_status()
                text = response.text
                if use_cache:
                    path.write_text(text, encoding="utf-8")
                    self._record_cache_response(path, "live_cached")
                return text
            except requests.RequestException as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"HTTP text request failed for {url}: {last_error}")

    def _record_cache_response(self, path: Path, cache_status: str) -> None:
        """Expose a stable cached-response timestamp for adapter provenance."""

        timestamp = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(
            timespec="seconds"
        ).replace("+00:00", "Z")
        self.last_response_metadata = {
            "cache_path": str(path),
            "cache_status": cache_status,
            "retrieval_time_utc": timestamp,
        }

    def post_file_text(
        self,
        url: str,
        file_path: str | Path,
        *,
        field_name: str = "input",
        data: dict | None = None,
        use_cache: bool = True,
        retries: int = 0,
    ) -> str:
        """POST a local file and cache the returned text by file digest and form data."""

        file_path = Path(file_path)
        payload_data = {key: str(value) for key, value in (data or {}).items()}
        digest = _file_sha256(file_path)
        cache_params = {"file_sha256": digest, "file_name": file_path.name, **payload_data}
        path = self._cache_path(url, cache_params, suffix=".txt")
        if use_cache and path.exists():
            return path.read_text(encoding="utf-8")

        last_error: Exception | None = None
        mime_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        for attempt in range(retries + 1):
            try:
                self._wait_for_rate_limit()
                with file_path.open("rb") as handle:
                    response = requests.post(
                        url,
                        files={field_name: (file_path.name, handle, mime_type)},
                        data=payload_data,
                        timeout=self.timeout,
                        headers={"User-Agent": self.user_agent},
                    )
                self._last_request = time.time()
                response.raise_for_status()
                text = response.text
                if use_cache:
                    path.write_text(text, encoding="utf-8")
                return text
            except requests.RequestException as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"HTTP file POST failed for {url}: {last_error}")


def _file_sha256(path: Path) -> str:
    """Return a stable digest for cache keys involving local files."""

    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
