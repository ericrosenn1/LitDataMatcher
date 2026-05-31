"""Small cached HTTP client for optional live adapters."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1
import json
from pathlib import Path
import time
from urllib.parse import urlencode

import requests


@dataclass(slots=True)
class CachedHttpClient:
    """Requests-based client with disk cache, retry, and polite user agent."""

    cache_dir: Path | str = Path(".cache/litdatamatcher/http")
    timeout: float = 20.0
    min_interval_seconds: float = 0.2
    user_agent: str = "LitDataMatcher/0.1 research pipeline"

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_request = 0.0

    def _cache_path(self, url: str, params: dict | None) -> Path:
        """Return a stable cache path for a request."""

        query = urlencode(sorted((params or {}).items()), doseq=True)
        key = sha1(f"{url}?{query}".encode("utf-8")).hexdigest()
        return self.cache_dir / f"{key}.json"

    def get_json(
        self,
        url: str,
        params: dict | None = None,
        *,
        use_cache: bool = True,
        retries: int = 2,
    ) -> dict:
        """Fetch JSON with retry and cache support."""

        path = self._cache_path(url, params)
        if use_cache and path.exists():
            return json.loads(path.read_text(encoding="utf-8"))

        wait = max(0.0, self.min_interval_seconds - (time.time() - self._last_request))
        if wait:
            time.sleep(wait)

        last_error: Exception | None = None
        for attempt in range(retries + 1):
            try:
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
                return data
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                if attempt < retries:
                    time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"HTTP JSON request failed for {url}: {last_error}")
