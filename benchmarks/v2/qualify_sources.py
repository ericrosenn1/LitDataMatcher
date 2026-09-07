"""Bounded official-source capture. Raw responses stay outside Git."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import urllib.error
import urllib.request
from datetime import datetime, timezone

REPOSITORIES = (
    "biomedicalinformaticsgroup/cadmus", "tecosaur/interaction_finder",
    "mims-harvard/OptimusKG", "mims-harvard/PrimeKG",
    "coledeisseroth/SNACKKSS", "coledeisseroth/SNACKKSS_NLP",
)
ACCESSIONS = ("GSE193336", "GSE128885", "GSE99787", "GSE133844", "GSE95435")


def capture(url: str, target: Path) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": "LitDataMatcher-v2-qualification/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            data = response.read(4_000_001)
            if len(data) > 4_000_000:
                raise ValueError("bounded response limit exceeded")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
            return {"url": url, "status": "CAPTURED", "http_status": response.status,
                    "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                    "path": target.name, "retrieved_at": datetime.now(timezone.utc).isoformat()}
    except (urllib.error.URLError, TimeoutError, ValueError) as exc:
        return {"url": url, "status": "UNAVAILABLE", "error": str(exc)}


def qualify_repository(repo: str, root: Path) -> dict:
    target = root / repo.replace("/", "__")
    result = {"repository": repo, "captures": []}
    for suffix, name in (("", "repository.json"), ("/commits?per_page=1", "commit.json"), ("/license", "license.json")):
        result["captures"].append(capture("https://api.github.com/repos/" + repo + suffix, target / name))
    if (target / "repository.json").exists():
        metadata = json.loads((target / "repository.json").read_bytes())
        result["license_spdx"] = (metadata.get("license") or {}).get("spdx_id", "UNKNOWN")
        result["default_branch"] = metadata.get("default_branch")
    if (target / "commit.json").exists():
        result["commit"] = json.loads((target / "commit.json").read_bytes())[0]["sha"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        repositories = list(pool.map(lambda r: qualify_repository(r, args.output), REPOSITORIES))
        sources = list(pool.map(lambda accession: {"accession": accession, **capture(
            "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=" + accession + "&targ=self&form=text&view=full",
            args.output / (accession + ".soft"))}, ACCESSIONS))
    summary = {"repositories": repositories, "sources": sources}
    (args.output / "qualification_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"repositories": [{k: v for k, v in r.items() if k != "captures"} for r in repositories], "sources": sources}, indent=2))


if __name__ == "__main__":
    main()
