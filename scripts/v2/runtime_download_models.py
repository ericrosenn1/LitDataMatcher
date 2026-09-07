"""Explicit connected model preparation; never called by application inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

MODELS = {
    "extractor": "Qwen/Qwen2.5-1.5B-Instruct",
    "embedding": "sentence-transformers/all-MiniLM-L6-v2",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--kind", choices=list(MODELS), required=True)
    parser.add_argument("--revision", help="Optional immutable 40-character model commit")
    args = parser.parse_args()
    repo = MODELS[args.kind]
    if args.revision and not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("--revision must be an immutable 40-character model commit")
    endpoint = f"https://huggingface.co/api/models/{repo}" + (f"/revision/{args.revision}" if args.revision else "")
    with urllib.request.urlopen(endpoint, timeout=60) as response:
        info = json.load(response)
    revision = info["sha"]
    folder = args.root / repo.split("/")[-1] / revision
    folder.mkdir(parents=True, exist_ok=True)
    existing_manifest = folder / "MODEL_MANIFEST.json"
    if existing_manifest.exists():
        existing = json.loads(existing_manifest.read_text(encoding="utf-8"))
        if existing.get("model_id") != repo or existing.get("revision") != revision:
            raise ValueError("Existing model identity differs; preserving it")
        for item in existing["files"]:
            path = (folder / item["path"]).resolve(strict=True)
            if not path.is_relative_to(folder.resolve()):
                raise ValueError("Existing manifest path escapes model directory")
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != item["sha256"]:
                raise ValueError(f"Existing model corruption at {item['path']}; refusing to re-bless or overwrite")
        print(json.dumps({"status": "verified_existing", "model_dir": str(folder)}), flush=True)
        return
    allowed = {"config.json", "generation_config.json", "tokenizer_config.json", "tokenizer.json",
               "special_tokens_map.json", "vocab.json", "vocab.txt", "merges.txt", "LICENSE", "README.md",
               "model.safetensors", "model.safetensors.index.json", "sentence_bert_config.json"}
    files = []
    for entry in info["siblings"]:
        name = entry["rfilename"]
        if name not in allowed and not (name.startswith("model-") and name.endswith(".safetensors")):
            continue
        dest = folder / name
        if not dest.exists():
            temporary = dest.with_suffix(dest.suffix + ".partial")
            print(f"Downloading {repo}@{revision}/{name}", flush=True)
            with urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/{revision}/{name}", timeout=180) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
            temporary.replace(dest)
        with dest.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        files.append({"path": name, "bytes": dest.stat().st_size, "sha256": digest})
    manifest = {"model_id": repo, "revision": revision, "license": info.get("cardData", {}).get("license", "unknown"),
                "retrieved_at": datetime.now(timezone.utc).isoformat(), "files": files,
                "source": f"https://huggingface.co/{repo}/tree/{revision}"}
    (folder / "MODEL_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"status": "downloaded_and_hashed", "model_dir": str(folder), "files": len(files)}), flush=True)


if __name__ == "__main__":
    main()
