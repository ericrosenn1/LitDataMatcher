"""Verify the frozen V2.0 hardened-alpha artifacts before Phase 2 work."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

ALPHA_BASELINE_COMMIT = "5747cbea2ae65c8570280d0e53f77bfabc968712"
EXPECTED_ARTIFACTS = {
    "final3_wheel": (
        "releases/0.2.0-hardened-alpha-final3/dist/litdatamatcher-0.2.0-py3-none-any.whl",
        "af2a6b6265aa89cad4ac0935bec34dbc3cc8d945c44c941ef9e3341d1794a389",
    ),
    "final3_sdist": (
        "releases/0.2.0-hardened-alpha-final3/dist/litdatamatcher-0.2.0.tar.gz",
        "7c1af07adff3bbec327bdbc57285e80369c46efad29e99c90f522ecaa1971be4",
    ),
    "acceptance_report": (
        "release/ACCEPTANCE_REPORT_FINAL3.json",
        "dc1bc32317f1430c3a100207fa9ff8e3f1b2e9310783e044eda096a6d04cb384",
    ),
    "closeout_audit": (
        "release_final3/FINAL_CLOSEOUT_AUDIT.json",
        "3c0a0c9ae876643c48cb182c658db973efde542d82a5d2f8ea1e807c05eef7e5",
    ),
    "release_manifest": (
        "release_final3/RELEASE_MANIFEST.json",
        "fe85c1985a5456f78d4261c000ad3a88d7b6f66cb18278454a129668936ab31f",
    ),
    "delivery_validation": (
        "release_final3/DELIVERY_VALIDATION.json",
        "20e08f9b325fb4b30fd86b1923308e203a8e1a3cc13c9d1cd88cb209bb8faf73",
    ),
    "sealed_holdout_manifest": (
        "evaluation/final_holdout_v4/run/RUN_MANIFEST.json",
        "79f92391aee7b8ba0afd13032417029796975dd7b8798c3dacb9f57343727c32",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate(source: Path, data: Path) -> dict:
    """Return a hash- and status-bound alpha baseline receipt."""

    artifact_checks = []
    for name, (relative_path, expected_sha256) in EXPECTED_ARTIFACTS.items():
        path = data / relative_path
        observed = _sha256(path) if path.is_file() else ""
        artifact_checks.append(
            {
                "name": name,
                "path": str(path),
                "expected_sha256": expected_sha256,
                "observed_sha256": observed,
                "status": "PASS" if observed == expected_sha256 else "FAIL",
            }
        )
    commit_present = (
        subprocess.run(
            ["git", "-C", str(source), "cat-file", "-e", f"{ALPHA_BASELINE_COMMIT}^{{commit}}"],
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )
    acceptance = _json(data / "release" / "ACCEPTANCE_REPORT_FINAL3.json")
    closeout = _json(data / "release_final3" / "FINAL_CLOSEOUT_AUDIT.json")
    delivery = _json(data / "release_final3" / "DELIVERY_VALIDATION.json")
    semantic_checks = {
        "baseline_commit_present": commit_present,
        "acceptance_hardened_alpha_ready": acceptance.get("product_status") == "HARDENED_ALPHA_READY",
        "acceptance_supervisor_disabled": acceptance.get("automation_status") == "DISABLED_AT_COMPLETION",
        "closeout_pass": not closeout.get("blockers") and closeout.get("summary", {}).get("PASS", 0) >= 109,
        "delivery_pass": delivery.get("status") == "PASS",
    }
    status = "PASS" if all(item["status"] == "PASS" for item in artifact_checks) and all(semantic_checks.values()) else "FAIL"
    return {
        "schema_version": "alpha-baseline-non-regression-v1",
        "baseline_commit": ALPHA_BASELINE_COMMIT,
        "status": status,
        "artifact_checks": artifact_checks,
        "semantic_checks": semantic_checks,
        "limitation": "This verifies frozen alpha artifacts and sealed-holdout manifest hashes; it does not rerun the holdout or rebuild alpha artifacts.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    result = validate(args.source.resolve(), args.data.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return int(result["status"] != "PASS")


if __name__ == "__main__":
    raise SystemExit(main())
