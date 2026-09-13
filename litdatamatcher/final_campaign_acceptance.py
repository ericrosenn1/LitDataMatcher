"""Read-only final campaign gates over retained, hash-bound observations.

This validator does not run acquisition, models, tests, or the sealed holdout.
Execution receipts are evidence, not authentication of their human producers.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
import re
import sqlite3
import stat
import subprocess
import tarfile
import xml.etree.ElementTree as ET
import zipfile
from contextlib import suppress
from datetime import datetime, timezone
from email import policy
from email.parser import BytesParser
from pathlib import Path, PurePosixPath

from .data_plane import atomic_json, digest

SCHEMA = "final_campaign_ledger_v1"
REQUIREMENT_IDS = frozenset(
    [f"ALPHA.G{n:02d}" for n in range(1, 17)]
    + [f"ALPHA.{n}" for n in ("COVERAGE", "WORKSTREAMS", "REFINEMENT", "LABEL_PROVENANCE")]
    + [f"PHASE2.{n}" for n in "ABCDEFGHIJKLMNO"]
    + ["EXPERT.HUMAN_LABELS", "EXPERT.CALIBRATION"]
    + [f"OPTIONAL.{n}" for n in ("SUGGESTED_MILESTONE_NAMES", "ALL_CANDIDATE_SOURCES", "TRAINING")]
    + [
        f"COMPONENT.{n}"
        for n in (
            "CADMUS",
            "INTERACTION_FINDER",
            "OPTIMUSKG",
            "PRIMEKG",
            "SNACKKSS",
            "SNACKKSS_NLP",
            "REACTOME",
        )
    ]
    + [
        f"FINAL.{n}"
        for n in (
            "MATRIX",
            "FOCUSED",
            "INTEGRATION",
            "FULL_SUITE",
            "DIFF_CHECK",
            "OFFLINE",
            "DETERMINISM",
            "CROSS_SOURCE",
            "RECOVERY",
            "PERFORMANCE",
            "PACKAGE",
            "CLEAN_INSTALL",
            "CLI_SMOKE",
            "ACCEPTANCE",
            "RECEIPTS",
            "SECURITY",
            "INDEPENDENT_REVIEW",
            "REPAIR_FINDINGS",
            "STATE",
            "PUSH",
            "CLOSEOUT",
            "FINAL_REPORT",
            "PROTECTED_ASSETS",
            "SUPERVISOR",
        )
    ]
    + [f"HISTORY.{n}" for n in ("ALPHA_AS_FINAL_ENDPOINT", "MODEL_DUTY_LIMITS", "SCHEDULE_ENABLE")]
)
OPERATIONS_IDS = frozenset(
    f"FINAL.{n}" for n in ("STATE", "PUSH", "CLOSEOUT", "FINAL_REPORT", "SUPERVISOR")
)
DERIVED_IDS = frozenset(f"FINAL.{n}" for n in ("MATRIX", "ACCEPTANCE", "RECEIPTS"))
ALPHA_HASHES = {
    "final3_wheel": "af2a6b6265aa89cad4ac0935bec34dbc3cc8d945c44c941ef9e3341d1794a389",
    "final3_sdist": "7c1af07adff3bbec327bdbc57285e80369c46efad29e99c90f522ecaa1971be4",
    "acceptance_report": "dc1bc32317f1430c3a100207fa9ff8e3f1b2e9310783e044eda096a6d04cb384",
    "closeout_audit": "3c0a0c9ae876643c48cb182c658db973efde542d82a5d2f8ea1e807c05eef7e5",
    "release_manifest": "fe85c1985a5456f78d4261c000ad3a88d7b6f66cb18278454a129668936ab31f",
    "delivery_validation": "20e08f9b325fb4b30fd86b1923308e203a8e1a3cc13c9d1cd88cb209bb8faf73",
    "sealed_holdout_manifest": "79f92391aee7b8ba0afd13032417029796975dd7b8798c3dacb9f57343727c32",
}
SHA = re.compile(r"[0-9a-f]{64}\Z")
COMMIT = re.compile(r"[0-9a-f]{40}\Z")


class InvalidEvidence(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidEvidence(message)


def sha256(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _json(path):
    def reject(value):
        raise InvalidEvidence(f"Nonfinite JSON value: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8-sig"), parse_constant=reject)


def _rows(path):
    rows = [
        json.loads(line) for line in Path(path).read_text("utf-8-sig").splitlines() if line.strip()
    ]
    require(rows and all(isinstance(row, dict) for row in rows), f"Empty/nonobject records: {path}")
    digest(rows)  # Reject nonfinite nested values too.
    return rows


def _time(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(stamp.tzinfo is not None, "Receipt timestamps require timezone")
    return stamp


def _git(root, *args):
    run = subprocess.run(["git", "-C", str(root), *args], capture_output=True, check=False)
    require(
        run.returncode == 0,
        f"Local git {' '.join(args[:3])}: {run.stderr.decode('utf-8', 'replace').strip()}",
    )
    return run.stdout


def _functional(name):
    p = PurePosixPath(name.replace("\\", "/"))
    return str(p) == "pyproject.toml" or (
        p.parts
        and (p.parts[0] in {"litdatamatcher", "tests"} or p.parts[:2] == ("scripts", "v2"))
        and p.suffix in {".py", ".json", ".jsonl", ".lock"}
        and not any(x in {"__pycache__", ".pytest_cache", ".ruff_cache"} for x in p.parts)
    )


def _code_digest(content):
    # Git checkout line-ending conversion must not invalidate identical code.
    return hashlib.sha256(content.replace(b"\r\n", b"\n")).hexdigest()


def source_snapshot(root: str | Path) -> dict:
    """Fingerprint current tracked and untracked functional bytes, including tests."""
    root = Path(root).resolve()
    names = (
        _git(root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
        .decode("utf-8")
        .split("\0")
    )
    files = {
        n: _code_digest((root / n).read_bytes())
        for n in sorted(set(names))
        if n and _functional(n) and (root / n).is_file()
    }
    require(files, "No functional source files")
    changed = _git(root, "diff", "HEAD", "--name-only", "-z").decode("utf-8").split("\0")
    untracked = (
        _git(root, "ls-files", "--others", "--exclude-standard", "-z").decode("utf-8").split("\0")
    )
    return {
        "schema_version": "final_functional_source_v1",
        "source_root": str(root),
        "source_commit": _git(root, "rev-parse", "HEAD").decode().strip(),
        "files": files,
        "digest": digest(files),
        "dirty_functional_paths": sorted({n for n in changed + untracked if n and _functional(n)}),
    }


def _committed_files(root, commit):
    require(isinstance(commit, str) and COMMIT.fullmatch(commit), "Expected full source commit")
    tree = _git(root, "ls-tree", "-r", "--name-only", commit).decode("utf-8").splitlines()
    prefixes = [
        p
        for p in ("litdatamatcher", "tests", "scripts/v2", "pyproject.toml")
        if any(n == p or n.startswith(p + "/") for n in tree)
    ]
    require(prefixes, "Commit has no functional files")
    raw = _git(root, "archive", "--format=tar", commit, *prefixes)
    with tarfile.open(fileobj=io.BytesIO(raw), mode="r:") as archive:
        return {
            m.name: _code_digest(archive.extractfile(m).read())
            for m in archive.getmembers()
            if m.isfile() and _functional(m.name)
        }


def _safe_member(name):
    p = PurePosixPath(name)
    require(
        name
        and not p.is_absolute()
        and not any(x in {"", ".", ".."} for x in name.split("/"))
        and "\\" not in name
        and ":" not in name
        and "\0" not in name,
        f"Unsafe archive path: {name!r}",
    )


def inspect_archive(path: str | Path, kind: str) -> dict:
    """Reopen without extracting; verify every ZIP CRC and the wheel RECORD."""
    path = Path(path)
    members = {}
    seen = set()
    total = 0

    def add(name, data):
        nonlocal total
        _safe_member(name)
        require(name.casefold() not in seen, f"Duplicate archive member: {name}")
        seen.add(name.casefold())
        total += len(data)
        require(total <= 512 * 1024 * 1024, "Unexpectedly large source/distribution archive")
        require(
            not any(
                p.lower() in {".git", "__pycache__", ".env", "cache", "checkpoints"}
                for p in PurePosixPath(name).parts
            ),
            f"Runtime/private artifact in archive: {name}",
        )
        require(
            not name.lower().endswith((".safetensors", ".pyc", ".sqlite3")),
            f"Runtime artifact in archive: {name}",
        )
        members[name] = data

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            require(
                sum(m.file_size for m in archive.infolist()) <= 512 * 1024 * 1024,
                "Archive size limit exceeded",
            )
            for member in archive.infolist():
                _safe_member(member.orig_filename.rstrip("/"))
                _safe_member(member.filename.rstrip("/"))
                require(
                    not stat.S_ISLNK(member.external_attr >> 16), "Archive symlink is forbidden"
                )
                content = archive.read(member)  # Includes directory entries in CRC verification.
                if not member.is_dir():
                    add(member.filename, content)
                else:
                    require(not content, "Directory archive entry contains payload bytes")
    else:
        with tarfile.open(path, "r:*") as archive:
            require(
                sum(m.size for m in archive.getmembers()) <= 512 * 1024 * 1024,
                "Archive size limit exceeded",
            )
            for member in archive.getmembers():
                _safe_member(member.name.rstrip("/"))
                require(
                    member.isfile() or member.isdir(), f"Archive link/special member: {member.name}"
                )
                if member.isfile():
                    add(member.name, archive.extractfile(member).read())
    require(members, "Empty archive")
    result = {"members": len(members), "uncompressed_bytes": total, "files": members}
    if kind in {"wheel", "sdist"}:
        suffix = ".dist-info/METADATA" if kind == "wheel" else "/PKG-INFO"
        metadata = [v for k, v in members.items() if k.endswith(suffix)]
        require(metadata, f"Missing {kind} package metadata")
        versions, names = set(), set()
        for item in metadata:
            # Core metadata is an email-style header block. Read only that block:
            # CRLF is valid, and README body examples are not package identity.
            message = BytesParser(policy=policy.default).parsebytes(item, headersonly=True)
            require(not message.defects, "Malformed package metadata headers")
            fields = {}
            for header in ("Name", "Version"):
                values = message.get_all(header, [])
                require(
                    len(values) == 1,
                    f"Duplicate or missing {header} package metadata header",
                )
                value = str(values[0]).strip()
                require(
                    value and not any(character.isspace() for character in value),
                    f"Invalid {header} package metadata header",
                )
                fields[header] = value
            versions.add(fields["Version"])
            names.add(fields["Name"])
        require(
            names == {"litdatamatcher"} and len(versions) == 1 and None not in versions,
            "Contradictory package name/version metadata",
        )
        result["version"] = versions.pop()
    if kind == "wheel":
        records = [k for k in members if k.endswith(".dist-info/RECORD")]
        require(len(records) == 1, "Wheel must contain one RECORD")
        record = records[0]
        rows = list(csv.reader(io.StringIO(members[record].decode("utf-8"))))
        require(all(len(row) == 3 for row in rows), "Malformed wheel RECORD")
        require(
            len(rows) == len(members) and {r[0] for r in rows} == set(members),
            "Wheel RECORD omits/duplicates/adds members",
        )
        for name, recorded_hash, size in rows:
            if name == record:
                require(
                    recorded_hash == size == "", "RECORD self-entry must have empty hash and size"
                )
            else:
                expected = (
                    base64.urlsafe_b64encode(hashlib.sha256(members[name]).digest())
                    .decode()
                    .rstrip("=")
                )
                require(
                    recorded_hash == "sha256=" + expected and size == str(len(members[name])),
                    f"Wheel RECORD mismatch: {name}",
                )
    return result


def junit_counts(path: str | Path) -> dict:
    root = ET.fromstring(Path(path).read_bytes())
    require(root.tag in {"testsuite", "testsuites"}, "Unrecognized JUnit root")
    cases = list(root.iter("testcase"))
    counts = {
        "tests": len(cases),
        "failures": sum(c.find("failure") is not None for c in cases),
        "errors": sum(c.find("error") is not None for c in cases),
        "skipped": sum(c.find("skipped") is not None for c in cases),
    }
    require(cases, "Empty JUnit collection")
    identities = [(c.get("classname"), c.get("name")) for c in cases]
    require(len(identities) == len(set(identities)), "Duplicate JUnit testcase identities")
    for suite in [root, *list(root.iter("testsuite"))]:
        actual = list(suite.iter("testcase"))
        for name, tag in (
            ("tests", None),
            ("failures", "failure"),
            ("errors", "error"),
            ("skipped", "skipped"),
        ):
            if name in suite.attrib:
                expected = (
                    len(actual) if tag is None else sum(c.find(tag) is not None for c in actual)
                )
                require(
                    int(suite.get(name)) == expected, f"JUnit declared {name} contradicts testcases"
                )
    return counts


def _record_id(row, kind):
    return (
        row.get("dataset_id")
        if kind == "dataset"
        else row.get("document_id") or row.get("source_id")
    )


def _provenance(row):
    return row.get("source_provenance") or row.get("metadata", {}).get("source_provenance") or {}


def _strip_cache(value):
    if isinstance(value, dict):
        return {k: _strip_cache(v) for k, v in value.items() if k != "cache_status"}
    if isinstance(value, list):
        return [_strip_cache(v) for v in value]
    return value


def source_span_matches(raw, span):
    """Compare text coordinates after decoding retained JSON/JSONL string fields."""
    texts = [raw]

    def visit(value):
        if isinstance(value, str):
            texts.append(value)
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    with suppress(ValueError):
        visit(json.loads(raw))
    if len(texts) == 1:
        for line in raw.splitlines():
            with suppress(ValueError):
                visit(json.loads(line))
    if isinstance(span, dict):
        text = span.get("text") or span.get("quote")
        if "start" in span or "end" in span:
            start, end = span.get("start"), span.get("end")
            return (
                type(start) is int
                and type(end) is int
                and 0 <= start < end
                and isinstance(text, str)
                and end - start == len(text)
                and any(value[start:end] == text for value in texts)
            )
    else:
        text = span
    return isinstance(text, str) and bool(text.strip()) and any(text in value for value in texts)


class Validator:
    def __init__(self, ledger_path, mode="technical"):
        self.path = Path(ledger_path).resolve()
        self.base = self.path.parent
        self.ledger = _json(self.path)
        require(self.ledger.get("schema_version") == SCHEMA, "Unsupported final ledger schema")
        require(
            self.ledger.get("expert_validation", "PENDING_EXPERT_REVIEW")
            == "PENDING_EXPERT_REVIEW",
            "Final campaign expert validation remains pending",
        )
        require(mode in {"technical", "closeout"}, "Unknown validation mode")
        self.mode = mode
        self.root = self.resolve(self.ledger["source"]["root"])
        self.snapshot = None
        self.checks = []
        self.real_dataset_ids = set()
        self.native_runs = {}

    def resolve(self, value, base=None):
        p = Path(value)
        return p.resolve() if p.is_absolute() else ((base or self.base) / p).resolve()

    def ref(self, value, *, base=None, size=True):
        require(isinstance(value, dict), "Artifact reference must be an object")
        p = self.resolve(value["path"], base)
        require(p.is_file() and not p.is_symlink(), f"Missing/nonregular artifact: {p}")
        require(
            isinstance(value.get("sha256"), str) and SHA.fullmatch(value["sha256"]),
            f"Missing/invalid artifact digest: {p}",
        )
        require(sha256(p) == value["sha256"], f"Artifact hash mismatch: {p}")
        if size or "size_bytes" in value:
            require(
                type(value.get("size_bytes")) is int and p.stat().st_size == value["size_bytes"],
                f"Artifact size mismatch: {p}",
            )
        return p

    def payload(self, value, **kwargs):
        return _json(self.ref(value, **kwargs))

    def source_bound(self, value, *, exact=True):
        require(self.snapshot is not None, "Source gate must pass first")
        require(
            value.get("source_fingerprint") == self.snapshot["digest"],
            "Stale source fingerprint in receipt",
        )
        commit = value.get("source_commit")
        require(
            isinstance(commit, str) and COMMIT.fullmatch(commit), "Missing receipt source commit"
        )
        _git(self.root, "merge-base", "--is-ancestor", commit, "HEAD")
        if exact and value.get("dirty_functional_paths") == []:
            require(
                digest(_committed_files(self.root, commit)) == self.snapshot["digest"],
                "Clean receipt commit differs from observed tested bytes",
            )

    def command(self, ref, *, successful=True, source=True):
        row = self.payload(ref)
        require(
            row.get("schema_version") == "final_command_receipt_v1", "Unsupported command receipt"
        )
        require(
            isinstance(row.get("argv"), list)
            and row["argv"]
            and all(isinstance(a, str) and a for a in row["argv"]),
            "Command argv is absent",
        )
        require(Path(row["cwd"]).is_absolute(), "Command cwd must be absolute")
        require(
            _time(row["finished_at"]) >= _time(row["started_at"]),
            "Command finished before starting",
        )
        require(
            type(row.get("exit_code")) is int and (not successful or row["exit_code"] == 0),
            "Recorded command failed",
        )
        require(
            isinstance(row.get("dirty_functional_paths"), list)
            and all(_functional(n) for n in row["dirty_functional_paths"]),
            "Missing/invalid dirty test source state",
        )
        if source:
            self.source_bound(row)
        self.ref(row["stdout"])
        for item in row.get("inputs", []) + row.get("outputs", []):
            self.ref(item)
        return row

    def run_check(self, name, fn, axis="technical"):
        try:
            detail = fn()
            self.checks.append(
                {"check": name, "axis": axis, "status": "PASS", "detail": detail, "errors": []}
            )
        except (
            InvalidEvidence,
            KeyError,
            TypeError,
            ValueError,
            OSError,
            ET.ParseError,
            zipfile.BadZipFile,
            tarfile.TarError,
            sqlite3.Error,
        ) as exc:
            self.checks.append(
                {
                    "check": name,
                    "axis": axis,
                    "status": "FAIL",
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
            )

    def check_source(self):
        expected = self.payload(self.ledger["source"]["fingerprint"])
        current = source_snapshot(self.root)
        require(
            expected.get("schema_version") == current["schema_version"],
            "Unknown source snapshot schema",
        )
        require(
            expected.get("files") == current["files"]
            and expected.get("digest") == current["digest"],
            "Current functional source differs from recorded fingerprint",
        )
        require(
            current["dirty_functional_paths"] == [],
            "Final functional source has uncommitted changes",
        )
        commit = self.ledger["source"]["functional_commit"]
        require(
            _committed_files(self.root, commit) == current["files"],
            "Functional commit does not contain current source bytes",
        )
        _git(self.root, "merge-base", "--is-ancestor", commit, "HEAD")
        self.snapshot = current
        return {
            "functional_commit": commit,
            "current_commit": current["source_commit"],
            "files": len(current["files"]),
            "fingerprint": current["digest"],
        }

    def check_matrix(self):
        matrix = self.payload(self.ledger["matrix"])
        rows = matrix["requirements"]
        require(
            isinstance(rows, list) and len(rows) == len(REQUIREMENT_IDS),
            "Requirement matrix must contain exactly 74 rows",
        )
        require(
            {r["id"] for r in rows} == REQUIREMENT_IDS,
            "Missing, duplicate, or unexpected requirement IDs",
        )
        counts = {}
        pending = []
        for row in rows:
            name, classification = row["id"], row["classification"]
            require(
                type(row.get("mandatory_for_software_complete")) is bool
                and type(row.get("blocks_project_completion")) is bool,
                f"Missing classification booleans: {name}",
            )
            require(
                row["mandatory_for_software_complete"]
                == name.startswith(("ALPHA.", "PHASE2.", "FINAL.")),
                f"Governing mandatory/optional scope changed: {name}",
            )
            counts[classification] = counts.get(classification, 0) + 1
            if name.startswith("EXPERT."):
                require(
                    classification == "OPTIONAL_FUTURE_VALIDATION"
                    and not row["mandatory_for_software_complete"]
                    and not row["blocks_project_completion"],
                    f"Expert evidence incorrectly promoted or blocking: {name}",
                )
                require(
                    row.get("expert_status", "PENDING_EXPERT_REVIEW") == "PENDING_EXPERT_REVIEW",
                    "This campaign has no completed independent expert validation",
                )
            elif name in OPERATIONS_IDS:
                if classification != "COMPLETE_VALIDATED" or row["blocks_project_completion"]:
                    pending.append(name)
            elif name in DERIVED_IDS and row.get("validation_contract") == "THIS_VALIDATOR":
                require(
                    classification in {"COMPLETE_VALIDATED", "COMPLETE_NEEDS_FINAL_REGRESSION"}
                    and not row["blocks_project_completion"],
                    f"Invalid derived validator disposition: {name}",
                )
            elif name.startswith(("ALPHA.", "PHASE2.", "FINAL.")):
                require(
                    classification == "COMPLETE_VALIDATED" and not row["blocks_project_completion"],
                    f"Unfinished mandatory requirement: {name} ({classification})",
                )
                require(
                    row["mandatory_for_software_complete"],
                    f"Mandatory requirement downgraded to optional: {name}",
                )
            else:
                require(
                    classification
                    in {
                        "COMPLETE_VALIDATED",
                        "NOT_APPLICABLE",
                        "SUPERSEDED",
                        "OPTIONAL_FUTURE_VALIDATION",
                    }
                    and not row["blocks_project_completion"],
                    f"Unjustified/nonclosed scope disposition: {name}",
                )
            require(
                isinstance(row.get("evidence"), list) and row["evidence"],
                f"No requirement evidence: {name}",
            )
            for ref in row["evidence"]:
                self.ref(ref, size=False, base=self.root)
            require(
                row.get("criterion") and row.get("requirement_sources"),
                f"No governing criterion: {name}",
            )
            for ref in row["requirement_sources"]:
                self.ref(ref, size=False, base=self.root)
        self.operations_pending = pending
        return {"rows": len(rows), "classification_counts": counts, "operations_pending": pending}

    def check_junit(self):
        items = self.ledger["junit"]
        require(isinstance(items, list) and items, "Missing executed test receipts")
        scopes = set()
        errors = []
        for row in items:
            scope = row["scope"]
            require(scope in {"full", "targeted", "integration"}, "Invalid JUnit scope")
            scopes.add(scope)
            path = self.ref(row["xml"])
            counts = junit_counts(path)
            self.junit.append({"scope": scope, "path": str(path), **counts})
            command = self.command(row["command"], successful=False)
            require(
                any(
                    self.resolve(x["path"]) == path and x["sha256"] == row["xml"]["sha256"]
                    for x in command.get("outputs", [])
                ),
                "JUnit XML not bound to executed command output",
            )
            require(
                any("pytest" in Path(arg).name.lower() for arg in command["argv"]),
                "JUnit command is not a recorded pytest execution",
            )
            if scope == "full":
                forbidden = (
                    "-k",
                    "-m",
                    "--deselect",
                    "--ignore",
                    "--last-failed",
                    "--lf",
                    "--collect-only",
                )
                argv = command["argv"]
                # Python's -m pytest is allowed; pytest's -m expression is selective.
                start = next(i for i, arg in enumerate(argv) if "pytest" in Path(arg).name.lower())
                require(
                    not any(
                        arg == f or arg.startswith(f + "=")
                        for arg in argv[start + 1 :]
                        for f in forbidden
                    ),
                    "Full suite receipt selected/excluded tests",
                )
                require(
                    not any("::" in arg or arg.endswith(".py") for arg in argv[start + 1 :]),
                    "Full suite receipt selects a test file/node",
                )
            if command["exit_code"] != 0 or any(
                counts[k] for k in ("failures", "errors", "skipped")
            ):
                errors.append(f"{scope}: exit={command['exit_code']} counts={counts}")
        require(
            "full" in scopes and bool(scopes & {"targeted", "integration"}),
            "Full and focused/integration test receipts are required",
        )
        require(not errors, "; ".join(errors))
        return self.junit

    def check_distribution(self):
        config = self.ledger["distribution"]
        version = config["version"]
        require(
            isinstance(version, str)
            and re.fullmatch(r"\d+\.\d+\.\d+", version)
            and version != "0.2.0",
            "Expected separate final release version",
        )
        archives = {
            kind: inspect_archive(self.ref(config[kind]), kind)
            for kind in ("wheel", "sdist", "source_archive")
        }
        require(
            archives["wheel"]["version"] == archives["sdist"]["version"] == version,
            "Wheel/sdist/ledger version disagreement",
        )
        require(self.snapshot is not None, "Source gate must pass first")
        for kind, archive in archives.items():
            files = archive["files"]
            for member in files:
                parts = PurePosixPath(member).parts
                candidates = [
                    "/".join(parts[i:])
                    for i in range(len(parts))
                    if _functional("/".join(parts[i:]))
                ]
                if candidates:
                    require(
                        candidates[0] in self.snapshot["files"],
                        f"{kind} contains unreviewed functional file: {member}",
                    )
            for name, expected in self.snapshot["files"].items():
                if kind != "source_archive" and not name.startswith("litdatamatcher/"):
                    continue
                hits = [
                    data
                    for member, data in files.items()
                    if member == name or (kind != "wheel" and member.endswith("/" + name))
                ]
                require(
                    len(hits) == 1 and _code_digest(hits[0]) == expected,
                    f"{kind} missing/stale functional file: {name}",
                )
        self.wheel_files = archives["wheel"]["files"]
        return {
            kind: {k: v for k, v in info.items() if k != "files"} for kind, info in archives.items()
        }

    def check_install(self):
        receipts = self.ledger["clean_install"]
        require(isinstance(receipts, list) and receipts, "Missing clean-install receipt")
        results = []
        for ref in receipts:
            command = self.command(ref)
            require(
                any(
                    item["sha256"] == self.ledger["distribution"]["wheel"]["sha256"]
                    for item in command.get("inputs", [])
                ),
                "Clean install does not bind delivered wheel",
            )
            observation = self.payload(command["stdout"])
            require(
                observation.get("schema_version") == "final_install_observation_v1",
                "Missing installed runtime observation",
            )
            require(
                observation["package_version"] == self.ledger["distribution"]["version"],
                "Installed package version differs from wheel",
            )
            prefix, package = (
                Path(observation["sys_prefix"]).resolve(),
                Path(observation["package_file"]).resolve(),
            )
            cwd, executable = (
                Path(observation["cwd"]).resolve(),
                Path(observation["executable"]).resolve(),
            )
            require(
                package.is_file()
                and package.is_relative_to(prefix)
                and not package.is_relative_to(self.root),
                "Installed import is missing or resolves inside source checkout",
            )
            require(
                executable.is_file()
                and executable.is_relative_to(prefix)
                and not cwd.is_relative_to(self.root),
                "Install observation did not run outside source in isolated environment",
            )
            require(package.name == "__init__.py", "Expected installed litdatamatcher.__file__")
            require(
                hasattr(self, "wheel_files")
                and package.read_bytes() == self.wheel_files["litdatamatcher/__init__.py"],
                "Installed import bytes differ from delivered wheel",
            )
            for name, data in self.wheel_files.items():
                if name.startswith("litdatamatcher/"):
                    installed = package.parent.parent / name
                    require(
                        installed.is_file() and installed.read_bytes() == data,
                        f"Installed package file differs from wheel: {name}",
                    )
            cli = observation["cli_results"]
            require(
                {item["name"] for item in cli} >= {"litdatamatcher", "litdatamatcher-v2"},
                "Both installed CLIs require smoke observations",
            )
            for item in cli:
                require(
                    type(item["exit_code"]) is int and item["exit_code"] == 0 and item.get("argv"),
                    "Installed CLI smoke failed",
                )
                log = self.ref(item["stdout"])
                require(log.stat().st_size > 0, "Installed CLI smoke output is empty")
            results.append(
                {
                    "package_file": str(package),
                    "package_version": observation["package_version"],
                    "cli_count": len(cli),
                }
            )
        return results

    def check_offline(self):
        receipts = self.ledger["offline"]
        require(isinstance(receipts, list) and receipts, "Missing offline execution receipts")
        results = []
        for ref in receipts:
            command = self.command(ref)
            observation = self.payload(command["stdout"])
            require(
                observation.get("schema_version") == "final_offline_observation_v1",
                "Unknown offline observation",
            )
            network = observation["network_control"]
            require(
                type(network.get("blocked_probe_count")) is int
                and network["blocked_probe_count"] >= 1
                and network.get("unexpected_requests") == 0,
                "Offline network guard untested/attempts observed",
            )
            values = []
            fingerprints = []
            for name, origin in (("fresh", "fresh_local_inference"), ("replay", "cache_replay")):
                item = observation[name]
                require(item["origin"] == origin, f"Offline {name} origin not observed")
                fingerprints.append(self.verify_run_manifest(item["manifest"], origin))
                values.append(self.payload(item["payload"]))
            require(
                values[0] and values[0] == values[1],
                "Fresh/replay scientific payloads differ or are empty",
            )
            require(
                fingerprints[0] == fingerprints[1],
                "Offline replay input/model/configuration fingerprint differs from fresh inference",
            )
            results.append(
                {
                    "payload_digest": digest(values[0]),
                    "network_probe_count": network["blocked_probe_count"],
                }
            )
        return results

    def verify_run_manifest(self, ref, origin=None):
        path = self.ref(ref)
        manifest = _json(path)
        require(
            (manifest["execution_status"] == "PASS" and not manifest["failures"])
            or (
                manifest["execution_status"] == "PARTIAL"
                and manifest["failures"]
                and all(
                    f.get("stage") == "source_guard" and f.get("rejections")
                    for f in manifest["failures"]
                )
            ),
            "Offline native run failed",
        )
        require(manifest["network"]["mode"] == "OFFLINE", "Run was not configured offline")
        require(
            manifest["models"]
            and all(
                m.get("id") and m.get("revision") and m.get("runtime") for m in manifest["models"]
            ),
            "Offline model identity/revision missing",
        )
        artifacts = {}
        for artifact in manifest["artifacts"]:
            p = self.resolve(artifact["path"], path.parent)
            require(
                p.is_relative_to(path.parent) and p.name not in artifacts,
                "Duplicate/outside native run artifact",
            )
            require(artifact["validation"] == "PASS", "Native artifact validation failed")
            artifacts[p.name] = self.ref(artifact, base=path.parent)
        require(
            {
                "questions.jsonl",
                "matches.jsonl",
                "scientific_dossiers.jsonl",
                "inferences.jsonl",
                "claims.jsonl",
            }
            <= set(artifacts),
            "Native scientific output inventory incomplete",
        )
        inferences = _rows(artifacts["inferences.jsonl"])
        require(
            all(
                (
                    i["origin"] == origin
                    if origin
                    else i["origin"] in {"fresh_local_inference", "cache_replay"}
                )
                and i["local_files_only"] is True
                and i.get("process_id")
                and i.get("execution_id")
                for i in inferences
            ),
            "Run inference origin/execution identity contradicts offline observation",
        )
        counts = manifest["inference"]
        require(
            counts["fresh_calls"] == sum(i["origin"] == "fresh_local_inference" for i in inferences)
            and counts["cache_replays"] == sum(i["origin"] == "cache_replay" for i in inferences),
            "Run inference counts contradict retained observations",
        )
        dossiers = [
            json.loads(line)
            for line in artifacts["scientific_dossiers.jsonl"].read_text("utf-8-sig").splitlines()
            if line.strip()
        ]
        claims = artifacts.get("claims.jsonl")
        claim_count = (
            len([line for line in claims.read_text("utf-8-sig").splitlines() if line.strip()])
            if claims
            else None
        )
        self.native_runs[str(path)] = {
            "manifest": ref,
            "execution_status": manifest["execution_status"],
            "failures": manifest["failures"],
            "coverage": manifest.get("coverage", {}),
            "inference": counts,
            "accepted_claim_count": claim_count,
            "dossier_count": len(dossiers),
            "dossiers": dossiers,
        }
        return sorted(
            (i["model_id"], i["model_revision"], digest(i["fingerprint"])) for i in inferences
        )

    def check_cases(self):
        from .scientific_dossier import validate_dossier

        manifest = self.payload(self.ledger["cases"]["manifest"])
        require(
            manifest.get("schema_version") == "final_cases_manifest_v1"
            and manifest.get("data_origin") == "real",
            "Cases must retain real-source provenance",
        )
        self.source_bound(manifest, exact=False)
        command = self.command(manifest["command"])
        catalog = _rows(self.ref(manifest["dataset_catalog"]))
        candidates = {row["dataset_id"]: row for row in catalog}
        require(len(candidates) == len(catalog), "Case catalog contains duplicate study identities")
        require(
            set(candidates) <= self.real_dataset_ids,
            "Case catalog contains identities outside verified acquired inputs",
        )
        locators = {}
        for item in manifest["source_locators"]:
            require(
                item.get("locator") and item["locator"] not in locators,
                "Duplicate/empty source locator",
            )
            locators[item["locator"]] = self.ref(item["artifact"]).read_text("utf-8-sig")
        cases = manifest["cases"]
        require(
            len(cases) >= 6 and len({r["case_id"] for r in cases}) == len(cases),
            "At least six distinct source-assisted cases required",
        )
        require(len({r["domain"] for r in cases}) >= 2, "Case evidence requires multiple domains")
        outputs = {self.resolve(ref["path"]): ref["sha256"] for ref in command.get("outputs", [])}
        identities = set()
        for case in cases:
            self.verify_run_manifest(case["run_manifest"])
            native = self.native_runs[str(self.resolve(case["run_manifest"]["path"]))]
            path = self.ref(case["dossier"])
            require(
                outputs.get(path) == case["dossier"]["sha256"],
                "Dossier is not bound to case execution output",
            )
            dossier = _json(path)
            require(validate_dossier(dossier), f"Invalid scientific dossier: {case['case_id']}")
            identity = (
                dossier["question"]["question_id"],
                dossier["candidate_dataset"]["dataset_id"],
            )
            require(
                identity not in identities,
                "Duplicate question/candidate presented as a distinct case",
            )
            identities.add(identity)
            require(
                dossier.get("calibration_status", "UNCALIBRATED_HEURISTIC")
                == "UNCALIBRATED_HEURISTIC",
                "Source-assisted dossier cannot claim expert calibration",
            )
            candidate = dossier["candidate_dataset"]
            require(
                candidate["dataset_id"] in candidates,
                "Dossier candidate is absent from real catalog",
            )
            for key, value in candidate.items():
                require(
                    key not in candidates[candidate["dataset_id"]]
                    or candidates[candidate["dataset_id"]][key] == value,
                    f"Dossier candidate contradicts retained catalog: {key}",
                )
            for item in dossier["source_evidence"]:
                require(
                    item["source_locator"] in locators,
                    "Dossier source locator has no retained hashed artifact",
                )
                span = (
                    item.get("evidence_span")
                    or item.get("quote")
                    or item.get("claim", {}).get("evidence_span")
                )
                if item.get("evidence_origin") == "deterministic_source_passage":
                    require(
                        item.get("role") == "background"
                        and item.get("claim_status") == "NOT_ASSERTED"
                        and item.get("direction") == "inconclusive"
                        and item.get("answers_question") is False,
                        "Deterministic source context was promoted to a scientific claim",
                    )
                if item.get("role") in {"direct_test", "replication", "perturbational_observation"}:
                    require(
                        bool(span),
                        "Direct scientific evidence requires a retained source span",
                    )
                if span:
                    require(
                        source_span_matches(locators[item["source_locator"]], span),
                        "Claim/context evidence span does not occur at its retained source location",
                    )
            require(
                dossier.get("review_status") == "SOURCE_ASSISTED_PENDING_EXPERT_REVIEW",
                "Expert validation masquerade in dossier",
            )
            require(
                dossier in native["dossiers"], "Dossier was not produced in its retained native run"
            )
        return {
            "case_count": len(cases),
            "domains": sorted({r["domain"] for r in cases}),
            "catalog_size": len(candidates),
            "expert_status": "PENDING_EXPERT_REVIEW",
        }

    def check_acquisition(self):
        config = self.ledger["acquisition"]
        receipt, plan = self.payload(config["receipt"]), self.payload(config["plan"])
        replay = self.payload(config["offline_replay"])
        require(
            receipt["status"] in {"PASS", "PASS_WITH_LIMITATIONS"} and replay["status"] == "PASS",
            "Acquisition/replay not successful",
        )
        require(receipt["plan_sha256"] == config["plan"]["sha256"], "Acquisition plan changed")
        require(
            _time(plan["declared_utc"]) < _time(receipt["completion_utc"]),
            "Acquisition plan not declared before completion",
        )
        require(
            replay.get("network_guard_calls") == 0
            and replay.get("ignored_field") == "cache_status only, recursively",
            "Acquisition replay made network requests or ignored more than cache origin",
        )
        by_kind = {}
        for name, metadata in receipt["files"].items():
            path = self.ref({"path": name, **metadata}, size=False)
            rows = _rows(path)
            kind = "datasets" if "dataset_id" in rows[0] else "literature"
            ids = [_record_id(r, "dataset" if kind == "datasets" else "literature") for r in rows]
            require(
                all(ids) and len(ids) == len(set(ids)) == metadata["rows"],
                "Acquisition rows/identity counts contradict retained input",
            )
            require(kind not in by_kind, "Multiple ambiguous final acquisition files")
            by_kind[kind] = rows
        counts = {kind: len(rows) for kind, rows in by_kind.items()}
        require(counts == receipt["counts"], "Acquisition total counts contradict files")
        targets = plan["targets"]
        require(
            counts["literature"] >= max(1000, targets["unique_literature"])
            and counts["datasets"] >= max(300, targets["unique_study_ids"]),
            "Declared expanded acquisition targets unmet",
        )
        repositories = {}
        for row in by_kind["datasets"]:
            repositories[row["source"]] = repositories.get(row["source"], 0) + 1
        require(
            repositories == receipt["dataset_sources"]
            and len(repositories) >= max(2, targets["dataset_repositories"]),
            "Repository counts/coverage contradict actual records",
        )
        partition_plan = {r["id"]: r for r in plan["partitions"]}
        summaries = {r["id"]: r for r in receipt["partitions"]}
        replay_rows = {r["partition"]: r for r in replay["partitions"]}
        require(
            len(partition_plan) == len(plan["partitions"])
            and set(partition_plan) == set(summaries) == set(replay_rows),
            "Partition coverage incomplete or duplicated",
        )
        observed = {}
        snapshot_count = 0
        for ref in config["partition_receipts"]:
            path = self.ref(ref)
            part = _json(path)
            key = part["partition"]["id"]
            require(
                key not in observed and part["partition"] == partition_plan[key],
                "Partition receipt disagrees with frozen plan",
            )
            require(
                part["status"] == summaries[key]["status"] == "ACQUIRED_BOUNDED"
                and not part["error"],
                "Failed acquisition partition",
            )
            require(
                part["plan_sha256"] == config["plan"]["sha256"],
                "Partition not bound to acquisition plan",
            )
            records = _rows(
                self.ref(
                    {"path": str(path.parent / "records.jsonl"), "sha256": part["records_sha256"]},
                    size=False,
                )
            )
            require(
                len(records)
                == part["accepted_rows"]
                == summaries[key]["accepted_rows"]
                == replay_rows[key]["rows"]
                <= part["partition"]["limit"],
                "Partition row/limit count contradiction",
            )
            comparable = hashlib.sha256(
                json.dumps(_strip_cache(records), sort_keys=True, ensure_ascii=False).encode()
            ).hexdigest()
            require(
                comparable
                == replay_rows[key]["original_comparable_sha256"]
                == replay_rows[key]["replay_comparable_sha256"],
                "Offline replay payload digest contradicts original records",
            )
            events = part["snapshot_events"]
            require(
                events and len(part["request_events"]) <= part["partition"]["max_requests"],
                "Missing source snapshots or request bound exceeded",
            )
            hashes = set()
            for event in events:
                if "status_code" not in event:
                    original = [
                        r["snapshot"]
                        for r in part["request_events"]
                        if r.get("snapshot", {}).get("sha256") == event["sha256"]
                        and r["snapshot"].get("url") == event["url"]
                    ]
                    require(
                        event.get("status") == "offline_replay" and len(original) == 1,
                        "Offline snapshot event does not resolve to original request metadata",
                    )
                    event = original[0]
                require(
                    event["status_code"] == 200 and event["url"].startswith("https://"),
                    "Unsuccessful/nonpublic source snapshot",
                )
                self.ref(
                    {
                        "path": event["object_path"],
                        "sha256": event["sha256"],
                        "size_bytes": event["size_bytes"],
                    }
                )
                hashes.add(event["sha256"])
                snapshot_count += 1
            for record in records:
                provenance = _provenance(record)
                require(
                    isinstance(provenance, dict) and provenance.get("source_locator"),
                    "Real acquisition record lacks source locator",
                )
                snapshot = provenance.get("metadata", {}).get("cache_snapshot", {})
                require(
                    snapshot.get("cache_content_sha256") in hashes,
                    "Record does not resolve to retained source snapshot",
                )
            observed[key] = records
        require(set(observed) == set(partition_plan), "Missing explicit partition receipt")
        membership = _rows(self.ref(config["membership"]))
        for member in membership:
            require(
                member["partition"] in observed
                and member["source_id"]
                in {
                    _record_id(
                        r,
                        "dataset"
                        if partition_plan[member["partition"]]["kind"] == "datasets"
                        else "literature",
                    )
                    for r in observed[member["partition"]]
                },
                "Membership points outside retained partition identities",
            )
        for kind, rows in by_kind.items():
            retained = {
                digest(_strip_cache(row))
                for key, items in observed.items()
                if partition_plan[key]["kind"] == kind
                for row in items
            }
            require(
                all(digest(_strip_cache(row)) in retained for row in rows),
                "Final acquired records do not occur in source partitions",
            )
        domains = {}
        for domain in {p["domain"] for p in partition_plan.values()}:
            domains[domain] = {}
            for kind in ("literature", "datasets"):
                keys = {
                    k
                    for k, p in partition_plan.items()
                    if p["domain"] == domain and p["kind"] == kind
                }
                domains[domain][kind] = len(
                    {m["identity_key"] for m in membership if m["partition"] in keys}
                )
        require(
            domains == receipt["domains"] and len(domains) >= max(3, targets["new_domains"]),
            "Domain counts contradict retained membership",
        )
        self.real_dataset_ids.update(r["dataset_id"] for r in by_kind["datasets"])
        return {
            "counts": counts,
            "repositories": repositories,
            "domains": domains,
            "verified_source_snapshots": snapshot_count,
        }

    def check_omics(self):
        config = self.ledger["omics"]
        receipt = self.payload(config["receipt"])
        self.ref(config["protocol"])
        require(
            receipt["schema_version"] == "omics_qualification_v1" and receipt["status"] == "PASS",
            "Omics qualification failed",
        )
        require(
            receipt["protocol_sha256"] == config["protocol"]["sha256"], "Omics protocol changed"
        )
        records = _rows(self.ref(receipt["artifact"]))
        ids = {r["dataset_id"] for r in records}
        require(
            len(ids) == len(records) == receipt["unique_study_ids"] and len(ids) >= 2,
            "Omics unique identity count mismatch",
        )
        require(
            {r["source"] for r in records} >= {"PRIDE", "Metabolomics Workbench"},
            "Real proteomics and metabolomics inputs required",
        )
        require(
            receipt.get("independent_study_groups") is None
            and receipt["scientific_status"].startswith("METADATA_COMPATIBILITY_ONLY"),
            "Omics metadata cannot establish independent biological sample counts",
        )
        network = receipt["network_control"]
        require(
            network["blocked_probe_count"] >= 1
            and network["unexpected_requests"] == 0
            and receipt["offline_byte_identity"] is True,
            "Omics offline verification failed",
        )
        require(
            sum(p["record_count"] for p in receipt["partitions"]) == len(records),
            "Omics partition count mismatch",
        )
        hashes = set()
        raw_inputs = []
        for snapshot in receipt["source_snapshots"]:
            raw_path = self.ref(
                {
                    "path": snapshot["object_path"],
                    "sha256": snapshot["sha256"],
                    "size_bytes": snapshot["size_bytes"],
                }
            )
            require(
                snapshot["status_code"] == 200 and snapshot["url"].startswith("https://"),
                "Invalid omics source snapshot",
            )
            hashes.add(snapshot["sha256"])
            raw_inputs.append(_json(raw_path))
        for record in records:
            provenance = _provenance(record)
            require(provenance.get("source_locator"), "Omics source locator missing")
            snapshot = provenance.get("metadata", {}).get("cache_snapshot", {})
            require(
                snapshot.get("cache_content_sha256") in hashes,
                "Omics record lacks matching raw snapshot",
            )
        require(
            len(raw_inputs) == len(receipt["partitions"]),
            "Omics partition/source snapshot inventory differs",
        )
        indexed = {r["dataset_id"]: r for r in records}
        for partition, raw in zip(receipt["partitions"], raw_inputs, strict=True):
            source_rows = (
                raw if isinstance(raw, list) else [raw] if "study_id" in raw else list(raw.values())
            )
            source_ids = [r.get("accession") or r.get("study_id") for r in source_rows]
            require(
                len(source_ids) == len(set(source_ids)) == partition["record_count"]
                and set(source_ids) <= ids,
                "Omics partition count/IDs disagree with raw source",
            )
            require(
                digest([indexed[key] for key in source_ids]) == partition["records_digest"],
                "Omics normalized partition payload digest changed",
            )
        self.real_dataset_ids.update(ids)
        return {
            "unique_study_ids": len(ids),
            "source_snapshots": len(hashes),
            "scientific_status": receipt["scientific_status"],
        }

    def benchmark_input(self, metadata):
        rows = _rows(self.ref(metadata, size=False))
        kind = metadata["identity_kind"]
        require(kind in {"dataset", "literature"}, "Unknown benchmark identity grain")
        records = {}
        for row in rows:
            key = _record_id(row, kind)
            require(
                key and (key not in records or records[key] == row),
                "Missing/conflicting benchmark identity",
            )
            records[key] = row
        order = sorted(records, key=lambda key: (hashlib.sha256(key.encode()).hexdigest(), key))
        require(
            len(rows) == metadata["input_rows"]
            and len(records) == metadata["unique_ids"]
            and len(rows) - len(records) == metadata["identical_duplicate_rows"],
            "Benchmark input counts contradict records",
        )
        require(
            digest(order) == metadata["identity_order_sha256"],
            "Benchmark ordered identity digest mismatch",
        )
        return [records[key] for key in order]

    def _catalog(self, path, kind):
        with sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True) as connection:
            rows = connection.execute(
                "SELECT c.id,c.digest,v.payload FROM current c JOIN versions v USING(kind,id,digest) WHERE c.kind=? AND c.valid=1 ORDER BY c.id",
                (kind,),
            ).fetchall()
            versions = connection.execute(
                "SELECT count(*) FROM versions WHERE kind=?", (kind,)
            ).fetchone()[0]
        result = []
        for identity, recorded, raw in rows:
            value = json.loads(raw)
            require(
                digest(value) == recorded and _record_id(value, kind) == identity,
                "Catalog stored payload or identity digest mismatch",
            )
            result.append(value)
        return result, versions

    def check_scale(self):
        path = self.ref(self.ledger["scale"]["receipt"])
        receipt = _json(path)
        require(
            receipt["schema_version"] == "phase2_real_metadata_benchmark_v1",
            "Unsupported native scale schema",
        )
        require(
            receipt["status"]
            == receipt["engineering_status"]
            == receipt["metadata_evaluation_status"]
            == "PASS",
            "Scale engineering or metadata evaluation failed",
        )
        for name in ("engineering_checks", "metadata_evaluation_checks"):
            require(
                receipt[name] and all(v is True for v in receipt[name].values()),
                f"Nonpassing {name}",
            )
        require(
            receipt["expert_validation"] == "PENDING_EXPERT_REVIEW"
            and receipt["calibration_status"] == "UNCALIBRATED_HEURISTIC",
            "Scale metadata/synthetic evidence masquerades as expert calibration",
        )
        require(receipt["network_attempts"] == [], "Scale run attempted external network")
        require(
            receipt["command"]
            and receipt["working_directory"]
            and COMMIT.fullmatch(receipt["source_commit"]),
            "Scale execution identity missing",
        )
        for name, expected in receipt["source_file_sha256"].items():
            current = self.root / name
            if sha256(current) != expected:
                original = Path(receipt["working_directory"]) / name
                require(
                    original.is_file()
                    and sha256(original) == expected
                    and _code_digest(original.read_bytes()) == _code_digest(current.read_bytes()),
                    f"Scale receipt bound to stale runtime file: {name}",
                )
        require(
            set(receipt["source_file_sha256"])
            >= {
                "litdatamatcher/phase2_benchmark.py",
                "litdatamatcher/data_plane.py",
                "litdatamatcher/scientific_v2.py",
                "litdatamatcher/v2.py",
                "litdatamatcher/semantic_runtime.py",
            },
            "Scale runtime source coverage incomplete",
        )
        artifacts = {}
        for name, expected in receipt["artifact_hashes"].items():
            candidate = self.resolve(name, path.parent)
            require(
                candidate.is_relative_to(path.parent), "Scale artifact escapes receipt directory"
            )
            artifacts[name.replace("\\", "/")] = self.ref(
                {"path": str(candidate), "sha256": expected}, size=False
            )
        plan = self.payload(
            {"path": receipt["plan_path"], "sha256": receipt["plan_sha256"]}, size=False
        )
        self.ref({"path": plan["protocol_path"], "sha256": plan["protocol_sha256"]}, size=False)
        for item in plan["acquisition_receipts"]:
            self.ref(item, size=False)
        require(
            plan["input_files"] == receipt["input_files"]
            and plan["combined_dataset"] == receipt["combined_dataset"],
            "Scale inputs differ from predeclared plan",
        )
        inputs = [
            (item["identity_kind"], self.benchmark_input(item)) for item in receipt["input_files"]
        ]
        literature = [r for kind, rows in inputs if kind == "literature" for r in rows]
        expected_datasets = {
            r["dataset_id"]: r for kind, rows in inputs if kind == "dataset" for r in rows
        }
        datasets = self.benchmark_input(receipt["combined_dataset"])
        require(
            {r["dataset_id"]: r for r in datasets} == expected_datasets,
            "Combined scale catalog differs from acquired records",
        )
        require(
            len(literature) >= 1000 and len(datasets) >= 300,
            "Real scale input below declared acquisition targets",
        )
        points = receipt["points"]
        require(
            len(points) >= 3
            and len({(p["literature_count"], p["dataset_count"]) for p in points}) >= 3,
            "Scale requires three distinct actual operating points",
        )
        require(
            points[-1]["literature_count"] == len(literature)
            and points[-1]["dataset_count"] == len(datasets),
            "Largest scale point omits final input",
        )
        bounds = plan["bounds"]
        require(
            0 < receipt["memory"]["sampled_process_tree_peak_rss_bytes"] <= bounds["rss_bytes"],
            "Scale memory exceeds declared bound",
        )
        for point in points:
            directory = f"literature_{point['literature_count']}_datasets_{point['dataset_count']}"
            require(
                _json(artifacts[directory + "/POINT_RECEIPT.json"]) == point,
                "Point summary differs from retained observation",
            )
            for kind, expected in (
                ("literature", literature[: point["literature_count"]]),
                ("dataset", datasets[: point["dataset_count"]]),
            ):
                rows, versions = self._catalog(
                    artifacts[directory + "/catalog/catalog.sqlite3"], kind
                )
                require(
                    rows == sorted(expected, key=lambda r: _record_id(r, kind))
                    and versions == len(rows),
                    "Scale catalog contains missing/duplicate/changed payloads",
                )
                require(
                    digest(rows) == point["catalog_payload_digests"][kind],
                    "Scale payload digest differs from catalog",
                )
                cache = point["cache_replay"][kind]
                require(
                    cache["hits"] == cache["items"] == len(rows)
                    and cache["misses"] == cache["reinserted_records"] == 0
                    and cache["hit_rate"] == 1.0,
                    "Cache count/denominator contradiction",
                )
            require(
                0 <= point["fts_queries"]["p95_seconds"] <= bounds["query_p95_seconds"]
                and point["fts_queries"]["n"] > 0,
                "Query latency bound failed/empty",
            )
            require(
                0
                <= point["matching"]["seconds_per_1000_assessments"]
                <= bounds["matching_seconds_per_1000"]
                and point["matching"]["items"] > 0,
                "Matching bound failed/empty",
            )
        recovery = receipt["recovery"]
        require(recovery["status"] == "PASS", "Recovery unsuccessful")
        runs = recovery["child_runs"]
        require(
            len(runs) == 2
            and [r["exit_code"] for r in runs] == [71, 0]
            and len({r["process_id"] for r in runs}) == 2,
            "Recovery lacks actual interrupted/resumed separate processes",
        )
        for run in runs:
            saved = self.payload(
                {"path": run["receipt_path"], "sha256": run["receipt_sha256"]}, size=False
            )
            require(
                all(run.get(k) == v for k, v in saved.items()),
                "Recovery summary contradicts child receipt",
            )
            require(
                run["input_sha256"] == receipt["combined_dataset"]["sha256"] and run["command"],
                "Recovery input/command missing",
            )
        first, resumed = runs
        require(
            0 < first["current_count"] < len(datasets)
            and first["current_count"] == first["version_count"] == resumed["existing_count"],
            "Recovery partial transaction counts contradictory",
        )
        require(
            set(first["final_ids"]) == set(resumed["skipped_existing_ids"])
            and not (set(resumed["inserted_ids"]) & set(first["final_ids"])),
            "Recovery reinserted committed IDs",
        )
        rows, versions = self._catalog(artifacts["recovery_catalog/catalog.sqlite3"], "dataset")
        require(
            rows == sorted(datasets, key=lambda r: r["dataset_id"])
            and len(rows) == versions == resumed["current_count"] == resumed["version_count"],
            "Recovery final catalog differs from complete input",
        )
        require(
            digest(rows)
            == resumed["final_payload_digest"]
            == recovery["clean_catalog_payload_digest"]
            and resumed["final_ids"] == [r["dataset_id"] for r in rows],
            "Recovery final IDs/payload digest mismatch",
        )
        rankings = _json(artifacts["RANKING_EVALUATION.json"])
        metrics = self.check_rankings(rankings, datasets, plan)
        require(
            metrics == receipt["metadata_label_counts"],
            "Scale label totals/denominators differ from retained pair labels",
        )
        semantic = receipt["semantic_baseline"]
        require(
            semantic["status"] == "PASS"
            and semantic["encoding"]["items"] == len(datasets)
            and SHA.fullmatch(semantic["vectors_sha256"]),
            "Pretrained retrieval baseline missing/empty",
        )
        require(
            semantic["model"]["revision"] and semantic["model"]["files"],
            "Pretrained model provenance missing",
        )
        return {
            "literature": len(literature),
            "datasets": len(datasets),
            "points": len(points),
            "queries": len(rankings),
            "pair_label_counts": metrics,
            "recovery_exits": [r["exit_code"] for r in runs],
        }

    def check_rankings(self, rankings, datasets, plan):
        records = {r["dataset_id"]: r for r in datasets}
        universe = sorted(records)
        require(
            rankings and [r["query"] for r in rankings] == plan["queries"],
            "Retrieval queries differ from frozen plan",
        )
        counts = {
            "OBSERVED_FIT": 0,
            "OBSERVED_MISMATCH": 0,
            "UNKNOWN": 0,
            "wrong_modality": 0,
            "wrong_organism": 0,
        }
        for row in rankings:
            require(
                row["candidate_count"] == len(records)
                and row["candidate_ids_sha256"] == digest(universe),
                "Retrieval candidate universe changed",
            )
            labels, query = row["labels"], row["query"]
            require(
                set(labels) == set(records) and query["label_origin"] == "source_determined",
                "Retrieval labels omit candidates or substitute expert labels",
            )
            require(
                query["anchor_record_sha256"] == digest(records[query["anchor_dataset_id"]]),
                "Query anchor changed",
            )
            for key, item in labels.items():
                record, states = records[key], []
                require(
                    item["label_origin"] == "source_determined"
                    and item["source_record_sha256"] == digest(record),
                    "Retrieval label lacks source-bound provenance",
                )
                provenance = _provenance(record)
                locators = provenance if isinstance(provenance, list) else [provenance]
                locator = next(
                    (
                        p.get("source_locator") or p.get("source_url")
                        for p in locators
                        if isinstance(p, dict) and (p.get("source_locator") or p.get("source_url"))
                    ),
                    None,
                )
                require(
                    locator and locator == item["source_locator"],
                    "Retrieval source locator changed",
                )
                require(
                    len(item["field_checks"]) == len(query["requirements"]) > 0,
                    "Missing requirement judgments",
                )
                for requirement, observed in zip(
                    query["requirements"], item["field_checks"], strict=True
                ):
                    field = requirement["field"]
                    require(
                        field in {"species", "modality"}, "Undeclared retrieval reference field"
                    )
                    raw = record.get("organisms" if field == "species" else "assay_types", [])
                    table = plan["reference_species" if field == "species" else "reference_assays"]
                    normalized = (
                        [" ".join(str(v).casefold().split()) for v in raw]
                        if isinstance(raw, list)
                        else []
                    )
                    known = {table[v] for v in normalized if v in table}
                    expected = requirement["expected"]
                    state = (
                        "MATCH"
                        if expected in known
                        else "UNKNOWN"
                        if not known or len(normalized) != sum(v in table for v in normalized)
                        else "MISMATCH"
                    )
                    require(
                        observed["field"] == field
                        and observed["expected"] == expected
                        and observed["recognized_source_values"] == sorted(known)
                        and observed["state"] == state,
                        "Retained field judgment contradicts source metadata/reference table",
                    )
                    if state == "MISMATCH":
                        counts["wrong_organism" if field == "species" else "wrong_modality"] += 1
                    states.append(state)
                expected_label = (
                    "OBSERVED_MISMATCH"
                    if "MISMATCH" in states
                    else "UNKNOWN"
                    if "UNKNOWN" in states
                    else "OBSERVED_FIT"
                )
                require(
                    item["label"] == expected_label,
                    "Pair label contradicts field-level observations",
                )
                counts[expected_label] += 1
            assessments = row["compatibility_assessments"]
            require(
                len(assessments) == len(records)
                and {a["dataset_id"] for a in assessments} == set(records),
                "Missing/duplicate compatibility assessments",
            )
            for item in assessments:
                label = labels[item["dataset_id"]]["label"]
                require(
                    type(item["is_qualified"]) is bool
                    and item["is_qualified"] == (label == "OBSERVED_FIT"),
                    "Observed fit excluded or mismatch/unknown promoted",
                )
                require(
                    item["score_type"] == "UNCALIBRATED_HEURISTIC" and math.isfinite(item["score"]),
                    "Invalid/calibrated retrieval score",
                )
            methods = row["methods"]
            require(
                set(methods)
                >= {
                    "lexical",
                    "pretrained_semantic",
                    "compatibility_only",
                    "compatibility_lexical",
                    "compatibility_semantic",
                    "heuristic_without_eligibility",
                },
                "Required real retrieval/ablation baseline absent",
            )
            for result in methods.values():
                expected = ranking_metrics(result["order"], labels)
                for key, value in expected.items():
                    actual = result["metrics"].get(key)
                    require(
                        (
                            isinstance(value, float)
                            and isinstance(actual, (float, int))
                            and math.isclose(value, actual, rel_tol=1e-12, abs_tol=1e-12)
                        )
                        or actual == value,
                        f"Retrieval metric/denominator contradiction: {key}",
                    )
        require(
            all(value > 0 for value in counts.values()),
            "Required observed fit/mismatch/unknown challenge denominator is empty",
        )
        return counts

    def check_review(self):
        review = self.payload(self.ledger["independent_review"])
        require(
            review["schema_version"] == "final_independent_review_v1"
            and review["status"] == "PASS",
            "Independent final review missing/failed",
        )
        self.source_bound(review, exact=False)
        require(
            review.get("reviewer_id")
            and review.get("implementer_id")
            and review["reviewer_id"] != review["implementer_id"],
            "Review producer is not independent of implementation",
        )
        self.ref(review["report"])
        require(
            review.get("expert_status") == "PENDING_EXPERT_REVIEW",
            "Functional review is not independent expert scientific validation",
        )
        require(isinstance(review["findings"], list), "Review finding disposition list missing")
        for finding in review["findings"]:
            severity = str(finding["severity"]).upper()
            require(
                severity in {"CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO", "P0", "P1", "P2", "P3"},
                "Unknown review severity",
            )
            if severity in {"CRITICAL", "HIGH", "P0", "P1"}:
                require(
                    finding["status"] in {"FIXED_VALIDATED", "DISMISSED_WITH_EVIDENCE"}
                    and finding.get("evidence"),
                    "Open critical/high independent review finding",
                )
                for ref in finding["evidence"]:
                    self.ref(ref)
        return {
            "reviewer_id": review["reviewer_id"],
            "findings": len(review["findings"]),
            "open_critical_high": 0,
        }

    def check_protected(self):
        refs = self.ledger["protected_alpha"]
        require(
            set(refs) == set(ALPHA_HASHES), "Exactly seven named protected alpha artifacts required"
        )
        for name, expected in ALPHA_HASHES.items():
            require(
                refs[name]["sha256"] == expected,
                f"Protected authority hash changed in ledger: {name}",
            )
            self.ref(refs[name])
        return {"read_only_hash_checks": len(refs), "sealed_holdout_executed": False}

    def check_git(self):
        config = self.ledger["git"]
        head = _git(self.root, "rev-parse", "HEAD").decode().strip()
        branch = _git(self.root, "branch", "--show-current").decode().strip()
        require(
            branch == config["branch"] and head == config["checkpoint"],
            "Final branch/checkpoint differs from current checkout",
        )
        require(
            not _git(self.root, "status", "--porcelain").strip(),
            "Final checkout contains uncommitted/untracked changes",
        )
        _git(
            self.root,
            "merge-base",
            "--is-ancestor",
            self.ledger["source"]["functional_commit"],
            head,
        )
        observed = self.command(config["remote_observation"], source=False)
        remote = _git(self.root, "remote", "get-url", config["remote"]).decode().strip()
        argv = observed["argv"]
        require(
            "ls-remote" in argv and (config["remote"] in argv or remote in argv),
            "Remote checkpoint observation did not query configured remote",
        )
        ref = "refs/heads/" + config["remote_branch"]
        require(ref in argv, "Remote observation did not query final branch")
        rows = self.ref(observed["stdout"]).read_text("utf-8-sig").splitlines()
        values = [
            line.split()[0] for line in rows if len(line.split()) == 2 and line.split()[1] == ref
        ]
        require(
            values == [head] and observed["source_commit"] == head,
            "Recorded remote checkpoint differs from final current commit",
        )
        timestamp = int(_git(self.root, "show", "-s", "--format=%ct", head).decode())
        require(
            _time(observed["started_at"]).timestamp() >= timestamp,
            "Remote observation predates checkpoint commit",
        )
        return {
            "branch": branch,
            "checkpoint": head,
            "recorded_remote_checkpoint": values[0],
            "remote_observed_at": observed["finished_at"],
            "network_requeried_by_validator": False,
        }

    def check_operations(self):
        require(hasattr(self, "operations_pending"), "Requirement matrix must pass first")
        require(
            not self.operations_pending,
            "Final operations requirements still pending: " + ", ".join(self.operations_pending),
        )
        return {"operations_requirements": len(OPERATIONS_IDS)}

    def validate(self):
        self.junit = []
        for name, fn in (
            ("source", self.check_source),
            ("requirements", self.check_matrix),
            ("junit", self.check_junit),
            ("distribution", self.check_distribution),
            ("clean_install", self.check_install),
            ("offline", self.check_offline),
            ("acquisition", self.check_acquisition),
            ("omics", self.check_omics),
            ("cases", self.check_cases),
            ("scale_retrieval_recovery", self.check_scale),
            ("independent_review", self.check_review),
            ("protected_alpha", self.check_protected),
        ):
            self.run_check(name, fn)
        self.run_check("operations_requirements", self.check_operations, "operations")
        self.run_check("git_closeout", self.check_git, "operations")
        ready = all(r["status"] == "PASS" for r in self.checks if r["axis"] == "technical")
        closed = ready and all(
            r["status"] == "PASS" for r in self.checks if r["axis"] == "operations"
        )
        return {
            "schema_version": "final_campaign_acceptance_v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "ledger": str(self.path),
            "ledger_sha256": sha256(self.path),
            "mode": self.mode,
            "status": "PASS" if ready and (self.mode == "technical" or closed) else "FAIL",
            "technical_readiness": "READY" if ready else "NOT_READY",
            "operations_closure": "COMPLETE" if closed else "PENDING",
            "expert_validation": "PENDING_EXPERT_REVIEW",
            "calibration_status": "UNCALIBRATED_HEURISTIC",
            "junit_counts": self.junit,
            "native_run_observations": [
                {k: v for k, v in item.items() if k != "dossiers"}
                for item in self.native_runs.values()
            ],
            "checks": self.checks,
            "limits": [
                "Retained execution evidence is hash-bound but its producers are not cryptographically authenticated.",
                "Remote equality is at the recorded git ls-remote observation; this validator does not contact the server.",
                "Metadata compatibility does not establish claim accuracy, biological independence, statistical power, or expert calibration.",
            ],
        }


def ranking_metrics(order, labels):
    require(
        len(order) == len(set(order)) and set(order) == set(labels),
        "Retrieval order omits/duplicates candidates",
    )
    positive = {k for k, v in labels.items() if v["label"] == "OBSERVED_FIT"}
    negative = {k for k, v in labels.items() if v["label"] == "OBSERVED_MISMATCH"}
    known = positive | negative
    judged = [k for k in order[:5] if k in known]
    hits5, hits10 = sum(k in positive for k in judged), sum(k in positive for k in order[:10])
    known_order = [k for k in order if k in known]
    dcg = sum((k in positive) / math.log2(i + 2) for i, k in enumerate(known_order[:5]))
    ideal = sum(1 / math.log2(i + 2) for i in range(min(5, len(positive))))
    first = next((i + 1 for i, k in enumerate(order) if k in positive), None)
    return {
        "candidate_count": len(order),
        "observed_fit_count": len(positive),
        "observed_mismatch_count": len(negative),
        "unknown_count": len(order) - len(known),
        "judged_precision_at_5": hits5 / len(judged) if judged else None,
        "precision_at_5_numerator": hits5,
        "precision_at_5_denominator": len(judged),
        "unknown_in_top5": len(order[:5]) - len(judged),
        "recall_at_10": hits10 / len(positive) if positive else None,
        "recall_at_10_numerator": hits10,
        "recall_at_10_denominator": len(positive),
        "reciprocal_rank": 1 / first if first else 0.0 if positive else None,
        "known_label_ndcg_at_5": dcg / ideal if ideal else None,
        "confirmed_invalid_top": bool(order and order[0] in negative),
        "unknown_top": bool(order and order[0] not in known),
    }


def validate_final_campaign(ledger_path, *, mode="technical", output=None):
    try:
        result = Validator(ledger_path, mode).validate()
    except (OSError, KeyError, TypeError, ValueError) as exc:
        result = {
            "schema_version": "final_campaign_acceptance_v1",
            "status": "FAIL",
            "technical_readiness": "NOT_READY",
            "operations_closure": "PENDING",
            "expert_validation": "PENDING_EXPERT_REVIEW",
            "junit_counts": [],
            "checks": [
                {
                    "check": "ledger",
                    "axis": "technical",
                    "status": "FAIL",
                    "errors": [f"{type(exc).__name__}: {exc}"],
                }
            ],
        }
    if output:
        atomic_json(output, result)
    return result


def example_ledger():
    def ref(name):
        return {"path": name, "sha256": "REPLACE_WITH_SHA256", "size_bytes": 0}

    return {
        "schema_version": SCHEMA,
        "source": {
            "root": "ABSOLUTE_SOURCE_CHECKOUT",
            "functional_commit": "FULL_COMMIT",
            "fingerprint": ref("SOURCE_FINGERPRINT.json"),
        },
        "matrix": ref("FINAL_REQUIREMENT_MATRIX.json"),
        "junit": [
            {"scope": scope, "xml": ref(scope + ".xml"), "command": ref(scope + "_command.json")}
            for scope in ("full", "targeted")
        ],
        "distribution": {
            "version": "0.3.0",
            "wheel": ref("dist/litdatamatcher-0.3.0-py3-none-any.whl"),
            "sdist": ref("dist/litdatamatcher-0.3.0.tar.gz"),
            "source_archive": ref("SOURCE.zip"),
        },
        "clean_install": [ref("INSTALL_COMMAND.json")],
        "offline": [ref("OFFLINE_COMMAND.json")],
        "cases": {"manifest": ref("CASES_MANIFEST.json")},
        "acquisition": {
            "receipt": ref("expanded/ACQUISITION_RECEIPT.json"),
            "plan": ref("expanded/PREDECLARED_PLAN.json"),
            "offline_replay": ref("expanded/OFFLINE_REPLAY.json"),
            "membership": ref("expanded/corpus/membership.jsonl"),
            "partition_receipts": [ref("expanded/partitions/ID/receipt.json")],
        },
        "omics": {
            "receipt": ref("omics_qualification/QUALIFICATION.json"),
            "protocol": ref("OMICS_PROTOCOL.json"),
        },
        "scale": {"receipt": ref("scale/RUN/BENCHMARK_RECEIPT.json")},
        "independent_review": ref("INDEPENDENT_REVIEW.json"),
        "protected_alpha": {name: ref(name) for name in ALPHA_HASHES},
        "git": {
            "branch": "codex/litdatamatcher-v2-build",
            "remote": "origin",
            "remote_branch": "codex/litdatamatcher-v2-build",
            "checkpoint": "FULL_COMMIT",
            "remote_observation": ref("REMOTE_COMMAND.json"),
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--mode", choices=("technical", "closeout"), default="technical")
    parser.add_argument("--snapshot-source", type=Path)
    parser.add_argument("--example-ledger", action="store_true")
    args = parser.parse_args(argv)
    if args.snapshot_source or args.example_ledger:
        result = source_snapshot(args.snapshot_source) if args.snapshot_source else example_ledger()
        if args.out:
            atomic_json(args.out, result)
        print(json.dumps(result, indent=2, ensure_ascii=True))
        return 0
    if not args.ledger:
        parser.error("--ledger is required unless generating a source snapshot/example")
    result = validate_final_campaign(args.ledger, mode=args.mode, output=args.out)
    print(json.dumps(result, indent=2, ensure_ascii=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
