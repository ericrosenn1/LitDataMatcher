"""Adversarial fixture contracts; these fixtures are not scientific evidence."""

import base64
import csv
import hashlib
import io
import json
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from litdatamatcher import final_campaign_acceptance as fc


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, ensure_ascii=False), encoding="utf-8")
    return ref(path)


def ref(path):
    return {"path": str(path), "sha256": fc.sha256(path), "size_bytes": path.stat().st_size}


def git(root, *argv):
    return (
        subprocess.run(["git", "-C", str(root), *argv], check=True, capture_output=True)
        .stdout.decode()
        .strip()
    )


@pytest.fixture
def workspace(tmp_path):
    root = tmp_path / "source"
    (root / "litdatamatcher").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "litdatamatcher/__init__.py").write_text('__version__ = "0.3.0"\n', encoding="utf-8")
    (root / "tests/test_sample.py").write_text(
        "def test_sample():\n    assert True\n", encoding="utf-8"
    )
    (root / "pyproject.toml").write_text(
        '[project]\nname="litdatamatcher"\nversion="0.3.0"\n', encoding="utf-8"
    )
    git(root, "init", "-b", "codex/final-fixture")
    git(root, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "add", ".")
    git(
        root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-m",
        "Fixture source",
    )
    snapshot = fc.source_snapshot(root)
    ledger = {
        "schema_version": fc.SCHEMA,
        "source": {
            "root": str(root),
            "functional_commit": snapshot["source_commit"],
            "fingerprint": write_json(tmp_path / "snapshot.json", snapshot),
        },
    }
    path = tmp_path / "ledger.json"
    write_json(path, ledger)
    validator = fc.Validator(path)
    validator.check_source()
    return validator


def command(v, name, *, outputs=(), stdout=None, exit_code=0, argv=None):
    if stdout is None:
        log = v.base / (name + ".log")
        log.write_text("retained process output\n", encoding="utf-8")
        stdout = ref(log)
    row = {
        "schema_version": "final_command_receipt_v1",
        "argv": argv or ["python", "-m", "pytest"],
        "cwd": str(v.root),
        "started_at": "2026-09-13T22:00:00+00:00",
        "finished_at": "2026-09-13T22:01:00+00:00",
        "exit_code": exit_code,
        "source_commit": v.snapshot["source_commit"],
        "source_fingerprint": v.snapshot["digest"],
        "dirty_functional_paths": [],
        "stdout": stdout,
        "inputs": [],
        "outputs": list(outputs),
    }
    return write_json(v.base / (name + "_command.json"), row)


def matrix(v):
    authority = write_json(v.base / "authority.json", {"test_fixture": True})
    rows = []
    for name in sorted(fc.REQUIREMENT_IDS):
        expert = name.startswith("EXPERT.")
        mandatory = name.startswith(("ALPHA.", "PHASE2.", "FINAL."))
        rows.append(
            {
                "id": name,
                "criterion": "Fixture criterion",
                "requirement_sources": [authority],
                "evidence": [authority],
                "classification": "OPTIONAL_FUTURE_VALIDATION" if expert else "COMPLETE_VALIDATED",
                "mandatory_for_software_complete": mandatory,
                "blocks_project_completion": False,
                "expert_status": "PENDING_EXPERT_REVIEW",
            }
        )
    return {"requirements": rows}


def package_archives(v):
    metadata = b"Name: litdatamatcher\nVersion: 0.3.0\n"
    files = {
        "litdatamatcher/__init__.py": (v.root / "litdatamatcher/__init__.py").read_bytes(),
        "litdatamatcher-0.3.0.dist-info/METADATA": metadata,
    }
    record = "litdatamatcher-0.3.0.dist-info/RECORD"
    stream = io.StringIO()
    writer = csv.writer(stream, lineterminator="\n")
    for name, content in files.items():
        encoded = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip("=")
        writer.writerow([name, "sha256=" + encoded, len(content)])
    writer.writerow([record, "", ""])
    files[record] = stream.getvalue().encode()
    wheel = v.base / "litdatamatcher-0.3.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, content in files.items():
            archive.writestr(name, content)
    sdist = v.base / "litdatamatcher-0.3.0.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        for name, content in {
            "litdatamatcher/__init__.py": files["litdatamatcher/__init__.py"],
            "PKG-INFO": metadata,
        }.items():
            item = tarfile.TarInfo("litdatamatcher-0.3.0/" + name)
            item.size = len(content)
            archive.addfile(item, io.BytesIO(content))
    source = v.base / "source.zip"
    with zipfile.ZipFile(source, "w") as archive:
        for name in v.snapshot["files"]:
            archive.writestr(name, (v.root / name).read_bytes())
    v.ledger["distribution"] = {
        "version": "0.3.0",
        "wheel": ref(wheel),
        "sdist": ref(sdist),
        "source_archive": ref(source),
    }
    return wheel


def test_exact_requirement_universe_is_pinned():
    assert len(fc.REQUIREMENT_IDS) == 74
    assert len(fc.ALPHA_HASHES) == 7


def test_source_changed_after_tests_is_rejected(workspace):
    v = workspace
    (v.root / "litdatamatcher/__init__.py").write_text("CHANGED = True\n", encoding="utf-8")
    with pytest.raises(fc.InvalidEvidence, match="differs"):
        v.check_source()


def test_new_untracked_functional_file_invalidates_source(workspace):
    v = workspace
    (v.root / "tests/test_unseen.py").write_text("def test_new(): pass\n", encoding="utf-8")
    with pytest.raises(fc.InvalidEvidence, match="differs"):
        v.check_source()


def test_jsonl_test_input_is_functional_source(workspace):
    v = workspace
    (v.root / "tests/input.jsonl").write_text('{"changed_test_input":true}\n', encoding="utf-8")
    with pytest.raises(fc.InvalidEvidence, match="differs"):
        v.check_source()


def test_committed_docs_change_does_not_force_scientific_rerun(workspace):
    v = workspace
    (v.root / "handoff.md").write_text("Documentation only\n", encoding="utf-8")
    git(v.root, "add", "handoff.md")
    git(
        v.root,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.invalid",
        "commit",
        "-m",
        "Document evidence",
    )
    assert v.check_source()["functional_commit"] != v.snapshot["source_commit"]


def test_recorded_dirty_test_state_is_preserved_and_same_bytes_accepted(workspace):
    v = workspace
    item = command(v, "dirty")
    data = fc._json(item["path"])
    data["dirty_functional_paths"] = ["tests/test_sample.py"]
    item = write_json(Path(item["path"]), data)
    assert v.command(item)["dirty_functional_paths"] == ["tests/test_sample.py"]


@pytest.mark.parametrize(
    "mutation", ["missing", "downgrade", "blocking", "expert", "mandatory_flag"]
)
def test_matrix_mutations_fail_closed(workspace, mutation):
    v, value = workspace, matrix(workspace)
    if mutation == "missing":
        value["requirements"].pop()
    elif mutation == "downgrade":
        value["requirements"][0]["classification"] = "OPTIONAL_FUTURE_VALIDATION"
    elif mutation == "blocking":
        value["requirements"][0]["blocks_project_completion"] = True
    elif mutation == "mandatory_flag":
        next(r for r in value["requirements"] if r["id"] == "FINAL.PUSH")[
            "mandatory_for_software_complete"
        ] = False
    else:
        next(r for r in value["requirements"] if r["id"] == "EXPERT.CALIBRATION")[
            "classification"
        ] = "COMPLETE_VALIDATED"
    v.ledger["matrix"] = write_json(v.base / "matrix.json", value)
    with pytest.raises(fc.InvalidEvidence):
        v.check_matrix()


def test_matrix_acceptance_is_not_circular(workspace):
    v, value = workspace, matrix(workspace)
    for row in value["requirements"]:
        if row["id"] in fc.DERIVED_IDS:
            row.update(
                classification="COMPLETE_NEEDS_FINAL_REGRESSION",
                validation_contract="THIS_VALIDATOR",
            )
        if row["id"] in fc.OPERATIONS_IDS:
            row.update(
                classification="MACHINE_COMPLETABLE_REMAINING", blocks_project_completion=True
            )
    v.ledger["matrix"] = write_json(v.base / "matrix.json", value)
    assert len(v.check_matrix()["operations_pending"]) == 5
    with pytest.raises(fc.InvalidEvidence, match="operations requirements"):
        v.check_operations()


def test_distribution_reopens_and_compares_current_source(workspace):
    package_archives(workspace)
    result = workspace.check_distribution()
    assert result["wheel"]["version"] == result["sdist"]["version"] == "0.3.0"


def test_tampered_wheel_with_rehashed_outer_artifact_fails_record(workspace):
    v = workspace
    wheel = package_archives(v)
    with zipfile.ZipFile(wheel) as archive:
        data = {n: archive.read(n) for n in archive.namelist()}
    data["litdatamatcher/__init__.py"] = b"tampered runtime\n"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, value in data.items():
            archive.writestr(name, value)
    v.ledger["distribution"]["wheel"] = ref(wheel)
    with pytest.raises(fc.InvalidEvidence, match="RECORD mismatch"):
        v.check_distribution()


def test_zip_crc_corruption_rejected(workspace):
    wheel = package_archives(workspace)
    with zipfile.ZipFile(wheel) as archive:
        member = archive.infolist()[0]
        offset = member.header_offset + 30 + len(member.filename.encode()) + len(member.extra)
    content = bytearray(wheel.read_bytes())
    content[offset] ^= 1
    wheel.write_bytes(content)
    with pytest.raises(zipfile.BadZipFile):
        fc.inspect_archive(wheel, "wheel")


@pytest.mark.parametrize(
    "name", ["../escape.py", "/absolute.py", "C:/drive.py", "a\\escape.py", "a//escape.py"]
)
def test_unsafe_archive_paths_rejected(tmp_path, name):
    path = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(name, b"bad")
    if "\\" in name:
        # Windows' ZIP writer normalizes separators; mutate retained header bytes.
        path.write_bytes(path.read_bytes().replace(name.replace("\\", "/").encode(), name.encode()))
    with pytest.raises(fc.InvalidEvidence, match="Unsafe archive"):
        fc.inspect_archive(path, "source_archive")


def test_archive_symlink_rejected(tmp_path):
    path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        member = tarfile.TarInfo("link")
        member.type = tarfile.SYMTYPE
        member.linkname = "/external"
        archive.addfile(member)
    with pytest.raises(fc.InvalidEvidence, match="link/special"):
        fc.inspect_archive(path, "sdist")


@pytest.mark.parametrize(
    "tag,counts", [("failure", (1, 0, 0)), ("error", (0, 1, 0)), ("skipped", (0, 0, 1))]
)
def test_junit_actual_nonpassing_counts_are_reported(workspace, tag, counts):
    v = workspace
    path = v.base / "tests.xml"
    path.write_text(
        f'<testsuite tests="1" failures="{counts[0]}" errors="{counts[1]}" skipped="{counts[2]}"><testcase classname="tests.sample" name="test_real"><{tag}/></testcase></testsuite>',
        encoding="utf-8",
    )
    xml = ref(path)
    v.ledger["junit"] = [
        {"scope": s, "xml": xml, "command": command(v, s, outputs=[xml])}
        for s in ("full", "targeted")
    ]
    v.junit = []
    with pytest.raises(fc.InvalidEvidence, match="counts="):
        v.check_junit()
    assert v.junit[0][{"failure": "failures", "error": "errors", "skipped": "skipped"}[tag]] == 1


def test_junit_cannot_hide_failure_in_declared_zero(tmp_path):
    path = tmp_path / "lies.xml"
    path.write_text(
        '<testsuite tests="1" failures="0"><testcase name="failed"><failure/></testcase></testsuite>',
        encoding="utf-8",
    )
    with pytest.raises(fc.InvalidEvidence, match="contradicts"):
        fc.junit_counts(path)


def test_missing_command_receipt_cannot_be_replaced_by_pass_flag(workspace):
    v = workspace
    item = command(v, "missing")
    Path(item["path"]).unlink()
    with pytest.raises(fc.InvalidEvidence, match="Missing"):
        v.command(item)


def test_unknowns_are_not_negative_metric_denominators():
    labels = {
        "p": {"label": "OBSERVED_FIT"},
        "n": {"label": "OBSERVED_MISMATCH"},
        "u": {"label": "UNKNOWN"},
    }
    result = fc.ranking_metrics(["u", "p", "n"], labels)
    assert result["precision_at_5_denominator"] == 2
    assert result["judged_precision_at_5"] == 0.5
    assert result["recall_at_10_denominator"] == 1
    assert result["unknown_count"] == 1


def test_synthetic_expert_campaign_promotion_fails(workspace):
    v = workspace
    v.ledger["expert_validation"] = "EXPERT_CALIBRATED"
    write_json(v.path, v.ledger)
    result = fc.validate_final_campaign(v.path)
    assert result["status"] == "FAIL"
    assert result["expert_validation"] == "PENDING_EXPERT_REVIEW"


def test_protected_hash_authority_cannot_be_rewritten_in_ledger(workspace):
    v = workspace
    fixture = write_json(v.base / "fake_protected.json", {"test_fixture": True})
    v.ledger["protected_alpha"] = {name: fixture for name in fc.ALPHA_HASHES}
    with pytest.raises(fc.InvalidEvidence, match="authority hash"):
        v.check_protected()


def test_missing_evidence_returns_all_gates_and_failing_cli(workspace):
    v = workspace
    result = fc.validate_final_campaign(v.path)
    assert result["status"] == "FAIL"
    assert len(result["checks"]) == 14
    assert result["checks"][0]["status"] == "PASS"
    assert all(r["status"] == "FAIL" for r in result["checks"][1:])
    assert fc.main(["--ledger", str(v.path), "--out", str(v.base / "FAIL.json")]) == 1


def native_run(v, name, origin):
    directory = v.base / name
    directory.mkdir()
    inference = {
        "origin": origin,
        "local_files_only": True,
        "process_id": 100,
        "execution_id": "fixture-execution",
        "model_id": "fixture-model",
        "model_revision": "revision",
        "fingerprint": {"input_sha256": "1" * 64, "implementation_sha256": "2" * 64},
    }
    artifacts = []
    for filename, value in (
        ("inferences.jsonl", inference),
        ("questions.jsonl", {"question_id": "q"}),
        ("claims.jsonl", {"claim_id": "fixture"}),
        ("matches.jsonl", {"dataset_id": "d"}),
        ("scientific_dossiers.jsonl", {"dossier_id": "fixture"}),
    ):
        item = write_json(directory / filename, value)
        item.update(path=filename, validation="PASS")
        artifacts.append(item)
    return write_json(
        directory / "RUN_MANIFEST.json",
        {
            "execution_status": "PASS",
            "failures": [],
            "network": {"mode": "OFFLINE"},
            "models": [{"id": "fixture-model", "revision": "revision", "runtime": "fixture"}],
            "artifacts": artifacts,
            "inference": {
                "fresh_calls": int(origin == "fresh_local_inference"),
                "cache_replays": int(origin == "cache_replay"),
            },
        },
    )


@pytest.mark.parametrize("mutation", [None, "payload", "artifact", "origin", "model", "network"])
def test_native_offline_observations_and_mutations(workspace, mutation):
    v = workspace
    fresh = native_run(v, "fresh", "fresh_local_inference")
    replay = native_run(v, "replay", "cache_replay")
    payload = write_json(
        v.base / "scientific_payload.json", {"question_id": "q", "dataset_id": "d"}
    )
    observation = {
        "schema_version": "final_offline_observation_v1",
        "network_control": {"blocked_probe_count": 1, "unexpected_requests": 0},
        "fresh": {"origin": "fresh_local_inference", "manifest": fresh, "payload": payload},
        "replay": {"origin": "cache_replay", "manifest": replay, "payload": payload},
    }
    if mutation == "payload":
        observation["replay"]["payload"] = write_json(
            v.base / "changed_payload.json", {"question_id": "changed"}
        )
    elif mutation in {"artifact", "origin", "model"}:
        manifest = fc._json(replay["path"])
        item = v.base / "replay/inferences.jsonl"
        value = fc._json(item)
        value[
            {"artifact": "process_id", "origin": "origin", "model": "model_revision"}[mutation]
        ] = "MUTATED"
        write_json(item, value)
        if mutation != "artifact":
            manifest["artifacts"][0].update(sha256=fc.sha256(item), size_bytes=item.stat().st_size)
            observation["replay"]["manifest"] = write_json(Path(replay["path"]), manifest)
    elif mutation == "network":
        observation["network_control"]["blocked_probe_count"] = 0
    observed = write_json(v.base / "offline_observation.json", observation)
    v.ledger["offline"] = [command(v, "offline", stdout=observed)]
    if mutation:
        with pytest.raises(fc.InvalidEvidence):
            v.check_offline()
    else:
        assert v.check_offline()[0]["network_probe_count"] == 1


@pytest.mark.parametrize("mutation", [None, "checkout_import", "failed_cli", "installed_tamper"])
def test_installed_location_and_wheel_bytes_are_checked(workspace, mutation):
    v = workspace
    package_archives(v)
    v.check_distribution()
    prefix = v.base / "isolated"
    package = prefix / "site-packages/litdatamatcher/__init__.py"
    package.parent.mkdir(parents=True)
    package.write_bytes(v.wheel_files["litdatamatcher/__init__.py"])
    executable = prefix / "python.exe"
    executable.write_bytes(b"FIXTURE_ONLY")
    stdout = v.base / "help.log"
    stdout.write_text("usage: fixture --help\n", encoding="utf-8")
    cli = [
        {"name": n, "argv": [n, "--help"], "exit_code": 0, "stdout": ref(stdout)}
        for n in ("litdatamatcher", "litdatamatcher-v2")
    ]
    observation = {
        "schema_version": "final_install_observation_v1",
        "package_file": str(package),
        "package_version": "0.3.0",
        "sys_prefix": str(prefix),
        "executable": str(executable),
        "cwd": str(v.base),
        "cli_results": cli,
    }
    if mutation == "checkout_import":
        observation["package_file"] = str(v.root / "litdatamatcher/__init__.py")
    elif mutation == "failed_cli":
        cli[0]["exit_code"] = 2
    elif mutation == "installed_tamper":
        package.write_text("TAMPERED = True\n", encoding="utf-8")
    observed = write_json(v.base / "installed.json", observation)
    receipt = command(v, "install", stdout=observed)
    row = fc._json(receipt["path"])
    row["inputs"] = [v.ledger["distribution"]["wheel"]]
    v.ledger["clean_install"] = [write_json(Path(receipt["path"]), row)]
    if mutation:
        with pytest.raises(fc.InvalidEvidence):
            v.check_install()
    else:
        assert v.check_install()[0]["package_version"] == "0.3.0"


@pytest.mark.parametrize("mutation", [None, "same_author", "high_open", "stale", "expert"])
def test_independent_review_is_source_bound_and_high_findings_closed(workspace, mutation):
    v = workspace
    report = write_json(v.base / "review_report.json", {"fixture_review": True})
    row = {
        "schema_version": "final_independent_review_v1",
        "status": "PASS",
        "source_commit": v.snapshot["source_commit"],
        "source_fingerprint": v.snapshot["digest"],
        "reviewer_id": "review-agent",
        "implementer_id": "writer-agent",
        "report": report,
        "expert_status": "PENDING_EXPERT_REVIEW",
        "findings": [],
    }
    if mutation == "same_author":
        row["reviewer_id"] = row["implementer_id"]
    elif mutation == "high_open":
        row["findings"] = [{"severity": "high", "status": "OPEN"}]
    elif mutation == "stale":
        row["source_fingerprint"] = "0" * 64
    elif mutation == "expert":
        row["expert_status"] = "EXPERT_CALIBRATED"
    v.ledger["independent_review"] = write_json(v.base / "review.json", row)
    if mutation:
        with pytest.raises(fc.InvalidEvidence):
            v.check_review()
    else:
        assert v.check_review()["open_critical_high"] == 0


def tiny_rankings():
    from litdatamatcher.data_plane import digest

    source = {"source_locator": "https://fixture.invalid/source"}
    datasets = [
        {
            "dataset_id": "p",
            "organisms": ["human"],
            "assay_types": ["rna"],
            "source_provenance": source,
        },
        {
            "dataset_id": "n",
            "organisms": ["rat"],
            "assay_types": ["protein"],
            "source_provenance": source,
        },
        {"dataset_id": "u", "organisms": [], "assay_types": [], "source_provenance": source},
    ]
    requirements = [
        {"field": "species", "expected": "human"},
        {"field": "modality", "expected": "rna"},
    ]
    query = {
        "anchor_dataset_id": "p",
        "anchor_record_sha256": digest(datasets[0]),
        "label_origin": "source_determined",
        "requirements": requirements,
    }
    labels = {}
    for index, record in enumerate(datasets):
        fields = [
            {
                "field": r["field"],
                "expected": r["expected"],
                "recognized_source_values": record[
                    "organisms" if r["field"] == "species" else "assay_types"
                ],
                "state": ["MATCH", "MISMATCH", "UNKNOWN"][index],
            }
            for r in requirements
        ]
        labels[record["dataset_id"]] = {
            "source_record_sha256": digest(record),
            "source_locator": source["source_locator"],
            "label_origin": "source_determined",
            "label": ["OBSERVED_FIT", "OBSERVED_MISMATCH", "UNKNOWN"][index],
            "field_checks": fields,
        }
    order = ["p", "u", "n"]
    methods = {
        n: {"order": order, "metrics": fc.ranking_metrics(order, labels)}
        for n in (
            "lexical",
            "pretrained_semantic",
            "compatibility_only",
            "compatibility_lexical",
            "compatibility_semantic",
            "heuristic_without_eligibility",
        )
    }
    assessments = [
        {
            "dataset_id": k,
            "is_qualified": k == "p",
            "score_type": "UNCALIBRATED_HEURISTIC",
            "score": 0.5,
        }
        for k in order
    ]
    row = {
        "query": query,
        "candidate_count": 3,
        "candidate_ids_sha256": digest(sorted(order)),
        "labels": labels,
        "methods": methods,
        "compatibility_assessments": assessments,
    }
    plan = {
        "queries": [query],
        "reference_species": {"human": "human", "rat": "rat"},
        "reference_assays": {"rna": "rna", "protein": "protein"},
    }
    return [row], datasets, plan


@pytest.mark.parametrize(
    "mutation",
    [None, "denominator", "label", "missing_candidate", "unknown_promotion", "expert_score"],
)
def test_real_metadata_pair_contract_mutations(workspace, mutation):
    rankings, datasets, plan = tiny_rankings()
    row = rankings[0]
    if mutation == "denominator":
        row["methods"]["lexical"]["metrics"]["precision_at_5_denominator"] = 3
    elif mutation == "label":
        row["labels"]["n"]["label"] = "OBSERVED_FIT"
    elif mutation == "missing_candidate":
        row["methods"]["lexical"]["order"] = ["p", "n"]
    elif mutation == "unknown_promotion":
        next(a for a in row["compatibility_assessments"] if a["dataset_id"] == "u")[
            "is_qualified"
        ] = True
    elif mutation == "expert_score":
        row["compatibility_assessments"][0]["score_type"] = "EXPERT_CALIBRATED"
    if mutation:
        with pytest.raises(fc.InvalidEvidence):
            workspace.check_rankings(rankings, datasets, plan)
    else:
        assert workspace.check_rankings(rankings, datasets, plan) == {
            "OBSERVED_FIT": 1,
            "OBSERVED_MISMATCH": 1,
            "UNKNOWN": 1,
            "wrong_modality": 1,
            "wrong_organism": 1,
        }


def case_manifest(v):
    from litdatamatcher.scientific_dossier import build_dossier

    v.real_dataset_ids = {"d"}  # Earlier acquisition gates supply these in a complete validation.
    catalog = write_json(v.base / "catalog.jsonl", {"dataset_id": "d", "source": "FIXTURE"})
    source = v.base / "source_evidence.txt"
    source.write_text("A retained source statement for fixture validation.\n", encoding="utf-8")
    cases = []
    for index in range(6):
        question = {
            "question_id": f"q{index}",
            "question": f"Source assisted fixture {index}?",
            "source_evidence_ids": ["e"],
        }
        bundle = {
            "question_id": f"q{index}",
            "gap_status": "insufficient-coverage",
            "novelty_claim": "Limited to searched coverage",
            "evidence_items": [
                {
                    "evidence_id": "e",
                    "source_locator": "https://fixture.invalid/record",
                    "role": "direct_test",
                    "evidence_span": "A retained source statement for fixture validation.",
                }
            ],
        }
        dossier = build_dossier(
            question,
            bundle,
            {"dataset_id": "d", "compatibility_status": "UNKNOWN", "requirements": []},
            {"dataset_id": "d", "source": "FIXTURE"},
            ["Source-derived fixture; not scientific evidence"],
        )
        cases.append(
            {
                "case_id": f"case{index}",
                "domain": "A" if index < 3 else "B",
                "dossier": write_json(v.base / f"dossier{index}.json", dossier),
                "run_manifest": case_native(v, f"case_native{index}", dossier),
            }
        )
    run = command(v, "cases", outputs=[c["dossier"] for c in cases])
    return {
        "schema_version": "final_cases_manifest_v1",
        "source_commit": v.snapshot["source_commit"],
        "source_fingerprint": v.snapshot["digest"],
        "command": run,
        "data_origin": "real",
        "dataset_catalog": catalog,
        "cases": cases,
        "source_locators": [{"locator": "https://fixture.invalid/record", "artifact": ref(source)}],
    }


def case_native(v, name, dossier):
    path = v.base / name / "RUN_MANIFEST.json"
    if not path.exists():
        native_run(v, name, "fresh_local_inference")
    row = fc._json(path)
    output = write_json(path.parent / "scientific_dossiers.jsonl", dossier)
    item = next(x for x in row["artifacts"] if x["path"] == "scientific_dossiers.jsonl")
    item.update(sha256=output["sha256"], size_bytes=output["size_bytes"])
    return write_json(path, row)


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "quote",
        "locator",
        "catalog",
        "expert",
        "duplicate",
        "missing_dossier",
        "unacquired_catalog",
    ],
)
def test_case_evidence_and_scientific_status_mutations(workspace, mutation):
    v = workspace
    manifest = case_manifest(v)
    first = manifest["cases"][0]
    if mutation in {"quote", "locator", "catalog", "expert"}:
        dossier = fc._json(first["dossier"]["path"])
        if mutation == "quote":
            dossier["source_evidence"][0]["evidence_span"] = "An invented experimental conclusion"
        elif mutation == "locator":
            dossier["source_evidence"][0]["source_locator"] = "https://fixture.invalid/absent"
        elif mutation == "catalog":
            dossier["candidate_dataset"]["dataset_id"] = "NOT_RETAINED"
        elif mutation == "expert":
            dossier["calibration_status"] = "EXPERT_CALIBRATED"
        first["dossier"] = write_json(Path(first["dossier"]["path"]), dossier)
        # Rebind the changed summary too; semantic gates must still reject it.
        manifest["command"] = command(v, "cases", outputs=[c["dossier"] for c in manifest["cases"]])
    elif mutation == "duplicate":
        manifest["cases"][1]["dossier"] = first["dossier"]
    elif mutation == "missing_dossier":
        Path(first["dossier"]["path"]).unlink()
    elif mutation == "unacquired_catalog":
        v.real_dataset_ids = {"different_acquired_identity"}
    v.ledger["cases"] = {"manifest": write_json(v.base / "cases_manifest.json", manifest)}
    if mutation:
        with pytest.raises(fc.InvalidEvidence):
            v.check_cases()
    else:
        assert v.check_cases()["case_count"] == 6


def test_git_checkpoint_does_not_accept_mismatching_recorded_remote(workspace):
    v = workspace
    git(v.root, "remote", "add", "origin", "https://fixture.invalid/repository.git")
    log = v.base / "remote.log"
    log.write_text("0" * 40 + "\trefs/heads/codex/final-fixture\n", encoding="utf-8")
    observed = command(
        v,
        "remote",
        stdout=ref(log),
        argv=["git", "ls-remote", "origin", "refs/heads/codex/final-fixture"],
    )
    v.ledger["git"] = {
        "branch": "codex/final-fixture",
        "checkpoint": v.snapshot["source_commit"],
        "remote": "origin",
        "remote_branch": "codex/final-fixture",
        "remote_observation": observed,
    }
    with pytest.raises(fc.InvalidEvidence, match="remote checkpoint"):
        v.check_git()


@pytest.mark.parametrize("bad_offset", [False, True])
def test_source_json_string_span_uses_decoded_text_coordinates(bad_offset):
    text = 'A source says "quoted text".\nAnother retained line.'
    span = {"start": 0 if not bad_offset else 2, "end": len(text), "text": text}
    assert fc.source_span_matches(json.dumps({"body": text}), span) is not bad_offset


@pytest.mark.parametrize("promotion", [False, True])
def test_deterministic_source_context_stays_unasserted(workspace, promotion):
    v = workspace
    manifest = case_manifest(v)
    first = manifest["cases"][0]
    dossier = fc._json(first["dossier"]["path"])
    evidence = dossier["source_evidence"][0]
    evidence.update(
        evidence_origin="deterministic_source_passage",
        role="background",
        claim_status="NOT_ASSERTED",
        direction="inconclusive",
        answers_question=promotion,
    )
    text = evidence["evidence_span"]
    evidence["evidence_span"] = {"start": 0, "end": len(text), "text": text}
    first["dossier"] = write_json(Path(first["dossier"]["path"]), dossier)
    first["run_manifest"] = case_native(v, "case_native0", dossier)
    manifest["command"] = command(v, "cases", outputs=[c["dossier"] for c in manifest["cases"]])
    v.ledger["cases"] = {"manifest": write_json(v.base / "cases_manifest.json", manifest)}
    if promotion:
        with pytest.raises(fc.InvalidEvidence, match="promoted"):
            v.check_cases()
    else:
        assert v.check_cases()["case_count"] == 6


@pytest.mark.parametrize(
    "status,stage,accepted",
    [
        ("FAIL", "source_guard", False),
        ("PARTIAL", "source_guard", True),
        ("PARTIAL", "inference", False),
    ],
)
def test_native_case_fail_cannot_hide_behind_controller_success(workspace, status, stage, accepted):
    v = workspace
    manifest = case_manifest(v)
    first = manifest["cases"][0]
    path = Path(first["run_manifest"]["path"])
    run = fc._json(path)
    run.update(
        execution_status=status,
        failures=[
            {
                "stage": stage,
                "description": "retained rejection",
                "rejections": [{"reason": "not_source_supported"}],
            }
        ],
    )
    first["run_manifest"] = write_json(path, run)
    v.ledger["cases"] = {"manifest": write_json(v.base / "cases_manifest.json", manifest)}
    if accepted:
        assert v.check_cases()["case_count"] == 6
        assert next(r for r in v.native_runs.values() if r["execution_status"] == "PARTIAL")[
            "failures"
        ]
    else:
        with pytest.raises(fc.InvalidEvidence, match="native run failed"):
            v.check_cases()


def test_archive_cannot_add_unreviewed_functional_file(workspace):
    v = workspace
    package_archives(v)
    path = Path(v.ledger["distribution"]["source_archive"]["path"])
    with zipfile.ZipFile(path, "a") as archive:
        archive.writestr("litdatamatcher/unreviewed.py", b"UNREVIEWED = True\n")
    v.ledger["distribution"]["source_archive"] = ref(path)
    with pytest.raises(fc.InvalidEvidence, match="unreviewed functional"):
        v.check_distribution()


def test_technical_readiness_is_separate_from_unfinished_closeout(workspace, monkeypatch):
    v = workspace
    # Isolate report aggregation; every evidence gate is exercised separately above.
    for method in (
        "check_source",
        "check_matrix",
        "check_junit",
        "check_distribution",
        "check_install",
        "check_offline",
        "check_cases",
        "check_acquisition",
        "check_omics",
        "check_scale",
        "check_review",
        "check_protected",
        "check_git",
    ):
        monkeypatch.setattr(v, method, lambda: {"aggregation_fixture": True})
    v.operations_pending = ["FINAL.PUSH"]
    technical = v.validate()
    assert technical["status"] == "PASS"
    assert technical["technical_readiness"] == "READY"
    assert technical["operations_closure"] == "PENDING"
    v.checks = []
    v.mode = "closeout"
    closed = v.validate()
    assert closed["status"] == "FAIL"
    assert closed["technical_readiness"] == "READY"
