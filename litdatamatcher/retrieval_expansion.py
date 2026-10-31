"""Frozen larger candidate pools; no model inference and no question-text reads."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from contextlib import suppress
from pathlib import Path

from .controller import integration_lease
from .data_plane import atomic_json, atomic_write, digest
from .strict_artifacts import strict_json_loads

VERSION = "unlinked-pubmed-candidate-expansion-1.1"


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path):
    return strict_json_loads(Path(path).read_text(encoding="utf-8"))


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return [strict_json_loads(line) for line in handle if line.strip()]


def normalize_text(value):
    return " ".join(value.casefold().split())


def canonical_identifier(kind, value):
    """Only typed IDs and explicit EvidenceBench namespaces can name PubMed papers."""
    value = str(value or "").strip()
    kind = str(kind or "").casefold()
    if kind == "evidencebench_paper_id":
        match = re.fullmatch(r"(pubmed|pmc)_([0-9]+)", value, re.I)
        if not match:
            raise ValueError("Unresolved EvidenceBench paper namespace; numeric IDs are not PMIDs")
        kind, value = ("pmid" if match[1].lower() == "pubmed" else "pmcid"), match[2]
    if kind in {"pmid", "pmcid"}:
        prefix = r"(?:pmid[:\s]*)?" if kind == "pmid" else r"(?:pmcid[:\s]*)?(?:pmc)?"
        match = re.fullmatch(prefix + r"([0-9]+)", value, re.I)
        if match and int(match[1]) > 0:
            return f"{kind}:{int(match[1])}"
        raise ValueError(f"Invalid typed {kind}: {value}")
    if kind == "doi":
        value = re.sub(
            r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", "", value, flags=re.I
        ).casefold()
        if re.fullmatch(r"10\.[0-9]{4,9}/\S+", value):
            return "doi:" + value
        raise ValueError("Invalid typed DOI")
    raise ValueError(f"Unrecognized source identifier type: {kind}")


def paper_aliases(row):
    aliases = set()
    for field in ("pmid", "pmcid", "doi"):
        if row.get(field):
            aliases.add(canonical_identifier(field, row[field]))
    return aliases


def _inside(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Source path escapes the corpus root")
    return path


def _unchanged(path, record):
    stat = path.stat()
    if stat.st_size != record["bytes"] or stat.st_mtime_ns != record["mtime_ns"]:
        raise ValueError(f"Source changed since its qualified receipt: {path}")


def qualified_shards(corpus_root, snapshot):
    if snapshot.get("status") != "PASS" or snapshot.get("raw_checksums_recomputed") is not True:
        raise ValueError("Require a PASS annual-baseline qualification with raw checksums")
    shards = snapshot.get("shards", [])
    if not shards:
        raise ValueError("No qualified shards")
    for row in shards:
        if row.get("source") != "baseline" or row.get("status") != "PASS":
            raise ValueError("Only qualified annual-baseline shards may supply candidates")
        if not re.fullmatch(r"[0-9a-f]{64}", row.get("sha256", "")):
            raise ValueError("Missing qualified shard hash")
        _unchanged(_inside(corpus_root, row["relative_path"]), row)
    return shards


def excluded_identifiers(edges_path):
    import pyarrow.parquet as pq

    values, counts = set(), Counter()
    before = file_hash(edges_path)
    for batch in pq.ParquetFile(edges_path).iter_batches(
        columns=["identifier_type", "identifier_value"], batch_size=4096
    ):
        for row in batch.to_pylist():
            counts[str(row["identifier_type"])] += 1
            values.add(canonical_identifier(row["identifier_type"], row["identifier_value"]))
    if file_hash(edges_path) != before:
        raise ValueError("Question-paper metadata changed during exclusion scan")
    return values, dict(counts), before


def resolve_aliases(corpus_root, shards, initial, checkpoint, fingerprint, progress):
    """Compute exact observed alias closure; backward links require another pass."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    aliases, processed, pass_number = set(initial), set(), 1
    pass_initial_count = len(aliases)
    if checkpoint.exists():
        saved = read_json(checkpoint)
        checksum = saved.pop("content_sha256", None)
        if digest(saved) != checksum or saved.get("fingerprint") != fingerprint:
            raise ValueError("Alias checkpoint content or fingerprint failed")
        aliases.update(saved["aliases"])
        if saved.get("complete"):
            return aliases, saved
        processed.update(saved["processed_shards"])
        pass_number = saved["pass_number"]
        pass_initial_count = saved["pass_initial_count"]

    def save(complete=False):
        receipt = {
            "fingerprint": fingerprint,
            "complete": complete,
            "aliases": sorted(aliases),
            "processed_shards": sorted(processed),
            "pass_number": pass_number,
            "pass_initial_count": pass_initial_count,
            "direct_alias_count": len(initial),
            "resolved_alias_count": len(aliases),
            "scope": "Normalized identifier-only fixed-point closure across qualified annual shards",
        }
        atomic_json(checkpoint, {**receipt, "content_sha256": digest(receipt)})
        return receipt

    def column(table, field):
        values = pc.utf8_lower(pc.utf8_trim_whitespace(table[field]))
        if field == "pmid":
            values = pc.replace_substring_regex(values, r"^pmid[:\s]*", "")
        elif field == "pmcid":
            values = pc.replace_substring_regex(values, r"^pmcid[:\s]*", "")
            values = pc.replace_substring_regex(values, r"^pmc", "")
        else:
            values = pc.replace_substring_regex(values, r"^(https?://(dx\.)?doi\.org/|doi:\s*)", "")
        if field in {"pmid", "pmcid"}:
            values = pc.replace_substring_regex(values, r"^0+", "")
        return values

    while pass_number <= 8:
        for record in shards:
            relative = record["relative_path"]
            if relative in processed:
                continue
            path = _inside(corpus_root, relative)
            _unchanged(path, record)
            table = pq.read_table(path, columns=["pmid", "pmcid", "doi"], use_threads=False)
            masks = []
            for field in ("pmid", "pmcid", "doi"):
                targets = [
                    value.split(":", 1)[1] for value in aliases if value.startswith(field + ":")
                ]
                if targets:
                    masks.append(pc.is_in(column(table, field), value_set=pa.array(targets)))
            mask = masks[0]
            for extra in masks[1:]:
                mask = pc.or_(mask, extra)
            for row in table.filter(pc.fill_null(mask, False)).to_pylist():
                for field in ("pmid", "pmcid", "doi"):
                    if row.get(field):
                        with suppress(ValueError):
                            # Unparseable optional aliases remain unknown, never coerced.
                            aliases.add(canonical_identifier(field, row[field]))
            _unchanged(path, record)
            processed.add(relative)
            if len(processed) % 25 == 0:
                progress(
                    {
                        "stage": "identifier_closure",
                        "pass": pass_number,
                        "shards": len(processed),
                        "aliases": len(aliases),
                    }
                )
                save()
        if len(aliases) == pass_initial_count:
            return aliases, save(complete=True)
        pass_number += 1
        pass_initial_count = len(aliases)
        processed.clear()
        save()
    raise ValueError("Alias closure did not converge within eight full passes")


def _seed_benchmark(root):
    manifest = read_json(root / "BENCHMARK_MANIFEST.json")
    if manifest.get("status") != "PASS":
        raise ValueError("Seed benchmark must be PASS")
    for entry in manifest["outputs"]:
        path = root / entry["path"]
        if (
            path.resolve().parent != root
            or file_hash(path) != entry["sha256"]
            or path.stat().st_size != entry["bytes"]
        ):
            raise ValueError("Seed benchmark integrity failure")
    return manifest, read_rows(root / "documents.jsonl")


def _publish(out, count, view, selected, seed_root, seed_manifest, provenance):
    destination = out / f"pool-{count}" / view
    manifest_path = destination / "BENCHMARK_MANIFEST.json"
    if manifest_path.exists():
        prior = read_json(manifest_path)
        if prior.get("expansion_fingerprint") != provenance["fingerprint"]:
            raise ValueError("Existing candidate pool fingerprint differs")
        for entry in prior["outputs"]:
            if file_hash(destination / entry["path"]) != entry["sha256"]:
                raise ValueError("Existing candidate output is corrupt; retained for diagnosis")
        return prior
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("train.jsonl", "development.jsonl"):
        atomic_write(destination / name, (seed_root / name).read_bytes())
    path = destination / "documents.jsonl"
    temp = destination / ".documents.tmp"
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            value = dict(row)
            value["text"] = row["abstract"] if view == "abstract_only" else row["text"]
            handle.write(
                json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )
    temp.replace(path)
    outputs = [
        {
            "path": name,
            "sha256": file_hash(destination / name),
            "bytes": (destination / name).stat().st_size,
        }
        for name in ("train.jsonl", "development.jsonl", "documents.jsonl")
    ]
    manifest = {
        **seed_manifest,
        "schema_version": VERSION,
        "status": "PASS",
        "candidate_view": view,
        "counts": {**seed_manifest["counts"], "documents": count},
        "outputs": outputs,
        "expansion_fingerprint": provenance["fingerprint"],
        "expansion": provenance,
        "ordered_candidate_ids_sha256": digest([row["id"] for row in selected]),
        "purpose": "Expanded original-paper recovery diagnostic; unjudged distractors are not proven irrelevant",
    }
    atomic_json(manifest_path, manifest)
    return manifest


def _build_expansion(
    corpus_root,
    snapshot_path,
    edges_path,
    seed_root,
    out,
    *,
    sizes=(50000, 100000),
    views=("full", "abstract_only"),
    seed="retrieval-scale-20261031",
    progress=None,
):
    import pyarrow.parquet as pq

    corpus_root, snapshot_path, edges_path, seed_root, out = map(
        lambda p: Path(p).resolve(), (corpus_root, snapshot_path, edges_path, seed_root, out)
    )
    sizes = sorted(set(sizes))
    views = sorted(set(views))
    if (
        not sizes
        or any(type(n) is not int or n < 1 for n in sizes)
        or not views
        or not set(views) <= {"full", "abstract_only"}
    ):
        raise ValueError("Positive sizes and full/abstract_only views are required")
    if (
        out.is_relative_to(corpus_root)
        or corpus_root.is_relative_to(out)
        or out.is_relative_to(seed_root)
        or seed_root.is_relative_to(out)
    ):
        raise ValueError("Expansion outputs must be separate from source and seed data")
    notify = progress or (lambda value: None)
    snapshot = read_json(snapshot_path)
    shards = qualified_shards(corpus_root, snapshot)
    seed_manifest, selected = _seed_benchmark(seed_root)
    if sizes[0] < len(selected):
        raise ValueError("A candidate pool cannot drop seed documents")
    exclusions, edge_counts, edge_hash = excluded_identifiers(edges_path)
    for row in selected:
        exclusions.update(paper_aliases(row))
        if not isinstance(row.get("abstract"), str) or not row["abstract"].strip():
            raise ValueError("Both candidate views require nonempty seed abstracts")
    identity = {
        "version": VERSION,
        "seed": seed,
        "sizes": sizes,
        "views": views,
        "snapshot_sha256": file_hash(snapshot_path),
        "edges_sha256": edge_hash,
        "benchmark_sha256": file_hash(seed_root / "BENCHMARK_MANIFEST.json"),
        "implementation_sha256": file_hash(Path(__file__)),
    }
    fingerprint = digest(identity)
    with integration_lease(out / ".owner", "candidate-pool-expansion"):
        if (out / "EXPANSION.json").exists():
            saved = read_json(out / "EXPANSION.json")
            if saved["fingerprint"] != fingerprint:
                raise ValueError("Expansion configuration changed; choose a new output directory")
            if saved.get("status") == "PASS":
                if file_hash(out / "SEED_BENCHMARK_MANIFEST.json") != identity["benchmark_sha256"]:
                    raise ValueError("Completed seed benchmark receipt changed")
                for item in saved["manifest_receipts"]:
                    manifest_path = out / item["path"]
                    if file_hash(manifest_path) != item["sha256"]:
                        raise ValueError("Completed candidate manifest changed")
                    manifest = read_json(manifest_path)
                    for entry in manifest["outputs"]:
                        if file_hash(manifest_path.parent / entry["path"]) != entry["sha256"]:
                            raise ValueError("Completed candidate output changed")
                return {**saved, "reused": True}
        atomic_json(
            out / "EXPANSION.json",
            {"fingerprint": fingerprint, "inputs": identity, "status": "RUNNING"},
        )
        atomic_write(
            out / "SEED_BENCHMARK_MANIFEST.json",
            (seed_root / "BENCHMARK_MANIFEST.json").read_bytes(),
        )
        aliases, alias_receipt = resolve_aliases(
            corpus_root, shards, exclusions, out / "EXCLUSIONS.json", fingerprint, notify
        )
        seen_ids = {row["id"] for row in selected}
        seen_full = {normalize_text(row["text"]) for row in selected}
        seen_abstract = {normalize_text(row["abstract"]) for row in selected}
        counts = Counter()
        used = []
        manifests = []
        ordered = sorted(
            shards,
            key=lambda row: (
                hashlib.sha256((seed + row["relative_path"]).encode()).hexdigest(),
                row["relative_path"],
            ),
        )

        def publish_ready():
            if len(selected) not in sizes:
                return
            if (
                file_hash(snapshot_path) != identity["snapshot_sha256"]
                or file_hash(edges_path) != edge_hash
                or file_hash(seed_root / "BENCHMARK_MANIFEST.json") != identity["benchmark_sha256"]
            ):
                raise ValueError("Source or seed identity changed before publication")
            _seed_benchmark(seed_root)
            provenance = {
                "fingerprint": fingerprint,
                "inputs": identity,
                "seed_documents": seed_manifest["counts"]["documents"],
                "distractors": len(selected) - seed_manifest["counts"]["documents"],
                "identifier_edge_counts": edge_counts,
                "resolved_exclusion_aliases": alias_receipt["resolved_alias_count"],
                "exclusion_receipt_sha256": file_hash(out / "EXCLUSIONS.json"),
                "used_shards": list(used),
                "counts": dict(counts),
                "sampling": "Hash-ordered qualified shards, then hash-ordered canonical PMID within each shard; not a global uniform sample",
                "hardness": "Not assumed; retrieval performance is measured separately",
                "limitations": "Exact exclusions use recorded source namespaces/aliases. Missing source metadata does not prove scientific irrelevance. Abstract-only is diagnostic, not a replacement endpoint.",
            }
            for view in views:
                manifests.append(
                    _publish(
                        out, len(selected), view, selected, seed_root, seed_manifest, provenance
                    )
                )
            notify({"stage": "pool_published", "candidates": len(selected), "views": views})

        publish_ready()
        for record in ordered:
            if len(selected) >= sizes[-1]:
                break
            path = _inside(corpus_root, record["relative_path"])
            _unchanged(path, record)
            if file_hash(path) != record["sha256"]:
                raise ValueError("Sampled shard differs from qualified hash")
            shard_rows = []
            offset = 0
            parquet = pq.ParquetFile(path)
            columns = ["pmid", "pmcid", "doi", "title", "abstract", "publication_types_json"]
            for group in range(parquet.num_row_groups):
                group_index = 0
                for batch in parquet.iter_batches(
                    batch_size=2048, row_groups=[group], columns=columns, use_threads=False
                ):
                    for row in batch.to_pylist():
                        counts["source_rows_examined"] += 1
                        location = {
                            "row_group": group,
                            "row_index": group_index,
                            "row_number": offset,
                        }
                        offset += 1
                        group_index += 1
                        if not row.get("pmid") or not str(row["pmid"]).isdigit():
                            counts["missing_or_invalid_pmid"] += 1
                            continue
                        if not isinstance(row.get("abstract"), str) or not row["abstract"].strip():
                            counts["missing_abstract"] += 1
                            continue
                        try:
                            row_aliases = paper_aliases(row)
                        except ValueError:
                            # A non-PMC local ID in a PMCID field is not a canonical identity.
                            counts["invalid_identifier_metadata"] += 1
                            continue
                        identity_id = canonical_identifier("PMID", row["pmid"])
                        if row_aliases & aliases or identity_id in seen_ids:
                            counts["linked_or_seeded_paper"] += 1
                            continue
                        publication_types = strict_json_loads(
                            row.get("publication_types_json") or "[]"
                        )
                        if not isinstance(publication_types, list):
                            raise ValueError("Publication types must be a structured list")
                        if {
                            re.sub(r"[^a-z]", "", str(value).lower()) for value in publication_types
                        } & {"retractedpublication", "retractionofpublication"}:
                            counts["retracted_or_retraction_notice"] += 1
                            continue
                        title = row.get("title") or ""
                        if not isinstance(title, str):
                            raise ValueError("Invalid title type")
                        text = f"{title}\n\n{row['abstract']}" if title else row["abstract"]
                        shard_rows.append(
                            {
                                "id": identity_id,
                                "text": text,
                                "title": title,
                                "abstract": row["abstract"],
                                "pmid": str(row["pmid"]),
                                "pmcid": row.get("pmcid"),
                                "doi": row.get("doi"),
                                "source_shard": record["relative_path"],
                                "source_sha256": record["sha256"],
                                "row_locator": location,
                            }
                        )
            _unchanged(path, record)
            used.append(
                {
                    "relative_path": record["relative_path"],
                    "sha256": record["sha256"],
                    "bytes": record["bytes"],
                }
            )
            for row in sorted(
                shard_rows,
                key=lambda item: (
                    hashlib.sha256((seed + item["id"]).encode()).hexdigest(),
                    item["id"],
                ),
            ):
                full, abstract = normalize_text(row["text"]), normalize_text(row["abstract"])
                if row["id"] in seen_ids or full in seen_full or abstract in seen_abstract:
                    counts["duplicate_identity_or_text"] += 1
                    continue
                selected.append(row)
                seen_ids.add(row["id"])
                seen_full.add(full)
                seen_abstract.add(abstract)
                publish_ready()
                if len(selected) == sizes[-1]:
                    break
        if len(selected) < sizes[-1]:
            raise ValueError(f"Only {len(selected)} qualifying candidates were available")
        if (
            file_hash(snapshot_path) != identity["snapshot_sha256"]
            or file_hash(edges_path) != edge_hash
        ):
            raise ValueError("Source qualification or exclusion inputs changed")
        result = {
            "status": "PASS",
            "fingerprint": fingerprint,
            "inputs": identity,
            "sizes": sizes,
            "views": views,
            "manifests": [
                str(
                    out
                    / f"pool-{m['counts']['documents']}"
                    / m["candidate_view"]
                    / "BENCHMARK_MANIFEST.json"
                )
                for m in manifests
            ],
        }
        result["manifest_receipts"] = [
            {"path": str(Path(value).relative_to(out)), "sha256": file_hash(Path(value))}
            for value in result["manifests"]
        ]
        atomic_json(out / "EXPANSION.json", result)
        return result


def build_expansion(corpus_root, snapshot_path, edges_path, seed_root, out, **kwargs):
    """Run the additive build and retain a clear failure receipt on interruption."""
    try:
        return _build_expansion(corpus_root, snapshot_path, edges_path, seed_root, out, **kwargs)
    except BaseException as exc:
        output = Path(out).resolve()
        protected = (Path(corpus_root).resolve(), Path(seed_root).resolve())
        if not any(
            output.is_relative_to(root) or root.is_relative_to(output) for root in protected
        ):
            state_path = output / "EXPANSION.json"
            if state_path.is_file():
                state = read_json(state_path)
                if state.get("status") == "RUNNING":
                    state["status"] = (
                        "INTERRUPTED"
                        if isinstance(exc, (KeyboardInterrupt, SystemExit))
                        else "FAIL"
                    )
                    state["error"] = {"type": type(exc).__name__, "message": str(exc)}
                    atomic_json(state_path, state)
        raise
