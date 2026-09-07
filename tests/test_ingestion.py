import json

from litdatamatcher.cli import main
from litdatamatcher.ingestion import (
    discover_literature_files,
    extract_body_text,
    ingest_literature_sources,
    infer_abstract,
    infer_title,
    load_literature_sources,
)
from litdatamatcher.pipeline import run_pipeline
from litdatamatcher.storage import read_jsonl


def test_infer_title_and_abstract_from_markdown_text(tmp_path):
    path = tmp_path / "paper.md"
    text = """# Microbiome Recovery After Antibiotics

Abstract
Future studies should examine whether antibiotic exposure and longitudinal
microbiome data explain recovery after treatment.

Methods
We sequenced samples.
"""

    assert infer_title(path, text) == "Microbiome Recovery After Antibiotics"
    assert "Future studies should examine" in infer_abstract(text)
    body = extract_body_text(text, title="Microbiome Recovery After Antibiotics")
    assert "Methods" in body
    assert "Future studies should examine" not in body


def test_load_literature_sources_reads_text_markdown_and_jsonl(tmp_path):
    text_path = tmp_path / "plain_text.txt"
    text_path.write_text(
        "Plain Text Study\n\nFurther research should examine whether metabolomics predicts remission.",
        encoding="utf-8",
    )
    markdown_path = tmp_path / "markdown_paper.md"
    markdown_path.write_text(
        "# Markdown Study\n\nFuture work should examine longitudinal microbiome recovery.",
        encoding="utf-8",
    )
    jsonl_path = tmp_path / "existing.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "title": "Existing Record",
                "abstract": "Further studies should examine antibiotic exposure.",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    records, skipped = load_literature_sources([text_path, markdown_path, jsonl_path])

    assert not skipped
    assert len(records) == 3
    assert {record["source_format"] for record in records} == {"txt", "md", "jsonl"}
    assert all(record["source_id"].startswith("source_") for record in records)
    assert all(record["document_id"].startswith("doc_") for record in records)
    assert all(record["source_sha256"] for record in records)
    assert all(record["source_size_bytes"] > 0 for record in records)
    assert all(record["source_modified_time_utc"] for record in records)
    assert all(record["source_record_count"] >= 1 for record in records)
    assert all(record["source_file_status"] == "ok" for record in records)
    assert records[0]["ingestion_schema_version"] == "literature_ingestion_v1"


def test_ingest_directory_recursive_writes_jsonl_and_manifest(tmp_path):
    source_dir = tmp_path / "papers"
    nested_dir = source_dir / "nested"
    nested_dir.mkdir(parents=True)
    (source_dir / "one.txt").write_text(
        "One Study\n\nFuture studies should examine antibiotic exposure.",
        encoding="utf-8",
    )
    (nested_dir / "two.md").write_text(
        "# Two Study\n\nFurther research should examine microbiome composition.",
        encoding="utf-8",
    )
    out_path = tmp_path / "literature.jsonl"

    metrics = ingest_literature_sources([source_dir], out_path, recursive=True)
    manifest = json.loads(out_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    report = out_path.with_suffix(".ingestion_report.md").read_text(encoding="utf-8")

    assert metrics["records"] == 2
    assert metrics["manifest"].endswith("literature.manifest.json")
    assert metrics["report"].endswith("literature.ingestion_report.md")
    assert len(read_jsonl(out_path)) == 2
    assert manifest["formats"] == {"md": 1, "txt": 1}
    assert manifest["recursive"] is True
    assert manifest["created_at_utc"]
    assert manifest["total_discovered_files"] == 2
    assert manifest["records_written"] == 2
    assert len(manifest["source_files"]) == 2
    assert {source["status"] for source in manifest["source_files"]} == {"ok"}
    assert all(source["sha256"] for source in manifest["source_files"])
    assert "Literature Ingestion Report" in report
    assert "Source Files" in report


def test_jsonl_passthrough_preserves_existing_source_and_document_ids(tmp_path):
    jsonl_path = tmp_path / "existing.jsonl"
    jsonl_path.write_text(
        json.dumps(
            {
                "source_id": "source-existing",
                "document_id": "doc-existing",
                "title": "Existing Record",
                "abstract": "Further studies should examine antibiotic exposure.",
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    records, skipped = load_literature_sources([jsonl_path])

    assert not skipped
    assert records[0]["source_id"] == "source-existing"
    assert records[0]["document_id"] == "doc-existing"
    assert records[0]["source_sha256"]
    assert records[0]["source_name"] == "existing.jsonl"
    assert records[0]["source_record_count"] == 1
    assert records[0]["source_file_status"] == "ok"


def test_jsonl_passthrough_records_file_level_record_count(tmp_path):
    jsonl_path = tmp_path / "existing.jsonl"
    rows = [
        {"title": "First Record", "abstract": "Future studies should examine exposure."},
        {"title": "Second Record", "abstract": "Future studies should examine outcomes."},
    ]
    jsonl_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )

    records, skipped = load_literature_sources([jsonl_path])

    assert not skipped
    assert len(records) == 2
    assert {record["source_record_count"] for record in records} == {2}
    assert {record["source_file_status"] for record in records} == {"ok"}


def test_duplicate_file_inputs_are_deduplicated(tmp_path):
    text_path = tmp_path / "paper.txt"
    text_path.write_text(
        "Duplicate Study\n\nFurther research should examine microbiome recovery.",
        encoding="utf-8",
    )

    records, skipped = load_literature_sources([text_path, text_path])

    assert not skipped
    assert len(records) == 1


def test_ingestion_reads_jats_xml_with_article_metadata(tmp_path):
    xml_path = tmp_path / "pmc_article.nxml"
    xml_path.write_text(
        """<?xml version="1.0"?>
<article>
  <front>
    <journal-meta><journal-title>Example Journal</journal-title></journal-meta>
    <article-meta>
      <article-id pub-id-type="doi">10.1234/example</article-id>
      <article-id pub-id-type="pmc">PMC123</article-id>
      <article-title>JATS Microbiome Study</article-title>
      <contrib-group><contrib contrib-type="author"><name><surname>Jones</surname><given-names>A</given-names></name></contrib></contrib-group>
      <abstract><p>Future studies should examine microbiome recovery.</p></abstract>
    </article-meta>
  </front>
  <body>
    <sec><title>Discussion</title><p>Further research should examine antibiotic exposure.</p></sec>
  </body>
</article>
""",
        encoding="utf-8",
    )

    records, skipped = load_literature_sources([xml_path])

    assert not skipped
    assert len(records) == 1
    assert records[0]["title"] == "JATS Microbiome Study"
    assert records[0]["doi"] == "10.1234/example"
    assert records[0]["ingestion_method"] == "jats"
    assert records[0]["metadata"]["pmcid"] == "PMC123"
    assert records[0]["metadata"]["authors"] == ["Jones A"]
    assert "Discussion" in records[0]["metadata"]["xml_sections"]
    assert records[0]["metadata"]["xml_section_records"][0]["title"] == "Discussion"


def test_ingestion_reads_grobid_tei_xml(tmp_path):
    xml_path = tmp_path / "grobid.tei"
    xml_path.write_text(
        """<?xml version="1.0"?>
<TEI xmlns="http://www.tei-c.org/ns/1.0">
  <teiHeader>
    <fileDesc><titleStmt><title>GROBID Parsed Study</title></titleStmt></fileDesc>
    <sourceDesc><biblStruct><analytic><author><persName><surname>Lee</surname><forename>B</forename></persName></author></analytic></biblStruct></sourceDesc>
    <profileDesc><abstract><p>Future work should examine host transcriptomics.</p></abstract></profileDesc>
    <idno type="DOI">10.5678/tei</idno>
  </teiHeader>
  <text><body><div><head>Results</head><p>Further studies should examine treatment response.</p></div></body></text>
</TEI>
""",
        encoding="utf-8",
    )

    records, skipped = load_literature_sources([xml_path])

    assert not skipped
    assert len(records) == 1
    assert records[0]["title"] == "GROBID Parsed Study"
    assert records[0]["doi"] == "10.5678/tei"
    assert records[0]["ingestion_method"] == "grobid_tei"
    assert records[0]["metadata"]["authors"] == ["Lee B"]
    assert records[0]["metadata"]["xml_section_records"][0]["paragraph_count"] == 1
    assert "treatment response" in records[0]["text"]


def test_discover_literature_files_rejects_unsupported_paths(tmp_path):
    bad_path = tmp_path / "notes.csv"
    bad_path.write_text("not,literature", encoding="utf-8")

    try:
        discover_literature_files([bad_path])
    except FileNotFoundError as exc:
        assert "No supported literature file" in str(exc)
    else:
        raise AssertionError("unsupported input should raise FileNotFoundError")


def test_ingest_cli_output_can_feed_pipeline(tmp_path, capsys):
    text_path = tmp_path / "paper.txt"
    text_path.write_text(
        (
            "Microbiome Recovery\n\n"
            "Further research should examine whether antibiotic exposure and "
            "longitudinal microbiome data explain clinical recovery."
        ),
        encoding="utf-8",
    )
    literature_path = tmp_path / "literature.jsonl"

    result = main(["ingest", "--input", str(text_path), "--out", str(literature_path)])
    captured = json.loads(capsys.readouterr().out)
    metrics = run_pipeline(literature_path, tmp_path / "run", top_n=5)

    assert result == 0
    assert captured["records"] == 1
    assert "report" in captured
    assert literature_path.exists()
    assert literature_path.with_suffix(".manifest.json").exists()
    assert literature_path.with_suffix(".ingestion_report.md").exists()
    assert metrics["documents"] == 1
    assert metrics["questions"] >= 1
    assert metrics["matches"] >= 1
