"""Title metadata remains searchable without becoming abstract evidence."""

import json

import pyarrow as pa
import pyarrow.parquet as pq

from litdatamatcher.corpus_snapshot import file_digest
from litdatamatcher.data_plane import atomic_json
from litdatamatcher.literature_index import LiteratureIndex, materialize_pubmed
from test_literature_index import event, fixture


def test_missing_abstract_keeps_title_searchable_without_evidence_substitution(tmp_path):
    corpus, manifest, database, _ = fixture(tmp_path, [[event(1)]])
    value = json.loads(manifest.read_text())
    normalized = corpus / value["shards"][0]["legacy_relative_path"]
    table = pq.read_table(normalized)
    rows = table.to_pylist()
    rows[0]["abstract"] = ""
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), normalized)
    value["shards"][0]["legacy"]["sha256"] = file_digest(normalized)
    atomic_json(manifest, value)
    materialize_pubmed(corpus, manifest, database)
    with LiteratureIndex(database, read_only=True) as index:
        record = index.get(1)
        assert record["title"] == "Cancer study 1 version 1"
        assert record["abstract"] == record["text"] == ""
        found = index.search("cancer")
        assert len(found) == 1 and found[0]["text"] == ""
        assert all(passage["field"] == "title" for passage in index.passages(1))
