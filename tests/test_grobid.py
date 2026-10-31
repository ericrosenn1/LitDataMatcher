from litdatamatcher.grobid import process_pdf_to_tei


class FakeGrobidClient:
    def __init__(self):
        self.calls = []

    def post_file_text(self, url, file_path, *, field_name="input", data=None, **kwargs):
        self.calls.append((url, file_path, field_name, data or {}))
        return "<TEI><teiHeader /><text><body><div><p>Parsed text.</p></div></body></text></TEI>"


def test_process_pdf_to_tei_writes_service_response(tmp_path):
    pdf_path = tmp_path / "paper.pdf"
    out_path = tmp_path / "paper.tei.xml"
    pdf_path.write_bytes(b"%PDF-1.4\n")
    client = FakeGrobidClient()

    metrics = process_pdf_to_tei(
        pdf_path,
        out_path,
        server_url="http://grobid.example",
        consolidate_header=True,
        client=client,
    )

    assert out_path.exists()
    assert out_path.read_text(encoding="utf-8").startswith("<TEI")
    assert metrics["out"] == str(out_path)
    assert client.calls[0][0] == "http://grobid.example/api/processFulltextDocument"
    assert client.calls[0][2] == "input"
    assert client.calls[0][3]["consolidateHeader"] == 1
