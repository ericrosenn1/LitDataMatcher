"""Cache replay must never use the network or publish a failed replacement."""

import json
from hashlib import sha256
from pathlib import Path

import pytest
import requests

from litdatamatcher import http_cache
from litdatamatcher.http_cache import CachedHttpClient

URL = "https://fixture.invalid/cache"
PARAMS = {"query": "cached", "page": [1, 2]}
FORM = {"consolidateHeader": 0, "includeRawAffiliations": False}
UPLOAD = b"%PDF-1.4\nsynthetic offline fixture\n"
# Existing request keys captured before this repair. Fixtures replay these exact names.
GET_KEY = "e7137d05aeefd369d614ff629cc8e94a12384cf2"
POST_KEY = "ec5b64fb95bd6a9a4849a862effa42500de1cb17"


@pytest.fixture(autouse=True)
def forbid_network(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("network attempted")

    for target in (
        "requests.get",
        "requests.post",
        "requests.sessions.Session.request",
        "socket.create_connection",
        "socket.socket.connect",
    ):
        monkeypatch.setattr(target, forbidden)


@pytest.fixture(params=["get_json", "get_text", "post_file_text"])
def cache_case(request, tmp_path):
    method = request.param
    client = CachedHttpClient(cache_dir=tmp_path / "cache", offline=True)
    upload = tmp_path / "paper.pdf"
    upload.write_bytes(UPLOAD)
    if method == "get_json":
        name, body, value = f"{GET_KEY}.json", b'{"message":"cached"}', {"message": "cached"}
    else:
        name = f"{POST_KEY if method == 'post_file_text' else GET_KEY}.txt"
        body, value = b"<TEI>cached</TEI>", "<TEI>cached</TEI>"
    return client, method, upload, client.cache_dir / name, body, value


def invoke(client, method, upload, **options):
    if method == "post_file_text":
        return client.post_file_text(URL, upload, data=FORM, **options)
    return getattr(client, method)(URL, PARAMS, **options)


def mock_response(monkeypatch, method):
    class Response:
        text = "<TEI>replacement</TEI>"

        def raise_for_status(self):
            return None

        def json(self):
            return {"message": "replacement"}

    calls = []

    def respond(url, **kwargs):
        calls.append((url, kwargs))
        if method == "post_file_text":
            assert kwargs["files"]["input"][1].read() == UPLOAD
            assert kwargs["data"] == {"consolidateHeader": "0", "includeRawAffiliations": "False"}
        else:
            assert kwargs["params"] == PARAMS
        return Response()

    verb = "post" if method == "post_file_text" else "get"
    monkeypatch.setattr(requests, verb, respond)
    return calls


def test_offline_hit_replays_existing_key_and_reports_validated_provenance(cache_case):
    client, method, upload, path, body, value = cache_case
    path.write_bytes(body)
    assert invoke(client, method, upload) == value
    assert path.read_bytes() == body
    assert client.last_response_metadata["cache_path"] == str(path)
    assert client.last_response_metadata["cache_status"] == "cache_hit"
    assert client.last_response_metadata["cache_content_sha256"] == sha256(body).hexdigest()
    assert client.last_response_metadata["retrieval_time_utc"]


def test_offline_miss_fails_without_network_or_stale_provenance(cache_case):
    client, method, upload, path, _, _ = cache_case
    client.last_response_metadata = {"cache_status": "previous_request"}
    with pytest.raises(FileNotFoundError, match="offline cache missing"):
        invoke(client, method, upload)
    assert not path.exists()
    assert client.last_response_metadata == {}


@pytest.mark.parametrize("cache_present", [False, True])
def test_offline_cache_bypass_cannot_enable_network(cache_case, cache_present):
    client, method, upload, path, body, _ = cache_case
    if cache_present:
        path.write_bytes(body)
    client.last_response_metadata = {"cache_status": "previous_request"}
    with pytest.raises(FileNotFoundError, match="offline cache missing"):
        invoke(client, method, upload, use_cache=False)
    assert path.exists() is cache_present
    if cache_present:
        assert path.read_bytes() == body
    assert client.last_response_metadata == {}


@pytest.mark.parametrize("offline", [False, True])
def test_invalid_utf8_cache_is_preserved_without_fallback_or_success_metadata(cache_case, offline):
    client, method, upload, path, _, _ = cache_case
    client.offline = offline
    path.write_bytes(b"\xffinvalid UTF-8")
    client.last_response_metadata = {"cache_status": "previous_request"}
    with pytest.raises(UnicodeDecodeError):
        invoke(client, method, upload)
    assert path.read_bytes() == b"\xffinvalid UTF-8"
    assert client.last_response_metadata == {}


def test_cache_path_directory_is_an_error_without_network_or_success_metadata(cache_case):
    client, method, upload, path, _, _ = cache_case
    path.mkdir()
    client.last_response_metadata = {"cache_status": "previous_request"}
    with pytest.raises(OSError):
        invoke(client, method, upload)
    assert path.is_dir()
    assert client.last_response_metadata == {}


@pytest.mark.parametrize("offline", [False, True])
@pytest.mark.parametrize("body", [b'{"truncated":', b""])
def test_malformed_json_cache_is_preserved_without_fallback_or_success_metadata(tmp_path, offline, body):
    client = CachedHttpClient(cache_dir=tmp_path, offline=offline)
    path = tmp_path / f"{GET_KEY}.json"
    path.write_bytes(body)
    client.last_response_metadata = {"cache_status": "previous_request"}
    with pytest.raises(json.JSONDecodeError):
        client.get_json(URL, PARAMS)
    assert path.read_bytes() == body
    assert client.last_response_metadata == {}


@pytest.mark.parametrize("use_cache", [False, True])
@pytest.mark.parametrize("body", [None, b'{"message":"cached"}', b'{"truncated":'])
def test_offline_refresh_never_fetches_or_replaces_existing_entry(tmp_path, use_cache, body):
    client = CachedHttpClient(cache_dir=tmp_path, offline=True)
    path = tmp_path / f"{GET_KEY}.json"
    if body is not None:
        path.write_bytes(body)
    client.last_response_metadata = {"cache_status": "previous_request"}
    with pytest.raises(FileNotFoundError, match="offline refresh unavailable"):
        client.get_json(URL, PARAMS, refresh=True, use_cache=use_cache)
    if body is not None:
        assert path.read_bytes() == body
    else:
        assert not path.exists()
    assert client.last_response_metadata == {}


@pytest.mark.parametrize("use_cache", [False, True])
def test_mocked_success_preserves_request_keys_and_clears_stale_metadata(cache_case, monkeypatch, use_cache):
    client, method, upload, path, _, _ = cache_case
    client.offline = False
    client.last_response_metadata = {"cache_status": "previous_request"}
    calls = mock_response(monkeypatch, method)
    expected = {"message": "replacement"} if method == "get_json" else "<TEI>replacement</TEI>"
    assert invoke(client, method, upload, use_cache=use_cache) == expected
    assert len(calls) == 1
    assert calls[0][0] == URL
    if use_cache:
        metadata = client.last_response_metadata
        assert metadata["cache_status"] == "live_cached"
        assert metadata["cache_path"] == str(path)
        assert metadata["cache_content_sha256"] == sha256(path.read_bytes()).hexdigest()
        client.offline = True
        assert invoke(client, method, upload) == expected
        assert len(calls) == 1
        assert sorted(client.cache_dir.iterdir()) == [path]
    else:
        assert client.last_response_metadata == {}
        assert list(client.cache_dir.iterdir()) == []


def test_failed_request_does_not_reuse_previous_response_metadata(cache_case, monkeypatch):
    client, method, upload, path, body, _ = cache_case
    client.offline = False
    path.write_bytes(body)
    client.last_response_metadata = {"cache_status": "previous_request"}

    def fail_request(*args, **kwargs):
        raise requests.RequestException("synthetic transport failure")

    monkeypatch.setattr(requests, "post" if method == "post_file_text" else "get", fail_request)
    with pytest.raises(RuntimeError, match="synthetic transport failure"):
        invoke(client, method, upload, use_cache=False, retries=0)
    assert path.read_bytes() == body
    assert client.last_response_metadata == {}


@pytest.mark.parametrize("failure", ["partial_write", "metadata_hash", "replace"])
def test_failed_cache_promotion_preserves_prior_entry_and_leaves_no_partial_cache(cache_case, monkeypatch, failure):
    client, method, upload, path, body, value = cache_case
    client.offline = False
    # JSON supports refresh; text and POST only write on cache misses.
    refresh = method == "get_json"
    if refresh:
        path.write_bytes(body)
    client.last_response_metadata = {"cache_status": "previous_request"}
    calls = mock_response(monkeypatch, method)

    with monkeypatch.context() as faults:
        if failure == "partial_write":
            def fail_write(target, text, **kwargs):
                target.write_bytes(b"partial replacement")
                raise OSError("synthetic partial write")

            faults.setattr(Path, "write_text", fail_write)
        elif failure == "metadata_hash":
            original_hash = http_cache._file_sha256

            def fail_new_metadata(target):
                if target == upload or (target == path and target.read_bytes() == body):
                    return original_hash(target)
                raise OSError("synthetic metadata hash failure")

            faults.setattr(http_cache, "_file_sha256", fail_new_metadata)
        else:
            def fail_replace(*args, **kwargs):
                raise OSError("synthetic replacement failure")

            faults.setattr(Path, "replace", fail_replace)

        with pytest.raises(OSError, match="synthetic"):
            invoke(client, method, upload, retries=0, **({"refresh": True} if refresh else {}))

    assert len(calls) == 1
    assert client.last_response_metadata == {}
    if refresh:
        assert path.read_bytes() == body
        client.offline = True
        assert invoke(client, method, upload) == value
        assert len(calls) == 1
        assert sorted(client.cache_dir.iterdir()) == [path]
    else:
        assert list(client.cache_dir.iterdir()) == []
