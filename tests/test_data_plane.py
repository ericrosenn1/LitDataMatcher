import hashlib
import pytest

from litdatamatcher.data_plane import Catalog, atomic_json


def test_invalidation_preserves_unrelated_and_replays(tmp_path):
    c = Catalog(tmp_path)
    assert c.upsert('document', 'd', {'text': 'first'})
    c.upsert('claim', 'a', {'claim': 'a'}, [('document', 'd')])
    c.upsert('match', 'm', {'match': 'm'}, [('claim', 'a')])
    c.upsert('dataset', 's', {'title': 'untouched'}, search_text='human inflammatory response')
    assert not c.upsert('document', 'd', {'text': 'first'})
    assert len(c.records('match')) == 1
    c.upsert('document', 'd', {'text': 'corrected'})
    assert not c.records('claim') and not c.records('match')
    assert len(c.records('dataset')) == 1
    assert c.search('dataset', 'inflammatory') == ['s']
    with pytest.raises(ValueError):
        c.upsert('match', 'x', {}, [('claim', 'a')])
    c.close()


def test_snapshot_detects_corruption_and_finite_json(tmp_path):
    c = Catalog(tmp_path)
    row = c.snapshot(b'source', {'url': 'https://example.org'})
    assert row['sha256'] == hashlib.sha256(b'source').hexdigest()
    from pathlib import Path
    Path(row['path']).write_bytes(b'broken')
    with pytest.raises(ValueError): c.snapshot(b'source', {})
    with pytest.raises(ValueError): atomic_json(tmp_path/'bad.json', {'score': float('nan')})
    assert not (tmp_path/'bad.json').exists()
    c.close()
