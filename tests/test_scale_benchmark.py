import pytest

from litdatamatcher.scale_benchmark import run_local_benchmark, validate_benchmark_receipt


def test_bounded_benchmark_measures_recovery_and_cache_hit(tmp_path):
    receipt = run_local_benchmark(tmp_path, count=4)
    assert receipt["validation_status"] == "PASS"
    assert receipt["measurements"]["recovery"] == {
        "interrupted_index_persisted_count": 2,
        "recovered_index_count": 4,
        "interrupted_cache_ignored": True,
    }
    assert receipt["measurements"]["cache_hit"]["replay_used_existing_manifest"] is True
    assert validate_benchmark_receipt(receipt)


@pytest.mark.parametrize("count", [0, -1, 1.5, "4", 1001])
def test_benchmark_rejects_nonzero_or_malformed_workload(count, tmp_path):
    with pytest.raises(ValueError, match="1..1000"):
        run_local_benchmark(tmp_path, count=count)


def test_malformed_benchmark_receipt_is_not_valid():
    assert not validate_benchmark_receipt({"validation_status": "PASS"})
    assert not validate_benchmark_receipt({"schema_version": "v2_5_local_benchmark_v1", "validation_status": "PASS", "measurements": {}})
