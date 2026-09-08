import pytest

from litdatamatcher.scale_benchmark import compare_benchmark_baseline, run_local_benchmark, validate_benchmark_receipt


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


def test_baseline_comparison_rejects_malformed_hardware_cache_and_recovery(tmp_path):
    receipt = run_local_benchmark(tmp_path, count=4)
    baseline = {"fixture": receipt["fixture"], "hardware": receipt["hardware"], "backend": receipt["backend"], "measurements": {key: float(receipt["measurements"][key]["seconds"] + 1) for key in ["catalog_ingestion", "index_query", "matching", "evidence_compilation"]}, "tolerances": {"max_latency_multiplier": 2.0}}
    assert compare_benchmark_baseline(receipt, baseline)["status"] == "PASS"
    assert compare_benchmark_baseline(receipt, {})["status"] == "INVALID_BASELINE_OR_RECEIPT"
    mismatch = dict(baseline, hardware={"cpu_count": -1})
    assert compare_benchmark_baseline(receipt, mismatch)["status"] == "INCOMPARABLE_HARDWARE"
    cached = dict(receipt); cached["measurements"] = dict(receipt["measurements"], cache_hit={"replay_used_existing_manifest": False})
    assert compare_benchmark_baseline(cached, baseline)["status"] == "FAIL_CACHE_MISS"
    recovered = dict(receipt); recovered["measurements"] = dict(receipt["measurements"], recovery={"recovered_index_count": 0, "interrupted_cache_ignored": False})
    assert compare_benchmark_baseline(recovered, baseline)["status"] == "FAIL_RECOVERY"
