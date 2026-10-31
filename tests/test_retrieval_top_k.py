"""Exact rank ordering, including ties at the partition boundary."""

import numpy as np
import pytest

from litdatamatcher.retrieval import top_k_indices


@pytest.mark.parametrize("size", [0, 1, 10, 1000])
@pytest.mark.parametrize("k", [1, 10, 2000])
def test_top_k_matches_reference_sort_for_ties_and_negative_scores(size, k):
    rng = np.random.default_rng(1729)
    scores = rng.integers(-3, 4, size=size).astype(np.float32)
    identifiers = [f"id-{i:06d}" for i in rng.permutation(size)]
    expected = sorted(range(size), key=lambda i: (-float(scores[i]), identifiers[i]))[:k]
    assert top_k_indices(scores, identifiers, k) == expected


def test_all_ties_select_identifiers_instead_of_partition_accident():
    assert top_k_indices(np.zeros(5), ["e", "d", "c", "b", "a"], 2) == [4, 3]


@pytest.mark.parametrize(
    "scores,ids,k",
    [
        (np.array([float("nan")]), ["a"], 1),
        (np.array([float("inf")]), ["a"], 1),
        (np.ones((1, 1)), ["a"], 1),
        (np.zeros(2), ["a"], 1),
        (np.zeros(1), ["a"], 0),
    ],
)
def test_invalid_rank_inputs_fail_closed(scores, ids, k):
    with pytest.raises(ValueError):
        top_k_indices(scores, ids, k)
