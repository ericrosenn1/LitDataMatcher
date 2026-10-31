import pytest

from litdatamatcher.retrieval_evaluation import evaluate_retrieval, paired_comparison


def test_each_query_including_missing_hit_enters_denominator():
    queries = [{"id": "one", "positive_doc_id": "a"}, {"id": "two", "positive_doc_id": "b"}]
    result = evaluate_retrieval(queries, [[{"id": "a"}], []])
    assert result["queries"] == 2
    assert result["mrr_at_k"] == result["known_link_recall_at_1"] == 0.5
    assert result["query_results"][1]["rank"] is None


def test_paired_queries_must_be_the_same():
    query = [{"id": "q", "positive_doc_id": "a"}]
    old, new = evaluate_retrieval(query, [[]]), evaluate_retrieval(query, [[{"id": "a"}]])
    assert paired_comparison(old, new)["mean_reciprocal_rank_delta"] == 1
    new["query_results"][0]["query_id"] = "changed"
    with pytest.raises(ValueError, match="identical"):
        paired_comparison(old, new)


def test_duplicates_are_rejected():
    query = [{"id": "q", "positive_doc_id": "a"}]
    with pytest.raises(ValueError, match="Duplicate candidate"):
        evaluate_retrieval(query, [[{"id": "a"}, {"id": "a"}]])
    with pytest.raises(ValueError, match="Duplicate query"):
        evaluate_retrieval(query + query, [[], []])
