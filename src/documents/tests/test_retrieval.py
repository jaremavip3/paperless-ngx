"""Tests for hybrid search retrieval and RRF."""


from documents.search._retrieval import RetrievalMode
from documents.search._retrieval import reciprocal_rank_fusion


class TestRetrievalMode:
    def test_values(self):
        assert RetrievalMode.KEYWORD == "keyword"
        assert RetrievalMode.SEMANTIC == "semantic"
        assert RetrievalMode.HYBRID == "hybrid"


class TestReciprocalRankFusion:
    def test_empty_lists(self):
        assert reciprocal_rank_fusion([], []) == []

    def test_keyword_only(self):
        result = reciprocal_rank_fusion([10, 20, 30], [])
        assert result == [10, 20, 30]

    def test_semantic_only(self):
        result = reciprocal_rank_fusion([], [30, 20, 10])
        assert result == [30, 20, 10]

    def test_both_rankings_boost_overlap(self):
        keyword_ids = [1, 2]
        semantic_ids = [3, 1]
        result = reciprocal_rank_fusion(keyword_ids, semantic_ids)
        # doc 1 appears in both -> highest score
        assert result[0] == 1
        # score(3) = 1/(60+1) = 0.01639
        # score(2) = 1/(60+2) = 0.01613
        # doc 3 slightly ahead
        assert result[1] == 3
        assert result[2] == 2

    def test_custom_weights(self):
        keyword_ids = [1, 2]
        semantic_ids = [2, 1]
        result = reciprocal_rank_fusion(keyword_ids, semantic_ids, keyword_weight=2.0)
        assert result[0] == 1  # #1 keyword (weighted 2x) > #2 semantic

    def test_custom_k(self):
        result_low_k = reciprocal_rank_fusion([1], [2], k=1)
        result_high_k = reciprocal_rank_fusion([1], [2], k=1000)
        assert result_low_k[0] == 1
        assert result_high_k[0] == 1

    def test_large_lists(self):
        keyword_ids = list(range(1, 101))
        semantic_ids = list(range(100, 0, -1))
        result = reciprocal_rank_fusion(keyword_ids, semantic_ids)
        assert len(result) == 100
        # Doc that's #1 in keyword AND #100 in semantic should still rank high
        assert 1 in result[:10]
