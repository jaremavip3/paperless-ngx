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


class TestHybridSearch:
    def test_hybrid_search_passes_search_mode_default(self):
        from unittest.mock import MagicMock
        from unittest.mock import patch

        from documents.search._backend import SearchMode
        from documents.search._retrieval import hybrid_search

        backend = MagicMock()
        backend.search_ids.return_value = [1, 2]
        user = MagicMock()
        filtered_qs = MagicMock()
        filtered_qs.filter.return_value.values_list.return_value = [1, 2]

        with patch(
            "paperless_ai.search.semantic_search_documents",
            return_value=[],
        ) as mock_semantic:
            result = hybrid_search(
                "test query",
                backend=backend,
                user=user,
                filtered_qs=filtered_qs,
            )

        backend.search_ids.assert_called_once_with(
            "test query",
            user=user,
            search_mode=SearchMode.QUERY,
            limit=200,
        )
        mock_semantic.assert_called_once_with(
            "test query",
            limit=200,
            chunk_k=200,
            min_score=None,
        )
        assert result == [1, 2]

    def test_hybrid_search_passes_custom_search_mode(self):
        from unittest.mock import MagicMock
        from unittest.mock import patch

        from documents.search._backend import SearchMode
        from documents.search._retrieval import hybrid_search

        backend = MagicMock()
        backend.search_ids.return_value = [10]
        user = MagicMock()
        filtered_qs = MagicMock()
        filtered_qs.filter.return_value.values_list.return_value = [10]

        with patch(
            "paperless_ai.search.semantic_search_documents",
            return_value=[],
        ):
            result = hybrid_search(
                "title search query",
                backend=backend,
                user=user,
                filtered_qs=filtered_qs,
                search_mode=SearchMode.TITLE,
            )

        backend.search_ids.assert_called_once_with(
            "title search query",
            user=user,
            search_mode=SearchMode.TITLE,
            limit=200,
        )
        assert result == [10]

    def test_hybrid_search_passes_min_score(self):
        from unittest.mock import MagicMock
        from unittest.mock import patch

        from documents.search._retrieval import hybrid_search

        backend = MagicMock()
        backend.search_ids.return_value = [1]
        user = MagicMock()
        filtered_qs = MagicMock()
        filtered_qs.filter.return_value.values_list.return_value = [1]

        with patch(
            "paperless_ai.search.semantic_search_documents",
            return_value=[],
        ) as mock_semantic:
            result = hybrid_search(
                "test query",
                backend=backend,
                user=user,
                filtered_qs=filtered_qs,
                min_score=0.85,
            )

        mock_semantic.assert_called_once_with(
            "test query",
            limit=200,
            chunk_k=200,
            min_score=0.85,
        )
        assert result == [1]

    def test_hybrid_search_no_eager_document_ids_fetching(self):
        from unittest.mock import MagicMock
        from unittest.mock import patch

        from documents.search._retrieval import hybrid_search
        from paperless_ai.search import SemanticDocumentHit

        backend = MagicMock()
        backend.search_ids.return_value = [1, 2]
        user = MagicMock()
        filtered_qs = MagicMock()
        filtered_qs.filter.return_value.values_list.return_value = [1, 2]

        semantic_hits = [SemanticDocumentHit(document_id=2, score=0.9, rank=1)]
        with patch(
            "paperless_ai.search.semantic_search_documents",
            return_value=semantic_hits,
        ) as mock_semantic:
            result = hybrid_search(
                "query",
                backend=backend,
                user=user,
                filtered_qs=filtered_qs,
            )

        # semantic_search_documents called without document_ids parameter
        mock_semantic.assert_called_once_with(
            "query",
            limit=200,
            chunk_k=200,
            min_score=None,
        )
        assert result == [2, 1]

    def test_hybrid_search_post_intersection(self):
        from unittest.mock import MagicMock
        from unittest.mock import patch

        from documents.search._retrieval import hybrid_search

        backend = MagicMock()
        backend.search_ids.return_value = [1, 2, 3]
        user = MagicMock()
        filtered_qs = MagicMock()
        # Doc 2 is not visible in filtered_qs
        filtered_qs.filter.return_value.values_list.return_value = [1, 3]

        with patch(
            "paperless_ai.search.semantic_search_documents",
            return_value=[],
        ):
            result = hybrid_search(
                "query",
                backend=backend,
                user=user,
                filtered_qs=filtered_qs,
            )

        assert result == [1, 3]

    def test_hybrid_search_empty_fusion_returns_empty(self):
        from unittest.mock import MagicMock
        from unittest.mock import patch

        from documents.search._retrieval import hybrid_search

        backend = MagicMock()
        backend.search_ids.return_value = []
        user = MagicMock()
        filtered_qs = MagicMock()

        with patch(
            "paperless_ai.search.semantic_search_documents",
            return_value=[],
        ):
            result = hybrid_search(
                "query",
                backend=backend,
                user=user,
                filtered_qs=filtered_qs,
            )

        assert result == []

    def test_hybrid_search_large_result_set_intersection(self):
        from unittest.mock import MagicMock
        from unittest.mock import patch

        from documents.search._retrieval import hybrid_search

        # 5,001 IDs to trigger > _INTERSECT_THRESHOLD branch
        large_ids = list(range(1, 5002))
        backend = MagicMock()
        backend.search_ids.return_value = large_ids
        user = MagicMock()
        filtered_qs = MagicMock()
        # Only even IDs are in filtered_qs
        filtered_qs.values_list.return_value = [i for i in large_ids if i % 2 == 0]

        with patch(
            "paperless_ai.search.semantic_search_documents",
            return_value=[],
        ):
            result = hybrid_search(
                "query",
                backend=backend,
                user=user,
                filtered_qs=filtered_qs,
            )

        # Expected only even IDs, preserving order
        expected = [i for i in large_ids if i % 2 == 0]
        assert result == expected
        filtered_qs.values_list.assert_called_once_with("pk", flat=True)


class TestUnifiedSearchViewSetRouting:
    def test_is_search_request_routing(self):
        from rest_framework.request import Request
        from rest_framework.test import APIRequestFactory

        from documents.views import UnifiedSearchViewSet

        factory = APIRequestFactory()
        view = UnifiedSearchViewSet()

        # No params -> False
        request = Request(factory.get("/api/documents/"))
        view.request = request
        assert view._is_search_request() is False

        # Non-search query params (e.g. page, ordering, tags) -> False
        request = Request(
            factory.get(
                "/api/documents/?page=2&page_size=25&ordering=-created&tags__id__in=1,2",
            ),
        )
        view.request = request
        assert view._is_search_request() is False

        # retrieval_mode alone without search params -> False
        for mode in ("keyword", "hybrid", "semantic"):
            request = Request(factory.get(f"/api/documents/?retrieval_mode={mode}"))
            view.request = request
            assert view._is_search_request() is False

        # Active search params alone -> True
        for param in (
            "query=invoice",
            "text=receipt",
            "title_search=tax",
            "more_like_id=5",
        ):
            request = Request(factory.get(f"/api/documents/?{param}"))
            view.request = request
            assert view._is_search_request() is True

        # Active search params + retrieval_mode -> True
        request = Request(
            factory.get("/api/documents/?query=invoice&retrieval_mode=hybrid"),
        )
        view.request = request
        assert view._is_search_request() is True

        request = Request(
            factory.get("/api/documents/?text=receipt&retrieval_mode=semantic"),
        )
        view.request = request
        assert view._is_search_request() is True

        # Also works when passing a raw Django HttpRequest
        raw_request = factory.get("/api/documents/?query=raw_test")
        view.request = raw_request
        assert view._is_search_request() is True
        raw_request_no_search = factory.get("/api/documents/?retrieval_mode=hybrid")
        view.request = raw_request_no_search
        assert view._is_search_request() is False

        # Handles None request safely
        view.request = None
        assert view._is_search_request(None) is False


class TestSemanticSearchHighlightFallback:
    """Verify that semantic search results produce valid SearchHit entries for all page_ids,
    even when Tantivy keyword highlighting returns no matches."""

    def test_semantic_search_produces_hits_when_tantivy_highlights_empty(self):
        from unittest.mock import MagicMock
        from unittest.mock import patch

        from rest_framework.test import APIRequestFactory

        from documents.views import UnifiedSearchViewSet
        from paperless_ai.search import SemanticDocumentHit

        factory = APIRequestFactory()
        view_func = UnifiedSearchViewSet.as_view({"get": "list"})
        raw_request = factory.get(
            "/api/documents/?query=proving+income&retrieval_mode=semantic",
        )
        mock_user = MagicMock()
        mock_user.is_superuser = True
        raw_request.user = mock_user

        mock_backend = MagicMock()
        # Tantivy highlight_hits drops non-keyword matches and returns []
        mock_backend.highlight_hits.return_value = []

        mock_qs = MagicMock()
        # Document IDs 1, 2, 3 exist in queryset
        mock_qs.filter.return_value.values_list.return_value = [1, 2, 3]

        semantic_hits = [
            SemanticDocumentHit(
                document_id=1,
                score=0.95,
                rank=1,
                best_chunk_text="Proof of income letter",
            ),
            SemanticDocumentHit(
                document_id=2,
                score=0.88,
                rank=2,
                best_chunk_text="Salary slip statement",
            ),
            SemanticDocumentHit(
                document_id=3,
                score=0.82,
                rank=3,
                best_chunk_text=None,
            ),
        ]

        with (
            patch.object(UnifiedSearchViewSet, "get_queryset", return_value=mock_qs),
            patch.object(UnifiedSearchViewSet, "filter_queryset", return_value=mock_qs),
            patch("documents.search.get_backend", return_value=mock_backend),
            patch(
                "paperless_ai.search.semantic_search_documents",
                return_value=semantic_hits,
            ),
            patch.object(UnifiedSearchViewSet, "get_serializer") as mock_ser,
        ):
            mock_ser.return_value.data = [{"id": 1}, {"id": 2}, {"id": 3}]
            response = view_func(raw_request)
            assert response.status_code == 200
            assert response.data["count"] == 3

        # Ensure serializer received the 3 SearchHits with chunk highlights and scores
        mock_ser.assert_called_once()
        page_arg = mock_ser.call_args[0][0]
        assert len(page_arg) == 3
        assert page_arg[0]["id"] == 1
        assert page_arg[0]["score"] == 0.95
        assert page_arg[0]["rank"] == 1
        assert page_arg[0]["highlights"] == {"content": "Proof of income letter"}
        assert page_arg[1]["id"] == 2
        assert page_arg[1]["score"] == 0.88
        assert page_arg[1]["rank"] == 2
        assert page_arg[1]["highlights"] == {"content": "Salary slip statement"}
        assert page_arg[2]["id"] == 3
        assert page_arg[2]["score"] == 0.82
        assert page_arg[2]["rank"] == 3
        assert page_arg[2]["highlights"] == {}


class TestTantivyRelevanceListFallback:
    """Verify TantivyRelevanceList gracefully falls back when page_hits is missing or partial."""

    def test_relevance_list_with_empty_page_hits(self):
        from documents.search._backend import TantivyRelevanceList

        ordered_ids = [10, 20, 30, 40, 50]
        # page_hits is empty because highlight_hits dropped non-keyword matches
        rl = TantivyRelevanceList(ordered_ids=ordered_ids, page_hits=[], page_offset=0)

        assert len(rl) == 5

        # Slicing page 1 (0:25)
        page = rl[0:25]
        assert len(page) == 5
        for idx, hit in enumerate(page):
            assert hit["id"] == ordered_ids[idx]
            assert hit["score"] == 0.0
            assert hit["rank"] == idx + 1
            assert hit["highlights"] == {}

    def test_relevance_list_with_partial_page_hits(self):
        from documents.search._backend import SearchHit
        from documents.search._backend import TantivyRelevanceList

        ordered_ids = [10, 20, 30]
        # Only doc 20 had a keyword highlight
        partial_hits = [
            SearchHit(
                id=20,
                score=4.5,
                rank=2,
                highlights={"content": "<b>matched</b>"},
            ),
        ]
        rl = TantivyRelevanceList(
            ordered_ids=ordered_ids,
            page_hits=partial_hits,
            page_offset=0,
        )

        page = rl[0:25]
        assert len(page) == 3
        assert page[0]["id"] == 10
        assert page[0]["score"] == 0.0
        assert page[1]["id"] == 20
        assert page[1]["score"] == 4.5
        assert page[1]["highlights"] == {"content": "<b>matched</b>"}
        assert page[2]["id"] == 30
        assert page[2]["score"] == 0.0

    def test_relevance_list_integer_indexing(self):
        from documents.search._backend import SearchHit
        from documents.search._backend import TantivyRelevanceList

        ordered_ids = [100, 200, 300]
        partial_hits = [
            SearchHit(id=200, score=1.2, rank=2, highlights={"content": "hit"}),
        ]
        rl = TantivyRelevanceList(
            ordered_ids=ordered_ids,
            page_hits=partial_hits,
            page_offset=0,
        )

        hit0 = rl[0]
        assert hit0["id"] == 100
        assert hit0["rank"] == 1

        hit1 = rl[1]
        assert hit1["id"] == 200
        assert hit1["score"] == 1.2
        assert hit1["highlights"] == {"content": "hit"}

        hit2 = rl[2]
        assert hit2["id"] == 300
        assert hit2["rank"] == 3

    def test_relevance_list_out_of_order_page_hits_preserves_ordered_ids(self):
        from documents.search._backend import SearchHit
        from documents.search._backend import TantivyRelevanceList

        ordered_ids = [10, 20, 30]
        # page_hits passed in reverse or arbitrary order
        out_of_order_hits = [
            SearchHit(id=30, score=3.0, rank=3, highlights={"content": "hit 30"}),
            SearchHit(id=10, score=1.0, rank=1, highlights={"content": "hit 10"}),
            SearchHit(id=20, score=2.0, rank=2, highlights={"content": "hit 20"}),
        ]
        rl = TantivyRelevanceList(
            ordered_ids=ordered_ids,
            page_hits=out_of_order_hits,
            page_offset=0,
        )

        page = rl[0:25]
        assert [h["id"] for h in page] == [10, 20, 30]
        assert page[0]["score"] == 1.0
        assert page[1]["score"] == 2.0
        assert page[2]["score"] == 3.0

    def test_relevance_list_negative_slice(self):
        from documents.search._backend import TantivyRelevanceList

        ordered_ids = [10, 20, 30, 40, 50]
        rl = TantivyRelevanceList(ordered_ids=ordered_ids, page_hits=[], page_offset=0)

        tail = rl[-2:]
        assert len(tail) == 2
        assert tail[0]["id"] == 40
        assert tail[0]["rank"] == 4
        assert tail[1]["id"] == 50
        assert tail[1]["rank"] == 5
