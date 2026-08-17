"""Tests for semantic document retrieval."""

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest

from paperless_ai.search import SemanticDocumentHit
from paperless_ai.search import semantic_search_documents


class TestSemanticDocumentHit:
    def test_creation(self):
        hit = SemanticDocumentHit(document_id=42, score=0.95, rank=1)
        assert hit.document_id == 42
        assert hit.score == 0.95
        assert hit.rank == 1
        assert hit.best_chunk_text is None

    def test_with_chunk_text(self):
        hit = SemanticDocumentHit(
            document_id=42,
            score=0.95,
            rank=1,
            best_chunk_text="some text",
        )
        assert hit.best_chunk_text == "some text"


class TestSemanticSearchDocuments:
    def test_returns_empty_when_index_disabled(self):
        mock_config = MagicMock()
        mock_config.llm_index_enabled = False
        with patch("paperless.config.AIConfig", return_value=mock_config):
            assert semantic_search_documents("test query") == []

    def test_returns_empty_on_embedding_failure(self):
        mock_config = MagicMock()
        mock_config.llm_index_enabled = True
        mock_config.llm_embedding_backend = "openai_like"
        mock_model = MagicMock()
        mock_model.get_text_embedding.side_effect = Exception("embed failed")
        with (
            patch("paperless.config.AIConfig", return_value=mock_config),
            patch(
                "paperless_ai.search.get_embedding_model",
                return_value=mock_model,
            ),
        ):
            assert semantic_search_documents("test query") == []

    @pytest.mark.django_db
    def test_returns_empty_for_empty_document_ids(self):
        assert semantic_search_documents("test", document_ids=[]) == []

    def test_search_documents_success(self):
        mock_config = MagicMock()
        mock_config.llm_index_enabled = True
        mock_model = MagicMock()
        mock_model.get_text_embedding.return_value = [0.1, 0.2, 0.3]

        mock_node1 = MagicMock()
        mock_node1.metadata = {"document_id": 42}
        mock_node1.get_content.return_value = "chunk text for 42"

        mock_node2 = MagicMock()
        mock_node2.metadata = {"document_id": 99}
        mock_node2.get_content.return_value = "chunk text for 99"

        mock_result = MagicMock()
        mock_result.nodes = [mock_node1, mock_node2]
        mock_result.similarities = [0.95, 0.85]

        mock_store = MagicMock()
        mock_store.query.return_value = mock_result
        mock_store_ctx = MagicMock()
        mock_store_ctx.__enter__.return_value = mock_store
        mock_store_ctx.__exit__.return_value = None

        with (
            patch("paperless.config.AIConfig", return_value=mock_config),
            patch("paperless_ai.search.get_embedding_model", return_value=mock_model),
            patch("paperless_ai.search.read_store", return_value=mock_store_ctx),
        ):
            hits = semantic_search_documents("query", limit=10, chunk_k=50)

        assert len(hits) == 2
        assert hits[0].document_id == 42
        assert hits[0].score == 0.95
        assert hits[0].rank == 1
        assert hits[0].best_chunk_text == "chunk text for 42"
        assert hits[1].document_id == 99
        assert hits[1].score == 0.85
        assert hits[1].rank == 2

    def test_returns_empty_when_no_nodes(self):
        mock_config = MagicMock()
        mock_config.llm_index_enabled = True
        mock_model = MagicMock()
        mock_model.get_text_embedding.return_value = [0.1, 0.2, 0.3]

        mock_result = MagicMock()
        mock_result.nodes = []

        mock_store = MagicMock()
        mock_store.query.return_value = mock_result
        mock_store_ctx = MagicMock()
        mock_store_ctx.__enter__.return_value = mock_store
        mock_store_ctx.__exit__.return_value = None

        with (
            patch("paperless.config.AIConfig", return_value=mock_config),
            patch("paperless_ai.search.get_embedding_model", return_value=mock_model),
            patch("paperless_ai.search.read_store", return_value=mock_store_ctx),
        ):
            hits = semantic_search_documents("query")

        assert hits == []

    def test_multiple_chunks_deduplication_and_missing_metadata(self):
        mock_config = MagicMock()
        mock_config.llm_index_enabled = True
        mock_model = MagicMock()
        mock_model.get_text_embedding.return_value = [0.1, 0.2, 0.3]

        # Two chunks for doc 10 (low score first, higher score second)
        mock_node1 = MagicMock()
        mock_node1.metadata = {"document_id": 10}
        mock_node1.get_content.return_value = "chunk 1 doc 10"

        mock_node2 = MagicMock()
        mock_node2.metadata = {"document_id": 10}
        mock_node2.get_content.return_value = "chunk 2 doc 10 (better match)"

        # One node with missing document_id
        mock_node3 = MagicMock()
        mock_node3.metadata = {}
        mock_node3.get_content.return_value = "chunk without doc_id"

        mock_result = MagicMock()
        mock_result.nodes = [mock_node1, mock_node2, mock_node3]
        mock_result.similarities = [0.5, 0.9, 0.99]

        mock_store = MagicMock()
        mock_store.query.return_value = mock_result
        mock_store_ctx = MagicMock()
        mock_store_ctx.__enter__.return_value = mock_store
        mock_store_ctx.__exit__.return_value = None

        with (
            patch("paperless.config.AIConfig", return_value=mock_config),
            patch("paperless_ai.search.get_embedding_model", return_value=mock_model),
            patch("paperless_ai.search.read_store", return_value=mock_store_ctx),
        ):
            hits = semantic_search_documents("query")

        assert len(hits) == 1
        assert hits[0].document_id == 10
        assert hits[0].score == 0.9
        assert hits[0].best_chunk_text == "chunk 2 doc 10 (better match)"

    def test_returns_empty_on_vector_store_query_failure(self):
        mock_config = MagicMock()
        mock_config.llm_index_enabled = True
        mock_model = MagicMock()
        mock_model.get_text_embedding.return_value = [0.1, 0.2, 0.3]

        mock_store = MagicMock()
        mock_store.query.side_effect = RuntimeError("sqlite-vec query failed")
        mock_store_ctx = MagicMock()
        mock_store_ctx.__enter__.return_value = mock_store
        mock_store_ctx.__exit__.return_value = None

        with (
            patch("paperless.config.AIConfig", return_value=mock_config),
            patch("paperless_ai.search.get_embedding_model", return_value=mock_model),
            patch("paperless_ai.search.read_store", return_value=mock_store_ctx),
        ):
            hits = semantic_search_documents("query")

        assert hits == []
