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
                "paperless_ai.embedding.get_embedding_model",
                return_value=mock_model,
            ),
        ):
            assert semantic_search_documents("test query") == []

    @pytest.mark.django_db
    def test_returns_empty_for_empty_document_ids(self):
        assert semantic_search_documents("test", document_ids=[]) == []
