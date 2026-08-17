"""Semantic document retrieval using the existing embedding index.

This module provides document-level semantic search by querying the sqlite-vec
vector store for similar chunks and aggregating them into document-level results.
No LLM is involved — only the embedding model used to build the index.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from django.db.models import QuerySet

from documents.models import Document
from paperless_ai.embedding import get_embedding_model
from paperless_ai.indexing import _document_id_filters, get_vector_store, read_store

if TYPE_CHECKING:
    from llama_index.core.vector_stores.types import (
        MetadataFilters,
        VectorStoreQuery,
    )

logger = logging.getLogger("paperless.search")


@dataclass
class SemanticDocumentHit:
    """A document-level semantic search result."""

    document_id: int
    score: float
    rank: int
    best_chunk_text: str | None = None


def semantic_search_documents(
    query: str,
    *,
    document_ids: list[int] | None = None,
    limit: int = 50,
    chunk_k: int = 200,
) -> list[SemanticDocumentHit]:
    """Search documents by semantic similarity.

    Queries the existing sqlite-vec index for chunks similar to *query*,
    then aggregates chunk-level scores into document-level scores using
    ``max(chunk_scores)`` — the best chunk represents the document.

    Parameters
    ----------
    query:
        The search query text.
    document_ids:
        If provided, restrict search to these document IDs (permission
        filtering). When ``None``, searches the entire index.
    limit:
        Maximum number of documents to return.
    chunk_k:
        Number of chunks to retrieve from the vector store before
        deduplication.  Higher values improve recall for large corpora.

    Returns
    -------
    list[SemanticDocumentHit]
        Document-level results sorted by descending score, with rank
        assigned starting from 1.
    """
    from paperless.config import AIConfig

    config = AIConfig()
    if not config.llm_index_enabled:
        logger.debug("LLM index not enabled, skipping semantic search")
        return []

    embed_model = get_embedding_model(config)

    try:
        query_embedding: list[float] = embed_model.get_text_embedding(query)
    except Exception:
        logger.warning("Failed to generate query embedding", exc_info=True)
        return []

    from llama_index.core.vector_stores.types import (
        FilterOperator,
        MetadataFilter,
        MetadataFilters,
        VectorStoreQuery,
    )

    filters: MetadataFilters | None = None
    if document_ids is not None:
        if len(document_ids) == 0:
            return []
        filters = _document_id_filters(document_ids)

    try:
        with read_store() as store:
            result = store.query(
                VectorStoreQuery(
                    query_embedding=query_embedding,
                    similarity_top_k=chunk_k,
                    filters=filters,
                )
            )
    except Exception:
        logger.warning("Vector store query failed", exc_info=True)
        return []

    if not result.nodes:
        return []

    # Aggregate chunks into documents.
    # Key: document_id, Value: (best_score, best_chunk_text)
    doc_scores: dict[int, tuple[float, str | None]] = {}
    for node, similarity in zip(result.nodes, result.similarities or []):
        doc_id_raw = node.metadata.get("document_id")
        if doc_id_raw is None:
            continue
        doc_id = int(doc_id_raw)
        score = float(similarity) if similarity is not None else 0.0
        chunk_text = node.get_content()[:500] if node.get_content() else None

        if doc_id not in doc_scores or score > doc_scores[doc_id][0]:
            doc_scores[doc_id] = (score, chunk_text)

    # Sort by score descending and assign ranks.
    sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1][0], reverse=True)[
        :limit
    ]

    return [
        SemanticDocumentHit(
            document_id=doc_id,
            score=score,
            rank=i + 1,
            best_chunk_text=chunk_text,
        )
        for i, (doc_id, (score, chunk_text)) in enumerate(sorted_docs)
    ]
