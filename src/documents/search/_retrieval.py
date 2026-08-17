"""Hybrid search orchestration with Reciprocal Rank Fusion (RRF).

Combines keyword (Tantivy BM25) and semantic (embedding vector) search
into a single ranked result set using RRF.  Neither Tantivy nor the
vector store is modified — this module only orchestrates them.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import TYPE_CHECKING

from documents.search._backend import SearchMode

if TYPE_CHECKING:
    from django.contrib.auth.models import AbstractUser
    from django.db.models import QuerySet

    from documents.models import Document
    from documents.search._backend import TantivyBackend

logger = logging.getLogger("paperless.search")


class RetrievalMode(StrEnum):
    """How documents are retrieved for a search query."""

    KEYWORD = "keyword"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


# ---------------------------------------------------------------------------
# RRF constants
# ---------------------------------------------------------------------------
RRF_K = 60
KEYWORD_WEIGHT = 1.0
SEMANTIC_WEIGHT = 1.0

# How many candidates to request from each backend before fusion.
KEYWORD_CANDIDATE_K = 200
SEMANTIC_CANDIDATE_K = 200
SEMANTIC_CHUNK_K = 200


def reciprocal_rank_fusion(
    keyword_ids: list[int],
    semantic_ids: list[int],
    *,
    keyword_weight: float = KEYWORD_WEIGHT,
    semantic_weight: float = SEMANTIC_WEIGHT,
    k: int = RRF_K,
) -> list[int]:
    """Merge two ranked ID lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    keyword_ids:
        Document IDs from keyword search, ordered by relevance.
    semantic_ids:
        Document IDs from semantic search, ordered by relevance.
    keyword_weight:
        Weight for keyword rank contribution.
    semantic_weight:
        Weight for semantic rank contribution.
    k:
        RRF constant (default 60, standard in literature).

    Returns
    -------
    list[int]
        Merged and re-ranked document IDs.
    """
    scores: dict[int, float] = {}

    for rank, doc_id in enumerate(keyword_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + keyword_weight / (k + rank)

    for rank, doc_id in enumerate(semantic_ids, start=1):
        scores[doc_id] = scores.get(doc_id, 0.0) + semantic_weight / (k + rank)

    return [
        doc_id for doc_id, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ]


def hybrid_search(
    query: str,
    *,
    backend: TantivyBackend,
    user: AbstractUser | None,
    filtered_qs: QuerySet[Document],
    search_mode: SearchMode = SearchMode.QUERY,
    keyword_limit: int = KEYWORD_CANDIDATE_K,
    semantic_limit: int = SEMANTIC_CANDIDATE_K,
    semantic_chunk_k: int = SEMANTIC_CHUNK_K,
    min_score: float | None = None,
) -> list[int]:
    """Run keyword + semantic search and fuse with RRF.

    Parameters
    ----------
    query:
        The user's search query.
    backend:
        The Tantivy search backend.
    user:
        Current user for permission filtering.
    filtered_qs:
        Django queryset already filtered by permissions and other
        document-list filters (tags, correspondents, dates, etc.).
    search_mode:
        Search mode for Tantivy query parsing (query, text, or title).
    keyword_limit:
        Max documents to request from Tantivy.
    semantic_limit:
        Max documents to request from semantic search.
    semantic_chunk_k:
        Chunks to retrieve from vector store.
    min_score:
        Minimum similarity score threshold for semantic search.

    Returns
    -------
    list[int]
        Fused document IDs sorted by RRF score, intersected with
        ``filtered_qs`` visibility.
    """
    # --- Keyword search (Tantivy) ---
    keyword_ids = backend.search_ids(
        query,
        user=user,
        search_mode=search_mode,
        limit=keyword_limit,
    )
    logger.debug("Keyword search returned %d documents", len(keyword_ids))

    # --- Semantic search (embedding index) ---
    from paperless_ai.search import semantic_search_documents

    semantic_hits = semantic_search_documents(
        query,
        limit=semantic_limit,
        chunk_k=semantic_chunk_k,
        min_score=min_score,
    )
    semantic_ids = [h.document_id for h in semantic_hits]
    logger.debug("Semantic search returned %d documents", len(semantic_ids))

    # --- Fuse ---
    fused_ids = reciprocal_rank_fusion(keyword_ids, semantic_ids)
    logger.debug("RRF fused to %d documents", len(fused_ids))

    # Intersect with the ORM queryset (preserves Django filter visibility).
    # Inline version of views.intersect_and_order (it's a nested function).
    if not fused_ids:
        return []
    _INTERSECT_THRESHOLD = 5_000
    if len(fused_ids) <= _INTERSECT_THRESHOLD:
        visible_ids = set(
            filtered_qs.filter(pk__in=fused_ids).values_list("pk", flat=True),
        )
    else:
        visible_ids = set(filtered_qs.values_list("pk", flat=True))
    return [doc_id for doc_id in fused_ids if doc_id in visible_ids]
