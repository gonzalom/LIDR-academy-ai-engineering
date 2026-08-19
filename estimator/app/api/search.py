"""HTTP layer for semantic search over the persisted corpus (Session 8).

Thin router: validation lives in ``SearchRequest`` (k bounds → 422), the
embed + SQL ranking lives in ``SemanticRetriever``. An empty corpus is a 200
with no results, not an error.
"""

from __future__ import annotations

import structlog
from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_retrieval_pipeline, get_semantic_retriever
from app.generation.rag.retrieval.pipeline import RetrievalPipeline
from app.generation.rag.retriever import SemanticRetriever
from app.generation.rag.schemas import (
    HybridSearchRequest,
    RetrievalRun,
    SearchRequest,
    SearchResponse,
)

log = structlog.get_logger()

router = APIRouter(tags=["search"])


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    retriever: SemanticRetriever | None = Depends(get_semantic_retriever),
) -> SearchResponse:
    """Return the k chunks closest to the query by cosine distance."""
    if retriever is None:
        # No OPENAI_API_KEY configured. Generic message to the client, detail logged.
        log.error("search_failed", reason="retriever_unavailable")
        raise HTTPException(status_code=500, detail="Embedding service is not available.")

    try:
        return await retriever.search(query=request.query, k=request.k)
    except Exception as exc:  # noqa: BLE001 — embedding/DB failures become a 500.
        log.error(
            "search_failed",
            reason="search_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(status_code=500, detail="Failed to run semantic search.") from exc


@router.post("/search/hybrid", response_model=RetrievalRun)
async def hybrid_search(
    request: HybridSearchRequest,
    pipeline: RetrievalPipeline | None = Depends(get_retrieval_pipeline),
) -> RetrievalRun:
    """Run one retrieval under an explicit configuration (Session 10).

    Kept apart from ``POST /search`` instead of bolted onto it: the Session 8
    contract (``SearchResponse``, cosine distance on every hit) still has
    clients, and a lexical-only hit has no distance to report. New capability,
    new endpoint.
    """
    if pipeline is None:
        log.error("hybrid_search_failed", reason="pipeline_unavailable")
        raise HTTPException(status_code=500, detail="Embedding service is not available.")

    try:
        return await pipeline.retrieve(
            query=request.query,
            top_k=request.k,
            mode=request.mode,
            rerank=request.rerank,
            recall_k=request.recall_k,
        )
    except Exception as exc:  # noqa: BLE001 — retrieval/rerank failures become a 500.
        log.error(
            "hybrid_search_failed",
            reason="retrieval_error",
            error_type=type(exc).__name__,
            error=str(exc)[:300],
        )
        raise HTTPException(status_code=500, detail="Failed to run hybrid search.") from exc
