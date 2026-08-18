"""Retrieval orchestration for Session 10: hybrid search + recall-then-rerank.

This is the only place that decides HOW MUCH to recall and HOW MUCH to keep.
The branches (vector k-NN in ``ChunkStore.search``, lexical FTS in
``ChunkStore.search_lexical``), the fusion (``fusion.reciprocal_rank_fusion``)
and the cross-encoder (``reranker.CrossEncoderReranker``) each do one thing and
know nothing about the others.

Recall-then-rerank, in one line: the cheap branches cast a WIDE net (top-50),
the expensive cross-encoder rescores only those and keeps the best (top-5).
Running the cross-encoder over the whole corpus would be correct and unusable;
running it over 50 candidates costs one forward pass and fixes exactly the
failure the exercise describes — "similar" not meaning "relevant".

Everything is a parameter with a settings-backed default, so the four
configurations of the brief are invocable without touching code:

    A  mode=vector  rerank=False      C  mode=vector  rerank=True
    B  mode=hybrid  rerank=False      D  mode=hybrid  rerank=True
"""

from __future__ import annotations

import asyncio
import time

import structlog
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.config import Settings
from app.generation.rag.embedding.embedder import OpenAIEmbedder
from app.generation.rag.errors import RetrievalError
from app.generation.rag.retrieval.fusion import reciprocal_rank_fusion
from app.generation.rag.retrieval.reranker import CrossEncoderReranker
from app.generation.rag.schemas import RankedHit, RetrievalRun, SearchMode
from app.generation.rag.store.repository import ChunkStore

log = structlog.get_logger()


class RetrievalPipeline:
    """Runs one retrieval under one configuration and reports how long it took."""

    def __init__(
        self,
        *,
        embedder: OpenAIEmbedder,
        session_factory: async_sessionmaker,
        store: ChunkStore,
        reranker: CrossEncoderReranker,
        settings: Settings,
    ) -> None:
        self._embedder = embedder
        self._session_factory = session_factory
        self._store = store
        self._reranker = reranker
        self._settings = settings

    async def retrieve(
        self,
        *,
        query: str,
        top_k: int | None = None,
        mode: SearchMode | None = None,
        rerank: bool | None = None,
        recall_k: int | None = None,
    ) -> RetrievalRun:
        """Retrieve ``top_k`` chunks for ``query`` under the resolved configuration.

        Every argument falls back to its setting, so the caller overrides only
        what it wants to vary — which is what makes the A/B/C/D sweep a loop over
        four dicts instead of four code paths.
        """
        settings = self._settings
        mode = mode if mode is not None else settings.SEARCH_MODE
        rerank = rerank if rerank is not None else settings.RERANK_ENABLED
        top_k = top_k if top_k is not None else settings.RERANK_TOP_N
        # Recall width: wide only when something downstream will re-rank it.
        # Recalling 50 and then keeping the first 5 by the SAME ranking would be
        # pure latency with no effect on the result.
        recall_width = (
            recall_k if recall_k is not None else (settings.RERANK_RECALL_K if rerank else top_k)
        )

        started = time.perf_counter()
        try:
            candidates = await self._recall(query=query, mode=mode, recall_width=recall_width)
        except Exception as exc:  # noqa: BLE001 — embedding/DB failures share one contract.
            raise RetrievalError(f"Recall failed for mode '{mode}': {exc}") from exc
        retrieval_ms = int((time.perf_counter() - started) * 1000)

        rerank_ms = 0
        if rerank and candidates:
            rerank_started = time.perf_counter()
            candidates = await self._rerank(query=query, candidates=candidates)
            rerank_ms = int((time.perf_counter() - rerank_started) * 1000)

        results = candidates[:top_k]
        total_ms = int((time.perf_counter() - started) * 1000)

        log.info(
            "retrieval_run_done",
            query=query[:80],
            mode=mode,
            reranked=rerank,
            candidates=len(candidates),
            returned=len(results),
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
        )
        return RetrievalRun(
            query=query,
            mode=mode,
            reranked=rerank,
            top_k=top_k,
            candidates=len(candidates),
            retrieval_ms=retrieval_ms,
            rerank_ms=rerank_ms,
            total_ms=total_ms,
            results=results,
        )

    # -- branches ------------------------------------------------------------

    async def _recall(
        self, *, query: str, mode: SearchMode, recall_width: int
    ) -> list[RankedHit]:
        """Run the configured branches and return one ranking, best first."""
        # The sync OpenAI client would block the event loop; same reasoning as
        # everywhere else in the ingest/query path.
        query_vector = await asyncio.to_thread(self._embedder.embed_one, query)

        async with self._session_factory() as session:
            vector_rows = await self._store.search(
                session, query_vector=query_vector, k=recall_width
            )
            lexical_rows = (
                await self._store.search_lexical(
                    session,
                    query=query,
                    k=recall_width,
                    fts_config=self._settings.FTS_CONFIG,
                )
                if mode == "hybrid"
                else []
            )

        hits: dict[int, RankedHit] = {}
        for position, row in enumerate(vector_rows, start=1):
            hits[row.id] = RankedHit(
                chunk_id=row.id,
                document_id=row.document_id,
                chunk_type=row.chunk_type,
                content=row.content,
                metadata=row.metadata_,
                # Higher is better across the whole module, so the distance is
                # negated. Only comparable within a run — never across runs.
                score=-float(row.distance),
                distance=float(row.distance),
                vector_position=position,
            )
        for position, row in enumerate(lexical_rows, start=1):
            hit = hits.get(row.id)
            if hit is None:
                hit = RankedHit(
                    chunk_id=row.id,
                    document_id=row.document_id,
                    chunk_type=row.chunk_type,
                    content=row.content,
                    metadata=row.metadata_,
                    score=0.0,
                )
                hits[row.id] = hit
            hit.lexical_rank = float(row.rank)
            hit.lexical_position = position

        if mode == "vector":
            return sorted(hits.values(), key=lambda hit: hit.vector_position or 0)

        fused = reciprocal_rank_fusion(
            [
                [row.id for row in vector_rows],
                [row.id for row in lexical_rows],
            ],
            k=self._settings.RRF_SMOOTHING_K,
        )
        ranked: list[RankedHit] = []
        for chunk_id, fused_score in fused:
            hit = hits[chunk_id]
            hit.score = fused_score
            ranked.append(hit)
        return ranked

    async def _rerank(self, *, query: str, candidates: list[RankedHit]) -> list[RankedHit]:
        """Rescore the recalled candidates with the cross-encoder, best first.

        ``score`` (not ``rerank``) on purpose: the wrapper's ``rerank`` returns
        the reordered candidates but drops the scores, and the scores are what
        makes the comparison table explainable.
        """
        scores = await asyncio.to_thread(
            self._reranker.score, query, [hit.content for hit in candidates]
        )
        for hit, score in zip(candidates, scores):
            hit.rerank_score = float(score)
            hit.score = float(score)
        # Stable sort: candidates keep their recall order when scores tie.
        return sorted(candidates, key=lambda hit: hit.rerank_score, reverse=True)
