"""Async data-access layer for the vector store.

The store never opens or commits sessions: the caller (ingest service,
retriever) owns the ``AsyncSession`` so a whole ingest — duplicate check,
document row, chunk rows — fits in ONE transaction. A failure anywhere rolls
everything back and leaves no orphan ``documents`` row.
"""

from __future__ import annotations

import re

from pgvector.sqlalchemy import HALFVEC
from sqlalchemy import Integer, Row, cast, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.generation.rag.schemas import EmbeddedChunk
from app.generation.rag.store.models import ChunkRow, DocumentRow, EMBEDDING_DIMENSIONS

# The structural chunker emits one chunk per budget component; the vocabulary
# is queryable thanks to the index on ``chunk_type`` (live-session filters).
BUDGET_COMPONENT = "budget_component"

# Tokens shorter than this carry no lexical signal ("a", "an", "of") and only
# widen the OR-query. Postgres would drop most of them as stop words anyway.
MIN_LEXICAL_TOKEN_LENGTH = 2

_WORD = re.compile(r"[a-z0-9]+")


def build_or_tsquery(query: str) -> str:
    """Turn a natural-language query into an OR-ed ``to_tsquery`` expression.

    Why not ``plainto_tsquery``/``websearch_to_tsquery``: both AND the terms.
    Our queries are full project descriptions ("a marketplace with Stripe
    payments and a seller dashboard"), and no single chunk contains every term,
    so an AND-query returns zero rows and the lexical branch contributes
    nothing. OR-ing the terms and letting ``ts_rank`` do the ordering is the
    standard shape of a lexical branch in hybrid retrieval: recall is the
    branch's job, precision is RRF's and the reranker's.

    Tokens are reduced to ``[a-z0-9]+`` so nothing the user types can reach
    ``to_tsquery``'s parser as an operator (``!``, ``&``, ``:*``) — that would
    be a syntax error at best and injection into the query language at worst.
    """
    seen: dict[str, None] = {}
    for token in _WORD.findall(query.lower()):
        if len(token) >= MIN_LEXICAL_TOKEN_LENGTH:
            seen.setdefault(token, None)
    return " | ".join(seen)


class ChunkStore:
    """CRUD + similarity search over ``documents``/``chunks``."""

    async def find_document_id(self, session: AsyncSession, source_path: str) -> int | None:
        """Return the id of the document already ingested from ``source_path``,
        or ``None``. Backs the application-level 409 duplicate guard."""
        stmt = select(DocumentRow.id).where(DocumentRow.source_path == source_path)
        return (await session.execute(stmt)).scalar_one_or_none()

    async def persist_document_with_chunks(
        self,
        session: AsyncSession,
        *,
        source_path: str,
        document_type: str,
        doc_metadata: dict,
        embedded_chunks: list[EmbeddedChunk],
        chunk_type: str = BUDGET_COMPONENT,
    ) -> int:
        """Insert the document row plus all its chunk rows. No commit here —
        the caller's transaction decides when (and whether) anything lands.

        ``chunk_type`` is stamped on every chunk (filterable column); it
        defaults to ``budget_component`` so existing callers are unaffected."""
        document = DocumentRow(
            source_path=source_path,
            document_type=document_type,
            metadata_=doc_metadata,
        )
        session.add(document)
        await session.flush()  # assigns document.id without committing

        session.add_all(
            ChunkRow(
                document_id=document.id,
                chunk_type=chunk_type,
                content=chunk.text,
                embedding=chunk.embedding,
                metadata_=chunk.metadata,
            )
            for chunk in embedded_chunks
        )
        return document.id

    async def search(
        self, session: AsyncSession, *, query_vector: list[float], k: int
    ) -> list[Row]:
        """k nearest chunks by cosine distance (``<=>``), sequential scan.

        Cosine over L2/inner product: OpenAI embeddings are normalized so the
        ranking would be equivalent, but cosine keeps us aligned with the RAG
        literature AND with the ``vector_cosine_ops`` operator class of the
        HNSW index the live session adds — operator/index mismatch makes
        Postgres silently ignore the index.
        """
        # distance = ChunkRow.embedding.cosine_distance(query_vector)
        distance = cast(ChunkRow.embedding, HALFVEC(EMBEDDING_DIMENSIONS)).cosine_distance(
            query_vector
        )
        stmt = (
            select(
                ChunkRow.id,
                ChunkRow.document_id,
                ChunkRow.chunk_type,
                ChunkRow.content,
                ChunkRow.metadata_,
                distance.label("distance"),
            )
            .order_by(distance)
            .limit(k)
        )
        return list((await session.execute(stmt)).all())

    async def search_lexical(
        self, session: AsyncSession, *, query: str, k: int, fts_config: str
    ) -> list[Row]:
        """Top-k chunks by lexical relevance over the ``content_tsv`` column.

        The keyword half of hybrid search. It catches exactly what the embedding
        misses: rare literals — "PSD2", "OAuth 2.0", a framework name — which a
        bi-encoder blurs into whatever it saw most during training, but which a
        lexeme match nails.

        ``@@`` is answered by the GIN index (migration 0003); ``ts_rank`` then
        scores only the rows that matched, so the cost is proportional to the
        matches, not to the corpus. An empty tsquery (a query with no usable
        tokens) matches nothing and returns an empty list — never every row.
        """
        tsquery_input = build_or_tsquery(query)
        if not tsquery_input:
            return []

        tsquery = func.to_tsquery(fts_config, tsquery_input)
        rank = func.ts_rank(ChunkRow.content_tsv, tsquery)
        stmt = (
            select(
                ChunkRow.id,
                ChunkRow.document_id,
                ChunkRow.chunk_type,
                ChunkRow.content,
                ChunkRow.metadata_,
                rank.label("rank"),
            )
            .where(ChunkRow.content_tsv.op("@@")(tsquery))
            .order_by(rank.desc(), ChunkRow.id)  # stable order for equal ranks
            .limit(k)
        )
        return list((await session.execute(stmt)).all())

    async def search_filtered(
        self,
        session: AsyncSession,
        *,
        query_vector: list[float],
        top_k: int = 10,
        distance_threshold: float = 0.6,
        sectors: list[str] | None = None,
        project_year_min: int | None = None,
        project_year_max: int | None = None,
        chunk_types: list[str] | None = None,
    ) -> tuple[list[Row], int]:
        """k-NN search with structural pre-filtering and a relevance threshold.

        Session 9 retrieval. Structural filters (sector / project year / chunk
        type) narrow the candidate space BEFORE the vector ranking — the metadata
        is persisted in JSONB (``client_sector``, ``year``) and the ``chunk_type``
        column. Each filter follows the ``(:filter IS NULL OR …)`` pattern: a
        ``None`` filter simply does not apply. The distance threshold then drops
        chunks that are not actually close (no "confidently retrieving garbage").

        Returns
        -------
        tuple[list[Row], int]
            ``(rows, candidates_evaluated)`` where ``rows`` are the top-k chunks
            under the threshold (ascending distance) and ``candidates_evaluated``
            is how many chunks matched the structural filters before the
            threshold/limit were applied.
        """
        sector_col = ChunkRow.metadata_["client_sector"].astext
        year_col = cast(ChunkRow.metadata_["year"].astext, Integer)

        structural_filters = []
        if sectors:
            structural_filters.append(sector_col.in_(sectors))
        if project_year_min is not None:
            structural_filters.append(year_col >= project_year_min)
        if project_year_max is not None:
            structural_filters.append(year_col <= project_year_max)
        if chunk_types:
            structural_filters.append(ChunkRow.chunk_type.in_(chunk_types))

        distance = cast(ChunkRow.embedding, HALFVEC(EMBEDDING_DIMENSIONS)).cosine_distance(
            query_vector
        )

        count_stmt = select(func.count()).select_from(ChunkRow).where(*structural_filters)
        candidates_evaluated = int((await session.execute(count_stmt)).scalar_one())

        stmt = (
            select(
                ChunkRow.id,
                ChunkRow.document_id,
                ChunkRow.chunk_type,
                ChunkRow.content,
                ChunkRow.metadata_,
                distance.label("distance"),
            )
            .where(*structural_filters)
            .where(distance <= distance_threshold)
            .order_by(distance)
            .limit(top_k)
        )
        rows = list((await session.execute(stmt)).all())
        return rows, candidates_evaluated
