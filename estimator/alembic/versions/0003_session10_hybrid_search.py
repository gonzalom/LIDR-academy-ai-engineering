"""Session 10 — full-text search column + GIN index on chunks.

Revision ID: 0003_session10_hybrid_search
Revises: 0002_session8_pgvector
Create Date: 2026-08-18 00:00:00

The lexical branch of hybrid search needs a ``tsvector``: the chunk content
already parsed into lexemes. It is a GENERATED column, so Postgres keeps it in
sync with ``content`` on every insert/update — no application code, no chance of
a stale index. The GIN index is what makes ``@@`` a lookup instead of a scan.

Text search configuration: **english**, not the ``spanish`` the exercise
statement assumes. Measured on 2026-08-15 against this corpus: the budgets and
the labels emitted by the structural chunker are written in English
("Component:", "Description:", "authentication backend"), and three test queries
with plural/singular variation retrieved 3/3 under ``english`` and 0/3 under
``spanish`` or ``simple`` — the Spanish stemmer does not conflate English
morphology. The deviation is deliberate and argued in the write-up.

A generated column hardcodes the configuration into the schema (it must be
IMMUTABLE), so switching languages is a migration, not a setting. That is the
correct trade-off here: the corpus language is a property of the data, not a
runtime knob.
"""

from __future__ import annotations

from typing import Sequence, Union

from alembic import op

revision: str = "0003_session10_hybrid_search"
down_revision: Union[str, None] = "0002_session8_pgvector"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

FTS_CONFIG = "english"


def upgrade() -> None:
    op.execute(
        f"""
        ALTER TABLE chunks
        ADD COLUMN content_tsv tsvector
        GENERATED ALWAYS AS (to_tsvector('{FTS_CONFIG}', content)) STORED
        """
    )
    op.execute("CREATE INDEX ix_chunks_content_tsv ON chunks USING GIN (content_tsv)")


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_chunks_content_tsv")
    op.execute("ALTER TABLE chunks DROP COLUMN IF EXISTS content_tsv")
