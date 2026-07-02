"""drop chunk embedding column (moved to qdrant)

Revision ID: 0003_drop_chunk_embedding
Revises: 0002_fix_embedding_dim
Create Date: 2026-07-03
"""

from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector

from alembic import op

revision: str = "0003_drop_chunk_embedding"
down_revision: str | None = "0002_fix_embedding_dim"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks", postgresql_using="hnsw")
    op.drop_column("chunks", "embedding")


def downgrade() -> None:
    # Embeddings now live in Qdrant and are not recoverable from Postgres, so the
    # restored column is nullable. Re-ingesting documents repopulates the vectors.
    op.add_column("chunks", sa.Column("embedding", Vector(384), nullable=True))
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
