"""fix chunk embedding dimension

Revision ID: 0002_fix_embedding_dim
Revises: 0001_initial
Create Date: 2026-06-14
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0002_fix_embedding_dim"
down_revision: str | None = "0001_initial"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks", postgresql_using="hnsw")
    op.execute("DELETE FROM chunks")
    op.execute("ALTER TABLE chunks ALTER COLUMN embedding TYPE vector(384)")
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )


def downgrade() -> None:
    op.drop_index("ix_chunks_embedding_hnsw", table_name="chunks", postgresql_using="hnsw")
    op.create_index(
        "ix_chunks_embedding_hnsw",
        "chunks",
        ["embedding"],
        postgresql_using="hnsw",
        postgresql_ops={"embedding": "vector_cosine_ops"},
    )
