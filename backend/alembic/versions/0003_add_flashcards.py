"""add session flashcards

Revision ID: 0003_add_flashcards
Revises: 0002_fix_embedding_dim
Create Date: 2026-06-20
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0003_add_flashcards"
down_revision: str | None = "0002_fix_embedding_dim"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "flashcards",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "session_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("answer", sa.Text(), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False, server_default="not_reviewed"),
        sa.Column("source_doc_name", sa.Text(), nullable=False),
        sa.Column("source_page", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.CheckConstraint(
            "status IN ('not_reviewed', 'learning', 'known')", name="ck_flashcards_status"
        ),
    )
    op.create_index("ix_flashcards_session_id", "flashcards", ["session_id"])


def downgrade() -> None:
    op.drop_index("ix_flashcards_session_id", table_name="flashcards")
    op.drop_table("flashcards")
