"""add_interview_message_metadata

Revision ID: msgmeta20260411
Revises: candctx20260411
Create Date: 2026-04-11 14:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "msgmeta20260411"
down_revision: str | None = "candctx20260411"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("interview_messages", sa.Column("lane", sa.Text(), nullable=True))
    op.add_column("interview_messages", sa.Column("focus", sa.Text(), nullable=True))
    op.add_column(
        "interview_messages",
        sa.Column("follow_up", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.add_column("interview_messages", sa.Column("scoreable", sa.Boolean(), nullable=True))
    op.alter_column("interview_messages", "follow_up", server_default=None)


def downgrade() -> None:
    op.drop_column("interview_messages", "scoreable")
    op.drop_column("interview_messages", "follow_up")
    op.drop_column("interview_messages", "focus")
    op.drop_column("interview_messages", "lane")
