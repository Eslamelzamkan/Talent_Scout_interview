"""add_candidate_context_to_interview_sessions

Revision ID: candctx20260411
Revises: aea756041e7c
Create Date: 2026-04-11 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "candctx20260411"
down_revision: str | None = "aea756041e7c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "interview_sessions",
        sa.Column(
            "candidate_context",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
    )
    op.alter_column("interview_sessions", "candidate_context", server_default=None)


def downgrade() -> None:
    op.drop_column("interview_sessions", "candidate_context")
