"""add_material_snapshots

Revision ID: 4d1f2f3a9b21
Revises: c904b3d706d3
Create Date: 2026-05-08
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "4d1f2f3a9b21"
down_revision: Union[str, None] = "c904b3d706d3"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "materials",
        sa.Column("source_row", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "materials",
        sa.Column("parsed_row", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("materials", "parsed_row")
    op.drop_column("materials", "source_row")

