"""add_hardness_ci_columns

Revision ID: 7b2f8e3a1c4d
Revises: 4d1f2f3a9b21
Create Date: 2026-05-12

Adds hardness_ci_lower and hardness_ci_upper columns to predictions table.
These columns were specified in the schema plan (§5.6) and implemented in
S5's InferenceEngine for 95% CI computation on vickers_hardness predictions.
The Alembic migration was missed during S5, so added here post-S7.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "7b2f8e3a1c4d"
down_revision: Union[str, None] = "4d1f2f3a9b21"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "predictions",
        sa.Column("hardness_ci_lower", sa.Float(), nullable=True),
    )
    op.add_column(
        "predictions",
        sa.Column("hardness_ci_upper", sa.Float(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("predictions", "hardness_ci_upper")
    op.drop_column("predictions", "hardness_ci_lower")
