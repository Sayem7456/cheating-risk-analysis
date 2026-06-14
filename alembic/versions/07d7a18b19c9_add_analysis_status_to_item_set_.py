"""add analysis_status to item_set_participant

Revision ID: 07d7a18b19c9
Revises: 66c6810e9907
Create Date: 2026-06-11 14:29:53.662099
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = '07d7a18b19c9'
down_revision: Union[str, None] = '66c6810e9907'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('item_set_participant', sa.Column('analysis_status', sa.String(length=20), nullable=True))


def downgrade() -> None:
    op.drop_column('item_set_participant', 'analysis_status')
