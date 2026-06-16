import uuid
from datetime import datetime

from sqlalchemy import Float, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, utcnow


class CheatingRiskAnalysis(Base):
    __tablename__ = "cheating_risk_analysis"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    participant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False, index=True
    )

    risk_score: Mapped[float] = mapped_column(Float, nullable=False)
    cheating_probability: Mapped[float] = mapped_column(Float, nullable=False)
    risk_level: Mapped[str] = mapped_column(String(20), nullable=False)

    activity_features: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    face_features: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    suspicious_timeline: Mapped[list | None] = mapped_column(JSONB, nullable=True)
    ai_summary: Mapped[str | None] = mapped_column(Text, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        nullable=False, default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        nullable=False, default=utcnow, onupdate=utcnow
    )
