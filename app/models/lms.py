import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, String
from sqlalchemy.dialects.postgresql import JSON, UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, utcnow


class ItemSetParticipant(Base):
    """Existing LMS table — actual schema discovered from DB introspection."""

    __tablename__ = "item_set_participant"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True
    )
    set_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )
    student_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), nullable=False
    )

    is_evaluated: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, index=True
    )
    submission_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    total_marks_after_eval: Mapped[float] = mapped_column(
        Float, nullable=False, default=0.0
    )
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), nullable=True
    )
    session_expire_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )

    activity_log: Mapped[list | None] = mapped_column(JSON, nullable=True)
    screen_records: Mapped[list | None] = mapped_column(JSON, nullable=True)
    face_records: Mapped[list | None] = mapped_column(JSON, nullable=True)

    final_result: Mapped[str | None] = mapped_column(
        String(50), nullable=True
    )

    analysis_status: Mapped[str | None] = mapped_column(
        String(20), nullable=True, default=None
    )
