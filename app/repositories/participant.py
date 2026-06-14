import uuid

from sqlalchemy import and_, select

from app.models.lms import ItemSetParticipant
from app.repositories.base import BaseRepository


class ItemSetParticipantRepository(BaseRepository[ItemSetParticipant]):
    entity_class = ItemSetParticipant

    async def update_analysis_status(
        self,
        participant_id: uuid.UUID,
        exam_id: uuid.UUID,
        status: str,
    ) -> None:
        participant = await self.find_by_participant_and_exam(
            str(participant_id), str(exam_id)
        )
        if participant is not None:
            participant.analysis_status = status
            await self.update(participant)

    async def find_completed_pending(self, limit: int = 100) -> list[ItemSetParticipant]:
        stmt = (
            select(ItemSetParticipant)
            .where(ItemSetParticipant.is_evaluated == True)
            .limit(limit)
        )
        result = await self.session.execute(stmt)
        return list(result.scalars().all())

    async def find_by_participant_and_exam(
        self,
        participant_id: str,
        exam_id: str,
    ) -> ItemSetParticipant | None:
        stmt = select(ItemSetParticipant).where(
            and_(
                ItemSetParticipant.student_id == uuid.UUID(participant_id),
                ItemSetParticipant.set_id == uuid.UUID(exam_id),
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def find_by_student_and_set(
        self,
        student_id: str,
        set_id: str,
    ) -> ItemSetParticipant | None:
        stmt = select(ItemSetParticipant).where(
            and_(
                ItemSetParticipant.student_id == uuid.UUID(student_id),
                ItemSetParticipant.set_id == uuid.UUID(set_id),
            )
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
