import uuid

from sqlalchemy import select

from app.models.lms import ItemSetParticipant
from app.repositories.base import BaseRepository


class ItemSetParticipantRepository(BaseRepository[ItemSetParticipant]):
    entity_class = ItemSetParticipant

    async def update_analysis_status(
        self,
        participant_id: uuid.UUID,
        status: str,
    ) -> None:
        participant = await self.find_by_id(participant_id)
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

    async def find_by_id(
        self,
        participant_id: uuid.UUID,
    ) -> ItemSetParticipant | None:
        stmt = select(ItemSetParticipant).where(
            ItemSetParticipant.id == participant_id,
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()
