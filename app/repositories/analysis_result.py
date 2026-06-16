from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.analysis import CheatingRiskAnalysis
from app.repositories.analysis import CheatingRiskAnalysisRepository
from app.repositories.participant import ItemSetParticipantRepository


class AnalysisResultRepository:
    """Facade that stores analysis results and updates participant status
    within a single transaction."""

    def __init__(self, session: AsyncSession) -> None:
        self._analysis_repo = CheatingRiskAnalysisRepository(session)
        self._participant_repo = ItemSetParticipantRepository(session)

    async def find_by_participant(
        self,
        participant_id: uuid.UUID,
    ) -> CheatingRiskAnalysis | None:
        return await self._analysis_repo.find_by_participant(
            str(participant_id)
        )

    async def delete_and_reset(
        self,
        participant_id: uuid.UUID,
    ) -> bool:
        deleted = await self._analysis_repo.delete_by_participant(
            str(participant_id)
        )
        if deleted:
            await self._participant_repo.update_analysis_status(
                participant_id=participant_id,
                status=None,
            )
        return deleted

    async def save_with_status(
        self,
        participant_id: uuid.UUID,
        risk_score: float,
        cheating_probability: float,
        risk_level: str,
        activity_features: dict[str, Any] | None = None,
        face_features: dict[str, Any] | None = None,
        suspicious_timeline: list[dict[str, Any]] | None = None,
        ai_summary: str | None = None,
    ) -> CheatingRiskAnalysis:
        analysis = await self._analysis_repo.upsert_analysis(
            participant_id=participant_id,
            risk_score=risk_score,
            cheating_probability=cheating_probability,
            risk_level=risk_level,
            activity_features=activity_features,
            face_features=face_features,
            suspicious_timeline=suspicious_timeline,
            ai_summary=ai_summary,
        )

        await self._participant_repo.update_analysis_status(
            participant_id=participant_id,
            status="COMPLETED",
        )

        return analysis
