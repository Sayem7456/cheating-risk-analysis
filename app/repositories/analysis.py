import uuid
from typing import Any

from sqlalchemy import select

from app.models.analysis import CheatingRiskAnalysis
from app.repositories.base import BaseRepository


class CheatingRiskAnalysisRepository(BaseRepository[CheatingRiskAnalysis]):
    entity_class = CheatingRiskAnalysis

    async def delete_by_participant_and_exam(
        self,
        participant_id: str,
        exam_id: str,
    ) -> bool:
        existing = await self.find_by_participant_and_exam(participant_id, exam_id)
        if existing is None:
            return False
        await self.session.delete(existing)
        await self.session.flush()
        return True

    async def find_by_participant_and_exam(
        self,
        participant_id: str,
        exam_id: str,
    ) -> CheatingRiskAnalysis | None:
        stmt = select(CheatingRiskAnalysis).where(
            CheatingRiskAnalysis.participant_id == uuid.UUID(participant_id),
            CheatingRiskAnalysis.exam_id == uuid.UUID(exam_id),
        )
        result = await self.session.execute(stmt)
        return result.scalar_one_or_none()

    async def upsert_analysis(
        self,
        participant_id: uuid.UUID,
        exam_id: uuid.UUID,
        risk_score: float,
        cheating_probability: float,
        risk_level: str,
        activity_features: dict[str, Any] | None = None,
        face_features: dict[str, Any] | None = None,
        suspicious_timeline: list[dict[str, Any]] | None = None,
        ai_summary: str | None = None,
    ) -> CheatingRiskAnalysis:
        existing = await self.find_by_participant_and_exam(
            str(participant_id), str(exam_id)
        )
        if existing:
            existing.risk_score = risk_score
            existing.cheating_probability = cheating_probability
            existing.risk_level = risk_level
            existing.activity_features = activity_features
            existing.face_features = face_features
            existing.suspicious_timeline = suspicious_timeline
            existing.ai_summary = ai_summary
            await self.update(existing)
            return existing

        analysis = CheatingRiskAnalysis(
            participant_id=participant_id,
            exam_id=exam_id,
            risk_score=risk_score,
            cheating_probability=cheating_probability,
            risk_level=risk_level,
            activity_features=activity_features,
            face_features=face_features,
            suspicious_timeline=suspicious_timeline,
            ai_summary=ai_summary,
        )
        await self.create(analysis)
        return analysis
