from datetime import datetime

from pydantic import BaseModel


class SuspiciousEvent(BaseModel):
    time: str
    event: str
    duration: float | None = None


class AnalysisResult(BaseModel):
    participant_id: str
    risk_score: float
    cheating_probability: float
    risk_level: str
    activity_features: dict | None = None
    face_features: dict | None = None
    suspicious_timeline: list[SuspiciousEvent] | None = None
    ai_summary: str | None = None


class AnalysisResultResponse(AnalysisResult):
    id: str
    created_at: datetime
    updated_at: datetime


class RunAnalysisResponse(BaseModel):
    status: str
    analysis_id: str
    risk_score: float | None = None
    risk_level: str | None = None


class ErrorResponse(BaseModel):
    detail: str
