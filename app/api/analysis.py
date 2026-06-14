from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import (
    get_analysis_orchestrator,
    get_analysis_result_repo,
    get_participant_repo,
)
from app.repositories.analysis_result import AnalysisResultRepository
from app.repositories.participant import ItemSetParticipantRepository
from app.schemas.analysis import (
    AnalysisResultResponse,
    ErrorResponse,
    RunAnalysisResponse,
    SuspiciousEvent,
)
from app.services.orchestrator import AnalysisOrchestrator

router = APIRouter(prefix="/analysis", tags=["analysis"])


@router.get(
    "/result/{set_id}/{student_id}",
    response_model=AnalysisResultResponse,
    responses={404: {"model": ErrorResponse}},
)
async def get_analysis_result(
    set_id: str,
    student_id: str,
    repo: AnalysisResultRepository = Depends(get_analysis_result_repo),
) -> AnalysisResultResponse:
    import uuid

    pid = uuid.UUID(student_id)
    eid = uuid.UUID(set_id)
    result = await repo.find_by_participant_and_exam(pid, eid)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Analysis result not found. Run analysis first.",
        )
    timeline = []
    if result.suspicious_timeline:
        timeline = [
            SuspiciousEvent(**e) if isinstance(e, dict) else e
            for e in result.suspicious_timeline
        ]
    return AnalysisResultResponse(
        id=str(result.id),
        participant_id=str(result.participant_id),
        exam_id=str(result.exam_id),
        risk_score=result.risk_score,
        cheating_probability=result.cheating_probability,
        risk_level=result.risk_level,
        activity_features=result.activity_features,
        face_features=result.face_features,
        suspicious_timeline=timeline,
        ai_summary=result.ai_summary,
        created_at=result.created_at,
        updated_at=result.updated_at,
    )


@router.post(
    "/run/{set_id}/{student_id}",
    response_model=RunAnalysisResponse,
    responses={404: {"model": ErrorResponse}, 409: {"model": ErrorResponse}},
)
async def run_analysis(
    set_id: str,
    student_id: str,
    orchestrator: AnalysisOrchestrator = Depends(get_analysis_orchestrator),
) -> RunAnalysisResponse:
    import uuid
    from uuid import UUID

    pid = UUID(student_id)
    eid = UUID(set_id)

    existing = await orchestrator.analysis_result_repo.find_by_participant_and_exam(
        pid, eid
    )
    if existing is not None:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Analysis already exists. Use POST /analysis/retry/{set_id}/{student_id} to re-run.",
        )

    result = await orchestrator.run_analysis(student_id, set_id)
    return RunAnalysisResponse(
        status="completed",
        analysis_id=str(result.id),
        risk_score=result.risk_score,
        risk_level=result.risk_level,
    )


@router.post(
    "/retry/{set_id}/{student_id}",
    response_model=RunAnalysisResponse,
    responses={404: {"model": ErrorResponse}},
)
async def retry_analysis(
    set_id: str,
    student_id: str,
    orchestrator: AnalysisOrchestrator = Depends(get_analysis_orchestrator),
) -> RunAnalysisResponse:
    import uuid
    from uuid import UUID

    pid = UUID(student_id)
    eid = UUID(set_id)

    deleted = await orchestrator.analysis_result_repo.delete_and_reset(pid, eid)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No existing analysis result to retry. Use POST /analysis/run/{set_id}/{student_id} first.",
        )

    result = await orchestrator.run_analysis(student_id, set_id)
    return RunAnalysisResponse(
        status="completed",
        analysis_id=str(result.id),
        risk_score=result.risk_score,
        risk_level=result.risk_level,
    )


@router.get("/participants/pending")
async def list_pending_participants(
    repo: ItemSetParticipantRepository = Depends(get_participant_repo),
) -> list[dict]:
    records = await repo.find_completed_pending()
    return [
        {
            "id": str(r.id),
            "participant_id": str(r.student_id),
            "exam_id": str(r.set_id),
        }
        for r in records
    ]
