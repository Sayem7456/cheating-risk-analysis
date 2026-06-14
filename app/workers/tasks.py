from app.core.database import async_session_factory
from app.workers.celery_app import celery_app
from app.core.logging import get_logger
from app.core.config import settings
from app.repositories.analysis_result import AnalysisResultRepository
from app.repositories.participant import ItemSetParticipantRepository
from app.analyzers.activity import ActivityAnalyzer
from app.analyzers.detection import ObjectDetectionAnalyzer
from app.analyzers.face import FaceAnalyzer
from app.analyzers.video import VideoAnalyzer
from app.scoring.aggregator import FeatureAggregator
from app.scoring.engine import RuleBasedRiskEngine
from app.llm.explanation import ExplanationGenerator
from app.utils.s3_downloader import S3VideoDownloader
from app.services.orchestrator import AnalysisOrchestrator
from app.scheduler import discover_and_dispatch

logger = get_logger(__name__)


@celery_app.task
def discover_exams() -> int:
    """Celery Beat task — discovers completed exams and enqueues analysis jobs."""
    return discover_and_dispatch()


def _build_service() -> AnalysisOrchestrator:
    """Build AnalysisService outside FastAPI for Celery worker context."""
    from sqlalchemy.ext.asyncio import AsyncSession

    import asyncio

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    session = loop.run_until_complete(
        async_session_factory().__aenter__()
    )
    downloader = S3VideoDownloader(
        access_key_id=settings.aws_access_key_id,
        secret_access_key=settings.aws_secret_access_key,
        region=settings.aws_region,
        bucket=settings.aws_s3_bucket,
        temp_dir=settings.video_temp_dir,
        verify_integrity=settings.s3_verify_integrity,
    )
    face_analyzer = FaceAnalyzer()
    object_detector = ObjectDetectionAnalyzer()
    return AnalysisOrchestrator(
        analysis_result_repo=AnalysisResultRepository(session),
        participant_repo=ItemSetParticipantRepository(session),
        activity_analyzer=ActivityAnalyzer(),
        video_analyzer=VideoAnalyzer(
            downloader=downloader,
            face_analyzer=face_analyzer,
            object_detector=object_detector,
            fps=settings.video_frame_fps,
        ),
        feature_aggregator=FeatureAggregator(),
        scoring_engine=RuleBasedRiskEngine(settings=settings),
        explanation_gen=ExplanationGenerator(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
        ),
    )


@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def analyze_exam_session(self, participant_id: str, exam_id: str) -> dict:
    """Celery task to run full analysis pipeline for one exam session."""
    import asyncio

    try:
        service = _build_service()
        result = asyncio.run(
            service.run_analysis(participant_id, exam_id)
        )
        logger.info(
            "task_completed",
            participant_id=participant_id,
            exam_id=exam_id,
            risk_score=result.risk_score,
        )
        return {
            "analysis_id": str(result.id),
            "risk_score": result.risk_score,
            "risk_level": result.risk_level,
        }
    except Exception as exc:
        logger.exception("task_failed", participant_id=participant_id, exam_id=exam_id)
        raise self.retry(exc=exc)
