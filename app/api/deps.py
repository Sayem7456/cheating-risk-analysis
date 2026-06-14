from collections.abc import AsyncGenerator

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from redis.asyncio import Redis

from app.core.config import settings
from app.core.database import async_session_factory
from app.repositories.analysis import CheatingRiskAnalysisRepository
from app.repositories.analysis_result import AnalysisResultRepository
from app.repositories.participant import ItemSetParticipantRepository
from app.services.orchestrator import AnalysisOrchestrator
from app.analyzers.activity import ActivityAnalyzer
from app.analyzers.video import VideoAnalyzer
from app.scoring.engine import RuleBasedRiskEngine
from app.llm.explanation import ExplanationGenerator
from app.analyzers.face import FaceAnalyzer
from app.analyzers.detection import ObjectDetectionAnalyzer
from app.scoring.aggregator import FeatureAggregator
from app.utils.s3_downloader import S3VideoDownloader


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def get_redis() -> AsyncGenerator[Redis, None]:
    r = Redis.from_url(settings.redis_url, decode_responses=True)
    try:
        yield r
    finally:
        await r.aclose()


def get_analysis_repo(
    db: AsyncSession = Depends(get_db),
) -> CheatingRiskAnalysisRepository:
    return CheatingRiskAnalysisRepository(db)


def get_participant_repo(
    db: AsyncSession = Depends(get_db),
) -> ItemSetParticipantRepository:
    return ItemSetParticipantRepository(db)


def get_activity_analyzer() -> ActivityAnalyzer:
    return ActivityAnalyzer()


def get_s3_downloader() -> S3VideoDownloader:
    return S3VideoDownloader(
        access_key_id=settings.aws_access_key_id,
        secret_access_key=settings.aws_secret_access_key,
        region=settings.aws_region,
        bucket=settings.aws_s3_bucket,
        temp_dir=settings.video_temp_dir,
        verify_integrity=settings.s3_verify_integrity,
    )


def get_face_analyzer() -> FaceAnalyzer:
    return FaceAnalyzer()

def get_object_detector() -> ObjectDetectionAnalyzer:
    return ObjectDetectionAnalyzer()

def get_video_analyzer(
    downloader: S3VideoDownloader = Depends(get_s3_downloader),
    face_analyzer: FaceAnalyzer = Depends(get_face_analyzer),
    object_detector: ObjectDetectionAnalyzer = Depends(get_object_detector),
) -> VideoAnalyzer:
    return VideoAnalyzer(
        downloader=downloader,
        face_analyzer=face_analyzer,
        object_detector=object_detector,
        fps=settings.video_frame_fps,
    )


def get_feature_aggregator() -> FeatureAggregator:
    return FeatureAggregator()


def get_scoring_engine() -> RuleBasedRiskEngine:
    return RuleBasedRiskEngine(settings=settings)


def get_llm_client() -> ExplanationGenerator:
    return ExplanationGenerator(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
    )


def get_analysis_result_repo(
    db: AsyncSession = Depends(get_db),
) -> AnalysisResultRepository:
    return AnalysisResultRepository(db)


def get_analysis_orchestrator(
    analysis_result_repo: AnalysisResultRepository = Depends(get_analysis_result_repo),
    participant_repo: ItemSetParticipantRepository = Depends(get_participant_repo),
    activity_analyzer: ActivityAnalyzer = Depends(get_activity_analyzer),
    video_analyzer: VideoAnalyzer = Depends(get_video_analyzer),
    feature_aggregator: FeatureAggregator = Depends(get_feature_aggregator),
    scoring_engine: RuleBasedRiskEngine = Depends(get_scoring_engine),
    explanation_gen: ExplanationGenerator = Depends(get_llm_client),
) -> AnalysisOrchestrator:
    return AnalysisOrchestrator(
        analysis_result_repo=analysis_result_repo,
        participant_repo=participant_repo,
        activity_analyzer=activity_analyzer,
        video_analyzer=video_analyzer,
        feature_aggregator=feature_aggregator,
        scoring_engine=scoring_engine,
        explanation_gen=explanation_gen,
    )
