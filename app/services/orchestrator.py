import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from typing import Any, TypeVar

from app.analyzers.activity import ActivityAnalyzer
from app.analyzers.timeline import SuspiciousTimelineBuilder
from app.analyzers.video import VideoAnalyzer
from app.core.logging import get_logger
from app.core.metrics import (
    analysis_duration,
    analysis_failures,
    analysis_step_duration,
    analysis_total,
    feature_gauge,
    risk_level_gauge,
    risk_score_gauge,
)
from app.llm.explanation import ExplanationGenerator
from app.models.analysis import CheatingRiskAnalysis
from app.repositories.analysis_result import AnalysisResultRepository
from app.repositories.participant import ItemSetParticipantRepository
from app.schemas.features import AggregatedFeatures
from app.scoring.aggregator import FeatureAggregator
from app.scoring.engine import RuleBasedRiskEngine

logger = get_logger(__name__)

T = TypeVar("T")

DEFAULT_RETRIES = 2
RETRY_DELAY_S = 2.0


async def _retry(
    fn: Callable[..., Awaitable[T]],
    *args: Any,
    retries: int = DEFAULT_RETRIES,
    step_name: str = "",
    **kwargs: Any,
) -> T:
    """Execute an async call with simple retry logic."""
    last_exc: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return await fn(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            logger.warning(
                "orchestrator_retry",
                step=step_name,
                attempt=attempt + 1,
                max_retries=retries,
                error=str(exc),
            )
            if attempt < retries:
                await asyncio.sleep(RETRY_DELAY_S * (attempt + 1))
    raise last_exc  # type: ignore[misc]


class AnalysisOrchestrator:
    """Orchestrates the full cheating-risk analysis pipeline with per-step retry.

    Pipeline steps
    -------------
    1. Load participant record from LMS.
    2. Analyze activity logs.
    3. Download videos and extract frames.
    4. Run face analysis.
    5. Run object detection (YOLO).
    6. Aggregate all features.
    7. Calculate risk score / level / probability.
    8. Build suspicious event timeline.
    9. Generate AI explanation.
    10. Save results to the database.
    """

    def __init__(
        self,
        analysis_result_repo: AnalysisResultRepository,
        participant_repo: ItemSetParticipantRepository,
        activity_analyzer: ActivityAnalyzer,
        video_analyzer: VideoAnalyzer,
        feature_aggregator: FeatureAggregator,
        scoring_engine: RuleBasedRiskEngine,
        explanation_gen: ExplanationGenerator,
    ) -> None:
        self.analysis_result_repo = analysis_result_repo
        self.participant_repo = participant_repo
        self.activity_analyzer = activity_analyzer
        self.video_analyzer = video_analyzer
        self.feature_aggregator = feature_aggregator
        self.scoring_engine = scoring_engine
        self.explanation_gen = explanation_gen

    async def run_analysis(
        self,
        participant_id: str,
    ) -> CheatingRiskAnalysis:
        pid = uuid.UUID(participant_id)
        start = time.monotonic()

        try:
            # Step 1 — load participant
            t0 = time.monotonic()
            participant = await _retry(
                self.participant_repo.find_by_id,
                pid,
                step_name="load_participant",
            )
            if participant is None:
                raise ValueError(
                    f"Participant {participant_id} not found"
                )
            analysis_step_duration.labels(step="load_participant").observe(
                time.monotonic() - t0
            )

            # Set status to ANALYZING
            await self.participant_repo.update_analysis_status(
                participant_id=pid,
                status="Analyzing",
            )

            activity_log = participant.activity_log or []
            face_records = participant.face_records or []

            # Step 2 — analyze activity logs
            t0 = time.monotonic()
            activity_features = await _retry(
                self.activity_analyzer.analyze,
                activity_log,
                step_name="activity_analysis",
            )
            analysis_step_duration.labels(step="activity_analysis").observe(
                time.monotonic() - t0
            )

            # Steps 3–5 — video download, frame extraction, face analysis, object detection
            t0 = time.monotonic()
            face_features, face_results, detection_results, merged_s3_key = await _retry(
                self.video_analyzer.analyze,
                face_records,
                step_name="video_analysis",
                session_id=participant_id,
            )
            analysis_step_duration.labels(step="video_analysis").observe(
                time.monotonic() - t0
            )

            # Step 6 — build suspicious timeline
            t0 = time.monotonic()
            timeline_builder = SuspiciousTimelineBuilder()
            suspicious_timeline = await _retry(
                _build_timeline,
                timeline_builder,
                activity_log,
                face_results,
                detection_results,
                float(self.video_analyzer.fps),
                step_name="timeline_build",
            )
            analysis_step_duration.labels(step="timeline_build").observe(
                time.monotonic() - t0
            )

            # Step 7 — aggregate features
            aggregated = self.feature_aggregator.aggregate(
                activity_features, face_features
            )

            # Step 8 — calculate risk
            score = self.scoring_engine.calculate_score(aggregated)
            probability = self.scoring_engine.calculate_probability(score)
            risk_level = self.scoring_engine.determine_risk_level(score)

            # Step 9 — generate AI explanation
            t0 = time.monotonic()
            summary = await _retry(
                self.explanation_gen.generate_explanation,
                risk_score=score,
                risk_level=risk_level,
                features=aggregated,
                step_name="explanation",
            )
            analysis_step_duration.labels(step="explanation").observe(
                time.monotonic() - t0
            )

            # Step 10 — save results + mark participant as COMPLETED
            t0 = time.monotonic()
            analysis = await _retry(
                self.analysis_result_repo.save_with_status,
                participant_id=pid,
                risk_score=score,
                cheating_probability=probability,
                risk_level=risk_level,
                activity_features=activity_features.model_dump(),
                face_features=face_features.model_dump(),
                suspicious_timeline=suspicious_timeline,
                ai_summary=summary,
                step_name="save_results",
            )
            analysis_step_duration.labels(step="save_results").observe(
                time.monotonic() - t0
            )

            # Step 11 — replace face_records with merged video link
            if merged_s3_key:
                from datetime import datetime, timezone
                from app.core.config import settings

                merged_url = (
                    f"https://{settings.aws_s3_bucket}"
                    f".s3.{settings.aws_region}.amazonaws.com"
                    f"/{merged_s3_key}"
                )
                await self.participant_repo.update_face_records(
                    participant_id=pid,
                    face_records=[{
                        "s3_key": merged_s3_key,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "video_url": merged_url,
                    }],
                )

            # Set status to COMPLETED
            await self.participant_repo.update_analysis_status(
                participant_id=pid,
                status="Completed",
            )

            # Record metrics
            duration = time.monotonic() - start
            analysis_duration.observe(duration)
            analysis_total.labels(status="success").inc()
            risk_score_gauge.set(score)
            _set_risk_level_gauge(risk_level)
            _record_feature_gauges(aggregated)

            logger.info(
                "analysis_completed",
                participant_id=participant_id,
                risk_score=score,
                risk_level=risk_level,
                duration_seconds=round(duration, 2),
            )
            return analysis

        except Exception as exc:
            analysis_total.labels(status="failed").inc()
            raise


RISK_LEVEL_MAP = {
    "Low": 0.0,
    "Moderate": 1.0,
    "Elevated": 2.0,
    "High": 3.0,
    "Critical": 4.0,
}


def _set_risk_level_gauge(level: str) -> None:
    risk_level_gauge.set(RISK_LEVEL_MAP.get(level, 0.0))


def _record_feature_gauges(features: AggregatedFeatures) -> None:
    for field_name in features.model_fields_set or features.model_fields:
        val = getattr(features, field_name, None)
        if isinstance(val, (int, float)):
            feature_gauge.labels(feature=field_name).set(val)


async def _build_timeline(
    builder: SuspiciousTimelineBuilder,
    activity_log: list[dict[str, Any]],
    face_results: list[Any],
    detection_results: list[Any],
    analysis_fps: float,
) -> list[dict[str, Any]]:
    """Synchronous timeline builder wrapped for retry compatibility."""
    return builder.build(
        activity_events=activity_log,
        face_results=face_results,
        detection_results=detection_results,
        analysis_fps=analysis_fps,
    )
