from app.analyzers.activity import ActivityAnalyzer
from app.analyzers.video import VideoAnalyzer
from app.schemas.features import ActivityFeatures, AggregatedFeatures, FaceFeatures
from app.core.logging import get_logger

logger = get_logger(__name__)


class FeatureAggregator:
    """Combines ActivityFeatures and FaceFeatures into a unified AggregatedFeatures vector.

    The unified vector serves as the single normalized input for the RiskScoringEngine.
    """

    def aggregate(
        self,
        activity: ActivityFeatures,
        face: FaceFeatures,
    ) -> AggregatedFeatures:
        """Merge activity and face features into a single scoring vector."""
        return AggregatedFeatures(
            tab_switches=activity.total_tab_switches,
            screenshot_attempts=activity.total_ss_attempt,
            page_refresh_attempts=activity.total_page_refresh_attempt,
            fullscreen_exit_attempts=activity.total_fullscreen_exit_attempt,
            switches_per_minute=activity.switches_per_minute,
            focus_loss_count=activity.focus_loss_count,
            auto_submit_count=activity.auto_submit_count,
            face_missing_duration=face.face_missing_duration,
            look_away_duration=face.look_away_duration,
            multiple_face_events=face.multiple_face_events,
            multiple_face_duration=face.multiple_face_duration,
            phone_detected_frames=face.phone_detected_frames,
            tablet_detected_frames=face.tablet_detected_frames,
            book_detected_frames=face.book_detected_frames,
            side_glance_count=face.side_glance_count,
            speaking_events=face.speaking_events,
            eyes_closed_duration=face.eyes_closed_duration,
        )
