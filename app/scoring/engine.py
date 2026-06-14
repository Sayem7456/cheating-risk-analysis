from app.core.config import Settings
from app.schemas.features import AggregatedFeatures
from app.scoring.weights import ScoreWeights
from app.core.logging import get_logger

logger = get_logger(__name__)


class RuleBasedRiskEngine:
    """Rule-based risk scoring engine.

    Combines weighted feature values into a single risk score (0-100),
    determines risk level, and computes a cheating probability estimate.

    Rules
    -----
    Each `AggregatedFeatures` field contributes risk via a configurable weight.
    The weights are loaded from ``Settings`` and can be overridden via env vars
    or a ``.env`` file. The final score is capped at 100.

    Risk levels
    -----------
    - 0–20   → Low
    - 21–40  → Moderate
    - 41–60  → Elevated
    - 61–80  → High
    - 81–100 → Critical
    """

    def __init__(self, settings: Settings) -> None:
        self.weights = ScoreWeights(
            tab_switch=settings.weight_tab_switch,
            screenshot_attempt=settings.weight_screenshot_attempt,
            page_refresh_attempt=settings.weight_page_refresh_attempt,
            fullscreen_exit_attempt=settings.weight_fullscreen_exit_attempt,
            face_missing_per_sec=settings.weight_face_missing_per_sec,
            look_away_per_sec=settings.weight_look_away_per_sec,
            multiple_face_event=settings.weight_multiple_face_event,
            phone_detected_frame=settings.weight_phone_detected_frame,
            tablet_detected_frame=settings.weight_tablet_detected_frame,
            book_detected_frame=settings.weight_book_detected_frame,
            side_glance=settings.weight_side_glance,
            speaking_event=settings.weight_speaking_event,
            eyes_closed_per_sec=settings.weight_eyes_closed_per_sec,
        )
        self.low_max = settings.risk_low_max
        self.moderate_max = settings.risk_moderate_max
        self.elevated_max = settings.risk_elevated_max
        self.high_max = settings.risk_high_max

    def calculate_score(self, features: AggregatedFeatures) -> float:
        raw = (
            features.tab_switches * self.weights.tab_switch
            + features.screenshot_attempts * self.weights.screenshot_attempt
            + features.page_refresh_attempts * self.weights.page_refresh_attempt
            + features.fullscreen_exit_attempts * self.weights.fullscreen_exit_attempt
            + features.face_missing_duration * self.weights.face_missing_per_sec
            + features.look_away_duration * self.weights.look_away_per_sec
            + features.multiple_face_events * self.weights.multiple_face_event
            + features.phone_detected_frames * self.weights.phone_detected_frame
            + features.tablet_detected_frames * self.weights.tablet_detected_frame
            + features.book_detected_frames * self.weights.book_detected_frame
            + features.side_glance_count * self.weights.side_glance
            + features.speaking_events * self.weights.speaking_event
            + features.eyes_closed_duration * self.weights.eyes_closed_per_sec
        )
        return min(round(raw, 1), 100.0)

    def calculate_probability(self, score: float) -> float:
        return round(min(score / 100.0, 1.0), 4)

    def determine_risk_level(self, score: float) -> str:
        if score <= self.low_max:
            return "Low"
        if score <= self.moderate_max:
            return "Moderate"
        if score <= self.elevated_max:
            return "Elevated"
        if score <= self.high_max:
            return "High"
        return "Critical"


# Backward-compatible alias
RiskScoringEngine = RuleBasedRiskEngine
