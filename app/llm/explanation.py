from openai import AsyncOpenAI

from app.core.config import settings
from app.schemas.features import AggregatedFeatures
from app.scoring.weights import ScoreWeights
from app.core.logging import get_logger

logger = get_logger(__name__)


SYSTEM_INSTRUCTION = (
    "You are an academic integrity analysis assistant. "
    "Your task is to summarize observed behaviors from an online exam monitoring system. "
    "Never claim cheating occurred. Only describe objective evidence. "
    "Mention the strongest contributing factors. "
    "Use simple, non-technical language that anyone can understand. "
    "Avoid terms like 'instances', 'frames', or 'events' — use plain words like 'detected', 'observed', or 'seen'. "
    "For phone detection, say 'phone was visible' or 'possible phone use was detected'. "
    "Keep the summary under 4 sentences. "
    "End with a manual review recommendation when the risk level is Elevated or higher."
)

FEATURE_MAP: list[tuple[str, str, str]] = [
    # (aggregated_field, weight_field, human_label)
    ("tab_switches", "tab_switch", "tab switches"),
    ("screenshot_attempts", "screenshot_attempt", "screenshot attempts"),
    ("page_refresh_attempts", "page_refresh_attempt", "page refresh attempts"),
    ("fullscreen_exit_attempts", "fullscreen_exit_attempt", "fullscreen exit attempts"),
    ("face_missing_duration", "face_missing_per_sec", "face missing (seconds)"),
    ("look_away_duration", "look_away_per_sec", "look away (seconds)"),
    ("multiple_face_events", "multiple_face_event", "multiple faces detected"),
    ("phone_detected_frames", "phone_detected_frame", "phone detected"),
    ("tablet_detected_frames", "tablet_detected_frame", "tablet detected"),
    ("book_detected_frames", "book_detected_frame", "book or notes detected"),
    ("side_glance_count", "side_glance", "side glances"),
    ("speaking_events", "speaking_event", "speaking detected"),
    ("eyes_closed_duration", "eyes_closed_per_sec", "eyes closed (seconds)"),
]


def _find_top_contributors(
    features: AggregatedFeatures,
    weights: ScoreWeights,
    top_n: int = 3,
) -> list[str]:
    """Identify the strongest contributing factors by weighted value."""
    scored: list[tuple[float, str]] = []
    for field_name, weight_name, label in FEATURE_MAP:
        value = getattr(features, field_name, 0) or 0
        weight = getattr(weights, weight_name, 0) or 0
        contribution = value * weight
        if contribution > 0:
            scored.append((contribution, label))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [label for _, label in scored[:top_n]]


class OpenAIExplanationService:
    """Generates human-readable explanations using the OpenAI Responses API.

    Uses the unified feature vector (AggregatedFeatures) as structured input,
    identifies strongest contributing factors, and produces concise academic summaries.
    """

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self.api_key = api_key or settings.openai_api_key
        self.model = model or settings.openai_model
        self._client: AsyncOpenAI | None = None

    @property
    def client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.api_key)
        return self._client

    async def generate_explanation(
        self,
        risk_score: float,
        risk_level: str,
        features: AggregatedFeatures,
    ) -> str:
        weights = ScoreWeights()
        top_factors = _find_top_contributors(features, weights)

        summary_line = (
            f"Risk Score: {risk_score}/100 ({risk_level}). "
            f"Strongest factors: {', '.join(top_factors)}."
        )
        feature_lines = "\n".join(
            f"- {label}: {getattr(features, field, 0)}"
            for field, _, label in FEATURE_MAP
        )

        user_input = (
            f"Summarize this exam session analysis:\n\n"
            f"{summary_line}\n\n"
            f"All observed features:\n{feature_lines}\n"
        )

        response = await self.client.responses.create(
            model=self.model,
            instructions=SYSTEM_INSTRUCTION,
            input=user_input,
            temperature=0.3,
            max_output_tokens=512,
        )
        summary = response.output_text
        logger.info(
            "explanation_generated",
            risk_score=risk_score,
            risk_level=risk_level,
            top_factors=top_factors,
        )
        return summary


# Backward-compatible alias
ExplanationGenerator = OpenAIExplanationService
