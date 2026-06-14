from datetime import datetime, timezone
from typing import Any

from app.analyzers.base import BaseAnalyzer
from app.schemas.features import ActivityFeatures
from app.core.logging import get_logger

logger = get_logger(__name__)

SUSPICIOUS_EVENTS = {
    "TAB_SWITCH",
    "WINDOW_BLUR",
    "FULLSCREEN_EXIT",
    "PAGE_REFRESH_ATTEMPT",
    "CLOSE_TAB_ATTEMPT",
    "NEW_TAB_ATTEMPT",
    "NEW_WINDOW_ATTEMPT",
    "ESCAPE_ATTEMPT",
    "FULLSCREEN_TOGGLE_ATTEMPT",
    "SCREENSHOT_ATTEMPT",
    "DEVTOOLS_ATTEMPT",
    "ZOOM_ATTEMPT",
    "CLIPBOARD_ATTEMPT",
    "CONTEXT_MENU_ATTEMPT",
    "BLOCKED_SHORTCUT",
}

SS_ATTEMPT_EVENTS = {"SCREENSHOT_ATTEMPT"}
PAGE_REFRESH_EVENTS = {"PAGE_REFRESH_ATTEMPT"}
FULLSCREEN_EXIT_EVENTS = {"FULLSCREEN_EXIT", "FULLSCREEN_TOGGLE_ATTEMPT"}
FOCUS_LOSS_EVENTS = {"WINDOW_BLUR"}
AUTO_SUBMIT_EVENTS = {"AUTO_SUBMIT"}


def _parse_timestamp(ts: Any) -> float | None:
    if isinstance(ts, (int, float)):
        return float(ts)
    if isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except (ValueError, TypeError):
            return None
    return None


class ActivityAnalyzer(BaseAnalyzer):
    async def analyze(self, activity_log: list[dict[str, Any]]) -> ActivityFeatures:
        if not activity_log:
            return ActivityFeatures()

        tab_switch_timestamps: list[float] = []
        ss_count = 0
        refresh_count = 0
        fullscreen_exit_count = 0
        focus_loss_count = 0
        auto_submit_count = 0
        all_timestamps: list[float] = []

        for event in activity_log:
            event_type = event.get("event", "")
            ts = _parse_timestamp(event.get("timestamp"))

            if ts is not None:
                all_timestamps.append(ts)

            if event_type == "TAB_SWITCH" and ts is not None:
                tab_switch_timestamps.append(ts)
            elif event_type in SS_ATTEMPT_EVENTS:
                ss_count += 1
            elif event_type in PAGE_REFRESH_EVENTS:
                refresh_count += 1
            elif event_type in FULLSCREEN_EXIT_EVENTS:
                fullscreen_exit_count += 1
            elif event_type in FOCUS_LOSS_EVENTS:
                focus_loss_count += 1
            elif event_type in AUTO_SUBMIT_EVENTS:
                auto_submit_count += 1

        first_switch_time: float | None = None
        last_switch_time: float | None = None
        switches_per_minute = 0.0

        if tab_switch_timestamps and all_timestamps:
            exam_start = min(all_timestamps)
            first_switch_time = round(tab_switch_timestamps[0] - exam_start, 2)
            last_switch_time = round(tab_switch_timestamps[-1] - exam_start, 2)

            time_span = tab_switch_timestamps[-1] - tab_switch_timestamps[0]
            if time_span > 0:
                switches_per_minute = round(
                    (len(tab_switch_timestamps) / time_span) * 60, 2
                )

        elif tab_switch_timestamps:
            first_switch_time = 0.0
            last_switch_time = 0.0

        features = ActivityFeatures(
            total_tab_switches=len(tab_switch_timestamps),
            total_ss_attempt=ss_count,
            total_page_refresh_attempt=refresh_count,
            total_fullscreen_exit_attempt=fullscreen_exit_count,
            switches_per_minute=switches_per_minute,
            focus_loss_count=focus_loss_count,
            auto_submit_count=auto_submit_count,
            first_switch_time=first_switch_time,
            last_switch_time=last_switch_time,
        )

        logger.debug(
            "activity_analysis_complete",
            features=features.model_dump(),
        )
        return features
