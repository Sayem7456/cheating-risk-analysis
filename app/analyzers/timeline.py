from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.analyzers.detection import FrameDetectionResult
from app.analyzers.face import FrameFaceResult
from app.core.logging import get_logger

logger = get_logger(__name__)

# Activity event types that appear in the timeline
TIMELINE_ACTIVITY_EVENTS = {
    "TAB_SWITCH",
    "SCREENSHOT_ATTEMPT",
    "PAGE_REFRESH_ATTEMPT",
    "FULLSCREEN_EXIT",
    "WINDOW_BLUR",
    "AUTO_SUBMIT",
    "ESCAPE_ATTEMPT",
    "NEW_TAB_ATTEMPT",
    "NEW_WINDOW_ATTEMPT",
    "CLIPBOARD_ATTEMPT",
    "CONTEXT_MENU_ATTEMPT",
    "DEVTOOLS_ATTEMPT",
    "BLOCKED_SHORTCUT",
}

# Frame event types
FACE_MISSING = "FACE_MISSING"
MULTIPLE_FACES = "MULTIPLE_FACES"
LOOK_AWAY = "LOOK_AWAY"
PHONE_DETECTED = "PHONE_DETECTED"
BOOK_DETECTED = "BOOK_DETECTED"

# Maximum gap (in frames) between same-type frame events to consider them merged
MERGE_MAX_GAP_FRAMES = 1


def _format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    total = int(round(seconds))
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _parse_activity_timestamp(
    ts: Any, reference: float | None
) -> float | None:
    """Parse an activity log timestamp and return seconds relative to reference."""
    if isinstance(ts, (int, float)):
        raw = float(ts)
    elif isinstance(ts, str):
        try:
            dt = datetime.fromisoformat(ts)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            raw = dt.timestamp()
        except (ValueError, TypeError):
            return None
    else:
        return None

    if reference is not None:
        return round(raw - reference, 2)
    return 0.0


def _find_earliest_timestamp(
    activity_events: list[dict[str, Any]],
) -> float | None:
    """Find the earliest absolute timestamp in activity events."""
    earliest = None
    for event in activity_events:
        ts = event.get("timestamp")
        if isinstance(ts, (int, float)):
            raw = float(ts)
        elif isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                raw = dt.timestamp()
            except (ValueError, TypeError):
                continue
        else:
            continue
        if earliest is None or raw < earliest:
            earliest = raw
    return earliest


@dataclass
class TimelineEvent:
    timestamp_seconds: float = 0.0
    event_type: str = ""
    duration: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "time": _format_timestamp(self.timestamp_seconds),
            "event": self.event_type,
            "duration": round(self.duration, 1),
        }


class SuspiciousTimelineBuilder:
    """Builds a chronological timeline of suspicious events from all analysis sources."""

    def build(
        self,
        activity_events: list[dict[str, Any]],
        face_results: list[FrameFaceResult],
        detection_results: list[FrameDetectionResult],
        analysis_fps: float = 1.0,
    ) -> list[dict[str, Any]]:
        """Build the full timeline.

        Args:
            activity_events: Raw activity log entries with ``event`` and ``timestamp`` keys.
            face_results: Per-frame face analysis results.
            detection_results: Per-frame object detection results.
            analysis_fps: Frame rate at which frames were analyzed (default 1 FPS).

        Returns:
            List of timeline event dicts, sorted chronologically.
        """
        events: list[TimelineEvent] = []

        # 1. Activity events
        ref = _find_earliest_timestamp(activity_events)
        for ae in activity_events:
            etype = ae.get("event", "")
            if etype not in TIMELINE_ACTIVITY_EVENTS:
                continue
            rel = _parse_activity_timestamp(ae.get("timestamp"), ref)
            if rel is not None:
                events.append(TimelineEvent(
                    timestamp_seconds=rel, event_type=etype, duration=0.0,
                ))

        sec_per_frame = 1.0 / max(analysis_fps, 1.0)

        # 2. Face frame events
        self._add_face_frame_events(events, face_results, sec_per_frame)

        # 3. Detection frame events
        self._add_detection_frame_events(events, detection_results, sec_per_frame)

        if not events:
            return []

        events.sort(key=lambda e: e.timestamp_seconds)
        merged = self._merge_events(events)
        return [e.to_dict() for e in merged]

    def _add_face_frame_events(
        self,
        events: list[TimelineEvent],
        results: list[FrameFaceResult],
        sec_per_frame: float,
    ) -> None:
        """Add face-related frame events (face missing, multiple faces, look away)."""
        for i, r in enumerate(results):
            ts = round(i * sec_per_frame, 2)
            if not r.face_detected:
                events.append(TimelineEvent(
                    timestamp_seconds=ts, event_type=FACE_MISSING, duration=sec_per_frame,
                ))
            elif r.face_count > 1:
                events.append(TimelineEvent(
                    timestamp_seconds=ts, event_type=MULTIPLE_FACES, duration=sec_per_frame,
                ))
            elif r.gaze_direction not in ("center", "unknown"):
                events.append(TimelineEvent(
                    timestamp_seconds=ts, event_type=LOOK_AWAY, duration=sec_per_frame,
                ))

    def _add_detection_frame_events(
        self,
        events: list[TimelineEvent],
        results: list[FrameDetectionResult],
        sec_per_frame: float,
    ) -> None:
        """Add detection-based frame events (phone, book detected)."""
        for i, r in enumerate(results):
            ts = round(i * sec_per_frame, 2)
            for obj in r.objects:
                etype = None
                if obj.label == "phone":
                    etype = PHONE_DETECTED
                elif obj.label == "book":
                    etype = BOOK_DETECTED
                if etype is not None:
                    events.append(TimelineEvent(
                        timestamp_seconds=ts, event_type=etype, duration=sec_per_frame,
                    ))

    def _merge_events(
        self, events: list[TimelineEvent]
    ) -> list[TimelineEvent]:
        """Merge consecutive same-type events, then sort chronologically.

        Events of the same type whose timestamps are within
        MERGE_MAX_GAP_FRAMES * sec_per_frame are merged into one.
        """
        if not events:
            return []

        by_type: dict[str, list[TimelineEvent]] = {}
        for e in events:
            by_type.setdefault(e.event_type, []).append(e)

        merged: list[TimelineEvent] = []
        for evs in by_type.values():
            evs.sort(key=lambda e: e.timestamp_seconds)
            current = evs[0]
            for next_ev in evs[1:]:
                gap = next_ev.timestamp_seconds - (
                    current.timestamp_seconds + current.duration
                )
                if gap <= MERGE_MAX_GAP_FRAMES * (next_ev.duration or 1.0):
                    new_end = next_ev.timestamp_seconds + next_ev.duration
                    current.duration = round(
                        new_end - current.timestamp_seconds, 2
                    )
                else:
                    merged.append(current)
                    current = next_ev
            merged.append(current)

        merged.sort(key=lambda e: e.timestamp_seconds)
        return merged
