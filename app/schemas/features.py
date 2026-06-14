from pydantic import BaseModel


class ActivityFeatures(BaseModel):
    total_tab_switches: int = 0
    total_ss_attempt: int = 0
    total_page_refresh_attempt: int = 0
    total_fullscreen_exit_attempt: int = 0
    switches_per_minute: float = 0.0
    focus_loss_count: int = 0
    auto_submit_count: int = 0
    first_switch_time: float | None = None
    last_switch_time: float | None = None


class FaceFeatures(BaseModel):
    face_missing_duration: float = 0.0
    multiple_face_events: int = 0
    multiple_face_duration: float = 0.0
    first_occurrence_timestamp: float = 0.0
    look_away_duration: float = 0.0
    side_glance_count: int = 0
    screen_attention_ratio: float = 1.0
    blink_rate: float = 0.0
    eyes_closed_duration: float = 0.0
    speaking_events: int = 0
    phone_detected_frames: int = 0
    tablet_detected_frames: int = 0
    book_detected_frames: int = 0


class AggregatedFeatures(BaseModel):
    """Unified feature vector combining activity + face + detection features for scoring."""

    tab_switches: int = 0
    screenshot_attempts: int = 0
    page_refresh_attempts: int = 0
    fullscreen_exit_attempts: int = 0
    switches_per_minute: float = 0.0
    focus_loss_count: int = 0
    auto_submit_count: int = 0
    face_missing_duration: float = 0.0
    look_away_duration: float = 0.0
    multiple_face_events: int = 0
    multiple_face_duration: float = 0.0
    phone_detected_frames: int = 0
    tablet_detected_frames: int = 0
    book_detected_frames: int = 0
    side_glance_count: int = 0
    speaking_events: int = 0
    eyes_closed_duration: float = 0.0
