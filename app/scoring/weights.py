from dataclasses import dataclass


@dataclass(frozen=True)
class ScoreWeights:
    """Configurable scoring weights loaded from Settings.

    Each attribute corresponds to a ``weight_*`` field in ``Settings``
    and can be overridden via environment variable or ``.env`` file.
    """

    tab_switch: float = 4.0
    screenshot_attempt: float = 8.0
    page_refresh_attempt: float = 6.0
    fullscreen_exit_attempt: float = 5.0
    face_missing_per_sec: float = 0.3
    look_away_per_sec: float = 0.1
    multiple_face_event: float = 20.0
    phone_detected_frame: float = 0.6
    tablet_detected_frame: float = 0.8
    book_detected_frame: float = 0.5
    side_glance: float = 1.5
    speaking_event: float = 5.0
    eyes_closed_per_sec: float = 0.2
