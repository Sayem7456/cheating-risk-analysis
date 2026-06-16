from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # --- Project ---
    project_name: str = "Cheating Risk Analysis"
    debug: bool = False
    is_dev: bool = True

    # --- Database (LMS PostgreSQL) ---
    postgres_server: str = "localhost"
    postgres_user: str = "postgres"
    postgres_password: str = ""
    postgres_db: str = "cheating_risk"
    database_url: str | None = Field(
        default=None,
        alias="SQLALCHEMY_DATABASE_URI",
    )

    @property
    def db_url(self) -> str:
        if self.database_url:
            raw = self.database_url
            if "+" not in raw.split("://")[0]:
                raw = raw.replace("postgresql://", "postgresql+asyncpg://")
            return raw
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_server}:5432/{self.postgres_db}"
        )

    @property
    def db_url_sync(self) -> str:
        if self.database_url:
            return self.database_url.replace("+asyncpg", "")
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_server}:5432/{self.postgres_db}"
        )

    # --- Redis ---
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_db: int = 0
    redis_password: str | None = None

    @property
    def redis_url(self) -> str:
        pwd = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{pwd}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    # --- Celery ---
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None

    @property
    def celery_broker(self) -> str:
        return self.celery_broker_url or self.redis_url

    @property
    def celery_backend(self) -> str:
        return self.celery_result_backend or self.redis_url

    # --- Scheduler ---
    scheduler_enabled: bool = True
    scheduler_interval_minutes: int = 5
    scheduler_start_hour: int = 18
    scheduler_end_hour: int = 23
    scheduler_max_dispatch: int = 100
    redis_lock_ttl_seconds: int = 300

    # --- AWS S3 ---
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    aws_region: str = "ap-southeast-1"
    aws_s3_bucket: str = "imch-medical-ai"
    s3_verify_integrity: bool = True
    s3_connect_timeout: int = 30
    s3_read_timeout: int = 120

    # --- OpenAI ---
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_concurrency: int = 2
    openai_self_consistency_runs: int = 3

    # --- Video Analysis ---
    model_dir: str = "models"
    video_frame_fps: int = 1
    video_temp_dir: str = "/tmp/cheating-analysis/videos"

    @property
    def model_asset_path(self) -> str:
        return str(
            Path(__file__).resolve().parent.parent.parent
            / self.model_dir
            / "face_landmarker.task"
        )

    # --- Scoring weights ---
    weight_tab_switch: float = 4.0
    weight_screenshot_attempt: float = 8.0
    weight_page_refresh_attempt: float = 6.0
    weight_fullscreen_exit_attempt: float = 5.0
    weight_face_missing_per_sec: float = 0.3
    weight_look_away_per_sec: float = 0.1
    weight_multiple_face_event: float = 20.0
    weight_phone_detected_frame: float = 3.0
    weight_tablet_detected_frame: float = 0.8
    weight_book_detected_frame: float = 0.5
    weight_side_glance: float = 1.5
    weight_speaking_event: float = 5.0
    weight_eyes_closed_per_sec: float = 0.2

    # --- Risk thresholds ---
    risk_low_max: float = 20.0
    risk_moderate_max: float = 40.0
    risk_elevated_max: float = 60.0
    risk_high_max: float = 80.0


settings = Settings()
