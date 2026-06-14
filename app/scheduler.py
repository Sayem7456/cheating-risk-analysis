"""Scheduler for discovering completed exams and dispatching analysis jobs.

Queries item_set_participant for is_evaluated=True records that haven't
been processed yet. Uses Redis-based distributed locking to avoid
duplicate job creation across multiple scheduler instances.
"""

from app.core.config import settings
from app.core.logging import get_logger
from app.workers.celery_app import celery_app

logger = get_logger(__name__)

LOCK_PREFIX = "analysis:lock:"
LOCK_TTL_SECONDS = 300
MAX_DISPATCH_PER_RUN = 100


def _acquire_lock(redis_client, lock_key: str) -> bool:
    """Try to acquire a Redis lock. Returns True if acquired."""
    import time

    now = int(time.time())
    acquired = redis_client.setnx(lock_key, now)
    if acquired:
        redis_client.expire(lock_key, LOCK_TTL_SECONDS)
        return True
    return False


def discover_and_dispatch() -> int:
    """Query LMS DB for completed exams and enqueue Celery tasks.

    Uses raw SQL for this synchronous polling path to avoid
    depending on the full async ORM stack in the scheduler process.
    """
    try:
        import redis as redis_module

        redis_client = redis_module.Redis.from_url(settings.redis_url)
    except Exception:
        logger.warning("redis_unavailable_fallback_no_locks")
        redis_client = None

    from sqlalchemy import create_engine, text

    engine = create_engine(settings.db_url_sync)
    dispatched = 0

    try:
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    "SELECT id, student_id, set_id "
                    "FROM item_set_participant "
                    "WHERE is_evaluated = TRUE "
                    "AND analysis_status IS NULL "
                    "LIMIT :limit"
                ),
                {"limit": MAX_DISPATCH_PER_RUN},
            ).fetchall()

            for row in rows:
                record_id = str(row[0])
                student_id = str(row[1])
                set_id = str(row[2])

                if redis_client:
                    lock_key = f"{LOCK_PREFIX}{record_id}"
                    if not _acquire_lock(redis_client, lock_key):
                        continue

                celery_app.send_task(
                    "app.workers.tasks.analyze_exam_session",
                    args=[student_id, set_id],
                )
                dispatched += 1

        if dispatched:
            logger.info("dispatched_analysis_jobs", count=dispatched)
        return dispatched

    except Exception:
        logger.exception("scheduler_discovery_failed")
        return 0
