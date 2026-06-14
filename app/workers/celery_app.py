import time
import uuid

import structlog
from celery import Celery
from celery.schedules import crontab
from celery.signals import (
    task_failure,
    task_postrun,
    task_prerun,
)

from app.core.config import settings
from app.core.metrics import celery_task_duration, celery_task_total
from app.core.tracing import generate_trace_id

celery_app = Celery(
    "cheating_risk_analysis",
    broker=settings.celery_broker,
    backend=settings.celery_backend,
    include=["app.workers.tasks"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,
    task_soft_time_limit=1800,
    task_time_limit=3600,
)

celery_app.conf.beat_schedule = {
    "discover-completed-exams": {
        "task": "app.workers.tasks.discover_exams",
        "schedule": crontab(
            minute=f"*/{settings.scheduler_interval_minutes}",
            hour=f"{settings.scheduler_start_hour}-{settings.scheduler_end_hour}",
        ),
        "options": {"expires": 60},
    },
}

# --- Tracing & metrics via Celery signals ---

_task_start_times: dict[str, float] = {}


@task_prerun.connect
def on_task_prerun(task_id, task, args, kwargs, **_):
    """Bind trace context and record start time for every task."""
    trace_id = kwargs.pop("trace_id", None) or generate_trace_id()
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        task_id=task_id,
        trace_id=trace_id,
        task_name=task.name,
    )
    _task_start_times[task_id] = time.monotonic()


@task_postrun.connect
def on_task_postrun(task_id, task, state, retval, **_):
    """Record duration and result metrics after task completes."""
    start = _task_start_times.pop(task_id, None)
    if start is not None:
        duration = time.monotonic() - start
        task_name = task.name
        celery_task_duration.labels(task_name=task_name).observe(duration)
        celery_task_total.labels(task_name=task_name, status=state).inc()

    structlog.contextvars.unbind_contextvars("task_id", "trace_id", "task_name")


@task_failure.connect
def on_task_failure(task_id, task, args, kwargs, einfo, **_):
    """Log task failures with full exception info."""
    logger = structlog.get_logger(__name__)
    logger.error(
        "celery_task_failed",
        task_id=task_id,
        task_name=task.name,
        error=str(einfo.exception),
        traceback=str(einfo.traceback),
    )
