"""Prometheus metrics for the cheating risk analysis service."""

from prometheus_client import Counter, Gauge, Histogram

# --- HTTP metrics ---
http_request_count = Counter(
    "http_requests_total",
    "Total HTTP requests",
    labelnames=["method", "path", "status"],
)

http_request_duration = Histogram(
    "http_request_duration_seconds",
    "HTTP request duration in seconds",
    labelnames=["method", "path"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0),
)

http_request_in_flight = Gauge(
    "http_requests_in_flight",
    "Number of HTTP requests currently in flight",
    labelnames=["method"],
)

# --- Analysis pipeline metrics ---
analysis_duration = Histogram(
    "analysis_duration_seconds",
    "Duration of the full analysis pipeline",
    buckets=(10, 30, 60, 120, 300, 600, 900, 1800),
)

analysis_step_duration = Histogram(
    "analysis_step_duration_seconds",
    "Duration of individual analysis pipeline steps",
    labelnames=["step"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600),
)

analysis_total = Counter(
    "analysis_total",
    "Total number of analysis runs",
    labelnames=["status"],
)

analysis_failures = Counter(
    "analysis_failures_total",
    "Analysis failures by step",
    labelnames=["step"],
)

# --- Celery task metrics ---
celery_task_duration = Histogram(
    "celery_task_duration_seconds",
    "Duration of Celery tasks",
    labelnames=["task_name"],
    buckets=(1, 5, 10, 30, 60, 120, 300, 600, 1800),
)

celery_task_total = Counter(
    "celery_task_total",
    "Total number of Celery task executions",
    labelnames=["task_name", "status"],
)

# --- Detection metrics ---
detection_objects_total = Counter(
    "detection_objects_total",
    "Detected objects by class",
    labelnames=["class"],
)

# --- Feature metrics ---
feature_gauge = Gauge(
    "analysis_feature_value",
    "Aggregated feature values from analysis runs",
    labelnames=["feature"],
)

# --- Risk score distribution ---
risk_score_gauge = Gauge("analysis_risk_score", "Risk score from last analysis run")

risk_level_gauge = Gauge(
    "analysis_risk_level",
    "Risk level as numeric value (0=Low, 1=Moderate, 2=Elevated, 3=High, 4=Critical)",
)
