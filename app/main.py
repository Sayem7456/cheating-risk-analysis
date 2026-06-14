from contextlib import asynccontextmanager

from fastapi import FastAPI
from prometheus_client import make_asgi_app

from app.api.router import router
from app.core.config import settings
from app.core.logging import setup_logging
from app.core.metrics import http_request_count, http_request_duration, http_request_in_flight
from app.core.tracing import TracingMiddleware


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging(debug=settings.debug)
    yield


app = FastAPI(
    title=settings.project_name,
    version="0.1.0",
    lifespan=lifespan,
)

# --- Middleware (order matters: tracing first, then metrics) ---
app.add_middleware(TracingMiddleware)


@app.middleware("http")
async def metrics_middleware(request, call_next):
    import time

    method = request.method
    path = request.url.path

    # Skip metrics scraping from the metrics endpoint itself
    if path == "/metrics":
        return await call_next(request)

    http_request_in_flight.labels(method=method).inc()
    start = time.monotonic()

    try:
        response = await call_next(request)
        status = str(response.status_code)
        http_request_count.labels(method=method, path=path, status=status).inc()
        return response
    except Exception:
        http_request_count.labels(method=method, path=path, status="500").inc()
        raise
    finally:
        http_request_duration.labels(method=method, path=path).observe(
            time.monotonic() - start
        )
        http_request_in_flight.labels(method=method).dec()


# --- Routes ---
app.include_router(router)

# --- Prometheus metrics endpoint ---
app.mount("/metrics", make_asgi_app())
