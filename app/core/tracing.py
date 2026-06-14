"""Request tracing — injects trace/request IDs into structlog context."""

import uuid

import structlog
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response


def generate_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def get_request_id(request: Request) -> str:
    """Extract request ID from header or generate a new one."""
    return request.headers.get("X-Request-ID") or generate_trace_id()


class TracingMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware that injects request_id and trace_id into
    structlog context vars and response headers."""

    async def dispatch(self, request: Request, call_next):
        req_id = get_request_id(request)
        trace_id = request.headers.get("X-Trace-ID") or req_id

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=req_id,
            trace_id=trace_id,
        )

        response: Response = await call_next(request)
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Trace-ID"] = trace_id

        structlog.contextvars.unbind_contextvars("request_id", "trace_id")
        return response
