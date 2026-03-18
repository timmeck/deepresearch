"""Auth middleware for DeepResearch."""

from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.config import DEEPRESEARCH_API_KEY


class AuthMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        if not DEEPRESEARCH_API_KEY:
            return await call_next(request)
        path = request.url.path
        if path in {"/", "/api/status", "/api/events/stream"}:
            return await call_next(request)
        if request.method == "GET" and path.startswith("/api/"):
            return await call_next(request)
        key = request.query_params.get("key") or request.headers.get("X-API-Key", "")
        if key != DEEPRESEARCH_API_KEY:
            return JSONResponse({"error": "Unauthorized"}, status_code=401)
        return await call_next(request)
