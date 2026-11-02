"""Public exhibition policy and bounded queues, including streamed responses."""

from __future__ import annotations

import asyncio
import re
from urllib.parse import parse_qs
from starlette.responses import JSONResponse


READ = re.compile(
    r"^/api/(?:health|session|favorites|ai/kimi/status|mainline/(?:status|cultures|catalog|random|upload_formats|tracks/[A-Za-z0-9_-]+|audio/[A-Za-z0-9_-]+))$"
)
WRITE = {
    "POST": {
        "/api/favorites",
        "/api/mainline/recommend",
        "/api/mainline/upload_recommend",
        "/api/ai/kimi/chat",
        "/api/ai/kimi/chat/stream",
    }
}
MEDIA = re.compile(
    r"^uploads/mainline/(?:raw|compressed|covers)/[0-9a-f]{32}_[^/\\%]+\.(?:mp3|m4a|mp4|aac|wav|wave|flac|ogg|oga|opus|webm|aif|aiff|wma|jpg|jpeg|png)$",
    re.I,
)


class ExhibitionGuard:
    def __init__(self, app, enabled=False, compute_concurrency=1, queue_limit=24, queue_timeout=45):
        self.app, self.enabled = app, enabled
        self.limits = {"compute": asyncio.Semaphore(compute_concurrency), "ai": asyncio.Semaphore(4)}
        self.waiting = {"compute": 0, "ai": 0}
        self.queue_limit, self.queue_timeout = queue_limit, queue_timeout

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.enabled:
            return await self.app(scope, receive, send)
        path, method = scope["path"], scope["method"]
        headers = {k.decode("latin1").lower(): v.decode("latin1") for k, v in scope.get("headers", [])}

        async def error(status, detail, retry=False):
            response = JSONResponse(
                {"detail": detail},
                status_code=status,
                headers={"Retry-After": "5", "Cache-Control": "no-store"} if retry else {},
            )
            await response(scope, receive, send)

        if path.startswith("/api/"):
            allowed = method in {"GET", "HEAD"} and bool(READ.fullmatch(path))
            allowed |= path in WRITE.get(method, set())
            allowed |= method == "DELETE" and (path == "/api/favorites" or path.startswith("/api/favorites/"))
            if path == "/api/files/download" and method in {"GET", "HEAD"}:
                value = parse_qs(scope.get("query_string", b"").decode("utf-8", errors="replace")).get("path", [""])[0]
                allowed = bool(MEDIA.fullmatch(value)) and ".." not in value
            if not allowed:
                return await error(404, "Not found")
        elif (
            method not in {"GET", "HEAD"}
            or path.startswith(("/docs", "/redoc", "/openapi.json", "/prototype", "/pal"))
            or "/." in path
        ):
            return await error(404, "Not found")
        if method not in {"GET", "HEAD", "OPTIONS"}:
            origin = headers.get("origin", "").rstrip("/")
            expected = f"{scope.get('scheme', 'http')}://{headers.get('host', '')}"
            if headers.get("sec-fetch-site") == "cross-site" or origin and origin.lower() != expected.lower():
                return await error(403, "Cross-site requests are not allowed")
        max_body = 210 * 1024 * 1024 if path == "/api/mainline/upload_recommend" else 1024 * 1024
        try:
            if int(headers.get("content-length", "0")) > max_body:
                return await error(413, "Request body is too large")
        except ValueError:
            return await error(400, "Invalid content length")
        kind = (
            "compute"
            if path in {"/api/mainline/recommend", "/api/mainline/upload_recommend"} and method == "POST"
            else ("ai" if path in {"/api/ai/kimi/chat", "/api/ai/kimi/chat/stream"} and method == "POST" else "")
        )
        if not kind:
            return await self.app(scope, receive, send)
        if self.waiting[kind] >= self.queue_limit:
            return await error(503, "The exhibition is busy. Please try again shortly.", True)
        self.waiting[kind] += 1
        acquired = False
        try:
            await asyncio.wait_for(self.limits[kind].acquire(), self.queue_timeout)
            acquired = True
        except asyncio.TimeoutError:
            return await error(503, "The exhibition is busy. Please try again shortly.", True)
        finally:
            self.waiting[kind] -= 1
        try:
            return await self.app(scope, receive, send)
        finally:
            if acquired:
                self.limits[kind].release()
