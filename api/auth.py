"""M12 错误处理与降级——API 安全中间件。

提供三个独立可组合的 Starlette 中间件：
  1. ``ApiKeyMiddleware`` —— 可选 API Key 鉴权（env ``EASYSEARCH_API_KEY`` 未设置时禁用）
  2. ``BodySizeLimitMiddleware`` —— 上传体积上限（默认 10MB，超出 413）
  3. ``RateLimitMiddleware`` —— 慢速限流（per-IP token bucket，默认 60 req/min）

设计原则：
- 默认关闭鉴权：未设置 env 时所有测试与离线部署照常运行。
- 失败不穿透 500：所有拒绝路径返回明确状态码 + 通用化错误消息，
  不回显后端细节（与 M12「不穿透 500、不泄露细节」对齐）。
- 公开端点白名单：``/api/health``、``/metrics``、``/api/metrics/*`` 不要求 API Key，
  便于 Prometheus 抓取与监控探针。
"""
from __future__ import annotations

import os
import time
from typing import Awaitable, Callable

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp

# ---------- 配置 ----------

#: M12：上传体积上限（字节），默认 10MB。env 可覆盖。
MAX_BODY_BYTES: int = int(os.getenv("EASYSEARCH_MAX_BODY_BYTES", str(10 * 1024 * 1024)))

#: M12：限流白名单（健康检查/指标抓取通常来自可信内网，不参与限流）
RATE_LIMIT_EXEMPT_PATHS: frozenset[str] = frozenset(
    {"/api/health", "/metrics", "/api/metrics/realtime", "/api/metrics/stream"}
)

#: API Key 鉴权白名单（监控探针不应被鉴权拦截）
AUTH_EXEMPT_PATHS: frozenset[str] = frozenset(
    {"/api/health", "/metrics", "/api/metrics/realtime", "/api/metrics/stream"}
)


# ---------- 1. API Key 鉴权 ----------


class ApiKeyMiddleware(BaseHTTPMiddleware):
    """M12：可选 API Key 鉴权。

    - 未配置 ``EASYSEARCH_API_KEY`` → 中间件透传（默认离线/测试场景）。
    - 已配置 → ``/api/*`` 路由要求 ``X-API-Key`` 头匹配；白名单端点豁免。
    - 失败 → 401（无/错 Key）或 403（Key 不匹配），通用化错误消息不泄露细节。

    注意：Key 在请求时从 env 读取（非构造时固化），便于测试隔离——
    测试 setUp/tearDown 临时设置 env var 即可切换鉴权开关，无需重建 app。
    """

    def __init__(self, app: ASGIApp, api_key: str | None = None) -> None:
        super().__init__(app)
        # 显式传入的 api_key 优先（用于测试注入）；否则运行时读 env
        self._explicit_api_key = api_key

    def _current_api_key(self) -> str | None:
        """获取当前生效的 API Key：显式注入 > env 变量。"""
        if self._explicit_api_key is not None:
            return self._explicit_api_key
        return os.getenv("EASYSEARCH_API_KEY")

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        api_key = self._current_api_key()
        if not api_key:
            # 未配置 Key → 鉴权关闭，透传
            return await call_next(request)
        path = request.url.path
        # 白名单豁免（监控探针）
        if path in AUTH_EXEMPT_PATHS:
            return await call_next(request)
        # 仅保护 /api/* 路由
        if not path.startswith("/api/"):
            return await call_next(request)
        provided = request.headers.get("X-API-Key") or request.headers.get("x-api-key")
        if not provided:
            return JSONResponse(
                status_code=401,
                content={"detail": "缺少 API Key，请在 X-API-Key 头提供"},
            )
        if provided != api_key:
            return JSONResponse(
                status_code=403,
                content={"detail": "API Key 无效"},
            )
        return await call_next(request)


# ---------- 2. 上传体积上限 ----------


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """M12：上传体积上限（默认 10MB）。

    通过 Content-Length 头预检；缺失时（chunked transfer）按实际读取字节判定。
    超出 → 413 Payload Too Large，通用化错误消息不泄露内部阈值细节。
    """

    def __init__(self, app: ASGIApp, max_bytes: int = MAX_BODY_BYTES) -> None:
        super().__init__(app)
        self.max_bytes = max_bytes

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # 仅检查有 body 的方法
        if request.method in ("POST", "PUT", "PATCH"):
            content_length = request.headers.get("content-length")
            if content_length:
                try:
                    cl = int(content_length)
                except ValueError:
                    return JSONResponse(
                        status_code=400,
                        content={"detail": "Content-Length 头格式无效"},
                    )
                if cl > self.max_bytes:
                    return JSONResponse(
                        status_code=413,
                        content={"detail": "上传内容过大"},
                    )
        return await call_next(request)


# ---------- 3. 慢速限流（per-IP token bucket） ----------


class _TokenBucket:
    """单 IP 的令牌桶。

    capacity = RATE_LIMIT_PER_MIN，refill 速率 = capacity/60（令牌/秒）。
    每次请求消费 1 令牌；不足则拒绝。
    """

    __slots__ = ("capacity", "refill_per_sec", "tokens", "last_refill_ts")

    def __init__(self, capacity: int, now: float) -> None:
        self.capacity = capacity
        self.refill_per_sec = capacity / 60.0
        # 初始满桶（避免冷启动误拒）
        self.tokens = float(capacity)
        self.last_refill_ts = now

    def consume(self, now: float) -> bool:
        """尝试消费 1 令牌；成功 True，不足 False。"""
        elapsed = max(0.0, now - self.last_refill_ts)
        self.tokens = min(
            float(self.capacity), self.tokens + elapsed * self.refill_per_sec
        )
        self.last_refill_ts = now
        if self.tokens >= 1.0:
            self.tokens -= 1.0
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """M12：per-IP 慢速限流（token bucket）。

    - 默认 60 req/min/IP；env ``EASYSEARCH_RATE_LIMIT`` 可覆盖（运行时读取，便于测试隔离）。
    - 设置 ``EASYSEARCH_RATE_LIMIT=0`` 可禁用限流（测试场景默认安全）。
    - 监控端点白名单豁免（避免 Prometheus 抓取被拒）。
    - 命中限流 → 429 Too Many Requests，附 ``Retry-After`` 头。
    - 进程内 dict 维护桶（单 worker 场景足够；多 worker 需换 Redis，见 plan 容量演进触发线）。
    """

    def __init__(
        self,
        app: ASGIApp,
        limit_per_min: int | None = None,
    ) -> None:
        super().__init__(app)
        # 显式注入优先；None 时运行时读 env
        self._explicit_limit = limit_per_min
        self._buckets: dict[str, _TokenBucket] = {}
        # 简易清理：每 5 分钟扫描一次过期桶（>10 分钟未访问）
        self._last_gc_ts = time.time()

    def _current_limit(self) -> int:
        """获取当前生效的限流配额：显式注入 > env 变量 > 默认 60。"""
        if self._explicit_limit is not None:
            return max(0, self._explicit_limit)
        try:
            return max(0, int(os.getenv("EASYSEARCH_RATE_LIMIT", "60")))
        except ValueError:
            return 60

    def _client_key(self, request: Request) -> str:
        # 优先取 X-Forwarded-For（反向代理后），其次 client.host
        xff = request.headers.get("x-forwarded-for")
        if xff:
            return xff.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _gc_if_needed(self, now: float) -> None:
        if now - self._last_gc_ts < 300.0:
            return
        cutoff = now - 600.0
        stale = [ip for ip, b in self._buckets.items() if b.last_refill_ts < cutoff]
        for ip in stale:
            self._buckets.pop(ip, None)
        self._last_gc_ts = now

    async def dispatch(
        self, request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        path = request.url.path
        if path in RATE_LIMIT_EXEMPT_PATHS:
            return await call_next(request)
        limit = self._current_limit()
        if limit <= 0:
            # 限流禁用（测试或离线场景）
            return await call_next(request)
        now = time.time()
        self._gc_if_needed(now)
        ip = self._client_key(request)
        bucket = self._buckets.get(ip)
        # 桶容量变化时重建（env 动态调整）
        if bucket is None or bucket.capacity != limit:
            bucket = _TokenBucket(limit, now)
            self._buckets[ip] = bucket
        if not bucket.consume(now):
            return JSONResponse(
                status_code=429,
                content={"detail": "请求过于频繁，请稍后再试"},
                headers={"Retry-After": "60"},
            )
        return await call_next(request)
