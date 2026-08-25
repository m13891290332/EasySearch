from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from typing import Any, Callable
from urllib import error, request

from .config import ASYNC_HTTP_POOL_SIZE, DASHSCOPE_API_KEY

logger = logging.getLogger(__name__)

# 模块级 httpx.AsyncClient 懒单例（M2：异步并发外部调用）
# 绑定到创建时的事件循环：若运行循环变化（asyncio.run 临时循环 / 多 loop）则重建，
# 避免跨 loop 使用已关闭 client。在线上单 FastAPI 循环下稳定复用连接池。
_async_client: Any = None
_async_client_loop: Any = None


def _get_async_client(timeout: int, pool_size: int = ASYNC_HTTP_POOL_SIZE) -> Any:
    """获取当前事件循环绑定的 httpx.AsyncClient 单例。"""
    global _async_client, _async_client_loop
    try:
        import httpx
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("httpx is required for async calls") from exc
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None  # 无运行循环（同步上下文调用异步方法时不应发生）
    if _async_client is None or _async_client_loop is not loop:
        # 旧 client 绑定了别的 loop，关闭重建
        if _async_client is not None:
            try:
                if _async_client_loop is not None:
                    _async_client_loop.create_task(_async_client.aclose())
            except Exception:
                pass
        _async_client = httpx.AsyncClient(
            timeout=timeout,
            limits=httpx.Limits(
                max_connections=pool_size,
                max_keepalive_connections=pool_size,
            ),
        )
        _async_client_loop = loop
    return _async_client


async def aclose_async_client() -> None:
    """应用关闭时关闭异步客户端（lifespan 调用）。"""
    global _async_client, _async_client_loop
    if _async_client is not None:
        await _async_client.aclose()
    _async_client = None
    _async_client_loop = None


# ---------- M12：远程失败重试分类 ----------
class RetryableHTTPError(RuntimeError):
    """5xx 服务端错误 / 超时 / 网络异常——可指数退避重试。"""


class NonRetryableHTTPError(RuntimeError):
    """4xx 客户端错误——不重试，原样抛出便于上层降级。"""


class DashScopeClient:
    """DashScope HTTP 客户端。

    通过 urllib 直接调用，不依赖官方 SDK，便于在没有 dashscope 包的环境运行。
    支持注入自定义 requester 用于单元测试（不触网）。

    M2：新增 post_json_async（httpx.AsyncClient 单例），保留同步 post_json 兼容测试。
    M12：post_json/post_json_async 加入指数退避重试（5xx/超时 2 次，4xx 不重试），
         降级打 WARN + 经 metrics_callback 上报 M10 计数。

    API Key 读取优先级：
        1. 构造参数 api_key（最高，用于测试/临时覆盖）
        2. 环境变量 DASHSCOPE_API_KEY（M1：源码零密钥，从 .env / 环境变量读取）
    """

    def __init__(
        self,
        api_key: str | None = None,
        requester: Callable[[str, bytes, dict[str, str]], dict[str, Any]] | None = None,
        timeout: int = 60,
        max_retries: int = 2,
        base_backoff: float = 0.5,
    ) -> None:
        self.api_key = api_key or DASHSCOPE_API_KEY or os.getenv("DASHSCOPE_API_KEY")
        self.requester = requester or self._default_requester
        self.timeout = timeout
        # M12：指数退避重试（base * 2^attempt），默认 0.5s/1.0s，最多 2 次
        self.max_retries = max_retries
        self.base_backoff = base_backoff
        # M10：外部调用埋点标识（dashscope/deepseek），由子类覆盖
        self.service_tag = "dashscope"
        # M10：可选 metrics 回调 (service_tag, ok, latency_ms) -> None；未设置则不埋点
        self.metrics_callback: "Callable[[str, bool, float], None] | None" = None

    @property
    def enabled(self) -> bool:
        """API Key 是否已配置（决定是否可发起远程调用）。

        用 property 而非 ``__init__`` 静态属性，使运行时动态修改 ``api_key``
        （如测试注入 placeholder Key 触发 LLM 路径）能即时反映。
        被 post_json/post_json_async、health 端点、embedding/guide/reranker
        共同依赖，是「离线 fallback vs 远程调用」的总开关。
        """
        return bool(self.api_key)

    def _record_external(self, ok: bool, latency_ms: float) -> None:
        """M10：向 metrics 回调上报一次外部调用结果（未设置回调时静默跳过）。

        M12：仅在重试循环结束后上报最终结果（成功或耗尽重试后的最终失败），
        避免每次失败尝试都计入 _external 计数导致 fail_rate 失真。
        """
        cb = self.metrics_callback
        if cb is None:
            return
        try:
            cb(self.service_tag, ok, latency_ms)
        except Exception:  # noqa: BLE001 - 埋点失败不影响主链路
            pass

    @staticmethod
    def _is_retryable(exc: BaseException) -> bool:
        """M12：判断异常是否可重试。

        - RetryableHTTPError（5xx / 超时 / 网络异常）→ 可重试
        - NonRetryableHTTPError（4xx）→ 不重试
        - 普通 RuntimeError（如测试 mock 抛出 / "API key is not configured"）→ 不重试
          保持向后兼容，避免对未知错误做无谓重试
        """
        if isinstance(exc, RetryableHTTPError):
            return True
        if isinstance(exc, NonRetryableHTTPError):
            return False
        return False

    def _sleep_backoff(self, attempt: int) -> None:
        """同步指数退避：attempt 0 → base，attempt 1 → base*2，..."""
        delay = self.base_backoff * (2 ** attempt)
        time.sleep(delay)

    async def _sleep_backoff_async(self, attempt: int) -> None:
        delay = self.base_backoff * (2 ** attempt)
        await asyncio.sleep(delay)

    def post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("DashScope API key is not configured")
        headers = {
            "Authorization": "Bearer " + (self.api_key or ""),
            "Content-Type": "application/json",
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        ok = False
        t0 = time.time()
        last_exc: BaseException | None = None
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    result = self.requester(url, body, headers)
                    ok = True
                    return result
                except NonRetryableHTTPError:
                    # 4xx 客户端错误：不重试，立即失败
                    raise
                except RetryableHTTPError as exc:
                    last_exc = exc
                    if attempt < self.max_retries:
                        logger.warning(
                            "%s request failed (attempt %d/%d): %s; retrying in %.2fs",
                            self.service_tag,
                            attempt + 1,
                            self.max_retries + 1,
                            exc,
                            self.base_backoff * (2 ** attempt),
                        )
                        self._sleep_backoff(attempt)
                        continue
                    # 重试耗尽：抛出最终失败
                    raise
                except RuntimeError:
                    # 未知 RuntimeError（非 M12 分类）——保持旧行为直接抛出，不重试
                    raise
            # 理论不可达（循环必在 return/raise 退出）
            if last_exc is not None:
                raise last_exc
            raise RuntimeError(f"{self.service_tag} request loop exited unexpectedly")
        finally:
            # M10/M12：仅在重试循环结束后上报最终结果（避免重复上报）
            self._record_external(ok, (time.time() - t0) * 1000.0)

    async def _async_post_safe(
        self, client: Any, url: str, content: bytes, headers: dict[str, str]
    ) -> Any:
        """M12：包装 httpx.post，将网络异常转为 RetryableHTTPError 走重试链路。

        httpx 的 ConnectError/TimeoutException/ReadError 等不是 RuntimeError 子类，
        直接抛出会导致上层 except RuntimeError 漏接 → 500。这里统一转为
        RetryableHTTPError（RuntimeError 子类），由 retry 循环 + 外层归一处理。
        """
        try:
            return await client.post(url, content=content, headers=headers)
        except Exception as exc:
            raise RetryableHTTPError(
                f"{self.service_tag} async network error: {exc}"
            ) from exc

    async def post_json_async(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        """M2：异步 POST JSON，底层 httpx.AsyncClient 单例连接池。

        M12：5xx/超时/网络异常指数退避重试 2 次；4xx 不重试；降级打 WARN。
        失败统一抛 RuntimeError（与同步版一致），上层 except RuntimeError 即可降级。
        """
        if not self.enabled:
            raise RuntimeError("DashScope API key is not configured")
        headers = {
            "Authorization": "Bearer " + (self.api_key or ""),
            "Content-Type": "application/json",
        }
        client = _get_async_client(self.timeout)
        ok = False
        t0 = time.time()
        last_exc: BaseException | None = None
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    # M12：_async_post_safe 把 httpx 网络异常转 RetryableHTTPError
                    resp = await self._async_post_safe(
                        client,
                        url,
                        json.dumps(payload, ensure_ascii=False).encode("utf-8"),
                        headers,
                    )
                    # M12：按状态码分类——5xx 可重试，4xx 不重试
                    if resp.status_code >= 500:
                        raise RetryableHTTPError(
                            f"{self.service_tag} server error: {resp.status_code} {resp.text[:200]}"
                        )
                    if resp.status_code >= 400:
                        raise NonRetryableHTTPError(
                            f"{self.service_tag} client error: {resp.status_code} {resp.text[:200]}"
                        )
                    content = resp.text
                    try:
                        parsed = json.loads(content)
                    except json.JSONDecodeError as exc:
                        # JSON 解析失败视为可重试（瞬时脏响应）
                        raise RetryableHTTPError(
                            f"{self.service_tag} response is not valid JSON"
                        ) from exc
                    ok = True
                    return parsed
                except NonRetryableHTTPError:
                    # 4xx 客户端错误：不重试，立即失败
                    raise
                except RetryableHTTPError as exc:
                    last_exc = exc
                    if attempt < self.max_retries:
                        logger.warning(
                            "%s async request failed (attempt %d/%d): %s; retrying",
                            self.service_tag,
                            attempt + 1,
                            self.max_retries + 1,
                            exc,
                        )
                        await self._sleep_backoff_async(attempt)
                        continue
                    raise
                except RuntimeError:
                    # 未知 RuntimeError（非 M12 分类）——保持旧行为直接抛出，不重试
                    raise
            if last_exc is not None:
                raise last_exc
            raise RuntimeError(f"{self.service_tag} async request loop exited unexpectedly")
        except (RetryableHTTPError, NonRetryableHTTPError) as exc:
            # 归一为 RuntimeError 便于上层 except 降级（与同步版语义一致）
            raise RuntimeError(str(exc)) from exc
        finally:
            # M10/M12：仅在重试循环结束后上报最终结果
            self._record_external(ok, (time.time() - t0) * 1000.0)

    def _default_requester(
        self, url: str, body: bytes, headers: dict[str, str]
    ) -> dict[str, Any]:
        req = request.Request(url=url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                content = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            # M12：5xx 可重试，4xx 不重试
            if exc.code >= 500:
                raise RetryableHTTPError(
                    f"{self.service_tag} server error: {exc.code} {message}"
                ) from exc
            raise NonRetryableHTTPError(
                f"{self.service_tag} client error: {exc.code} {message}"
            ) from exc
        except (error.URLError, TimeoutError, OSError) as exc:
            # TimeoutError / OSError（含 socket 超时）统一为可重试
            reason = getattr(exc, "reason", exc)
            raise RetryableHTTPError(
                f"{self.service_tag} network error: {reason}"
            ) from exc

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise RetryableHTTPError(
                f"{self.service_tag} response is not valid JSON"
            ) from exc
