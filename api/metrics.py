"""M14 实时性能监控——实时大盘端点。

复用 M10 ``MetricsCollector`` 指标底座，新增两个端点：
  - ``GET /api/metrics/realtime?window=60``：返回最近 60s 滚动窗口的各阶段
    P50/P95/P99、QPS、错误率、缓存命中率、降级计数、DB 池占用、embedding 是否进行中。
  - ``GET /api/metrics/stream``：Server-Sent Events，每秒推送一次 realtime_summary，
    供前端大盘页 ``frontend/dashboard.html`` 1s 刷新。

设计要点：
  - 不直接持有 engine 引用：realtime_summary 来自进程内 ``MetricsCollector`` 单例，
    engine.search 已在 M10 埋点；SSE 流仅读单例，避免与 engine 生命周期耦合。
  - SSE 流默认无限推送直到客户端断开；测试可传 ``max_events`` 限制推送次数。
  - 单进程 async worker 下，1s 推送 + 轻量锁足够，无需额外线程。
"""
from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse

from easysearch.metrics import get_metrics


def _sse(payload: dict[str, Any]) -> str:
    """格式化一条无事件名的 SSE data 行（``data: <json>\\n\\n``）。"""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def register_metrics_routes(app: FastAPI) -> None:
    """在 app 上注册 M14 实时大盘路由。"""

    @app.get("/api/metrics/realtime", tags=["meta"])
    def realtime(
        window: int = Query(60, ge=1, le=600, description="滚动窗口秒数（1-600）"),
    ) -> dict[str, Any]:
        """M14：实时大盘快照（最近 ``window`` 秒聚合）。

        返回 QPS / 错误率 / 缓存命中率 / 降级计数 / 各阶段 P50/P95/P99 +
        外部调用健康度 + DB 池占用 + embedding 是否进行中。窗口内无事件返回全 0。
        """
        return get_metrics().realtime_summary(window_seconds=window)

    @app.get("/api/metrics/stream", tags=["meta"])
    async def realtime_stream(
        window: int = Query(60, ge=1, le=600, description="滚动窗口秒数"),
        interval: float = Query(1.0, ge=0.1, le=10, description="推送间隔秒数"),
        max_events: int | None = Query(
            None, ge=1, le=3600, description="最大推送次数（测试用；默认无限）"
        ),
    ) -> StreamingResponse:
        """M14：SSE 实时大盘流，每 ``interval`` 秒推送一次 realtime_summary。

        前端 ``dashboard.html`` 通过 ``EventSource`` 订阅，1s 刷新各阶段延迟 /
        QPS 折线 / 降级高亮 / DB 池占用。客户端断开即停止推送。
        """

        async def event_gen():
            count = 0
            while True:
                payload = get_metrics().realtime_summary(window_seconds=window)
                # 标注推送时间，便于前端按秒对齐
                import time as _time

                payload["pushed_at"] = _time.time()
                yield _sse(payload)
                count += 1
                if max_events is not None and count >= max_events:
                    return
                await asyncio.sleep(interval)

        return StreamingResponse(
            event_gen(), media_type="text/event-stream; charset=utf-8"
        )
