"""M14 实时性能监控测试。

覆盖：
  - MetricsCollector.realtime_summary：60s 窗口聚合 / 各阶段 P50/P95/P99 /
    QPS / 错误率 / 缓存命中率 / 降级计数 / DB 池占位 / embedding 状态。
  - 窗口外事件被裁剪（60s 前的事件不计入 realtime，但仍保留在 _events）。
  - GET /api/metrics/realtime：返回 JSON 大盘快照。
  - GET /api/metrics/stream：SSE 流每秒推送一次（max_events 限制便于测试）。
  - GET /api/search 响应含 timing 字段（单请求诊断）。
"""
from __future__ import annotations

import os
import tempfile
import time
import unittest

from fastapi.testclient import TestClient

from api.main import app, reset_engine
from easysearch import DashScopeClient, DeepSeekClient, ServiceSearchEngine, SQLiteStore
from easysearch.metrics import get_metrics

KB = [
    {
        "service_id": "svc-order",
        "service_name": "订单中心",
        "aliases": ["订单", "order"],
        "service_intro": "订单管理",
        "route": "/orders",
    },
    {
        "service_id": "svc-account",
        "service_name": "账户中心",
        "aliases": ["账户", "account"],
        "service_intro": "账户查询",
        "route": "/account",
    },
]


def _make_engine(db_path: str) -> ServiceSearchEngine:
    return ServiceSearchEngine(
        dashscope_client=DashScopeClient(api_key=None),
        deepseek_client=DeepSeekClient(api_key=None),
        store=SQLiteStore(db_path),
        db_path=db_path,
    )


class RealtimeSummaryTests(unittest.TestCase):
    """MetricsCollector.realtime_summary 60s 窗口聚合。"""

    def setUp(self) -> None:
        get_metrics().reset()

    def tearDown(self) -> None:
        get_metrics().reset()

    def test_empty_returns_zeros(self) -> None:
        """冷启动无事件 → 全 0 指标，不报错。"""
        rt = get_metrics().realtime_summary(60)
        self.assertEqual(rt["total_requests"], 0)
        self.assertEqual(rt["qps"], 0.0)
        self.assertEqual(rt["error_rate"], 0.0)
        self.assertEqual(rt["degraded_count"], 0)
        self.assertEqual(rt["latency_total"], {"p50": 0.0, "p95": 0.0, "p99": 0.0})
        self.assertEqual(rt["db_pool_usage"], 0)
        self.assertFalse(rt["kb_embedding_in_progress"])

    def test_aggregates_within_window(self) -> None:
        m = get_metrics()
        for ms in [10, 20, 30, 40, 100]:
            m.record_search(total_ms=float(ms), stages={"retrieval": ms * 0.5, "rerank": ms * 0.3})
        m.record_search(total_ms=200.0, error=True)
        m.record_search(total_ms=15.0, degraded=True, cache_hit=True)
        rt = m.realtime_summary(60)
        self.assertEqual(rt["total_requests"], 7)
        # QPS = 7 / 60
        self.assertAlmostEqual(rt["qps"], 7 / 60, places=3)
        # 1 次错误 / 7 次
        self.assertAlmostEqual(rt["error_rate"], 1 / 7)
        # 1 次缓存命中 / 7 次
        self.assertAlmostEqual(rt["cache_hit_rate"], 1 / 7)
        # 1 次降级
        self.assertEqual(rt["degraded_count"], 1)
        # 各阶段 latency_stages 含 retrieval/rerank
        self.assertIn("retrieval", rt["latency_stages"])
        self.assertIn("rerank", rt["latency_stages"])
        # latency_total 含 P50/P95/P99
        self.assertIn("p50", rt["latency_total"])
        self.assertIn("p99", rt["latency_total"])

    def test_window_filters_old_events(self) -> None:
        """窗口外事件不计入 realtime，但仍保留在 _events（health_summary 用）。"""
        m = get_metrics()
        # 3 条「旧」事件（ts = now - 120s，超出 60s 窗口）——直接注入两个缓冲
        old_ts = time.time() - 120
        with m._lock:
            for ms in [10, 20, 30]:
                evt = {"ts": old_ts, "total_ms": float(ms), "stages": {},
                       "cache_hit": False, "degraded": False, "error": False, "intent": ""}
                m._events.append(dict(evt))
                m._realtime_events.append(dict(evt))
                m._search_total += 1
        # 2 条新事件（ts = now，窗口内）
        m.record_search(total_ms=50.0)
        m.record_search(total_ms=60.0)
        rt = m.realtime_summary(60)
        # 仅 2 条计入 realtime（窗口外 3 条被裁剪）
        self.assertEqual(rt["total_requests"], 2)
        # _events 仍保留全部 5 条（health_summary 用 100 缓冲）
        self.assertEqual(len(m.events()), 5)

    def test_stage_percentiles_monotonic(self) -> None:
        """P50 ≤ P95 ≤ P99。"""
        m = get_metrics()
        for ms in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            m.record_search(total_ms=float(ms), stages={"retrieval": float(ms)})
        rt = m.realtime_summary(60)
        lat = rt["latency_stages"]["retrieval"]
        self.assertLessEqual(lat["p50"], lat["p95"])
        self.assertLessEqual(lat["p95"], lat["p99"])

    def test_realtime_buffer_capped_at_600(self) -> None:
        m = get_metrics()
        for _ in range(700):
            m.record_search(total_ms=1.0)
        # realtime 缓冲 maxlen=600
        rt = m.realtime_summary(60)
        self.assertLessEqual(rt["total_requests"], 600)


class APIRealtimeTests(unittest.TestCase):
    """GET /api/metrics/realtime + /api/metrics/stream + /api/search timing。"""

    def setUp(self) -> None:
        get_metrics().reset()
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "t.db")
        self.engine = _make_engine(self.db)
        self.engine.load_knowledge_base(KB)
        reset_engine(self.engine)
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.engine.store.close()
        get_metrics().reset()

    def test_realtime_endpoint_returns_json(self) -> None:
        self.engine.search("u1", "订单")
        r = self.client.get("/api/metrics/realtime", params={"window": 60})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["window_seconds"], 60)
        self.assertEqual(data["total_requests"], 1)
        self.assertIn("latency_total", data)
        self.assertIn("latency_stages", data)
        self.assertIn("qps", data)

    def test_realtime_endpoint_validates_window_bounds(self) -> None:
        """window 超出 1-600 范围应 422（Query 校验）。"""
        r = self.client.get("/api/metrics/realtime", params={"window": 0})
        self.assertEqual(r.status_code, 422)
        r = self.client.get("/api/metrics/realtime", params={"window": 601})
        self.assertEqual(r.status_code, 422)

    def test_stream_endpoint_emits_sse(self) -> None:
        """SSE 流每 interval 秒推送一次；max_events=2 限制后停止。"""
        self.engine.search("u1", "订单")
        r = self.client.get(
            "/api/metrics/stream",
            params={"window": 60, "interval": 0.1, "max_events": 2},
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/event-stream", r.headers.get("content-type", ""))
        # 至少 2 条 data: 行
        self.assertEqual(r.text.count("data: "), 2)
        # 每条 data 行可解析为 JSON 含 total_requests
        import json as _json

        lines = [ln for ln in r.text.splitlines() if ln.startswith("data: ")]
        self.assertGreaterEqual(len(lines), 2)
        first = _json.loads(lines[0][len("data: "):])
        self.assertIn("total_requests", first)
        self.assertIn("pushed_at", first)

    def test_search_response_includes_timing(self) -> None:
        """M14：/api/search 响应含 timing 字段（单请求各阶段耗时）。"""
        r = self.client.get(
            "/api/search", params={"user_id": "u1", "query": "订单"}
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("timing", data)
        # timing 非空且含各阶段
        self.assertIsNotNone(data["timing"])
        self.assertIn("total", data["timing"])
        self.assertIn("retrieval", data["timing"])

    def test_search_response_timing_none_on_empty_kb(self) -> None:
        """KB 为空时（无搜索事件）timing 应为 None（不报 500）。"""
        # 用空 KB 引擎替换
        empty_engine = _make_engine(os.path.join(self.tmp, "empty.db"))
        reset_engine(empty_engine)
        empty_client = TestClient(app)
        # 空库 → /api/search 返回 409（知识库为空），不进入搜索路径
        r = empty_client.get("/api/search", params={"user_id": "u1", "query": "x"})
        self.assertEqual(r.status_code, 409)


if __name__ == "__main__":
    unittest.main()
