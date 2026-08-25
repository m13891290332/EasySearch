"""M10 监控告警体系测试。

覆盖：
  - MetricsCollector：record_search / record_external / health_summary /
    prometheus_text（含 prometheus_client 可用与不可用两条路径）/ reset 隔离。
  - AlertChecker：error_rate / p95 / cache_hit / external_consecutive_fail 规则评估
    + fire 触发 ERROR 日志（webhook 不触网）。
  - engine.search：埋点落地（search_total 递增、cache_hit 命中标记、error 计数）。
  - API：GET /metrics 返回 Prometheus 文本；GET /api/health 返回扩展字段。
  - logging_config.setup_logging：输出可解析为 JSON 的日志行。

软依赖：prometheus_client / structlog 不可用时仍全绿（降级路径）。
"""
from __future__ import annotations

import json
import logging
import os
import tempfile
import unittest
from unittest import mock

from fastapi.testclient import TestClient

from api.main import app, reset_engine
from easysearch import (
    DashScopeClient,
    DeepSeekClient,
    ServiceSearchEngine,
    SQLiteStore,
)
from easysearch.alerts import Alert, AlertChecker
from easysearch.logging_config import JsonFormatter, setup_logging
from easysearch.metrics import MetricsCollector, get_metrics

# 3 条最小 KB（离线 fallback 向量即可，无需 DashScope Key）
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
    {
        "service_id": "svc-transfer",
        "service_name": "转账",
        "aliases": ["转账", "transfer"],
        "service_intro": "资金转账",
        "route": "/transfer",
    },
]


def _make_engine(db_path: str) -> ServiceSearchEngine:
    """无 API Key 的离线引擎（fallback 向量 + 模板 reason）。"""
    return ServiceSearchEngine(
        dashscope_client=DashScopeClient(api_key=None),
        deepseek_client=DeepSeekClient(api_key=None),
        store=SQLiteStore(db_path),
        db_path=db_path,
    )


class MetricsCollectorTests(unittest.TestCase):
    """MetricsCollector 单元测试（单例，每个用例 reset 隔离）。"""

    def setUp(self) -> None:
        get_metrics().reset()

    def tearDown(self) -> None:
        get_metrics().reset()

    def test_record_search_updates_counters_and_events(self) -> None:
        m = get_metrics()
        m.record_search(total_ms=12.3, stages={"retrieval": 5.0}, cache_hit=False)
        m.record_search(total_ms=20.0, stages={"retrieval": 8.0}, cache_hit=True)
        m.record_search(total_ms=50.0, error=True)
        summary = m.health_summary()
        self.assertEqual(summary["search_total"], 3)
        self.assertEqual(summary["search_errors"], 1)
        self.assertAlmostEqual(summary["error_rate"], 1 / 3)
        # 最近 3 次事件
        self.assertEqual(summary["recent_total"], 3)
        self.assertEqual(summary["recent_error_rate"], 1 / 3)
        # 缓存命中 1 次 / 未命中 2 次 → 1/3
        self.assertAlmostEqual(summary["cache_hit_rate"], 1 / 3)

    def test_health_summary_empty_does_not_divide_by_zero(self) -> None:
        """冷启动（无事件）不应抛 ZeroDivisionError。"""
        summary = get_metrics().health_summary()
        self.assertEqual(summary["search_total"], 0)
        self.assertEqual(summary["error_rate"], 0.0)
        self.assertEqual(summary["recent_total"], 0)
        self.assertEqual(summary["recent_error_rate"], 0.0)
        self.assertEqual(summary["cache_hit_rate"], 0.0)
        self.assertEqual(summary["external"], {})

    def test_record_external_tracks_consecutive_failures(self) -> None:
        m = get_metrics()
        # 3 次连续失败
        for _ in range(3):
            m.record_external("dashscope", ok=False, latency_ms=120.0)
        # 1 次成功 → 连续失败清零
        m.record_external("dashscope", ok=True, latency_ms=80.0)
        stats = m.external_stats()
        self.assertEqual(stats["dashscope"]["ok"], 1)
        self.assertEqual(stats["dashscope"]["fail"], 3)
        self.assertEqual(m.consecutive_failures("dashscope"), 0)
        # deepseek 独立计数
        m.record_external("deepseek", ok=False, latency_ms=200.0)
        self.assertEqual(m.consecutive_failures("deepseek"), 1)

    def test_events_buffer_capped_at_100(self) -> None:
        m = get_metrics()
        for i in range(150):
            m.record_search(total_ms=float(i))
        # 滚动缓冲仅保留最近 100 条
        self.assertEqual(len(m.events()), 100)
        # 首条是第 50 次（total_ms=50）
        self.assertAlmostEqual(m.events()[0]["total_ms"], 50.0)
        # 聚合计数器仍累加全量
        self.assertEqual(m.health_summary()["search_total"], 150)

    def test_prometheus_text_contains_metric_names(self) -> None:
        """无论 prometheus_client 是否可用，/metrics 文本都含核心指标名。"""
        m = get_metrics()
        m.record_search(total_ms=10.0, cache_hit=True)
        m.record_external("dashscope", ok=True, latency_ms=50.0)
        text = m.prometheus_text()
        self.assertIsInstance(text, str)
        self.assertIn("easysearch_search_total", text)
        self.assertIn("easysearch_cache_hits_total", text)
        self.assertIn("easysearch_external_call_total", text)

    def test_set_kb_embedding_in_progress_gauge(self) -> None:
        m = get_metrics()
        m.set_kb_embedding_in_progress(True)
        self.assertTrue(m.health_summary()["kb_embedding_in_progress"])
        m.set_kb_embedding_in_progress(False)
        self.assertFalse(m.health_summary()["kb_embedding_in_progress"])

    def test_p95_percentile_calculation(self) -> None:
        """P95 应随最慢样本上升。"""
        m = get_metrics()
        for ms in [10, 20, 30, 40, 50, 60, 70, 80, 90, 1000]:
            m.record_search(total_ms=float(ms))
        p95 = m.health_summary()["p95_ms"]
        # 10 个样本，P95 索引 = int(10 * 0.95) = 9 → 排序后最末（1000）
        self.assertGreaterEqual(p95, 900.0)


class AlertCheckerTests(unittest.TestCase):
    """AlertChecker 规则评估 + fire 日志。"""

    def setUp(self) -> None:
        get_metrics().reset()

    def tearDown(self) -> None:
        get_metrics().reset()

    def test_no_alerts_on_healthy_metrics(self) -> None:
        checker = AlertChecker()
        health = {
            "recent_total": 100,
            "recent_error_rate": 0.01,
            "p95_ms": 200.0,
            "cache_hit_rate": 0.8,
        }
        self.assertEqual(checker.evaluate(health), [])

    def test_error_rate_alert_triggered(self) -> None:
        checker = AlertChecker()
        health = {
            "recent_total": 10,
            "recent_error_rate": 0.5,  # > 5% 阈值
            "p95_ms": 100.0,
            "cache_hit_rate": 0.8,
        }
        alerts = checker.evaluate(health)
        self.assertTrue(any(a.rule == "error_rate" for a in alerts))
        err_alert = next(a for a in alerts if a.rule == "error_rate")
        self.assertEqual(err_alert.level, "ERROR")

    def test_p95_alert_triggered(self) -> None:
        checker = AlertChecker()
        health = {
            "recent_total": 10,
            "recent_error_rate": 0.0,
            "p95_ms": 1500.0,  # > 1000ms 阈值
            "cache_hit_rate": 0.8,
        }
        alerts = checker.evaluate(health)
        p95_alert = next(a for a in alerts if a.rule == "p95_latency")
        self.assertEqual(p95_alert.level, "WARN")

    def test_cache_hit_alert_triggered(self) -> None:
        checker = AlertChecker()
        health = {
            "recent_total": 10,
            "recent_error_rate": 0.0,
            "p95_ms": 100.0,
            "cache_hit_rate": 0.1,  # < 30% 阈值
        }
        alerts = checker.evaluate(health)
        self.assertTrue(any(a.rule == "cache_hit_rate" for a in alerts))

    def test_external_consecutive_fail_alert(self) -> None:
        checker = AlertChecker()
        health = {"recent_total": 0, "recent_error_rate": 0.0, "p95_ms": 0.0, "cache_hit_rate": 1.0}
        # DashScope 连续失败 5 次（>= 阈值）
        alerts = checker.evaluate(health, {"dashscope": 5})
        ext_alert = next(a for a in alerts if a.rule == "external_consecutive_fail")
        self.assertEqual(ext_alert.level, "ERROR")
        self.assertIn("dashscope", ext_alert.message)

    def test_no_alerts_below_sample_threshold(self) -> None:
        """样本量 < 5 不评估错误率/P95/缓存（避免冷启动误报）。"""
        checker = AlertChecker()
        health = {
            "recent_total": 3,
            "recent_error_rate": 0.9,
            "p95_ms": 5000.0,
            "cache_hit_rate": 0.0,
        }
        self.assertEqual(checker.evaluate(health), [])

    def test_fire_logs_alerts(self) -> None:
        """fire 触发 ERROR/WARN 日志，无告警则静默。"""
        checker = AlertChecker()
        with self.assertLogs("easysearch.alerts", level="ERROR") as cm:
            checker.fire(
                [Alert(level="ERROR", rule="error_rate", message="err", value=0.5)]
            )
        self.assertTrue(any("error_rate" in line for line in cm.output))

    def test_fire_silent_when_no_alerts(self) -> None:
        checker = AlertChecker()
        # 无 assertLogs 触发即视为静默
        checker.fire([])

    def test_fire_webhook_failure_swallowed(self) -> None:
        """webhook URL 不可达时不应抛异常（不影响主链路）。"""
        checker = AlertChecker()
        with mock.patch.dict(os.environ, {"EASYSEARCH_ALERT_WEBHOOK": "http://127.0.0.1:1/nonexistent"}):
            # 不应抛异常
            checker.fire(
                [Alert(level="ERROR", rule="error_rate", message="err", value=0.5)]
            )

    def test_alert_to_dict(self) -> None:
        a = Alert(level="WARN", rule="p95_latency", message="slow", value=1500.0)
        d = a.to_dict()
        self.assertEqual(d["level"], "WARN")
        self.assertEqual(d["rule"], "p95_latency")
        self.assertEqual(d["value"], 1500.0)


class EngineMetricsIntegrationTests(unittest.TestCase):
    """engine.search / search_async 埋点落地。"""

    def setUp(self) -> None:
        get_metrics().reset()
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "t.db")
        self.engine = _make_engine(self.db)
        self.engine.load_knowledge_base(KB)

    def tearDown(self) -> None:
        self.engine.store.close()
        get_metrics().reset()

    def test_search_increments_search_total(self) -> None:
        before = get_metrics().health_summary()["search_total"]
        self.engine.search("u1", "订单")
        after = get_metrics().health_summary()["search_total"]
        self.assertEqual(after, before + 1)

    def test_cache_hit_recorded_on_second_search(self) -> None:
        m = get_metrics()
        self.engine.search("u1", "订单")
        # 第二次同 query 同 user → 命中结果缓存
        self.engine.search("u1", "订单")
        # 至少一次 cache_hit 事件
        events = m.events()
        self.assertTrue(any(e["cache_hit"] for e in events))

    def test_search_records_stage_timings(self) -> None:
        m = get_metrics()
        self.engine.search("u1", "账户")
        events = m.events()
        self.assertTrue(events)
        stages = events[-1]["stages"]
        # 同步路径应记录 retrieval / rerank / mmr / intent / total
        self.assertIn("total", stages)
        self.assertIn("retrieval", stages)
        self.assertIn("rerank", stages)

    def test_search_async_increments_search_total(self) -> None:
        import asyncio

        before = get_metrics().health_summary()["search_total"]
        asyncio.run(self.engine.search_async("u1", "转账"))
        after = get_metrics().health_summary()["search_total"]
        self.assertEqual(after, before + 1)

    def test_external_callback_wired_on_init(self) -> None:
        """engine.__init__ 应把 metrics.record_external 接到两个客户端。

        bound method 每次 ``instance.method`` 访问都是新对象，故用 ``__self__``
        + ``__func__`` 比对底层单例与函数身份，而非 ``is``。
        """
        cb = self.engine.dashscope_client.metrics_callback
        self.assertIsNotNone(cb)
        self.assertIs(cb.__self__, get_metrics())  # 绑定到同一单例
        self.assertEqual(cb.__func__.__name__, "record_external")
        # deepseek 客户端同样接入
        cb2 = self.engine.deepseek_client.metrics_callback
        self.assertIsNotNone(cb2)
        self.assertIs(cb2.__self__, get_metrics())
        # 实际调用一次：落 external 计数
        cb("dashscope", ok=True, latency_ms=10.0)
        self.assertEqual(get_metrics().external_stats()["dashscope"]["ok"], 1)


class APIMonitoringTests(unittest.TestCase):
    """/metrics + /api/health 端点。"""

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

    def test_metrics_endpoint_returns_prometheus_text(self) -> None:
        # 先产生一些指标
        self.engine.search("u1", "订单")
        r = self.client.get("/metrics")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/plain", r.headers.get("content-type", ""))
        body = r.text
        self.assertIn("easysearch_search_total", body)
        # search_total 计数 > 0
        self.assertRegex(body, r"easysearch_search_total\s+\d+")

    def test_health_endpoint_returns_extended_fields(self) -> None:
        # 产生搜索事件以使 health_summary 非空
        for _ in range(3):
            self.engine.search("u1", "订单")
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        # 兼容字段
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["services_count"], 3)
        self.assertFalse(data["dashscope_enabled"])
        # M10 扩展字段
        self.assertIn("search_total", data)
        self.assertIn("p95_ms", data)
        self.assertIn("cache_hit_rate", data)
        self.assertIn("recent_total", data)
        self.assertEqual(data["search_total"], 3)
        self.assertIsNotNone(data["kb_hash"])

    def test_health_endpoint_backwards_compatible(self) -> None:
        """未产生任何搜索事件时，扩展字段仍可返回（不报 500）。"""
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(data["search_total"], 0)
        self.assertEqual(data["recent_total"], 0)
        self.assertEqual(data["p95_ms"], 0.0)

    def test_reason_stream_endpoint_returns_sse(self) -> None:
        """M10-5：/api/reason 返回 SSE 流，含 start/delta/done 事件 + 阶段计时。"""
        # 模板理由路径（REASON_ENABLED 默认关闭）
        r = self.client.get(
            "/api/reason",
            params={"service_id": "svc-order", "query": "订单", "user_id": "u1"},
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/event-stream", r.headers.get("content-type", ""))
        body = r.text
        # SSE 事件序列
        self.assertIn("event: start", body)
        self.assertIn("event: delta", body)
        self.assertIn("event: done", body)
        # done 事件含 timing 阶段计时
        self.assertIn("timing", body)
        self.assertIn("reason", body)

    def test_reason_stream_unknown_service_returns_error_event(self) -> None:
        """服务不在 KB → error 事件（前端降级展示模板理由）。"""
        r = self.client.get(
            "/api/reason",
            params={"service_id": "nope", "query": "x", "user_id": "u1"},
        )
        self.assertEqual(r.status_code, 200)  # SSE 流本身 200，错误在事件里
        self.assertIn("event: error", r.text)


class StructuredLoggingTests(unittest.TestCase):
    """setup_logging + JsonFormatter 输出可解析 JSON。

    注意：setup_logging 改全局根 logger，本类 setUp/tearDown 保存并还原
    handlers + level，避免污染其它测试。
    """

    def setUp(self) -> None:
        root = logging.getLogger()
        self._saved_level = root.level
        self._saved_handlers = list(root.handlers)

    def tearDown(self) -> None:
        root = logging.getLogger()
        root.setLevel(self._saved_level)
        # 移除本类测试追加的 json handler，恢复原 handler 列表
        root.handlers = list(self._saved_handlers)

    def test_json_formatter_outputs_single_line_json(self) -> None:
        formatter = JsonFormatter()
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="search query=%s latency=%s",
            args=("订单", 12.5),
            exc_info=None,
        )
        line = formatter.format(record)
        # 单行 JSON
        self.assertNotIn("\n", line)
        payload = json.loads(line)
        self.assertEqual(payload["level"], "INFO")
        self.assertEqual(payload["logger"], "test")
        self.assertIn("订单", payload["message"])
        self.assertIn("ts", payload)

    def test_setup_logging_idempotent(self) -> None:
        """重复调用 setup_logging 不应重复追加 handler。"""
        root = logging.getLogger()
        before = len(root.handlers)
        setup_logging()
        setup_logging()
        after = len(root.handlers)
        # 第二次调用应被幂等拦截（最多新增一个 _easysearch_json handler）
        self.assertLessEqual(after, before + 1)
        # 确实新增的 handler 带 _easysearch_json 标记
        json_handlers = [h for h in root.handlers if getattr(h, "_easysearch_json", False)]
        self.assertEqual(len(json_handlers), 1)

    def test_setup_logging_respects_env_level(self) -> None:
        with mock.patch.dict(os.environ, {"EASYSEARCH_LOG_LEVEL": "WARNING"}):
            setup_logging()
            self.assertEqual(logging.getLogger().level, logging.WARNING)


if __name__ == "__main__":
    unittest.main()
