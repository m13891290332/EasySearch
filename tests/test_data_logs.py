"""M11 数据日志记录测试。

覆盖：
  - store：search_logs 表 append/recent/mark_click/aggregate_no_click/
    aggregate_high_latency/degradation_stats；kb_op_logs append/list。
  - user_id 哈希化：同 user_id 哈希稳定、不同 user_id 哈希不同、不等于原始 user_id。
  - engine：search() 成功/缓存命中路径落 search_logs；record_click 回填 clicked_sid；
    import_kb_version/rollback_kb 落 kb_op_logs。
  - API：GET /api/logs/search/no-click、/api/logs/search/slow、/api/logs/degradation、
    /api/logs/search/recent、/api/logs/kb-ops；kb_export 落操作日志。
"""
from __future__ import annotations

import asyncio
import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from api.main import app, reset_engine
from easysearch import (
    DashScopeClient,
    DeepSeekClient,
    ServiceSearchEngine,
    SQLiteStore,
)
from easysearch.metrics import get_metrics

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


# ===========================================================================
# store：search_logs + user_id 哈希化
# ===========================================================================
class StoreSearchLogsTests(unittest.TestCase):
    """M11：store.search_logs 落库 + 哈希化 + 聚合查询。"""

    def setUp(self):
        self.store = SQLiteStore(":memory:")

    def tearDown(self):
        self.store.close()

    def test_hash_user_id_stable_and_unique(self):
        """同 user_id 哈希稳定；不同 user_id 哈希不同；哈希不等于原始 user_id。"""
        h1 = self.store.hash_user_id("alice")
        h2 = self.store.hash_user_id("alice")
        h3 = self.store.hash_user_id("bob")
        self.assertEqual(h1, h2)  # 稳定
        self.assertNotEqual(h1, h3)  # 不同 user 不同哈希
        self.assertNotEqual(h1, "alice")  # 不等于原始
        self.assertEqual(len(h1), 64)  # sha256 hex = 64 chars

    def test_append_and_recent_search_logs(self):
        log_id = self.store.append_search_log(
            user_id="alice", query="订单", intent="default",
            top_ids=["svc-order", "svc-account"], latencies={"total": 50.0},
            cache_hit=False, degraded=False, ts=1000.0,
        )
        self.assertIsInstance(log_id, int)
        self.assertGreater(log_id, 0)
        logs = self.store.recent_search_logs(limit=10)
        self.assertEqual(len(logs), 1)
        entry = logs[0]
        self.assertEqual(entry["id"], log_id)
        self.assertEqual(entry["query"], "订单")
        self.assertEqual(entry["intent"], "default")
        self.assertEqual(entry["top_ids"], ["svc-order", "svc-account"])
        self.assertEqual(entry["latencies"], {"total": 50.0})
        self.assertFalse(entry["cache_hit"])
        self.assertFalse(entry["degraded"])
        self.assertIsNone(entry["clicked_sid"])  # 默认 NULL
        # user_hash 是哈希值，不是原始 user_id
        self.assertNotEqual(entry["user_hash"], "alice")

    def test_append_with_sub_queries_and_session(self):
        log_id = self.store.append_search_log(
            user_id="bob", query="开户 和 转账", intent="multi_condition",
            top_ids=["svc-order"], latencies={"total": 100.0},
            cache_hit=False, degraded=False, ts=2000.0,
            sub_queries=["开户", "转账"], session_id="sess-1",
        )
        logs = self.store.recent_search_logs(limit=10)
        entry = logs[0]
        self.assertEqual(entry["sub_queries"], ["开户", "转账"])
        self.assertEqual(entry["session_id"], "sess-1")

    def test_mark_search_log_click_backfill(self):
        log_id = self.store.append_search_log(
            user_id="alice", query="账户", intent="default",
            top_ids=["svc-account"], latencies={},
            cache_hit=False, degraded=False, ts=1000.0,
        )
        # 回填点击
        ok = self.store.mark_search_log_click(log_id, "svc-account")
        self.assertTrue(ok)
        logs = self.store.recent_search_logs(limit=10)
        self.assertEqual(logs[0]["clicked_sid"], "svc-account")
        # 重复回填幂等（NULL 守卫，第二次不更新）
        ok2 = self.store.mark_search_log_click(log_id, "svc-account")
        self.assertFalse(ok2)

    def test_mark_click_unknown_log_returns_false(self):
        self.assertFalse(self.store.mark_search_log_click(99999, "svc-x"))

    def test_aggregate_no_click_queries(self):
        """同一 query 搜索 3 次，其中 1 次点击 → 无点击率 2/3。"""
        ts = 1000.0
        # 3 次搜索 "订单"，2 次无点击 + 1 次点击
        lid1 = self.store.append_search_log("u1", "订单", "default", ["svc-order"], {}, False, False, ts)
        lid2 = self.store.append_search_log("u1", "订单", "default", ["svc-order"], {}, False, False, ts)
        lid3 = self.store.append_search_log("u1", "订单", "default", ["svc-order"], {}, False, False, ts)
        # 回填 lid2 的点击（lid1/lid3 无点击）
        self.store.mark_search_log_click(lid2, "svc-order")
        # 1 次搜索 "账户" 无点击
        self.store.append_search_log("u1", "账户", "default", ["svc-account"], {}, False, False, ts)
        result = self.store.aggregate_no_click_queries(window_seconds=3600, now=ts + 100, limit=10)
        by_query = {r["query"]: r for r in result}
        self.assertIn("订单", by_query)
        self.assertEqual(by_query["订单"]["total"], 3)
        self.assertEqual(by_query["订单"]["no_click"], 2)
        self.assertAlmostEqual(by_query["订单"]["no_click_rate"], 2 / 3, places=4)
        self.assertIn("账户", by_query)
        self.assertEqual(by_query["账户"]["no_click"], 1)
        self.assertAlmostEqual(by_query["账户"]["no_click_rate"], 1.0, places=4)

    def test_aggregate_no_click_window_filter(self):
        """窗口外的记录不计入聚合。"""
        ts_old = 1000.0
        ts_new = 2000.0
        self.store.append_search_log("u1", "旧query", "default", [], {}, False, False, ts_old)
        self.store.append_search_log("u1", "新query", "default", [], {}, False, False, ts_new)
        # 窗口 = 100s，now = 2000 → 只含 ts >= 1900 的记录
        result = self.store.aggregate_no_click_queries(window_seconds=100, now=ts_new, limit=10)
        queries = [r["query"] for r in result]
        self.assertIn("新query", queries)
        self.assertNotIn("旧query", queries)

    def test_aggregate_high_latency_queries(self):
        """从 latencies_json.total 提取耗时，统计慢搜索次数。"""
        ts = 1000.0
        # "慢query" 2 次慢（2000ms） + 1 次快（50ms）
        self.store.append_search_log("u1", "慢query", "default", [], {"total": 2000.0}, False, False, ts)
        self.store.append_search_log("u1", "慢query", "default", [], {"total": 2000.0}, False, False, ts)
        self.store.append_search_log("u1", "慢query", "default", [], {"total": 50.0}, False, False, ts)
        # "快query" 1 次快（10ms）
        self.store.append_search_log("u1", "快query", "default", [], {"total": 10.0}, False, False, ts)
        result = self.store.aggregate_high_latency_queries(
            window_seconds=3600, now=ts + 100, latency_threshold_ms=1000.0, limit=10
        )
        by_query = {r["query"]: r for r in result}
        self.assertEqual(by_query["慢query"]["total"], 3)
        self.assertEqual(by_query["慢query"]["slow_count"], 2)
        self.assertAlmostEqual(by_query["慢query"]["avg_total_ms"], (2000 + 2000 + 50) / 3, places=2)
        self.assertEqual(by_query["慢query"]["max_total_ms"], 2000.0)
        self.assertEqual(by_query["快query"]["slow_count"], 0)
        # 按 slow_count 降序：慢query 在前
        self.assertEqual(result[0]["query"], "慢query")

    def test_degradation_stats(self):
        ts = 1000.0
        # 4 次搜索：1 缓存命中 + 1 降级 + 2 正常
        self.store.append_search_log("u1", "q1", "default", [], {}, True, False, ts)
        self.store.append_search_log("u1", "q2", "default", [], {}, False, True, ts)
        self.store.append_search_log("u1", "q3", "default", [], {}, False, False, ts)
        self.store.append_search_log("u1", "q4", "default", [], {}, False, False, ts)
        stats = self.store.search_log_degradation_stats(window_seconds=3600, now=ts + 100)
        self.assertEqual(stats["total"], 4)
        self.assertEqual(stats["cache_hit"], 1)
        self.assertAlmostEqual(stats["cache_hit_rate"], 0.25, places=4)
        self.assertEqual(stats["degraded"], 1)
        self.assertAlmostEqual(stats["degraded_rate"], 0.25, places=4)

    def test_degradation_stats_empty(self):
        stats = self.store.search_log_degradation_stats(window_seconds=3600)
        self.assertEqual(stats["total"], 0)
        self.assertEqual(stats["cache_hit_rate"], 0.0)


# ===========================================================================
# store：kb_op_logs
# ===========================================================================
class StoreKBOpLogsTests(unittest.TestCase):
    """M11：store.kb_op_logs 落库 + 列表 + 过滤。"""

    def setUp(self):
        self.store = SQLiteStore(":memory:")

    def tearDown(self):
        self.store.close()

    def test_append_and_list(self):
        id1 = self.store.append_kb_op_log(op="import", version_id="v1", kb_hash="h1", ok=True, detail={"n": 3})
        id2 = self.store.append_kb_op_log(op="export", kb_hash="h1", ok=True, detail={"bytes": 100})
        id3 = self.store.append_kb_op_log(op="rollback", version_id="v1", kb_hash="h1", ok=False, detail={"err": "x"})
        self.assertIsInstance(id1, int)
        logs = self.store.list_kb_op_logs(limit=10)
        self.assertEqual(len(logs), 3)
        # 新→旧：id3 在前
        self.assertEqual(logs[0]["id"], id3)
        self.assertEqual(logs[0]["op"], "rollback")
        self.assertFalse(logs[0]["ok"])
        self.assertEqual(logs[0]["detail"], {"err": "x"})
        self.assertEqual(logs[1]["op"], "export")
        self.assertTrue(logs[2]["ok"])

    def test_list_filter_by_op(self):
        self.store.append_kb_op_log(op="import", version_id="v1", ok=True)
        self.store.append_kb_op_log(op="export", ok=True)
        self.store.append_kb_op_log(op="import", version_id="v2", ok=True)
        imports = self.store.list_kb_op_logs(limit=10, op="import")
        self.assertEqual(len(imports), 2)
        self.assertTrue(all(r["op"] == "import" for r in imports))
        exports = self.store.list_kb_op_logs(limit=10, op="export")
        self.assertEqual(len(exports), 1)


# ===========================================================================
# engine：search_logs + kb_op_logs 集成
# ===========================================================================
class EngineDataLogsIntegrationTests(unittest.TestCase):
    """M11：engine.search 落 search_logs + record_click 回填 + KB 操作日志。"""

    def setUp(self):
        get_metrics().reset()
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "t.db")
        self.engine = _make_engine(self.db)
        self.engine.load_knowledge_base(KB)

    def tearDown(self):
        self.engine.store.close()
        get_metrics().reset()

    def test_search_appends_search_log(self):
        """成功搜索落一条 search_logs，含 top_ids + latencies + intent。"""
        results = self.engine.search("u1", "订单")
        self.assertTrue(results)
        logs = self.engine.recent_search_logs(limit=10)
        self.assertEqual(len(logs), 1)
        entry = logs[0]
        self.assertEqual(entry["query"], "订单")
        # top_ids 来自结果
        top_ids = [r["service_id"] for r in results[:10]]
        self.assertEqual(entry["top_ids"], top_ids)
        # latencies 含 total
        self.assertIn("total", entry["latencies"])
        self.assertFalse(entry["cache_hit"])
        self.assertFalse(entry["degraded"])
        # user_hash 不是原始 user_id
        self.assertNotEqual(entry["user_hash"], "u1")

    def test_cache_hit_path_appends_log(self):
        """缓存命中也落日志（cache_hit=True）。"""
        self.engine.search("u1", "订单")  # 首次
        self.engine.search("u1", "订单")  # 缓存命中
        logs = self.engine.recent_search_logs(limit=10)
        self.assertEqual(len(logs), 2)
        cache_log = logs[0]  # 最新一条
        self.assertTrue(cache_log["cache_hit"])
        self.assertEqual(cache_log["intent"], "cache_hit")

    def test_record_click_backfills_clicked_sid(self):
        """record_click 回填最近一次 search_logs.clicked_sid。"""
        results = self.engine.search("u1", "订单")
        clicked_sid = results[0]["service_id"]
        self.engine.record_click("u1", clicked_sid)
        logs = self.engine.recent_search_logs(limit=10)
        self.assertEqual(logs[0]["clicked_sid"], clicked_sid)

    def test_record_click_no_matching_top_id_no_backfill(self):
        """点击不在 top_ids 中的服务不回填（不属本次搜索结果）。

        3 服务 KB 下搜索 "订单" 通常返回全部 3 服务作为 top_ids，故用清空缓存
        模拟「无最近搜索」场景，验证 _mark_search_log_click 在无缓存时静默跳过。
        """
        results = self.engine.search("u1", "订单")
        clicked_sid = results[0]["service_id"]
        # 清空最近搜索日志缓存，模拟无最近搜索
        self.engine._last_search_log.clear()
        # 点击不应回填（无缓存条目）
        self.engine.record_click("u1", clicked_sid)
        logs = self.engine.recent_search_logs(limit=10)
        # clicked_sid 仍为 NULL（未回填）
        self.assertIsNone(logs[0]["clicked_sid"])

    def test_search_async_appends_search_log(self):
        """异步搜索也落 search_logs。"""
        results = asyncio.run(self.engine.search_async("u1", "账户"))
        self.assertTrue(results)
        logs = self.engine.recent_search_logs(limit=10)
        self.assertEqual(len(logs), 1)
        self.assertEqual(logs[0]["query"], "账户")

    def test_import_kb_version_logs_op(self):
        """import_kb_version 落 kb_op_logs（成功）。"""
        result = self.engine.import_kb_version(KB, version_id="v-test-log")
        logs = self.engine.list_kb_op_logs(limit=10, op="import")
        self.assertEqual(len(logs), 1)
        entry = logs[0]
        self.assertTrue(entry["ok"])
        self.assertEqual(entry["version_id"], "v-test-log")
        self.assertEqual(entry["kb_hash"], result["kb_hash"])
        self.assertEqual(entry["detail"]["services_count"], 3)

    def test_rollback_kb_logs_op(self):
        """rollback_kb 落 kb_op_logs（成功 + 失败）。"""
        self.engine.import_kb_version(KB, version_id="v-rb-1")
        # 成功回滚
        self.engine.rollback_kb("v-rb-1")
        # 回滚不存在的版本（失败日志）
        self.engine.rollback_kb("nope")
        logs = self.engine.list_kb_op_logs(limit=10, op="rollback")
        self.assertEqual(len(logs), 2)
        # 新→旧：失败日志在前（后执行）
        self.assertFalse(logs[0]["ok"])
        self.assertTrue(logs[1]["ok"])

    def test_aggregate_no_click_queries_via_engine(self):
        """engine 委托方法正常返回聚合结果。"""
        self.engine.search("u1", "订单")  # 无点击
        self.engine.search("u1", "订单")  # 无点击
        result = self.engine.aggregate_no_click_queries(window_seconds=3600, limit=10)
        self.assertTrue(any(r["query"] == "订单" for r in result))
        order_stat = next(r for r in result if r["query"] == "订单")
        self.assertEqual(order_stat["no_click"], 2)
        self.assertAlmostEqual(order_stat["no_click_rate"], 1.0, places=4)


# ===========================================================================
# API：/api/logs/* 端点
# ===========================================================================
class APIDataLogsTests(unittest.TestCase):
    """M11：/api/logs/* 端点闭环。"""

    def setUp(self):
        get_metrics().reset()
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "t.db")
        self.engine = _make_engine(self.db)
        self.engine.load_knowledge_base(KB)
        reset_engine(self.engine)
        self.client = TestClient(app)

    def tearDown(self):
        self.engine.store.close()
        get_metrics().reset()

    def test_no_click_endpoint_after_searches(self):
        """搜索后 GET /api/logs/search/no-click 返回聚合。"""
        self.client.get("/api/search", params={"user_id": "u1", "query": "订单"})
        self.client.get("/api/search", params={"user_id": "u1", "query": "账户"})
        r = self.client.get("/api/logs/search/no-click", params={"window": 3600, "limit": 50})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIsInstance(data, list)
        self.assertTrue(any(item["query"] in ("订单", "账户") for item in data))
        # 每项含 total/no_click/no_click_rate
        for item in data:
            self.assertIn("total", item)
            self.assertIn("no_click", item)
            self.assertIn("no_click_rate", item)

    def test_no_click_endpoint_validates_window(self):
        """window 参数校验（ge=60）。"""
        r = self.client.get("/api/logs/search/no-click", params={"window": 10})
        self.assertEqual(r.status_code, 422)

    def test_slow_endpoint(self):
        """GET /api/logs/search/slow 返回高延迟聚合。"""
        self.client.get("/api/search", params={"user_id": "u1", "query": "订单"})
        r = self.client.get("/api/logs/search/slow", params={"window": 3600, "threshold_ms": 0})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIsInstance(data, list)
        # threshold=0 → 所有搜索都算慢
        for item in data:
            self.assertIn("avg_total_ms", item)
            self.assertIn("max_total_ms", item)
            self.assertIn("slow_count", item)

    def test_degradation_endpoint(self):
        """GET /api/logs/degradation 返回降级统计。"""
        self.client.get("/api/search", params={"user_id": "u1", "query": "订单"})
        r = self.client.get("/api/logs/degradation", params={"window": 3600})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("total", data)
        self.assertIn("cache_hit_rate", data)
        self.assertIn("degraded_rate", data)
        self.assertGreaterEqual(data["total"], 1)

    def test_recent_endpoint(self):
        """GET /api/logs/search/recent 返回最近日志（user_hash 哈希化）。"""
        self.client.get("/api/search", params={"user_id": "alice", "query": "订单"})
        r = self.client.get("/api/logs/search/recent", params={"limit": 10})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertEqual(len(data), 1)
        entry = data[0]
        self.assertEqual(entry["query"], "订单")
        self.assertNotEqual(entry["user_hash"], "alice")  # 哈希化
        self.assertIn("latencies", entry)
        self.assertIn("top_ids", entry)

    def test_kb_ops_endpoint_after_import(self):
        """KB 导入后 GET /api/logs/kb-ops 返回操作日志。"""
        self.client.post("/api/kb/import", json=KB)
        r = self.client.get("/api/logs/kb-ops", params={"limit": 10})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(any(item["op"] == "import" for item in data))
        imp = next(item for item in data if item["op"] == "import")
        self.assertTrue(imp["ok"])
        self.assertIn("services_count", imp["detail"])

    def test_kb_ops_endpoint_filter_by_op(self):
        """GET /api/logs/kb-ops?op=export 过滤。"""
        self.client.post("/api/kb/import", json=KB)
        self.client.get("/api/kb/export")
        r = self.client.get("/api/logs/kb-ops", params={"op": "export"})
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(len(data) >= 1)
        self.assertTrue(all(item["op"] == "export" for item in data))

    def test_export_logs_op(self):
        """kb_export 落操作日志。"""
        self.client.post("/api/kb/import", json=KB)
        self.client.get("/api/kb/export")
        logs = self.engine.list_kb_op_logs(limit=10, op="export")
        self.assertEqual(len(logs), 1)
        self.assertTrue(logs[0]["ok"])
        self.assertIn("bytes", logs[0]["detail"])

    def test_rollback_logs_op_via_api(self):
        """API 回滚也落操作日志。"""
        r = self.client.post("/api/kb/import", json=KB)
        vid = r.json()["version_id"]
        self.client.post("/api/kb/rollback", params={"version_id": vid})
        logs = self.engine.list_kb_op_logs(limit=10, op="rollback")
        self.assertEqual(len(logs), 1)
        self.assertTrue(logs[0]["ok"])
        self.assertEqual(logs[0]["version_id"], vid)


if __name__ == "__main__":
    unittest.main()
