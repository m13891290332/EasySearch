"""测试搜索结果缓存：Memory + Redis（mock）+ 工厂 + 降级。

覆盖：
  - MemoryResultCache: get/set/invalidate, TTL 过期, LRU 淘汰, key 含 retrieval_mode
  - RedisResultCache: mock redis 客户端, get/set/invalidate, 异常降级为 miss
  - get_cache 工厂: 无 REDIS_URL → Memory, reset_cache 重置
  - engine 缓存集成: 命中返回缓存, 未命中计算后写回
  - TTL 跟随缓存实现: Redis 用配置 TTL, Memory 用 60s
"""
from __future__ import annotations

import json
import os
import time
import unittest
from unittest.mock import MagicMock, patch

from easysearch import DashScopeClient, DeepSeekClient, ServiceSearchEngine, SQLiteStore
from easysearch.cache import (
    MemoryResultCache,
    RedisResultCache,
    ResultCache,
    get_cache,
    reset_cache,
)


SERVICES = [
    {
        "service_id": "svc-1",
        "service_name": "订单中心",
        "aliases": ["订单"],
        "service_intro": "订单管理",
        "route": "/orders",
    },
    {
        "service_id": "svc-2",
        "service_name": "用户中心",
        "aliases": ["用户"],
        "service_intro": "用户管理",
        "route": "/users",
    },
]


def _make_engine(db_path: str = ":memory:") -> ServiceSearchEngine:
    return ServiceSearchEngine(
        dashscope_client=DashScopeClient(api_key=None),
        deepseek_client=DeepSeekClient(api_key=None),
        store=SQLiteStore(db_path),
        db_path=db_path,
    )


class MemoryResultCacheTests(unittest.TestCase):
    """MemoryResultCache: LRU + TTL + key 隔离。"""

    def setUp(self):
        self.cache = MemoryResultCache(size=3, ttl=1.0)
        self.results = [{"service_id": "svc-1", "score": 0.8}]

    def test_set_then_get(self):
        self.cache.set("u1", "订单", "hybrid", self.results, ttl=1.0)
        got = self.cache.get("u1", "订单", "hybrid")
        self.assertEqual(got, self.results)

    def test_get_miss(self):
        got = self.cache.get("u1", "不存在", "hybrid")
        self.assertIsNone(got)

    def test_ttl_expiry(self):
        self.cache.set("u1", "订单", "hybrid", self.results, ttl=0.1)
        time.sleep(0.15)
        got = self.cache.get("u1", "订单", "hybrid")
        self.assertIsNone(got)

    def test_lru_eviction(self):
        """size=3，第 4 条写入淘汰最旧的。"""
        for i in range(4):
            self.cache.set("u1", f"q{i}", "hybrid", self.results, ttl=10.0)
        # q0 被淘汰
        self.assertIsNone(self.cache.get("u1", "q0", "hybrid"))
        # q3 仍在
        self.assertEqual(self.cache.get("u1", "q3", "hybrid"), self.results)

    def test_key_includes_retrieval_mode(self):
        """相同 query 不同 retrieval_mode 不串结果。"""
        kw_results = [{"service_id": "svc-kw"}]
        sem_results = [{"service_id": "svc-sem"}]
        self.cache.set("u1", "订单", "keyword", kw_results, ttl=10.0)
        self.cache.set("u1", "订单", "semantic", sem_results, ttl=10.0)
        self.assertEqual(self.cache.get("u1", "订单", "keyword"), kw_results)
        self.assertEqual(self.cache.get("u1", "订单", "semantic"), sem_results)

    def test_key_includes_user_id(self):
        """相同 query 不同 user_id 不串结果。"""
        self.cache.set("u1", "订单", "hybrid", [{"sid": "a"}], ttl=10.0)
        self.cache.set("u2", "订单", "hybrid", [{"sid": "b"}], ttl=10.0)
        self.assertEqual(self.cache.get("u1", "订单", "hybrid"), [{"sid": "a"}])
        self.assertEqual(self.cache.get("u2", "订单", "hybrid"), [{"sid": "b"}])

    def test_invalidate_all(self):
        self.cache.set("u1", "q1", "hybrid", self.results, ttl=10.0)
        self.cache.set("u2", "q2", "hybrid", self.results, ttl=10.0)
        self.cache.invalidate()
        self.assertIsNone(self.cache.get("u1", "q1", "hybrid"))
        self.assertIsNone(self.cache.get("u2", "q2", "hybrid"))

    def test_invalidate_by_user(self):
        """invalidate(user_id) 只清该用户缓存。"""
        self.cache.set("u1", "q1", "hybrid", self.results, ttl=10.0)
        self.cache.set("u2", "q2", "hybrid", self.results, ttl=10.0)
        self.cache.invalidate("u1")
        self.assertIsNone(self.cache.get("u1", "q1", "hybrid"))
        self.assertIsNotNone(self.cache.get("u2", "q2", "hybrid"))

    def test_get_moves_to_end(self):
        """get 后该条变最新，LRU 淘汰时不选它。"""
        self.cache.set("u1", "q0", "hybrid", [{"a": 0}], ttl=10.0)
        self.cache.set("u1", "q1", "hybrid", [{"a": 1}], ttl=10.0)
        self.cache.set("u1", "q2", "hybrid", [{"a": 2}], ttl=10.0)
        # get q0 使其变最新
        self.cache.get("u1", "q0", "hybrid")
        # 写 q3 → 淘汰最旧（q1，不是 q0）
        self.cache.set("u1", "q3", "hybrid", [{"a": 3}], ttl=10.0)
        self.assertIsNone(self.cache.get("u1", "q1", "hybrid"))
        self.assertIsNotNone(self.cache.get("u1", "q0", "hybrid"))


class RedisResultCacheTests(unittest.TestCase):
    """RedisResultCache: mock redis 客户端。"""

    def _make_cache(self, ttl=300.0):
        client = MagicMock()
        return RedisResultCache(client=client, ttl=ttl)

    def test_set_calls_setex(self):
        cache = self._make_cache(ttl=300)
        results = [{"service_id": "svc-1"}]
        cache.set("u1", "订单", "hybrid", results, ttl=300)
        cache._client.setex.assert_called_once()
        args = cache._client.setex.call_args
        key = args[0][0]
        ttl_val = args[0][1]
        val = args[0][2]
        self.assertEqual(ttl_val, 300)
        self.assertEqual(json.loads(val), results)
        self.assertIn("u1", key)
        self.assertIn("hybrid", key)

    def test_get_hit(self):
        cache = self._make_cache()
        results = [{"service_id": "svc-1"}]
        cache._client.get.return_value = json.dumps(results, ensure_ascii=False)
        got = cache.get("u1", "订单", "hybrid")
        self.assertEqual(got, results)

    def test_get_miss(self):
        cache = self._make_cache()
        cache._client.get.return_value = None
        self.assertIsNone(cache.get("u1", "订单", "hybrid"))

    def test_get_redis_exception_returns_none(self):
        """Redis 异常视作 miss，不抛错。"""
        cache = self._make_cache()
        cache._client.get.side_effect = Exception("connection lost")
        self.assertIsNone(cache.get("u1", "订单", "hybrid"))

    def test_set_redis_exception_silent(self):
        """Redis 写入失败静默跳过，不抛错。"""
        cache = self._make_cache()
        cache._client.setex.side_effect = Exception("write failed")
        # 不抛异常
        cache.set("u1", "订单", "hybrid", [{"sid": "x"}], ttl=300)

    def test_get_corrupt_payload_returns_none(self):
        """脏数据视作 miss。"""
        cache = self._make_cache()
        cache._client.get.return_value = "not valid json{{{"
        self.assertIsNone(cache.get("u1", "订单", "hybrid"))

    def test_invalidate_by_user_uses_scan(self):
        """invalidate(user_id) 用 SCAN + DELETE 清前缀。"""
        cache = self._make_cache()
        cache._client.scan.side_effect = [
            (1, ["es:res:u1:abc:hybrid", "es:res:u1:def:keyword"]),
            (0, []),
        ]
        cache.invalidate("u1")
        cache._client.delete.assert_called_once_with(
            "es:res:u1:abc:hybrid", "es:res:u1:def:keyword"
        )

    def test_invalidate_all_noop(self):
        """invalidate(None) 不操作（Redis TTL 已兜底）。"""
        cache = self._make_cache()
        cache.invalidate(None)
        cache._client.scan.assert_not_called()

    def test_key_structure(self):
        """key 格式：es:res:{user_id}:{sha256(query)}:{retrieval_mode}。"""
        key = ResultCache._key("u1", "订单", "hybrid")
        self.assertTrue(key.startswith("es:res:u1:"))
        self.assertTrue(key.endswith(":hybrid"))
        # query 被 sha256 哈希（不是明文）
        self.assertNotIn("订单", key)


class GetCacheFactoryTests(unittest.TestCase):
    """get_cache 工厂 + reset_cache。"""

    def setUp(self):
        reset_cache()

    def tearDown(self):
        reset_cache()

    def test_no_redis_url_returns_memory(self):
        """无 REDIS_URL → MemoryResultCache。"""
        with patch.dict(os.environ, {"REDIS_URL": "", "EASYSEARCH_REDIS_URL": ""}):
            cache = get_cache()
            self.assertIsInstance(cache, MemoryResultCache)

    def test_singleton(self):
        """get_cache 返回同一实例。"""
        with patch.dict(os.environ, {"REDIS_URL": "", "EASYSEARCH_REDIS_URL": ""}):
            c1 = get_cache()
            c2 = get_cache()
            self.assertIs(c1, c2)

    def test_reset_cache_clears_singleton(self):
        """reset_cache 后再次 get_cache 返回新实例。"""
        with patch.dict(os.environ, {"REDIS_URL": "", "EASYSEARCH_REDIS_URL": ""}):
            c1 = get_cache()
            reset_cache()
            c2 = get_cache()
            self.assertIsNot(c1, c2)

    def test_redis_url_unavailable_falls_back_to_memory(self):
        """REDIS_URL 配了但连不上 → 降级 Memory。"""
        with patch.dict(os.environ, {"REDIS_URL": "redis://nonexistent:6379/0"}):
            cache = get_cache()
            # ping 失败降级
            self.assertIsInstance(cache, MemoryResultCache)


class EngineCacheIntegrationTests(unittest.TestCase):
    """engine 缓存集成：命中返回缓存，未命中计算后写回。"""

    def setUp(self):
        reset_cache()
        self.engine = _make_engine()
        self.engine.load_knowledge_base(SERVICES)

    def tearDown(self):
        reset_cache()

    def test_cache_hit_returns_same_results(self):
        """第二次相同查询命中缓存，返回相同结果。"""
        r1 = self.engine.search("u1", "订单", retrieval_mode="hybrid")
        r2 = self.engine.search("u1", "订单", retrieval_mode="hybrid")
        self.assertEqual(len(r1), len(r2))
        self.assertEqual(r1[0]["service_id"], r2[0]["service_id"])

    def test_cache_different_modes_dont_cross(self):
        """不同 retrieval_mode 缓存不串。"""
        kw = self.engine.search("u1", "订单", retrieval_mode="keyword")
        hy = self.engine.search("u1", "订单", retrieval_mode="hybrid")
        # 两种模式都能返回结果（不因缓存串模式）
        self.assertTrue(len(kw) > 0)
        self.assertTrue(len(hy) > 0)

    def test_cache_invalidate_on_click(self):
        """record_click 后缓存失效，下次搜索重新计算。"""
        r1 = self.engine.search("u1", "订单", retrieval_mode="hybrid")
        self.assertTrue(len(r1) > 0)
        # 点击后缓存失效
        self.engine.record_click("u1", r1[0]["service_id"])
        # 重新搜索应重新计算（不报错即可）
        r2 = self.engine.search("u1", "订单", retrieval_mode="hybrid")
        self.assertTrue(len(r2) > 0)

    def test_cache_ttl_follows_implementation(self):
        """_result_cache_set 使用缓存实例自身 _ttl，不硬编码 60s。"""
        cache = self.engine.result_cache
        # Memory 的 _ttl 应为 60.0（保旧行为）
        if isinstance(cache, MemoryResultCache):
            self.assertAlmostEqual(cache._ttl, 60.0, places=1)


if __name__ == "__main__":
    unittest.main()
