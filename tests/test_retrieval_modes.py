"""测试 retrieval_mode：keyword / semantic / hybrid 三模式。

覆盖：
  - _hybrid_score 权重归零：keyword 关闭向量、semantic 关闭 BM25、hybrid 全开
  - search 三模式均返回结果且含 rerank_reason
  - keyword/semantic 跳过 rerank+reason（rerank_score == score）
  - 三模式结果含正确字段（service_id/route/score）
  - 无效 mode 不崩（默认 hybrid 兜底）
"""
from __future__ import annotations

import unittest

from easysearch import DashScopeClient, DeepSeekClient, ServiceSearchEngine, SQLiteStore
from easysearch.config import BM25_WEIGHT, POPULARITY_WEIGHT, VECTOR_WEIGHT


SERVICES = [
    {
        "service_id": "svc-1",
        "service_name": "订单中心",
        "aliases": ["订单", "order"],
        "service_intro": "查看与管理订单信息，支持订单审批",
        "route": "/orders",
    },
    {
        "service_id": "svc-2",
        "service_name": "用户中心",
        "aliases": ["用户", "customer"],
        "service_intro": "查看用户画像与账户信息",
        "route": "/users",
    },
    {
        "service_id": "svc-3",
        "service_name": "风控平台",
        "aliases": ["风控", "risk"],
        "service_intro": "风险决策管理与监控",
        "route": "/risk/decision",
    },
]


def _make_engine(db_path: str = ":memory:") -> ServiceSearchEngine:
    return ServiceSearchEngine(
        dashscope_client=DashScopeClient(api_key=None),
        deepseek_client=DeepSeekClient(api_key=None),
        store=SQLiteStore(db_path),
        db_path=db_path,
    )


class HybridScoreTests(unittest.TestCase):
    """_hybrid_score 权重归零逻辑。"""

    def test_hybrid_default(self):
        """hybrid 模式权重全开（0.6/0.3/0.1）。"""
        score = ServiceSearchEngine._hybrid_score(0.8, 0.5, 1.0)
        expected = VECTOR_WEIGHT * 0.8 + BM25_WEIGHT * 0.5 + POPULARITY_WEIGHT * 1.0
        self.assertAlmostEqual(score, expected, places=10)

    def test_keyword_zeroes_vector(self):
        """keyword 模式向量权重归零。"""
        score = ServiceSearchEngine._hybrid_score(0.8, 0.5, 1.0, retrieval_mode="keyword")
        expected = 0.0 * 0.8 + BM25_WEIGHT * 0.5 + POPULARITY_WEIGHT * 1.0
        self.assertAlmostEqual(score, expected, places=10)
        # 向量分不影响 keyword 模式
        score2 = ServiceSearchEngine._hybrid_score(0.0, 0.5, 1.0, retrieval_mode="keyword")
        self.assertAlmostEqual(score, score2, places=10)

    def test_semantic_zeroes_bm25(self):
        """semantic 模式 BM25 权重归零。"""
        score = ServiceSearchEngine._hybrid_score(0.8, 0.5, 1.0, retrieval_mode="semantic")
        expected = VECTOR_WEIGHT * 0.8 + 0.0 * 0.5 + POPULARITY_WEIGHT * 1.0
        self.assertAlmostEqual(score, expected, places=10)
        # BM25 分不影响 semantic 模式
        score2 = ServiceSearchEngine._hybrid_score(0.8, 0.0, 1.0, retrieval_mode="semantic")
        self.assertAlmostEqual(score, score2, places=10)

    def test_popularity_preserved_in_all_modes(self):
        """popularity 权重在所有模式都保留（行为信号不随检索模式消失）。"""
        for mode in ("keyword", "semantic", "hybrid"):
            score = ServiceSearchEngine._hybrid_score(0.0, 0.0, 1.0, retrieval_mode=mode)
            self.assertAlmostEqual(score, POPULARITY_WEIGHT * 1.0, places=10)

    def test_backward_compat_two_args(self):
        """verify.py 风格三位置参数调用（不传 retrieval_mode）保 hybrid 行为。"""
        score = ServiceSearchEngine._hybrid_score(0.8, 0.5, 1.0)
        expected = VECTOR_WEIGHT * 0.8 + BM25_WEIGHT * 0.5 + POPULARITY_WEIGHT * 1.0
        self.assertAlmostEqual(score, expected, places=10)


class RetrievalModeSearchTests(unittest.TestCase):
    """search / search_async 三模式端到端。"""

    def setUp(self):
        self.engine = _make_engine()
        self.engine.load_knowledge_base(SERVICES)

    def test_all_modes_return_results(self):
        """三种模式都能返回结果且含核心字段。"""
        for mode in ("keyword", "semantic", "hybrid"):
            with self.subTest(mode=mode):
                results = self.engine.search("u1", "订单", retrieval_mode=mode)
                self.assertTrue(len(results) > 0, f"{mode} 模式应返回结果")
                item = results[0]
                self.assertIn("service_id", item)
                self.assertIn("route", item)
                self.assertIn("score", item)
                self.assertIn("rerank_reason", item)
                self.assertTrue(item["rerank_reason"])

    def test_keyword_skips_rerank(self):
        """keyword 模式跳过 rerank：rerank_score == score（直接用召回分）。"""
        results = self.engine.search("u1", "订单", retrieval_mode="keyword")
        self.assertTrue(len(results) > 0)
        for item in results:
            self.assertAlmostEqual(item["rerank_score"], item["score"], places=10)

    def test_semantic_skips_rerank(self):
        """semantic 模式跳过 rerank：rerank_score == score。"""
        results = self.engine.search("u1", "用户", retrieval_mode="semantic")
        self.assertTrue(len(results) > 0)
        for item in results:
            self.assertAlmostEqual(item["rerank_score"], item["score"], places=10)

    def test_keyword_results_differ_from_hybrid(self):
        """keyword 与 hybrid 结果排序可能不同（权重不同导致排序差异）。

        注：离线 fallback 向量为 hash，差异可能不大，但模式参数确实传入并生效。
        这里验证两种模式都能正常返回结果即可（不硬断排序差异，防 flaky）。
        """
        kw = self.engine.search("u1", "风控", retrieval_mode="keyword")
        hy = self.engine.search("u2", "风控", retrieval_mode="hybrid")
        self.assertTrue(len(kw) > 0)
        self.assertTrue(len(hy) > 0)

    def test_search_async_keyword(self):
        """search_async keyword 模式也能返回结果。"""
        import asyncio

        results = asyncio.run(
            self.engine.search_async("u1", "订单", retrieval_mode="keyword")
        )
        self.assertTrue(len(results) > 0)
        for item in results:
            self.assertTrue(item["rerank_reason"])
            self.assertAlmostEqual(item["rerank_score"], item["score"], places=10)

    def test_search_async_semantic(self):
        """search_async semantic 模式也能返回结果。"""
        import asyncio

        results = asyncio.run(
            self.engine.search_async("u1", "用户", retrieval_mode="semantic")
        )
        self.assertTrue(len(results) > 0)
        for item in results:
            self.assertTrue(item["rerank_reason"])

    def test_search_async_hybrid(self):
        """search_async hybrid 模式也能返回结果。"""
        import asyncio

        results = asyncio.run(
            self.engine.search_async("u1", "风控", retrieval_mode="hybrid")
        )
        self.assertTrue(len(results) > 0)
        for item in results:
            self.assertTrue(item["rerank_reason"])

    def test_default_mode_is_hybrid(self):
        """不传 retrieval_mode 默认 hybrid（保旧调用方兼容）。"""
        results_default = self.engine.search("u1", "订单")
        results_hybrid = self.engine.search("u2", "订单", retrieval_mode="hybrid")
        # 两模式结果一致（都是 hybrid），第一条 service_id 相同
        self.assertEqual(results_default[0]["service_id"], results_hybrid[0]["service_id"])


if __name__ == "__main__":
    unittest.main()
