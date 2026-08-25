"""M15 意图驱动二次深度检索测试。

覆盖：
  - IntentRouter.evaluate_confidence 各触发条件与不触发场景。
  - engine._rrf_fuse RRF 融合排序。
  - engine._deep_expand_query 查询扩展（含 base + KB 共现词）。
  - engine._maybe_deep_search 触发/不触发/标签/防递归。
"""
from __future__ import annotations

import unittest

from easysearch import (
    CONVERSATIONAL,
    DEFAULT,
    INFORMATIONAL,
    MULTI_CONDITION,
    NAVIGATIONAL,
    DashScopeClient,
    DeepSeekClient,
    IntentRouter,
    ServiceSearchEngine,
    SQLiteStore,
)

SERVICES = [
    {
        "service_id": "svc-1",
        "service_name": "订单中心",
        "aliases": ["订单", "order"],
        "service_intro": "查看与管理订单信息，支持订单审批",
        "route": {"path": "/orders", "component": "OrderTable", "action_button": "ApproveOrder"},
    },
    {
        "service_id": "svc-2",
        "service_name": "用户中心",
        "aliases": ["用户", "customer"],
        "service_intro": "查看用户画像",
        "route": {"path": "/users", "component": "UserProfile", "action_button": "ConfirmUser"},
    },
    {
        "service_id": "svc-3",
        "service_name": "风控平台",
        "aliases": ["风控", "risk"],
        "service_intro": "风险决策管理",
        "route": "/risk/decision",
    },
]


def make_engine():
    return ServiceSearchEngine(
        dashscope_client=DashScopeClient(api_key=None),
        deepseek_client=DeepSeekClient(api_key=None),
        store=SQLiteStore(":memory:"),
        db_path=":memory:",  # 使 _embeddings_dir=None，不落 .npz 避免跨测试污染
    )


def _item(sid: str, score: float) -> dict:
    return {
        "service_id": sid,
        "service_name": sid,
        "aliases": [],
        "service_intro": "",
        "route": f"/{sid}",
        "component": "C",
        "decision_button": "B",
        "derived": False,
        "score": score,
        "rerank_score": score,
    }


class ConfidenceTests(unittest.TestCase):
    def setUp(self):
        self.router = IntentRouter()

    def test_navigational_no_trigger(self):
        r = self.router.evaluate_confidence([_item("a", 0.1)], NAVIGATIONAL)
        self.assertFalse(r.should_deep_search)

    def test_multi_condition_no_trigger(self):
        r = self.router.evaluate_confidence([_item("a", 0.1)], MULTI_CONDITION)
        self.assertFalse(r.should_deep_search)

    def test_conversational_no_trigger(self):
        r = self.router.evaluate_confidence([_item("a", 0.1)], CONVERSATIONAL)
        self.assertFalse(r.should_deep_search)

    def test_empty_results_triggers(self):
        r = self.router.evaluate_confidence([], DEFAULT)
        self.assertTrue(r.should_deep_search)
        self.assertIn("无命中", r.reason)

    def test_low_delta_triggers(self):
        results = [_item("a", 0.30), _item("b", 0.28), _item("c", 0.27)]
        r = self.router.evaluate_confidence(results, DEFAULT)
        self.assertTrue(r.should_deep_search)
        self.assertIn("头部分离不足", r.reason)

    def test_sparse_hits_triggers(self):
        # n_hits=1（其余 score=0），delta 大但仍因命中稀疏触发
        results = [_item("a", 0.9), _item("b", 0.0), _item("c", 0.0)]
        r = self.router.evaluate_confidence(results, DEFAULT)
        self.assertTrue(r.should_deep_search)
        self.assertIn("命中稀疏", r.reason)

    def test_informational_low_relevance_triggers(self):
        results = [_item("a", 0.2), _item("b", 0.1)]
        r = self.router.evaluate_confidence(results, INFORMATIONAL)
        self.assertTrue(r.should_deep_search)

    def test_strong_results_no_trigger(self):
        results = [_item("a", 0.9), _item("b", 0.1), _item("c", 0.05)]
        r = self.router.evaluate_confidence(results, DEFAULT)
        self.assertFalse(r.should_deep_search)
        self.assertIn("充足", r.reason)

    def test_cold_user_factor_lowers_confidence(self):
        results = [_item("a", 0.9), _item("b", 0.1), _item("c", 0.05)]
        warm = self.router.evaluate_confidence(results, DEFAULT, is_cold_user=False)
        cold = self.router.evaluate_confidence(results, DEFAULT, is_cold_user=True)
        # 强结果两者都不触发，但冷启动 confidence 数值更低
        self.assertLessEqual(cold.confidence, warm.confidence)


class RRFFuseTests(unittest.TestCase):
    def test_fuse_double_appearance_ranks_first(self):
        a = [_item("a", 0.9), _item("b", 0.5)]
        b = [_item("b", 0.8), _item("c", 0.7), _item("d", 0.6)]
        fused = ServiceSearchEngine._rrf_fuse(a, b, k=60, top_k=3)
        ids = [x["service_id"] for x in fused]
        # b 同时出现在两表 → RRF 最高
        self.assertEqual(ids[0], "b")
        self.assertIn("a", ids)
        self.assertEqual(len(fused), 3)

    def test_fuse_deep_only_item_gets_template_reason(self):
        a = [_item("a", 0.9)]
        b = [_item("d", 0.6)]
        fused = ServiceSearchEngine._rrf_fuse(a, b, k=60, top_k=5)
        deep_only = next(x for x in fused if x["service_id"] == "d")
        self.assertEqual(deep_only["rerank_reason"], "深度检索补充结果。")

    def test_fuse_empty_deep_returns_first(self):
        a = [_item("a", 0.9), _item("b", 0.5)]
        fused = ServiceSearchEngine._rrf_fuse(a, [], k=60, top_k=10)
        self.assertEqual([x["service_id"] for x in fused], ["a", "b"])


class DeepExpandQueryTests(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()
        self.engine.load_knowledge_base(SERVICES)

    def test_expand_includes_base(self):
        from easysearch.utils import tokenize

        base = set(tokenize("订单"))
        expanded = set(self.engine._deep_expand_query("订单"))
        self.assertTrue(base <= expanded)

    def test_expand_adds_kb_cooccurrence(self):
        # "订单" 命中 svc-1 name "订单中心" → 扩展应含 "中心"
        expanded = set(self.engine._deep_expand_query("订单"))
        self.assertIn("中心", expanded)

    def test_expand_capped(self):
        expanded = self.engine._deep_expand_query("订单", max_tokens=3)
        self.assertLessEqual(len(expanded), 3)


class MaybeDeepSearchTests(unittest.TestCase):
    def setUp(self):
        self.engine = make_engine()
        self.engine.load_knowledge_base(SERVICES)

    def test_triggers_on_low_delta_and_tags(self):
        weak = [
            _item("svc-1", 0.30),
            _item("svc-2", 0.28),
            _item("svc-3", 0.27),
        ]
        out = self.engine._maybe_deep_search("u-weak", "订单", weak, DEFAULT)
        self.assertGreater(len(out), 0)
        self.assertTrue(out[0].get("deep_searched"))
        self.assertIn("头部分离不足", out[0].get("deep_reason", ""))

    def test_skips_strong_results(self):
        strong = [
            _item("svc-1", 0.95),
            _item("svc-2", 0.10),
            _item("svc-3", 0.05),
        ]
        out = self.engine._maybe_deep_search("u-strong", "订单", strong, DEFAULT)
        self.assertFalse(any(x.get("deep_searched") for x in out))

    def test_skips_navigational_even_if_weak(self):
        weak = [_item("svc-1", 0.30), _item("svc-2", 0.28), _item("svc-3", 0.27)]
        out = self.engine._maybe_deep_search("u-nav", "订单", weak, NAVIGATIONAL)
        self.assertFalse(any(x.get("deep_searched") for x in out))

    def test_no_recursion_structure(self):
        """_maybe_deep_search 内部仅调 _build_top_candidates（不调自身/search），
        故单次 search 至多触发一次深度检索。此处验证返回项被打标且不异常。"""
        weak = [_item("svc-1", 0.30), _item("svc-2", 0.29)]
        out = self.engine._maybe_deep_search("u-rec", "订单", weak, DEFAULT)
        self.assertTrue(all(x.get("deep_searched") for x in out))

    def test_empty_first_results_returns_empty(self):
        out = self.engine._maybe_deep_search("u-empty", "订单", [], DEFAULT)
        self.assertEqual(out, [])


if __name__ == "__main__":
    unittest.main()
