"""测试 _build_template_reason：差异化模板排序理由。

覆盖：
  - 空 query → 通用模板
  - 服务名命中 → "服务名「xxx」直接命中查询"
  - 别名命中 → "别名「xxx」命中查询"
  - 简介命中 → "服务简介包含查询关键词"
  - 无字面命中 → "语义相似匹配（无关键词字面命中）"
  - 分数尾巴格式 "综合分X.XXXX。"
  - search/search_async 在 keyword/semantic 模式下使用模板 reason
"""
from __future__ import annotations

import unittest

from easysearch import DashScopeClient, DeepSeekClient, Qwen3VLReranker, ServiceSearchEngine, SQLiteStore


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


class TemplateReasonTests(unittest.TestCase):
    """_build_template_reason 各分支覆盖。"""

    def test_empty_query_returns_generic(self):
        item = {"service_name": "订单中心", "aliases": ["订单"], "service_intro": "订单管理", "score": 0.8266}
        reason = Qwen3VLReranker._build_template_reason("", item)
        self.assertIn("综合相关性与关键词覆盖", reason)
        self.assertIn("综合分0.8266", reason)

    def test_score_tail_format(self):
        item = {"service_name": "x", "aliases": [], "service_intro": "", "score": 0.5}
        reason = Qwen3VLReranker._build_template_reason("x", item)
        self.assertTrue(reason.endswith("综合分0.5000。"))

    def test_rerank_score_preferred_over_score(self):
        item = {"service_name": "x", "aliases": [], "service_intro": "", "score": 0.1, "rerank_score": 0.9}
        reason = Qwen3VLReranker._build_template_reason("x", item)
        self.assertIn("综合分0.9000", reason)
        self.assertNotIn("综合分0.1000", reason)

    def test_name_hit(self):
        """query token 命中服务名。"""
        item = {"service_name": "订单中心", "aliases": [], "service_intro": "其他内容", "score": 0.7}
        reason = Qwen3VLReranker._build_template_reason("订单", item)
        self.assertIn("服务名「订单中心」直接命中查询", reason)

    def test_name_whole_in_query(self):
        """服务名整体出现在 query 中也算命中。"""
        item = {"service_name": "风控", "aliases": [], "service_intro": "无关内容", "score": 0.6}
        reason = Qwen3VLReranker._build_template_reason("风控平台", item)
        self.assertIn("服务名「风控」直接命中查询", reason)

    def test_alias_hit(self):
        """query token 命中别名。"""
        item = {"service_name": "用户中心", "aliases": ["customer"], "service_intro": "无关", "score": 0.5}
        reason = Qwen3VLReranker._build_template_reason("customer", item)
        self.assertIn("别名「customer」命中查询", reason)

    def test_alias_whole_in_query(self):
        """别名整体出现在 query 中也算命中。"""
        item = {"service_name": "用户中心", "aliases": ["账户"], "service_intro": "无关", "score": 0.5}
        reason = Qwen3VLReranker._build_template_reason("账户管理", item)
        self.assertIn("别名「账户」命中查询", reason)

    def test_intro_hit(self):
        """名称/别名未命中但简介包含查询关键词。"""
        item = {"service_name": "X服务", "aliases": [], "service_intro": "支持订单审批流程", "score": 0.4}
        reason = Qwen3VLReranker._build_template_reason("审批", item)
        self.assertIn("服务简介包含查询关键词", reason)

    def test_no_literal_hit_returns_semantic(self):
        """无字面命中 → 语义相似模板。"""
        item = {"service_name": "X服务", "aliases": ["y"], "service_intro": "完全不相关的内容", "score": 0.3}
        reason = Qwen3VLReranker._build_template_reason("查询", item)
        self.assertIn("语义相似匹配", reason)
        self.assertIn("无关键词字面命中", reason)

    def test_name_takes_priority_over_alias(self):
        """名称命中优先于别名命中。"""
        item = {"service_name": "订单", "aliases": ["order"], "service_intro": "订单管理", "score": 0.5}
        reason = Qwen3VLReranker._build_template_reason("订单", item)
        self.assertIn("服务名「订单」", reason)
        self.assertNotIn("别名", reason)

    def test_alias_takes_priority_over_intro(self):
        """别名命中优先于简介命中。"""
        item = {"service_name": "X服务", "aliases": ["订单"], "service_intro": "订单管理", "score": 0.5}
        reason = Qwen3VLReranker._build_template_reason("订单", item)
        self.assertIn("别名「订单」", reason)
        self.assertNotIn("服务简介", reason)


class EngineTemplateReasonTests(unittest.TestCase):
    """keyword/semantic 模式下 search/search_async 使用模板 reason。"""

    def setUp(self):
        self.engine = _make_engine()
        self.engine.load_knowledge_base(SERVICES)

    def test_keyword_mode_uses_template_reason(self):
        """keyword 模式跳过 rerank+reason，结果含模板 reason。"""
        results = self.engine.search("u1", "订单", retrieval_mode="keyword")
        self.assertTrue(len(results) > 0)
        for item in results:
            self.assertTrue(item["rerank_reason"])
            # 模板 reason 都以「综合分」结尾
            self.assertIn("综合分", item["rerank_reason"])

    def test_semantic_mode_uses_template_reason(self):
        """semantic 模式同样跳过 rerank+reason，结果含模板 reason。"""
        results = self.engine.search("u1", "用户", retrieval_mode="semantic")
        self.assertTrue(len(results) > 0)
        for item in results:
            self.assertTrue(item["rerank_reason"])
            self.assertIn("综合分", item["rerank_reason"])

    def test_keyword_mode_name_hit_reason(self):
        """keyword 模式下名称命中应生成「服务名直接命中」模板。"""
        results = self.engine.search("u1", "订单", retrieval_mode="keyword")
        top = results[0]
        # 订单中心名称命中
        self.assertIn("服务名「订单中心」", top["rerank_reason"])

    def test_hybrid_mode_default_has_reason(self):
        """hybrid 默认模式也应有 reason（离线 fallback 也是模板）。"""
        results = self.engine.search("u1", "风控")
        self.assertTrue(len(results) > 0)
        for item in results:
            self.assertTrue(item["rerank_reason"])


if __name__ == "__main__":
    unittest.main()
