"""M5 意图识别测试。

覆盖：
  - IntentRouter 各意图分类规则与优先级。
  - engine.classify_intent 与 navigational 直达置顶（_pin_navigational_to_top）。
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
    ServiceRecord,
    ServiceSearchEngine,
    SQLiteStore,
)

# 复用 test_search_engine 的知识库形态
SERVICES = [
    {
        "service_id": "svc-1",
        "service_name": "订单中心",
        "aliases": ["订单", "order"],
        "service_intro": "查看与管理订单信息，支持订单审批",
        "route": {
            "path": "/orders",
            "component": "OrderTable",
            "action_button": "ApproveOrder",
        },
    },
    {
        "service_id": "svc-2",
        "service_name": "用户中心",
        "aliases": ["用户", "customer"],
        "service_intro": "查看用户画像",
        "route": {
            "path": "/users",
            "component": "UserProfile",
            "action_button": "ConfirmUser",
        },
    },
    {
        "service_id": "svc-3",
        "service_name": "风控平台",
        "aliases": ["风控", "risk"],
        "service_intro": "风险决策管理",
        "route": "/risk/decision",
    },
]


def make_engine(api_key=None, db_path=":memory:"):
    client = DashScopeClient(api_key=api_key)
    ds_client = DeepSeekClient(api_key=api_key)
    store = SQLiteStore(db_path)
    # db_path 必须透传给 engine：否则 engine 默认 db_path="data/easysearch.db"，
    # _embeddings_dir 指向 data/embeddings/，M4 .npz 跨测试污染。
    return ServiceSearchEngine(
        dashscope_client=client, deepseek_client=ds_client, store=store, db_path=db_path
    )


def _services_dict() -> dict[str, ServiceRecord]:
    return {
        "svc-1": ServiceRecord(
            service_id="svc-1",
            service_name="订单中心",
            aliases=["订单", "order"],
            service_intro="查看与管理订单信息",
            route="/orders",
        ),
        "svc-2": ServiceRecord(
            service_id="svc-2",
            service_name="用户中心",
            aliases=["用户", "customer"],
            service_intro="查看用户画像",
            route="/users",
        ),
        "svc-3": ServiceRecord(
            service_id="svc-3",
            service_name="风控平台",
            aliases=["风控", "risk"],
            service_intro="风险决策管理",
            route="/risk/decision",
        ),
    }


class IntentRouterTests(unittest.TestCase):
    def setUp(self):
        self.router = IntentRouter()
        self.services = _services_dict()

    def test_navigational_name_exact_match(self):
        r = self.router.classify("订单中心", services=self.services)
        self.assertEqual(r.intent, NAVIGATIONAL)
        self.assertEqual(r.matched_service_id, "svc-1")

    def test_navigational_alias_exact_match(self):
        r = self.router.classify("订单", services=self.services)
        self.assertEqual(r.intent, NAVIGATIONAL)
        self.assertEqual(r.matched_service_id, "svc-1")

    def test_navigational_case_insensitive_and_whitespace(self):
        # service_name 大小写不敏感 + 前后空白剥离
        services = {
            "svc-x": ServiceRecord(
                service_id="svc-x",
                service_name="OrderCenter",
                aliases=[],
                service_intro="",
                route="/oc",
            )
        }
        r = self.router.classify("  ordercenter  ", services=services)
        self.assertEqual(r.intent, NAVIGATIONAL)
        self.assertEqual(r.matched_service_id, "svc-x")

    def test_multi_condition_connector_and_split(self):
        r = self.router.classify("开户 和 转账", services=self.services)
        self.assertEqual(r.intent, MULTI_CONDITION)
        self.assertEqual(r.sub_queries, ["开户", "转账"])

    def test_multi_condition_plus_connector(self):
        r = self.router.classify("订单+用户", services=self.services)
        self.assertEqual(r.intent, MULTI_CONDITION)
        self.assertEqual(r.sub_queries, ["订单", "用户"])

    def test_multi_condition_multiple_segments(self):
        r = self.router.classify("订单 和 用户 且 风控", services=self.services)
        self.assertEqual(r.intent, MULTI_CONDITION)
        self.assertEqual(r.sub_queries, ["订单", "用户", "风控"])

    def test_informational_hint(self):
        for q in ["怎么开户", "如何转账", "是什么", "为什么打不开", "哪个服务"]:
            r = self.router.classify(q, services=self.services)
            self.assertEqual(r.intent, INFORMATIONAL, f"query={q} got {r.intent}")

    def test_conversational_has_session(self):
        # 非精确命中、无连接词、无疑问词、有会话上下文 → conversational
        r = self.router.classify("还想要的那个", services=self.services, has_session=True)
        self.assertEqual(r.intent, CONVERSATIONAL)

    def test_default(self):
        r = self.router.classify("订单管理", services=self.services)
        self.assertEqual(r.intent, DEFAULT)
        self.assertEqual(r.matched_service_id, None)
        self.assertEqual(r.sub_queries, [])

    def test_empty_query(self):
        r = self.router.classify("   ", services=self.services)
        self.assertEqual(r.intent, DEFAULT)
        self.assertEqual(r.raw_query, "")

    def test_priority_navigational_over_multi(self):
        # 服务名本身含连接词：精确命中优先于 multi_condition
        services = {
            "svc-special": ServiceRecord(
                service_id="svc-special",
                service_name="风控+审计",
                aliases=[],
                service_intro="",
                route="/risk-audit",
            )
        }
        r = self.router.classify("风控+审计", services=services)
        self.assertEqual(r.intent, NAVIGATIONAL)
        self.assertEqual(r.matched_service_id, "svc-special")

    def test_priority_multi_over_conversational(self):
        # 有会话但含连接词 → multi_condition（plan 优先级：multi 在 conversational 之前）
        r = self.router.classify(
            "订单 和 用户", services=self.services, has_session=True
        )
        self.assertEqual(r.intent, MULTI_CONDITION)

    def test_no_services_skips_navigational(self):
        r = self.router.classify("订单中心", services=None)
        # 无 KB 时无法精确命中，"订单中心" 无连接词/疑问词 → default
        self.assertEqual(r.intent, DEFAULT)


class EngineIntentTests(unittest.TestCase):
    """engine.classify_intent + navigational 直达置顶。"""

    def setUp(self):
        self.engine = make_engine()
        self.engine.load_knowledge_base(SERVICES)

    def test_classify_intent_navigational(self):
        r = self.engine.classify_intent("订单中心")
        self.assertEqual(r.intent, NAVIGATIONAL)
        self.assertEqual(r.matched_service_id, "svc-1")

    def test_classify_intent_multi_condition(self):
        r = self.engine.classify_intent("订单 和 用户")
        self.assertEqual(r.intent, MULTI_CONDITION)
        self.assertEqual(r.sub_queries, ["订单", "用户"])

    def test_classify_intent_default(self):
        r = self.engine.classify_intent("订单管理")
        self.assertEqual(r.intent, DEFAULT)

    def test_navigational_pins_exact_name_to_top(self):
        results = self.engine.search("u-nav", "订单中心")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["service_id"], "svc-1")

    def test_navigational_pins_alias_to_top(self):
        results = self.engine.search("u-nav2", "订单")
        self.assertTrue(len(results) > 0)
        self.assertEqual(results[0]["service_id"], "svc-1")

    def test_navigational_pin_prepend_when_not_in_results(self):
        """命中服务未进 Top-10 时，_pin_navigational_to_top 用 KB 记录前置直达。"""
        engine = make_engine()
        engine.load_knowledge_base(SERVICES)
        # 模拟 final 未含 svc-2（用户中心）的场景
        final = [
            {
                "service_id": "svc-1",
                "service_name": "订单中心",
                "aliases": ["订单"],
                "service_intro": "...",
                "route": "/orders",
                "component": "OrderTable",
                "decision_button": "ApproveOrder",
                "derived": False,
                "score": 0.9,
                "rerank_reason": "高匹配",
            }
        ]
        pinned = engine._pin_navigational_to_top(final, "svc-2")
        self.assertEqual(len(pinned), 2)
        self.assertEqual(pinned[0]["service_id"], "svc-2")
        self.assertEqual(pinned[0]["service_name"], "用户中心")
        self.assertIn("直达", pinned[0].get("rerank_reason", ""))
        # 原结果保留下方
        self.assertEqual(pinned[1]["service_id"], "svc-1")

    def test_navigational_pin_moves_existing_to_top(self):
        """命中服务已在结果中（非首位）→ 移到 index 0，原因不变。"""
        engine = make_engine()
        engine.load_knowledge_base(SERVICES)
        final = [
            {
                "service_id": "svc-2",
                "service_name": "用户中心",
                "aliases": [],
                "service_intro": "",
                "route": "/users",
                "component": "UserProfile",
                "decision_button": "ConfirmUser",
                "derived": False,
                "score": 0.8,
                "rerank_reason": "原原因",
            },
            {
                "service_id": "svc-1",
                "service_name": "订单中心",
                "aliases": [],
                "service_intro": "",
                "route": "/orders",
                "component": "OrderTable",
                "decision_button": "ApproveOrder",
                "derived": False,
                "score": 0.9,
                "rerank_reason": "高匹配",
            },
        ]
        pinned = engine._pin_navigational_to_top(final, "svc-1")
        self.assertEqual(pinned[0]["service_id"], "svc-1")
        self.assertEqual(pinned[0]["rerank_reason"], "高匹配")  # 原因保留
        self.assertEqual(len(pinned), 2)

    def test_default_search_unchanged(self):
        """非 navigational 意图走正常检索链路，结果非空。"""
        results = self.engine.search("u-def", "订单管理")
        self.assertTrue(len(results) > 0)


if __name__ == "__main__":
    unittest.main()
