"""M8 页面内组件与执行测试。

覆盖：
  - ServiceRecord.from_dict 解析 components（含清洗 / 旧 KB 向后兼容 / 非法项过滤）。
  - engine 搜索结果项与 get_service 携带 components。
  - engine.execute_component_action 白名单校验（命中 / 未命中 / 未知服务）。
  - API /api/action/execute 打桩端点（ok+echo / 404 / 400）。
  - /api/service 返回 components。
"""
from __future__ import annotations

import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from api.main import app, reset_engine
from easysearch import (
    DashScopeClient,
    DeepSeekClient,
    ServiceRecord,
    ServiceSearchEngine,
    SQLiteStore,
)

# 含 components 的知识库（混合 dict/string route）
KB = [
    {
        "service_id": "svc-order",
        "service_name": "订单中心",
        "aliases": ["订单", "order"],
        "service_intro": "查看与管理订单信息，支持订单审批与导出",
        "route": {
            "path": "/orders",
            "component": "OrderTable",
            "action_button": "ApproveOrder",
        },
        "components": [
            {"name": "审批订单", "action": "approve", "params": {"order_id": "O-001"}},
            {"name": "导出订单", "action": "export"},
        ],
    },
    {
        "service_id": "svc-user",
        "service_name": "用户中心",
        "aliases": ["用户", "customer"],
        "service_intro": "查看用户画像",
        "route": "/users",
        # 无 components 字段（向后兼容）
    },
    {
        "service_id": "svc-bad",
        "service_name": "异常组件服务",
        "aliases": ["异常"],
        "service_intro": "用于校验非法 components 过滤",
        "route": "/bad",
        "components": [
            {"name": "缺动作", "action": ""},  # action 空 → 过滤
            {"name": "", "action": "missing_name"},  # name 空 → 过滤
            "not-a-dict",  # 非 dict → 过滤
            {"name": "合法动作", "action": "valid_act"},
            {"action": "no_name"},  # 无 name → 过滤
        ],
    },
]


def _make_engine(db_path: str = ":memory:") -> ServiceSearchEngine:
    return ServiceSearchEngine(
        dashscope_client=DashScopeClient(api_key=None),
        deepseek_client=DeepSeekClient(api_key=None),
        store=SQLiteStore(db_path),
        db_path=db_path,
    )


class ComponentSchemaTests(unittest.TestCase):
    """ServiceRecord.from_dict 解析与清洗。"""

    def test_parse_components_with_params(self):
        rec = ServiceRecord.from_dict(KB[0])
        comps = rec.components
        self.assertEqual(len(comps), 2)
        self.assertEqual(comps[0]["name"], "审批订单")
        self.assertEqual(comps[0]["action"], "approve")
        self.assertEqual(comps[0]["params"], {"order_id": "O-001"})
        self.assertNotIn("params", comps[1])  # 未传 params 不带键

    def test_missing_components_defaults_empty(self):
        rec = ServiceRecord.from_dict(KB[1])
        self.assertEqual(rec.components, [])

    def test_invalid_components_filtered(self):
        rec = ServiceRecord.from_dict(KB[2])
        # 仅 {合法动作, valid_act} 存活
        self.assertEqual(len(rec.components), 1)
        self.assertEqual(rec.components[0]["name"], "合法动作")
        self.assertEqual(rec.components[0]["action"], "valid_act")

    def test_non_list_components_becomes_empty(self):
        payload = dict(KB[0])
        payload["components"] = {"name": "x", "action": "y"}  # 非 list
        rec = ServiceRecord.from_dict(payload)
        self.assertEqual(rec.components, [])

    def test_params_non_dict_dropped(self):
        payload = dict(KB[0])
        payload["components"] = [
            {"name": "审批", "action": "approve", "params": "not-a-dict"},
        ]
        rec = ServiceRecord.from_dict(payload)
        self.assertEqual(len(rec.components), 1)
        self.assertNotIn("params", rec.components[0])


class EngineComponentsTests(unittest.TestCase):
    """engine 搜索结果 / get_service / execute_component_action。"""

    def setUp(self):
        self.engine = _make_engine()
        self.engine.load_knowledge_base(KB)

    def test_search_results_carry_components(self):
        results = self.engine.search("u1", "订单")
        self.assertTrue(results)
        order = next(r for r in results if r["service_id"] == "svc-order")
        self.assertIn("components", order)
        self.assertEqual(len(order["components"]), 2)
        self.assertEqual(order["components"][0]["name"], "审批订单")

    def test_old_kb_results_components_empty(self):
        results = self.engine.search("u1", "用户")
        self.assertTrue(results)
        user = next(r for r in results if r["service_id"] == "svc-user")
        self.assertEqual(user.get("components", "MISSING"), [])

    def test_get_service_returns_components(self):
        item = self.engine.get_service("svc-order")
        self.assertIsNotNone(item)
        self.assertIn("components", item)
        self.assertEqual(len(item["components"]), 2)

    def test_execute_whitelist_hit_returns_echo(self):
        r = self.engine.execute_component_action(
            "svc-order", "审批订单", "approve", {"order_id": "O-001"}
        )
        self.assertIsNotNone(r)
        self.assertTrue(r["ok"])
        self.assertEqual(r["service_id"], "svc-order")
        self.assertEqual(r["component"], "审批订单")
        self.assertEqual(r["action"], "approve")
        self.assertEqual(r["echo"]["params"], {"order_id": "O-001"})

    def test_execute_action_without_params(self):
        r = self.engine.execute_component_action("svc-order", "导出订单", "export")
        self.assertIsNotNone(r)
        self.assertTrue(r["ok"])
        self.assertNotIn("params", r["echo"])

    def test_execute_non_whitelisted_action_returns_none(self):
        # component 命中但 action 不匹配 → None（白名单校验 component+action 同时命中）
        r = self.engine.execute_component_action(
            "svc-order", "审批订单", "export"
        )
        self.assertIsNone(r)

    def test_execute_unknown_component_returns_none(self):
        r = self.engine.execute_component_action(
            "svc-order", "不存在组件", "approve"
        )
        self.assertIsNone(r)

    def test_execute_unknown_service_returns_none(self):
        r = self.engine.execute_component_action("nope", "x", "y")
        self.assertIsNone(r)


class APIActionExecuteTests(unittest.TestCase):
    """/api/action/execute 打桩端点。"""

    def setUp(self):
        tmp = tempfile.mkdtemp()
        self.db = os.path.join(tmp, "t.db")
        client = DashScopeClient(api_key=None)
        store = SQLiteStore(self.db)
        self.engine = ServiceSearchEngine(
            dashscope_client=client,
            deepseek_client=DeepSeekClient(api_key=None),
            store=store,
            db_path=self.db,
        )
        self.engine.load_knowledge_base(KB)
        reset_engine(self.engine)
        self.client = TestClient(app)

    def tearDown(self):
        self.engine.store.close()

    def test_execute_success(self):
        r = self.client.post(
            "/api/action/execute",
            json={
                "user_id": "u1",
                "service_id": "svc-order",
                "component": "审批订单",
                "action": "approve",
                "params": {"order_id": "O-001"},
            },
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["ok"])
        self.assertEqual(body["service_id"], "svc-order")
        self.assertEqual(body["component"], "审批订单")
        self.assertEqual(body["action"], "approve")
        self.assertEqual(body["echo"]["params"], {"order_id": "O-001"})

    def test_execute_no_params(self):
        r = self.client.post(
            "/api/action/execute",
            json={
                "user_id": "u1",
                "service_id": "svc-order",
                "component": "导出订单",
                "action": "export",
            },
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertTrue(body["ok"])

    def test_execute_unknown_service_404(self):
        r = self.client.post(
            "/api/action/execute",
            json={
                "user_id": "u1",
                "service_id": "nope",
                "component": "审批订单",
                "action": "approve",
            },
        )
        self.assertEqual(r.status_code, 404)

    def test_execute_non_whitelisted_action_400(self):
        r = self.client.post(
            "/api/action/execute",
            json={
                "user_id": "u1",
                "service_id": "svc-order",
                "component": "审批订单",
                "action": "export",  # component 命中但 action 不在白名单
            },
        )
        self.assertEqual(r.status_code, 400)

    def test_search_response_has_components(self):
        r = self.client.get(
            "/api/search", params={"user_id": "u1", "query": "订单"}
        )
        self.assertEqual(r.status_code, 200)
        first = r.json()["results"][0]
        self.assertIn("components", first)
        self.assertTrue(first["components"])

    def test_service_detail_has_components(self):
        r = self.client.get(
            "/api/service", params={"service_id": "svc-order"}
        )
        self.assertEqual(r.status_code, 200)
        self.assertIn("components", r.json())
        self.assertEqual(len(r.json()["components"]), 2)


if __name__ == "__main__":
    unittest.main()
