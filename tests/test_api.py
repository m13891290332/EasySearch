import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from api.main import app, reset_engine
from easysearch import DashScopeClient, ServiceSearchEngine, SQLiteStore

# 内联知识库（避免跨测试模块导入在 unittest discover 下的路径脆弱性）
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


def _setup_engine() -> ServiceSearchEngine:
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "t.db")
    client = DashScopeClient(api_key=None)  # 离线模式
    store = SQLiteStore(db)
    # db_path 必须透传：否则 engine 默认 db_path="data/easysearch.db"，
    # _embeddings_dir 指向 data/embeddings/，M4 .npz 持久化跨测试/污染应用数据。
    engine = ServiceSearchEngine(dashscope_client=client, store=store, db_path=db)
    engine.load_knowledge_base(SERVICES)
    reset_engine(engine)
    return engine


class APITests(unittest.TestCase):
    def setUp(self):
        self.engine = _setup_engine()
        self.client = TestClient(app)

    def tearDown(self):
        self.engine.store.close()

    def test_health(self):
        r = self.client.get("/api/health")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["status"], "ok")
        self.assertEqual(body["services_count"], len(SERVICES))
        self.assertFalse(body["dashscope_enabled"])  # 离线模式

    def test_search_and_click_and_dropdown(self):
        r = self.client.get(
            "/api/search", params={"user_id": "u1", "query": "订单"}
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertTrue(data["results"])
        first = data["results"][0]
        for key in ("route", "component", "decision_button", "rerank_reason"):
            self.assertIn(key, first)

        sid = first["service_id"]
        r2 = self.client.post(
            "/api/click", json={"user_id": "u1", "service_id": sid}
        )
        self.assertEqual(r2.status_code, 200)

        r3 = self.client.get("/api/dropdown", params={"user_id": "u1"})
        self.assertEqual(r3.status_code, 200)
        dropdown = r3.json()
        self.assertIn("recent_queries", dropdown)
        self.assertEqual(dropdown["recent_queries"], ["订单"])
        # global_hot 返回对象数组 [{service_id, service_name}]
        self.assertEqual(
            [item["service_name"] for item in dropdown["global_hot_services"]],
            [first["service_name"]],
        )
        self.assertEqual(dropdown["global_hot_services"][0]["service_id"], sid)

        # /api/service 单服务详情
        r4 = self.client.get("/api/service", params={"service_id": sid})
        self.assertEqual(r4.status_code, 200)
        sd = r4.json()
        self.assertEqual(sd["service_id"], sid)
        for key in ("route", "component", "decision_button"):
            self.assertIn(key, sd)

        # /api/service 未知 404
        r5 = self.client.get("/api/service", params={"service_id": "nope"})
        self.assertEqual(r5.status_code, 404)

    def test_search_empty_query_rejected(self):
        r = self.client.get("/api/search", params={"user_id": "u1", "query": "  "})
        self.assertEqual(r.status_code, 400)

    def test_click_unknown_service_returns_200_with_deprecated(self):
        """M12：已下线服务仍记点击（标 deprecated），不硬 404。"""
        r = self.client.post(
            "/api/click", json={"user_id": "u1", "service_id": "nope"}
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})

    def test_upload_knowledge_base(self):
        r = self.client.post("/api/knowledge-base/upload", json=SERVICES)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["services_count"], len(SERVICES))

    def test_homepage_served(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
