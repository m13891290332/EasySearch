"""协同过滤「猜你想用」推荐测试。

覆盖：
- store.collaborative_filter：冷启动 / 共现推荐 / 排除已点过
- engine.guess_you_like：CF → 内容相关 → 热门 三级降级链
- engine.homepage_dropdown：recommended_services 字段 + 四列去重
- /api/dropdown：返回 recommended_services
"""
import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from api.main import app, reset_engine
from easysearch import DashScopeClient, ServiceSearchEngine, SQLiteStore

SERVICES = [
    {
        "service_id": "svc-1",
        "service_name": "订单中心",
        "aliases": ["订单", "order"],
        "service_intro": "查看与管理订单信息",
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
    {
        "service_id": "svc-4",
        "service_name": "审批中心",
        "aliases": ["审批", "approval"],
        "service_intro": "统一审批流",
        "route": "/approval/center",
    },
    {
        "service_id": "svc-5",
        "service_name": "报表平台",
        "aliases": ["报表", "report"],
        "service_intro": "数据报表分析",
        "route": "/report/view",
    },
]
VALID_IDS = {s["service_id"] for s in SERVICES}


def _make_engine(services=SERVICES) -> ServiceSearchEngine:
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "t.db")
    client = DashScopeClient(api_key=None)  # 离线模式
    store = SQLiteStore(db)
    engine = ServiceSearchEngine(dashscope_client=client, store=store, db_path=db)
    if services:
        engine.load_knowledge_base(services)
    return engine


class CollaborativeFilterStoreTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.mkdtemp()
        self.db = os.path.join(tmp, "t.db")
        self.store = SQLiteStore(self.db)

    def tearDown(self):
        self.store.close()

    def _click(self, user, sid, ts):
        self.store.append_click(user, sid, ts)

    def test_cold_start_user_returns_empty(self):
        """无点击记录的用户 → 空列表（由上层降级）。"""
        self.assertEqual(self.store.collaborative_filter("nobody", limit=3), [])

    def test_co_occurrence_recommend(self):
        """u1 点 svc-1/svc-2；u2 共点 svc-1+svc-3+svc-4；u3 共点 svc-2+svc-3+svc-5。

        u1 的推荐应排除自身已点的 svc-1/svc-2，svc-3 被两位用户共点 → co_count 最高排首位。
        """
        self._click("u1", "svc-1", 1.0)
        self._click("u1", "svc-2", 2.0)
        self._click("u2", "svc-1", 3.0)
        self._click("u2", "svc-3", 4.0)
        self._click("u2", "svc-4", 5.0)
        self._click("u3", "svc-2", 6.0)
        self._click("u3", "svc-3", 7.0)
        self._click("u3", "svc-5", 8.0)
        rec = self.store.collaborative_filter("u1", limit=3)
        # svc-3 被 u2/u3 共点（co_count=2）应排第一
        self.assertEqual(rec[0], "svc-3")
        # 不含 u1 已点过的服务
        self.assertNotIn("svc-1", rec)
        self.assertNotIn("svc-2", rec)
        self.assertEqual(len(rec), 3)

    def test_exclude_ids_filtered(self):
        """exclude_ids 显式排除的候选不出现在结果中。"""
        self._click("u1", "svc-1", 1.0)
        self._click("u2", "svc-1", 2.0)
        self._click("u2", "svc-3", 3.0)
        rec = self.store.collaborative_filter("u1", exclude_ids={"svc-3"}, limit=3)
        self.assertNotIn("svc-3", rec)

    def test_deprecated_clicks_excluded(self):
        """deprecated=1 的点击不参与 CF 建模（不作为种子也不作为候选）。"""
        self.store.append_click("u1", "svc-1", 1.0, deprecated=False)
        self.store.append_click("u1", "svc-down", 2.0, deprecated=True)
        # 仅有一条非下线点击，且无其他用户共点 → 空
        rec = self.store.collaborative_filter("u1", limit=3)
        self.assertEqual(rec, [])


class GuessYouLikeTests(unittest.TestCase):
    def setUp(self):
        self.engine = _make_engine()

    def tearDown(self):
        self.engine.store.close()

    def test_cold_start_falls_back_to_hot(self):
        """新用户无点击：CF 空 → 内容相关空 → 热门兜底，返回 ≤3 且在 KB 内。"""
        # 制造一些全局点击形成热度
        for i in range(5):
            self.engine.store.append_click("other", "svc-3", float(i))
        rec = self.engine.guess_you_like("new_user", limit=3)
        self.assertLessEqual(len(rec), 3)
        for item in rec:
            self.assertIn(item["service_id"], VALID_IDS)
            self.assertEqual(item["service_name"], self.engine.services[item["service_id"]].service_name)

    def test_cf_recommend_excludes_exclude_ids(self):
        """有共现数据时 CF 推荐，且 exclude_ids 被过滤。"""
        self.engine.store.append_click("u1", "svc-1", 1.0)
        self.engine.store.append_click("u1", "svc-2", 2.0)
        self.engine.store.append_click("u2", "svc-1", 3.0)
        self.engine.store.append_click("u2", "svc-3", 4.0)
        rec = self.engine.guess_you_like("u1", exclude_ids={"svc-1", "svc-2"}, limit=3)
        ids = [x["service_id"] for x in rec]
        self.assertIn("svc-3", ids)
        self.assertNotIn("svc-1", ids)
        self.assertNotIn("svc-2", ids)

    def test_limit_respected(self):
        rec = self.engine.guess_you_like("any", limit=2)
        self.assertLessEqual(len(rec), 2)

    def test_all_services_excluded_returns_empty(self):
        """所有 KB 服务都在 exclude_ids 中 → 返回空列表（不抛异常）。"""
        rec = self.engine.guess_you_like(
            "u1", exclude_ids=set(VALID_IDS), limit=3
        )
        self.assertEqual(rec, [])

    def test_empty_kb_returns_empty(self):
        """未加载任何服务 → guess_you_like 返回空列表（不抛异常）。"""
        engine = _make_engine(services=None)
        try:
            rec = engine.guess_you_like("u1", limit=3)
            self.assertEqual(rec, [])
        finally:
            engine.store.close()

    def test_recommend_items_have_name_matching_kb(self):
        """返回的 service_name 与 KB 中实际服务名一致。"""
        for i in range(5):
            self.engine.store.append_click("other", "svc-3", float(i))
        rec = self.engine.guess_you_like("new_user", limit=3)
        for item in rec:
            sid = item["service_id"]
            self.assertEqual(
                item["service_name"], self.engine.services[sid].service_name
            )


class HomepageDropdownTests(unittest.TestCase):
    def setUp(self):
        self.engine = _make_engine()

    def tearDown(self):
        self.engine.store.close()

    def test_dropdown_has_recommended_field(self):
        """homepage_dropdown 返回 recommended_services 字段。"""
        data = self.engine.homepage_dropdown("u1")
        self.assertIn("recommended_services", data)
        self.assertIsInstance(data["recommended_services"], list)

    def test_recommended_no_overlap_with_recent_and_hot(self):
        """猜你想用与最近点击 / 热门服务四列去重。"""
        self.engine.store.append_click("u1", "svc-1", 1.0)
        self.engine.store.append_click("u1", "svc-2", 2.0)
        self.engine.store.append_click("other", "svc-3", 3.0)
        data = self.engine.homepage_dropdown("u1")
        recent_ids = {x["service_id"] for x in data["recent_clicked_services"]}
        hot_ids = {x["service_id"] for x in data["global_hot_services"]}
        rec_ids = {x["service_id"] for x in data["recommended_services"]}
        self.assertEqual(rec_ids & recent_ids, set())
        self.assertEqual(rec_ids & hot_ids, set())

    def test_recommended_at_most_three(self):
        data = self.engine.homepage_dropdown("u1")
        self.assertLessEqual(len(data["recommended_services"]), 3)


class DropdownAPITests(unittest.TestCase):
    def setUp(self):
        self.engine = _make_engine()
        reset_engine(self.engine)
        self.client = TestClient(app)

    def tearDown(self):
        self.engine.store.close()

    def test_dropdown_returns_recommended_services(self):
        r = self.client.get("/api/dropdown?user_id=u1")
        self.assertEqual(r.status_code, 200, r.text)
        body = r.json()
        self.assertIn("recommended_services", body)
        self.assertIsInstance(body["recommended_services"], list)
        for item in body["recommended_services"]:
            self.assertIn("service_id", item)
            self.assertIn("service_name", item)
            self.assertIn(item["service_id"], VALID_IDS)


if __name__ == "__main__":
    unittest.main()
