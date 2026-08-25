"""相关服务离线预计算（engine._build_related_services / get_related_services）测试。

预计算在 KB 加载时完成（embedding cosine top-3，排除自身），落盘
``related_{kb_hash}.json``（:memory: 测试库不落盘，仅内存 dict）。
进入路由占位视图时 ``get_related_services`` O(1) 查表返回 top-k 详情，
未预计算则即时兜底计算。

离线模式下 embedding 走 local_hash_vector 确定性 fallback，cosine 仍可计算。
"""
import os
import tempfile
import unittest

from easysearch import DashScopeClient, ServiceSearchEngine, SQLiteStore

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

VALID_IDS = {"svc-1", "svc-2", "svc-3"}

DETAIL_FIELDS = {
    "service_id", "service_name", "aliases", "service_intro",
    "route", "component", "decision_button", "derived", "components",
}


def _make_engine(services=SERVICES, db_path: str | None = None) -> ServiceSearchEngine:
    if db_path is None:
        tmp = tempfile.mkdtemp()
        db_path = os.path.join(tmp, "t.db")
    client = DashScopeClient(api_key=None)  # 离线模式
    store = SQLiteStore(db_path)
    # db_path 必须透传，避免持久化目录污染应用数据
    engine = ServiceSearchEngine(dashscope_client=client, store=store, db_path=db_path)
    if services:
        engine.load_knowledge_base(services)
    return engine


class BuildRelatedServicesTests(unittest.TestCase):
    def setUp(self):
        self.engine = _make_engine()

    def tearDown(self):
        self.engine.store.close()

    def test_precomputed_on_load(self):
        """KB 加载后 _related_services 已为每个服务预计算（非空，除自身外有候选）。"""
        for sid in VALID_IDS:
            self.assertIn(sid, self.engine._related_services)
            ids = self.engine._related_services[sid]
            # 3 个服务时，每个的 top-3 应含其余 2 个（排除自身）
            self.assertNotIn(sid, ids)
            self.assertGreaterEqual(len(ids), 1)

    def test_precomputed_excludes_self(self):
        """预计算结果不含自身（cosine 自相似恒为 1.0，必须排除）。"""
        for sid, related in self.engine._related_services.items():
            self.assertNotIn(sid, related)

    def test_precomputed_ids_in_kb(self):
        """预计算 ID 均在 KB 内（无悬挂引用）。"""
        for related in self.engine._related_services.values():
            for rid in related:
                self.assertIn(rid, VALID_IDS)

    def test_persistence_file_created(self):
        """非 :memory: 库加载后落盘 related_{kb_hash}.json。"""
        self.assertTrue(self.engine.kb_hash, "kb_hash 应在加载后设置")
        self.assertIsNotNone(self.engine._related_dir)
        path = os.path.join(
            self.engine._related_dir, f"related_{self.engine.kb_hash}.json"
        )
        self.assertTrue(os.path.exists(path), f"预计算文件应落盘：{path}")

    def test_reload_uses_persisted_cache(self):
        """同库同目录重新加载命中磁盘缓存（写 sentinel 验证非重算）。"""
        import json
        kb_hash = self.engine.kb_hash
        path = os.path.join(
            self.engine._related_dir, f"related_{kb_hash}.json"
        )
        self.assertTrue(os.path.exists(path))
        # 写入 sentinel：若 engine2 重算则不会出现该结构，只有读盘才命中
        sentinel = {"svc-1": ["svc-2", "svc-3"], "svc-2": ["svc-1"], "svc-3": ["svc-1"]}
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(sentinel, fp, ensure_ascii=False)
        # 同 db_path 构造 engine2（共享 _related_dir + kb_hash）
        engine2 = _make_engine(db_path=self.engine.store.db_path)
        try:
            self.assertEqual(engine2.kb_hash, kb_hash)
            # 命中磁盘缓存 → 与 sentinel 完全一致（非重算结果）
            self.assertEqual(engine2._related_services, sentinel)
        finally:
            engine2.store.close()


class GetRelatedServicesTests(unittest.TestCase):
    def setUp(self):
        self.engine = _make_engine()

    def tearDown(self):
        self.engine.store.close()

    def test_returns_at_most_k(self):
        """k 参数生效：k=1 → ≤1，k=2 → ≤2。"""
        self.assertLessEqual(len(self.engine.get_related_services("svc-1", k=1)), 1)
        self.assertLessEqual(len(self.engine.get_related_services("svc-1", k=2)), 2)

    def test_default_k_is_3(self):
        """默认 k=3。"""
        items = self.engine.get_related_services("svc-1")
        self.assertLessEqual(len(items), 3)

    def test_excludes_self(self):
        """返回项不含 service_id 自身。"""
        for sid in VALID_IDS:
            items = self.engine.get_related_services(sid)
            for it in items:
                self.assertNotEqual(it["service_id"], sid)

    def test_nonexistent_service_returns_empty(self):
        """不存在的 service_id → []。"""
        self.assertEqual(self.engine.get_related_services("nope"), [])

    def test_item_structure(self):
        """返回项为 ServiceDetail 同构 dict（含全部详情字段，id 在 KB 内）。"""
        items = self.engine.get_related_services("svc-1")
        self.assertTrue(items)
        for it in items:
            self.assertTrue(DETAIL_FIELDS.issubset(it.keys()), f"缺字段：{it.keys()}")
            self.assertIn(it["service_id"], VALID_IDS)
            self.assertIsInstance(it["service_name"], str)
            self.assertIsInstance(it["aliases"], list)
            self.assertIsInstance(it["route"], str)

    def test_returns_distinct_services(self):
        """返回项互不相同（无重复 service_id）。"""
        for sid in VALID_IDS:
            items = self.engine.get_related_services(sid)
            ids = [it["service_id"] for it in items]
            self.assertEqual(len(ids), len(set(ids)))


class SingleServiceKBTests(unittest.TestCase):
    def setUp(self):
        self.engine = _make_engine(services=[SERVICES[0]])

    def tearDown(self):
        self.engine.store.close()

    def test_single_service_no_related(self):
        """KB 仅 1 个服务时无相关服务（排除自身后无候选）→ []。"""
        self.assertEqual(self.engine.get_related_services("svc-1"), [])
        self.assertEqual(self.engine._related_services.get("svc-1", []), [])

    def test_single_service_related_map_empty(self):
        """单服务库 _related_services[sid] 为空列表（无其他候选）。"""
        self.assertEqual(self.engine._related_services.get("svc-1"), [])


class FallbackTests(unittest.TestCase):
    """未预计算（清空 _related_services）时即时兜底计算。"""

    def setUp(self):
        self.engine = _make_engine()

    def tearDown(self):
        self.engine.store.close()

    def test_fallback_when_cache_cleared(self):
        """清空 _related_services 后 get_related_services 即时算兜底，仍非空。"""
        self.engine._related_services.clear()
        items = self.engine.get_related_services("svc-1")
        self.assertTrue(items)
        for it in items:
            self.assertNotEqual(it["service_id"], "svc-1")

    def test_fallback_respects_k(self):
        """即时兜底也尊重 k 参数。"""
        self.engine._related_services.clear()
        self.assertLessEqual(len(self.engine.get_related_services("svc-1", k=1)), 1)


class StaleCacheValidationTests(unittest.TestCase):
    """缓存中的 service_id 与当前 KB 不一致时，应丢弃缓存并重算。"""

    def setUp(self):
        self.engine = _make_engine()

    def tearDown(self):
        self.engine.store.close()

    def test_stale_cache_ids_recomputed(self):
        """缓存含已不存在的 service_id → 丢弃缓存重算，结果 ID 均在 KB 内。"""
        # 篡改内存缓存：注入一个不在 KB 中的悬挂 ID
        self.engine._related_services["svc-1"] = ["ghost-1", "ghost-2", "ghost-3"]
        items = self.engine.get_related_services("svc-1")
        # 全部 ghost 被过滤 → fallback 即时算 → 返回有效结果
        self.assertTrue(items, "悬挂 ID 全过滤后应 fallback 即时算")
        for it in items:
            self.assertIn(it["service_id"], VALID_IDS)

    def test_stale_json_cache_invalidated_on_load(self):
        """磁盘 JSON 缓存含不在 KB 的 ID → 加载时校验失败 → 重算。"""
        import json

        kb_hash = self.engine.kb_hash
        path = os.path.join(
            self.engine._related_dir, f"related_{kb_hash}.json"
        )
        self.assertTrue(os.path.exists(path))
        # 写入含悬挂 ID 的缓存
        stale = {"svc-1": ["ghost-x", "svc-2"], "svc-2": ["svc-1"], "svc-3": ["svc-1"]}
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(stale, fp, ensure_ascii=False)
        # 重新加载：ghost-x 不在 KB → 缓存校验失败 → 重算
        engine2 = _make_engine(db_path=self.engine.store.db_path)
        try:
            # 重算结果中不应出现 ghost-x
            for related in engine2._related_services.values():
                self.assertNotIn("ghost-x", related)
            # 结果应非空（3 服务 KB）
            self.assertTrue(engine2._related_services)
        finally:
            engine2.store.close()

    def test_partial_stale_ids_fallback(self):
        """预计算列表中部分 ID 失效 → 过滤后仍能返回有效结果。"""
        # svc-2/svc-3 仍在 KB，ghost 不在
        self.engine._related_services["svc-1"] = ["ghost", "svc-2", "svc-3"]
        items = self.engine.get_related_services("svc-1")
        # ghost 被过滤，svc-2/svc-3 有效 → 结果非空
        self.assertTrue(items)
        ids = {it["service_id"] for it in items}
        self.assertNotIn("ghost", ids)
        self.assertTrue(ids & {"svc-2", "svc-3"})


class MarkdownCleanOnKBLoadTests(unittest.TestCase):
    """KB 导入时 service_intro 中的 ## 标题标记应被 sanitize_text 剥除。"""

    def test_markdown_headers_stripped_on_load(self):
        """含 ## 标题的 service_intro 导入后被清洗，不含 # 标记。"""
        services_with_markdown = [
            {
                "service_id": "svc-md-1",
                "service_name": "行内转账",
                "aliases": ["内部转账", "同行转账"],
                "service_intro": (
                    "# 行内转账## 一、功能描述支持行内、跨行、跨境等多种转账方式，"
                    "提供实时到账与预约转账功能，确保资金流转高效安全。"
                    "## 二、使用方法1. 选择转账方式；2. 输入收款方信息；"
                    "3. 填写转账金额；4. 验证身份并提交；5. 查看转账结果。"
                ),
                "route": "/transfer/inline",
            },
            {
                "service_id": "svc-md-2",
                "service_name": "跨行转账",
                "aliases": ["跨行"],
                "service_intro": "### 三级标题内容### 另一个标题",
                "route": "/transfer/cross",
            },
        ]
        engine = _make_engine(services=services_with_markdown)
        try:
            for sid in ("svc-md-1", "svc-md-2"):
                svc = engine.services[sid]
                self.assertNotIn("#", svc.service_intro, (
                    f"{sid} 的 service_intro 仍含 # 标记：{svc.service_intro}"
                ))
                self.assertNotIn("##", svc.service_intro)
            # 相关服务仍正常计算（markdown 清洗不影响 embedding 流程）
            related = engine.get_related_services("svc-md-1")
            self.assertLessEqual(len(related), 1)  # 仅 2 服务，排除自身后 ≤1
        finally:
            engine.store.close()


if __name__ == "__main__":
    unittest.main()
