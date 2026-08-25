"""DCN v2 保底重排器测试。

覆盖：
- DCNReranker 启发式降级（未训练 / 无 torch）：复用 embedding cosine 的线性打分
- 特征构造、rerank 排序、空候选
- engine 离线搜索走 DCN 保底路径（无 API Key）
- train_dcn_reranker：样本不足返回 trained=False；torch 可用时端到端训练 + 落盘 + 加载
  （torch 不可用则跳过深度模型断言，仅验证不抛异常）

torch 为可选依赖：缺失时深度模型路径被 skip，启发式与集成测试仍全绿。
"""
import os
import tempfile
import unittest

from easysearch import DashScopeClient, ServiceSearchEngine, SQLiteStore, DCNReranker

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
]
VALID_IDS = {"svc-1", "svc-2", "svc-3"}

try:
    import torch  # noqa: F401
    from easysearch.dcn_v2 import DCN_V2  # 需 torch/numpy/pandas/sklearn 齐备
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


def _make_engine(services=SERVICES) -> ServiceSearchEngine:
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "t.db")
    client = DashScopeClient(api_key=None)  # 离线模式
    store = SQLiteStore(db)
    engine = ServiceSearchEngine(dashscope_client=client, store=store, db_path=db)
    if services:
        engine.load_knowledge_base(services)
    return engine


class DCNRerankerHeuristicTests(unittest.TestCase):
    """未训练状态下的启发式降级（不依赖 torch）。"""

    def setUp(self):
        self.engine = _make_engine()
        self.dcn = self.engine._dcn_reranker

    def tearDown(self):
        self.engine.store.close()

    def test_empty_candidates_returns_empty(self):
        self.assertEqual(self.dcn.rerank("订单", []), [])

    def test_rerank_sets_scores_and_sorts_desc(self):
        """启发式重排：每项附 rerank_score + dcn_score，按降序排列。"""
        candidates = [
            {"service_id": sid, "service_name": self.engine.services[sid].service_name,
             "aliases": list(self.engine.services[sid].aliases),
             "service_intro": self.engine.services[sid].service_intro, "score": 0.5}
            for sid in ("svc-1", "svc-2", "svc-3")
        ]
        ranked = self.dcn.rerank("订单", candidates)
        self.assertEqual(len(ranked), 3)
        scores = [x["rerank_score"] for x in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for item in ranked:
            self.assertIn("rerank_score", item)
            self.assertIn("dcn_score", item)

    def test_heuristic_prefers_semantic_match(self):
        """query 命中服务名（字面+语义）的候选应排在纯语义不匹配者之前。"""
        candidates = [
            {"service_id": "svc-2", "service_name": "用户中心",
             "aliases": ["用户"], "service_intro": "查看用户画像", "score": 0.4},
            {"service_id": "svc-1", "service_name": "订单中心",
             "aliases": ["订单"], "service_intro": "查看与管理订单信息", "score": 0.4},
        ]
        ranked = self.dcn.rerank("订单", candidates)
        # svc-1 名称/别名命中「订单」→ name_overlap 高，应排第一
        self.assertEqual(ranked[0]["service_id"], "svc-1")

    def test_available_false_before_training(self):
        """未训练时 available=False，走启发式。"""
        self.assertFalse(self.dcn.available)

    def test_rebuild_index_built_on_kb_load(self):
        """KB 加载后 service_id→idx 词表已构建。"""
        for sid in VALID_IDS:
            self.assertIn(sid, self.dcn.service_id_to_idx)

    def test_single_candidate_uses_heuristic(self):
        """单候选（batch=1）跳过 DCN forward，走启发式打分，不抛异常。"""
        candidates = [
            {"service_id": "svc-1", "service_name": "订单中心",
             "aliases": ["订单"], "service_intro": "查看与管理订单信息", "score": 0.6}
        ]
        ranked = self.dcn.rerank("订单", candidates)
        self.assertEqual(len(ranked), 1)
        self.assertIn("rerank_score", ranked[0])
        self.assertIn("dcn_score", ranked[0])
        self.assertGreaterEqual(ranked[0]["rerank_score"], 0.0)

    def test_rerank_preserves_original_fields(self):
        """重排结果保留候选原有的所有字段，仅追加 rerank_score/dcn_score。"""
        original = {
            "service_id": "svc-1", "service_name": "订单中心",
            "aliases": ["订单", "order"], "service_intro": "查看与管理订单信息",
            "score": 0.42, "route": "/orders", "component": "OrderTable",
        }
        ranked = self.dcn.rerank("订单", [original])
        self.assertEqual(ranked[0]["service_id"], "svc-1")
        self.assertEqual(ranked[0]["service_name"], "订单中心")
        self.assertEqual(ranked[0]["aliases"], ["订单", "order"])
        self.assertEqual(ranked[0]["score"], 0.42)
        self.assertEqual(ranked[0]["route"], "/orders")
        self.assertEqual(ranked[0]["component"], "OrderTable")

    def test_empty_query_does_not_crash(self):
        """空 query（无 token）时仍返回合法结果，启发式降级不依赖 query 命中。"""
        candidates = [
            {"service_id": "svc-1", "service_name": "订单中心",
             "aliases": [], "service_intro": "查看与管理订单信息", "score": 0.5},
            {"service_id": "svc-2", "service_name": "用户中心",
             "aliases": [], "service_intro": "查看用户画像", "score": 0.3},
        ]
        ranked = self.dcn.rerank("", candidates)
        self.assertEqual(len(ranked), 2)
        # 空 query 时 name_overlap/intro_overlap=0，排序主要靠 cosine+hybrid
        scores = [x["rerank_score"] for x in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_heuristic_is_deterministic(self):
        """相同输入两次重排结果一致（启发式为纯函数）。"""
        candidates = [
            {"service_id": sid, "service_name": self.engine.services[sid].service_name,
             "aliases": list(self.engine.services[sid].aliases),
             "service_intro": self.engine.services[sid].service_intro, "score": 0.5}
            for sid in ("svc-1", "svc-2", "svc-3")
        ]
        first = self.dcn.rerank("订单", candidates)
        second = self.dcn.rerank("订单", candidates)
        self.assertEqual(
            [x["rerank_score"] for x in first],
            [x["rerank_score"] for x in second],
        )
        self.assertEqual(
            [x["service_id"] for x in first],
            [x["service_id"] for x in second],
        )


class EngineDCNIntegrationTests(unittest.TestCase):
    """离线搜索（无 API Key）走 DCN 保底路径。"""

    def setUp(self):
        self.engine = _make_engine()

    def tearDown(self):
        self.engine.store.close()

    def test_offline_search_uses_dcn_fallback(self):
        """离线模式 qwen-rerank 不可用 → 走 DCN 保底，结果合法。"""
        results = self.engine.search("u1", "订单")
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        for item in results:
            self.assertIn(item["service_id"], VALID_IDS)
            self.assertTrue(item.get("rerank_reason"))

    def test_offline_search_async_uses_dcn_fallback(self):
        import asyncio
        results = asyncio.get_event_loop().run_until_complete(
            self.engine.search_async("u1", "用户")
        )
        self.assertGreater(len(results), 0)
        for item in results:
            self.assertIn(item["service_id"], VALID_IDS)

    def test_dcn_reranker_attached(self):
        """engine 的 reranker 持有 DCN 保底器引用。"""
        self.assertIs(self.engine.reranker.dcn_reranker, self.engine._dcn_reranker)

    def test_train_insufficient_data_returns_not_trained(self):
        """无点击日志 → 训练返回 trained=False（不抛异常）。"""
        summary = self.engine.train_dcn_reranker()
        self.assertFalse(summary["trained"])
        self.assertTrue(summary["reason"])


@unittest.skipUnless(_HAS_TORCH, "torch unavailable: skip deep-model training path")
class DCNTrainAndLoadTests(unittest.TestCase):
    """torch 可用时：端到端训练 + 落盘 + 重载。"""

    def setUp(self):
        self.engine = _make_engine()
        # 构造足够正样本：每次搜索→点击同一服务，建立 (query, sid) 对
        for _ in range(12):
            self.engine.store.append_query("u1", "订单", 1.0)
            self.engine.store.append_click("u1", "svc-1", 2.0)
        for _ in range(8):
            self.engine.store.append_query("u2", "风控", 3.0)
            self.engine.store.append_click("u2", "svc-3", 4.0)

    def tearDown(self):
        self.engine.store.close()

    def test_train_succeeds_and_persists(self):
        summary = self.engine.train_dcn_reranker()
        self.assertTrue(summary["trained"], summary)
        self.assertGreater(summary["positives"], 0)
        self.assertTrue(self.engine.dcn_reranker_available)
        self.assertIsNotNone(summary["path"])
        self.assertTrue(os.path.exists(summary["path"]))

    def test_trained_model_used_in_rerank(self):
        """训练后 rerank 走 DCN_V2 forward（available=True），结果仍合法排序。"""
        self.engine.train_dcn_reranker()
        self.assertTrue(self.engine.dcn_reranker_available)
        candidates = [
            {"service_id": sid, "service_name": self.engine.services[sid].service_name,
             "aliases": list(self.engine.services[sid].aliases),
             "service_intro": self.engine.services[sid].service_intro, "score": 0.5}
            for sid in ("svc-1", "svc-2", "svc-3")
        ]
        ranked = self.engine._dcn_reranker.rerank("订单", candidates)
        self.assertEqual(len(ranked), 3)
        scores = [x["rerank_score"] for x in ranked]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_weights_reloaded_on_kb_reload(self):
        """同库重新加载命中磁盘权重（try_load 成功）。"""
        self.engine.train_dcn_reranker()
        path = self.engine._dcn_reranker._model_path()
        self.assertTrue(path and os.path.exists(path))
        # 重建 engine（同 db_path → 同 kb_hash → 同 model_dir）
        db_path = self.engine.store.db_path
        client = DashScopeClient(api_key=None)
        store = SQLiteStore(db_path)
        engine2 = ServiceSearchEngine(dashscope_client=client, store=store, db_path=db_path)
        try:
            engine2.load_knowledge_base(SERVICES)
            self.assertTrue(engine2.dcn_reranker_available)
        finally:
            engine2.store.close()


if __name__ == "__main__":
    unittest.main()
