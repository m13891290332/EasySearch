import json
import os
import tempfile
import unittest
from unittest.mock import patch

from easysearch import DashScopeClient, DeepSeekClient, ServiceSearchEngine, SQLiteStore, route_info
from easysearch.utils import tokenize


# 混合 dict / string 两种 route 形态的知识库
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


def make_engine(api_key=None, requester=None, db_path=":memory:"):
    # dashscope 与 deepseek 共用同一 requester（测试按 URL 分流）
    client = DashScopeClient(api_key=api_key, requester=requester)
    ds_client = DeepSeekClient(api_key=api_key, requester=requester)
    store = SQLiteStore(db_path)
    # db_path 必须透传给 engine：否则 engine.__init__ 的 db_path 取默认值
    # "data/easysearch.db"，_embeddings_dir 指向 data/embeddings/，M4 .npz 持久化
    # 会被激活——跨测试共享同一 .npz，导致 DashScopePayloadTests 的 embedding
    # 调用被缓存跳过、calls 列表缺 KB 批量 embedding 项，any() 断言误触 rerank
    # payload 的 KeyError: 'texts'。传 db_path 使 :memory: 测试不落 .npz。
    return ServiceSearchEngine(
        dashscope_client=client, deepseek_client=ds_client, store=store, db_path=db_path
    )


class RouteInfoTests(unittest.TestCase):
    def test_string_route_derives_component_and_button(self):
        info = route_info("/go/account/open-account")
        self.assertEqual(info["route"], "/go/account/open-account")
        self.assertEqual(info["component"], "OpenAccount")
        self.assertEqual(info["decision_button"], "进入")
        self.assertTrue(info["derived"])

    def test_dict_route_extracts_original_fields(self):
        info = route_info(
            {"path": "/orders", "component": "OrderTable", "action_button": "ApproveOrder"}
        )
        self.assertEqual(info["route"], "/orders")
        self.assertEqual(info["component"], "OrderTable")
        self.assertEqual(info["decision_button"], "ApproveOrder")
        self.assertFalse(info["derived"])


class HybridScoreTests(unittest.TestCase):
    def test_formula_weights(self):
        # 0.6*0.8 + 0.3*0.5 + 0.1*1.0 = 0.48 + 0.15 + 0.1 = 0.73
        self.assertAlmostEqual(ServiceSearchEngine._hybrid_score(0.8, 0.5, 1.0), 0.73)


class TokenizeTests(unittest.TestCase):
    def test_chinese_segmented_into_words(self):
        tokens = tokenize("我来到北京清华大学")
        # jieba 应切出 北京 / 清华；未装 jieba 时回退为整串，子串仍命中
        self.assertTrue(any(("北京" in t) or ("清华" in t) for t in tokens))
        self.assertTrue(len(tokens) > 0)

    def test_punctuation_filtered(self):
        tokens = tokenize("# 开户 ## 一、功能")
        self.assertNotIn("#", tokens)
        self.assertTrue("开户" in tokens or any("开户" in t for t in tokens))


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "t.db")
        self.engine = make_engine(db_path=self.db)
        self.engine.load_knowledge_base(SERVICES)

    def tearDown(self):
        self.engine.store.close()

    def test_search_returns_clickable_fields_and_reason(self):
        results = self.engine.search("u-1", "订单审批")
        self.assertTrue(results)
        first = results[0]
        for key in ("route", "component", "decision_button", "rerank_reason", "score"):
            self.assertIn(key, first)
        # string route 的 svc-3 应被派生出 component/decision_button
        svc3 = next(r for r in results if r["service_id"] == "svc-3")
        self.assertEqual(svc3["component"], "Decision")
        self.assertEqual(svc3["decision_button"], "进入")
        self.assertTrue(svc3["derived"])

    def test_search_top10_limit(self):
        results = self.engine.search("u-1", "订单")
        self.assertLessEqual(len(results), 10)

    def test_homepage_dropdown_recent_and_hot(self):
        self.engine.search("u-1", "订单")
        self.engine.search("u-1", "用户")
        self.engine.search("u-1", "风控")

        self.engine.record_click("u-1", "svc-1")
        self.engine.record_click("u-1", "svc-2")
        self.engine.record_click("u-1", "svc-3")
        self.engine.record_click("u-2", "svc-3")

        dropdown = self.engine.homepage_dropdown("u-1")
        self.assertEqual(dropdown["recent_queries"], ["风控", "用户", "订单"])
        # 点击/热门返回 [{service_id, service_name}]，便于前端直接定位
        self.assertEqual(
            [item["service_name"] for item in dropdown["recent_clicked_services"]],
            ["风控平台", "用户中心", "订单中心"],
        )
        self.assertEqual(
            dropdown["recent_clicked_services"][0]["service_id"], "svc-3"
        )
        self.assertEqual(
            dropdown["global_hot_services"][0]["service_name"], "风控平台"
        )

    def test_dropdown_dedup_recent_queries(self):
        # 同一搜索词多次出现，下拉只保留最近一次（去重）
        for _ in range(3):
            self.engine.search("u-d", "开户")
        self.engine.search("u-d", "转账")
        dd = self.engine.homepage_dropdown("u-d")
        self.assertEqual(dd["recent_queries"], ["转账", "开户"])

    def test_dropdown_dedup_recent_clicks(self):
        # 同一服务多次点击，下拉只保留最近一次（去重）
        for _ in range(3):
            self.engine.record_click("u-c", "svc-1")
        self.engine.record_click("u-c", "svc-2")
        dd = self.engine.homepage_dropdown("u-c")
        ids = [item["service_id"] for item in dd["recent_clicked_services"]]
        self.assertEqual(ids, ["svc-2", "svc-1"])

    def test_get_service_returns_detail(self):
        detail = self.engine.get_service("svc-3")
        self.assertIsNotNone(detail)
        self.assertEqual(detail["service_id"], "svc-3")
        self.assertEqual(detail["component"], "Decision")
        self.assertEqual(detail["decision_button"], "进入")
        self.assertTrue(detail["derived"])

    def test_get_service_unknown_returns_none(self):
        self.assertIsNone(self.engine.get_service("nope"))

    def test_din_threshold_over_ten_triggers_without_error(self):
        for idx in range(12):
            results = self.engine.search("u-h", f"查询{idx}")
            self.assertLessEqual(len(results), 10)
        # 第 11 次搜索后 count=11 > 10，DIN 路径已触发
        self.assertEqual(self.engine.store.query_count("u-h"), 12)

    def test_persistence_across_store_instances(self):
        self.engine.search("u-p", "开户")
        self.engine.record_click("u-p", "svc-1")
        self.engine.store.close()

        store2 = SQLiteStore(self.db)
        self.assertEqual(store2.query_count("u-p"), 1)
        self.assertEqual(store2.recent_queries("u-p"), ["开户"])
        self.assertEqual(store2.recent_clicks("u-p"), ["svc-1"])
        self.assertEqual(store2.hot_services(1), ["svc-1"])
        store2.close()

    def test_record_click_unknown_service_marks_deprecated(self):
        """M12：已下线服务仍记点击（标 deprecated），不抛 ValueError。

        原行为：unknown service → raise ValueError（API 透 404）。
        M12 新行为：unknown service → 仍记 user_clicks(deprecated=1)，
        不污染 global_clicks 热度榜，不阻塞前端交互。
        """
        # 不抛 ValueError
        self.engine.record_click("u-1", "nope")
        # 验证 deprecated=1 标记
        with self.engine.store._lock:
            row = self.engine.store._conn.execute(
                "SELECT deprecated FROM user_clicks "
                "WHERE service_id=? ORDER BY id DESC LIMIT 1",
                ("nope",),
            ).fetchone()
        self.assertIsNotNone(row)
        self.assertEqual(int(row["deprecated"]), 1)
        # 已下线服务不污染 global_clicks
        self.assertNotIn("nope", self.engine.store.global_click_counter())


class DashScopePayloadTests(unittest.TestCase):
    def test_dashscope_payloads_used_when_api_key_exists(self):
        calls: list[tuple[str, dict, dict]] = []

        def requester(url: str, body: bytes, headers: dict) -> dict:
            payload = json.loads(body.decode("utf-8"))
            calls.append((url, payload, headers))
            if "text-embedding" in url:
                n = len(payload["input"]["texts"])
                return {
                    "output": {
                        "embeddings": [
                            {"embedding": [1.0, 0.0, 0.0]} for _ in range(n)
                        ]
                    }
                }
            if "text-rerank" in url:
                docs = payload["input"]["documents"]
                return {
                    "output": {
                        "results": [
                            {"index": i, "relevance_score": float(len(docs) - i)}
                            for i in range(len(docs))
                        ]
                    }
                }
            if "chat/completions" in url:
                return {
                    "choices": [
                        {
                            "message": {
                                "content": json.dumps(
                                    [
                                        {"service_id": "svc-1", "reason": "与订单语义最相关"},
                                        {"service_id": "svc-2", "reason": "与用户信息次相关"},
                                    ],
                                    ensure_ascii=False,
                                )
                            }
                        }
                    ]
                }
            raise AssertionError(f"Unexpected URL: {url}")

        client = DashScopeClient(api_key="placeholder-api-key", requester=requester)
        deepseek_client = DeepSeekClient(api_key="placeholder-api-key", requester=requester)
        # db_path=":memory:" 使 _embeddings_dir=None → 不加载/保存 .npz，
        # 保证每次 load_knowledge_base 都真实调用 embedding 远程接口（mock requester），
        # calls 列表含 KB 批量 embedding 项，any(len(texts)>1) 断言能短路命中。
        engine = ServiceSearchEngine(
            dashscope_client=client,
            deepseek_client=deepseek_client,
            store=SQLiteStore(":memory:"),
            db_path=":memory:",
        )
        engine.load_knowledge_base(SERVICES)
        # M2 后 REASON_ENABLED 默认 False、REASON_EFFORT 默认 low；本用例验证
        # "有 API Key 时 DashScope/DeepSeek 全链路被调用"，需临时开启 reason + high effort
        with patch("easysearch.reranker.REASON_ENABLED", True), \
             patch("easysearch.reranker.REASON_EFFORT", "high"):
            results = engine.search("u-api", "订单审批")

        self.assertTrue(results)
        self.assertTrue(any("text-embedding/text-embedding" in url for url, _, _ in calls))
        self.assertTrue(any("rerank/text-rerank/text-rerank" in url for url, _, _ in calls))
        # A7 修正：DeepSeek 实际 endpoint 是 api.deepseek.com/chat/completions（原断言漂移已修）
        self.assertTrue(any("api.deepseek.com/chat/completions" in url for url, _, _ in calls))
        embedding_call = next(
            payload for url, payload, _ in calls if "text-embedding/text-embedding" in url
        )
        self.assertEqual(embedding_call["model"], "qwen3.7-text-embedding")
        self.assertIn("texts", embedding_call["input"])
        # 批量 embed：服务装载时一次性传入多条
        self.assertTrue(any(len(p["input"]["texts"]) > 1 for _, p, _ in calls))
        # 模型名校验
        rerank_call = next(p for u, p, _ in calls if "text-rerank" in u)
        self.assertEqual(rerank_call["model"], "qwen3-vl-rerank")
        # reasoner 现用 deepseek-v4-flash（endpoint: api.deepseek.com/chat/completions）
        chat_call = next(p for u, p, _ in calls if "chat/completions" in u)
        self.assertEqual(chat_call["model"], "deepseek-v4-flash")
        self.assertIn("thinking", chat_call)
        self.assertEqual(chat_call.get("reasoning_effort"), "high")
        self.assertTrue(results[0]["rerank_reason"])
        engine.store.close()


# ============================================================
# A 组 + B 组新增测试（VectorIndex / MultiFieldBM25 / SynonymExpander
# / LevenshteinCorrector / MMRReranker / popularity_decayed）
# ============================================================

class VectorIndexTests(unittest.TestCase):
    """B2：FAISS IndexFlatIP 向量检索（无 faiss 时降级 Python 循环）。"""

    def test_build_and_score_all(self):
        from easysearch import VectorIndex

        idx = VectorIndex()
        idx.build({
            "a": [1.0, 0.0, 0.0],
            "b": [0.0, 1.0, 0.0],
            "c": [1.0, 1.0, 0.0],
        })
        scores = idx.score_all([1.0, 0.0, 0.0])
        # a 与 query 完全一致（cos=1），b 正交（cos=0），c 部分匹配
        self.assertAlmostEqual(scores["a"], 1.0, places=5)
        self.assertLess(scores["b"], 0.01)
        self.assertGreater(scores["c"], scores["b"])

    def test_search_top_k(self):
        from easysearch import VectorIndex

        idx = VectorIndex()
        idx.build({
            "a": [1.0, 0.0],
            "b": [0.9, 0.1],
            "c": [0.0, 1.0],
        })
        result = idx.search([1.0, 0.0], top_k=2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], "a")  # 最相似
        self.assertGreater(result[0][1], result[1][1])

    def test_get_and_contains(self):
        from easysearch import VectorIndex

        idx = VectorIndex()
        idx.build({"a": [1.0, 0.0]})
        self.assertEqual(idx.get("a"), [1.0, 0.0])
        self.assertIn("a", idx)
        self.assertNotIn("b", idx)
        self.assertEqual(len(idx), 1)

    def test_empty_build(self):
        from easysearch import VectorIndex

        idx = VectorIndex()
        idx.build({})
        self.assertEqual(idx.score_all([1.0]), {})
        self.assertEqual(len(idx), 0)


class MultiFieldBM25Tests(unittest.TestCase):
    """A2 + C3：多字段加权 BM25 + 一次 tokenize。"""

    def test_field_weighting_name_dominates(self):
        from easysearch import MultiFieldBM25Index

        idx = MultiFieldBM25Index()
        idx.build({
            "svc-1": {"name": "订单", "aliases": "order", "intro": "x", "route": "/orders"},
            "svc-2": {"name": "用户", "aliases": "订单", "intro": "y", "route": "/users"},
        })
        # query "订单" 在 svc-1 的 name 字段命中（权重 3.0），svc-2 在 aliases 命中（权重 2.0）
        scores = idx.batch_score_tokens(["订单"])
        self.assertGreater(scores["svc-1"], scores["svc-2"])

    def test_batch_score_tokens_empty_query(self):
        from easysearch import MultiFieldBM25Index

        idx = MultiFieldBM25Index()
        idx.build({"svc-1": {"name": "a", "aliases": "", "intro": "", "route": ""}})
        self.assertEqual(idx.batch_score_tokens([]), {"svc-1": 0.0})

    def test_vocabulary_merged(self):
        from easysearch import MultiFieldBM25Index

        idx = MultiFieldBM25Index()
        idx.build({
            "svc-1": {"name": "订单", "aliases": "order", "intro": "管理", "route": "/orders"},
        })
        vocab = idx.vocabulary()
        self.assertIn("订单", vocab)
        self.assertIn("order", vocab)
        self.assertIn("管理", vocab)


class SynonymExpanderTests(unittest.TestCase):
    """A1：同义词扩展（领域词典 + KB 动态抽取）。"""

    def test_expand_adds_synonyms(self):
        from easysearch import SynonymExpander

        expander = SynonymExpander()
        # "开户" 的同义词组应包含 "网上开户" 等
        expanded = expander.expand(["开户"])
        self.assertIn("开户", expanded)
        # 至少追加一个同义词
        self.assertGreater(len(expanded), 1)

    def test_normalize_to_canonical(self):
        from easysearch import SynonymExpander

        expander = SynonymExpander()
        # "网上开户" 应归一到规范词（最短者，通常是 "开户"）
        normalized = expander.normalize("网上开户")
        self.assertIn("开户", normalized.split())

    def test_update_from_kb_links_alias_and_name(self):
        from easysearch import ServiceRecord, SynonymExpander, route_info

        services = {
            "svc-1": ServiceRecord(
                service_id="svc-1",
                service_name="银证转账",
                aliases=["资金", "转账"],
                service_intro="",
                route="/transfer",
            ),
        }
        expander = SynonymExpander()
        expander.update_from_kb(services)
        # "资金" 与 "银证转账" 应互为同义词
        syns = expander.synonyms_of("资金")
        self.assertIn("银证转账", syns)
        syns2 = expander.synonyms_of("银证转账")
        self.assertIn("资金", syns2)


class SpellCorrectionTests(unittest.TestCase):
    """A6：Levenshtein OOV 纠错。"""

    def test_correct_oov_token(self):
        from easysearch import LevenshteinCorrector

        corrector = LevenshteinCorrector({"订单", "开户", "转账"}, max_distance=2)
        # "订丹" 与 "订单" 编辑距离 1
        self.assertEqual(corrector.correct("订丹"), "订单")

    def test_iv_token_unchanged(self):
        from easysearch import LevenshteinCorrector

        corrector = LevenshteinCorrector({"订单"}, max_distance=2)
        # IV token 直接返回
        self.assertEqual(corrector.correct("订单"), "订单")

    def test_correct_tokens_appends_not_replaces(self):
        from easysearch import LevenshteinCorrector

        corrector = LevenshteinCorrector({"订单"}, max_distance=2)
        result = corrector.correct_tokens(["订丹"])
        # 原 token 保留 + 追加纠错结果
        self.assertIn("订丹", result)
        self.assertIn("订单", result)

    def test_no_correction_when_too_far(self):
        from easysearch import LevenshteinCorrector

        corrector = LevenshteinCorrector({"订单"}, max_distance=2)
        # "完全不同" 距离 > 2，原样返回
        self.assertEqual(corrector.correct("完全不同"), "完全不同")


class MMRRerankerTests(unittest.TestCase):
    """A5：MMR 多样性重排。"""

    def test_lambda_one_disables_diversity(self):
        from easysearch import MMRReranker

        reranker = MMRReranker(lambda_=1.0)
        candidates = [
            {"service_id": f"s{i}", "score": 10 - i, "rerank_score": 10 - i}
            for i in range(20)
        ]
        embeddings = {f"s{i}": [1.0 if i == j else 0.0 for j in range(20)] for i in range(20)}
        result = reranker.select(candidates, embeddings, top_k=5)
        # lambda=1 完全按相关性顺序，前 5 个
        self.assertEqual([r["service_id"] for r in result], ["s0", "s1", "s2", "s3", "s4"])

    def test_mmr_promotes_diversity(self):
        from easysearch import MMRReranker

        reranker = MMRReranker(lambda_=0.5)
        # s0/s1 高 relevance 但 embedding 完全相同（高度重复），
        # s2 relevance 较低但与 s0 完全不同（多样性补偿）
        candidates = [
            {"service_id": "s0", "score": 1.0, "rerank_score": 1.0},
            {"service_id": "s1", "score": 0.9, "rerank_score": 0.9},
            {"service_id": "s2", "score": 0.5, "rerank_score": 0.5},
        ]
        embeddings = {
            "s0": [1.0, 0.0],  # s0/s1 完全相同
            "s1": [1.0, 0.0],
            "s2": [0.0, 1.0],  # 与 s0 完全不同
        }
        result = reranker.select(candidates, embeddings, top_k=2)
        ids = [r["service_id"] for r in result]
        # 第一个仍是 s0（最高 relevance），第二个应是 s2（多样性）而非 s1
        self.assertEqual(ids[0], "s0")
        self.assertIn("s2", ids)
        self.assertNotIn("s1", ids)

    def test_fewer_candidates_returns_all(self):
        from easysearch import MMRReranker

        reranker = MMRReranker(lambda_=0.7)
        candidates = [{"service_id": "s0", "score": 1.0, "rerank_score": 1.0}]
        result = reranker.select(candidates, {"s0": [1.0]}, top_k=10)
        self.assertEqual(len(result), 1)


class PopularityDecayTests(unittest.TestCase):
    """A4：时间衰减 popularity。"""

    def test_recent_click_higher_than_old(self):
        import time
        from easysearch import SQLiteStore

        store = SQLiteStore(":memory:")
        now = time.time()
        # svc-1 近期点击（1 天前），svc-2 老点击（60 天前）
        store.append_click("u", "svc-1", now - 86400)
        store.append_click("u", "svc-2", now - 60 * 86400)
        scores = store.popularity_decayed(tau=2592000.0, now=now, window_days=90)
        self.assertGreater(scores["svc-1"], scores["svc-2"])
        store.close()

    def test_old_click_outside_window_excluded(self):
        import time
        from easysearch import SQLiteStore

        store = SQLiteStore(":memory:")
        now = time.time()
        # 100 天前的点击超出 90 天窗口
        store.append_click("u", "svc-old", now - 100 * 86400)
        scores = store.popularity_decayed(tau=2592000.0, now=now, window_days=90)
        self.assertNotIn("svc-old", scores)
        store.close()

    def test_hot_services_still_uses_raw_count(self):
        """A4 兼容：hot_services 仍用 raw count（不受时间衰减影响）。"""
        import time
        from easysearch import SQLiteStore

        store = SQLiteStore(":memory:")
        now = time.time()
        # svc-1 老点击 5 次，svc-2 近期点击 1 次
        for _ in range(5):
            store.append_click("u", "svc-1", now - 100 * 86400)
        store.append_click("u", "svc-2", now - 86400)
        # hot_services 按 raw count，svc-1 应排第 1
        self.assertEqual(store.hot_services(1), ["svc-1"])
        store.close()


if __name__ == "__main__":
    unittest.main()
