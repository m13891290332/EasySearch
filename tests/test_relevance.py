"""M13 相关性提升测试。

覆盖：
  - LevenshteinCorrector：BK-tree 加速（与线性 correct 同口径）+ suggest 整条建议。
  - PinyinIndex：拼音索引构建（pypinyin 可用时；不可用时优雅降级）。
  - normalize_scores：minmax / rank / zscore 三模式基本正确性。
  - store：append_feedback / negative_signals（快速跳出计数）。
  - engine：_apply_negative_penalty（无负样本零影响 / 有负样本降权）+ spell_suggest。
  - API：/api/feedback 端点 + /api/search 返回 spell_suggestion 字段。
"""
from __future__ import annotations

import os
import tempfile
import time
import unittest

from fastapi.testclient import TestClient

from api.main import app, reset_engine
from easysearch import (
    DashScopeClient,
    DeepSeekClient,
    LevenshteinCorrector,
    ServiceSearchEngine,
    SQLiteStore,
)
from easysearch.spell import BKTree, PinyinIndex, levenshtein
from easysearch.utils import normalize_scores

# 含易拼错词的知识库（中英混合，避免 jieba 分词非确定性）
KB = [
    {
        "service_id": "svc-order",
        "service_name": "订单中心",
        "aliases": ["order", "订单"],
        "service_intro": "订单管理",
        "route": "/orders",
    },
    {
        "service_id": "svc-account",
        "service_name": "账户中心",
        "aliases": ["account", "账户"],
        "service_intro": "账户查询",
        "route": "/account",
    },
    {
        "service_id": "svc-transfer",
        "service_name": "转账",
        "aliases": ["transfer"],
        "service_intro": "资金转账",
        "route": "/transfer",
    },
]


def _make_engine(db_path: str = ":memory:") -> ServiceSearchEngine:
    return ServiceSearchEngine(
        dashscope_client=DashScopeClient(api_key=None),
        deepseek_client=DeepSeekClient(api_key=None),
        store=SQLiteStore(db_path),
        db_path=db_path,
    )


class BKTreeTests(unittest.TestCase):
    """M13：BK-tree 近邻检索与线性扫描同口径。"""

    def setUp(self):
        self.vocab = ["订单", "开户", "转账", "order", "account"]
        self.tree = BKTree(self.vocab)

    def test_search_returns_within_distance(self):
        # "ordr" 与 "order" 距离 1
        hits = self.tree.search("ordr", 2)
        words = {w for w, _ in hits}
        self.assertIn("order", words)

    def test_parity_with_linear_scan(self):
        """BK-tree 结果集 == 线性扫描结果集（同口径 levenshtein）。"""
        token = "ordr"
        max_d = 2
        linear = {
            w for w in self.vocab
            if abs(len(w) - len(token)) <= max_d and levenshtein(token, w) <= max_d
        }
        bktree = {w for w, d in self.tree.search(token, max_d) if d <= max_d}
        self.assertEqual(linear, bktree)

    def test_empty_query_returns_empty(self):
        self.assertEqual(self.tree.search("", 2), [])

    def test_no_match_within_distance(self):
        # "zzz" 与全部 vocab 距离 > 2
        self.assertEqual(self.tree.search("zzzzzzz", 1), [])


class SuggestTests(unittest.TestCase):
    """M13：整条 query 的拼写建议。"""

    def setUp(self):
        self.corrector = LevenshteinCorrector(
            {"order", "account", "transfer"}, max_distance=2
        )

    def test_suggest_corrects_oov_typo(self):
        # "ordr" OOV → "order"（距离 1）
        self.assertEqual(self.corrector.suggest("ordr"), "order")

    def test_suggest_returns_none_for_clean_query(self):
        # "order" 在 vocab → 无 OOV → None
        self.assertIsNone(self.corrector.suggest("order"))

    def test_suggest_returns_none_when_too_far(self):
        # "zzzzzz" 与所有 vocab 距离 > 2
        self.assertIsNone(self.corrector.suggest("zzzzzz"))

    def test_suggest_preserves_iv_tokens(self):
        # 混合：一个 OOV + 一个 IV → 纠正 OOV，保留 IV
        corrector = LevenshteinCorrector(
            {"order", "account", "transa"}, max_distance=2
        )
        # "ordr account" → "order account"
        self.assertEqual(corrector.suggest("ordr account"), "order account")


class PinyinIndexTests(unittest.TestCase):
    """M13：拼音同音纠错索引。"""

    def test_unavailable_without_pypinyin(self):
        # 仅当 pypinyin 不可用时验证降级；可用时跳过（无法强制卸载）
        try:
            import pypinyin  # noqa: F401
            self.skipTest("pypinyin 已安装，跳过降级用例")
        except ImportError:
            idx = PinyinIndex(["订单"])
            self.assertFalse(idx.available)
            self.assertEqual(idx.suggest_by_pinyin("订丹"), [])

    def test_pinyin_suggest_when_available(self):
        try:
            import pypinyin  # noqa: F401
        except ImportError:
            self.skipTest("pypinyin 未安装，跳过拼音用例")
        # 「订丹」「订单」同音（dingdan），拼音索引应召回
        idx = PinyinIndex(["订单", "开户"])
        self.assertTrue(idx.available)
        hits = idx.suggest_by_pinyin("订丹")
        self.assertIn("订单", hits)


class NormalizeScoresTests(unittest.TestCase):
    """M13：归一化三模式。"""

    def test_minmax_default(self):
        out = normalize_scores({"a": 1.0, "b": 3.0})
        self.assertEqual(out["a"], 0.0)
        self.assertEqual(out["b"], 1.0)

    def test_minmax_all_equal_returns_zero(self):
        out = normalize_scores({"a": 2.0, "b": 2.0})
        self.assertEqual(out, {"a": 0.0, "b": 0.0})

    def test_rank_preserves_order(self):
        out = normalize_scores({"a": 1.0, "b": 2.0, "c": 3.0}, mode="rank")
        # 降序：b<c... 实际 c=3 最高 → rank0 → 1.0；b=2 → rank1 → 0.5；a=1 → rank2 → 0.0
        self.assertEqual(out["c"], 1.0)
        self.assertEqual(out["b"], 0.5)
        self.assertEqual(out["a"], 0.0)

    def test_zscore_zero_mean(self):
        import math
        out = normalize_scores({"a": 0.0, "b": 2.0}, mode="zscore")
        # mean=1, std=1 → a=-1, b=1
        self.assertAlmostEqual(out["a"], -1.0)
        self.assertAlmostEqual(out["b"], 1.0)

    def test_empty(self):
        self.assertEqual(normalize_scores({}), {})
        self.assertEqual(normalize_scores({}, mode="rank"), {})


class NegativeFeedbackStoreTests(unittest.TestCase):
    """M13：store 负反馈落库。"""

    def setUp(self):
        self.store = SQLiteStore(":memory:")
        self.now = time.time()

    def tearDown(self):
        self.store.close()

    def test_negative_signals_counts_quick_bounce(self):
        self.store.append_feedback("u", "svc-1", 1000, self.now)
        self.store.append_feedback("u", "svc-1", 500, self.now)
        self.store.append_feedback("u", "svc-2", 5000, self.now)  # 非负
        neg = self.store.negative_signals(now=self.now, quick_bounce_ms=3000)
        self.assertEqual(neg, {"svc-1": 2})

    def test_negative_signals_window_excludes_old(self):
        # 100 天前超出 90 天窗口
        self.store.append_feedback("u", "svc-old", 100, self.now - 100 * 86400)
        neg = self.store.negative_signals(
            now=self.now, window_days=90, quick_bounce_ms=3000
        )
        self.assertEqual(neg, {})

    def test_no_feedback_returns_empty(self):
        self.assertEqual(
            self.store.negative_signals(now=self.now, quick_bounce_ms=3000), {}
        )


class EngineNegativePenaltyTests(unittest.TestCase):
    """M13：engine._apply_negative_penalty。"""

    def setUp(self):
        self.engine = _make_engine()
        self.engine.load_knowledge_base(KB)

    def tearDown(self):
        self.engine.store.close()

    def test_no_negatives_returns_unchanged(self):
        pop = {"svc-order": 1.0, "svc-account": 0.5}
        out = self.engine._apply_negative_penalty(pop, time.time())
        self.assertEqual(out, pop)

    def test_negative_reduces_popularity(self):
        now = time.time()
        # 对 svc-order 记 2 次快速跳出
        self.engine.store.append_feedback("u", "svc-order", 500, now)
        self.engine.store.append_feedback("u", "svc-order", 800, now)
        out = self.engine._apply_negative_penalty({"svc-order": 1.0}, now)
        self.assertLess(out["svc-order"], 1.0)
        self.assertGreaterEqual(out["svc-order"], 0.0)


class EngineSpellSuggestTests(unittest.TestCase):
    """M13：engine.spell_suggest。"""

    def setUp(self):
        self.engine = _make_engine()
        self.engine.load_knowledge_base(KB)

    def tearDown(self):
        self.engine.store.close()

    def test_suggest_for_typo(self):
        # KB 含 "order"；query "ordr" OOV → 建议 "order"
        sug = self.engine.spell_suggest("ordr")
        self.assertEqual(sug, "order")

    def test_suggest_none_for_clean(self):
        self.assertIsNone(self.engine.spell_suggest("order"))

    def test_suggest_none_when_spell_disabled(self):
        # 直接构造一个未初始化纠错器的场景（spell_corrector=None）
        eng = _make_engine()
        # 未 load KB → spell_corrector 为 None
        self.assertIsNone(eng.spell_suggest("ordr"))


class APIFeedbackAndSuggestionTests(unittest.TestCase):
    """M13：/api/feedback 端点 + /api/search.spell_suggestion。"""

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

    def test_feedback_endpoint_ok(self):
        r = self.client.post(
            "/api/feedback",
            json={"user_id": "u1", "service_id": "svc-order", "dwell_ms": 500},
        )
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json(), {"status": "ok"})
        # 负样本已落库
        neg = self.engine.store.negative_signals(quick_bounce_ms=3000)
        self.assertEqual(neg.get("svc-order"), 1)

    def test_feedback_long_dwell_not_negative(self):
        r = self.client.post(
            "/api/feedback",
            json={"user_id": "u1", "service_id": "svc-order", "dwell_ms": 5000},
        )
        self.assertEqual(r.status_code, 200)
        neg = self.engine.store.negative_signals(quick_bounce_ms=3000)
        self.assertNotIn("svc-order", neg)

    def test_feedback_unknown_service_returns_200(self):
        """M12：已下线服务仍记 dwell（行为分析信号），不硬 404。"""
        r = self.client.post(
            "/api/feedback",
            json={"user_id": "u1", "service_id": "nope", "dwell_ms": 500},
        )
        self.assertEqual(r.status_code, 200)

    def test_search_returns_spell_suggestion(self):
        # query "ordr" OOV → 建议 "order"
        r = self.client.get(
            "/api/search", params={"user_id": "u1", "query": "ordr"}
        )
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertIn("spell_suggestion", body)
        self.assertEqual(body["spell_suggestion"], "order")

    def test_search_no_suggestion_for_clean_query(self):
        r = self.client.get(
            "/api/search", params={"user_id": "u1", "query": "order"}
        )
        self.assertEqual(r.status_code, 200)
        # "order" 在 vocab → 无建议
        self.assertIsNone(r.json()["spell_suggestion"])


if __name__ == "__main__":
    unittest.main()
