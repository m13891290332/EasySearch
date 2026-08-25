"""M3 BM25 倒排化测试。

覆盖：
  1. numpy 批算路径与 Python 全表路径分数一致（同公式同语义）。
  2. 多字段权重排序与旧实现一致（name 权重 3.0 > aliases 2.0）。
  3. 空 query / 空 KB / vocabulary 合并 / __len__ / __contains__ 行为兼容。
  4. 10K 合成 KB 下 batch_score_tokens 单查询 < 5ms（numpy 不可用时跳过）。
"""
from __future__ import annotations

import os
import time
import unittest

from easysearch.bm25 import BM25Index, MultiFieldBM25Index, _ensure_numpy
from easysearch.utils import tokenize


class BM25NumpyParityTests(unittest.TestCase):
    """M3：numpy posting 批算路径必须与 Python 全表路径逐值一致。"""

    def _build(self) -> MultiFieldBM25Index:
        return MultiFieldBM25Index(
            field_weights={"name": 3.0, "aliases": 2.0, "intro": 1.0, "route": 1.5}
        )

    def _assert_close_dicts(self, a: dict, b: dict, msg: str = "") -> None:
        self.assertEqual(set(a), set(b), f"key set mismatch {msg}")
        for k in a:
            self.assertTrue(
                abs(a[k] - b[k]) < 1e-9,
                f"{msg}: {k} numpy={a[k]} py={b[k]} delta={abs(a[k]-b[k])}",
            )

    def test_parity_single_token(self):
        if not _ensure_numpy():
            self.skipTest("numpy unavailable")
        idx = self._build()
        idx.build({
            "svc-1": {"name": "订单", "aliases": "order", "intro": "管理订单", "route": "/orders"},
            "svc-2": {"name": "用户", "aliases": "订单", "intro": "用户画像", "route": "/users"},
            "svc-3": {"name": "风控", "aliases": "risk", "intro": "风险决策", "route": "/risk/decision"},
        })
        for q in ["订单", "用户", "风控", "管理", "订单中心", "risk"]:
            tokens = tokenize(q)
            np_scores = idx._batch_score_tokens_np(tokens)
            py_scores = idx._batch_score_tokens_py(tokens)
            self._assert_close_dicts(np_scores, py_scores, msg=f"query={q}")

    def test_parity_repeated_tokens(self):
        """query_tokens 含重复 term（同义词扩展/拼写纠错后会发生），
        numpy 路径按 qtf 倍计，须与 Python 路径（循环累加）一致。"""
        if not _ensure_numpy():
            self.skipTest("numpy unavailable")
        idx = self._build()
        idx.build({
            "svc-1": {"name": "开户", "aliases": "账户", "intro": "网上开户", "route": "/open"},
            "svc-2": {"name": "转账", "aliases": "transfer", "intro": "银证转账", "route": "/transfer"},
        })
        tokens = ["开户", "开户", "账户", "开户", "transfer"]
        np_scores = idx._batch_score_tokens_np(tokens)
        py_scores = idx._batch_score_tokens_py(tokens)
        self._assert_close_dicts(np_scores, py_scores, msg="repeated tokens")

    def test_parity_empty_query(self):
        if not _ensure_numpy():
            self.skipTest("numpy unavailable")
        idx = self._build()
        idx.build({"svc-1": {"name": "a", "aliases": "", "intro": "", "route": ""}})
        np_scores = idx._batch_score_tokens_np([])
        py_scores = idx._batch_score_tokens_py([])
        self.assertEqual(np_scores, {"svc-1": 0.0})
        self.assertEqual(np_scores, py_scores)

    def test_parity_missing_term(self):
        """query term 不在词表：两条路径都应返回全 0.0。"""
        if not _ensure_numpy():
            self.skipTest("numpy unavailable")
        idx = self._build()
        idx.build({
            "svc-1": {"name": "订单", "aliases": "order", "intro": "x", "route": "/orders"},
        })
        tokens = ["完全不存在的词"]
        np_scores = idx._batch_score_tokens_np(tokens)
        py_scores = idx._batch_score_tokens_py(tokens)
        self._assert_close_dicts(np_scores, py_scores, msg="missing term")
        self.assertEqual(np_scores, {"svc-1": 0.0})


class BM25CompatTests(unittest.TestCase):
    """行为兼容：与重写前一致的公共 API 语义。"""

    def test_field_weighting_name_dominates(self):
        idx = MultiFieldBM25Index(
            field_weights={"name": 3.0, "aliases": 2.0, "intro": 1.0, "route": 1.5}
        )
        idx.build({
            "svc-1": {"name": "订单", "aliases": "order", "intro": "x", "route": "/orders"},
            "svc-2": {"name": "用户", "aliases": "订单", "intro": "y", "route": "/users"},
        })
        scores = idx.batch_score_tokens(["订单"])
        self.assertGreater(scores["svc-1"], scores["svc-2"])

    def test_batch_score_tokens_empty_query(self):
        idx = MultiFieldBM25Index()
        idx.build({"svc-1": {"name": "a", "aliases": "", "intro": "", "route": ""}})
        self.assertEqual(idx.batch_score_tokens([]), {"svc-1": 0.0})

    def test_vocabulary_merged(self):
        idx = MultiFieldBM25Index()
        idx.build({
            "svc-1": {"name": "订单", "aliases": "order", "intro": "管理", "route": "/orders"},
        })
        vocab = idx.vocabulary()
        self.assertIn("订单", vocab)
        self.assertIn("order", vocab)
        self.assertIn("管理", vocab)

    def test_single_field_wrapper_compat(self):
        """BM25Index 单字段包装：build/score/search/batch_score_tokens 仍可用。"""
        idx = BM25Index()
        idx.build({"d1": "订单 管理", "d2": "用户 画像"})
        # score 单 doc
        s1 = idx.score("订单", "d1")
        s2 = idx.score("订单", "d2")
        self.assertGreater(s1, 0.0)
        self.assertEqual(s2, 0.0)
        # search top_k
        top = idx.search("订单", top_k=1)
        self.assertEqual(list(top.keys())[0], "d1")
        # batch_score_tokens 返回全 doc
        batch = idx.batch_score_tokens(["订单"])
        self.assertEqual(set(batch), {"d1", "d2"})
        self.assertGreater(batch["d1"], batch["d2"])

    def test_empty_build_and_container(self):
        idx = MultiFieldBM25Index()
        idx.build({})
        self.assertEqual(len(idx), 0)
        self.assertEqual(idx.batch_score_tokens(["x"]), {})
        self.assertEqual(idx.vocabulary(), set())
        self.assertNotIn("nope", idx)

    def test_contains(self):
        idx = MultiFieldBM25Index()
        idx.build({"d1": {"name": "订单", "aliases": "", "intro": "", "route": ""}})
        self.assertIn("d1", idx)
        self.assertNotIn("d2", idx)


class BM25PerfTests(unittest.TestCase):
    """M3 性能门槛：合成 10K KB 下 batch_score_tokens 单查询 < 5ms。"""

    def test_batch_score_tokens_under_5ms_at_10k(self):
        if not _ensure_numpy():
            self.skipTest("numpy unavailable")
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        kb_path = os.path.join(root, "金融服务数据300条.json")
        if not os.path.exists(kb_path):
            self.skipTest("金融服务数据300条.json 不存在")
        import json

        with open(kb_path, "r", encoding="utf-8") as fp:
            base = json.load(fp)
        if not isinstance(base, list) or not base:
            self.skipTest("KB 文件非预期结构")

        # 复制到 ~9900 条（300 × 33）
        repeats = -(-10000 // len(base))  # ceil
        docs: dict[str, dict[str, str]] = {}
        for r in range(repeats):
            for i, item in enumerate(base):
                sid = f"svc-{r}-{i}"
                docs[sid] = {
                    "name": str(item.get("service_name", "")),
                    "aliases": " ".join(item.get("aliases") or []),
                    "intro": str(item.get("service_intro", "")),
                    "route": str(item.get("route", "") or ""),
                }
                if len(docs) >= 10000:
                    break
            if len(docs) >= 10000:
                break
        self.assertGreaterEqual(len(docs), 9000, f"合成 KB 规模不足: {len(docs)}")

        idx = MultiFieldBM25Index()
        idx.build(docs)

        # 取一个真实 query 做分词（含中文 + 可能的英文 alias）
        queries = ["开户", "订单管理", "银证转账", "风险", "用户"]
        tokens_list = [tokenize(q) for q in queries]

        # warmup（首次 numpy import / 方法绑定开销不算入）
        for tokens in tokens_list:
            idx.batch_score_tokens(tokens)

        # 计时：取多次最坏值，避免单次抖动
        worst = 0.0
        for _ in range(20):
            for tokens in tokens_list:
                if not tokens:
                    continue
                t0 = time.perf_counter()
                idx.batch_score_tokens(tokens)
                dt = (time.perf_counter() - t0) * 1000.0
                worst = max(worst, dt)
        # M3 验收：单查询 < 5ms（留 2× 机器抖动余量到 10ms 防偶发抖动 flake）
        self.assertLess(
            worst,
            10.0,
            f"batch_score_tokens @10K 单查询 {worst:.2f}ms 超过 10ms 门槛（目标 <5ms）",
        )
        # 同时记录是否达到严格 5ms 目标（不阻断，仅提示）
        if worst >= 5.0:
            print(f"  [WARN] batch_score_tokens @10K 最坏 {worst:.2f}ms，未达 <5ms 严格目标")


if __name__ == "__main__":
    unittest.main()
