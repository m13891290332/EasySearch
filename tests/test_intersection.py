"""M6 多条件交集检索测试。

覆盖：
  - _intersect_candidate_lists：纯逻辑（交集/空集降级 union/单列表/空），无 IO 可单测。
  - _rrf_fuse_multi：多列表 RRF 融合。
  - search_intersection：端到端 (Top-10, match_mode)。
"""
from __future__ import annotations

import unittest

from easysearch import (
    DashScopeClient,
    DeepSeekClient,
    ServiceSearchEngine,
    SQLiteStore,
)

# svc-C 同时命中 "开户" 与 "转账"（交集）；svc-D 仅命中 "风控"
KB = [
    {"service_id": "svc-A", "service_name": "开户平台", "aliases": ["开户", "open"],
     "service_intro": "账户开户 网上开户", "route": "/open"},
    {"service_id": "svc-B", "service_name": "转账平台", "aliases": ["转账", "transfer"],
     "service_intro": "银证转账 资金划转", "route": "/transfer"},
    {"service_id": "svc-C", "service_name": "综合金融门户", "aliases": ["开户", "转账", "一站式"],
     "service_intro": "开户与转账一站式综合服务", "route": "/portal"},
    {"service_id": "svc-D", "service_name": "风控平台", "aliases": ["风控", "risk"],
     "service_intro": "风险决策管理", "route": "/risk"},
]


def make_engine():
    return ServiceSearchEngine(
        dashscope_client=DashScopeClient(api_key=None),
        deepseek_client=DeepSeekClient(api_key=None),
        store=SQLiteStore(":memory:"),
    )


def _cand(sid: str, score: float = 0.5) -> dict:
    return {
        "service_id": sid, "service_name": sid, "aliases": [], "service_intro": "",
        "route": f"/{sid}", "component": "C", "decision_button": "B",
        "derived": False, "score": score,
    }


class IntersectCandidateListsTests(unittest.TestCase):
    """纯逻辑：对已召回候选列表求交集/降级 union（确定性单测）。"""

    def test_intersection_picks_only_shared(self):
        # "开户" 命中 {svc-A, svc-C}；"转账" 命中 {svc-B, svc-C}；交集 {svc-C}
        per_q = [[_cand("svc-A"), _cand("svc-C")], [_cand("svc-B"), _cand("svc-C")]]
        candidates, mode = ServiceSearchEngine._intersect_candidate_lists(per_q)
        self.assertEqual(mode, "intersection")
        ids = [c["service_id"] for c in candidates]
        self.assertEqual(ids, ["svc-C"])  # 仅同时命中两者的服务

    def test_empty_intersection_degrades_to_union(self):
        # "开户" {svc-A, svc-C}；"风控" {svc-D}；交集空 → union
        per_q = [[_cand("svc-A"), _cand("svc-C")], [_cand("svc-D")]]
        candidates, mode = ServiceSearchEngine._intersect_candidate_lists(per_q)
        self.assertEqual(mode, "union")
        ids = {c["service_id"] for c in candidates}
        self.assertEqual(ids, {"svc-A", "svc-C", "svc-D"})

    def test_three_way_intersection(self):
        per_q = [
            [_cand("a"), _cand("b"), _cand("c")],
            [_cand("b"), _cand("c")],
            [_cand("c"), _cand("d")],
        ]
        candidates, mode = ServiceSearchEngine._intersect_candidate_lists(per_q)
        self.assertEqual(mode, "intersection")
        self.assertEqual([c["service_id"] for c in candidates], ["c"])

    def test_single_list_default(self):
        candidates, mode = ServiceSearchEngine._intersect_candidate_lists(
            [[_cand("a"), _cand("b")]]
        )
        self.assertEqual(mode, "default")
        self.assertEqual([c["service_id"] for c in candidates], ["a", "b"])

    def test_empty_lists_default(self):
        candidates, mode = ServiceSearchEngine._intersect_candidate_lists([])
        self.assertEqual(mode, "default")
        self.assertEqual(candidates, [])

    def test_intersection_preserves_first_list_order(self):
        # 交集项按首个子查询候选的原序输出
        per_q = [[_cand("c"), _cand("a"), _cand("b")], [_cand("b"), _cand("a"), _cand("c")]]
        candidates, _ = ServiceSearchEngine._intersect_candidate_lists(per_q)
        self.assertEqual([x["service_id"] for x in candidates], ["c", "a", "b"])


class RRFFuseMultiTests(unittest.TestCase):
    def test_three_lists_double_appearance_ranks_high(self):
        lists = [
            [_cand("a", 0.9), _cand("b", 0.5)],
            [_cand("b", 0.8), _cand("c", 0.7)],
            [_cand("b", 0.6), _cand("d", 0.4)],
        ]
        fused = ServiceSearchEngine._rrf_fuse_multi(lists, k=60, top_k=4)
        # b 同时在三表出现，RRF 最高
        self.assertEqual(fused[0]["service_id"], "b")
        self.assertEqual(len(fused), 4)


class SearchIntersectionTests(unittest.TestCase):
    """端到端：search_intersection 调用真实召回 + rerank + MMR。

    注：无 API key 时 vector 走 local_hash，小库 top-30 含全量服务，
    故交集总非空（不触发 union 降级）；union 降级由纯逻辑测试覆盖。
    """

    def setUp(self):
        self.engine = make_engine()
        self.engine.load_knowledge_base(KB)

    def test_intersection_end_to_end(self):
        # svc-C 同时命中 "开户" 与 "转账"，必在两子查询 top-30 交集中
        results, mode = self.engine.search_intersection(
            "u-int", ["开户", "转账"], original_query="开户 和 转账"
        )
        self.assertEqual(mode, "intersection")
        self.assertGreater(len(results), 0)
        ids = {r["service_id"] for r in results}
        self.assertIn("svc-C", ids)

    def test_single_query_default(self):
        results, mode = self.engine.search_intersection(
            "u-sing", ["开户"], original_query="开户"
        )
        self.assertEqual(mode, "default")
        self.assertGreater(len(results), 0)

    def test_all_blank_subqueries_returns_empty(self):
        results, mode = self.engine.search_intersection("u-none", ["  ", ""])
        self.assertEqual(mode, "default")
        self.assertEqual(results, [])

    def test_navigational_not_pinned_in_intersection(self):
        # 多条件模式不做 navigational 直达；svc-C 即便精确命中 alias 也不强制置顶
        results, mode = self.engine.search_intersection(
            "u-nav", ["开户", "转账"], original_query="开户 和 转账"
        )
        self.assertEqual(mode, "intersection")
        # 仅校验 svc-C 在结果中，不强制 #0（多条件按 rerank 排序）
        self.assertIn("svc-C", {r["service_id"] for r in results})


if __name__ == "__main__":
    unittest.main()
