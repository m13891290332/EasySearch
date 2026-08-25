"""M7 长程对话搜索测试。

覆盖：
  - store 会话持久化（append/exists/turns/rollback）。
  - engine.search_session：首轮宽召回 / 后续轮精化 / 撤回 / 空撤回。
  - classify_intent：session_id 传入且会话存在 → conversational。
  - search_session_async：异步路径 rerank+reason 并发。
"""
from __future__ import annotations

import asyncio
import unittest

from easysearch import (
    CONVERSATIONAL,
    DashScopeClient,
    DeepSeekClient,
    ServiceSearchEngine,
    SQLiteStore,
)

# 复用多条件测试的知识库形态：svc-C 同时覆盖开户/转账
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


def make_engine(db_path: str = ":memory:"):
    return ServiceSearchEngine(
        dashscope_client=DashScopeClient(api_key=None),
        deepseek_client=DeepSeekClient(api_key=None),
        store=SQLiteStore(db_path),
        db_path=db_path,
    )


class StoreSessionTests(unittest.TestCase):
    """SQLiteStore 会话表 CRUD。"""

    def setUp(self):
        self.store = SQLiteStore(":memory:")

    def test_empty_session_not_exists(self):
        self.assertFalse(self.store.session_exists("s1"))
        self.assertEqual(self.store.session_last_turn_idx("s1"), -1)
        self.assertEqual(self.store.session_turns("s1"), [])

    def test_append_and_turns_ordering(self):
        self.store.append_session_turn("s1", "u1", 0, "开户", ["svc-A", "svc-C"], 1.0)
        self.store.append_session_turn("s1", "u1", 1, "转账", ["svc-B", "svc-C"], 2.0)
        self.assertTrue(self.store.session_exists("s1"))
        self.assertEqual(self.store.session_last_turn_idx("s1"), 1)
        turns = self.store.session_turns("s1")
        self.assertEqual(len(turns), 2)
        self.assertEqual(int(turns[0]["turn_idx"]), 0)
        self.assertEqual(turns[0]["query"], "开户")
        self.assertEqual(int(turns[1]["turn_idx"]), 1)

    def test_delete_last_turn(self):
        self.store.append_session_turn("s1", "u1", 0, "q0", ["a"], 1.0)
        self.store.append_session_turn("s1", "u1", 1, "q1", ["b"], 2.0)
        self.assertTrue(self.store.session_delete_last_turn("s1"))
        self.assertEqual(self.store.session_last_turn_idx("s1"), 0)
        # 再撤一次 → 剩 0 轮
        self.assertTrue(self.store.session_delete_last_turn("s1"))
        self.assertFalse(self.store.session_exists("s1"))
        self.assertEqual(self.store.session_last_turn_idx("s1"), -1)
        # 空会话再撤 → False
        self.assertFalse(self.store.session_delete_last_turn("s1"))


class EngineSessionTests(unittest.TestCase):
    """engine.search_session 端到端（无 API key，向量走 local_hash）。"""

    def setUp(self):
        self.engine = make_engine()
        self.engine.load_knowledge_base(KB)

    def test_first_turn_wide_recall(self):
        r = self.engine.search_session("s1", "u1", "开户")
        self.assertEqual(r["action"], "search")
        self.assertEqual(r["turn_idx"], 0)
        self.assertEqual(r["match_mode"], "session")
        self.assertGreater(len(r["results"]), 0)
        # 首轮后 history 含 1 轮
        self.assertEqual(len(r["history"]), 1)
        self.assertEqual(r["history"][0]["turn_idx"], 0)
        self.assertEqual(r["history"][0]["query"], "开户")
        # 会话已存在
        self.assertTrue(self.engine._session_exists("s1"))

    def test_second_turn_refinement_increments_turn(self):
        self.engine.search_session("s1", "u1", "开户")
        r = self.engine.search_session("s1", "u1", "转账")
        self.assertEqual(r["turn_idx"], 1)
        self.assertEqual(len(r["history"]), 2)
        # 后续轮仍返回结果（精化重排）
        self.assertGreater(len(r["results"]), 0)

    def test_rollback_returns_previous_turn(self):
        # 两轮后撤回 → 回到第 0 轮
        self.engine.search_session("s1", "u1", "开户")
        self.engine.search_session("s1", "u1", "转账")
        r = self.engine.search_session("s1", "u1", "", action="rollback")
        self.assertEqual(r["action"], "rollback")
        self.assertEqual(r["turn_idx"], 0)
        self.assertEqual(r["query"], "开户")
        self.assertEqual(r["match_mode"], "rollback")
        # 撤回后结果来自第 0 轮 top_ids（重建项）
        self.assertGreater(len(r["results"]), 0)
        self.assertIn("撤回至上一轮", r["results"][0].get("rerank_reason", ""))
        # history 只剩 1 轮
        self.assertEqual(len(r["history"]), 1)

    def test_rollback_to_empty(self):
        # 仅 1 轮 → 撤回弹出末轮后无上一轮可返回 → 空会话
        self.engine.search_session("s1", "u1", "开户")
        r = self.engine.search_session("s1", "u1", "", action="rollback")
        self.assertEqual(r["turn_idx"], -1)
        self.assertEqual(r["match_mode"], "empty")
        self.assertEqual(r["results"], [])
        self.assertEqual(r["history"], [])
        # 会话已空，再撤仍空（不报错）
        r2 = self.engine.search_session("s1", "u1", "", action="rollback")
        self.assertEqual(r2["turn_idx"], -1)
        self.assertEqual(r2["results"], [])

    def test_rollback_empty_session_from_scratch(self):
        r = self.engine.search_session("s-nope", "u1", "", action="rollback")
        self.assertEqual(r["turn_idx"], -1)
        self.assertEqual(r["match_mode"], "empty")
        self.assertEqual(r["results"], [])

    def test_empty_query_search_returns_empty(self):
        r = self.engine.search_session("s1", "u1", "   ")
        self.assertEqual(r["match_mode"], "empty")
        self.assertEqual(r["results"], [])

    def test_first_turn_candidates_capped_at_40(self):
        # 落库的 top_ids 不应超过 40（宽召回上限）
        self.engine.search_session("s1", "u1", "开户")
        turns = self.engine.store.session_turns("s1")
        import json as _json
        top_ids = _json.loads(turns[0]["top_ids_json"])
        self.assertLessEqual(len(top_ids), 40)


class ClassifyIntentSessionTests(unittest.TestCase):
    """classify_intent：session_id 传入且会话存在 → conversational。"""

    def setUp(self):
        self.engine = make_engine()
        self.engine.load_knowledge_base(KB)

    def test_no_session_id_not_conversational(self):
        # 无 session_id → has_session=False，generic query 走 default
        r = self.engine.classify_intent("还想要的那个", user_id="u1")
        self.assertNotEqual(r.intent, CONVERSATIONAL)

    def test_session_id_nonexistent_not_conversational(self):
        # session_id 传入但会话不存在 → has_session=False
        r = self.engine.classify_intent(
            "还想要的那个", user_id="u1", session_id="s-nope"
        )
        self.assertNotEqual(r.intent, CONVERSATIONAL)

    def test_session_id_existing_is_conversational(self):
        # 先建一会话（首轮）
        self.engine.search_session("s1", "u1", "开户")
        # generic query + 会话存在 → conversational
        r = self.engine.classify_intent(
            "还想要的那个", user_id="u1", session_id="s1"
        )
        self.assertEqual(r.intent, CONVERSATIONAL)

    def test_navigational_still_priority_over_conversational(self):
        # 会话存在但精确命中服务名 → navigational 优先
        self.engine.search_session("s1", "u1", "开户")
        r = self.engine.classify_intent(
            "风控平台", user_id="u1", session_id="s1"
        )
        self.assertEqual(r.intent, "navigational")
        self.assertEqual(r.matched_service_id, "svc-D")


class AsyncSessionTests(unittest.IsolatedAsyncioTestCase):
    """search_session_async 异步路径。"""

    def setUp(self):
        self.engine = make_engine()
        self.engine.load_knowledge_base(KB)

    async def test_async_first_and_rollback(self):
        r = await self.engine.search_session_async("s1", "u1", "开户")
        self.assertEqual(r["turn_idx"], 0)
        self.assertEqual(r["match_mode"], "session")
        self.assertGreater(len(r["results"]), 0)
        # 异步撤回
        r2 = await self.engine.search_session_async("s1", "u1", "", action="rollback")
        self.assertEqual(r2["action"], "rollback")
        self.assertEqual(r2["turn_idx"], -1)
        self.assertEqual(r2["match_mode"], "empty")

    async def test_async_multi_turn(self):
        await self.engine.search_session_async("s1", "u1", "开户")
        r = await self.engine.search_session_async("s1", "u1", "转账")
        self.assertEqual(r["turn_idx"], 1)
        self.assertEqual(len(r["history"]), 2)


if __name__ == "__main__":
    unittest.main()
