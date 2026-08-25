"""搜索框灰色补全建议测试。

覆盖：
  - QuerySuggester._parse_completion：JSON 解析 / 裸字符串回退 / 前缀强制 / 限长 / list content。
  - QuerySuggester.suggest(_async)：enabled=False / 空 partial / RuntimeError 降级 / mock 成功。
  - engine.suggest_query(_async)：历史聚合 / 已下线服务过滤 / 异常静默。
  - /api/search/suggest 端点：参数校验 / 无 key 降级 / mock 成功 / 前缀不匹配降级。
"""
from __future__ import annotations

import asyncio
import unittest

from easysearch import (
    DashScopeClient,
    DeepSeekClient,
    QuerySuggester,
    ServiceSearchEngine,
    SQLiteStore,
)

KB = [
    {"service_id": "svc-A", "service_name": "开户平台", "aliases": ["开户"],
     "service_intro": "账户开户 网上开户", "route": "/open"},
    {"service_id": "svc-B", "service_name": "转账平台", "aliases": ["转账"],
     "service_intro": "银证转账 资金划转", "route": "/transfer"},
]


def make_engine(db_path: str = ":memory:"):
    """构造测试引擎。db_path=:memory: 避免 .npz embedding 缓存污染（M12 教训）。"""
    return ServiceSearchEngine(
        dashscope_client=DashScopeClient(api_key=None),
        deepseek_client=DeepSeekClient(api_key=None),
        store=SQLiteStore(db_path),
        db_path=db_path,
    )


def _resp(content):
    """构造 chat/completions 响应（content 为 str 或 list）。"""
    return {"choices": [{"message": {"content": content}}]}


class SuggestParseTests(unittest.TestCase):
    """QuerySuggester._parse_completion 纯逻辑。"""

    def setUp(self):
        # _parse_completion 不依赖 client，disabled client 即可
        self.s = QuerySuggester(DeepSeekClient(api_key=None))

    def test_parse_json_valid(self):
        r = self.s._parse_completion(_resp('{"completion":"开户流程"}'), "开户")
        self.assertEqual(r, "开户流程")

    def test_parse_json_string_literal(self):
        # JSON 字符串字面量 '"开户流程"' → extract_json 返回 str
        r = self.s._parse_completion(_resp('"开户流程指南"'), "开户")
        self.assertEqual(r, "开户流程指南")

    def test_parse_raw_string_fallback(self):
        # 非 JSON 裸字符串 → extract_json 返 None → 走裸字符串回退
        r = self.s._parse_completion(_resp("开户流程指南"), "开户")
        self.assertEqual(r, "开户流程指南")

    def test_parse_strips_code_block(self):
        # 带 ```json 代码块 → extract_json 剥块后解析
        r = self.s._parse_completion(
            _resp('```json\n{"completion":"开户流程"}\n```'), "开户"
        )
        self.assertEqual(r, "开户流程")

    def test_parse_returns_none_when_not_prefixed(self):
        # LLM 忽略前缀指令 → 服务端硬性校验 → None
        r = self.s._parse_completion(_resp('{"completion":"转账流程"}'), "开户")
        self.assertIsNone(r)

    def test_parse_returns_none_when_equal_to_partial(self):
        # 补全等于前缀（无新内容）→ None
        r = self.s._parse_completion(_resp('{"completion":"开户"}'), "开户")
        self.assertIsNone(r)

    def test_parse_returns_none_on_non_prefixed_garbage(self):
        # 裸字符串回退得到原串，但不以 partial 开头 → None
        r = self.s._parse_completion(_resp("不是JSON也不是补全"), "开户")
        self.assertIsNone(r)

    def test_parse_returns_none_on_empty(self):
        r = self.s._parse_completion(_resp(""), "开户")
        self.assertIsNone(r)

    def test_parse_truncates_long_completion(self):
        long = "开户" + "x" * 100
        r = self.s._parse_completion(_resp(f'{{"completion":"{long}"}}'), "开户")
        self.assertIsNotNone(r)
        self.assertLessEqual(len(r), 50)
        self.assertTrue(r.startswith("开户"))

    def test_parse_truncated_still_validates_prefix(self):
        # 截断后若不再以 partial 开头（极端情况）→ None；这里构造正常截断
        long = "开户流程" + "细节" * 30
        r = self.s._parse_completion(_resp(f'{{"completion":"{long}"}}'), "开户")
        self.assertIsNotNone(r)
        self.assertTrue(r.startswith("开户"))

    def test_parse_list_content_form(self):
        # 兼容 message.content 为 list（与 GuideGenerator._extract_content 一致）
        r = self.s._parse_completion(
            {"choices": [{"message": {"content": [
                {"type": "text", "text": '{"completion":"开户流程"}'}
            ]}}]},
            "开户",
        )
        self.assertEqual(r, "开户流程")


class SuggestEnabledTests(unittest.TestCase):
    """QuerySuggester.suggest：enabled / 异常降级 / mock 成功。"""

    def test_disabled_client_returns_none(self):
        s = QuerySuggester(DeepSeekClient(api_key=None))  # enabled=False
        self.assertIsNone(s.suggest("开户", ["开户流程"], ["开户平台"]))

    def test_empty_partial_returns_none(self):
        # 即使 enabled=True 也不调用 LLM
        class _Fake:
            enabled = True
            def post_json(self, url, payload):
                raise AssertionError("should not call LLM on empty partial")
        s = QuerySuggester(_Fake())
        self.assertIsNone(s.suggest("", ["x"], ["y"]))

    def test_runtime_error_silenced(self):
        class _Failing:
            enabled = True
            def post_json(self, url, payload):
                raise RuntimeError("timeout")
        s = QuerySuggester(_Failing())
        self.assertIsNone(s.suggest("开户", ["开户流程"], ["开户平台"]))

    def test_mocked_success(self):
        class _Fake:
            enabled = True
            def post_json(self, url, payload):
                return _resp('{"completion":"开户流程指南"}')
        s = QuerySuggester(_Fake())
        self.assertEqual(s.suggest("开户", ["开户流程"], ["开户平台"]), "开户流程指南")

    def test_mocked_prefix_mismatch_returns_none(self):
        class _Fake:
            enabled = True
            def post_json(self, url, payload):
                return _resp('{"completion":"转账流程"}')  # 不以"开户"开头
        s = QuerySuggester(_Fake())
        self.assertIsNone(s.suggest("开户", [], []))

    def test_async_path_success(self):
        class _FakeAsync:
            enabled = True
            async def post_json_async(self, url, payload):
                return _resp('{"completion":"开户流程"}')
        s = QuerySuggester(_FakeAsync())
        r = asyncio.run(s.suggest_async("开户", [], []))
        self.assertEqual(r, "开户流程")

    def test_async_path_runtime_error_silenced(self):
        class _FakeAsync:
            enabled = True
            async def post_json_async(self, url, payload):
                raise RuntimeError("timeout")
        s = QuerySuggester(_FakeAsync())
        self.assertIsNone(asyncio.run(s.suggest_async("开户", [], [])))


class EngineSuggestTests(unittest.TestCase):
    """engine.suggest_query(_async)：历史聚合 + 异常静默。"""

    def setUp(self):
        self.engine = make_engine()
        self.engine.load_knowledge_base(KB)

    def test_history_aggregation(self):
        # 注入历史
        self.engine.store.append_query("u1", "开户流程", 1.0)
        self.engine.store.append_click("u1", "svc-A", 1.0)
        captured = {}

        class _Cap:
            def suggest(self, partial, rq, rc):
                captured.update(partial=partial, rq=rq, rc=rc)
                return None
        self.engine.query_suggester = _Cap()
        self.engine.suggest_query("u1", "开户")
        self.assertEqual(captured["partial"], "开户")
        self.assertIn("开户流程", captured["rq"])
        self.assertIn("开户平台", captured["rc"])

    def test_offline_service_id_filtered(self):
        # 点击了 svc-X（不在 KB）→ 不应抛 KeyError，且不在 recent_clicked_names 里
        self.engine.store.append_click("u1", "svc-X", 1.0)
        captured = {}

        class _Cap:
            def suggest(self, partial, rq, rc):
                captured["rc"] = list(rc)
                return None
        self.engine.query_suggester = _Cap()
        # 不应抛异常
        self.assertIsNone(self.engine.suggest_query("u1", "x"))
        # svc-X 被过滤掉
        self.assertNotIn("svc-X", captured["rc"])

    def test_exception_silenced(self):
        class _Boom:
            def suggest(self, *a, **k):
                raise RuntimeError("boom")
        self.engine.query_suggester = _Boom()
        # 不能抛
        self.assertIsNone(self.engine.suggest_query("u1", "开户"))

    def test_empty_partial_returns_none(self):
        self.assertIsNone(self.engine.suggest_query("u1", ""))

    def test_no_history_still_calls_suggester(self):
        # 无历史也应能调用 suggester（传空 list）
        called = {}

        class _Cap:
            def suggest(self, partial, rq, rc):
                called["yes"] = True
                called["rq"], called["rc"] = rq, rc
                return None
        self.engine.query_suggester = _Cap()
        self.engine.suggest_query("u2", "开户")
        self.assertTrue(called.get("yes"))
        self.assertEqual(called["rq"], [])
        self.assertEqual(called["rc"], [])

    def test_async_path(self):
        class _Async:
            async def suggest_async(self, partial, rq, rc):
                return "开户流程"
        self.engine.query_suggester = _Async()
        r = asyncio.run(self.engine.suggest_query_async("u1", "开户"))
        self.assertEqual(r, "开户流程")

    def test_async_exception_silenced(self):
        class _Async:
            async def suggest_async(self, partial, rq, rc):
                raise RuntimeError("boom")
        self.engine.query_suggester = _Async()
        self.assertIsNone(asyncio.run(self.engine.suggest_query_async("u1", "开户")))


class SuggestEndpointTests(unittest.TestCase):
    """/api/search/suggest：参数校验 + 降级 + mock 成功。"""

    def setUp(self):
        from fastapi.testclient import TestClient
        from api.main import create_app, reset_engine
        self.app = create_app()
        self.client = TestClient(self.app)
        self.engine = make_engine()
        self.engine.load_knowledge_base(KB)
        reset_engine(self.engine)

    def test_missing_partial_returns_422(self):
        r = self.client.get("/api/search/suggest?user_id=u1")
        self.assertEqual(r.status_code, 422)

    def test_empty_partial_returns_422(self):
        r = self.client.get("/api/search/suggest?user_id=u1&partial=")
        self.assertEqual(r.status_code, 422)  # min_length=1

    def test_missing_user_id_returns_422(self):
        r = self.client.get("/api/search/suggest?partial=开户")
        self.assertEqual(r.status_code, 422)

    def test_too_long_partial_returns_422(self):
        r = self.client.get(f"/api/search/suggest?user_id=u1&partial={'x' * 101}")
        self.assertEqual(r.status_code, 422)  # max_length=100

    def test_degraded_when_no_api_key(self):
        # engine 用 DeepSeekClient(api_key=None) → enabled=False → 返 None → 降级
        r = self.client.get("/api/search/suggest?user_id=u1&partial=开户")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["completion"], "")
        self.assertEqual(body["source"], "none")

    def test_mocked_completion(self):
        class _Fake:
            async def suggest_async(self, partial, rq, rc):
                return "开户流程"
        self.engine.query_suggester = _Fake()
        r = self.client.get("/api/search/suggest?user_id=u1&partial=开户")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["completion"], "开户流程")
        self.assertEqual(body["source"], "llm")

    def test_mocked_prefix_mismatch_returns_empty(self):
        # suggester 返回不以 partial 开头的串 → 端点二次校验 → 降级
        class _Fake:
            async def suggest_async(self, partial, rq, rc):
                return "转账流程"  # 不以"开户"开头
        self.engine.query_suggester = _Fake()
        r = self.client.get("/api/search/suggest?user_id=u1&partial=开户")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["completion"], "")
        self.assertEqual(body["source"], "none")

    def test_mocked_equal_to_partial_returns_empty(self):
        # 补全等于 partial（无新内容）→ 端点长度校验 → 降级
        class _Fake:
            async def suggest_async(self, partial, rq, rc):
                return "开户"
        self.engine.query_suggester = _Fake()
        r = self.client.get("/api/search/suggest?user_id=u1&partial=开户")
        body = r.json()
        self.assertEqual(body["completion"], "")

    def test_whitespace_only_partial_stripped_to_empty(self):
        # partial="  " → strip 后为空 → 端点返回降级（但 min_length=1 让 "  " 通过Query校验）
        r = self.client.get("/api/search/suggest?user_id=u1&partial=%20%20")
        self.assertEqual(r.status_code, 200)
        body = r.json()
        self.assertEqual(body["completion"], "")
        self.assertEqual(body["source"], "none")


if __name__ == "__main__":
    unittest.main()
