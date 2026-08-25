"""M16 答案模式测试。

覆盖：
  - IntentRouter：guide 意图分类 + 优先级（guide > informational）。
  - GuideGenerator._parse_guide：内联 [[service_id]] 解析 + 白名单过滤 + 非法引用剔除。
  - engine.search_guide(_async)：LLM 不可用降级 list 模式；mock 后返回 answer_guide。
"""
from __future__ import annotations

import asyncio
import unittest

from easysearch import (
    DEFAULT,
    GUIDE,
    INFORMATIONAL,
    DashScopeClient,
    DeepSeekClient,
    GuideGenerator,
    IntentRouter,
    ServiceSearchEngine,
    SQLiteStore,
)

KB = [
    {"service_id": "svc-A", "service_name": "开户平台", "aliases": ["开户", "open"],
     "service_intro": "账户开户 网上开户", "route": "/open"},
    {"service_id": "svc-B", "service_name": "转账平台", "aliases": ["转账", "transfer"],
     "service_intro": "银证转账 资金划转", "route": "/transfer"},
    {"service_id": "svc-C", "service_name": "综合金融门户", "aliases": ["一站式"],
     "service_intro": "开户与转账一站式综合服务", "route": "/portal"},
]


def make_engine(db_path: str = ":memory:"):
    return ServiceSearchEngine(
        dashscope_client=DashScopeClient(api_key=None),
        deepseek_client=DeepSeekClient(api_key=None),
        store=SQLiteStore(db_path),
        db_path=db_path,
    )


def _cand(sid):
    return {
        "service_id": sid, "service_name": sid + "-name", "aliases": [],
        "service_intro": "intro", "route": f"/{sid}", "component": "Comp",
        "decision_button": "进入", "derived": False, "score": 0.5,
    }


class IntentGuideTests(unittest.TestCase):
    def setUp(self):
        self.router = IntentRouter()

    def test_guide_hints(self):
        for q in ["新手如何开始", "新手入门", "操作流程", "怎么操作", "怎么玩", "使用流程"]:
            r = self.router.classify(q, services={})
            self.assertEqual(r.intent, GUIDE, f"query={q} got {r.intent}")

    def test_guide_priority_over_informational(self):
        # "怎么操作" 含通用疑问词「怎么」，但 guide 短语优先 → guide
        r = self.router.classify("怎么操作", services={})
        self.assertEqual(r.intent, GUIDE)

    def test_info_not_swallowed_by_guide(self):
        # "如何转账" 含「如何」但非 guide 短语 → informational
        r = self.router.classify("如何转账", services={})
        self.assertEqual(r.intent, INFORMATIONAL)

    def test_default_unrelated(self):
        r = self.router.classify("开户", services={})
        self.assertEqual(r.intent, DEFAULT)


class GuideParseTests(unittest.TestCase):
    """GuideGenerator._parse_guide 纯逻辑（构造 fake LLM 响应）。"""

    def setUp(self):
        # 无 key 的 client，generate_guide 会返回 None，但 _parse_guide 可直接调
        self.gen = GuideGenerator(DeepSeekClient(api_key=None))
        self.candidates = [_cand("svc-A"), _cand("svc-B"), _cand("svc-C")]

    def _resp(self, content):
        return {"choices": [{"message": {"content": content}}]}

    def test_parse_valid_inline_refs(self):
        resp = self._resp(
            '{"steps":[{"step_text":"1. 开卡 [[svc-A]]"},{"step_text":"2. 转账 [[svc-B]] 和 [[svc-C]]"}]}'
        )
        guide = self.gen._parse_guide(resp, self.candidates)
        self.assertIsNotNone(guide)
        self.assertEqual(len(guide["steps"]), 2)
        self.assertEqual(guide["steps"][0]["services"][0]["service_id"], "svc-A")
        self.assertEqual(guide["steps"][0]["services"][0]["route"], "/svc-A")
        self.assertEqual(len(guide["steps"][1]["services"]), 2)

    def test_parse_filters_invalid_refs(self):
        # svc-X 不在白名单 → 过滤；保留 svc-A
        resp = self._resp(
            '{"steps":[{"step_text":"1. 开卡 [[svc-A]] 然后 [[svc-X]]"}]}'
        )
        guide = self.gen._parse_guide(resp, self.candidates)
        self.assertIsNotNone(guide)
        sids = [s["service_id"] for s in guide["steps"][0]["services"]]
        self.assertEqual(sids, ["svc-A"])  # svc-X 被剔除

    def test_parse_array_form_compatible(self):
        # LLM 直接输出数组（非 {steps:[]}）
        resp = self._resp('["1. 开卡 [[svc-A]]", "2. 转账 [[svc-B]]"]')
        guide = self.gen._parse_guide(resp, self.candidates)
        self.assertIsNotNone(guide)
        self.assertEqual(len(guide["steps"]), 2)

    def test_parse_empty_steps_returns_none(self):
        resp = self._resp('{"steps":[]}')
        self.assertIsNone(self.gen._parse_guide(resp, self.candidates))

    def test_parse_garbage_returns_none(self):
        resp = self._resp("这不是JSON")
        self.assertIsNone(self.gen._parse_guide(resp, self.candidates))

    def test_parse_strips_html_and_limits_length(self):
        long_text = "步骤" + "x" * 400
        resp = self._resp(f'{{"steps":[{{"step_text":"<script>{long_text}</script> [[svc-A]]"}}]}}')
        guide = self.gen._parse_guide(resp, self.candidates)
        self.assertIsNotNone(guide)
        txt = guide["steps"][0]["step_text"]
        self.assertNotIn("<script>", txt)
        self.assertLessEqual(len(txt), 303)  # 300 + 省略号


class EngineGuideTests(unittest.TestCase):
    """engine.search_guide：降级 + mock answer_guide。"""

    def setUp(self):
        self.engine = make_engine()
        self.engine.load_knowledge_base(KB)

    def test_fallback_to_list_when_no_api_key(self):
        # 无 DeepSeek key → generate_guide 返回 None → 降级 list 模式
        r = self.engine.search_guide("u1", "新手如何开始")
        self.assertIsNone(r["answer_guide"])
        self.assertGreater(len(r["results"]), 0)

    def test_mocked_guide_returns_answer(self):
        # 注入 fake generator 返回 canned guide
        class _FakeGen:
            def generate_guide(self, query, candidates):
                return {"steps": [
                    {"step_text": f"1. 开户 [[svc-A]]", "services": []}
                ]}
        self.engine.guide_generator = _FakeGen()
        r = self.engine.search_guide("u1", "新手如何开始")
        self.assertIsNotNone(r["answer_guide"])
        self.assertEqual(r["answer_guide"]["query"], "新手如何开始")
        self.assertEqual(len(r["answer_guide"]["steps"]), 1)
        # answer_guide 与 results 互斥
        self.assertEqual(r["results"], [])

    def test_mocked_guide_async(self):
        class _FakeGen:
            async def generate_guide_async(self, query, candidates):
                return {"steps": [
                    {"step_text": f"1. 开户 [[svc-A]] → 2. 转账 [[svc-B]]", "services": []}
                ]}
        self.engine.guide_generator = _FakeGen()
        r = asyncio.run(self.engine.search_guide_async("u1", "操作流程"))
        self.assertIsNotNone(r["answer_guide"])
        self.assertEqual(len(r["answer_guide"]["steps"]), 1)

    def test_mocked_guide_empty_steps_falls_back(self):
        # generator 返回空 steps → 降级 list
        class _FakeGen:
            def generate_guide(self, query, candidates):
                return {"steps": []}
        self.engine.guide_generator = _FakeGen()
        r = self.engine.search_guide("u1", "新手如何开始")
        self.assertIsNone(r["answer_guide"])
        self.assertGreater(len(r["results"]), 0)


if __name__ == "__main__":
    unittest.main()
