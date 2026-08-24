import unittest

import json

from easysearch import DashScopeClient, ServiceSearchEngine


SERVICES = [
    {
        "service_id": "svc-1",
        "service_name": "订单中心",
        "aliases": ["订单", "order"],
        "service_intro": "查看与管理订单信息",
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
        "route": "/risk",
    },
]


class ServiceSearchEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ServiceSearchEngine()
        self.engine.load_knowledge_base(SERVICES)

    def test_search_returns_clickable_fields_and_reason(self) -> None:
        results = self.engine.search("u-1", "订单审批")
        self.assertTrue(results)
        first = results[0]
        self.assertIn("route", first)
        self.assertIn("component", first)
        self.assertIn("decision_button", first)
        self.assertIn("rerank_reason", first)

    def test_homepage_dropdown_contains_recent_and_hot(self) -> None:
        self.engine.search("u-1", "订单")
        self.engine.search("u-1", "用户")
        self.engine.search("u-1", "风控")

        self.engine.record_click("u-1", "svc-1")
        self.engine.record_click("u-1", "svc-2")
        self.engine.record_click("u-1", "svc-3")
        self.engine.record_click("u-2", "svc-3")

        dropdown = self.engine.homepage_dropdown("u-1")

        self.assertEqual(dropdown["recent_queries"], ["风控", "用户", "订单"])
        self.assertEqual(dropdown["recent_clicked_services"], ["风控平台", "用户中心", "订单中心"])
        self.assertEqual(dropdown["global_hot_services"][0], "风控平台")

    def test_hybrid_score_formula(self) -> None:
        score = self.engine._hybrid_score(0.8, 0.5, 1.0)
        self.assertAlmostEqual(score, 0.73)

    def test_query_history_over_ten_uses_optimizer_path(self) -> None:
        for idx in range(11):
            self.engine.search("u-h", f"查询{idx}")
        self.assertEqual(len(self.engine.user_query_history["u-h"]), 11)

    def test_dashscope_payloads_are_used_when_api_key_exists(self) -> None:
        calls: list[tuple[str, dict, dict]] = []

        def requester(url: str, body: bytes, headers: dict[str, str]) -> dict:
            payload = json.loads(body.decode("utf-8"))
            calls.append((url, payload, headers))

            if "text-embedding" in url:
                return {"output": {"embeddings": [{"embedding": [1.0, 0.0, 0.0]}]}}
            if "text-rerank" in url:
                docs = payload["input"]["documents"]
                return {
                    "output": {
                        "results": [
                            {"index": index, "relevance_score": float(len(docs) - index)}
                            for index in range(len(docs))
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
        engine = ServiceSearchEngine(dashscope_client=client)
        engine.load_knowledge_base(SERVICES)
        results = engine.search("u-api", "订单审批")

        self.assertTrue(results)
        self.assertTrue(any("text-embedding/text-embedding" in url for url, _, _ in calls))
        self.assertTrue(any("rerank/text-rerank/text-rerank" in url for url, _, _ in calls))
        self.assertTrue(any("compatible-mode/v1/chat/completions" in url for url, _, _ in calls))
        embedding_call = next(payload for url, payload, _ in calls if "text-embedding/text-embedding" in url)
        self.assertEqual(embedding_call["model"], "qwen3.7-text-embedding")
        self.assertIn("texts", embedding_call["input"])
        self.assertTrue(results[0]["rerank_reason"])


if __name__ == "__main__":
    unittest.main()
