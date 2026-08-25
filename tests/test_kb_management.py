"""M9 知识库管理测试。

覆盖：
  - engine：import_kb_version 建快照 + 置 active；export_kb 往返 hash 一致；
    rollback 恢复旧版本内容并切换 active；embedding_status 字段齐全；
    rollback 未知版本返回 None。
  - store：kb_versions 元数据增/列/查/切 active。
  - API：POST /api/kb/import、GET /api/kb/export、GET /api/kb/versions、
    POST /api/kb/rollback、GET /api/kb/embedding-status。
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from fastapi.testclient import TestClient

from api.main import app, reset_engine
from easysearch import (
    DashScopeClient,
    DeepSeekClient,
    ServiceSearchEngine,
    SQLiteStore,
)

# V1：3 个服务（order / account / transfer）
KB_V1 = [
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

# V2：2 个服务（order / account），且 intro 变化，用于验证回滚恢复 V1 内容
KB_V2 = [
    {
        "service_id": "svc-order",
        "service_name": "订单中心",
        "aliases": ["order"],
        "service_intro": "订单管理（新版）",
        "route": "/orders",
    },
    {
        "service_id": "svc-account",
        "service_name": "账户中心",
        "aliases": ["account"],
        "service_intro": "账户查询（新版）",
        "route": "/account",
    },
]


def _make_engine(db_path: str) -> ServiceSearchEngine:
    """构造带持久化目录的测试引擎（非 :memory:，快照目录才可用）。"""
    return ServiceSearchEngine(
        dashscope_client=DashScopeClient(api_key=None),
        deepseek_client=DeepSeekClient(api_key=None),
        store=SQLiteStore(db_path),
        db_path=db_path,
    )


class StoreKBVersionTests(unittest.TestCase):
    """M9：store.kb_versions 元数据增/列/查/切 active。"""

    def setUp(self):
        self.store = SQLiteStore(":memory:")

    def tearDown(self):
        self.store.close()

    def test_add_and_list(self):
        self.store.kb_version_add("v1", "hash1", "/p/v1.json", 1000.0, active=True)
        self.store.kb_version_add("v2", "hash2", "/p/v2.json", 2000.0, active=True)
        versions = self.store.kb_version_list()
        # 新→旧：v2 在前
        self.assertEqual(versions[0]["version_id"], "v2")
        self.assertTrue(versions[0]["active"])
        self.assertFalse(versions[1]["active"])

    def test_get_returns_none_for_missing(self):
        self.assertIsNone(self.store.kb_version_get("nope"))

    def test_set_active_switches(self):
        self.store.kb_version_add("v1", "hash1", "/p/v1.json", 1000.0, active=True)
        self.store.kb_version_add("v2", "hash2", "/p/v2.json", 2000.0, active=True)
        # 切回 v1
        self.assertTrue(self.store.kb_version_set_active("v1"))
        active = self.store.kb_version_active()
        self.assertEqual(active["version_id"], "v1")
        self.assertTrue(active["active"])

    def test_set_active_missing_returns_false(self):
        self.assertFalse(self.store.kb_version_set_active("nope"))


class EngineKBManagementTests(unittest.TestCase):
    """M9：engine 导入/导出/回滚/embedding 状态。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "t.db")
        self.engine = _make_engine(self.db)

    def tearDown(self):
        self.engine.store.close()

    def test_import_creates_active_version_and_snapshot(self):
        result = self.engine.import_kb_version(KB_V1, version_id="v-test-1")
        self.assertEqual(result["version_id"], "v-test-1")
        self.assertTrue(result["active"])
        self.assertEqual(result["services_count"], 3)
        self.assertTrue(result["kb_hash"])
        # 快照文件落盘
        self.assertTrue(os.path.exists(result["path"]))
        # 版本列表含一条且 active
        versions = self.engine.list_kb_versions()
        self.assertEqual(len(versions), 1)
        self.assertTrue(versions[0]["active"])
        # kb_hash 暴露在 engine 上
        self.assertEqual(self.engine.kb_hash, result["kb_hash"])

    def test_export_reimport_hash_consistent(self):
        r1 = self.engine.import_kb_version(KB_V1, version_id="v-orig")
        exported = self.engine.export_kb()
        self.assertEqual(len(exported), 3)
        # 导出形态再导入 → kb_hash 一致（内容寻址）
        r2 = self.engine.import_kb_version(exported, version_id="v-reimport")
        self.assertEqual(r2["kb_hash"], r1["kb_hash"])

    def test_rollback_restores_previous_content(self):
        r1 = self.engine.import_kb_version(KB_V1, version_id="v-1")
        r2 = self.engine.import_kb_version(KB_V2, version_id="v-2")
        # 当前为 V2（2 服务，hash 与 V1 不同）
        self.assertEqual(len(self.engine.services), 2)
        self.assertNotEqual(r2["kb_hash"], r1["kb_hash"])
        # 回滚到 V1
        result = self.engine.rollback_kb("v-1")
        self.assertIsNotNone(result)
        self.assertTrue(result["active"])
        self.assertEqual(result["services_count"], 3)
        # 内容恢复：3 服务 + kb_hash 回到 V1
        self.assertEqual(len(self.engine.services), 3)
        self.assertIn("svc-transfer", self.engine.services)
        self.assertEqual(self.engine.kb_hash, r1["kb_hash"])
        # active 已切换：V1 active、V2 非 active
        versions = {v["version_id"]: v for v in self.engine.list_kb_versions()}
        self.assertTrue(versions["v-1"]["active"])
        self.assertFalse(versions["v-2"]["active"])

    def test_rollback_unknown_returns_none(self):
        self.engine.import_kb_version(KB_V1, version_id="v-1")
        self.assertIsNone(self.engine.rollback_kb("nope"))

    def test_embedding_status_fields(self):
        self.engine.import_kb_version(KB_V1, version_id="v-1")
        status = self.engine.embedding_status()
        self.assertEqual(status["total"], 3)
        self.assertEqual(status["embedded"], 3)  # 离线 fallback 向量已索引
        self.assertFalse(status["in_progress"])
        self.assertTrue(status["kb_hash"])
        self.assertEqual(status["last_error"], "")

    def test_import_rejects_empty_payload(self):
        with self.assertRaises(ValueError):
            self.engine.import_kb_version([], version_id="v-empty")

    def test_import_rejects_non_list(self):
        with self.assertRaises(ValueError):
            self.engine.import_kb_version({"not": "a list"}, version_id="v-bad")  # type: ignore[arg-type]


class APIKBManagementTests(unittest.TestCase):
    """M9：/api/kb/* 端点闭环。"""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "t.db")
        self.engine = _make_engine(self.db)
        reset_engine(self.engine)
        self.client = TestClient(app)

    def tearDown(self):
        self.engine.store.close()

    def test_import_export_versions_rollback_loop(self):
        # 1) 导入 V1
        r = self.client.post("/api/kb/import", json=KB_V1)
        self.assertEqual(r.status_code, 200)
        v1 = r.json()
        self.assertEqual(v1["services_count"], 3)
        self.assertTrue(v1["active"])
        v1_id = v1["version_id"]
        v1_hash = v1["kb_hash"]

        # 2) 版本列表含一条 active
        r = self.client.get("/api/kb/versions")
        self.assertEqual(r.status_code, 200)
        versions = r.json()
        self.assertEqual(len(versions), 1)
        self.assertTrue(versions[0]["active"])

        # 3) embedding-status
        r = self.client.get("/api/kb/embedding-status")
        self.assertEqual(r.status_code, 200)
        st = r.json()
        self.assertEqual(st["total"], 3)
        self.assertFalse(st["in_progress"])
        self.assertEqual(st["kb_hash"], v1_hash)

        # 4) 导出 → 再导入，hash 一致（内容寻址）
        r = self.client.get("/api/kb/export")
        self.assertEqual(r.status_code, 200)
        exported = r.json()
        self.assertEqual(len(exported), 3)
        r = self.client.post("/api/kb/import", json=exported)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["kb_hash"], v1_hash)

        # 5) 导入 V2（不同内容），再回滚到 V1
        r = self.client.post("/api/kb/import", json=KB_V2)
        self.assertEqual(r.status_code, 200)
        v2_id = r.json()["version_id"]
        self.assertNotEqual(v2_id, v1_id)

        # 当前 active 为 V2
        versions = self.client.get("/api/kb/versions").json()
        active = next(v for v in versions if v["active"])
        self.assertEqual(active["version_id"], v2_id)

        # 回滚到 V1
        r = self.client.post(
            "/api/kb/rollback", params={"version_id": v1_id}
        )
        self.assertEqual(r.status_code, 200)
        rb = r.json()
        self.assertEqual(rb["version_id"], v1_id)
        self.assertEqual(rb["services_count"], 3)

        # 回滚后 active 切换到 V1
        versions = self.client.get("/api/kb/versions").json()
        active = next(v for v in versions if v["active"])
        self.assertEqual(active["version_id"], v1_id)

    def test_rollback_unknown_404(self):
        r = self.client.post("/api/kb/rollback", params={"version_id": "nope"})
        self.assertEqual(r.status_code, 404)

    def test_import_empty_400(self):
        r = self.client.post("/api/kb/import", json=[])
        self.assertEqual(r.status_code, 400)

    def test_export_empty_kb_409(self):
        # 未导入任何 KB → 导出 409
        r = self.client.get("/api/kb/export")
        self.assertEqual(r.status_code, 409)


if __name__ == "__main__":
    unittest.main()
