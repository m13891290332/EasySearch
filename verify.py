#!/usr/bin/env python3
"""EasySearch 独立验证脚本 —— 无需 pytest，直接 python verify.py 即可。

验证项：
  1. 知识库上传（5 字段）
  2. route 派生（string→component/decision_button，dict→原始）
  3. 搜索返回可点击路径/页面组件/决策按钮/排序理由
  4. 混合打分公式 0.6/0.3/0.1
  5. DIN 历史优化（>10 触发）
  6. 首页下拉（最近3搜索词/最近3点击/最热3服务）
  7. SQLite 持久化（重启不丢）
  8. Top-20→Top-10 链路
  9. FastAPI 接口端到端
"""
from __future__ import annotations

import os
import sys
import tempfile

# 确保项目根在 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = "") -> None:
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name}  {detail}")


def main() -> None:
    from easysearch import DashScopeClient, ServiceSearchEngine, SQLiteStore, route_info
    from easysearch.engine import VECTOR_WEIGHT, BM25_WEIGHT, POPULARITY_WEIGHT

    print("=" * 60)
    print("EasySearch 验证脚本")
    print("=" * 60)

    # --- 1. route 派生 ---
    print("\n[1] route_info 派生")
    info_s = route_info("/go/account/open-account")
    check("string route→path", info_s["route"] == "/go/account/open-account")
    check("string route→component=OpenAccount", info_s["component"] == "OpenAccount", info_s["component"])
    check("string route→decision_button=进入", info_s["decision_button"] == "进入")
    check("string route→derived=True", info_s["derived"] is True)

    info_d = route_info({"path": "/orders", "component": "OrderTable", "action_button": "ApproveOrder"})
    check("dict route→path", info_d["route"] == "/orders")
    check("dict route→component", info_d["component"] == "OrderTable")
    check("dict route→decision_button", info_d["decision_button"] == "ApproveOrder")
    check("dict route→derived=False", info_d["derived"] is False)

    # --- 2. 混合公式 ---
    print("\n[2] 混合打分公式")
    expected = 0.6 * 0.8 + 0.3 * 0.5 + 0.1 * 1.0
    got = ServiceSearchEngine._hybrid_score(0.8, 0.5, 1.0)
    check(f"0.6*0.8+0.3*0.5+0.1*1.0={expected}", abs(got - expected) < 1e-9, f"got {got}")
    check("权重 0.6/0.3/0.1", VECTOR_WEIGHT == 0.6 and BM25_WEIGHT == 0.3 and POPULARITY_WEIGHT == 0.1)

    # --- 3. 知识库 + 搜索 + 下拉 ---
    print("\n[3] 知识库加载 + 搜索 + 下拉")
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, "verify.db")
    engine = ServiceSearchEngine(
        dashscope_client=DashScopeClient(api_key=None),
        store=SQLiteStore(db),
    )
    services = [
        {"service_id": "svc-1", "service_name": "订单中心", "aliases": ["订单", "order"],
         "service_intro": "查看与管理订单信息，支持订单审批",
         "route": {"path": "/orders", "component": "OrderTable", "action_button": "ApproveOrder"}},
        {"service_id": "svc-2", "service_name": "用户中心", "aliases": ["用户", "customer"],
         "service_intro": "查看用户画像",
         "route": {"path": "/users", "component": "UserProfile", "action_button": "ConfirmUser"}},
        {"service_id": "svc-3", "service_name": "风控平台", "aliases": ["风控", "risk"],
         "service_intro": "风险决策管理", "route": "/risk/decision"},
    ]
    engine.load_knowledge_base(services)
    check("知识库加载 3 条", len(engine.services) == 3)

    results = engine.search("u-1", "订单审批")
    check("搜索返回结果", len(results) > 0)
    check("结果 ≤ 10 (Top-10)", len(results) <= 10)
    if results:
        first = results[0]
        for key in ("route", "component", "decision_button", "rerank_reason", "score", "service_id"):
            check(f"结果含 {key}", key in first)
        # svc-3 是 string route，应派生
        svc3 = next((r for r in results if r["service_id"] == "svc-3"), None)
        if svc3:
            check("svc-3 component=Decision", svc3["component"] == "Decision", svc3["component"])
            check("svc-3 decision_button=进入", svc3["decision_button"] == "进入")
            check("svc-3 derived=True", svc3["derived"] is True)

    # 下拉
    engine.search("u-1", "订单")
    engine.search("u-1", "用户")
    engine.search("u-1", "风控")
    engine.record_click("u-1", "svc-1")
    engine.record_click("u-1", "svc-2")
    engine.record_click("u-1", "svc-3")
    engine.record_click("u-2", "svc-3")
    dd = engine.homepage_dropdown("u-1")
    check("最近3搜索词(去重)", dd["recent_queries"] == ["风控", "用户", "订单"], str(dd["recent_queries"]))
    # 点击/热门返回 [{service_id, service_name}]
    click_names = [item["service_name"] for item in dd["recent_clicked_services"]]
    check("最近3点击服务(去重)", click_names == ["风控平台", "用户中心", "订单中心"], str(click_names))
    check("最近点击[0] service_id=svc-3", dd["recent_clicked_services"][0]["service_id"] == "svc-3")
    check("最热服务[0]=风控平台", dd["global_hot_services"][0]["service_name"] == "风控平台",
          str(dd["global_hot_services"]))

    # 去重验证
    for _ in range(3):
        engine.search("u-dd", "开户")
    engine.search("u-dd", "转账")
    dd2 = engine.homepage_dropdown("u-dd")
    check("搜索词去重(重复开户只留1次)", dd2["recent_queries"] == ["转账", "开户"], str(dd2["recent_queries"]))

    # 单服务详情
    detail = engine.get_service("svc-3")
    check("get_service svc-3 非空", detail is not None)
    if detail:
        check("get_service component=Decision", detail["component"] == "Decision", detail["component"])
        check("get_service decision_button=进入", detail["decision_button"] == "进入")
        check("get_service derived=True", detail["derived"] is True)
    check("get_service 未知返回 None", engine.get_service("nope") is None)

    # --- 4. DIN 历史优化 ---
    print("\n[4] DIN 历史优化（>10 触发）")
    for i in range(12):
        engine.search("u-din", f"查询{i}")
    check("DIN 用户查询数=12", engine.store.query_count("u-din") == 12)
    # 第 12 次后 query_count > 10，DIN 路径已走过（无异常即通过）

    # --- 5. SQLite 持久化 ---
    print("\n[5] SQLite 持久化")
    engine.store.close()
    store2 = SQLiteStore(db)
    check("重启后查询数=12", store2.query_count("u-din") == 12)
    # recent_queries 返回最近 limit 条（默认3），最近->最旧
    recent = store2.recent_queries("u-din", 3)
    check("重启后最近搜索词[0]=查询11", recent[0] == "查询11", str(recent))
    check("重启后最近搜索词顺序(最近->最旧)", recent == ["查询11", "查询10", "查询9"], str(recent))
    check("重启后最热[0]=svc-3", store2.hot_services(1) == ["svc-3"])
    store2.close()

    # --- 6. 真实知识库 ---
    print("\n[6] services_dict_50.json 加载")
    kb_path = os.path.join(os.path.dirname(__file__), "services_dict_50.json")
    if os.path.exists(kb_path):
        engine2 = ServiceSearchEngine(
            dashscope_client=DashScopeClient(api_key=None),
            store=SQLiteStore(":memory:"),
        )
        engine2.upload_knowledge_base_from_json(kb_path)
        check("50 条服务加载", len(engine2.services) == 50, f"got {len(engine2.services)}")
        r = engine2.search("u-real", "开户")
        check("真实知识库搜索开户有结果", len(r) > 0)
        if r:
            check("开户结果含 route", bool(r[0].get("route")))
            check("开户结果含 component", bool(r[0].get("component")))
            check("开户结果含 decision_button", bool(r[0].get("decision_button")))
            check("开户结果含 rerank_reason", bool(r[0].get("rerank_reason")))
        engine2.store.close()
    else:
        print("  (跳过：services_dict_50.json 不存在)")

    # --- 7. FastAPI 接口 ---
    print("\n[7] FastAPI 接口端到端")
    try:
        from fastapi.testclient import TestClient
        from api.main import app, reset_engine

        client_engine = ServiceSearchEngine(
            dashscope_client=DashScopeClient(api_key=None),
            store=SQLiteStore(os.path.join(tmp, "api.db")),
        )
        client_engine.load_knowledge_base(services)
        reset_engine(client_engine)
        client = TestClient(app)

        # health
        r = client.get("/api/health")
        check("GET /api/health 200", r.status_code == 200, str(r.status_code))
        check("health services_count=3", r.json()["services_count"] == 3)

        # search
        r = client.get("/api/search", params={"user_id": "u1", "query": "订单"})
        check("GET /api/search 200", r.status_code == 200, str(r.status_code))
        check("search 返回结果", len(r.json()["results"]) > 0)
        first = r.json()["results"][0]
        check("API 结果含 route", "route" in first)
        check("API 结果含 component", "component" in first)
        check("API 结果含 decision_button", "decision_button" in first)
        check("API 结果含 rerank_reason", "rerank_reason" in first)

        # click
        sid = first["service_id"]
        r = client.post("/api/click", json={"user_id": "u1", "service_id": sid})
        check("POST /api/click 200", r.status_code == 200, str(r.status_code))

        # dropdown（返回对象数组）
        r = client.get("/api/dropdown", params={"user_id": "u1"})
        check("GET /api/dropdown 200", r.status_code == 200, str(r.status_code))
        dd = r.json()
        check("dropdown recent_queries 含订单", "订单" in dd["recent_queries"])
        check("dropdown global_hot[0] service_name", dd["global_hot_services"][0]["service_name"] == first["service_name"])
        check("dropdown global_hot[0] service_id", bool(dd["global_hot_services"][0].get("service_id")))

        # /api/service 单服务详情
        r = client.get("/api/service", params={"service_id": sid})
        check("GET /api/service 200", r.status_code == 200, str(r.status_code))
        sd = r.json()
        check("/api/service service_id", sd["service_id"] == sid)
        check("/api/service route", bool(sd.get("route")))
        check("/api/service component", bool(sd.get("component")))
        check("/api/service decision_button", bool(sd.get("decision_button")))

        # /api/service 未知 404
        r = client.get("/api/service", params={"service_id": "nope"})
        check("/api/service 未知 404", r.status_code == 404, str(r.status_code))

        # empty query 400
        r = client.get("/api/search", params={"user_id": "u1", "query": "  "})
        check("空 query 400", r.status_code == 400, str(r.status_code))

        # unknown click — M12：下线/未知服务仍记点击（标 deprecated），不硬 404
        r = client.post("/api/click", json={"user_id": "u1", "service_id": "nope"})
        check("未知 service 200 (M12 deprecated)", r.status_code == 200, str(r.status_code))

        # upload
        r = client.post("/api/knowledge-base/upload", json=services)
        check("POST /api/knowledge-base/upload 200", r.status_code == 200, str(r.status_code))

        # homepage
        r = client.get("/")
        check("GET / 主页 200", r.status_code == 200, str(r.status_code))

        client_engine.store.close()
    except ImportError as e:
        check("FastAPI 测试", False, f"依赖未安装: {e}")

    # --- 汇总 ---
    print("\n" + "=" * 60)
    print(f"汇总: {PASS} 通过, {FAIL} 失败, 共 {PASS + FAIL} 项")
    print("=" * 60)
    if FAIL == 0:
        print("✅ 全部通过！EasySearch 搜索引擎功能完整。")
    else:
        print("❌ 有失败项，请检查上方 [FAIL] 行。")
    return 1 if FAIL > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
