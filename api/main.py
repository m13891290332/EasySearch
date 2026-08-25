from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from easysearch import ServiceSearchEngine
from easysearch.dashscope import aclose_async_client
from easysearch.intent import CONVERSATIONAL, GUIDE, MULTI_CONDITION
from easysearch.logging_config import setup_logging
from easysearch.metrics import get_metrics
from easysearch.models import route_info
from easysearch.safety import PromptInjectionError
from easysearch.utils import tokenize

from .schemas import (
    ActionExecuteRequest,
    ActionExecuteResponse,
    AnswerGuide,
    AutocompleteItem,
    AutocompleteResponse,
    AutocompleteTag,
    ClickRequest,
    CombinationGroup,
    CombinationStep,
    DegradationStats,
    DeepComponentItem,
    DeepComponentRequest,
    DeepComponentResponse,
    DropdownItem,
    DropdownResponse,
    EmbeddingStatusResponse,
    FeedbackRequest,
    HealthResponse,
    HighLatencyQueryStat,
    IntersectionSearchRequest,
    KBImportResponse,
    KBOpLogItem,
    KBRollbackResponse,
    KBVersionInfo,
    KnowledgeBaseItem,
    NoClickQueryStat,
    NotFoundInfo,
    SearchLogItem,
    SearchResponse,
    SearchResultItem,
    ServiceDetail,
    SessionSearchRequest,
    SessionSearchResponse,
    SuggestResponse,
    UploadResponse,
)
from .metrics import register_metrics_routes

logger = logging.getLogger(__name__)

_engine: ServiceSearchEngine | None = None
_FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
_DEFAULT_KB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "services_dict_50.json")
_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_ACTIVE_KB = os.path.join(_DATA_DIR, "active_kb.json")


def _persist_kb(payload: list[dict[str, Any]]) -> None:
    """将上传/导入的 KB 持久化到 data/active_kb.json，重启后自动加载。"""
    os.makedirs(_DATA_DIR, exist_ok=True)
    with open(_ACTIVE_KB, "w", encoding="utf-8") as fp:
        json.dump(payload, fp, ensure_ascii=False, indent=2)


def get_engine() -> ServiceSearchEngine:
    global _engine
    if _engine is None:
        os.makedirs(_DATA_DIR, exist_ok=True)
        db_path = os.path.join(_DATA_DIR, "easysearch.db")
        _engine = ServiceSearchEngine(db_path=db_path)
        # 优先加载用户上传的持久化 KB，其次环境变量指定的 KB，最后默认 KB
        kb_path = os.getenv("EASYSEARCH_KB", "")
        if kb_path and os.path.exists(kb_path):
            _engine.upload_knowledge_base_from_json(kb_path)
        elif os.path.exists(_ACTIVE_KB):
            _engine.upload_knowledge_base_from_json(_ACTIVE_KB)
        elif os.path.exists(_DEFAULT_KB):
            _engine.upload_knowledge_base_from_json(_DEFAULT_KB)
    return _engine


def reset_engine(engine: ServiceSearchEngine) -> None:
    """测试/重载入口。"""
    global _engine
    _engine = engine


def _sse(event: str, data: dict[str, Any]) -> str:
    """M10-5：格式化一条 Server-Sent Events 消息（``event: <name>\ndata: <json>\n\n``）。"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # M10：启动即配置结构化 JSON 日志（structlog 可选降级 stdlib）
    setup_logging()
    get_engine()  # 启动时初始化并加载默认知识库
    # 缓存预热：触发 Redis 连接（不可用时自动降级内存 LRU）
    from easysearch.cache import get_cache

    get_cache()
    yield
    # M2：关闭 httpx.AsyncClient 连接池，避免资源泄漏
    await aclose_async_client()
    # 关闭结果缓存（Redis 连接池等资源）
    from easysearch.cache import get_cache as _get_cache, reset_cache

    try:
        _get_cache().close()
    except Exception:  # pragma: no cover - 关闭失败不影响退出
        pass
    reset_cache()


def create_app() -> FastAPI:
    app = FastAPI(title="EasySearch 服务搜索引擎", version="2.0.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    # M12：错误处理与降级——API Key 鉴权 + 上传体积上限 + 慢速限流
    # Starlette 中间件为 LIFO 栈：后添加的为外层。请求执行顺序 = 限流 → 体积 → 鉴权 → 路由
    # 限流在最外层挡突发流量；体积上限防止超大 payload 耗资源；鉴权最后验证身份
    # 默认配置：env 未设置则中间件透传，离线/测试不受影响
    from .auth import ApiKeyMiddleware, BodySizeLimitMiddleware, RateLimitMiddleware

    app.add_middleware(ApiKeyMiddleware)
    app.add_middleware(BodySizeLimitMiddleware)
    app.add_middleware(RateLimitMiddleware)

    @app.get("/api/health", response_model=HealthResponse, tags=["meta"])
    def health() -> HealthResponse:
        """健康检查 + M10 监控摘要。

        前三字段（status/services_count/dashscope_enabled）保持兼容；
        M10 字段叠加最近 100 次搜索 P95/错误率/缓存命中率 + 外部调用健康度 + embedding 状态。
        """
        engine = get_engine()
        metrics_summary = engine.metrics.health_summary()
        emb = engine.embedding_status()
        return HealthResponse(
            status="ok",
            services_count=len(engine.services),
            dashscope_enabled=engine.dashscope_client.enabled,
            search_total=metrics_summary["search_total"],
            error_rate=metrics_summary["error_rate"],
            recent_total=metrics_summary["recent_total"],
            recent_error_rate=metrics_summary["recent_error_rate"],
            p95_ms=metrics_summary["p95_ms"],
            cache_hit_rate=metrics_summary["cache_hit_rate"],
            external=metrics_summary["external"],
            kb_embedding_in_progress=metrics_summary["kb_embedding_in_progress"],
            kb_hash=emb.get("kb_hash") or None,
            last_error=emb.get("last_error") or None,
        )

    @app.get("/metrics", tags=["meta"])
    def prometheus_metrics() -> Response:
        """M10：Prometheus 抓取端点。

        prometheus_client 可用时走标准 ``generate_latest``（含 Histogram 分桶）；
        不可用时由 MetricsCollector 手写 exposition 文本，抓取方仍可解析。
        响应 Content-Type 遵循 Prometheus 文本 exposition 约定。
        """
        body = get_metrics().prometheus_text()
        return Response(content=body, media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.get("/api/search", response_model=SearchResponse, tags=["search"])
    async def search(
        user_id: str = Query(..., description="用户ID"),
        query: str = Query(..., description="搜索词"),
        mode: str = Query("auto", description="检索模式：auto=按意图自动路由"),
        retrieval_mode: str = Query(
            "hybrid",
            description="检索模式：keyword=仅关键词 / semantic=仅语义 / hybrid=混合",
        ),
        session_id: str | None = Query(None, description="会话ID（传入且为会话型意图时走 M7 长程对话）"),
    ) -> SearchResponse:
        if not query.strip():
            raise HTTPException(status_code=400, detail="query 不能为空")
        if retrieval_mode not in ("keyword", "semantic", "hybrid"):
            raise HTTPException(
                status_code=400,
                detail="retrieval_mode 必须为 keyword|semantic|hybrid",
            )
        engine = get_engine()
        if not engine.services:
            raise HTTPException(status_code=409, detail="知识库为空，请先上传")
        # 需求1：DeepSeek 语义意图预分类——query 进入检索前的第一道分类。
        # normal_financial/colloquial → 既有规则路由 + 检索；
        # generalized_combination → 组合卡片包组（需求2）；
        # irrelevant → 未命中提示，不调检索、不胡编服务（需求3）。
        classification = await engine.classify_query_async(query)
        intent_category = classification.category

        # 需求3：无关消息 / 无关 prompt / 提示词攻击 → 未命中提示，不胡编不存在服务
        if intent_category == "irrelevant":
            # 安全网：LLM 可能误判金融查询为 irrelevant（如 "大宗" 被判无关，
            # 但 KB 有 "大宗交易"）。先做 BM25 + 名称/别名子串匹配，
            # 有命中则降级为 normal_financial 继续检索，不阻断。
            _q_tokens = tokenize(query)
            _bm25 = engine._mf_bm25.batch_score_tokens(_q_tokens) if _q_tokens else {}
            _has_bm25 = any(s > 0 for s in _bm25.values())
            _ql = query.lower().strip()
            _has_substr = any(
                _ql in svc.service_name.lower()
                or any(_ql in a.lower() for a in svc.aliases)
                for svc in engine.services.values()
            ) if _ql else False
            if not _has_bm25 and not _has_substr:
                return SearchResponse(
                    user_id=user_id,
                    query=query,
                    intent="irrelevant",
                    intent_category="irrelevant",
                    match_mode="not_found",
                    not_found=NotFoundInfo(
                        message="未命中：您查询的内容不在金融服务范围内，未找到相关服务。",
                        category=classification.sub_category or "off_topic",
                        hint="请输入与平台金融服务相关的查询，例如：开户、银证转账、新股申购。",
                    ),
                    results=[],
                )
            # BM25 或子串命中 → LLM 误判，降级为正常检索
            intent_category = "normal_financial"

        # 需求2：泛化需求组合回复 → 每步 top1 按序组合为卡片包组
        if (
            intent_category == "generalized_combination"
            and classification.combination_steps
        ):
            step_bundles = await engine.search_combination_async(
                user_id, classification.combination_steps
            )
            combo_steps = [
                CombinationStep(
                    step_label=b["step_label"],
                    step_query=b["step_query"],
                    results=[SearchResultItem(**it) for it in b["results"]],
                )
                for b in step_bundles
            ]
            title = "组合查找：" + " → ".join(classification.combination_steps)
            return SearchResponse(
                user_id=user_id,
                query=query,
                intent="generalized_combination",
                intent_category="generalized_combination",
                match_mode="combination",
                combination=CombinationGroup(title=title, steps=combo_steps),
                results=[],
            )

        # 正常金融服务查找 / 口语化查找 → 既有规则意图路由 + 检索
        # colloquial：用「按金融术语理解 + 追加金融名词」后的 augmented_query 做检索，
        # 其余用原 query（normal_financial 下与原行为完全一致）。
        # 安全网：若原始 query 已在 KB 中有 BM25/子串命中（说明是标准术语），
        # 则不 augment，避免 LLM 追加无关词稀释检索。
        if intent_category == "colloquial" and classification.augmented_query:
            _q_tokens = tokenize(query)
            _bm25 = engine._mf_bm25.batch_score_tokens(_q_tokens) if _q_tokens else {}
            _has_bm25 = any(s > 0 for s in _bm25.values())
            _ql = query.lower().strip()
            _has_substr = any(
                _ql in svc.service_name.lower()
                or any(_ql in a.lower() for a in svc.aliases)
                for svc in engine.services.values()
            ) if _ql else False
            if _has_bm25 or _has_substr:
                # query 已是标准术语，不需要 augment
                effective_query = query
                augmented_query = None
            else:
                effective_query = classification.augmented_query
                augmented_query = classification.augmented_query
        else:
            effective_query = query
            augmented_query = None
        # M5：规则意图识别（navigational 直达 / multi_condition 转 M6 / conversational 转 M7）
        intent_result = engine.classify_intent(
            effective_query, user_id=user_id, session_id=session_id
        )
        match_mode = "default"
        session_turn_idx: int | None = None
        answer_guide: AnswerGuide | None = None
        try:
            if (
                mode == "auto"
                and intent_result.intent == CONVERSATIONAL
                and session_id
            ):
                # M7：会话上下文存在 → 长程对话搜索（首轮宽召回 / 后续轮精化）
                session_result = await engine.search_session_async(
                    session_id=session_id,
                    user_id=user_id,
                    query=effective_query,
                    action="search",
                )
                results = session_result["results"]
                match_mode = "session"
                session_turn_idx = session_result["turn_idx"]
            elif (
                mode == "auto"
                and intent_result.intent == GUIDE
            ):
                # M16：指引型意图 → LLM 步骤化答案（失败降级 list 模式）
                guide_result = await engine.search_guide_async(
                    user_id=user_id, query=effective_query
                )
                results = guide_result["results"]
                if guide_result["answer_guide"]:
                    answer_guide = AnswerGuide(
                        **guide_result["answer_guide"]
                    )
                    match_mode = "guide"
                else:
                    match_mode = "default"
            elif (
                mode == "auto"
                and intent_result.intent == MULTI_CONDITION
                and intent_result.sub_queries
            ):
                # M6：多条件交集检索（交集为空降级 union）
                results, match_mode = await engine.search_intersection_async(
                    user_id=user_id,
                    queries=intent_result.sub_queries,
                    original_query=effective_query,
                )
            else:
                # M2：默认异步路径，rerank 与 reason 并发 gather
                # retrieval_mode 仅在此默认分支透传；session/guide/multi_condition 分支
                # 是意图路由与 retrieval_mode 正交，本期默认 hybrid 保现有行为
                results = await engine.search_async(
                    user_id=user_id,
                    query=effective_query,
                    retrieval_mode=retrieval_mode,
                )
        except PromptInjectionError as exc:
            # M1：提示词注入命中 → 400，不穿透 500、不泄露后端细节
            raise HTTPException(status_code=400, detail=str(exc))
        # M15：二次深度检索标签从首条结果读取（engine 触发时已打标）
        deep_searched = bool(results and results[0].get("deep_searched"))
        deep_reason = (results[0].get("deep_reason") if results else "") or ""
        # M13：拼写建议（仅 list 模式给；guide 答案模式不打扰）
        spell_suggestion = (
            None if answer_guide is not None else engine.spell_suggest(effective_query)
        )
        # M14：单请求 timing——从最近一次 metrics 事件提取（单 worker 下即本请求）。
        # engine.search 返回 list[dict] 不变，timing 经 metrics 旁路提取，不破坏兼容。
        last_event = engine.metrics.events()[-1] if engine.metrics.events() else None
        timing = last_event["stages"] if last_event else None
        return SearchResponse(
            user_id=user_id,
            query=query,
            intent=intent_result.intent,
            sub_queries=intent_result.sub_queries,
            match_mode=match_mode,
            retrieval_mode=retrieval_mode,
            intent_category=intent_category,
            augmented_query=augmented_query,
            deep_searched=deep_searched,
            deep_reason=deep_reason,
            session_id=session_id,
            turn_idx=session_turn_idx,
            answer_guide=answer_guide,
            spell_suggestion=spell_suggestion,
            results=[SearchResultItem(**item) for item in results],
            timing=timing,
        )

    @app.post("/api/click", tags=["search"])
    def click(payload: ClickRequest) -> dict[str, str]:
        engine = get_engine()
        try:
            engine.record_click(user_id=payload.user_id, service_id=payload.service_id)
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"status": "ok"}

    @app.post("/api/feedback", tags=["search"])
    def feedback(payload: FeedbackRequest) -> dict[str, str]:
        """M13：上报结果停留时长（dwell_ms）。

        dwell < QUICK_BOUNCE_MS 记为负样本，对「点后快速跳出」服务在 popularity 中降权。
        """
        engine = get_engine()
        try:
            engine.record_feedback(
                user_id=payload.user_id,
                service_id=payload.service_id,
                dwell_ms=payload.dwell_ms,
            )
        except ValueError as exc:
            raise HTTPException(status_code=404, detail=str(exc))
        return {"status": "ok"}

    @app.post(
        "/api/search/session",
        response_model=SessionSearchResponse,
        tags=["search"],
    )
    async def search_session(
        payload: SessionSearchRequest,
    ) -> SessionSearchResponse:
        """M7 长程对话搜索端点。

        action=search：首轮宽召回 Top-40，后续轮基于会话上下文精化；
        action=rollback：弹出末轮，返回上一轮 Top-N 与上下文。
        """
        engine = get_engine()
        if not engine.services:
            raise HTTPException(status_code=409, detail="知识库为空，请先上传")
        if payload.action not in ("search", "rollback"):
            raise HTTPException(
                status_code=400, detail="action 必须为 search 或 rollback"
            )
        if payload.action == "search" and not payload.query.strip():
            raise HTTPException(status_code=400, detail="query 不能为空")
        try:
            result = await engine.search_session_async(
                session_id=payload.session_id,
                user_id=payload.user_id,
                query=payload.query,
                action=payload.action,
            )
        except PromptInjectionError as exc:
            # M1：提示词注入命中 → 400
            raise HTTPException(status_code=400, detail=str(exc))
        return SessionSearchResponse(
            session_id=result["session_id"],
            action=result["action"],
            turn_idx=result["turn_idx"],
            query=result["query"],
            match_mode=result["match_mode"],
            results=[SearchResultItem(**item) for item in result["results"]],
            history=[
                {"turn_idx": t["turn_idx"], "query": t["query"]}
                for t in result.get("history", [])
            ],
        )

    @app.post(
        "/api/search/intersection",
        response_model=SearchResponse,
        tags=["search"],
    )
    async def search_intersection(
        payload: IntersectionSearchRequest,
    ) -> SearchResponse:
        """M6 高级多条件交集搜索端点。

        前端 +/- 行输入多个子查询后直接调用本端点（不经意图分词自动路由）。
        复用 ``engine.search_intersection_async``：每子查询独立 Top-30 召回 →
        求交集（空降级 RRF union）→ qwen3-vl-rerank 重排 + 理由生成并发 →
        MMR Top-10。空交集返回空结果 + ``match_mode=union``。
        """
        engine = get_engine()
        if not engine.services:
            raise HTTPException(status_code=409, detail="知识库为空，请先上传")
        # 去空去重（保序），至少 2 条非空子查询才进入交集逻辑
        seen: set[str] = set()
        clean: list[str] = []
        for q in payload.queries:
            s = (q or "").strip()
            if s and s not in seen:
                seen.add(s)
                clean.append(s)
        if len(clean) < 2:
            raise HTTPException(
                status_code=400, detail="至少需要 2 个非空子查询"
            )
        original = payload.original_query or " ".join(clean)
        try:
            results, match_mode = await engine.search_intersection_async(
                user_id=payload.user_id,
                queries=clean,
                original_query=original,
            )
        except PromptInjectionError as exc:
            # M1：提示词注入命中 → 400，不穿透 500、不泄露后端细节
            raise HTTPException(status_code=400, detail=str(exc))
        # M14：单请求 timing——从最近一次 metrics 事件旁路提取（不破坏 list[dict] 返回契约）
        last_event = (
            engine.metrics.events()[-1] if engine.metrics.events() else None
        )
        timing = last_event["stages"] if last_event else None
        # M13：拼写建议（多条件场景同样适用）
        spell_suggestion = engine.spell_suggest(original)
        return SearchResponse(
            user_id=payload.user_id,
            query=original,
            intent="multi_condition",
            sub_queries=clean,
            match_mode=match_mode,
            retrieval_mode="hybrid",
            results=[SearchResultItem(**item) for item in results],
            spell_suggestion=spell_suggestion,
            timing=timing,
        )

    @app.post(
        "/api/search/deep-components",
        response_model=DeepComponentResponse,
        tags=["search"],
    )
    async def deep_components(
        payload: DeepComponentRequest,
    ) -> DeepComponentResponse:
        """深度组件检索端点。

        前端勾选「深度检索」后，搜索完成自动调用：对 top-10 结果分别抓取
        服务 route 页面（仅 http(s)，SSRF 防御在 ``page_fetcher`` 内），
        调 LLM 分析「最契合 query 的组件」，渲染到每条结果右侧的可点击 chip。
        无 Key / LLM 失败 / 抓取失败 → 启发式降级（``pick_heuristic``）。
        """
        engine = get_engine()
        if not engine.services:
            raise HTTPException(status_code=409, detail="知识库为空，请先上传")
        if not payload.service_ids:
            raise HTTPException(status_code=400, detail="service_ids 不能为空")
        try:
            items = await engine.analyze_deep_components_async(
                user_id=payload.user_id,
                query=payload.query,
                service_ids=payload.service_ids,
            )
        except PromptInjectionError as exc:
            # M1：提示词注入命中 → 400，不穿透 500、不泄露后端细节
            raise HTTPException(status_code=400, detail=str(exc))
        return DeepComponentResponse(
            items=[DeepComponentItem(**item) for item in items]
        )

    @app.get("/api/dropdown", response_model=DropdownResponse, tags=["search"])
    def dropdown(user_id: str = Query(..., description="用户ID")) -> DropdownResponse:
        engine = get_engine()
        data = engine.homepage_dropdown(user_id)
        return DropdownResponse(
            recent_queries=data["recent_queries"],
            recent_clicked_services=[
                DropdownItem(**item) for item in data["recent_clicked_services"]
            ],
            global_hot_services=[
                DropdownItem(**item) for item in data["global_hot_services"]
            ],
            recommended_services=[
                DropdownItem(**item) for item in data["recommended_services"]
            ],
        )

    @app.get("/api/search/suggest", response_model=SuggestResponse, tags=["search"])
    async def search_suggest(
        user_id: str = Query(..., description="用户ID"),
        partial: str = Query(
            ...,
            min_length=1,
            max_length=100,
            description="用户已输入的查询前缀",
        ),
    ) -> SuggestResponse:
        """搜索框灰色补全建议（Chrome omnibox 风格）。

        调用 DeepSeek 基于用户历史 + partial 生成补全；LLM 不可用/失败/前缀不匹配 →
        返回空 completion + source=none（前端隐藏灰色建议）。
        M12 重试/超时由 DeepSeekClient 处理；本端点仅做参数校验 + 降级包装。
        """
        engine = get_engine()
        partial = partial.strip()
        if not partial:
            return SuggestResponse(completion="", source="none")
        try:
            completion = await engine.suggest_query_async(user_id, partial)
        except Exception:  # noqa: BLE001 - 任何异常都降级，不打扰输入体验
            completion = None
        # 端点二次前缀校验（防御性：suggester 实现被替换后契约仍成立）
        if (
            completion
            and completion.startswith(partial)
            and len(completion) > len(partial)
        ):
            return SuggestResponse(completion=completion, source="llm")
        return SuggestResponse(completion="", source="none")

    @app.get(
        "/api/search/autocomplete",
        response_model=AutocompleteResponse,
        tags=["search"],
    )
    async def search_autocomplete(
        user_id: str = Query(..., description="用户ID"),
        query: str = Query(
            ...,
            min_length=1,
            max_length=100,
            description="已输入的查询（每次修改 query 触发，尚未点击搜索）",
        ),
    ) -> AutocompleteResponse:
        """搜索框自动补全：边输入边返回 top-10 推荐服务。

        每行只展示匹配到、标蓝的 service_name 或 alias（点击进入路由占位视图），
        不生成排序理由，改为右侧 4 种红色标签：关键词完全匹配 / 语义相似（>0.5）/
        过去常点 / 意图匹配。autocomplete ≠ 真实搜索，不记查询历史、不触结果缓存。
        engine.autocomplete 为同步实现（embed/bm25/local_rerank 全同步），
        经 asyncio.to_thread 包装避免阻塞事件循环（单 worker 下不卡其他请求）。
        """
        engine = get_engine()
        if not engine.services:
            raise HTTPException(status_code=409, detail="知识库为空，请先上传")
        q = query.strip()
        if not q:
            return AutocompleteResponse(query=query, items=[])
        try:
            items = await asyncio.to_thread(engine.autocomplete, user_id, q, 10)
        except PromptInjectionError as exc:
            # M1：提示词注入命中 → 400，不穿透 500、不泄露后端细节
            raise HTTPException(status_code=400, detail=str(exc))
        return AutocompleteResponse(
            query=query, items=[AutocompleteItem(**it) for it in items]
        )

    @app.get("/api/service", response_model=ServiceDetail, tags=["search"])
    def get_service(service_id: str = Query(..., description="服务ID")) -> ServiceDetail:
        """按 service_id 直接返回服务详情（不经检索/rerank），供下拉点击直接进入。"""
        engine = get_engine()
        item = engine.get_service(service_id)
        if item is None:
            raise HTTPException(status_code=404, detail=f"service not found: {service_id}")
        return ServiceDetail(**item)

    @app.get("/api/service/related", response_model=list[ServiceDetail], tags=["search"])
    def service_related(
        service_id: str = Query(..., description="服务ID"),
        k: int = Query(3, ge=1, le=10, description="返回条数（默认 3）"),
    ) -> list[ServiceDetail]:
        """路由占位视图：返回与该服务最相关的 top-k 服务（离线预计算 cosine top-3）。

        测试环境下所有 route 界面不可访问，进入路由界面时用搜索结果卡片临时代替，
        下方展示该服务相关性最高的 3 个服务卡片。预计算在 KB 加载时完成并落盘，
        进入 route 界面直接复用（O(1) 查表），提速。
        """
        engine = get_engine()
        if not engine.services:
            raise HTTPException(status_code=409, detail="知识库为空，请先上传")
        items = engine.get_related_services(service_id, k=k)
        if not items:
            # service 不存在 或 无相关服务（KB 仅 1 个服务时）
            if engine.services.get(service_id) is None:
                raise HTTPException(
                    status_code=404, detail=f"service not found: {service_id}"
                )
            return []
        return [ServiceDetail(**it) for it in items]

    @app.get("/api/reason", tags=["search"])
    async def reason_stream(
        service_id: str = Query(..., description="目标服务ID"),
        query: str = Query(..., description="原始搜索词"),
        user_id: str = Query("anon", description="用户ID（仅记日志，不影响结果）"),
    ) -> StreamingResponse:
        """M10-5：reason 流式端点（M2 懒加载用）。

        前端卡片展开时调用，SSE 流式返回排序理由 + 阶段计时。
        - REASON_ENABLED 开启（默认）且有 API Key → LLM 生成理由，按字符块增量推送
        - 无 API Key / LLM 失败 → 立即返回模板理由
        - 服务不在 KB → error 事件（前端降级展示模板理由）
        每次调用经 engine.metrics 记录 reason 阶段耗时（不记为 search 事件）。
        """
        async def event_gen():
            import time as _time

            t0 = _time.time()
            stages: dict[str, float] = {}
            engine = get_engine()
            svc = engine.services.get(service_id)
            if svc is None:
                yield _sse("error", {"message": f"service not found: {service_id}"})
                return
            info = route_info(svc.route)
            candidate = {
                "service_id": svc.service_id,
                "service_name": svc.service_name,
                "service_intro": svc.service_intro,
                "route": info["route"],
                "score": 1.0,
                "rerank_score": 1.0,
            }
            t_reason = _time.time()
            try:
                reasons = await engine.reasoner.generate_reasons_async(query, [candidate])
            except Exception:  # noqa: BLE001 - LLM 失败降级模板
                reasons = {}
            stages["reason"] = (_time.time() - t_reason) * 1000.0
            reason = reasons.get(service_id) or (
                f"综合相关性与关键词覆盖，综合分1.0000。"
            )
            source = "llm" if reasons.get(service_id) else "template"
            # SSE 增量推送：按 20 字符切块，前端可逐块渲染
            yield _sse("start", {"service_id": service_id, "source": source})
            for i in range(0, len(reason), 20):
                yield _sse("delta", {"text": reason[i : i + 20]})
            stages["total"] = (_time.time() - t0) * 1000.0
            yield _sse("done", {"timing": stages, "source": source})
            # M10：LLM 调用耗时经 client.metrics_callback 自动记入 external 指标；
            # reason 阶段总耗时随 done 事件返回前端，便于单请求诊断。

        return StreamingResponse(
            event_gen(), media_type="text/event-stream; charset=utf-8"
        )

    @app.post(
        "/api/action/execute",
        response_model=ActionExecuteResponse,
        tags=["search"],
    )
    def execute_action(payload: ActionExecuteRequest) -> ActionExecuteResponse:
        """M8：发起页面内组件动作执行（本期打桩）。

        校验 service_id + component + action 命中 KB components 白名单后回显。
        后续可在此转发实际服务总线；M11 接入后将落 search_logs / action_logs。
        """
        engine = get_engine()
        if not engine.services:
            raise HTTPException(status_code=409, detail="知识库为空，请先上传")
        result = engine.execute_component_action(
            service_id=payload.service_id,
            component=payload.component,
            action=payload.action,
            params=payload.params,
        )
        if result is None:
            # service 不存在 或 component/action 不在白名单
            if engine.services.get(payload.service_id) is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"service not found: {payload.service_id}",
                )
            raise HTTPException(
                status_code=400,
                detail=f"component/action 不在服务白名单: {payload.component}/{payload.action}",
            )
        # M11 接入前用标准日志打点（用户级行为日志后续落库）
        logger.info(
            "action_execute user=%s service=%s component=%s action=%s",
            payload.user_id,
            payload.service_id,
            payload.component,
            payload.action,
        )
        return ActionExecuteResponse(**result)

    @app.post("/api/knowledge-base/upload", response_model=UploadResponse, tags=["kb"])
    def upload_knowledge_base(items: list[KnowledgeBaseItem]) -> UploadResponse:
        if not items:
            raise HTTPException(status_code=400, detail="知识库不能为空")
        engine = get_engine()
        payload: list[dict[str, Any]] = [item.model_dump() for item in items]
        engine.load_knowledge_base(payload)
        _persist_kb(payload)
        return UploadResponse(status="ok", services_count=len(engine.services))

    # ---------- M9 知识库管理：导入导出 / 版本 / embedding 状态 ----------
    @app.post("/api/kb/import", response_model=KBImportResponse, tags=["kb"])
    def kb_import(items: list[KnowledgeBaseItem]) -> KBImportResponse:
        """M9：导入 KB → 重建索引 → 落快照 → 置为 active 版本。

        入参为 KnowledgeBaseItem 列表（与 /api/knowledge-base/upload 同构，
        但额外建立版本快照并支持回滚）。
        """
        if not items:
            raise HTTPException(status_code=400, detail="知识库不能为空")
        engine = get_engine()
        payload: list[dict[str, Any]] = [item.model_dump() for item in items]
        try:
            result = engine.import_kb_version(payload)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        _persist_kb(payload)
        logger.info(
            "kb_import version=%s hash=%s services=%s",
            result["version_id"],
            result["kb_hash"][:12],
            result["services_count"],
        )
        return KBImportResponse(
            version_id=result["version_id"],
            kb_hash=result["kb_hash"],
            created_at=result["created_at"],
            active=result["active"],
            services_count=result["services_count"],
        )

    @app.get("/api/kb/export", tags=["kb"])
    def kb_export() -> Response:
        """M9：导出当前 KB 为 JSON 文件（Content-Disposition 触发浏览器下载）。

        M11：落 kb_op_logs 操作日志（export 成功记录服务数 + kb_hash）。
        """
        engine = get_engine()
        if not engine.services:
            raise HTTPException(status_code=409, detail="知识库为空，请先上传")
        payload = engine.export_kb()
        body = json.dumps(payload, ensure_ascii=False, indent=2)
        # M11：export 落操作日志（成功）
        engine._append_kb_op_log(
            op="export", kb_hash=engine.kb_hash or None, ok=True,
            detail={"services_count": len(engine.services), "bytes": len(body)},
        )
        return Response(
            content=body,
            media_type="application/json",
            headers={
                "Content-Disposition": 'attachment; filename="knowledge_base.json"'
            },
        )

    @app.get("/api/kb/versions", response_model=list[KBVersionInfo], tags=["kb"])
    def kb_versions() -> list[KBVersionInfo]:
        """M9：列出全部 KB 版本快照（新→旧），active 标记当前生效版本。"""
        engine = get_engine()
        versions = engine.list_kb_versions()
        return [
            KBVersionInfo(
                version_id=ver["version_id"],
                kb_hash=ver["kb_hash"],
                created_at=ver["created_at"],
                active=ver["active"],
            )
            for ver in versions
        ]

    @app.post("/api/kb/rollback", response_model=KBRollbackResponse, tags=["kb"])
    def kb_rollback(version_id: str = Query(..., description="目标版本ID")) -> KBRollbackResponse:
        """M9：回滚到指定版本——读快照 → 重建索引 → 置为 active。"""
        engine = get_engine()
        result = engine.rollback_kb(version_id)
        if result is None:
            raise HTTPException(
                status_code=404, detail=f"version not found: {version_id}"
            )
        # 同步持久化回滚后的 KB
        _persist_kb([s.to_dict() for s in engine.services.values()])
        logger.info(
            "kb_rollback version=%s hash=%s services=%s",
            result["version_id"],
            result["kb_hash"][:12],
            result["services_count"],
        )
        return KBRollbackResponse(
            version_id=result["version_id"],
            kb_hash=result["kb_hash"],
            created_at=result["created_at"],
            active=result["active"],
            services_count=result["services_count"],
        )

    @app.get(
        "/api/kb/embedding-status",
        response_model=EmbeddingStatusResponse,
        tags=["kb"],
    )
    def kb_embedding_status() -> EmbeddingStatusResponse:
        """M9：当前 KB 的 embedding 状态（total/embedded/in_progress/kb_hash/last_error）。"""
        engine = get_engine()
        return EmbeddingStatusResponse(**engine.embedding_status())

    # ---------- M11 数据日志分析 ----------
    @app.get(
        "/api/logs/search/no-click",
        response_model=list[NoClickQueryStat],
        tags=["logs"],
    )
    def search_logs_no_click(
        window: int = Query(
            86400, ge=60, le=2592000,
            description="聚合窗口秒数（60s-30d），默认 24h",
        ),
        limit: int = Query(50, ge=1, le=500, description="返回条数上限"),
    ) -> list[NoClickQueryStat]:
        """M11：按 query 聚合「无点击率」——召回优化的核心信号。

        无点击 = 用户搜索后 clicked_sid 仍为 NULL（没点任何结果）。
        高频无点击 query → 召回/相关性问题，用于 M13 同义词/负反馈挖掘。
        """
        engine = get_engine()
        rows = engine.aggregate_no_click_queries(
            window_seconds=window, limit=limit
        )
        return [NoClickQueryStat(**r) for r in rows]

    @app.get(
        "/api/logs/search/slow",
        response_model=list[HighLatencyQueryStat],
        tags=["logs"],
    )
    def search_logs_slow(
        window: int = Query(
            86400, ge=60, le=2592000,
            description="聚合窗口秒数（60s-30d），默认 24h",
        ),
        threshold_ms: float = Query(
            1000.0, ge=0.0, le=60000.0,
            description="高延迟阈值（毫秒），默认 1000ms；0=所有搜索均算慢",
        ),
        limit: int = Query(50, ge=1, le=500, description="返回条数上限"),
    ) -> list[HighLatencyQueryStat]:
        """M11：按 query 聚合「高延迟搜索」——性能优化信号。

        从 search_logs.latencies_json.total 提取每次搜索总耗时，
        统计每个 query 的平均/最大/慢搜索次数（超过 threshold_ms）。
        慢 query → 检索/外部调用性能优化。
        """
        engine = get_engine()
        rows = engine.aggregate_high_latency_queries(
            window_seconds=window,
            latency_threshold_ms=threshold_ms,
            limit=limit,
        )
        return [HighLatencyQueryStat(**r) for r in rows]

    @app.get(
        "/api/logs/degradation",
        response_model=DegradationStats,
        tags=["logs"],
    )
    def search_logs_degradation(
        window: int = Query(
            3600, ge=60, le=86400,
            description="统计窗口秒数（60s-24h），默认 1h",
        ),
    ) -> DegradationStats:
        """M11：窗口内降级/缓存命中频次——外部服务健康信号。

        降级频次高 → 外部服务（embed/rerank/reason）健康度下降；
        缓存命中率低 → 重复 query 占比低或缓存失效过快。
        """
        engine = get_engine()
        stats = engine.search_log_degradation_stats(window_seconds=window)
        return DegradationStats(**stats)

    @app.get(
        "/api/logs/search/recent",
        response_model=list[SearchLogItem],
        tags=["logs"],
    )
    def search_logs_recent(
        limit: int = Query(50, ge=1, le=500, description="返回条数上限"),
    ) -> list[SearchLogItem]:
        """M11：最近 search_logs 记录（新→旧），用于调试/巡检。

        user_id 已哈希化（sha256+盐），不暴露原始用户标识。
        """
        engine = get_engine()
        rows = engine.recent_search_logs(limit=limit)
        return [SearchLogItem(**r) for r in rows]

    @app.get(
        "/api/logs/kb-ops",
        response_model=list[KBOpLogItem],
        tags=["logs"],
    )
    def kb_op_logs(
        limit: int = Query(50, ge=1, le=500, description="返回条数上限"),
        op: str | None = Query(
            None, description="按操作类型过滤（import/export/rollback）"
        ),
    ) -> list[KBOpLogItem]:
        """M11：KB 操作日志（import/export/rollback），运维审计。

        新→旧排序，可按 op 过滤。记录成功/失败及错误详情。
        """
        engine = get_engine()
        rows = engine.list_kb_op_logs(limit=limit, op=op)
        return [KBOpLogItem(**r) for r in rows]

    # M14：实时大盘路由（/api/metrics/realtime + /api/metrics/stream SSE）
    register_metrics_routes(app)

    # 前端静态资源：/static/app.js, /static/styles.css
    if os.path.isdir(_FRONTEND_DIR):
        app.mount("/static", StaticFiles(directory=_FRONTEND_DIR), name="static")

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(os.path.join(_FRONTEND_DIR, "index.html"))

        @app.get("/kb", include_in_schema=False)
        def kb_page() -> FileResponse:
            """M9：知识库管理页（导入/进度/版本列表/回滚/导出）。"""
            return FileResponse(os.path.join(_FRONTEND_DIR, "kb.html"))

        @app.get("/dashboard", include_in_schema=False)
        def dashboard_page() -> FileResponse:
            """M14：实时性能大盘页（SSE 1s 刷新各阶段延迟/QPS/降级高亮）。"""
            return FileResponse(os.path.join(_FRONTEND_DIR, "dashboard.html"))

    return app


app = create_app()
