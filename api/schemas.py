from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class KnowledgeBaseItem(BaseModel):
    """知识库单条记录（5 字段 + M8 components）。route 可为字符串或 dict。"""

    service_id: str
    service_name: str
    aliases: list[str] = Field(default_factory=list)
    service_intro: str
    route: Any
    # M8/M9：页面内组件动作列表（缺省空，旧 KB / 旧上传请求向后兼容）
    components: list[dict[str, Any]] = Field(default_factory=list)


class ClickRequest(BaseModel):
    user_id: str
    service_id: str


class ComponentAction(BaseModel):
    """M8：页面内组件动作定义（与 KB components 项同构）。"""

    name: str
    action: str
    params: dict[str, Any] | None = None


class SearchResultItem(BaseModel):
    service_id: str
    service_name: str
    aliases: list[str] = Field(default_factory=list)
    service_intro: str
    route: str
    component: str
    decision_button: str
    derived: bool = False
    # M8：页面内组件动作列表（缺省空，旧 KB 向后兼容）
    components: list[ComponentAction] = Field(default_factory=list)
    score: float
    rerank_score: float | None = None
    rerank_reason: str = ""


class ActionExecuteRequest(BaseModel):
    """M8：发起页面内组件动作执行。

    service_id 定位服务；component 为组件名（与 KB components[].name 对齐）；
    action 为动作标识；params 可选。本期打桩：后端校验 + echo 返回。
    """

    user_id: str
    service_id: str
    component: str
    action: str
    params: dict[str, Any] | None = None


class ActionExecuteResponse(BaseModel):
    """M8：组件执行结果（本期打桩：ok + echo）。"""

    ok: bool
    service_id: str
    component: str
    action: str
    echo: dict[str, Any]


class FeedbackRequest(BaseModel):
    """M13：上报结果停留时长（dwell_ms）以驱动负反馈降权。"""

    user_id: str
    service_id: str
    dwell_ms: int


class AnswerServiceRef(BaseModel):
    """M16 步骤中内嵌的服务引用（点击可跳转 route）。"""

    service_id: str
    service_name: str
    route: str
    component: str
    decision_button: str


class AnswerStep(BaseModel):
    """M16 答案单步：步骤文本 + 该步引用的服务列表。"""

    step_text: str
    services: list[AnswerServiceRef] = Field(default_factory=list)


class AnswerGuide(BaseModel):
    """M16 步骤化指引答案（与 results 互斥）。"""

    query: str
    steps: list[AnswerStep]


class SearchResponse(BaseModel):
    user_id: str
    query: str
    intent: str = "default"
    sub_queries: list[str] = Field(default_factory=list)
    match_mode: str = "default"
    # 检索模式：keyword=仅关键词 / semantic=仅语义 / hybrid=混合（默认）
    retrieval_mode: str = "hybrid"
    deep_searched: bool = False
    deep_reason: str = ""
    session_id: str | None = None
    turn_idx: int | None = None
    answer_guide: AnswerGuide | None = None
    # M13：拼写建议（query 含 OOV/同音错字时给出「您是不是要找」候选；无可纠错为 None）
    spell_suggestion: str | None = None
    results: list[SearchResultItem] = Field(default_factory=list)
    # M14：单请求各阶段耗时（毫秒），便于单请求诊断；None 表示未采集
    # （engine.search 仍返回 list[dict] 兼容 verify.py，timing 由 API 层从最近一次
    # metrics 事件提取，单 worker 下即本请求的埋点）。
    timing: dict[str, float] | None = None


class SessionTurnSummary(BaseModel):
    """M7 会话轮次摘要（前端轮次列表展示）。"""

    turn_idx: int
    query: str


class SessionSearchRequest(BaseModel):
    """M7 长程对话请求。"""

    session_id: str
    user_id: str
    query: str = ""
    action: str = "search"  # "search" | "rollback"


class SessionSearchResponse(BaseModel):
    """M7 长程对话响应：含当前轮结果 + 全部轮次摘要。"""

    session_id: str
    action: str
    turn_idx: int
    query: str
    match_mode: str = "session"  # session | rollback | empty
    results: list[SearchResultItem]
    history: list[SessionTurnSummary] = Field(default_factory=list)


class DropdownItem(BaseModel):
    service_id: str
    service_name: str


class DropdownResponse(BaseModel):
    recent_queries: list[str] = Field(default_factory=list)
    recent_clicked_services: list[DropdownItem] = Field(default_factory=list)
    global_hot_services: list[DropdownItem] = Field(default_factory=list)


class SuggestResponse(BaseModel):
    """搜索框补全建议响应（Chrome omnibox 风格）。

    completion 为完整补全串（partial + 后缀）；空串表示无建议
    （LLM 不可用/降级/前缀不匹配）。source 标识来源便于调试。
    """
    completion: str = ""
    source: str = "none"  # "llm" | "none"


class ServiceDetail(BaseModel):
    service_id: str
    service_name: str
    aliases: list[str] = Field(default_factory=list)
    service_intro: str
    route: str
    component: str
    decision_button: str
    derived: bool = False
    # M8：页面内组件动作列表
    components: list[ComponentAction] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """健康检查响应。

    M10 扩展：在原 status/services_count/dashscope_enabled 基础上，
    叠加最近 100 次搜索的 P95/错误率/缓存命中率/外部调用健康度/embedding 状态。
    所有 M10 字段均为 Optional（默认 None），旧消费者只读前三字段不受影响。
    """

    status: str
    services_count: int
    dashscope_enabled: bool
    # M10 监控扩展（默认 None，未采集时省略）
    search_total: int | None = None
    error_rate: float | None = None
    recent_total: int | None = None
    recent_error_rate: float | None = None
    p95_ms: float | None = None
    cache_hit_rate: float | None = None
    external: dict[str, Any] | None = None
    kb_embedding_in_progress: bool | None = None
    kb_hash: str | None = None
    last_error: str | None = None


class UploadResponse(BaseModel):
    status: str
    services_count: int


# ---------- M9 知识库管理 ----------
class KBVersionInfo(BaseModel):
    """M9：KB 版本快照元数据。"""

    version_id: str
    kb_hash: str
    created_at: float
    active: bool = False
    services_count: int | None = None


class KBImportResponse(BaseModel):
    """M9：导入 KB 的响应（含新版本元数据 + 服务数）。"""

    version_id: str
    kb_hash: str
    created_at: float
    active: bool
    services_count: int


class KBRollbackResponse(BaseModel):
    """M9：回滚 KB 版本的响应。"""

    version_id: str
    kb_hash: str
    created_at: float
    active: bool
    services_count: int


class EmbeddingStatusResponse(BaseModel):
    """M9：当前 KB 的 embedding 状态。"""

    total: int
    embedded: int
    in_progress: bool
    kb_hash: str
    last_error: str = ""


# ---------- M11 数据日志 ----------
class NoClickQueryStat(BaseModel):
    """M11：无点击 query 聚合项（召回优化信号）。"""

    query: str
    total: int
    no_click: int
    no_click_rate: float


class HighLatencyQueryStat(BaseModel):
    """M11：高延迟 query 聚合项（性能优化信号）。"""

    query: str
    total: int
    avg_total_ms: float
    max_total_ms: float
    slow_count: int


class SearchLogItem(BaseModel):
    """M11：单条 search_logs 记录（调试/巡检）。"""

    id: int
    user_hash: str
    query: str
    intent: str
    sub_queries: list[str] = Field(default_factory=list)
    top_ids: list[str] = Field(default_factory=list)
    latencies: dict[str, float] = Field(default_factory=dict)
    cache_hit: bool = False
    degraded: bool = False
    session_id: str | None = None
    clicked_sid: str | None = None
    ts: float


class DegradationStats(BaseModel):
    """M11：窗口内降级/缓存命中统计（外部服务健康信号）。"""

    window_seconds: float
    total: int
    cache_hit: int
    cache_hit_rate: float
    degraded: int
    degraded_rate: float


class KBOpLogItem(BaseModel):
    """M11：KB 操作日志项（import/export/rollback 等运维审计）。"""

    id: int
    op: str
    version_id: str | None = None
    kb_hash: str | None = None
    ok: bool
    detail: dict[str, Any] = Field(default_factory=dict)
    ts: float
