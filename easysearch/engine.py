from __future__ import annotations

import asyncio
import hashlib
import heapq
import json
import logging
import os
import time
from collections import OrderedDict, defaultdict
from typing import Any, Iterable

from .alerts import AlertChecker
from .bm25 import BM25Index, MultiFieldBM25Index
from .config import (
    BM25_FIELD_WEIGHTS,
    BM25_WEIGHT,
    DIN_HISTORY_THRESHOLD,
    DIN_HISTORY_WINDOW,
    DIN_MIX_RATIO,
    DIN_RECENCY_WEIGHT,
    DIN_RELEVANCE_WEIGHT,
    MMR_LAMBDA,
    NEGATIVE_FEEDBACK_ENABLED,
    NEGATIVE_PENALTY,
    NORMALIZE_MODE,
    NORMALIZE_VECTOR,
    POPULARITY_TAU,
    POPULARITY_WEIGHT,
    POPULARITY_WINDOW_DAYS,
    QUICK_BOUNCE_MS,
    REASON_ENABLED,
    SPELL_ENABLED,
    SPELL_MAX_DISTANCE,
    SYNONYM_ENABLED,
    VECTOR_WEIGHT,
)
from .dashscope import DashScopeClient
from .deepseek import DeepSeekClient
from .din import DINHistoryOptimizer
from .embedding import Qwen37TextEmbedding
from .guide import GuideGenerator
from .intent import (
    CONVERSATIONAL,
    GUIDE,
    MULTI_CONDITION,
    NAVIGATIONAL,
    IntentResult,
    IntentRouter,
)
from .metrics import get_metrics
from .mmr import MMRReranker
from .models import ServiceRecord, route_info
from .reranker import Qwen3VLReranker, DeepSeekReasoner
from .safety import PromptInjectionError, sanitize_query, sanitize_text, safe_route
from .spell import LevenshteinCorrector
from .store import SQLiteStore
from .suggest import QuerySuggester
from .cache import get_cache, reset_cache, ResultCache, MemoryResultCache
from .synonyms import SynonymExpander
from .utils import normalize_scores, tokenize
from .vector_index import VectorIndex

logger = logging.getLogger(__name__)

# 模块级再导出（保 verify.py:65 / test_search_engine.py:75 通过）
__all__ = [
    "ServiceSearchEngine",
    "VECTOR_WEIGHT",
    "BM25_WEIGHT",
    "POPULARITY_WEIGHT",
]


class ServiceSearchEngine:
    """服务搜索引擎编排器：知识库 -> 向量化 -> 混合检索 Top-20 -> rerank -> MMR -> Top-10。

    行为对外保持兼容：search / record_click / homepage_dropdown 三个公共方法签名不变。

    本轮改进（A 组 + B 组 A1-A7）：
      - B2 VectorIndex (FAISS IndexFlatIP) 替代 dict + Python cosine
      - C3 tokenize(query) 只切一次，bm25.batch_score_tokens 一次拿全表
      - A1 SynonymExpander（领域词典 + KB 动态抽取 alias↔name）
      - A2 多字段 BM25（name/aliases/intro/route 加权）
      - A3 配置项全部从 config 读（环境变量可覆盖）
      - A4 popularity_decayed 时间衰减替代 raw count
      - A5 MMRReranker（lambda=0.85）在 rerank 后做多样性选择
      - A6 LevenshteinCorrector OOV 拼写纠错
      - A7 reranker 一致性 prompt + 校验（在 reranker.py 内）
    """

    def __init__(
        self,
        dashscope_client: DashScopeClient | None = None,
        deepseek_client: DeepSeekClient | None = None,
        store: SQLiteStore | None = None,
        db_path: str = "data/easysearch.db",
    ) -> None:
        self.dashscope_client = dashscope_client or DashScopeClient()
        self.deepseek_client = deepseek_client or DeepSeekClient()
        self.embedding_model = Qwen37TextEmbedding(self.dashscope_client)
        self.history_optimizer = DINHistoryOptimizer()
        self.reasoner = DeepSeekReasoner(self.deepseek_client)
        self.reranker = Qwen3VLReranker(self.dashscope_client, self.reasoner)
        self.bm25 = BM25Index()  # 兼容旧访问（单字段）
        self._mf_bm25 = MultiFieldBM25Index(field_weights=BM25_FIELD_WEIGHTS)  # A2 多字段
        self.store = store or SQLiteStore(db_path)
        # M16：答案模式 generator（复用 DeepSeek 客户端；无 key 时 generate_guide 返回 None → 降级列表）
        self.guide_generator = GuideGenerator(self.deepseek_client)
        # 搜索框灰色补全建议器（复用 DeepSeek 客户端；无 key/前缀不匹配 → 返 None 隐藏灰色）
        self.query_suggester = QuerySuggester(self.deepseek_client)

        # A 组 / B 组新组件
        self.vector_index = VectorIndex()
        self.synonym_expander = SynonymExpander()
        self.spell_corrector: LevenshteinCorrector | None = None
        self.mmr = MMRReranker(lambda_=MMR_LAMBDA)
        # M5：意图识别（规则路由 navigational/multi_condition/informational/conversational/default）
        self.intent_router = IntentRouter()

        self.services: dict[str, ServiceRecord] = {}
        # service_embeddings 保留为属性（兼容外部访问），从 VectorIndex 反查
        self._service_embeddings_cache: dict[str, list[float]] = {}
        # M4：向量持久化目录（:memory: 测试库不持久化，避免污染临时文件）
        if db_path == ":memory:":
            self._embeddings_dir: str | None = None
        else:
            base = os.path.dirname(os.path.abspath(db_path)) or os.getcwd()
            self._embeddings_dir = os.path.join(base, "embeddings")
        # M9：KB 版本快照目录（与 embeddings 同级；:memory: 测试库不持久化快照）
        self._kb_versions_dir: str | None = (
            None if self._embeddings_dir is None
            else os.path.join(os.path.dirname(self._embeddings_dir), "kb_versions")
        )
        # M9：当前 KB 内容 hash + 最近一次 embedding 错误（embedding-status 端点用）
        self.kb_hash: str = ""
        self._kb_last_error: str = ""
        # M4：结果 LRU 缓存（size=512, TTL=60s），key=(user_id, query)；点击后失效
        self._result_cache: "OrderedDict[tuple[str, str], tuple[list[dict[str, Any]], float]]" = OrderedDict()
        self._result_cache_size = 512
        self._result_cache_ttl = 60.0
        # 检索模式缓存（Redis 可选降级）：key 含 retrieval_mode 防串结果，value 含 reason 完整 results
        # 无 Redis 时 per-instance Memory（保旧 per-engine 隔离，防跨测试/跨引擎串结果）；
        # 配 REDIS_URL 时用共享 Redis 单例（跨 worker 共享，TTL 5min）
        redis_url = os.getenv("REDIS_URL") or os.getenv("EASYSEARCH_REDIS_URL") or ""
        if redis_url:
            self.result_cache: ResultCache = get_cache()
        else:
            self.result_cache = MemoryResultCache()
        # M7：会话历史 LRU(1000)，key=session_id；append/rollback 后失效
        self._session_cache: "OrderedDict[str, list[dict[str, Any]]]" = OrderedDict()
        self._session_cache_size = 1000
        # M11：最近一次 search_logs 自增 id（per user_id），供 record_click 回填 clicked_sid。
        # 单 worker 1-20 并发下 per-user 最近搜索即该用户上次展示的结果，回填 clicked_sid 准确。
        self._last_search_log: "OrderedDict[str, tuple[int, list[str]]]" = OrderedDict()
        self._last_search_log_size = 1000
        # M10：监控告警——metrics 单例（事件缓冲 + Prometheus 镜像）+ AlertChecker
        # 外部调用（dashscope/deepseek）经 metrics_callback 回调埋点，主链路 search/search_async
        # 内部对各阶段计时并 record_search，告警在每次搜索后评估。
        self.metrics = get_metrics()
        self.alert_checker = AlertChecker()
        self.dashscope_client.metrics_callback = self.metrics.record_external
        self.deepseek_client.metrics_callback = self.metrics.record_external

    # ---------- M10 监控告警 ----------
    def _evaluate_and_fire_alerts(self) -> None:
        """每次搜索后评估告警规则并触发（ERROR 日志 + 可选 webhook）。

        - 基于进程内滚动窗口（最近 100 次）评估错误率/P95/缓存命中率
        - 叠加外部服务连续失败计数（dashscope/deepseek）
        - 无告警则静默；触发也不抛异常，确保不影响主链路
        """
        try:
            health = self.metrics.health_summary()
            external_fails = {
                svc: self.metrics.consecutive_failures(svc)
                for svc in ("dashscope", "deepseek")
            }
            alerts = self.alert_checker.evaluate(health, external_fails)
            self.alert_checker.fire(alerts)
        except Exception:  # noqa: BLE001 - 告警链路异常不影响搜索
            logger.warning("alert evaluation failed", exc_info=True)

    # ---------- M11 数据日志 ----------
    def _append_search_log(
        self,
        user_id: str,
        query: str,
        intent: str,
        results: list[dict[str, Any]],
        stages: dict[str, float],
        cache_hit: bool,
        degraded: bool,
        sub_queries: list[str] | None = None,
        session_id: str | None = None,
    ) -> int | None:
        """M11：落一条 search_logs 记录并缓存 log_id 供 record_click 回填。

        - 从 results 提取 Top-10 service_id 作为 top_ids
        - latencies 直接用 stages dict（含 total/retrieval/rerank/mmr/intent）
        - 失败静默（日志落库异常不影响搜索主链路）
        - 返回 log_id（失败返回 None）
        """
        try:
            top_ids = [
                r.get("service_id", "")
                for r in results[:10]
                if r.get("service_id")
            ]
            log_id = self.store.append_search_log(
                user_id=user_id,
                query=query,
                intent=intent,
                top_ids=top_ids,
                latencies=stages,
                cache_hit=cache_hit,
                degraded=degraded,
                ts=time.time(),
                sub_queries=sub_queries,
                session_id=session_id,
            )
            # 缓存 (log_id, top_ids) per user_id，供 record_click 回填 clicked_sid
            self._last_search_log[user_id] = (log_id, top_ids)
            self._last_search_log.move_to_end(user_id)
            while len(self._last_search_log) > self._last_search_log_size:
                self._last_search_log.popitem(last=False)
            return log_id
        except Exception:  # noqa: BLE001 - 日志落库异常不影响搜索
            logger.warning("search_log append failed", exc_info=True)
            return None

    def _mark_search_log_click(self, user_id: str, service_id: str) -> None:
        """M11：回填 clicked_sid 到最近一次该用户的 search_logs。

        - 命中条件：缓存中有该 user_id 的最近搜索，且 service_id 在该次 top_ids 中
        - mark_search_log_click 带 NULL 守卫，重复回填幂等
        - 失败静默（回填异常不影响点击主链路）
        """
        try:
            entry = self._last_search_log.get(user_id)
            if entry is None:
                return
            log_id, top_ids = entry
            if service_id in top_ids:
                self.store.mark_search_log_click(log_id, service_id)
        except Exception:  # noqa: BLE001 - 回填异常不影响点击
            logger.warning("search_log click backfill failed", exc_info=True)

    def _append_kb_op_log(
        self,
        op: str,
        version_id: str | None = None,
        kb_hash: str | None = None,
        ok: bool = True,
        detail: dict[str, Any] | None = None,
    ) -> None:
        """M11：落 KB 操作日志（import/export/rollback）。

        失败静默（日志异常不影响 KB 管理主链路）。
        """
        try:
            self.store.append_kb_op_log(
                op=op, version_id=version_id, kb_hash=kb_hash,
                ok=ok, detail=detail, ts=time.time(),
            )
        except Exception:  # noqa: BLE001
            logger.warning("kb_op_log append failed", exc_info=True)

    def aggregate_no_click_queries(
        self, window_seconds: float = 86400.0, limit: int = 50
    ) -> list[dict[str, Any]]:
        """M11：按 query 聚合无点击率（召回优化信号）。委托 store。"""
        return self.store.aggregate_no_click_queries(
            window_seconds=window_seconds, limit=limit
        )

    def aggregate_high_latency_queries(
        self,
        window_seconds: float = 86400.0,
        latency_threshold_ms: float = 1000.0,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """M11：按 query 聚合高延迟搜索（性能优化信号）。委托 store。"""
        return self.store.aggregate_high_latency_queries(
            window_seconds=window_seconds,
            latency_threshold_ms=latency_threshold_ms,
            limit=limit,
        )

    def search_log_degradation_stats(
        self, window_seconds: float = 3600.0
    ) -> dict[str, Any]:
        """M11：窗口内降级/缓存命中频次（外部服务健康信号）。委托 store。"""
        return self.store.search_log_degradation_stats(
            window_seconds=window_seconds
        )

    def recent_search_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        """M11：最近 search_logs 记录（调试/巡检）。委托 store。"""
        return self.store.recent_search_logs(limit=limit)

    def list_kb_op_logs(
        self, limit: int = 50, op: str | None = None
    ) -> list[dict[str, Any]]:
        """M11：列出 KB 操作日志（新→旧）。委托 store。"""
        return self.store.list_kb_op_logs(limit=limit, op=op)

    # ---------- 向量访问兼容 ----------
    @property
    def service_embeddings(self) -> dict[str, list[float]]:
        """兼容旧属性：从 VectorIndex 反查返回 dict[str, list[float]]。"""
        return self._service_embeddings_cache

    def _embeddings_npz_path(self, kb_hash: str) -> str | None:
        """M4：返回向量持久化 npz 路径；未启用持久化（:memory: 测试库）返回 None。"""
        if not self._embeddings_dir:
            return None
        return os.path.join(self._embeddings_dir, f"emb_{kb_hash}.npz")

    # ---------- 知识库 ----------
    def upload_knowledge_base_from_json(self, json_path: str) -> None:
        with open(json_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if not isinstance(payload, list):
            raise ValueError("Knowledge base JSON must be a list")
        self.load_knowledge_base(payload)

    def load_knowledge_base(self, payload: list[dict[str, Any]]) -> None:
        self._load_services(payload)

    def _load_services(self, payload: list[dict[str, Any]]) -> None:
        self.services.clear()
        self._service_embeddings_cache.clear()
        self.vector_index.build({})  # 清空

        if not payload:
            self.bm25.build({})
            self._mf_bm25.build({})
            self.synonym_expander = SynonymExpander()  # 重置为基础词典
            self.spell_corrector = None
            return

        records: list[ServiceRecord] = []
        # M1：清洗 KB 字段（剥控制/零宽字符、限长）+ 校验路由（不安全 path 置空）
        # M8：components 透传原始值，由 ServiceRecord.from_dict → _sanitize_components 统一清洗
        for item in payload:
            safe_item = {
                "service_id": sanitize_text(item.get("service_id", ""), 100),
                "service_name": sanitize_text(item.get("service_name", ""), 200),
                "aliases": [
                    sanitize_text(a, 100)
                    for a in (item.get("aliases") or [])
                    if a is not None
                ],
                "service_intro": sanitize_text(item.get("service_intro", "")),
                "route": safe_route(item.get("route")),
                "components": item.get("components"),
            }
            records.append(ServiceRecord.from_dict(safe_item))
        # 去重：同一 service_id 后者覆盖
        for record in records:
            self.services[record.service_id] = record

        # A1：从 KB 抽取 alias ↔ service_name 同义词
        self.synonym_expander = SynonymExpander()
        if SYNONYM_ENABLED:
            self.synonym_expander.update_from_kb(self.services)

        # A2/C3：多字段 BM25 构建（name/aliases/intro/route 加权）
        multi_docs: dict[str, dict[str, str]] = {}
        single_docs: dict[str, str] = {}  # 兼容旧 BM25Index.build
        for record in records:
            info = route_info(record.route)
            route_text = " ".join(
                part for part in [info["route"], info["component"], info["decision_button"]] if part
            )
            multi_docs[record.service_id] = {
                "name": record.service_name,
                "aliases": " ".join(record.aliases),
                "intro": record.service_intro,
                "route": route_text,
            }
            single_docs[record.service_id] = record.searchable_text
        self.bm25.build(single_docs)  # 兼容旧访问
        self._mf_bm25.build(multi_docs)  # A2 多字段加权

        # A6：拼写纠错词表（来自多字段 BM25 vocabulary）
        if SPELL_ENABLED:
            self.spell_corrector = LevenshteinCorrector(
                self._mf_bm25.vocabulary(),
                max_distance=SPELL_MAX_DISTANCE,
            )
        else:
            self.spell_corrector = None

        # B2 + M4：批量向量化 + VectorIndex 构建（命中持久化 npz 则跳过 embedding）
        texts = [record.searchable_text for record in records]
        # KB 内容 hash：按 (service_id, searchable_text) 序列化，内容寻址持久化
        kb_hash = hashlib.sha256(
            "\n".join(f"{sid}\t{t}" for sid, t in zip(
                [r.service_id for r in records], texts
            )).encode("utf-8")
        ).hexdigest()
        # M9：暴露当前 KB 内容 hash（embedding-status / 版本管理用）
        self.kb_hash = kb_hash
        npz_path = self._embeddings_npz_path(kb_hash)
        if npz_path is not None and self.vector_index.load_npz(npz_path):
            # M4：命中持久化缓存，跳过 embedding 调用（重启不重 embed）
            pass
        else:
            vectors = self.embedding_model.embed_batch(texts)
            items: dict[str, list[float]] = {}
            for record, vector in zip(records, vectors):
                items[record.service_id] = vector
            self.vector_index.build(items)
            if npz_path is not None:
                self.vector_index.save_npz(npz_path)
        # 兼容缓存：从 VectorIndex 反查
        self._service_embeddings_cache = {sid: self.vector_index.get(sid) for sid in self.services}

    @staticmethod
    def _project_to_single_text(
        docs: dict[str, dict[str, str]], records: list[ServiceRecord]
    ) -> dict[str, str]:
        """把多字段 docs 投影成单字段 searchable_text（兼容旧 BM25Index.build 接口）。

        旧 BM25Index 内部委托给 MultiFieldBM25Index 单字段实例（field="text"），
        故仍需传入 {"text": searchable_text} 形态——这里直接构造单字段 text 即可。
        """
        result: dict[str, str] = {}
        for record in records:
            result[record.service_id] = record.searchable_text
        return result

    # ---------- M9 知识库管理：导入导出 / 版本 / embedding 状态 ----------
    def export_kb(self) -> list[dict[str, Any]]:
        """导出当前 KB 为 JSON 条目列表（与 from_dict 输入同构）。

        每项 {service_id, service_name, aliases, service_intro, route, components}；
        route 保留原始形态（dict/string），re-import 时 safe_route 幂等。
        """
        return [svc.to_dict() for svc in self.services.values()]

    def import_kb_version(
        self,
        payload: list[dict[str, Any]],
        version_id: str | None = None,
    ) -> dict[str, Any]:
        """M9：导入 KB → 重建索引 → 落快照 → 置为 active 版本。

        - 先 load_knowledge_base 重建索引（内部计算 self.kb_hash）
        - 用导出形态（已 sanitize）写快照文件，保证 re-import hash 一致
        - 落 kb_versions 元数据并置 active
        - 失效结果缓存（KB 已变更，旧缓存不再有效）
        - M11：落 kb_op_logs 操作日志（成功/失败均记）
        返回 {version_id, kb_hash, path, created_at, active, services_count}。
        """
        if self._kb_versions_dir is None:
            raise ValueError("KB 版本持久化未启用（:memory: 测试库不支持版本管理）")
        if not isinstance(payload, list) or not payload:
            raise ValueError("知识库不能为空")
        self._kb_last_error = ""
        try:
            self.load_knowledge_base(payload)
        except Exception as exc:  # noqa: BLE001 - 上层 API 决定如何映射
            self._kb_last_error = str(exc)
            # M11：失败也落操作日志（故障诊断）
            self._append_kb_op_log(
                op="import", ok=False,
                detail={"error": str(exc), "items": len(payload) if isinstance(payload, list) else 0},
            )
            raise
        created_at = time.time()
        if version_id is None:
            version_id = f"v-{self.kb_hash[:12]}-{int(created_at)}"
        os.makedirs(self._kb_versions_dir, exist_ok=True)
        path = os.path.join(self._kb_versions_dir, f"{version_id}.json")
        # 快照用导出形态（已 sanitize），保证 re-import 后 searchable_text 不变 → hash 一致
        snapshot_payload = self.export_kb()
        with open(path, "w", encoding="utf-8") as fp:
            json.dump(snapshot_payload, fp, ensure_ascii=False, indent=2)
        self.store.kb_version_add(
            version_id, self.kb_hash, path, created_at, active=True
        )
        self._result_cache_invalidate()
        # M11：成功落操作日志（version_id/kb_hash/services_count）
        self._append_kb_op_log(
            op="import", version_id=version_id, kb_hash=self.kb_hash, ok=True,
            detail={"services_count": len(self.services), "path": path},
        )
        return {
            "version_id": version_id,
            "kb_hash": self.kb_hash,
            "path": path,
            "created_at": created_at,
            "active": True,
            "services_count": len(self.services),
        }

    def list_kb_versions(self) -> list[dict[str, Any]]:
        """M9：列出全部 KB 版本快照（新→旧）。"""
        return self.store.kb_version_list()

    def rollback_kb(self, version_id: str) -> dict[str, Any] | None:
        """M9：回滚到指定版本——读快照文件 → 重建索引 → 置为 active。

        版本不存在或快照文件缺失返回 None；重建成功返回版本元数据（含 services_count）。
        M11：落 kb_op_logs 操作日志（成功/失败/版本缺失均记）。
        """
        meta = self.store.kb_version_get(version_id)
        if meta is None:
            # M11：版本不存在也落日志（运维审计）
            self._append_kb_op_log(
                op="rollback", version_id=version_id, ok=False,
                detail={"error": "version not found"},
            )
            return None
        path = meta.get("path") or ""
        if not path or not os.path.exists(path):
            # M11：快照文件缺失也落日志
            self._append_kb_op_log(
                op="rollback", version_id=version_id, ok=False,
                detail={"error": "snapshot file missing", "path": path},
            )
            return None
        with open(path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        self._kb_last_error = ""
        try:
            self.load_knowledge_base(payload)
        except Exception as exc:  # noqa: BLE001
            self._kb_last_error = str(exc)
            # M11：重建失败落日志
            self._append_kb_op_log(
                op="rollback", version_id=version_id, kb_hash=meta.get("kb_hash"),
                ok=False, detail={"error": str(exc)},
            )
            raise
        self.store.kb_version_set_active(version_id)
        self._result_cache_invalidate()
        # 重新读取 meta 以反映 active 状态切换（set_active 前旧 meta 的 active=False）
        meta = self.store.kb_version_get(version_id)
        meta["services_count"] = len(self.services)
        # M11：成功回滚落日志
        self._append_kb_op_log(
            op="rollback", version_id=version_id, kb_hash=meta.get("kb_hash"),
            ok=True, detail={"services_count": len(self.services)},
        )
        return meta

    def embedding_status(self) -> dict[str, Any]:
        """M9：当前 KB 的 embedding 状态。

        - total：KB 服务总数
        - embedded：VectorIndex 已索引向量数（命中持久化 npz 也会重建到 VectorIndex）
        - in_progress：是否正在后台重建（本期同步导入，恒 False）
        - kb_hash：当前 KB 内容 hash
        - last_error：最近一次导入/回滚的错误信息（无错误为空串）
        """
        return {
            "total": len(self.services),
            "embedded": len(self.vector_index),
            "in_progress": False,
            "kb_hash": self.kb_hash,
            "last_error": self._kb_last_error,
        }

    # ---------- 搜索 ----------
    def _build_top_candidates(
        self,
        user_id: str,
        query: str,
        query_tokens: list[str] | None = None,
        query_embedding: list[float] | None = None,
        top_n: int = 20,
        retrieval_mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """共享召回：embed→DIN→BM25/向量/popularity→混合打分→Top-20。

        query 由调用方（search/search_async）已清洗并记录到 store，此处不再重复。
        返回 top20 候选列表（空列表表示无候选）。

        query_tokens / query_embedding 可由调用方注入：
          - M15 深度检索注入扩展后的 query_tokens
          - M7 会话后续轮注入 session-level DIN 融合后的 query_embedding（跳过 embed + 用户级 DIN）
        """
        if not query or not self.services:
            return []

        timestamp = time.time()

        if query_embedding is None:
            # A1 向量路径：归一到规范词再 embed，避免同义词稀释向量语义
            embed_query = (
                self.synonym_expander.normalize(query)
                if SYNONYM_ENABLED
                else query
            )
            query_embedding = self.embedding_model.embed(embed_query)

            # DIN：用户历史查询数 > 阈值时，用历史序列优化 query 向量
            if self.store.query_count(user_id) > DIN_HISTORY_THRESHOLD:
                history_queries = self.store.all_queries(user_id, DIN_HISTORY_WINDOW)
                # 去掉当前刚追加的 query（它是最后一条）
                if history_queries and history_queries[-1] == query:
                    history_queries = history_queries[:-1]
                if history_queries:
                    history_embeddings = self.embedding_model.embed_batch(history_queries)
                    query_embedding = self.history_optimizer.optimize(
                        query_embedding,
                        history_embeddings,
                        relevance_weight=DIN_RELEVANCE_WEIGHT,
                        recency_weight=DIN_RECENCY_WEIGHT,
                        mix_ratio=DIN_MIX_RATIO,
                    )

        # C3：tokenize 一次，下游复用；M15 深度检索可传入预扩展 tokens 跳过本步
        if query_tokens is None:
            query_tokens = tokenize(query)

            # A1 BM25 路径：expand 追加同义词 token
            if SYNONYM_ENABLED:
                query_tokens = self.synonym_expander.expand(query_tokens)

            # A6：OOV 拼写纠错（追加候选 token，不替换原 token）
            if SPELL_ENABLED and self.spell_corrector is not None:
                query_tokens = self.spell_corrector.correct_tokens(query_tokens)

        # C3 + A2：一次 batch_score_tokens 拿全表多字段加权 BM25 分数
        bm25_raw: dict[str, float] = self._mf_bm25.batch_score_tokens(query_tokens)

        # B2：VectorIndex 一次算全部 cosine
        vector_raw: dict[str, float] = self.vector_index.score_all(query_embedding)

        # A4：时间衰减 popularity（替代 raw global_click_counter）
        popularity_raw: dict[str, float] = self.store.popularity_decayed(
            tau=POPULARITY_TAU,
            now=timestamp,
            window_days=POPULARITY_WINDOW_DAYS,
        )
        # M13：点后快速跳出服务降权（无负样本时原样返回，零影响）
        popularity_raw = self._apply_negative_penalty(popularity_raw, timestamp)

        # M13：归一化模式 + 向量分可选归一（默认 minmax + 不归一向量，保旧行为）
        bm25_norm = normalize_scores(bm25_raw, mode=NORMALIZE_MODE)
        popularity_norm = normalize_scores(popularity_raw, mode=NORMALIZE_MODE)
        if NORMALIZE_VECTOR:
            vector_raw = normalize_scores(vector_raw, mode=NORMALIZE_MODE)

        # M3：大库（>200）候选集 = BM25 Top-100 ∪ 向量 Top-100，避免全表 N 次构造候选；
        # hybrid 分对各 doc 不变（bm25_norm/vector_raw/popularity_norm 仍按全表归一），
        # 被裁掉的 doc 必为 bm25 与向量双低，无法进 Top-20，故结果与全表一致。
        # 小库走全量，与旧行为逐字节一致（保 verify.py / 旧测试通过）。
        if len(self.services) > 200:
            bm25_top = {
                sid
                for sid, _ in heapq.nlargest(
                    100, bm25_raw.items(), key=lambda kv: kv[1]
                )
            }
            vec_top = {
                sid
                for sid, _ in heapq.nlargest(
                    100, vector_raw.items(), key=lambda kv: kv[1]
                )
            }
            candidate_ids = bm25_top | vec_top
        else:
            candidate_ids = self.services.keys()

        candidates: list[dict[str, Any]] = []
        for service_id in candidate_ids:
            service = self.services.get(service_id)
            if service is None:
                continue
            info = route_info(service.route)
            score = self._hybrid_score(
                vector_similarity=vector_raw.get(service_id, 0.0),
                bm25_score=bm25_norm.get(service_id, 0.0),
                popularity_score=popularity_norm.get(service_id, 0.0),
                retrieval_mode=retrieval_mode,
            )
            candidates.append(
                {
                    "service_id": service.service_id,
                    "service_name": service.service_name,
                    "aliases": list(service.aliases),
                    "service_intro": service.service_intro,
                    "route": info["route"],
                    "component": info["component"],
                    "decision_button": info["decision_button"],
                    "derived": info["derived"],
                    "components": list(service.components),
                    "score": score,
                }
            )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_n]

    # ---------- M5 意图识别 ----------
    def classify_intent(
        self,
        query: str,
        user_id: str | None = None,
        session_id: str | None = None,
    ) -> IntentResult:
        """对 query 做规则意图分类（M5）。

        has_session：M7 长程对话接入后由会话存在性决定；session_id 传入且会话存在
        → has_session=True（conversational 意图据此触发）；否则 False。
        """
        has_session = (
            session_id is not None and self._session_exists(session_id)
        )
        return self.intent_router.classify(
            query, services=self.services, has_session=has_session
        )

    def _pin_navigational_to_top(
        self, results: list[dict[str, Any]], service_id: str | None
    ) -> list[dict[str, Any]]:
        """navigational 意图：跳过 MMR，把精确命中服务置顶。

        命中服务在结果中 → 移到 index 0；不在结果中 → 用 KB 记录构造一项前置
        （保证直达唯一服务，即使打分未进 Top-10）。
        """
        if not service_id:
            return results
        match_idx = next(
            (i for i, x in enumerate(results) if x.get("service_id") == service_id),
            -1,
        )
        if match_idx == 0:
            return results
        if match_idx > 0:
            item = results.pop(match_idx)
            results.insert(0, item)
            return results
        # 不在 Top-10：从 KB 构造直达项
        svc = self.services.get(service_id)
        if svc is None:
            return results
        info = route_info(svc.route)
        direct = {
            "service_id": svc.service_id,
            "service_name": svc.service_name,
            "aliases": list(svc.aliases),
            "service_intro": svc.service_intro,
            "route": info["route"],
            "component": info["component"],
            "decision_button": info["decision_button"],
            "derived": info["derived"],
            "components": list(svc.components),
            "score": 1.0,
            "rerank_reason": "精确命中服务名称，直达该服务。",
        }
        results.insert(0, direct)
        return results

    # ---------- M15 二次深度检索 ----------
    def _deep_expand_query(self, query: str, max_tokens: int = 40) -> list[str]:
        """M15 深度检索的查询扩展：base + 同义词 + KB 共现词。

        KB 共现扩展：命中任一 base token 的服务，追加其 name/alias token
        （作为 embedding 近邻词的替代；M13 接入后可换 embedding top-5）。
        """
        base = tokenize(query)
        expanded = list(base)
        if SYNONYM_ENABLED:
            expanded = self.synonym_expander.expand(expanded)
        extra: set[str] = set(expanded)
        base_set = set(base)
        for svc in self.services.values():
            svc_tokens = set(tokenize(svc.service_name))
            for alias in svc.aliases:
                svc_tokens.update(tokenize(alias))
            if svc_tokens & base_set:
                extra.update(svc_tokens)
                if len(extra) >= max_tokens:
                    break
        return list(extra)[:max_tokens]

    @staticmethod
    def _rrf_fuse(
        first_results: list[dict[str, Any]],
        deep_candidates: list[dict[str, Any]],
        k: int = 60,
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """RRF 融合：score(d) = Σ 1/(k + rank) over 两个排序列表。

        保留首次项的完整字段（含 rerank_reason 等）；深度独有项补模板 reason。
        """
        by_id: dict[str, dict[str, Any]] = {}
        rrf: dict[str, float] = defaultdict(float)

        for rank, item in enumerate(first_results):
            sid = item.get("service_id")
            if not sid:
                continue
            by_id[sid] = item
            rrf[sid] += 1.0 / (k + rank + 1)
        for rank, item in enumerate(deep_candidates):
            sid = item.get("service_id")
            if not sid:
                continue
            if sid not in by_id:
                merged = dict(item)
                merged.setdefault("rerank_reason", "深度检索补充结果。")
                by_id[sid] = merged
            rrf[sid] += 1.0 / (k + rank + 1)

        ordered = sorted(by_id, key=lambda sid: rrf[sid], reverse=True)[:top_k]
        return [by_id[sid] for sid in ordered]

    def _maybe_deep_search(
        self,
        user_id: str,
        query: str,
        first_results: list[dict[str, Any]],
        intent: str,
    ) -> list[dict[str, Any]]:
        """M15：置信度不足时触发二次深度检索 + RRF 融合，仅触发一次（不递归）。

        navigational/multi_condition/conversational 不触发（evaluate_confidence 内判定）。
        触发后给结果项打 deep_searched/deep_reason 标签，供响应与前端展示。
        """
        is_cold_user = self.store.query_count(user_id) <= 1
        conf = self.intent_router.evaluate_confidence(
            first_results, intent, is_cold_user=is_cold_user
        )
        if not conf.should_deep_search or not first_results:
            return first_results
        # 二次深度检索：扩展 query → 重检索 Top-30 → RRF 融合（仅一次，不再递归）
        expanded_tokens = self._deep_expand_query(query)
        deep_candidates = self._build_top_candidates(
            user_id, query, query_tokens=expanded_tokens, top_n=30
        )
        fused = self._rrf_fuse(first_results, deep_candidates)
        for item in fused:
            item["deep_searched"] = True
            item["deep_reason"] = conf.reason
        return fused

    # ---------- M6 多条件交集 ----------
    @staticmethod
    def _rrf_fuse_multi(
        lists: list[list[dict[str, Any]]],
        k: int = 60,
        top_k: int = 30,
    ) -> list[dict[str, Any]]:
        """多排序列表 RRF 融合：score(d) = Σ 1/(k + rank) over 各列表。

        用于 M6 空交集降级：合并各子查询 Top-30，不补模板 reason（后续 rerank 统一加）。
        """
        by_id: dict[str, dict[str, Any]] = {}
        rrf: dict[str, float] = defaultdict(float)
        for lst in lists:
            for rank, item in enumerate(lst):
                sid = item.get("service_id")
                if not sid:
                    continue
                if sid not in by_id:
                    by_id[sid] = item
                rrf[sid] += 1.0 / (k + rank + 1)
        ordered = sorted(by_id, key=lambda sid: rrf[sid], reverse=True)[:top_k]
        return [by_id[sid] for sid in ordered]

    def _intersect_candidates(
        self, user_id: str, queries: list[str], top_n: int = 30
    ) -> tuple[list[dict[str, Any]], str]:
        """M6：每个子查询独立召回 Top-N → service_id 集合求交集。

        交集非空 → match_mode="intersection"，按首个子查询原序保序；
        交集为空 → match_mode="union"，RRF 融合各子查询 Top-N 取 Top-30。
        单子查询 → match_mode="default"。
        返回 (候选列表, match_mode)，候选待 rerank + MMR。
        """
        clean = [sanitize_query(q) for q in queries if q and q.strip()]
        clean = [q for q in clean if q]
        if not clean:
            return ([], "default")
        per_q = [self._build_top_candidates(user_id, qi, top_n=top_n) for qi in clean]
        return self._intersect_candidate_lists(per_q)

    @staticmethod
    def _intersect_candidate_lists(
        per_q: list[list[dict[str, Any]]],
    ) -> tuple[list[dict[str, Any]], str]:
        """纯逻辑：对已召回的各子查询候选列表求交集/降级 union（无 IO，可单测）。"""
        if not per_q:
            return ([], "default")
        if len(per_q) == 1:
            return (list(per_q[0]), "default")

        id_sets = [set(c.get("service_id") for c in lst) for lst in per_q]
        intersection = set(id_sets[0])
        for s in id_sets[1:]:
            intersection &= s

        if intersection:
            match_mode = "intersection"
            first_by_id = {c["service_id"]: c for c in per_q[0]}
            # 按首个子查询的候选顺序保序取交集项（稳定排序）
            candidates = [
                first_by_id[sid]
                for sid in (c["service_id"] for c in per_q[0])
                if sid in intersection
            ]
        else:
            match_mode = "union"
            candidates = ServiceSearchEngine._rrf_fuse_multi(per_q, k=60, top_k=30)
        return (candidates, match_mode)

    def search_intersection(
        self,
        user_id: str,
        queries: list[str],
        original_query: str | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        """M6 同步：多条件交集检索（rerank + MMR 后返回 Top-10，match_mode）。"""
        candidates, match_mode = self._intersect_candidates(user_id, queries)
        if not candidates:
            # M11：空交集也落日志（degraded=True，sub_queries=queries）
            q = original_query or " ".join(queries)
            self._append_search_log(
                user_id, q, "multi_condition", [], {},
                cache_hit=False, degraded=True, sub_queries=queries,
            )
            return ([], match_mode)
        q = original_query or " ".join(queries)
        reranked = self.reranker.rerank(q, candidates)
        final = self.mmr.select(
            reranked, embeddings=self._service_embeddings_cache, top_k=10
        )
        # M11：多条件交集成功落日志（sub_queries=queries）
        self._append_search_log(
            user_id, q, "multi_condition", final, {},
            cache_hit=False, degraded=False, sub_queries=queries,
        )
        return (final, match_mode)

    async def search_intersection_async(
        self,
        user_id: str,
        queries: list[str],
        original_query: str | None = None,
    ) -> tuple[list[dict[str, Any]], str]:
        """M6 异步：多条件交集检索，rerank 与 reason 并发 gather。"""
        candidates, match_mode = self._intersect_candidates(user_id, queries)
        if not candidates:
            # M11：空交集也落日志（degraded=True，sub_queries=queries）
            q = original_query or " ".join(queries)
            self._append_search_log(
                user_id, q, "multi_condition", [], {},
                cache_hit=False, degraded=True, sub_queries=queries,
            )
            return ([], match_mode)
        q = original_query or " ".join(queries)
        reranked, reasons = await asyncio.gather(
            self.reranker.rerank_async(q, candidates),
            self.reasoner.generate_reasons_async(q, candidates),
        )
        if reasons:
            for item in reranked:
                sid = item["service_id"]
                if sid in reasons:
                    item["rerank_reason"] = reasons[sid]
        final = self.mmr.select(
            reranked, embeddings=self._service_embeddings_cache, top_k=10
        )
        # M11：多条件交集成功落日志（sub_queries=queries）
        self._append_search_log(
            user_id, q, "multi_condition", final, {},
            cache_hit=False, degraded=False, sub_queries=queries,
        )
        return (final, match_mode)

    # ---------- M7 长程对话 ----------
    def _session_exists(self, session_id: str) -> bool:
        """会话是否存在（至少一轮）。"""
        return self.store.session_exists(session_id)

    def _session_cache_get(self, session_id: str) -> list[dict[str, Any]] | None:
        entry = self._session_cache.get(session_id)
        if entry is None:
            return None
        self._session_cache.move_to_end(session_id)
        return entry

    def _session_cache_set(self, session_id: str, history: list[dict[str, Any]]) -> None:
        self._session_cache[session_id] = history
        self._session_cache.move_to_end(session_id)
        while len(self._session_cache) > self._session_cache_size:
            self._session_cache.popitem(last=False)

    def _session_cache_invalidate(self, session_id: str) -> None:
        self._session_cache.pop(session_id, None)

    def _session_history(self, session_id: str) -> list[dict[str, Any]]:
        """返回会话历史轮次（旧→新），每项 {turn_idx, query, top_ids}。

        进程内 LRU 命中直接返回；未命中查 DB 并写回缓存。
        """
        cached = self._session_cache_get(session_id)
        if cached is not None:
            return cached
        rows = self.store.session_turns(session_id)
        history: list[dict[str, Any]] = [
            {
                "turn_idx": int(row["turn_idx"]),
                "query": row["query"],
                "top_ids": json.loads(row["top_ids_json"]),
            }
            for row in rows
        ]
        self._session_cache_set(session_id, history)
        return history

    @staticmethod
    def _session_summary(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """会话轮次摘要（供前端轮次列表展示）。"""
        return [{"turn_idx": t["turn_idx"], "query": t["query"]} for t in history]

    def _session_refine(
        self,
        user_id: str,
        query: str,
        history: list[dict[str, Any]],
        top_n: int = 40,
    ) -> list[dict[str, Any]]:
        """M7 后续轮：新 query embedding 与历史轮 query embedding 融合（session-level DIN），
        对累积候选 Top-40 重排精化。

        - 累积候选 = 历史各轮 top_ids 并集 ∪ 新 query（fused embedding）召回 Top-40
        - 对累积候选用 fused embedding + 新 query BM25 + popularity 重打分，取 Top-40
        """
        new_emb = self.embedding_model.embed(query)
        history_queries = [t["query"] for t in history if t.get("query")]
        if history_queries:
            history_embs = self.embedding_model.embed_batch(history_queries)
            fused_emb = self.history_optimizer.optimize(
                new_emb,
                history_embs,
                relevance_weight=DIN_RELEVANCE_WEIGHT,
                recency_weight=DIN_RECENCY_WEIGHT,
                mix_ratio=DIN_MIX_RATIO,
            )
        else:
            fused_emb = new_emb

        accumulated_ids: set[str] = set()
        for t in history:
            accumulated_ids.update(t.get("top_ids") or [])
        # 新检索补充候选（注入 fused embedding 跳过 embed + 用户级 DIN）
        new_candidates = self._build_top_candidates(
            user_id, query, query_embedding=fused_emb, top_n=40
        )
        for c in new_candidates:
            accumulated_ids.add(c["service_id"])
        return self._rescore_candidates(user_id, query, fused_emb, accumulated_ids, top_n=top_n)

    def _rescore_candidates(
        self,
        user_id: str,
        query: str,
        query_embedding: list[float],
        candidate_ids: set[str] | Iterable[str],
        top_n: int = 40,
        retrieval_mode: str = "hybrid",
    ) -> list[dict[str, Any]]:
        """对指定候选 id 集合用 (vector_sim with query_embedding, BM25(query), popularity) 重打分。

        用于 M7 会话后续轮对累积候选集的精化重排。
        """
        if not candidate_ids or not self.services:
            return []
        timestamp = time.time()
        query_tokens = tokenize(query)
        if SYNONYM_ENABLED:
            query_tokens = self.synonym_expander.expand(query_tokens)
        if SPELL_ENABLED and self.spell_corrector is not None:
            query_tokens = self.spell_corrector.correct_tokens(query_tokens)
        bm25_raw = self._mf_bm25.batch_score_tokens(query_tokens)
        vector_raw = self.vector_index.score_all(query_embedding)
        popularity_raw = self.store.popularity_decayed(
            tau=POPULARITY_TAU,
            now=timestamp,
            window_days=POPULARITY_WINDOW_DAYS,
        )
        # M13：负反馈降权 + 归一化模式 + 向量分可选归一（与 _build_top_candidates 同口径）
        popularity_raw = self._apply_negative_penalty(popularity_raw, timestamp)
        bm25_norm = normalize_scores(bm25_raw, mode=NORMALIZE_MODE)
        popularity_norm = normalize_scores(popularity_raw, mode=NORMALIZE_MODE)
        if NORMALIZE_VECTOR:
            vector_raw = normalize_scores(vector_raw, mode=NORMALIZE_MODE)

        candidates: list[dict[str, Any]] = []
        for service_id in candidate_ids:
            service = self.services.get(service_id)
            if service is None:
                continue
            info = route_info(service.route)
            score = self._hybrid_score(
                vector_similarity=vector_raw.get(service_id, 0.0),
                bm25_score=bm25_norm.get(service_id, 0.0),
                popularity_score=popularity_norm.get(service_id, 0.0),
                retrieval_mode=retrieval_mode,
            )
            candidates.append(
                {
                    "service_id": service.service_id,
                    "service_name": service.service_name,
                    "aliases": list(service.aliases),
                    "service_intro": service.service_intro,
                    "route": info["route"],
                    "component": info["component"],
                    "decision_button": info["decision_button"],
                    "derived": info["derived"],
                    "components": list(service.components),
                    "score": score,
                }
            )
        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:top_n]

    def _reconstruct_results(self, top_ids: list[str]) -> list[dict[str, Any]]:
        """M7 撤回：从上一轮 top_ids 用 KB 记录重建结果项（无 rerank 分，保序展示）。"""
        results: list[dict[str, Any]] = []
        for idx, sid in enumerate(top_ids[:10]):
            service = self.services.get(sid)
            if service is None:
                continue
            info = route_info(service.route)
            results.append(
                {
                    "service_id": service.service_id,
                    "service_name": service.service_name,
                    "aliases": list(service.aliases),
                    "service_intro": service.service_intro,
                    "route": info["route"],
                    "component": info["component"],
                    "decision_button": info["decision_button"],
                    "derived": info["derived"],
                    "components": list(service.components),
                    "score": max(0.0, 1.0 - idx * 0.01),
                    "rerank_reason": "撤回至上一轮结果。",
                }
            )
        return results

    def _session_rollback(self, session_id: str) -> dict[str, Any]:
        """M7 撤回：弹出末轮，返回上一轮 Top-N 与上下文。"""
        deleted = self.store.session_delete_last_turn(session_id)
        self._session_cache_invalidate(session_id)
        history = self._session_history(session_id)
        if not history:
            return {
                "session_id": session_id,
                "action": "rollback",
                "turn_idx": -1,
                "query": "",
                "match_mode": "empty",
                "results": [],
                "history": [],
            }
        last = history[-1]
        results = self._reconstruct_results(last.get("top_ids") or [])
        return {
            "session_id": session_id,
            "action": "rollback",
            "turn_idx": last["turn_idx"],
            "query": last["query"],
            "match_mode": "rollback",
            "results": results,
            "history": self._session_summary(history),
        }

    def _session_search_core(
        self,
        session_id: str,
        user_id: str,
        query: str,
    ) -> tuple[list[dict[str, Any]], list[str], int, list[dict[str, Any]]]:
        """M7 search 动作核心逻辑（rerank/reason 由 sync/async 包装器分别处理）。

        返回 (top40_candidates, top_ids, turn_idx, history_summary)。
        - 首轮：宽召回 Top-40
        - 后续轮：session-level DIN 融合 + 累积候选重排精化 Top-40
        落库 + 失效缓存。
        """
        history = self._session_history(session_id)
        turn_idx = len(history)  # 下一轮序号（0-based）
        # 记录用户级查询（DIN 历史 / 下拉一致性）
        self.store.append_query(user_id, query, time.time())
        if not history:
            top40 = self._build_top_candidates(user_id, query, top_n=40)
        else:
            top40 = self._session_refine(user_id, query, history, top_n=40)
        top_ids = [c["service_id"] for c in top40[:40]]
        # 落库并失效缓存
        self.store.append_session_turn(
            session_id, user_id, turn_idx, query, top_ids, time.time()
        )
        self._session_cache_invalidate(session_id)
        new_history = history + [
            {"turn_idx": turn_idx, "query": query, "top_ids": top_ids}
        ]
        return (top40, top_ids, turn_idx, self._session_summary(new_history))

    def search_session(
        self,
        session_id: str,
        user_id: str,
        query: str,
        action: str = "search",
    ) -> dict[str, Any]:
        """M7 同步长程对话搜索。

        action: "search"（首轮宽召回/后续轮精化）或 "rollback"（撤回上一轮）。
        返回 {session_id, action, turn_idx, query, match_mode, results, history}。
        """
        if action == "rollback":
            return self._session_rollback(session_id)
        # search
        q = sanitize_query(query)
        if not q or not self.services:
            return {
                "session_id": session_id,
                "action": action,
                "turn_idx": -1,
                "query": q,
                "match_mode": "empty",
                "results": [],
                "history": self._session_summary(self._session_history(session_id)),
            }
        top40, _, turn_idx, summary = self._session_search_core(session_id, user_id, q)
        if not top40:
            return {
                "session_id": session_id,
                "action": "search",
                "turn_idx": turn_idx,
                "query": q,
                "match_mode": "empty",
                "results": [],
                "history": summary,
            }
        reranked = self.reranker.rerank(q, top40)
        final = self.mmr.select(
            reranked, embeddings=self._service_embeddings_cache, top_k=10
        )
        # M11：会话搜索也落 search_logs（intent=conversational + session_id）
        self._append_search_log(
            user_id, q, "conversational", final, {},
            cache_hit=False, degraded=False, session_id=session_id,
        )
        return {
            "session_id": session_id,
            "action": "search",
            "turn_idx": turn_idx,
            "query": q,
            "match_mode": "session",
            "results": final,
            "history": summary,
        }

    async def search_session_async(
        self,
        session_id: str,
        user_id: str,
        query: str,
        action: str = "search",
    ) -> dict[str, Any]:
        """M7 异步长程对话搜索：rerank 与 reason 并发 gather。"""
        if action == "rollback":
            return self._session_rollback(session_id)
        q = sanitize_query(query)
        if not q or not self.services:
            return {
                "session_id": session_id,
                "action": action,
                "turn_idx": -1,
                "query": q,
                "match_mode": "empty",
                "results": [],
                "history": self._session_summary(self._session_history(session_id)),
            }
        top40, _, turn_idx, summary = self._session_search_core(session_id, user_id, q)
        if not top40:
            return {
                "session_id": session_id,
                "action": "search",
                "turn_idx": turn_idx,
                "query": q,
                "match_mode": "empty",
                "results": [],
                "history": summary,
            }
        reranked, reasons = await asyncio.gather(
            self.reranker.rerank_async(q, top40),
            self.reasoner.generate_reasons_async(q, top40),
        )
        if reasons:
            for item in reranked:
                sid = item["service_id"]
                if sid in reasons:
                    item["rerank_reason"] = reasons[sid]
        final = self.mmr.select(
            reranked, embeddings=self._service_embeddings_cache, top_k=10
        )
        # M11：会话搜索也落 search_logs（intent=conversational + session_id）
        self._append_search_log(
            user_id, q, "conversational", final, {},
            cache_hit=False, degraded=False, session_id=session_id,
        )
        return {
            "session_id": session_id,
            "action": "search",
            "turn_idx": turn_idx,
            "query": q,
            "match_mode": "session",
            "results": final,
            "history": summary,
        }

    # ---------- M16 答案模式 ----------
    def search_guide(
        self, user_id: str, query: str
    ) -> dict[str, Any]:
        """M16 同步：guide 意图 → LLM 步骤化答案；失败降级 list 模式 Top-10。

        返回 {"answer_guide": {query, steps} | None, "results": [...]}（互斥）。
        """
        q = sanitize_query(query)
        if not q or not self.services:
            return {"answer_guide": None, "results": []}
        self.store.append_query(user_id, q, time.time())
        top20 = self._build_top_candidates(user_id, q, top_n=20)
        if not top20:
            # M11：guide 无候选也落日志（degraded=True）
            self._append_search_log(
                user_id, q, "guide", [], {}, cache_hit=False, degraded=True,
            )
            return {"answer_guide": None, "results": []}
        # Top-10 候选作为 LLM 生成上下文（白名单源）
        guide = self.guide_generator.generate_guide(q, top20[:10])
        if guide and guide.get("steps"):
            # M11：guide 成功落日志（top_ids = guide 引用的服务）
            guide_sids = [
                ref.get("service_id", "")
                for step in guide["steps"]
                for ref in (step.get("services") or [])
                if ref.get("service_id")
            ][:10]
            self._append_search_log(
                user_id, q, "guide", [{"service_id": sid} for sid in guide_sids], {},
                cache_hit=False, degraded=False,
            )
            return {
                "answer_guide": {"query": q, "steps": guide["steps"]},
                "results": [],
            }
        # 降级：LLM 不可用/解析失败 → list 模式 Top-10（rerank + MMR）
        reranked = self.reranker.rerank(q, top20)
        final = self.mmr.select(
            reranked, embeddings=self._service_embeddings_cache, top_k=10
        )
        # M11：guide 降级为 list 模式也落日志
        self._append_search_log(
            user_id, q, "guide", final, {}, cache_hit=False, degraded=True,
        )
        return {"answer_guide": None, "results": final}

    async def search_guide_async(
        self, user_id: str, query: str
    ) -> dict[str, Any]:
        """M16 异步：guide 意图 → LLM 步骤化答案；失败降级 list 模式 Top-10。"""
        q = sanitize_query(query)
        if not q or not self.services:
            return {"answer_guide": None, "results": []}
        self.store.append_query(user_id, q, time.time())
        top20 = self._build_top_candidates(user_id, q, top_n=20)
        if not top20:
            # M11：guide 无候选也落日志（degraded=True）
            self._append_search_log(
                user_id, q, "guide", [], {}, cache_hit=False, degraded=True,
            )
            return {"answer_guide": None, "results": []}
        guide = await self.guide_generator.generate_guide_async(q, top20[:10])
        if guide and guide.get("steps"):
            # M11：guide 成功落日志（top_ids = guide 引用的服务）
            guide_sids = [
                ref.get("service_id", "")
                for step in guide["steps"]
                for ref in (step.get("services") or [])
                if ref.get("service_id")
            ][:10]
            self._append_search_log(
                user_id, q, "guide", [{"service_id": sid} for sid in guide_sids], {},
                cache_hit=False, degraded=False,
            )
            return {
                "answer_guide": {"query": q, "steps": guide["steps"]},
                "results": [],
            }
        # 降级：list 模式 Top-10（rerank + reason 并发 gather）
        reranked, reasons = await asyncio.gather(
            self.reranker.rerank_async(q, top20),
            self.reasoner.generate_reasons_async(q, top20),
        )
        if reasons:
            for item in reranked:
                sid = item["service_id"]
                if sid in reasons:
                    item["rerank_reason"] = reasons[sid]
        final = self.mmr.select(
            reranked, embeddings=self._service_embeddings_cache, top_k=10
        )
        # M11：guide 降级为 list 模式也落日志
        self._append_search_log(
            user_id, q, "guide", final, {}, cache_hit=False, degraded=True,
        )
        return {"answer_guide": None, "results": final}

    # ---------- M4 结果缓存 ----------
    def _result_cache_get(
        self, user_id: str, query: str, retrieval_mode: str = "hybrid"
    ) -> list[dict[str, Any]] | None:
        # 默认 retrieval_mode="hybrid" 保旧调用方（verify.py / 测试 2 参数调用）兼容。
        # key 含 retrieval_mode 避免不同模式串结果（keyword 缓存不被 hybrid 命中）。
        return self.result_cache.get(user_id, query, retrieval_mode)

    def _result_cache_set(
        self,
        user_id: str,
        query: str,
        results: list[dict[str, Any]],
        retrieval_mode: str = "hybrid",
    ) -> None:
        # TTL 跟随缓存实现配置：Redis 用 EASYSEARCH_CACHE_TTL（默认 300s=5min），
        # Memory 用 60s 保旧行为。getattr 取缓存实例自身 _ttl，避免硬编码 60s 覆盖 Redis 5min。
        ttl = getattr(self.result_cache, "_ttl", self._result_cache_ttl)
        self.result_cache.set(user_id, query, retrieval_mode, results, ttl=ttl)

    def _result_cache_invalidate(self, user_id: str | None = None) -> None:
        """record_click 后失效缓存（默认清全部；传 user_id 只清该用户）。"""
        self.result_cache.invalidate(user_id)

    def search(
        self, user_id: str, query: str, retrieval_mode: str = "hybrid"
    ) -> list[dict[str, Any]]:
        """同步搜索（兼容旧链路 / 测试 / verify.py）。

        内部走同步 rerank + 同步 reason（reason 默认关闭即模板理由）。
        retrieval_mode="hybrid" 默认保 verify.py:88 的 2 位置参数调用兼容；
        keyword/semantic 模式跳过 rerank+reason，仅附差异化模板 reason。
        M4：结果 LRU 缓存命中直接返回，未命中计算后写回；点击后失效。
        M10：各阶段计时埋点 + record_search + 告警评估（失败也记 error=True 后 re-raise）。
        M11：各返回点落 search_logs（timing/intent/cache_hit/degraded/top_ids），
        cache_hit/no_candidates/successful 路径均记录；empty query 路径跳过（噪声）。
        """
        # M10：阶段计时 + 错误/缓存命中埋点
        t0 = time.time()
        stages: dict[str, float] = {}
        intent_str = ""
        try:
            # M1：入口清洗查询词，命中提示词注入抛 PromptInjectionError（API 层捕获→400）
            q = sanitize_query(query)
            if not q or not self.services:
                stages["total"] = (time.time() - t0) * 1000
                self.metrics.record_search(
                    total_ms=stages["total"], stages=stages,
                    degraded=True, intent="empty",
                )
                # M11：空 query / 空 KB 不落日志（噪声，无分析价值）
                return []
            # 记录查询（即使缓存命中也要记录，供下拉/DIN 历史）
            self.store.append_query(user_id, q, time.time())
            cached = self._result_cache_get(user_id, q, retrieval_mode)
            if cached is not None:
                stages["total"] = (time.time() - t0) * 1000
                self.metrics.record_search(
                    total_ms=stages["total"], stages=stages,
                    cache_hit=True, intent="cache_hit",
                )
                self._evaluate_and_fire_alerts()
                # M11：缓存命中也落日志（cache_hit=True，分析缓存命中率）
                self._append_search_log(
                    user_id, q, "cache_hit", cached, stages,
                    cache_hit=True, degraded=False,
                )
                return cached
            t_rec = time.time()
            top20 = self._build_top_candidates(user_id, q, retrieval_mode=retrieval_mode)
            stages["retrieval"] = (time.time() - t_rec) * 1000
            if not top20:
                stages["total"] = (time.time() - t0) * 1000
                self.metrics.record_search(
                    total_ms=stages["total"], stages=stages,
                    degraded=True, intent="no_candidates",
                )
                self._evaluate_and_fire_alerts()
                # M11：无候选也落日志（degraded=True，召回优化的核心信号）
                self._append_search_log(
                    user_id, q, "no_candidates", [], stages,
                    cache_hit=False, degraded=True,
                )
                return []
            t_rerank = time.time()
            if retrieval_mode in ("keyword", "semantic"):
                # keyword/semantic 模式：跳过 rerank + reason，仅附差异化模板 reason（保字段非空）
                # rerank_score 直接用召回 hybrid 分；仍跑 MMR 保多样性
                reranked = [
                    {
                        **item,
                        "rerank_score": item["score"],
                        "rerank_reason": self.reranker._build_template_reason(q, item),
                    }
                    for item in top20
                ]
            else:
                reranked = self.reranker.rerank(q, top20)
            stages["rerank"] = (time.time() - t_rerank) * 1000
            t_mmr = time.time()
            # A5：MMR 多样性重排（在 rerank 之后从 top20 选 top10）
            final = self.mmr.select(
                reranked,
                embeddings=self._service_embeddings_cache,
                top_k=10,
            )
            stages["mmr"] = (time.time() - t_mmr) * 1000
            # M5：navigational 直达——跳过 MMR 把精确命中服务置顶
            t_intent = time.time()
            intent = self.classify_intent(q, user_id=user_id)
            stages["intent"] = (time.time() - t_intent) * 1000
            intent_str = intent.intent
            # M15：置信度不足触发二次深度检索 + RRF 融合（仅一次；navigational/multi/conversational 不触发）
            final = self._maybe_deep_search(user_id, q, final, intent.intent)
            if intent.intent == NAVIGATIONAL:
                final = self._pin_navigational_to_top(final, intent.matched_service_id)
            self._result_cache_set(user_id, q, final, retrieval_mode)
            stages["total"] = (time.time() - t0) * 1000
            self.metrics.record_search(
                total_ms=stages["total"], stages=stages,
                intent=intent_str,
            )
            self._evaluate_and_fire_alerts()
            # M11：成功搜索落日志（top_ids/intent/latencies 供无点击聚合/高延迟分析）
            self._append_search_log(
                user_id, q, intent_str, final, stages,
                cache_hit=False, degraded=False,
                sub_queries=intent.sub_queries,
            )
            return final
        except Exception:
            stages["total"] = (time.time() - t0) * 1000
            self.metrics.record_search(
                total_ms=stages["total"], stages=stages,
                error=True, intent=intent_str,
            )
            # M11：异常搜索也落日志（degraded=True，故障诊断信号）
            self._append_search_log(
                user_id, query, intent_str or "error", [], stages,
                cache_hit=False, degraded=True,
            )
            raise

    async def search_async(
        self, user_id: str, query: str, retrieval_mode: str = "hybrid"
    ) -> list[dict[str, Any]]:
        """M2 异步搜索：rerank 与 reason 并发（asyncio.gather）。

        reason 输入用 rerank 前的混合分 Top-20（不依赖重排顺序），
        最终展示用 rerank 顺序 + 对应 reason（按 service_id 对齐覆盖模板理由）。
        retrieval_mode="hybrid" 默认保旧调用方兼容；keyword/semantic 模式跳过
        rerank+reason，仅附差异化模板 reason（仍跑 MMR 保多样性）。
        M4：结果 LRU 缓存命中直接返回，未命中计算后写回；点击后失效。
        M10：各阶段计时埋点 + record_search + 告警评估（失败也记 error=True 后 re-raise）。
        M11：各返回点落 search_logs（timing/intent/cache_hit/degraded/top_ids），
        cache_hit/no_candidates/successful 路径均记录；empty query 路径跳过（噪声）。
        """
        # M10：阶段计时 + 错误/缓存命中埋点
        t0 = time.time()
        stages: dict[str, float] = {}
        intent_str = ""
        try:
            # M1：入口清洗查询词，命中提示词注入抛 PromptInjectionError（API 层捕获→400）
            q = sanitize_query(query)
            if not q or not self.services:
                stages["total"] = (time.time() - t0) * 1000
                self.metrics.record_search(
                    total_ms=stages["total"], stages=stages,
                    degraded=True, intent="empty",
                )
                # M11：空 query / 空 KB 不落日志（噪声，无分析价值）
                return []
            self.store.append_query(user_id, q, time.time())
            cached = self._result_cache_get(user_id, q, retrieval_mode)
            if cached is not None:
                stages["total"] = (time.time() - t0) * 1000
                self.metrics.record_search(
                    total_ms=stages["total"], stages=stages,
                    cache_hit=True, intent="cache_hit",
                )
                self._evaluate_and_fire_alerts()
                # M11：缓存命中也落日志（cache_hit=True，分析缓存命中率）
                self._append_search_log(
                    user_id, q, "cache_hit", cached, stages,
                    cache_hit=True, degraded=False,
                )
                return cached
            t_rec = time.time()
            top20 = self._build_top_candidates(user_id, q, retrieval_mode=retrieval_mode)
            stages["retrieval"] = (time.time() - t_rec) * 1000
            if not top20:
                stages["total"] = (time.time() - t0) * 1000
                self.metrics.record_search(
                    total_ms=stages["total"], stages=stages,
                    degraded=True, intent="no_candidates",
                )
                self._evaluate_and_fire_alerts()
                # M11：无候选也落日志（degraded=True，召回优化的核心信号）
                self._append_search_log(
                    user_id, q, "no_candidates", [], stages,
                    cache_hit=False, degraded=True,
                )
                return []
            t_rerank = time.time()
            if retrieval_mode in ("keyword", "semantic"):
                # keyword/semantic 模式：跳过 rerank + reason（不调 LLM/不调远程 rerank），
                # 仅附差异化模板 reason 保字段非空；rerank_score 直接用召回 hybrid 分
                reranked = [
                    {
                        **item,
                        "rerank_score": item["score"],
                        "rerank_reason": self.reranker._build_template_reason(q, item),
                    }
                    for item in top20
                ]
                stages["rerank_reason"] = (time.time() - t_rerank) * 1000
            else:
                # hybrid：rerank 与 reason 并发 gather（reason 默认关闭即模板，REASON_ENABLED=True 时 LLM 覆盖）
                reranked, reasons = await asyncio.gather(
                    self.reranker.rerank_async(q, top20),
                    self.reasoner.generate_reasons_async(q, top20),
                )
                stages["rerank_reason"] = (time.time() - t_rerank) * 1000
                # 用 LLM reason 覆盖模板 reason（按 service_id 对齐到 rerank 顺序）
                if reasons:
                    for item in reranked:
                        sid = item["service_id"]
                        if sid in reasons:
                            item["rerank_reason"] = reasons[sid]
            t_mmr = time.time()
            # A5：MMR 多样性重排（在 rerank 之后从 top20 选 top10）
            final = self.mmr.select(
                reranked,
                embeddings=self._service_embeddings_cache,
                top_k=10,
            )
            stages["mmr"] = (time.time() - t_mmr) * 1000
            # M5：navigational 直达——跳过 MMR 把精确命中服务置顶
            t_intent = time.time()
            intent = self.classify_intent(q, user_id=user_id)
            stages["intent"] = (time.time() - t_intent) * 1000
            intent_str = intent.intent
            # M15：置信度不足触发二次深度检索 + RRF 融合（仅一次；navigational/multi/conversational 不触发）
            final = self._maybe_deep_search(user_id, q, final, intent.intent)
            if intent.intent == NAVIGATIONAL:
                final = self._pin_navigational_to_top(final, intent.matched_service_id)
            self._result_cache_set(user_id, q, final, retrieval_mode)
            stages["total"] = (time.time() - t0) * 1000
            self.metrics.record_search(
                total_ms=stages["total"], stages=stages,
                intent=intent_str,
            )
            self._evaluate_and_fire_alerts()
            # M11：成功搜索落日志（top_ids/intent/latencies 供无点击聚合/高延迟分析）
            self._append_search_log(
                user_id, q, intent_str, final, stages,
                cache_hit=False, degraded=False,
                sub_queries=intent.sub_queries,
            )
            return final
        except Exception:
            stages["total"] = (time.time() - t0) * 1000
            self.metrics.record_search(
                total_ms=stages["total"], stages=stages,
                error=True, intent=intent_str,
            )
            # M11：异常搜索也落日志（degraded=True，故障诊断信号）
            self._append_search_log(
                user_id, query, intent_str or "error", [], stages,
                cache_hit=False, degraded=True,
            )
            raise

    # ---------- 行为记录 ----------
    def record_click(self, user_id: str, service_id: str) -> None:
        """记录点击。M12：已下线服务仍记点击（标 deprecated），不硬 404。

        - service_id 在 KB：正常路径，落 user_clicks + global_clicks + 回填 search_logs
        - service_id 不在 KB（已下线/未知）：仍记 user_clicks(deprecated=1) 用于行为分析，
          不污染 global_clicks 热度榜；打 WARN 日志便于运维感知下线服务的残留点击。
          不抛 ValueError，避免 API 透 404 阻塞前端交互（M12「不硬 404」要求）。
        """
        if service_id not in self.services:
            # M12：下线服务仍记点击，标 deprecated；不污染热度榜
            self.store.append_click(
                user_id, service_id, time.time(), deprecated=True
            )
            logger.warning(
                "record_click on deprecated service user=%s sid=%s",
                user_id, service_id,
            )
            return
        self.store.append_click(user_id, service_id, time.time())
        # M11：回填最近一次 search_logs.clicked_sid（无点击聚合分析的信号源）
        self._mark_search_log_click(user_id, service_id)

    def record_feedback(self, user_id: str, service_id: str, dwell_ms: int) -> None:
        """M13：记录结果停留时长（dwell_ms）；dwell < QUICK_BOUNCE_MS 记为负样本。

        负样本经 _apply_negative_penalty 在 popularity 中降权，实现「点后快速跳出降权」。
        M12：对已下线服务仍记录 dwell（行为分析信号），不抛 ValueError。
        """
        # M12：服务下线时仍记录 dwell，便于行为分析；不阻塞前端
        self.store.append_feedback(user_id, service_id, dwell_ms, time.time())
        if service_id not in self.services:
            logger.warning(
                "record_feedback on deprecated service user=%s sid=%s dwell=%dms",
                user_id, service_id, dwell_ms,
            )

    # ---------- 首页下拉 ----------
    def homepage_dropdown(self, user_id: str) -> dict[str, list[Any]]:
        """下拉默认项：最近3未重复搜索词 / 最近3未重复点击服务 / 全局最热3服务。

        recent_queries 为字符串列表；点击/热门返回 [{service_id, service_name}]，
        前端据此可直接调 /api/service 进入该服务详情，无需重新搜索。
        注：hot_services 仍用 raw count（A4 的 popularity_decayed 仅用于搜索混合打分）。
        """
        recent_queries = self.store.recent_queries(user_id, 3)  # 最近->最旧
        recent_click_ids = self.store.recent_clicks(user_id, 3)  # 最近->最旧
        recent_clicked_services = [
            {"service_id": sid, "service_name": self.services[sid].service_name}
            for sid in recent_click_ids
            if sid in self.services
        ]
        hot_ids = self.store.hot_services(3)
        hot_services = [
            {"service_id": sid, "service_name": self.services[sid].service_name}
            for sid in hot_ids
            if sid in self.services
        ]
        return {
            "recent_queries": recent_queries,
            "recent_clicked_services": recent_clicked_services,
            "global_hot_services": hot_services,
        }

    # ---------- 单服务详情（不经过检索流程）----------
    def get_service(self, service_id: str) -> dict[str, Any] | None:
        service = self.services.get(service_id)
        if service is None:
            return None
        info = route_info(service.route)
        return {
            "service_id": service.service_id,
            "service_name": service.service_name,
            "aliases": list(service.aliases),
            "service_intro": service.service_intro,
            "route": info["route"],
            "component": info["component"],
            "decision_button": info["decision_button"],
            "derived": info["derived"],
            "components": list(service.components),
        }

    # ---------- M13 拼写建议 ----------
    def spell_suggest(self, query: str) -> str | None:
        """对 query 生成「您是不是要找」建议字符串；无 OOV/无可纠错返回 None。

        - SPELL_ENABLED 关闭或纠错器未初始化 → None
        - 复用 LevenshteinCorrector.suggest（BK-tree 编辑距离 + 拼音同音纠错）
        """
        if not SPELL_ENABLED or self.spell_corrector is None or not query:
            return None
        return self.spell_corrector.suggest(query)

    # ---------- 搜索框灰色补全建议 ----------
    def suggest_query(self, user_id: str, partial: str) -> str | None:
        """生成搜索框补全建议（基于用户搜索/点击历史 + 已输入前缀）。

        LLM 不可用/异常/前缀不匹配 → 返回 None（调用方隐藏灰色建议，不影响主链路）。
        异常静默：建议是「锦上添花」，绝不让后端异常打断输入体验。
        """
        if not partial:
            return None
        try:
            recent_queries = self.store.recent_queries(user_id, 5)
            recent_click_ids = self.store.recent_clicks(user_id, 5)
            # service_id → service_name；已下线服务（不在 KB）过滤掉，避免 KeyError
            recent_clicked_names = [
                self.services[sid].service_name
                for sid in recent_click_ids
                if sid in self.services
            ]
            return self.query_suggester.suggest(
                partial, recent_queries, recent_clicked_names
            )
        except Exception:  # noqa: BLE001 - 任何异常都不影响主链路
            logger.warning(
                "suggest_query failed user=%s partial=%r",
                user_id,
                partial,
                exc_info=True,
            )
            return None

    async def suggest_query_async(
        self, user_id: str, partial: str
    ) -> str | None:
        """异步版本（M2 双入口模式）。供 /api/search/suggest 异步端点调用。"""
        if not partial:
            return None
        try:
            recent_queries = self.store.recent_queries(user_id, 5)
            recent_click_ids = self.store.recent_clicks(user_id, 5)
            recent_clicked_names = [
                self.services[sid].service_name
                for sid in recent_click_ids
                if sid in self.services
            ]
            return await self.query_suggester.suggest_async(
                partial, recent_queries, recent_clicked_names
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "suggest_query_async failed user=%s partial=%r",
                user_id,
                partial,
                exc_info=True,
            )
            return None

    # ---------- M8 页面内组件执行（本期打桩）----------
    def execute_component_action(
        self,
        service_id: str,
        component: str,
        action: str,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """M8 打桩：校验 service_id/component/action 命中 KB components 白名单后回显。

        - service_id 必须在 KB；component/action 必须在 components[].name/action 白名单
        - 命中 → 返回 {ok, service_id, component, action, echo}（echo = 请求参数快照）
        - 未命中 → 返回 None（API 层映射 404/400）
        后续可在此处转发到实际服务总线（M11 接入日志埋点）。
        """
        service = self.services.get(service_id)
        if service is None:
            return None
        matched = next(
            (
                c
                for c in service.components
                if c.get("name") == component and c.get("action") == action
            ),
            None,
        )
        if matched is None:
            return None
        echo: dict[str, Any] = {
            "service_id": service_id,
            "component": component,
            "action": action,
        }
        if params:
            echo["params"] = params
        return {
            "ok": True,
            "service_id": service_id,
            "component": component,
            "action": action,
            "echo": echo,
        }

    # ---------- 静态工具 ----------
    @staticmethod
    def _hybrid_score(
        vector_similarity: float,
        bm25_score: float,
        popularity_score: float,
        retrieval_mode: str = "hybrid",
    ) -> float:
        # 检索模式按权重归零：keyword 关闭向量信号，semantic 关闭 BM25 信号；
        # popularity 是行为信号（用户偏好），三模式都保留，不随检索模式消失。
        # 默认 "hybrid" 保 verify.py:60 静态调用 _hybrid_score(0.8,0.5,1.0) 逐字节兼容。
        vw, bw = VECTOR_WEIGHT, BM25_WEIGHT
        if retrieval_mode == "keyword":
            vw = 0.0
        elif retrieval_mode == "semantic":
            bw = 0.0
        return (
            vw * vector_similarity
            + bw * bm25_score
            + POPULARITY_WEIGHT * popularity_score
        )

    def _apply_negative_penalty(
        self, popularity_raw: dict[str, float], now: float
    ) -> dict[str, float]:
        """M13：对「点后快速跳出」服务在归一前从 popularity_raw 扣除惩罚。

        - NEGATIVE_FEEDBACK_ENABLED 关闭 → 原样返回（无负样本时也无影响）
        - 负样本计数 * NEGATIVE_PENALTY 从该服务 popularity 中扣除（不低于 0）
        - 无负样本时返回原 dict（值不变，对 normalize 与既有测试零影响）
        """
        if not NEGATIVE_FEEDBACK_ENABLED or not popularity_raw:
            return popularity_raw
        negatives = self.store.negative_signals(
            now=now,
            window_days=POPULARITY_WINDOW_DAYS,
            quick_bounce_ms=QUICK_BOUNCE_MS,
        )
        if not negatives:
            return popularity_raw
        adjusted = dict(popularity_raw)
        for sid, count in negatives.items():
            if sid in adjusted:
                adjusted[sid] = max(0.0, adjusted[sid] - NEGATIVE_PENALTY * count)
            else:
                # 服务有负样本但无正向点击 → 置一个负值基准，归一后仍为 0
                adjusted[sid] = max(0.0, 0.0 - NEGATIVE_PENALTY * count)
        return adjusted

    def service_name(self, service_id: str) -> str | None:
        service = self.services.get(service_id)
        return service.service_name if service else None
