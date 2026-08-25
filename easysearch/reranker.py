from __future__ import annotations

import json
import logging
from typing import Any

from .config import REASON_ENABLED, REASON_EFFORT
from .dashscope import DashScopeClient
from .deepseek import DeepSeekClient
from .safety import sanitize_for_prompt, validate_llm_output
from .utils import extract_json, tokenize

logger = logging.getLogger(__name__)

# A7：rank=1 的 reason 若含以下负面词，视为与排序矛盾，回退到模板
_REASON_NEGATIVE_HINTS = ("次", "不", "较弱", "次要", "较差", "无关", "更低", "靠后")

# M12：rank 较小（位于 top-K 前半）的 reason 含负面词视为单调性矛盾 → 回退模板
# 用于扩展 A7 的 rank=1 校验到全 rank 单调性（plan 要求"扩展到全 rank 单调性"）
_REASON_POSITIVE_HINTS = ("最相关", "最佳", "首选", "top1", "第一位", "最优", "强相关")


class DeepSeekReasoner:
    """调用 deepseek-v4-flash 对比 Query 与 Top-20 候选内容，生成排序理由。

    M2：reason 默认关闭（REASON_ENABLED=False），开启时 effort 默认 low（high 是 SLA 杀手）。
        提供 generate_reasons（同步，兼容旧链路）与 generate_reasons_async（并发 gather 用）。
    A7：prompt 中嵌入 rank + rerank_score，要求 LLM 按排序顺序生成 reason；
        后处理校验 rank=1 的 reason 不含负面词，矛盾则回退模板。
    """

    model_name = "deepseek-v4-flash"
    endpoint = "https://api.deepseek.com/chat/completions"

    def __init__(self, deepseek_client: DeepSeekClient) -> None:
        self.client = deepseek_client

    # ---------- 共享：构造 prompt payload ----------
    def _build_payload(self, query: str, ranked_candidates: list[dict[str, Any]]) -> dict[str, Any]:
        # M1：prompt 构造前对 query/候选文本做防御性清洗，防越狱话术污染
        safe_query = sanitize_for_prompt(query)
        compact = [
            {
                "rank": idx + 1,
                "service_id": sanitize_for_prompt(item["service_id"]),
                "service_name": sanitize_for_prompt(item["service_name"]),
                "service_intro": sanitize_for_prompt(item["service_intro"]),
                "route": sanitize_for_prompt(item.get("route", "")),
                "rerank_score": float(item.get("rerank_score", item.get("score", 0.0))),
            }
            for idx, item in enumerate(ranked_candidates)
        ]
        prompt = (
            "你是搜索重排解释器。以下是按相关性降序排列的候选"
            "（rank=1 为最相关，rerank_score 为重排分数）。"
            "请严格按 candidates 数组顺序输出排序理由："
            "rank 越小的 reason 应表达越强的相关性（rank=1 表达『最相关』语义），"
            "禁止出现 rank=1 但 reason 说『次相关/不相关/较弱』之类的矛盾。"
            '仅输出JSON数组，每项格式为{"service_id":"...","reason":"..."}。'
            f"query: {safe_query}\n"
            f"candidates: {json.dumps(compact, ensure_ascii=False)}"
        )
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "enabled"},
            "reasoning_effort": REASON_EFFORT,  # M2：默认 low
            "stream": False,
        }

    # ---------- 共享：解析 + 校验 LLM 输出 ----------
    def _parse_reasons(
        self, response: dict[str, Any], ranked_candidates: list[dict[str, Any]]
    ) -> dict[str, str]:
        raw_content = (
            response.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        if isinstance(raw_content, list):
            raw_content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw_content
            )
        parsed = extract_json(raw_content)
        if not isinstance(parsed, list):
            return {}
        reasons: dict[str, str] = {}
        for row in parsed:
            if not isinstance(row, dict):
                continue
            service_id = str(row.get("service_id", "")).strip()
            reason = str(row.get("reason", "")).strip()
            if service_id and reason:
                reasons[service_id] = reason
        # M1：校验 LLM 输出（service_id 白名单 + reason 剥 HTML + 限长 200）
        kb_ids = {item["service_id"] for item in ranked_candidates}
        reasons = validate_llm_output(reasons, kb_ids)
        # M12：扩展 A7 单调性校验到全 rank——top 前半不应含负面词，
        # 底部不应含强正面词（避免与排序矛盾）；命中则删除该条回退模板。
        self._validate_rank_monotonicity(ranked_candidates, reasons)
        return reasons

    # ---------- 同步入口（兼容旧链路 / 测试） ----------
    def generate_reasons(self, query: str, ranked_candidates: list[dict[str, Any]]) -> dict[str, str]:
        # M2：reason 默认关闭，避免 high effort 拖垮端到端时延
        if not REASON_ENABLED or not self.client.enabled or not ranked_candidates:
            return {}
        payload = self._build_payload(query, ranked_candidates)
        # M12：LLM 输出 JSON 解析失败 → 重试 1 次；仍失败则降级模板（返回 {}）
        response: dict[str, Any] | None = None
        try:
            response = self.client.post_json(self.endpoint, payload)
        except RuntimeError:
            return {}
        reasons = self._parse_reasons(response, ranked_candidates)
        if not reasons:
            # 第一次解析为空——可能是 LLM 输出脏 JSON，重试 1 次
            logger.warning(
                "deepseek reasons parse empty on first attempt, retrying once (query=%s)",
                query[:80],
            )
            try:
                response = self.client.post_json(self.endpoint, payload)
            except RuntimeError:
                return {}
            reasons = self._parse_reasons(response, ranked_candidates)
        return reasons

    # ---------- 异步入口（M2：与 rerank 并发 gather） ----------
    async def generate_reasons_async(
        self, query: str, ranked_candidates: list[dict[str, Any]]
    ) -> dict[str, str]:
        if not REASON_ENABLED or not self.client.enabled or not ranked_candidates:
            return {}
        payload = self._build_payload(query, ranked_candidates)
        # M12：LLM 输出 JSON 解析失败 → 重试 1 次；仍失败则降级模板（返回 {}）
        response: dict[str, Any] | None = None
        try:
            response = await self.client.post_json_async(self.endpoint, payload)
        except RuntimeError:
            return {}
        reasons = self._parse_reasons(response, ranked_candidates)
        if not reasons:
            logger.warning(
                "deepseek reasons parse empty on first async attempt, retrying once (query=%s)",
                query[:80],
            )
            try:
                response = await self.client.post_json_async(self.endpoint, payload)
            except RuntimeError:
                return {}
            reasons = self._parse_reasons(response, ranked_candidates)
        return reasons

    @staticmethod
    def _validate_top_reason_consistency(
        ranked_candidates: list[dict[str, Any]], reasons: dict[str, str]
    ) -> None:
        """A7 兼容保留：若 rank=1 的 reason 含负面词，删除该条让上层回退到模板。

        M12 已将此校验扩展为 _validate_rank_monotonicity（覆盖全 rank），
        但保留本方法以兼容旧测试断言。
        """
        if not ranked_candidates:
            return
        top = ranked_candidates[0]
        top_sid = top.get("service_id")
        top_reason = reasons.get(top_sid, "")
        if top_reason and any(hint in top_reason for hint in _REASON_NEGATIVE_HINTS):
            reasons.pop(top_sid, None)

    @staticmethod
    def _validate_rank_monotonicity(
        ranked_candidates: list[dict[str, Any]], reasons: dict[str, str]
    ) -> None:
        """M12：扩展 A7 到全 rank 单调性校验。

        - 前半 rank（rank <= N//2）：reason 不应含负面词（"次"/"较弱"/"无关" 等），
          命中则删除该条 → 上层回退模板。
        - 后半 rank（rank > N//2）：reason 不应含强正面词（"最相关"/"最佳" 等），
          命中则删除该条 → 避免与排序矛盾。

        校验只删除矛盾的条目，不影响其他 rank 的合法 reason。
        """
        if not ranked_candidates:
            return
        n = len(ranked_candidates)
        half = max(1, n // 2)
        for idx, item in enumerate(ranked_candidates):
            sid = item.get("service_id")
            if not sid:
                continue
            rank = idx + 1
            reason = reasons.get(sid, "")
            if not reason:
                continue
            if rank <= half:
                if any(hint in reason for hint in _REASON_NEGATIVE_HINTS):
                    reasons.pop(sid, None)
            else:
                if any(hint in reason for hint in _REASON_POSITIVE_HINTS):
                    reasons.pop(sid, None)


# 向后兼容别名
Qwen3VLPlusReasoner = DeepSeekReasoner


class Qwen3VLReranker:
    """qwen3-vl-rerank 适配器：对 Top-20 候选重排，并附带排序理由。

    rerank（qwen3-vl-rerank，DashScope）+ 排序理由（deepseek-v4-flash，DeepSeek）
    无 API Key / 失败：本地关键词重合度 rerank + 默认理由

    M2：rerank_async 仅做重排 + 模板理由（不取 LLM reason），
        LLM reason 由 engine.search_async 通过 asyncio.gather 并发获取后覆盖。
    """

    model_name = "qwen3-vl-rerank"
    endpoint = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

    def __init__(self, dashscope_client: DashScopeClient, reasoner: DeepSeekReasoner) -> None:
        self.client = dashscope_client
        self.reasoner = reasoner

    # ---------- 同步入口（兼容旧链路 / 测试） ----------
    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        reranked = (
            self._remote_rerank(query, candidates)
            if self.client.enabled
            else self._local_rerank(query, candidates)
        )
        reasons = self.reasoner.generate_reasons(query, reranked)
        for item in reranked:
            if not item.get("rerank_reason"):
                item["rerank_reason"] = reasons.get(
                    item["service_id"],
                    self._build_template_reason(query, item),
                )
        return reranked

    # ---------- 异步入口（M2：并发重排，仅模板理由） ----------
    async def rerank_async(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []
        if self.client.enabled:
            reranked = await self._remote_rerank_async(query, candidates)
        else:
            reranked = self._local_rerank(query, candidates)
        # 仅附模板 reason；LLM reason 由 engine.search_async gather 后覆盖
        for item in reranked:
            if not item.get("rerank_reason"):
                item["rerank_reason"] = self._build_template_reason(query, item)
        return reranked

    @staticmethod
    def _build_template_reason(query: str, item: dict[str, Any]) -> str:
        """基于 query token 在 item 各字段命中分布的差异化模板 reason。

        REASON_ENABLED=False 时的默认理由（替代千篇一律的"综合相关性与关键词覆盖"）。
        所有分支都拼"综合分X.XXXX。"尾巴，保前端展示一致性 + 分数透明。
        REASON_ENABLED=True 时由 engine 层 LLM reason 覆盖（见 search_async）。
        """
        score = item.get("rerank_score", item.get("score", 0.0))
        score_tail = f"综合分{score:.4f}。"
        if not query:
            return f"综合相关性与关键词覆盖，{score_tail}"
        q_tokens = set(tokenize(query))
        if not q_tokens:
            return f"综合相关性与关键词覆盖，{score_tail}"
        name = str(item.get("service_name", "") or "")
        aliases = item.get("aliases", []) or []
        intro = str(item.get("service_intro", "") or "")
        # 名称命中：token 命中名称，或名称整体出现在 query 中
        name_hit = any(t in name for t in q_tokens) or (name and name in query)
        if name_hit:
            return f"服务名「{name}」直接命中查询，{score_tail}"
        # 别名命中
        for a in aliases:
            a = str(a or "")
            if not a:
                continue
            if a in query or any(t in a for t in q_tokens):
                return f"别名「{a}」命中查询，{score_tail}"
        # 简介命中
        if any(t in intro for t in q_tokens):
            return f"服务简介包含查询关键词，{score_tail}"
        # 无字面命中：纯语义
        return f"语义相似匹配（无关键词字面命中），{score_tail}"

    def _remote_rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        docs = [self._candidate_text(item) for item in candidates]
        payload = {
            "model": self.model_name,
            "input": {"query": query, "documents": docs},
            "parameters": {"return_documents": True, "top_n": len(candidates)},
        }
        try:
            response = self.client.post_json(self.endpoint, payload)
            return self._map_rerank_response(response, query, candidates)
        except RuntimeError:
            return self._local_rerank(query, candidates)

    async def _remote_rerank_async(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        docs = [self._candidate_text(item) for item in candidates]
        payload = {
            "model": self.model_name,
            "input": {"query": query, "documents": docs},
            "parameters": {"return_documents": True, "top_n": len(candidates)},
        }
        try:
            response = await self.client.post_json_async(self.endpoint, payload)
            return self._map_rerank_response(response, query, candidates)
        except RuntimeError:
            return self._local_rerank(query, candidates)

    def _map_rerank_response(
        self, response: dict[str, Any], query: str, candidates: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """把 rerank 接口响应映射为按 rerank_score 降序的候选列表；空则回退本地。"""
        rows = response.get("output", {}).get("results", [])
        if not isinstance(rows, list) or not rows:
            return self._local_rerank(query, candidates)
        mapped: list[dict[str, Any]] = []
        for row in rows:
            index = row.get("index")
            if not isinstance(index, int) or index < 0 or index >= len(candidates):
                continue
            item = dict(candidates[index])
            relevance = row.get("relevance_score")
            item["rerank_score"] = (
                float(relevance) if isinstance(relevance, (int, float)) else item["score"]
            )
            mapped.append(item)
        if not mapped:
            return self._local_rerank(query, candidates)
        mapped.sort(key=lambda x: x["rerank_score"], reverse=True)
        return mapped

    def _local_rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        query_tokens = set(tokenize(query))
        reranked: list[dict[str, Any]] = []
        for item in candidates:
            text_tokens = set(tokenize(self._candidate_text(item)))
            overlap = len(query_tokens & text_tokens)
            ranked = dict(item)
            ranked["rerank_score"] = ranked["score"] + 0.01 * overlap
            ranked["rerank_reason"] = self._build_template_reason(query, ranked)
            reranked.append(ranked)
        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked

    @staticmethod
    def _candidate_text(item: dict[str, Any]) -> str:
        return (
            f"{item['service_name']} {' '.join(item.get('aliases', []))} "
            f"{item.get('service_intro', '')} {item.get('route', '')} "
            f"{item.get('component', '')} {item.get('decision_button', '')}"
        )
