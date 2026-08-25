"""M16 答案模式：LLM 生成步骤化指引答案 + 内嵌服务跳转。

guide 意图（M5 增）触发：DeepSeek 基于 KB 命中服务生成结构化步骤答案，
服务引用用 ``[[service_id]]`` 内联标记，后端解析为
``{steps: [{step_text, services: [{service_id,name,route,component,decision_button}]}]}``。

- 复用 M2 DeepSeek 客户端（async/sync 双入口）
- 复用 M1 ``sanitize_for_prompt`` / ``sanitize_text`` 做输入输出清洗
- service_id 必须在 KB 白名单（非法引用过滤，打 WARN 由调用方）
- LLM 不可用 / 解析失败 → 返回 None，调用方降级为 list 模式（不影响主链路）
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from .dashscope import DashScopeClient
from .safety import sanitize_for_prompt, sanitize_text
from .utils import extract_json

logger = logging.getLogger(__name__)

# 内联服务引用标记：[[service_id]]
_REF_PATTERN = re.compile(r"\[\[\s*([^\[\]]+?)\s*\]\]")
# HTML/script 标签剥离（前端用 text 渲染，剥离仅为洁净展示，非 XSS 防御）
_TAG_RE = re.compile(r"<[^>]+>")

_MAX_STEPS = 8
_MAX_STEP_TEXT = 300


class GuideGenerator:
    """DeepSeek 步骤化指引答案生成器（M16）。"""

    model_name = "deepseek-v4-flash"
    endpoint = "https://api.deepseek.com/chat/completions"

    def __init__(self, deepseek_client: DeepSeekClient) -> None:
        self.client = deepseek_client

    # ---------- 共享：构造 prompt payload ----------
    def _build_payload(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        safe_query = sanitize_for_prompt(query)
        compact = [
            {
                "service_id": sanitize_for_prompt(item["service_id"]),
                "service_name": sanitize_for_prompt(item["service_name"]),
                "service_intro": sanitize_for_prompt(item["service_intro"]),
                "route": sanitize_for_prompt(item.get("route", "")),
            }
            for item in candidates
        ]
        prompt = (
            "你是平台服务导航助手。用户提出一个指引型问题，请基于下列候选服务"
            "生成简洁的步骤化操作指引。每一步是一句话，若该步骤需要使用某服务，"
            "在文本中用 [[service_id]] 内联标记（service_id 必须来自下列 services，"
            "不得编造）。步骤数 2-6 步，按操作先后排序。"
            '仅输出JSON：{"steps":[{"step_text":"..."}]}，step_text 中内嵌 [[service_id]]。'
            f"query: {safe_query}\n"
            f"services: {json.dumps(compact, ensure_ascii=False)}"
        )
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "thinking": {"type": "enabled"},
            "stream": False,
        }

    # ---------- 共享：解析 + 校验 LLM 输出 ----------
    @staticmethod
    def _extract_content(response: dict[str, Any]) -> str:
        raw_content = (
            response.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        if isinstance(raw_content, list):
            raw_content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw_content
            )
        return raw_content if isinstance(raw_content, str) else str(raw_content)

    def _parse_guide(
        self, response: dict[str, Any], candidates: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        raw = self._extract_content(response)
        parsed = extract_json(raw)
        steps_raw: Any = None
        if isinstance(parsed, dict):
            steps_raw = parsed.get("steps")
        elif isinstance(parsed, list):
            # 兼容 LLM 直接输出数组
            steps_raw = parsed
        if not isinstance(steps_raw, list) or not steps_raw:
            return None

        kb = {item["service_id"]: item for item in candidates}
        kb_ids = set(kb)

        steps: list[dict[str, Any]] = []
        for s in steps_raw[:_MAX_STEPS]:
            if isinstance(s, str):
                step_text = s
            elif isinstance(s, dict):
                step_text = str(s.get("step_text", ""))
            else:
                continue
            # M1：剥 HTML/script 标签 + 控制字符 + 限长（前端用 text 渲染，剥离为洁净展示）
            step_text = _TAG_RE.sub("", step_text)
            step_text = sanitize_text(step_text).strip()
            if not step_text:
                continue
            if len(step_text) > _MAX_STEP_TEXT:
                step_text = step_text[:_MAX_STEP_TEXT] + "…"
            # 提取内联 [[service_id]] 引用，白名单过滤
            refs = _REF_PATTERN.findall(step_text)
            services: list[dict[str, Any]] = []
            seen: set[str] = set()
            invalid_refs: list[str] = []
            for sid in refs:
                sid = sid.strip()
                if sid in kb_ids and sid not in seen:
                    c = kb[sid]
                    services.append(
                        {
                            "service_id": sid,
                            "service_name": c.get("service_name", ""),
                            "route": c.get("route", ""),
                            "component": c.get("component", ""),
                            "decision_button": c.get("decision_button", ""),
                        }
                    )
                    seen.add(sid)
                elif sid and sid not in kb_ids:
                    invalid_refs.append(sid)
            if invalid_refs:
                # 非法引用：过滤并打 WARN（不抛错，保主链路）
                logger.warning(
                    "guide answer 含非法 service 引用，已过滤: %s", invalid_refs
                )
            steps.append({"step_text": step_text, "services": services})
        if not steps:
            return None
        return {"steps": steps}

    # ---------- 同步入口 ----------
    def generate_guide(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        if not self.client.enabled or not candidates:
            return None
        payload = self._build_payload(query, candidates)
        try:
            response = self.client.post_json(self.endpoint, payload)
        except RuntimeError:
            return None
        return self._parse_guide(response, candidates)

    # ---------- 异步入口（M2） ----------
    async def generate_guide_async(
        self, query: str, candidates: list[dict[str, Any]]
    ) -> dict[str, Any] | None:
        if not self.client.enabled or not candidates:
            return None
        payload = self._build_payload(query, candidates)
        try:
            response = await self.client.post_json_async(self.endpoint, payload)
        except RuntimeError:
            return None
        return self._parse_guide(response, candidates)
