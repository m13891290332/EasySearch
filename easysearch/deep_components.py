"""深度组件检索：分析服务页面/简介，挑出最契合 query 的组件。

对每个 top-10 服务：
    1. route 为 http(s) → ``page_fetcher.fetch_page_async`` 抓页面文本；
       抓取失败 / route 为相对路径 → 用 ``service_intro`` 兜底
    2. 调 DeepSeek LLM 分析 query 与组件相关性，挑最佳组件
    3. LLM 不可用 / 解析失败 / 超时 → ``pick_heuristic`` 启发式降级

启发式降级策略（无 Key / LLM 失败时全程走此路径）：
    - 有 components 列表 + query token 命中 component name → 选首个命中
    - 否则用 ``route_info`` 派生的入口按钮（component + decision_button）
    - route 为 http(s) 或相对路径 → href=route 供前端跳转
"""
from __future__ import annotations

import json
import logging
from typing import Any

from .dashscope import DashScopeClient
from .models import ServiceRecord, route_info
from .safety import sanitize_for_prompt, strip_html
from .utils import extract_json, tokenize

logger = logging.getLogger(__name__)

_MODEL_NAME = "deepseek-v4-flash"
_ENDPOINT = "https://api.deepseek.com/chat/completions"
_MAX_COMPONENTS_IN_PROMPT = 20
_MAX_PAGE_TEXT = 2000
_MAX_LABEL_LEN = 40
_MAX_REASON_LEN = 200


def _component_in_whitelist(
    service: ServiceRecord, component: str, action: str
) -> bool:
    """component/action 是否命中 service.components 白名单（与 engine 校验一致）。

    LLM 可能幻觉出不存在的组件名/动作；命中才允许前端触发
    ``/api/action/execute``，否则置空改走 route 跳转，避免注定 400。
    """
    for c in service.components or []:
        if not isinstance(c, dict):
            continue
        if str(c.get("name", "")) == component and str(c.get("action", "")) == action:
            return True
    return False


class ComponentAnalyzer:
    """分析服务页面/简介，挑出最契合 query 的组件。

    依赖 ``DeepSeekClient``（无 Key 时 ``enabled=False`` → 全程启发式降级，
    与 reranker/guide 等离线降级语义一致）。
    """

    def __init__(self, client: DashScopeClient) -> None:
        self.client = client

    async def analyze_async(
        self, query: str, service: ServiceRecord, page_text: str
    ) -> dict[str, Any]:
        """分析单服务，返回组件推荐项。

        返回 dict 形态：
            {label, reason, component, action, href, route, source}
        其中 source ∈ {"llm", "heuristic"}，便于前端展示来源 badge。
        """
        if not self.client.enabled:
            return self.pick_heuristic(query, service)
        prompt = self._build_prompt(query, service, page_text)
        try:
            resp = await self.client.post_json_async(
                _ENDPOINT, self._build_payload(prompt)
            )
            parsed = self._parse_response(resp, service)
            if parsed:
                parsed["source"] = "llm"
                return parsed
        except Exception as exc:
            logger.warning(
                "Component LLM analyze failed for %s: %s; fallback to heuristic",
                service.service_id,
                exc,
            )
        heur = self.pick_heuristic(query, service)
        heur["source"] = "heuristic"
        return heur

    # ---------- LLM 路径 ----------
    def _build_prompt(
        self, query: str, service: ServiceRecord, page_text: str
    ) -> str:
        info = route_info(service.route)
        components = (service.components or [])[:_MAX_COMPONENTS_IN_PROMPT]
        comp_json = json.dumps(
            [
                {"name": c.get("name", ""), "action": c.get("action", "")}
                for c in components
                if isinstance(c, dict)
            ],
            ensure_ascii=False,
        )
        page = sanitize_for_prompt(page_text)[:_MAX_PAGE_TEXT]
        return (
            "你是服务页面组件分析器。从给定服务的页面文本与组件列表中，"
            "挑出最契合用户 query 的一个组件（或页面入口动作），"
            "供用户直接点击执行或跳转。\n"
            f"query: {sanitize_for_prompt(query)}\n"
            f"service_id: {sanitize_for_prompt(service.service_id)}\n"
            f"service_name: {sanitize_for_prompt(service.service_name)}\n"
            f"route: {sanitize_for_prompt(info['route'])}\n"
            f"components: {comp_json}\n"
            f"page_text: {page}\n"
            '仅输出JSON对象：{"label":"按钮文案",'
            '"reason":"为何契合query(<=80字)",'
            '"component":"组件名(可空)","action":"动作标识(可空)",'
            '"href":"跳转链接(可空)"}。'
            "label 必填且<=20字。href 仅在可直接跳转时给出。"
        )

    def _build_payload(self, prompt: str) -> dict[str, Any]:
        return {
            "model": _MODEL_NAME,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "thinking": {"type": "enabled"},
            "reasoning_effort": "low",
        }

    def _parse_response(
        self, resp: dict[str, Any], service: ServiceRecord
    ) -> dict[str, Any]:
        content = (
            resp.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        if isinstance(content, list):
            content = "".join(
                p.get("text", "") if isinstance(p, dict) else str(p)
                for p in content
            )
        parsed = extract_json(content)
        if not isinstance(parsed, dict):
            return {}
        # 用 `or ""` 把 JSON null / 缺失统一为空串：str(None) == "None" 会让
        # 前端把字符串 "None" 当合法 component/action 调 /api/action/execute，
        # 后端白名单匹配失败返回 400 "component/action 不在服务白名单: None/None"。
        label = str(parsed.get("label") or "").strip()
        if not label:
            return {}
        reason = strip_html(str(parsed.get("reason") or "")).strip()[:_MAX_REASON_LEN]
        component = str(parsed.get("component") or "").strip()[:100]
        action = str(parsed.get("action") or "").strip()[:100]
        href = str(parsed.get("href") or "").strip()
        info = route_info(service.route)
        # 白名单校验：LLM 返回的 component/action 必须命中 service.components，
        # 否则置空（防 LLM 幻觉组件触发注定 400 的 /api/action/execute）。
        # 置空后前端走 href 跳转分支，chip 仍可点击。
        if not (component and action and _component_in_whitelist(service, component, action)):
            component, action = "", ""
        # 兜底跳转：无白名单组件可执行时，用 route 入口确保 chip 仍可点击
        if not href:
            href = info["route"]
        return {
            "label": label[:_MAX_LABEL_LEN],
            "reason": reason,
            "component": component,
            "action": action,
            "href": href,
            "route": info["route"],
            "source": "llm",
        }

    # ---------- 启发式降级 ----------
    def pick_heuristic(self, query: str, service: ServiceRecord) -> dict[str, Any]:
        """启发式降级：基于 query token 命中 component name 选最佳组件。

        - 有 components + query token 命中 component name → 选首个命中
        - 否则用 route_info 派生的入口按钮（decision_button / component）
        """
        info = route_info(service.route)
        components = service.components or []
        q_tokens = set(tokenize(query)) if query else set()
        if q_tokens and components:
            for comp in components:
                if not isinstance(comp, dict):
                    continue
                name = str(comp.get("name", ""))
                action = str(comp.get("action", ""))
                if not name or not action:
                    continue
                name_tokens = set(tokenize(name))
                if name_tokens & q_tokens:
                    return {
                        "label": name[:_MAX_LABEL_LEN],
                        "reason": f"组件「{name}」与查询词命中，可直接执行。",
                        "component": name,
                        "action": action,
                        "href": "",
                        "route": info["route"],
                        "source": "heuristic",
                    }
        # 无命中 / 无 components → 用 route 派生的入口按钮
        label = info["decision_button"] or "进入"
        return {
            "label": label[:_MAX_LABEL_LEN],
            "reason": "按路由入口推荐，可点击进入该服务页面。",
            "component": info["component"],
            "action": "",
            "href": info["route"],
            "route": info["route"],
            "source": "heuristic",
        }
