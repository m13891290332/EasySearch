"""搜索框灰色补全建议：DeepSeek 基于用户历史 + 当前前缀生成 omnibox 风格补全。

触发场景：用户在首页搜索框输入过程中，后端基于「已输入前缀 + 最近搜索/点击历史」
调用 DeepSeek 生成一个补全串，前端以灰色叠加层展示（Chrome omnibox 风格），
用户按 Tab 即可接受补全。

- 复用 M2 DeepSeek 客户端（async/sync 双入口）
- 复用 M1 ``sanitize_for_prompt`` / ``sanitize_text`` 输入输出清洗
- 复用 ``utils.extract_json`` 鲁棒解析 JSON（兼容裸字符串回退）
- LLM 不可用 / 解析失败 / 前缀不匹配 → 返回 None（调用方隐藏灰色建议，不影响主链路）
- thinking 关闭：per-keystroke 延迟敏感，无需多步推理（区别于 guide 的 enabled）
"""
from __future__ import annotations

import json
import logging
from typing import Any

from .dashscope import DashScopeClient
from .safety import sanitize_for_prompt, sanitize_text
from .utils import extract_json

logger = logging.getLogger(__name__)

# 补全串最大长度：查询补全不需过长，超过即截断后再校验前缀
_MAX_COMPLETION = 50


class QuerySuggester:
    """DeepSeek 搜索框补全建议生成器。

    仿 :class:`easysearch.guide.GuideGenerator` 模式：构造 payload → 调远程 →
    解析 + 前缀强制校验 → 失败返 None 降级。
    """

    model_name = "deepseek-v4-flash"
    endpoint = "https://api.deepseek.com/chat/completions"

    def __init__(self, deepseek_client: DashScopeClient) -> None:
        self.client = deepseek_client

    # ---------- 共享：构造 prompt payload ----------
    def _build_payload(
        self,
        partial: str,
        recent_queries: list[str],
        recent_clicked_names: list[str],
    ) -> dict[str, Any]:
        safe_partial = sanitize_for_prompt(partial)
        safe_qs = [sanitize_for_prompt(q) for q in recent_queries[:5]]
        safe_cs = [sanitize_for_prompt(n) for n in recent_clicked_names[:5]]
        prompt = (
            "你是搜索框补全助手。基于用户已输入的前缀 + 历史，生成一个补全建议。"
            "建议必须严格以用户前缀开头（区分大小写），且长度大于前缀本身。"
            "结合用户最近搜索/点击偏好生成更贴合的补全。"
            '仅输出JSON：{"completion":"..."}，不要解释/Markdown/引号包裹。'
            f"用户前缀：{safe_partial}\n"
            f"最近搜索：{json.dumps(safe_qs, ensure_ascii=False)}\n"
            f"最近点击服务：{json.dumps(safe_cs, ensure_ascii=False)}"
        )
        return {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            # per-keystroke 延迟敏感：关闭 thinking 加速响应
            "thinking": {"type": "disabled"},
            "stream": False,
        }

    # ---------- 共享：提取 LLM 响应内容 ----------
    @staticmethod
    def _extract_content(response: dict[str, Any]) -> str:
        """从 chat/completions 响应提取 message.content（兼容 list content 形式）。"""
        raw_content = (
            response.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        if isinstance(raw_content, list):
            raw_content = "".join(
                part.get("text", "") if isinstance(part, dict) else str(part)
                for part in raw_content
            )
        return raw_content if isinstance(raw_content, str) else str(raw_content)

    # ---------- 共享：解析 + 前缀强制校验 ----------
    def _parse_completion(
        self, response: dict[str, Any], partial: str
    ) -> str | None:
        """解析 LLM 输出为补全串；前缀不匹配/无内容 → None。

        三重保险：
        1. 双模式解析：先 JSON ``{"completion":"..."}``，失败回退裸字符串（剥引号/代码块）
        2. ``sanitize_text`` 剥控制字符 + 限长 50
        3. 硬性前缀校验：``completion.startswith(partial)`` 且比 partial 长
        """
        raw = self._extract_content(response)
        parsed = extract_json(raw)
        completion: Any = None
        if isinstance(parsed, dict):
            completion = parsed.get("completion")
        elif isinstance(parsed, str):
            completion = parsed
        if completion is None:
            # 裸字符串回退：LLM 偶尔不遵循 JSON 指令，剥外层引号/代码块标记
            completion = raw.strip().strip("`").strip('"').strip("'")
        if not isinstance(completion, str) or not completion:
            return None
        # 清洗：剥控制字符 + 限长
        completion = sanitize_text(completion).strip()
        if not completion:
            return None
        if len(completion) > _MAX_COMPLETION:
            completion = completion[:_MAX_COMPLETION]
        # 硬性前缀校验：必须严格以 partial 开头且比 partial 长
        # （宁可丢弃也不展示错误建议——Chrome omnibox 契约）
        if not completion.startswith(partial):
            return None
        if len(completion) <= len(partial):
            return None
        return completion

    # ---------- 同步入口 ----------
    def suggest(
        self,
        partial: str,
        recent_queries: list[str],
        recent_clicked_names: list[str],
    ) -> str | None:
        """生成补全建议。LLM 不可用/异常/前缀不匹配 → 返回 None。"""
        if not self.client.enabled or not partial:
            return None
        payload = self._build_payload(partial, recent_queries, recent_clicked_names)
        try:
            response = self.client.post_json(self.endpoint, payload)
        except RuntimeError:
            # M12：5xx/超时重试 2 次后仍失败 → 静默降级
            return None
        return self._parse_completion(response, partial)

    # ---------- 异步入口（M2） ----------
    async def suggest_async(
        self,
        partial: str,
        recent_queries: list[str],
        recent_clicked_names: list[str],
    ) -> str | None:
        """异步版本。供 /api/search/suggest 异步端点调用，避免阻塞事件循环。"""
        if not self.client.enabled or not partial:
            return None
        payload = self._build_payload(partial, recent_queries, recent_clicked_names)
        try:
            response = await self.client.post_json_async(self.endpoint, payload)
        except RuntimeError:
            return None
        return self._parse_completion(response, partial)
