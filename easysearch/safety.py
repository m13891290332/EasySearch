"""EasySearch 安全基座：输入清洗 / 提示词注入防御 / 路由校验 / LLM 输出校验。

M1 目标：消除密钥泄露；防御 LLM 提示词注入与恶意特殊字符。

对外能力：
    - sanitize_query(q)        : 搜索词清洗，命中注入关键词则抛 PromptInjectionError
    - sanitize_for_prompt(t)  : prompt 构造前清洗（注入关键词替换为 [filtered]，不抛异常）
    - sanitize_text(text)      : KB 字段清洗（剥控制/零宽字符，限长）
    - validate_route_url(route): 仅允许相对路径或 http(s)://mailto，拒绝 javascript:/data: 等
    - safe_route(route)        : 清洗路由，不安全路径置空（保留结构）
    - strip_html(text)         : 剥除 HTML/script 标签并转义实体
    - validate_llm_output(obj, kb_ids): 校验 LLM 排序理由输出
"""
from __future__ import annotations

import html
import re
from typing import Any, Iterable

# 控制字符 + 零宽字符（含 \t\n\r；U+200B-U+200F / U+2028-U+202E / U+2060 / U+FEFF 等）
_CONTROL_RE = re.compile(
    r"[\x00-\x1F\x7F\u200B-\u200F\u2028\u2029\u202A-\u202E\u2060\uFEFF]"
)

# 提示词注入关键词（中英文，覆盖常见越狱话术）
_INJECTION_PATTERNS = [
    "忽略上述", "忽略上面", "忽略之前", "忽略前面",
    "无视上述", "无视上面", "无视之前",
    "ignore previous", "ignore above", "ignore prior", "ignore all previous",
    "disregard previous", "disregard above",
    "system:", "</system>", "<|system|>", "<|im_start|>",
    "role:", "现在你是", "现在你扮演", "现在你是一个",
    "你的新指令", "新指令如下", "override previous", "override instructions",
    "你被重新设定", "jailbreak",
]
_INJECTION_RE = re.compile(
    "|".join(re.escape(p) for p in _INJECTION_PATTERNS),
    re.IGNORECASE,
)

# HTML/script 标签剥离
_TAG_RE = re.compile(r"<[^>]+>")

# 路由危险协议
_UNSAFE_SCHEMES = ("javascript:", "data:", "vbscript:", "file:", "about:")

MAX_QUERY_LEN = 200
MAX_TEXT_LEN = 2000
MAX_REASON_LEN = 200
MAX_ID_LEN = 100


class PromptInjectionError(ValueError):
    """检测到疑似提示词注入，查询被拦截。"""


# ---------------------------------------------------------------- 查询清洗
def sanitize_query(query: Any) -> str:
    """搜索词清洗：限长 200；剥控制/零宽字符；命中注入关键词则拦截。

    命中注入关键词 → 抛 PromptInjectionError（由 API 层捕获返回 400）。
    """
    if not isinstance(query, str):
        query = str(query or "")
    cleaned = _CONTROL_RE.sub("", query).strip()
    if len(cleaned) > MAX_QUERY_LEN:
        cleaned = cleaned[:MAX_QUERY_LEN]
    if _INJECTION_RE.search(cleaned):
        raise PromptInjectionError("检测到疑似提示词注入，查询已被拦截")
    return cleaned


def sanitize_for_prompt(text: Any) -> str:
    """prompt 构造前清洗：剥控制/零宽字符；注入关键词替换为 [filtered]（不抛异常）。

    用于 reranker/reasoner 拼 LLM prompt 时，对 query/候选文本做防御性清洗，
    避免候选服务文本里的越狱话术污染排序理由生成。
    """
    if not isinstance(text, str):
        text = str(text or "")
    cleaned = _CONTROL_RE.sub("", text)
    cleaned = _INJECTION_RE.sub("[filtered]", cleaned)
    return cleaned


def is_prompt_injection(query: Any) -> bool:
    """非抛出版注入检测：query 是否命中提示词注入关键词（供分类器降级判断）。

    与 sanitize_query 的区别：sanitize_query 命中即抛 PromptInjectionError（→400）；
    本函数仅返回布尔，供 query_classifier 在 DeepSeek 不可用时把攻击类 query 归入
    irrelevant/prompt_attack（需求3：未命中提示而非 400），不阻断主链路。
    """
    if not isinstance(query, str):
        query = str(query or "")
    return _INJECTION_RE.search(query) is not None


# ---------------------------------------------------------------- KB 字段
def sanitize_text(text: Any, max_len: int = MAX_TEXT_LEN) -> str:
    """KB 字段清洗：剥控制/零宽字符，限长（默认 2000）。"""
    if not isinstance(text, str):
        text = str(text or "")
    cleaned = _CONTROL_RE.sub("", text)
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len]
    return cleaned


def strip_html(text: Any) -> str:
    """剥除 HTML/script 标签并转义 HTML 实体。"""
    if not isinstance(text, str):
        return ""
    no_tags = _TAG_RE.sub("", text)
    return html.escape(no_tags, quote=False)


# ---------------------------------------------------------------- 路由校验
def validate_route_url(route: Any) -> bool:
    """仅允许相对路径或 http(s)://mailto；拒绝 javascript:/data:/vbscript: 等。

    route 可为字符串或 dict（取 path 字段校验）。
    """
    if route is None:
        return False
    if isinstance(route, dict):
        route = route.get("path", "")
    route = str(route or "").strip()
    if not route:
        return False
    lower = route.lower()
    # 显式拒绝危险协议
    if any(lower.startswith(s) for s in _UNSAFE_SCHEMES):
        return False
    # 带 scheme 的（xxx:// 或 xxx:）只允许 http/https/mailto
    m = re.match(r"^([a-z][a-z0-9+.\-]*):(?://|(?=$))", lower)
    if m:
        return m.group(1) in ("http", "https", "mailto")
    # 无 scheme：相对路径（含锚点），视为安全
    return True


def safe_route(route: Any) -> Any:
    """清洗路由：保留 dict 结构，不安全 path 置空（不阻断 KB 加载）。

    - dict 路由：path 不安全则置空，保留 component/action_button
    - string 路由：不安全则返回空串
    """
    if route is None:
        return ""
    if isinstance(route, dict):
        path = str(route.get("path", ""))
        if not validate_route_url(path):
            return {**route, "path": ""}
        return route
    if isinstance(route, str):
        return route if validate_route_url(route) else ""
    return str(route) if validate_route_url(str(route)) else ""


# ---------------------------------------------------------------- LLM 输出
def validate_llm_output(
    reasons: dict[str, str], kb_ids: Iterable[str]
) -> dict[str, str]:
    """校验 LLM 排序理由输出：service_id 必须在 KB 白名单；reason 限长并剥 HTML。

    返回清洗后的 {service_id: reason}，非法 service_id 与空 reason 被过滤。
    """
    whitelist = set(kb_ids)
    safe: dict[str, str] = {}
    for service_id, reason in reasons.items():
        sid = str(service_id or "").strip()[:MAX_ID_LEN]
        if sid not in whitelist:
            continue
        text = strip_html(str(reason or "")).strip()
        if len(text) > MAX_REASON_LEN:
            text = text[:MAX_REASON_LEN]
        if text:
            safe[sid] = text
    return safe
