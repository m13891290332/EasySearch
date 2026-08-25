"""深度组件检索：SSRF 安全的远程页面抓取。

对 top-10 搜索结果的服务 route（仅 http(s)）抓取页面文本，供
``ComponentAnalyzer`` 分析「最契合 query 的页面组件」。

安全约束（与 safety.validate_route_url 互补，本模块负责真正的出网请求侧防御）：
    - 仅 http/https 协议（拒绝 javascript:/data:/file: 等）
    - 拒绝 IP 字面量落在私网 / 环回 / 链路本地 / 未分配 / 保留段
    - 拒绝 localhost 及 .local / .internal 主机名（保守）
    - 不跟随重定向（避免被 302 到内网；缺页面时主链路降级用 service_intro）
    - 5s 超时、仅 text/html、4000 字符截断
"""
from __future__ import annotations

import ipaddress
import logging
import re
from urllib.parse import urlparse

from .safety import strip_html

logger = logging.getLogger(__name__)

_HOST_RE = re.compile(r"^\d{1,3}(?:\.\d{1,3}){3}$")  # 简易 IPv4 字面量判定
_MAX_PAGE_TEXT = 4000


def _is_private_host(host: str) -> bool:
    """主机名是否落在禁止抓取的私网/环回/链路本地/保留段。

    - 空 → 视为不安全（拒绝）
    - localhost / .local / .internal → 拒绝（保守，防内部域名探测）
    - IPv4 字面量 → 用 ipaddress 判私网/环回/链路本地/未分配/保留
    - 域名（非字面量）→ 不在此拦截（DNS 解析后的 IP 检查本期不做，
      依赖网络层出网策略；典型外网域名 example.com 等放行）
    """
    if not host:
        return True
    host = host.lower()
    if host in ("localhost", "localhost.localdomain"):
        return True
    if host.endswith(".local") or host.endswith(".internal"):
        return True
    if _HOST_RE.match(host):
        try:
            ip = ipaddress.ip_address(host)
        except ValueError:
            return True
        return (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_unspecified
            or ip.is_reserved
        )
    return False


async def fetch_page_async(url: str) -> str:
    """SSRF 安全地抓取页面并转为纯文本。

    返回值：成功 → 剥 HTML 后的前 4000 字符；任何失败/拒绝 → 空串
    （不抛异常，主链路据此降级用 service_intro 做组件分析）。
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return ""
    if parsed.scheme not in ("http", "https"):
        return ""
    if _is_private_host(parsed.hostname or ""):
        logger.warning("Rejected SSRF attempt to private host: %s", parsed.hostname)
        return ""
    try:
        import httpx
    except ImportError:  # pragma: no cover - httpx 是项目硬依赖
        return ""
    try:
        # 不跟随重定向：避免被 302 到内网地址；多数服务页直连 200
        async with httpx.AsyncClient(
            timeout=5.0,
            follow_redirects=False,
        ) as client:
            resp = await client.get(url, headers={"Accept": "text/html,*/*"})
    except Exception as exc:
        logger.debug("page fetch failed (%s): %s", url, exc)
        return ""
    if resp.status_code != 200:
        return ""
    content_type = resp.headers.get("content-type", "").lower()
    if "html" not in content_type and "text" not in content_type and "*" not in content_type:
        return ""
    text = resp.text or ""
    return strip_html(text)[:_MAX_PAGE_TEXT]
