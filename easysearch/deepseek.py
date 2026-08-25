from __future__ import annotations

import os
from typing import Any, Callable

from .config import DEEPSEEK_API_KEY
from .dashscope import DashScopeClient


class DeepSeekClient(DashScopeClient):
    """DeepSeek HTTP 客户端，复用 DashScopeClient 的通用逻辑，仅改 Key 来源。

    API Key 读取优先级：
        1. 构造参数 api_key
        2. config.py 写死的 DEEPSEEK_API_KEY
        3. 环境变量 DEEPSEEK_API_KEY
    """

    def __init__(
        self,
        api_key: str | None = None,
        requester: Callable[[str, bytes, dict[str, str]], dict[str, Any]] | None = None,
        timeout: int = 120,
    ) -> None:
        resolved = api_key or DEEPSEEK_API_KEY or os.getenv("DEEPSEEK_API_KEY")
        super().__init__(api_key=resolved, requester=requester, timeout=timeout)
        # M10：外部调用埋点标识（覆盖父类 dashscope → deepseek）
        self.service_tag = "deepseek"
