"""M10 结构化日志：JSON 行输出，每搜索一条含各阶段耗时/缓存命中/降级标志。

软依赖 structlog：可用时用其 JSON 处理链；不可用时退回 stdlib + 自定义 JsonFormatter，
输出形态一致（一行一个 JSON 对象）。遵循项目「软依赖」约定。
"""
from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any

# 软依赖 structlog
_structlog_available: bool | None = None


def _structlog_ready() -> bool:
    global _structlog_available
    if _structlog_available is not None:
        return _structlog_available
    try:
        import structlog  # noqa: F401

        _structlog_available = True
    except ImportError:
        _structlog_available = False
    return _structlog_available


# stdlib JSON 日志保留字段（其余进入 data）
_RESERVED = {
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
    "created", "msecs", "relativeCreated", "thread", "threadName",
    "processName", "process", "message", "taskName",
}


class JsonFormatter(logging.Formatter):
    """把 LogRecord 序列化为单行 JSON；extra 字段并入 data。"""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # extra 字段
        for key, val in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = val
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def setup_logging(level: str | None = None) -> None:
    """配置根日志为 JSON 输出。重复调用幂等（只追加一次 handler）。"""
    log_level = level or os.getenv("EASYSEARCH_LOG_LEVEL", "INFO").upper()
    root = logging.getLogger()
    root.setLevel(log_level)
    # 避免重复追加
    if any(getattr(h, "_easysearch_json", False) for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(log_level)
    handler.setFormatter(JsonFormatter())
    handler._easysearch_json = True  # type: ignore[attr-defined]
    root.addHandler(handler)

    if _structlog_ready():
        try:
            import structlog

            structlog.configure(
                processors=[
                    structlog.processors.add_log_level,
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.JSONRenderer(ensure_ascii=False),
                ],
                wrapper_class=structlog.make_filtering_bound_logger(
                    getattr(logging, log_level, logging.INFO)
                ),
                logger_factory=structlog.stdlib.LoggerFactory(),
            )
        except Exception:  # pragma: no cover - structlog 配置失败回退 stdlib
            pass
