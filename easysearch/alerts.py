"""M10 轻量告警：基于 MetricsCollector 滚动窗口评估规则，触发 ERROR 日志 + 可选 webhook。

规则（plan.md M10 §3）：
  - error_rate > 5%            （最近窗口错误率）
  - P95 > 1s                   （端到端延迟）
  - cache_hit_rate < 30%       （缓存命中率过低）
  - DashScope 连续失败 ≥ 5 次  （外部服务连续故障）
  - DB 池满                    （本期无连接池 gauge，预留扩展点）

webhook：环境变量 ``EASYSEARCH_ALERT_WEBHOOK`` 设置后，触发告警时 POST JSON。
遵循「软依赖」约定：无 prometheus_client 时仍可用进程内指标评估。
"""
from __future__ import annotations

import json
import logging
import os
import urllib.request
from typing import Any

logger = logging.getLogger(__name__)

# 阈值（可经环境变量覆盖）
_ERROR_RATE_THRESHOLD = float(os.getenv("EASYSEARCH_ALERT_ERROR_RATE", "0.05"))
_P95_MS_THRESHOLD = float(os.getenv("EASYSEARCH_ALERT_P95_MS", "1000"))
_P95_MS_THRESHOLD_DISPLAY = int(_P95_MS_THRESHOLD)
_CACHE_HIT_THRESHOLD = float(os.getenv("EASYSEARCH_ALERT_CACHE_HIT", "0.30"))
_EXT_CONSEC_FAIL_THRESHOLD = int(os.getenv("EASYSEARCH_ALERT_EXT_FAIL", "5"))


class Alert:
    """单条告警。"""

    __slots__ = ("level", "rule", "message", "value")

    def __init__(self, level: str, rule: str, message: str, value: Any) -> None:
        self.level = level
        self.rule = rule
        self.message = message
        self.value = value

    def to_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "rule": self.rule,
            "message": self.message,
            "value": self.value,
        }


class AlertChecker:
    """评估告警规则，返回告警列表并触发日志/webhook。

    与 MetricsCollector 解耦：调用方传入 health_summary + external consecutive failures。
    """

    def evaluate(
        self,
        health: dict[str, Any],
        external_consecutive_failures: dict[str, int] | None = None,
    ) -> list[Alert]:
        alerts: list[Alert] = []
        recent_total = health.get("recent_total", 0)
        # 需足够样本才评估错误率/缓存（避免冷启动误报；最少 5 次）
        if recent_total >= 5:
            err_rate = health.get("recent_error_rate", 0.0)
            if err_rate > _ERROR_RATE_THRESHOLD:
                alerts.append(
                    Alert(
                        level="ERROR",
                        rule="error_rate",
                        message=f"最近错误率 {err_rate:.1%} 超过阈值 {_ERROR_RATE_THRESHOLD:.1%}",
                        value=round(err_rate, 4),
                    )
                )
            p95 = health.get("p95_ms", 0.0)
            if p95 > _P95_MS_THRESHOLD:
                alerts.append(
                    Alert(
                        level="WARN",
                        rule="p95_latency",
                        message=f"P95 延迟 {p95:.0f}ms 超过阈值 {_P95_MS_THRESHOLD_DISPLAY}ms",
                        value=p95,
                    )
                )
            cache_rate = health.get("cache_hit_rate", 1.0)
            if cache_rate < _CACHE_HIT_THRESHOLD:
                alerts.append(
                    Alert(
                        level="WARN",
                        rule="cache_hit_rate",
                        message=f"缓存命中率 {cache_rate:.1%} 低于阈值 {_CACHE_HIT_THRESHOLD:.0%}",
                        value=cache_rate,
                    )
                )

        # 外部服务连续失败（独立于样本量，立刻告警）
        for svc, n in (external_consecutive_failures or {}).items():
            if n >= _EXT_CONSEC_FAIL_THRESHOLD:
                alerts.append(
                    Alert(
                        level="ERROR",
                        rule="external_consecutive_fail",
                        message=f"{svc} 连续失败 {n} 次，超过阈值 {_EXT_CONSEC_FAIL_THRESHOLD}",
                        value=n,
                    )
                )
        return alerts

    def fire(self, alerts: list[Alert]) -> None:
        """告警落地：ERROR 日志 + 可选 webhook。无告警则静默。"""
        if not alerts:
            return
        for a in alerts:
            logger.log(
                logging.ERROR if a.level == "ERROR" else logging.WARNING,
                "ALERT rule=%s value=%s msg=%s",
                a.rule,
                a.value,
                a.message,
            )
        webhook = os.getenv("EASYSEARCH_ALERT_WEBHOOK", "").strip()
        if webhook:
            payload = json.dumps(
                {"alerts": [a.to_dict() for a in alerts]}, ensure_ascii=False
            ).encode("utf-8")
            try:
                req = urllib.request.Request(
                    url=webhook,
                    data=payload,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310 - 可配置 webhook
                    resp.read()
            except Exception as exc:  # noqa: BLE001 - webhook 失败不影响主链路
                logger.warning("alert webhook failed: %s", exc)
