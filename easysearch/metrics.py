"""M10 监控告警：指标采集 + 滚动事件缓冲 + Prometheus 导出（可选降级）。

设计：
- 进程内 ``MetricsCollector`` 单例是唯一真实源：计数器 + 滚动事件缓冲（供
  /api/health「最近100次」与告警规则评估）。
- prometheus_client 可用时，同步镜像到 Counter/Histogram/Gauge，/metrics 走标准
  ``generate_latest``；不可用时 /metrics 用进程内状态手写 Prometheus exposition 文本，
  抓取方仍可解析。遵循项目「软依赖」约定（faiss/numpy/jieba 同）。
- 放在 easysearch/ 而非 api/：engine 需直接上报（search 事件 + 外部调用回调），
  api/ 层仅负责暴露端点。避免 easysearch→api 的逆向依赖。
- 线程安全：单 worker + TestClient 同步调用下加锁即可。
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any

# 软依赖 prometheus_client
_PROM = None
_PROM_LOADED = False


def _ensure_prometheus() -> bool:
    global _PROM, _PROM_LOADED
    if _PROM_LOADED:
        return _PROM is not None
    _PROM_LOADED = True
    try:
        import prometheus_client as pc  # noqa: WPS433

        _PROM = pc
    except ImportError:
        _PROM = None
    return _PROM is not None


class MetricsCollector:
    """进程内指标采集器 + Prometheus 镜像（可选）。

    线程不安全的并发场景由 ``_lock`` 保护；uvicorn 单 worker 下足够。
    """

    _instance: "MetricsCollector | None" = None

    def __new__(cls) -> "MetricsCollector":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init()
        return cls._instance

    def _init(self) -> None:
        self._lock = threading.Lock()
        # 搜索事件滚动缓冲（最近 100 条，供 /api/health 与告警评估）
        self._events: deque[dict[str, Any]] = deque(maxlen=100)
        # M14：实时大盘缓冲（最近 600 条，60s 滚动窗口聚合，避免 Prometheus
        # scrape 间隔盲区；与 _events 分离以保 M10 health_summary 语义不变）
        self._realtime_events: deque[dict[str, Any]] = deque(maxlen=600)
        # 聚合计数器
        self._search_total = 0
        self._search_errors = 0
        self._cache_hits = 0
        self._cache_misses = 0
        # 外部调用：{service: {"ok": n, "fail": n, "total_ms": float, "consecutive_fail": int}}
        self._external: dict[str, dict[str, Any]] = {}
        # KB embedding 进行中标记（Gauge 语义；本期同步导入恒 0）
        self._kb_embedding_in_progress = 0
        self._prom = self._build_prometheus()  # None或 dict[str, Any]

    def _build_prometheus(self) -> Any:
        """prometheus_client 可用时创建 Counter/Histogram/Gauge；否则 None。"""
        if not _ensure_prometheus():
            return None
        pc = _PROM
        try:
            return {
                "search_total": pc.Counter(
                    "easysearch_search_total", "Total search requests"
                ),
                "search_errors": pc.Counter(
                    "easysearch_search_errors_total", "Failed search requests"
                ),
                "cache_hits": pc.Counter(
                    "easysearch_cache_hits_total", "Cache hit count"
                ),
                "cache_misses": pc.Counter(
                    "easysearch_cache_misses_total", "Cache miss count"
                ),
                "latency": pc.Histogram(
                    "easysearch_search_latency_seconds",
                    "Search latency by stage (seconds)",
                    ["stage"],
                    buckets=(0.005, 0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0),
                ),
                "external": pc.Counter(
                    "easysearch_external_call_total",
                    "External LLM call count",
                    ["service", "status"],
                ),
                "external_latency": pc.Histogram(
                    "easysearch_external_call_latency_seconds",
                    "External call latency (seconds)",
                    ["service"],
                    buckets=(0.05, 0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0),
                ),
                "kb_embedding": pc.Gauge(
                    "easysearch_kb_embedding_in_progress",
                    "1 if KB embedding rebuild in progress",
                ),
                "db_pool": pc.Gauge(
                    "easysearch_db_pool_usage",
                    "DB connection pool usage (placeholder)",
                ),
            }
        except Exception:  # pragma: no cover - 重复注册等异常降级
            return None

    # ---------- 搜索事件 ----------
    def record_search(
        self,
        total_ms: float,
        stages: dict[str, float] | None = None,
        cache_hit: bool = False,
        degraded: bool = False,
        error: bool = False,
        intent: str = "",
    ) -> None:
        """记录一次搜索事件：落滚动缓冲 + 更新计数器 + 镜像 Prometheus。"""
        event = {
            "ts": time.time(),
            "total_ms": float(total_ms),
            "stages": stages or {},
            "cache_hit": bool(cache_hit),
            "degraded": bool(degraded),
            "error": bool(error),
            "intent": intent,
        }
        with self._lock:
            self._events.append(event)
            self._realtime_events.append(event)
            self._search_total += 1
            if error:
                self._search_errors += 1
            if cache_hit:
                self._cache_hits += 1
            else:
                self._cache_misses += 1
        if self._prom:
            try:
                self._prom["search_total"].inc()
                if error:
                    self._prom["search_errors"].inc()
                if cache_hit:
                    self._prom["cache_hits"].inc()
                else:
                    self._prom["cache_misses"].inc()
                if stages:
                    for stage, ms in stages.items():
                        self._prom["latency"].labels(stage=stage).observe(
                            ms / 1000.0
                        )
            except Exception:  # pragma: no cover
                pass

    # ---------- 外部调用 ----------
    def record_external(
        self, service: str, ok: bool, latency_ms: float
    ) -> None:
        """记录一次外部 LLM 调用（dashscope/deepseek），含连续失败计数。"""
        with self._lock:
            stats = self._external.setdefault(
                service,
                {"ok": 0, "fail": 0, "total_ms": 0.0, "consecutive_fail": 0},
            )
            if ok:
                stats["ok"] += 1
                stats["consecutive_fail"] = 0
            else:
                stats["fail"] += 1
                stats["consecutive_fail"] += 1
            stats["total_ms"] += float(latency_ms)
        if self._prom:
            try:
                self._prom["external"].labels(
                    service=service, status="ok" if ok else "fail"
                ).inc()
                self._prom["external_latency"].labels(service=service).observe(
                    latency_ms / 1000.0
                )
            except Exception:  # pragma: no cover
                pass

    # ---------- Gauge ----------
    def set_kb_embedding_in_progress(self, value: bool) -> None:
        with self._lock:
            self._kb_embedding_in_progress = 1 if value else 0
        if self._prom:
            try:
                self._prom["kb_embedding"].set(self._kb_embedding_in_progress)
            except Exception:  # pragma: no cover
                pass

    # ---------- 查询 ----------
    @staticmethod
    def _percentile(sorted_vals: list[float], p: float) -> float:
        if not sorted_vals:
            return 0.0
        idx = int(len(sorted_vals) * p / 100.0)
        if idx >= len(sorted_vals):
            idx = len(sorted_vals) - 1
        return sorted_vals[idx]

    @staticmethod
    def _percentiles(sorted_vals: list[float]) -> dict[str, float]:
        """返回 P50/P95/P99（输入需已排序）。空列表返回全 0。"""
        if not sorted_vals:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        return {
            "p50": MetricsCollector._percentile(sorted_vals, 50),
            "p95": MetricsCollector._percentile(sorted_vals, 95),
            "p99": MetricsCollector._percentile(sorted_vals, 99),
        }

    @staticmethod
    def _stage_percentiles(events: list[dict[str, Any]], stage: str) -> dict[str, float]:
        """对指定 stage 跨事件取 P50/P95/P99（毫秒）。"""
        vals = sorted(
            e["stages"].get(stage, 0.0)
            for e in events
            if e.get("stages")
        )
        return MetricsCollector._percentiles(vals)

    def health_summary(self) -> dict[str, Any]:
        """供 /api/health：最近100次成功率/P95/缓存命中率 + 外部调用健康度。"""
        with self._lock:
            events = list(self._events)
            total = self._search_total
            errors = self._search_errors
            hits = self._cache_hits
            misses = self._cache_misses
            external = {k: dict(v) for k, v in self._external.items()}
            in_progress = self._kb_embedding_in_progress
        latencies = sorted(e["total_ms"] for e in events)
        p95 = self._percentile(latencies, 95)
        recent_total = len(events)
        recent_errors = sum(1 for e in events if e["error"])
        recent_degraded = sum(1 for e in events if e["degraded"])
        cache_rate = (hits / (hits + misses)) if (hits + misses) else 0.0
        external_summary = {
            svc: {
                "success_rate": (
                    s["ok"] / (s["ok"] + s["fail"])
                    if (s["ok"] + s["fail"])
                    else 1.0
                ),
                "consecutive_fail": int(s["consecutive_fail"]),
                "total_calls": int(s["ok"] + s["fail"]),
            }
            for svc, s in external.items()
        }
        return {
            "search_total": total,
            "search_errors": errors,
            "error_rate": (errors / total) if total else 0.0,
            "recent_total": recent_total,
            "recent_error_rate": (recent_errors / recent_total)
            if recent_total
            else 0.0,
            "recent_degraded": recent_degraded,
            "p95_ms": round(p95, 2),
            "cache_hit_rate": cache_rate,
            "external": external_summary,
            "kb_embedding_in_progress": bool(in_progress),
        }

    def events(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._events)

    def realtime_summary(self, window_seconds: int = 60) -> dict[str, Any]:
        """M14：实时大盘——最近 ``window_seconds`` 秒滚动窗口聚合。

        返回 QPS / 错误率 / 缓存命中率 / 降级计数 / 各阶段 P50/P95/P99 +
        DB 池占用（占位 0，无连接池 gauge）+ embedding 是否进行中。
        窗口内无事件时返回全 0 指标（冷启动不报错）。
        """
        now = time.time()
        cutoff = now - float(window_seconds)
        with self._lock:
            events = [e for e in self._realtime_events if e["ts"] >= cutoff]
            in_progress = self._kb_embedding_in_progress
            external = {k: dict(v) for k, v in self._external.items()}
        n = len(events)
        errors = sum(1 for e in events if e["error"])
        degraded = sum(1 for e in events if e["degraded"])
        cache_hits = sum(1 for e in events if e["cache_hit"])
        total_latencies = sorted(e["total_ms"] for e in events)
        # 汇总各出现过的 stage（取并集，缺失项该事件计 0）
        stage_names: set[str] = set()
        for e in events:
            stage_names.update(e.get("stages", {}).keys())
        stage_latency = {
            stage: self._stage_percentiles(events, stage)
            for stage in sorted(stage_names)
        }
        qps = (n / float(window_seconds)) if window_seconds > 0 else 0.0
        external_rt = {
            svc: {
                "success_rate": (
                    s["ok"] / (s["ok"] + s["fail"])
                    if (s["ok"] + s["fail"])
                    else 1.0
                ),
                "consecutive_fail": int(s["consecutive_fail"]),
                "total_calls": int(s["ok"] + s["fail"]),
            }
            for svc, s in external.items()
        }
        return {
            "window_seconds": window_seconds,
            "total_requests": n,
            "qps": round(qps, 4),
            "error_rate": (errors / n) if n else 0.0,
            "cache_hit_rate": (cache_hits / n) if n else 0.0,
            "degraded_count": degraded,
            "latency_total": self._percentiles(total_latencies),
            "latency_stages": stage_latency,
            "external": external_rt,
            "kb_embedding_in_progress": bool(in_progress),
            "db_pool_usage": 0,  # 占位：本期无连接池 gauge，预留扩展点
        }

    def external_stats(self) -> dict[str, dict[str, Any]]:
        with self._lock:
            return {k: dict(v) for k, v in self._external.items()}

    def consecutive_failures(self, service: str) -> int:
        with self._lock:
            stats = self._external.get(service)
            return int(stats["consecutive_fail"]) if stats else 0

    # ---------- Prometheus 导出 ----------
    def prometheus_text(self) -> str:
        """prometheus_client 可用走 generate_latest；否则手写 exposition 文本。"""
        if self._prom and _ensure_prometheus():
            try:
                return _PROM.generate_latest().decode("utf-8")
            except Exception:  # pragma: no cover
                pass
        return self._manual_exposition()

    def _manual_exposition(self) -> str:
        """无 prometheus_client 时手写 Prometheus exposition 格式（可被抓取解析）。"""
        summary = self.health_summary()
        lines: list[str] = []
        lines.append("# HELP easysearch_search_total Total search requests")
        lines.append("# TYPE easysearch_search_total counter")
        lines.append(f"easysearch_search_total {summary['search_total']}")
        lines.append("# HELP easysearch_search_errors_total Failed search requests")
        lines.append("# TYPE easysearch_search_errors_total counter")
        lines.append(f"easysearch_search_errors_total {summary['search_errors']}")
        lines.append("# HELP easysearch_cache_hits_total Cache hit count")
        lines.append("# TYPE easysearch_cache_hits_total counter")
        lines.append(f"easysearch_cache_hits_total {self._cache_hits}")
        lines.append("# HELP easysearch_cache_misses_total Cache miss count")
        lines.append("# TYPE easysearch_cache_misses_total counter")
        lines.append(f"easysearch_cache_misses_total {self._cache_misses}")
        lines.append("# HELP easysearch_search_latency_seconds Search latency p95 (seconds)")
        lines.append("# TYPE easysearch_search_latency_seconds summary")
        lines.append(
            f'easysearch_search_latency_seconds{{quantile="0.95"}} '
            f"{summary['p95_ms'] / 1000.0:.6f}"
        )
        lines.append("# HELP easysearch_external_call_total External LLM call count")
        lines.append("# TYPE easysearch_external_call_total counter")
        with self._lock:
            external = {k: dict(v) for k, v in self._external.items()}
        for svc, s in external.items():
            lines.append(
                f'easysearch_external_call_total{{service="{svc}",status="ok"}} {s["ok"]}'
            )
            lines.append(
                f'easysearch_external_call_total{{service="{svc}",status="fail"}} {s["fail"]}'
            )
        lines.append("# HELP easysearch_kb_embedding_in_progress KB embedding rebuild in progress")
        lines.append("# TYPE easysearch_kb_embedding_in_progress gauge")
        lines.append(
            f"easysearch_kb_embedding_in_progress {1 if summary['kb_embedding_in_progress'] else 0}"
        )
        return "\n".join(lines) + "\n"

    # ---------- 测试辅助 ----------
    def reset(self) -> None:
        """清空进程内状态（测试隔离用）。Prometheus 注册对象不可重置，仅清进程内。"""
        with self._lock:
            self._events.clear()
            self._realtime_events.clear()
            self._search_total = 0
            self._search_errors = 0
            self._cache_hits = 0
            self._cache_misses = 0
            self._external.clear()
            self._kb_embedding_in_progress = 0


def get_metrics() -> MetricsCollector:
    """获取全局 MetricsCollector 单例。"""
    return MetricsCollector()
