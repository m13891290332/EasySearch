#!/usr/bin/env python3
"""M3 BM25 倒排化基准：合成 10K KB，验证单查询 batch_score_tokens < 5ms。

用法：python bench_bm25.py
依赖：numpy（推荐）、jieba（推荐）；缺失时自动降级。
"""
from __future__ import annotations

import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from easysearch.bm25 import MultiFieldBM25Index, _ensure_numpy
from easysearch.utils import tokenize


def build_synthetic_kb(target: int = 10000) -> dict[str, dict[str, str]]:
    root = os.path.dirname(os.path.abspath(__file__))
    kb_path = os.path.join(root, "金融服务数据300条.json")
    if not os.path.exists(kb_path):
        kb_path = os.path.join(root, "services_dict_50.json")
    with open(kb_path, "r", encoding="utf-8") as fp:
        base = json.load(fp)
    if not isinstance(base, list):
        raise RuntimeError(f"{kb_path} 非数组结构")

    docs: dict[str, dict[str, str]] = {}
    r = 0
    while len(docs) < target:
        for i, item in enumerate(base):
            sid = f"svc-{r}-{i}"
            docs[sid] = {
                "name": str(item.get("service_name", "")),
                "aliases": " ".join(item.get("aliases") or []),
                "intro": str(item.get("service_intro", "")),
                "route": str(item.get("route", "") or ""),
            }
            if len(docs) >= target:
                break
        r += 1
    return docs


def main() -> int:
    has_numpy = _ensure_numpy()
    print(f"numpy: {'available' if has_numpy else 'UNAVAILABLE (will fall back to Python)'}")

    docs = build_synthetic_kb(10000)
    print(f"synthetic KB size: {len(docs)}")

    idx = MultiFieldBM25Index()
    t0 = time.perf_counter()
    idx.build(docs)
    build_ms = (time.perf_counter() - t0) * 1000.0
    print(f"build time: {build_ms:.2f} ms")
    print(f"vocabulary size: {len(idx.vocabulary())}")

    queries = ["开户", "订单管理", "银证转账", "风险控制", "用户画像", "transfer", "审批"]
    tokens_list = [tokenize(q) for q in queries]

    # 正确性：numpy 路径 vs Python 路径（仅在 numpy 可用时校验）
    if has_numpy:
        max_delta = 0.0
        for tokens in tokens_list:
            np_s = idx._batch_score_tokens_np(tokens)
            py_s = idx._batch_score_tokens_py(tokens)
            for k in np_s:
                max_delta = max(max_delta, abs(np_s[k] - py_s[k]))
        print(f"parity numpy-vs-python max delta: {max_delta:.2e} (expect < 1e-9)")

    # warmup
    for tokens in tokens_list:
        idx.batch_score_tokens(tokens)

    # 计时 batch_score_tokens（不含 tokenize，纯索引打分）
    samples: list[float] = []
    rounds = 50
    for _ in range(rounds):
        for tokens in tokens_list:
            if not tokens:
                continue
            t0 = time.perf_counter()
            idx.batch_score_tokens(tokens)
            samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()
    n = len(samples)
    p50 = samples[n // 2]
    p95 = samples[int(n * 0.95)]
    p99 = samples[int(n * 0.99)]
    worst = samples[-1]
    print(f"batch_score_tokens @10K ({n} samples):")
    print(f"  P50  = {p50:.3f} ms")
    print(f"  P95  = {p95:.3f} ms")
    print(f"  P99  = {p99:.3f} ms")
    print(f"  worst= {worst:.3f} ms")

    # 对比 Python 全表路径（量化 M3 收益）
    py_samples: list[float] = []
    for _ in range(5):
        for tokens in tokens_list:
            if not tokens:
                continue
            t0 = time.perf_counter()
            idx._batch_score_tokens_py(tokens)
            py_samples.append((time.perf_counter() - t0) * 1000.0)
    if py_samples:
        py_med = statistics.median(py_samples)
        print(f"python full-scan median (对比): {py_med:.3f} ms")
        if p50 > 0:
            print(f"speedup vs python: ~{py_med / p50:.1f}x")

    target_ms = 5.0
    ok = worst < target_ms
    print("=" * 50)
    if ok:
        print(f"✅ PASS: worst {worst:.3f}ms < {target_ms}ms 目标")
        return 0
    if p95 < target_ms:
        print(f"⚠️  worst {worst:.3f}ms 超 {target_ms}ms，但 P95 {p95:.3f}ms 达标")
        return 0
    print(f"❌ FAIL: worst {worst:.3f}ms / P95 {p95:.3f}ms 超过 {target_ms}ms 目标")
    return 1


if __name__ == "__main__":
    sys.exit(main())
