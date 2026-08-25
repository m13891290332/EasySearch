from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

# 至少含一个中文或单词字符的 token 才保留（过滤标点/空白）
_KEEP_PATTERN = re.compile(r"[\w\u4e00-\u9fff]", re.UNICODE)

# 懒加载 jieba：未安装时回退到正则单字/词切分
_jieba = None
_jieba_loaded = False


def _ensure_jieba() -> None:
    global _jieba, _jieba_loaded
    if _jieba_loaded:
        return
    _jieba_loaded = True
    try:
        import jieba  # noqa: WPS433

        _jieba = jieba
    except Exception:  # pragma: no cover - jieba 未安装时降级
        _jieba = None


def tokenize(text: str) -> list[str]:
    """中文用 jieba 分词，英文/数字按 \\w 切分；过滤纯标点空白。"""
    if not text:
        return []
    _ensure_jieba()
    if _jieba is not None:
        tokens = [t for t in _jieba.cut(text) if t.strip() and _KEEP_PATTERN.search(t)]
    else:
        tokens = re.findall(r"[\w\u4e00-\u9fff]+", text, flags=re.UNICODE)
    return [t.lower() for t in tokens]


def safe_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def l2_normalize(vector: list[float]) -> list[float]:
    norm = safe_norm(vector)
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = safe_norm(left)
    right_norm = safe_norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot = sum(l * r for l, r in zip(left, right))
    return dot / (left_norm * right_norm)


def normalize_scores(
    raw: dict[str, float], mode: str = "minmax"
) -> dict[str, float]:
    """分数归一化；默认 min-max（向后兼容）。

    - ``minmax``：(v-min)/(max-min)，输出 [0,1]；全相等时置 0。
    - ``rank``：按值降序排名映射到 (n-rank)/(n-1) ∈ [0,1]，抗极值；n<=1 时置 0。
    - ``zscore``：(v-mean)/std；std=0 时置 0（输出未裁剪，调用方需按场景裁剪）。
    """
    if not raw:
        return {}
    if mode == "rank":
        items = sorted(raw.items(), key=lambda kv: kv[1], reverse=True)
        n = len(items)
        if n <= 1:
            return {k: 0.0 for k in raw}
        denom = n - 1
        result: dict[str, float] = {}
        for rank, (k, _v) in enumerate(items):
            result[k] = (n - 1 - rank) / denom
        return result
    if mode == "zscore":
        values = list(raw.values())
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        std = math.sqrt(var)
        if math.isclose(std, 0.0):
            return {k: 0.0 for k in raw}
        return {k: (v - mean) / std for k, v in raw.items()}
    # 默认 minmax
    max_value = max(raw.values())
    min_value = min(raw.values())
    if math.isclose(max_value, min_value):
        return {key: 0.0 for key in raw}
    denom = max_value - min_value
    return {key: (value - min_value) / denom for key, value in raw.items()}


def extract_json(text: str) -> Any:
    """从模型输出中抽取 JSON 数组/对象，兼容代码块包裹。"""
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:].strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("[")
        end = stripped.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(stripped[start : end + 1])
            except json.JSONDecodeError:
                return None
        return None


def local_hash_vector(text: str, dim: int) -> list[float]:
    """离线 fallback 向量：基于 token 的 sha256 哈希累加后 L2 归一。"""
    vector = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for i in range(dim):
            vector[i] += (digest[i % len(digest)] / 255.0) - 0.5
    return l2_normalize(vector)
