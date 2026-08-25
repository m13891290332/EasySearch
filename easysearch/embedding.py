from __future__ import annotations

import hashlib
from collections import OrderedDict
from typing import Any

from .dashscope import DashScopeClient
from .utils import l2_normalize, local_hash_vector


class Qwen37TextEmbedding:
    """qwen3.7-text-embedding 适配器，支持批量向量化与离线 fallback。

    - 有 API Key：调用 DashScope text-embedding 接口（批量 texts），返回真实向量
    - 无 API Key / 调用失败：使用本地 hash 向量（local_hash_vector），保证可演示
    - 首次成功调用后记录真实维度，fallback 与之对齐，避免 cosine 维度不匹配

    M4：embed LRU(1024) 缓存（key=sha256(text)），重复文本不再重复调远程/计算；
        embed_batch 内部按缓存命中切分，仅对未命中部分批量调远程，兼顾缓存与批量效率。
    """

    model_name = "qwen3.7-text-embedding"
    endpoint = (
        "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"
    )

    def __init__(self, dashscope_client: DashScopeClient, dim: int = 1024) -> None:
        self.client = dashscope_client
        self.dim = dim
        self._actual_dim: int | None = None
        # M4：LRU 缓存（OrderedDict，容量 1024）
        self._cache: "OrderedDict[str, list[float]]" = OrderedDict()
        self._cache_size = 1024

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def embed(self, text: str) -> list[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float] | None] = [None] * len(texts)
        # 命中缓存的部分直接复用，未命中的收集起来批量调远程
        uncached_texts: list[str] = []
        uncached_pos: list[int] = []
        for i, text in enumerate(texts):
            key = self._key(text)
            cached = self._cache.get(key)
            if cached is not None:
                self._cache.move_to_end(key)
                results[i] = cached
            else:
                uncached_texts.append(text)
                uncached_pos.append(i)
        if uncached_texts:
            vectors = self._embed_uncached(uncached_texts)
            for text, vec, pos in zip(uncached_texts, vectors, uncached_pos):
                key = self._key(text)
                self._cache[key] = vec
                self._cache.move_to_end(key)
                results[pos] = vec
                # LRU 淘汰
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
        return results  # type: ignore[return-value]

    def _embed_uncached(self, texts: list[str]) -> list[list[float]]:
        """对未命中缓存的 texts 做实际向量化（远程优先，失败走本地 hash）。"""
        if self.client.enabled:
            try:
                vectors = self._remote_batch(texts)
                if vectors:
                    self._actual_dim = len(vectors[0])
                    return [l2_normalize(v) for v in vectors]
            except RuntimeError:
                pass
        return [self._local_embed(text) for text in texts]

    def _remote_batch(self, texts: list[str]) -> list[list[float]]:
        payload = {
            "model": self.model_name,
            "input": {"texts": texts},
        }
        response = self.client.post_json(self.endpoint, payload)
        embeddings = response.get("output", {}).get("embeddings", [])
        vectors: list[list[float]] = []
        for item in embeddings:
            vector = item.get("embedding") if isinstance(item, dict) else item
            if isinstance(vector, list) and vector:
                vectors.append([float(x) for x in vector])
        if len(vectors) != len(texts):
            raise RuntimeError("embedding count mismatch with input texts")
        return vectors

    def _local_embed(self, text: str) -> list[float]:
        return local_hash_vector(text, self._actual_dim or self.dim)
