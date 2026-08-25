from __future__ import annotations

from .utils import cosine_similarity, l2_normalize


class DINHistoryOptimizer:
    """DIN 风格的用户历史序列注意力，用于优化 query 向量。

    当用户历史查询数 > 10 时触发：
    对每条历史 query 向量计算与当前 query 的相关性（cosine，截断到 >=0），
    叠加近因权重（越新权重越大），加权求和后与当前 query 做 0.8/0.2 混合再归一。
    """

    @staticmethod
    def optimize(
        query_embedding: list[float],
        history_embeddings: list[list[float]],
        relevance_weight: float = 0.7,
        recency_weight: float = 0.3,
        mix_ratio: float = 0.8,
    ) -> list[float]:
        if not history_embeddings:
            return query_embedding
        weighted_sum = [0.0] * len(query_embedding)
        total_weight = 0.0
        # 倒序遍历：最新历史 index=1（权重最大）
        for index, embedding in enumerate(reversed(history_embeddings), start=1):
            if len(embedding) != len(query_embedding):
                continue
            relevance = max(0.0, cosine_similarity(query_embedding, embedding))
            recency = 1.0 / index
            weight = relevance_weight * relevance + recency_weight * recency
            total_weight += weight
            for i, value in enumerate(embedding):
                weighted_sum[i] += value * weight
        if total_weight == 0:
            return query_embedding
        history_vector = [weighted_sum[i] / total_weight for i in range(len(query_embedding))]
        mixed = [
            mix_ratio * query_embedding[i] + (1 - mix_ratio) * history_vector[i]
            for i in range(len(query_embedding))
        ]
        return l2_normalize(mixed)
