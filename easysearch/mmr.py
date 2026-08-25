"""A5 MMR 多样性重排：Maximal Marginal Relevance。

公式：mmr_score(i) = lambda * rel(i) - (1-lambda) * max_{j in S} cos(emb_i, emb_j)
贪心选最高者入集合 S，从 top20 选 top10。

相似度用向量余弦（捕捉语义重复），lambda=0.85（用户确认，轻微多样性）。
"""
from __future__ import annotations

from typing import Any

from .utils import cosine_similarity


class MMRReranker:
    """MMR 多样性重排器。

    输入：已按 relevance 排序的候选 + 每个候选的 embedding。
    输出：按 MMR 选出的 top_k 子集。
    """

    def __init__(self, lambda_: float = 0.85) -> None:
        self.lambda_ = lambda_

    def select(
        self,
        candidates: list[dict[str, Any]],
        embeddings: dict[str, list[float]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """从 candidates（已按 relevance 降序）中按 MMR 选 top_k。

        要求 candidates 已按 rerank_score / score 降序排列。
        embeddings: {service_id: vector}，候选必须全部命中。
        """
        if not candidates:
            return []
        if len(candidates) <= top_k:
            return list(candidates)
        if self.lambda_ >= 1.0:
            # 完全关闭多样性，纯按相关性顺序
            return list(candidates[:top_k])

        # 取每个候选的 relevance（用 rerank_score 优先，回退到 score）
        rels: list[float] = []
        embs: list[list[float]] = []
        valid_idx: list[int] = []
        for i, cand in enumerate(candidates):
            sid = cand.get("service_id")
            if sid not in embeddings:
                continue
            rel = float(cand.get("rerank_score", cand.get("score", 0.0)))
            rels.append(rel)
            embs.append(embeddings[sid])
            valid_idx.append(i)

        if len(valid_idx) <= top_k:
            return [candidates[i] for i in valid_idx]

        # MMR 贪心选择
        selected_pos: list[int] = []
        remaining = set(range(len(valid_idx)))
        # 第一个：直接选 relevance 最高的（candidates 已按 rerank_score 降序，
        # valid_idx[0] 对应 candidates[0] 即 rerank 第 1 名）
        first = 0
        selected_pos.append(first)
        remaining.discard(first)

        while len(selected_pos) < top_k and remaining:
            best_pos: int | None = None
            best_score: float = float("-inf")
            for pos in remaining:
                # max_{j in S} cos(emb_i, emb_j)
                max_sim = 0.0
                for sel_pos in selected_pos:
                    sim = max(0.0, cosine_similarity(embs[pos], embs[sel_pos]))
                    if sim > max_sim:
                        max_sim = sim
                mmr = self.lambda_ * rels[pos] - (1.0 - self.lambda_) * max_sim
                if mmr > best_score:
                    best_score = mmr
                    best_pos = pos
            if best_pos is None:
                break
            selected_pos.append(best_pos)
            remaining.discard(best_pos)

        return [candidates[valid_idx[p]] for p in selected_pos]
