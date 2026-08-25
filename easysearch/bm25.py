from __future__ import annotations

import heapq
import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

from .config import BM25_B, BM25_FIELD_WEIGHTS, BM25_K1
from .utils import tokenize

# M3：懒加载 numpy，未安装时降级到纯 Python 全表遍历（与 vector_index.py 一致）
_HAS_NUMPY: bool | None = None


def _ensure_numpy() -> bool:
    """返回 numpy 是否可用（首次调用时检测，结果缓存）。"""
    global _HAS_NUMPY
    if _HAS_NUMPY is not None:
        return _HAS_NUMPY
    try:
        import numpy  # noqa: F401  仅探测可用性

        _HAS_NUMPY = True
    except ImportError:  # pragma: no cover - numpy 是推荐依赖
        _HAS_NUMPY = False
    return _HAS_NUMPY


class BM25Index:
    """BM25 倒排索引（中文 jieba 分词，utils.tokenize）。

    标准 BM25 公式：
        score(q,d) = Σ_t idf(t) · (f(t,d)·(k1+1)) / (f(t,d) + k1·(1-b+b·|d|/avgdl))

    兼容包装：内部委托给 MultiFieldBM25Index 的单字段实例（field="text"，权重 1.0）。
    旧 build/score/search 签名不变，保 verify.py 与旧测试通过。
    """

    def __init__(self, k1: float = BM25_K1, b: float = BM25_B) -> None:
        self.k1 = k1
        self.b = b
        self._inner: MultiFieldBM25Index = MultiFieldBM25Index(
            k1=k1, b=b, field_weights={"text": 1.0}
        )

    # 旧单字段接口：docs[doc_id] = text
    def build(self, docs: dict[str, str]) -> None:
        multi_docs = {doc_id: {"text": text} for doc_id, text in docs.items()}
        self._inner.build(multi_docs)

    def score(self, query: str, doc_id: str) -> float:
        return self._inner.score(query, doc_id)

    def search(self, query: str, top_k: int | None = None) -> dict[str, float]:
        scores = self._inner.batch_score(query)
        if top_k is not None:
            ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            return dict(ordered)
        return scores

    # 暴露多字段能力（engine 用）
    def batch_score_tokens(self, query_tokens: list[str]) -> dict[str, float]:
        return self._inner.batch_score_tokens(query_tokens)

    def vocabulary(self) -> set[str]:
        return self._inner.vocabulary()


@dataclass
class _FieldStats:
    """单字段的 BM25 统计量。

    兼容层：doc_freq / doc_lengths / term_freqs / avg_doc_len / num_docs 保留为
        dict/Counter 结构，供 _term_field_score / score_tokens / vocabulary
        逐 doc 查询使用（行为与重写前逐字节一致）。
    M3 增量：postings（term -> (doc_idx np.array, tf np.array)）+ doc_lengths_np
        对齐到全局 doc 顺序，供 batch_score_tokens 的 numpy 批算路径使用，
        避免全表 N×T 的 Python 循环。
    """

    doc_freq: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    doc_lengths: dict[str, int] = field(default_factory=dict)
    term_freqs: dict[str, Counter[str]] = field(default_factory=dict)
    avg_doc_len: float = 0.0
    num_docs: int = 0
    # M3 倒排表（numpy 路径用；numpy 不可用时保持空，batch 自动降级）
    postings: dict[str, tuple[Any, Any]] = field(default_factory=dict)
    doc_lengths_np: Any = None  # np.ndarray shape [N]，按全局 doc 顺序


class MultiFieldBM25Index:
    """多字段 BM25：每字段独立统计，打分按字段权重加权融合。

    A2：name=3.0 / aliases=2.0 / intro=1.0 / route=1.5（来自 config.BM25_FIELD_WEIGHTS）
    C3：batch_score_tokens 一次遍历所有 doc，避免 score() 被调 N 次
    M3：build 产出每字段每 term 的 posting list（doc_idx + tf 数组），
        batch_score_tokens 改走 numpy 批算：对 query token 取并集 doc_idx，
        向量化计算 idf 与 tf 项，按字段权重加和；不再遍历全表 N×T。
        单查询 @10K 目标 < 5ms。numpy 不可用时自动降级到原 Python 全表实现。

    打分公式：
        score(q,d) = Σ_field w_field · Σ_t idf(t,field) · tf_term(t,d,field)
    其中 tf_term(t,d,field) = idf · (f·(k1+1)) / (f + k1·(1-b+b·|d_field|/avgdl_field))，
    query 中同一 term 重复出现 qtf 次则按 qtf 倍计（与重写前一致）。
    """

    def __init__(
        self,
        k1: float = BM25_K1,
        b: float = BM25_B,
        field_weights: dict[str, float] | None = None,
    ) -> None:
        self.k1 = k1
        self.b = b
        self.field_weights: dict[str, float] = dict(field_weights or BM25_FIELD_WEIGHTS)
        self.fields: dict[str, _FieldStats] = {}
        self.doc_ids: list[str] = []
        # M3：全局 doc_id -> 行号（所有字段共享同一顺序）
        self._id_to_idx: dict[str, int] = {}

    def build(self, docs: dict[str, dict[str, str]]) -> None:
        """docs[doc_id] = {"name": ..., "aliases": ..., "intro": ..., "route": ...}

        缺失字段视为空字符串。仅对 field_weights 中列出的字段建索引。
        同时填充 dict 兼容结构（_term_field_score/score/vocabulary 用）与
        M3 posting 数组（batch_score_tokens numpy 路径用）。
        """
        self.fields = {fname: _FieldStats() for fname in self.field_weights}
        self.doc_ids = list(docs.keys())
        self._id_to_idx = {sid: i for i, sid in enumerate(self.doc_ids)}
        n_docs = len(self.doc_ids)

        for fname, stats in self.fields.items():
            total_len = 0
            for doc_id, fields in docs.items():
                text = str(fields.get(fname, "") or "")
                tokens = tokenize(text)
                counts = Counter(tokens)
                stats.term_freqs[doc_id] = counts
                stats.doc_lengths[doc_id] = len(tokens)
                total_len += len(tokens)
                for term in counts:
                    stats.doc_freq[term] += 1
            stats.num_docs = n_docs
            stats.avg_doc_len = (total_len / n_docs) if n_docs else 0.0

            # M3：构建 posting list + doc_lengths_np（numpy 不可用时跳过，batch 自动降级）
            if _ensure_numpy() and n_docs:
                self._build_field_postings(stats)

    def _build_field_postings(self, stats: _FieldStats) -> None:
        """为单字段构建倒排 posting list（numpy 数组）。

        postings[term] = (doc_idx: np.ndarray[int64], tf: np.ndarray[float64])，
        doc_idx 对齐到全局 self.doc_ids 顺序；doc_lengths_np 同样按全局顺序对齐。
        """
        import numpy as np

        # doc_lengths 按全局顺序对齐（缺失视为 0）
        stats.doc_lengths_np = np.array(
            [stats.doc_lengths.get(sid, 0) for sid in self.doc_ids],
            dtype=np.float64,
        )

        # 汇总 term -> [(idx, tf), ...]，再转为紧凑 numpy 数组
        bucket: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for sid in self.doc_ids:
            counts = stats.term_freqs.get(sid)
            if not counts:
                continue
            idx = self._id_to_idx[sid]
            for term, tf in counts.items():
                bucket[term].append((idx, tf))

        postings: dict[str, tuple[Any, Any]] = {}
        for term, pairs in bucket.items():
            pairs.sort(key=lambda p: p[0])  # 按 doc_idx 升序（便于可读/调试，非必需）
            idxs = np.fromiter((p[0] for p in pairs), dtype=np.int64, count=len(pairs))
            tfs = np.fromiter(
                (float(p[1]) for p in pairs), dtype=np.float64, count=len(pairs)
            )
            postings[term] = (idxs, tfs)
        stats.postings = postings

    # ---------- 单 token 的字段级 BM25（dict 路径，逐 doc 查询用） ----------
    def _term_field_score(
        self, term: str, doc_id: str, fname: str, stats: _FieldStats
    ) -> float:
        freq = stats.term_freqs.get(doc_id, {}).get(term, 0)
        if freq == 0:
            return 0.0
        df = stats.doc_freq.get(term, 0)
        num_docs = max(1, stats.num_docs)
        idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
        doc_len = stats.doc_lengths.get(doc_id, 0)
        if stats.avg_doc_len == 0:
            return 0.0
        denom = freq + self.k1 * (1 - self.b + self.b * doc_len / stats.avg_doc_len)
        return idf * (freq * (self.k1 + 1)) / denom

    # ---------- 旧 score 接口（内部 tokenize 一次） ----------
    def score(self, query: str, doc_id: str) -> float:
        return self.score_tokens(tokenize(query), doc_id)

    def score_tokens(self, query_tokens: list[str], doc_id: str) -> float:
        score = 0.0
        for fname, stats in self.fields.items():
            w = self.field_weights.get(fname, 1.0)
            if w == 0:
                continue
            field_score = 0.0
            for term in query_tokens:
                field_score += self._term_field_score(term, doc_id, fname, stats)
            score += w * field_score
        return score

    # ---------- C3 + M3：一次遍历全 doc（numpy 优先，降级 Python） ----------
    def batch_score_tokens(self, query_tokens: list[str]) -> dict[str, float]:
        """一次返回 {doc_id: score}（含 0.0 的未命中 doc，保 normalize_scores 行为一致）。

        M3：numpy 可用时走 posting list 批算（O(Σ_t |postings_t|) 向量化），
        不再遍历全表 N×T；numpy 不可用时降级到原 Python 全表实现。
        """
        if _ensure_numpy() and self.doc_ids:
            return self._batch_score_tokens_np(query_tokens)
        return self._batch_score_tokens_py(query_tokens)

    def _batch_score_tokens_np(self, query_tokens: list[str]) -> dict[str, float]:
        import numpy as np

        n = len(self.doc_ids)
        result: dict[str, float] = {sid: 0.0 for sid in self.doc_ids}
        if not query_tokens or not n:
            return result

        scores = np.zeros(n, dtype=np.float64)
        k1 = self.k1
        b = self.b
        # query term 频次：重复 token 按 qtf 倍计（与重写前一致）
        qcounts = Counter(query_tokens)

        for fname, stats in self.fields.items():
            w = self.field_weights.get(fname, 1.0)
            if w == 0 or stats.num_docs == 0 or stats.avg_doc_len == 0:
                continue
            n_field = stats.num_docs
            avgdl = stats.avg_doc_len
            doc_len_np = stats.doc_lengths_np
            postings = stats.postings
            if doc_len_np is None or not postings:
                continue
            for term, qtf in qcounts.items():
                posting = postings.get(term)
                if not posting:
                    continue
                idxs, tfs = posting
                df = idxs.shape[0]
                idf = math.log((n_field - df + 0.5) / (df + 0.5) + 1.0)
                dlen = doc_len_np[idxs]
                denom = tfs + k1 * (1.0 - b + b * dlen / avgdl)
                term_score = idf * (tfs * (k1 + 1.0)) / denom  # shape [P]
                # np.add.at：散列累加到 scores，对 idxs 唯一性无假设（稳健）；
                # 跨 term 顺序累加到同一 scores 数组
                np.add.at(scores, idxs, w * qtf * term_score)

        # 散列回 dict（保持与旧实现一致的 {doc_id: float} 全量返回）
        for i, sid in enumerate(self.doc_ids):
            result[sid] = float(scores[i])
        return result

    def _batch_score_tokens_py(self, query_tokens: list[str]) -> dict[str, float]:
        """numpy 不可用时的降级路径：原 Python 全表遍历实现。"""
        result: dict[str, float] = {doc_id: 0.0 for doc_id in self.doc_ids}
        if not query_tokens:
            return result
        for fname, stats in self.fields.items():
            w = self.field_weights.get(fname, 1.0)
            if w == 0:
                continue
            for doc_id in self.doc_ids:
                tf_doc = stats.term_freqs.get(doc_id)
                if not tf_doc:
                    continue
                field_score = 0.0
                for term in query_tokens:
                    freq = tf_doc.get(term, 0)
                    if freq == 0:
                        continue
                    df = stats.doc_freq.get(term, 0)
                    num_docs = max(1, stats.num_docs)
                    idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
                    doc_len = stats.doc_lengths.get(doc_id, 0)
                    if stats.avg_doc_len == 0:
                        continue
                    denom = freq + self.k1 * (
                        1 - self.b + self.b * doc_len / stats.avg_doc_len
                    )
                    field_score += idf * (freq * (self.k1 + 1)) / denom
                result[doc_id] += w * field_score
        return result

    def batch_score(self, query: str) -> dict[str, float]:
        """便捷封装：先 tokenize 一次再调 batch_score_tokens。"""
        return self.batch_score_tokens(tokenize(query))

    def search(self, query: str, top_k: int | None = None) -> dict[str, float]:
        scores = self.batch_score(query)
        if top_k is not None:
            ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
            return dict(ordered)
        return scores

    def vocabulary(self) -> set[str]:
        """A6 拼写纠错词表：合并所有字段的 term set。"""
        vocab: set[str] = set()
        for stats in self.fields.values():
            vocab.update(stats.doc_freq.keys())
        return vocab

    def __len__(self) -> int:
        return len(self.doc_ids)

    def __contains__(self, doc_id: object) -> bool:
        return any(doc_id in stats.term_freqs for stats in self.fields.values())
