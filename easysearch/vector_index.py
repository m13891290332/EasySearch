"""B2 向量索引：FAISS IndexFlatIP 批量检索，无 faiss 时降级 Python 循环。

embedding.py 已对向量做 L2 归一，归一后内积 = cosine 相似度，
故选用 IndexFlatIP（内积，越大越相似），与原 utils.cosine_similarity 语义完全一致。

M4：build 后 save_npz 持久化（按 KB 内容 hash 命名），下次启动 load_npz 命中即跳过 embedding。
"""
from __future__ import annotations

import logging
import os
from typing import Any

from .utils import cosine_similarity, l2_normalize

# 懒加载 faiss：未安装时降级到 Python 循环
_faiss = None
_faiss_loaded = False
_HAS_FAISS: bool | None = None


def _ensure_faiss() -> bool:
    """懒加载 faiss，返回是否可用。"""
    global _faiss, _faiss_loaded, _HAS_FAISS
    if _faiss_loaded:
        return bool(_HAS_FAISS)
    _faiss_loaded = True
    try:
        import faiss  # noqa: WPS433

        _faiss = faiss
        _HAS_FAISS = True
    except ImportError:
        _faiss = None
        _HAS_FAISS = False
        logging.warning("faiss unavailable, falling back to Python loop for VectorIndex")
    return bool(_HAS_FAISS)


class VectorIndex:
    """向量检索索引：优先 FAISS IndexFlatIP，降级 Python dict + cosine_similarity。

    对外接口：
        build(items: dict[str, list[float]])  构建索引
        score_all(query) -> dict[str, float]  一次性返回所有 id 的相似度
        search(query, top_k) -> list[(id, score)]  返回 top_k
        get(id) -> list[float] | None  反查原始向量（兼容旧 dict 访问）
        ids() -> list[str]  全部 id
        __len__ / __contains__ / __iter__
    """

    def __init__(self) -> None:
        self._ids: list[str] = []                  # 行号 -> service_id
        self._id_to_idx: dict[str, int] = {}        # service_id -> 行号
        self._vectors: dict[str, list[float]] = {} # 降级路径用
        self._matrix: Any = None                   # FAISS / numpy 矩阵（faiss 可用时）
        self._index: Any = None                     # faiss.IndexFlatIP
        self._dim: int | None = None

    # ---------- 构建 ----------
    def build(self, items: dict[str, list[float]]) -> None:
        self._ids = []
        self._id_to_idx = {}
        self._vectors = {}
        self._matrix = None
        self._index = None
        self._dim = None
        if not items:
            return

        for service_id, vec in items.items():
            self._vectors[service_id] = list(vec)
            self._id_to_idx[service_id] = len(self._ids)
            self._ids.append(service_id)
            if self._dim is None:
                self._dim = len(vec)

        if _ensure_faiss() and self._dim is not None:
            self._build_faiss()

    def _build_faiss(self) -> None:
        """构建 FAISS IndexFlatIP。"""
        import numpy as np  # faiss 必依赖 numpy

        # 归一化（双重保险：embedding 已归一，但 build 入参可能未归一时兜底）
        raw = [self._vectors[sid] for sid in self._ids]
        normalized = [l2_normalize(v) if v else v for v in raw]
        matrix = np.array(normalized, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            return
        self._matrix = matrix
        self._index = _faiss.IndexFlatIP(self._dim)
        self._index.add(matrix)

    # ---------- M4 持久化 ----------
    def save_npz(self, path: str) -> None:
        """把当前索引的 ids + 向量矩阵存为 .npz，重启后 load_npz 命中即跳过 embedding。

        numpy 不可用或索引为空时静默跳过（不影响主链路）。
        """
        if not self._ids:
            return
        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            return
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
        raw = [self._vectors[sid] for sid in self._ids]
        normalized = [l2_normalize(v) if v else v for v in raw]
        matrix = np.array(normalized, dtype=np.float32)
        ids_arr = np.array(self._ids)
        np.savez(path, ids=ids_arr, matrix=matrix)

    def load_npz(self, path: str) -> bool:
        """从 .npz 恢复索引。文件不存在 / numpy 不可用 / 解析失败均返回 False。"""
        if not path or not os.path.exists(path):
            return False
        try:
            import numpy as np
        except ImportError:  # pragma: no cover
            return False
        try:
            data = np.load(path, allow_pickle=False)
            ids_arr = data["ids"]
            matrix = data["matrix"]
        except Exception:
            return False
        if matrix.ndim != 2 or matrix.shape[0] == 0:
            return False
        self._ids = [str(s) for s in ids_arr]
        self._id_to_idx = {sid: i for i, sid in enumerate(self._ids)}
        self._vectors = {
            sid: [float(x) for x in matrix[i]] for i, sid in enumerate(self._ids)
        }
        self._dim = int(matrix.shape[1])
        self._matrix = None
        self._index = None
        if _ensure_faiss():
            self._build_faiss()
        return True

    # ---------- 查询 ----------
    def score_all(self, query: list[float]) -> dict[str, float]:
        """返回 {service_id: similarity}，对全部已索引向量。"""
        if not self._ids:
            return {}
        if self._index is not None and _ensure_faiss():
            return self._score_all_faiss(query)
        # 降级路径
        q = l2_normalize(list(query)) if query else query
        return {sid: max(0.0, cosine_similarity(q, self._vectors[sid])) for sid in self._ids}

    def _score_all_faiss(self, query: list[float]) -> dict[str, float]:
        import numpy as np

        q = l2_normalize(list(query)) if query else query
        q_arr = np.array([q], dtype=np.float32)
        scores, indices = self._index.search(q_arr, len(self._ids))
        result: dict[str, float] = {}
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self._ids):
                continue
            result[self._ids[idx]] = float(max(0.0, score))
        return result

    def search(
        self, query: list[float], top_k: int | None = None
    ) -> list[tuple[str, float]]:
        """返回 [(service_id, score)]，按 score 降序。"""
        scored = self.score_all(query)
        ordered = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)
        return ordered[:top_k] if top_k is not None else ordered

    # ---------- 反查 / 容器协议 ----------
    def get(self, service_id: str) -> list[float] | None:
        """反查原始向量，兼容旧 dict 访问。"""
        return self._vectors.get(service_id)

    def ids(self) -> list[str]:
        return list(self._ids)

    def __len__(self) -> int:
        return len(self._ids)

    def __contains__(self, service_id: object) -> bool:
        return service_id in self._id_to_idx

    def __iter__(self):
        return iter(self._ids)

    @property
    def uses_faiss(self) -> bool:
        """是否实际走 FAISS 路径（便于诊断/日志）。"""
        return self._index is not None and _ensure_faiss()
