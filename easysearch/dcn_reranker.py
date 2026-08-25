"""DCN v2 保底重排器：qwen-rerank 不可用 / 失败时的降级重排。

设计要点
--------
- 复用项目既有的 embedding（query / 候选向量）与召回侧信号构造特征：
  ``service_id``(sparse) + ``cosine`` + ``hybrid`` + ``name_overlap`` +
  ``intro_overlap`` + ``popularity``(dense)，喂给 ``dcn_v2.DCN_V2`` 网络。
- 训练样本来自 ``store.click_query_pairs``（点击→点击前最近 query，label=1）+
  随机负采样（label=0），用 BCE 损失在线训练，权重按 kb_hash 落盘复用。
- 推理：torch 可用且已加载匹配 kb_hash 的权重 → 走 DCN_V2 forward；
  否则走「线性启发式」降级（同一组特征的加权线性组合，仍复用 embedding 的 cosine），
  保证无 torch / 未训练时也是比纯 token-overlap 更稳的保底。
- 模型与 service_id 词表强绑定（embedding 维度 = num_services），故权重文件按
  ``kb_hash`` 命名，加载时校验 kb_hash + service_ids 一致，否则视为未训练。
- 不设置 ``rerank_reason``：由调用方（``Qwen3VLReranker``）统一附模板理由。
"""
from __future__ import annotations

import json
import logging
import os
import random
from typing import Any

from .utils import cosine_similarity, tokenize

logger = logging.getLogger(__name__)

# 启发式降级的线性权重（cosine 复用 embedding，hybrid 已含 popularity 故不重复计）
_HEURISTIC_WEIGHTS = {
    "cosine": 0.5,
    "hybrid": 0.3,
    "name_overlap": 0.1,
    "intro_overlap": 0.1,
}
_EMBEDDING_SIZE = 8
_MIN_POS_TO_TRAIN = 10
_DNN_HIDDEN = (64, 32)
_TRAIN_EPOCHS = 30
_TRAIN_LR = 1e-2
_NEG_RATIO = 3


class DCNReranker:
    """DCN v2 保底重排器：复用 embedding 等既有组件。

    构造时仅持有各组件引用，torch / DCN_V2 延迟导入（未安装则全程走启发式）。
    KB 加载后由 engine 调 ``rebuild_index`` + ``try_load`` 装载匹配权重。
    """

    def __init__(
        self,
        embedding_model: Any,
        vector_index: Any,
        store: Any,
        model_dir: str | None = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.vector_index = vector_index
        self.store = store
        self.model_dir = model_dir
        # service_id -> 稳定整数下标（sorted 后顺序，随 KB 重建）
        self.service_ids: list[str] = []
        self.service_id_to_idx: dict[str, int] = {}
        self.kb_hash: str = ""
        # 已加载的模型（torch.nn.Module）或 None
        self._model: Any = None
        self._torch: Any = None

    # ---------- 词表 / 权重管理 ----------
    def rebuild_index(self, services: dict[str, Any], kb_hash: str) -> None:
        """KB 加载后重建 service_id→idx 词表；若 kb_hash 变化则丢弃旧模型。"""
        if kb_hash and kb_hash != self.kb_hash:
            self._model = None
        self.kb_hash = kb_hash or ""
        self.service_ids = sorted(services.keys())
        self.service_id_to_idx = {
            sid: i for i, sid in enumerate(self.service_ids)
        }

    def _model_path(self) -> str | None:
        if not self.model_dir or not self.kb_hash:
            return None
        return os.path.join(self.model_dir, f"dcn_reranker_{self.kb_hash}.pt")

    def _load_torch(self) -> tuple[Any, Any] | tuple[None, None]:
        """延迟导入 torch + DCN_V2；任一不可用返回 (None, None)。"""
        if self._torch is not None:
            return self._torch, self._DCN_V2  # type: ignore[attr-defined]
        try:
            import torch  # type: ignore
            from .dcn_v2 import DCN_V2  # 触发 numpy/pandas/sklearn 顶部导入
            self._torch = torch
            self._DCN_V2 = DCN_V2  # type: ignore[attr-defined]
            return torch, DCN_V2
        except Exception:  # noqa: BLE001 - 依赖缺失时静默降级
            self._torch = None
            return None, None

    @property
    def available(self) -> bool:
        """是否已加载可用的 DCN 模型（torch + 权重就绪）。"""
        return self._model is not None

    def try_load(self) -> bool:
        """尝试加载与当前 kb_hash 匹配的权重；失败静默返回 False。"""
        torch, _ = self._load_torch()
        if torch is None:
            return False
        path = self._model_path()
        if not path or not os.path.exists(path):
            return False
        try:
            state = torch.load(path, map_location="cpu", weights_only=False)
            if not isinstance(state, dict):
                return False
            if state.get("kb_hash") != self.kb_hash:
                return False
            saved_ids = state.get("service_ids")
            if saved_ids != self.service_ids:
                return False
            model = self._build_model()
            model.load_state_dict(state["state_dict"])
            model.eval()
            self._model = model
            logger.info(
                "DCN reranker weights loaded (kb_hash=%s, services=%d)",
                self.kb_hash[:8], len(self.service_ids),
            )
            return True
        except Exception:  # noqa: BLE001
            logger.warning("DCN reranker load failed", exc_info=True)
            self._model = None
            return False

    def _build_model(self) -> Any:
        """按当前词表大小构造 DCN_V2 网络。

        feat_size 按插入顺序决定 feature_index 列下标，须与 _build_features 的
        行列序 [svc_idx, cosine, hybrid, name_ov, intro_ov, pop] 严格一致。
        linear/dnn 特征列均传全量（DCN_V2 内部按 'sparse'/'dense' 过滤）。
        """
        torch, DCN_V2 = self._load_torch()
        num_services = max(2, len(self.service_ids))
        feat_size = {
            "service_id": num_services,
            "cosine": 1,
            "hybrid": 1,
            "name_overlap": 1,
            "intro_overlap": 1,
            "popularity": 1,
        }
        all_cols = [
            ("service_id", "sparse"),
            ("cosine", "dense"),
            ("hybrid", "dense"),
            ("name_overlap", "dense"),
            ("intro_overlap", "dense"),
            ("popularity", "dense"),
        ]
        return DCN_V2(
            feat_size, _EMBEDDING_SIZE, all_cols, all_cols,
            dnn_hidden_units=_DNN_HIDDEN, drop_rate=0.2,
        )

    # ---------- 特征 ----------
    def _build_features(
        self,
        query: str,
        query_embedding: list[float] | None,
        candidates: list[dict[str, Any]],
        pop_norm_map: dict[str, float] | None = None,
    ) -> list[list[float]]:
        """为每个候选构造一行特征 [svc_idx, cosine, hybrid, name_ov, intro_ov, pop]。

        pop_norm_map 可由训练侧预计算（全局 min-max 归一）传入；推理时缺省按
        当前候选集 min-max 归一（单候选时退化为 0，仅影响训练，故训练侧传全局 map）。
        """
        if query_embedding is None:
            query_embedding = self.embedding_model.embed(query) if query else []
        q_tokens = set(tokenize(query)) if query else set()
        q_norm = max(len(q_tokens), 1)
        if pop_norm_map is None:
            # 推理路径：跨候选 min-max 归一
            pop_raw = self.store.popularity_decayed()
            pop_vals = [pop_raw.get(item["service_id"], 0.0) for item in candidates]
            pop_max = max(pop_vals) if pop_vals else 0.0
            pop_min = min(pop_vals) if pop_vals else 0.0
            pop_range = (pop_max - pop_min) or 1.0
            pop_norm_map = {
                item["service_id"]: (pv - pop_min) / pop_range
                for item, pv in zip(candidates, pop_vals)
            }

        rows: list[list[float]] = []
        for item in candidates:
            sid = item["service_id"]
            idx = self.service_id_to_idx.get(sid, 0)
            cand_vec = self.vector_index.get(sid)
            cosine = (
                cosine_similarity(query_embedding, cand_vec)
                if query_embedding and cand_vec else 0.0
            )
            name_tokens = set(tokenize(str(item.get("service_name", ""))))
            intro_tokens = set(tokenize(str(item.get("service_intro", ""))))
            name_ov = len(q_tokens & name_tokens) / q_norm
            intro_ov = len(q_tokens & intro_tokens) / q_norm
            hybrid = float(item.get("score", 0.0))
            pop_norm = pop_norm_map.get(sid, 0.0)
            rows.append([float(idx), cosine, hybrid, name_ov, intro_ov, pop_norm])
        return rows

    # ---------- 推理 ----------
    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """对候选重排：DCN_V2 可用则走深度模型，否则走线性启发式。

        返回候选浅拷贝列表（按 rerank_score 降序），每项附加：
          ``rerank_score``（排序依据）、``dcn_score``（原始分，便于诊断）。
        不附加 ``rerank_reason``（由调用方统一处理）。
        """
        if not candidates:
            return []
        rows = self._build_features(query, query_embedding, candidates)
        scores = self._score_rows(rows)
        ranked: list[dict[str, Any]] = []
        for item, score in zip(candidates, scores):
            ranked.append({**item, "rerank_score": float(score), "dcn_score": float(score)})
        ranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return ranked

    def _score_rows(self, rows: list[list[float]]) -> list[float]:
        """对特征行打分：DCN 模型可用走 forward，否则线性启发式。

        单候选（batch=1）时跳过 DCN：参考实现 CrossNetMix 用 ``squeeze()`` 会抹掉
        batch 维导致形状错位，且单候选无需排序，直接走启发式即可。
        """
        if not rows:
            return []
        if self._model is not None and len(rows) >= 2:
            torch, _ = self._load_torch()
            if torch is not None:
                try:
                    x = torch.tensor(rows, dtype=torch.float32)
                    self._model.eval()
                    with torch.no_grad():
                        y = self._model(x)
                    y = y.squeeze(-1) if y.dim() > 1 else y
                    return [float(v) for v in y.tolist()]
                except Exception:  # noqa: BLE001 - 推理异常降级启发式
                    logger.warning("DCN forward failed, fallback to heuristic", exc_info=True)
        return [self._heuristic_score(row) for row in rows]

    @staticmethod
    def _heuristic_score(row: list[float]) -> float:
        """线性启发式：cosine 0.5 + hybrid 0.3 + name_ov 0.1 + intro_ov 0.1。

        popularity 已隐含在 hybrid（召回侧混合分含 popularity 权重），故不单列，
        避免双重计数。复用 embedding 的 cosine 作为主信号。
        """
        _idx, cosine, hybrid, name_ov, intro_ov, _pop = row
        return (
            _HEURISTIC_WEIGHTS["cosine"] * cosine
            + _HEURISTIC_WEIGHTS["hybrid"] * hybrid
            + _HEURISTIC_WEIGHTS["name_overlap"] * name_ov
            + _HEURISTIC_WEIGHTS["intro_overlap"] * intro_ov
        )

    # ---------- 训练 ----------
    def train_from_store(self, services: dict[str, Any]) -> dict[str, Any]:
        """从 store 的点击/搜索日志构造正负样本训练 DCN_V2 并落盘。

        返回训练摘要 {trained, positives, negatives, epochs, path}。
        torch 不可用 / 样本不足 / 词表未建 → trained=False 并说明原因。
        """
        torch, _ = self._load_torch()
        summary = {
            "trained": False, "positives": 0, "negatives": 0,
            "epochs": 0, "path": None, "reason": "",
        }
        if torch is None:
            summary["reason"] = "torch/dcn_v2 unavailable"
            return summary
        if len(self.service_ids) < 2:
            summary["reason"] = "service index not built or too few services"
            return summary

        pairs = self.store.click_query_pairs()
        positives = [(q, sid) for sid, q in pairs if sid in self.service_id_to_idx]
        summary["positives"] = len(positives)
        if len(positives) < _MIN_POS_TO_TRAIN:
            summary["reason"] = f"insufficient positives ({len(positives)} < {_MIN_POS_TO_TRAIN})"
            return summary

        rng = random.Random(2022)
        sid_list = list(self.service_ids)
        # 负采样：每条正样本采 _NEG_RATIO 个未点击服务
        samples: list[tuple[str, str, int]] = [(q, sid, 1) for q, sid in positives]
        for q, pos_sid in positives:
            drawn = 0
            while drawn < _NEG_RATIO:
                neg_sid = rng.choice(sid_list)
                if neg_sid != pos_sid:
                    samples.append((q, neg_sid, 0))
                    drawn += 1
        summary["negatives"] = len(samples) - len(positives)

        # 全局 popularity min-max 归一（跨所有服务），供训练特征稳定使用；
        # 单样本逐条调 _build_features 时候选集为单元素，跨候选归一会退化成 0。
        pop_raw = self.store.popularity_decayed()
        pop_all_vals = list(pop_raw.values()) or [0.0]
        pop_max = max(pop_all_vals)
        pop_min = min(pop_all_vals)
        pop_range = (pop_max - pop_min) or 1.0
        pop_norm_map = {sid: (v - pop_min) / pop_range for sid, v in pop_raw.items()}

        # 构造特征（候选 = {service_id, score=0, name/intro 从 services 取）
        feat_rows: list[list[float]] = []
        labels: list[float] = []
        for q, sid, label in samples:
            svc = services.get(sid)
            if svc is None:
                continue
            item = {
                "service_id": sid,
                "service_name": svc.service_name,
                "service_intro": svc.service_intro,
                "score": 0.0,
            }
            row = self._build_features(q, None, [item], pop_norm_map)[0]
            feat_rows.append(row)
            labels.append(float(label))

        if len(feat_rows) < _MIN_POS_TO_TRAIN:
            summary["reason"] = "feature build yielded too few rows"
            return summary

        x = torch.tensor(feat_rows, dtype=torch.float32)
        y = torch.tensor(labels, dtype=torch.float32).unsqueeze(1)
        model = self._build_model()
        model.train()
        loss_fn = torch.nn.BCELoss(reduction="mean")
        opt = torch.optim.Adam(model.parameters(), lr=_TRAIN_LR, weight_decay=1e-3)
        for epoch in range(_TRAIN_EPOCHS):
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            summary["epochs"] = epoch + 1
        model.eval()
        self._model = model

        path = self._model_path()
        if path:
            try:
                os.makedirs(self.model_dir or ".", exist_ok=True)
                torch.save(
                    {
                        "kb_hash": self.kb_hash,
                        "service_ids": self.service_ids,
                        "state_dict": model.state_dict(),
                    },
                    path,
                )
                summary["path"] = path
            except OSError:
                logger.warning("DCN reranker weights save failed", exc_info=True)
        summary["trained"] = True
        logger.info(
            "DCN reranker trained: pos=%d neg=%d epochs=%d",
            summary["positives"], summary["negatives"], summary["epochs"],
        )
        return summary
