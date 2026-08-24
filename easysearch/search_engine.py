from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
import json
import math
import os
import re
from typing import Any, Callable
from urllib import error, request


TOKEN_PATTERN = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


@dataclass(frozen=True)
class ServiceRecord:
    service_id: str
    service_name: str
    aliases: list[str]
    service_intro: str
    route: Any

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ServiceRecord":
        required = {"service_id", "service_name", "aliases", "service_intro", "route"}
        missing = required.difference(payload)
        if missing:
            raise ValueError(f"Missing required fields: {sorted(missing)}")
        aliases = payload.get("aliases") or []
        if not isinstance(aliases, list):
            aliases = [str(aliases)]
        return cls(
            service_id=str(payload["service_id"]),
            service_name=str(payload["service_name"]),
            aliases=[str(item) for item in aliases],
            service_intro=str(payload["service_intro"]),
            route=payload["route"],
        )


class DashScopeClient:
    def __init__(
        self,
        api_key: str | None = None,
        requester: Callable[[str, bytes, dict[str, str]], dict[str, Any]] | None = None,
        timeout: int = 15,
    ) -> None:
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        self.requester = requester or self._default_requester
        self.timeout = timeout

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    def post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("DashScope API key is not configured")
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        return self.requester(url, body, headers)

    def _default_requester(self, url: str, body: bytes, headers: dict[str, str]) -> dict[str, Any]:
        req = request.Request(url=url, data=body, headers=headers, method="POST")
        try:
            with request.urlopen(req, timeout=self.timeout) as resp:
                content = resp.read().decode("utf-8")
        except error.HTTPError as exc:
            message = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DashScope request failed: {exc.code} {message}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"DashScope network error: {exc.reason}") from exc

        try:
            return json.loads(content)
        except json.JSONDecodeError as exc:
            raise RuntimeError("DashScope response is not valid JSON") from exc


class Qwen37TextEmbedding:
    """qwen3.7-text-embedding adapter with offline fallback."""

    model_name = "qwen3.7-text-embedding"
    endpoint = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"

    def __init__(self, dashscope_client: DashScopeClient, dim: int = 32) -> None:
        self.client = dashscope_client
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        if not text:
            return [0.0] * self.dim

        if self.client.enabled:
            payload = {
                "model": self.model_name,
                "input": {
                    "texts": [text],
                },
            }
            try:
                response = self.client.post_json(self.endpoint, payload)
                embeddings = response.get("output", {}).get("embeddings", [])
                if embeddings:
                    first = embeddings[0]
                    vector = first.get("embedding") if isinstance(first, dict) else first
                    if isinstance(vector, list) and vector:
                        return _l2_normalize([float(item) for item in vector])
            except RuntimeError:
                pass

        return self._local_embed(text)

    def _local_embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in _tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for i in range(self.dim):
                vector[i] += (digest[i] / 255.0) - 0.5
        return _l2_normalize(vector)


class DINHistoryOptimizer:
    """Simple DIN-style user sequence attention for query embedding."""

    def optimize(
        self,
        query_embedding: list[float],
        history_embeddings: list[list[float]],
    ) -> list[float]:
        if not history_embeddings:
            return query_embedding
        weighted_sum = [0.0] * len(query_embedding)
        total_weight = 0.0
        for index, embedding in enumerate(reversed(history_embeddings), start=1):
            relevance = max(0.0, _cosine_similarity(query_embedding, embedding))
            recency_weight = 1.0 / index
            weight = 0.7 * relevance + 0.3 * recency_weight
            total_weight += weight
            for i, value in enumerate(embedding):
                weighted_sum[i] += value * weight
        if total_weight == 0:
            return query_embedding
        mixed = [
            0.8 * query_embedding[i] + 0.2 * (weighted_sum[i] / total_weight)
            for i in range(len(query_embedding))
        ]
        return _l2_normalize(mixed)


class Qwen3VLPlusReasoner:
    model_name = "qwen3-vl-plus"
    endpoint = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

    def __init__(self, dashscope_client: DashScopeClient) -> None:
        self.client = dashscope_client

    def generate_reasons(self, query: str, ranked_candidates: list[dict[str, Any]]) -> dict[str, str]:
        if not self.client.enabled or not ranked_candidates:
            return {}

        compact_docs = [
            {
                "service_id": item["service_id"],
                "service_name": item["service_name"],
                "service_intro": item["service_intro"],
                "route": item["route"],
            }
            for item in ranked_candidates
        ]
        prompt = (
            "你是搜索重排解释器。请根据query与候选服务内容，"
            "为每个service_id给出一句排序理由。"
            "仅输出JSON数组，每项格式为{\"service_id\":\"...\",\"reason\":\"...\"}。"
            f"query: {query}\n"
            f"candidates: {json.dumps(compact_docs, ensure_ascii=False)}"
        )

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt,
                        }
                    ],
                }
            ],
            "stream": False,
            "enable_thinking": True,
            "thinking_budget": 81920,
        }

        try:
            response = self.client.post_json(self.endpoint, payload)
            raw_content = (
                response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
            if isinstance(raw_content, list):
                raw_content = "".join(
                    part.get("text", "") if isinstance(part, dict) else str(part)
                    for part in raw_content
                )
            parsed = _extract_json(raw_content)
            if not isinstance(parsed, list):
                return {}

            reasons: dict[str, str] = {}
            for row in parsed:
                if not isinstance(row, dict):
                    continue
                service_id = str(row.get("service_id", "")).strip()
                reason = str(row.get("reason", "")).strip()
                if service_id and reason:
                    reasons[service_id] = reason
            return reasons
        except RuntimeError:
            return {}


class Qwen3VLReranker:
    """qwen3-vl-rerank adapter with local fallback and optional qwen3-vl-plus reasons."""

    model_name = "qwen3-vl-rerank"
    endpoint = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"

    def __init__(self, dashscope_client: DashScopeClient, reasoner: Qwen3VLPlusReasoner) -> None:
        self.client = dashscope_client
        self.reasoner = reasoner

    def rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if not candidates:
            return []

        reranked = self._remote_rerank(query, candidates) if self.client.enabled else self._local_rerank(query, candidates)
        reasons = self.reasoner.generate_reasons(query, reranked)

        for item in reranked:
            if not item.get("rerank_reason"):
                item["rerank_reason"] = reasons.get(
                    item["service_id"],
                    f"Qwen3-VL-plus: 综合相关性与关键词覆盖，混合分{item['score']:.4f}。",
                )
        return reranked

    def _remote_rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        docs = [self._candidate_text(item) for item in candidates]
        payload = {
            "model": self.model_name,
            "input": {
                "query": query,
                "documents": docs,
            },
            "parameters": {
                "return_documents": True,
                "top_n": len(candidates),
            },
        }

        try:
            response = self.client.post_json(self.endpoint, payload)
            rows = response.get("output", {}).get("results", [])
            if not isinstance(rows, list) or not rows:
                return self._local_rerank(query, candidates)

            mapped: list[dict[str, Any]] = []
            for row in rows:
                index = row.get("index")
                if not isinstance(index, int) or index < 0 or index >= len(candidates):
                    continue
                item = dict(candidates[index])
                relevance = row.get("relevance_score")
                if isinstance(relevance, (int, float)):
                    item["rerank_score"] = float(relevance)
                else:
                    item["rerank_score"] = item["score"]
                mapped.append(item)

            if not mapped:
                return self._local_rerank(query, candidates)
            mapped.sort(key=lambda x: x["rerank_score"], reverse=True)
            return mapped
        except RuntimeError:
            return self._local_rerank(query, candidates)

    def _local_rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        query_tokens = set(_tokenize(query))
        reranked: list[dict[str, Any]] = []
        for item in candidates:
            text_tokens = set(_tokenize(self._candidate_text(item)))
            overlap = len(query_tokens.intersection(text_tokens))
            ranked = dict(item)
            ranked["rerank_score"] = ranked["score"] + 0.01 * overlap
            ranked["rerank_reason"] = (
                f"Qwen3-VL-plus: query与服务关键词重合{overlap}个，基础混合分{ranked['score']:.4f}。"
            )
            reranked.append(ranked)

        reranked.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked

    @staticmethod
    def _candidate_text(item: dict[str, Any]) -> str:
        return (
            f"{item['service_name']} {' '.join(item.get('aliases', []))} "
            f"{item['service_intro']} {item.get('route', '')} {item.get('component', '')} {item.get('decision_button', '')}"
        )


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_freq: dict[str, int] = defaultdict(int)
        self.doc_lengths: dict[str, int] = {}
        self.term_freqs: dict[str, Counter[str]] = {}
        self.avg_doc_len: float = 0.0

    def build(self, docs: dict[str, str]) -> None:
        self.doc_freq.clear()
        self.doc_lengths.clear()
        self.term_freqs.clear()
        total_len = 0
        for doc_id, text in docs.items():
            tokens = _tokenize(text)
            counts = Counter(tokens)
            self.term_freqs[doc_id] = counts
            self.doc_lengths[doc_id] = len(tokens)
            total_len += len(tokens)
            for term in counts:
                self.doc_freq[term] += 1
        self.avg_doc_len = (total_len / len(docs)) if docs else 0.0

    def score(self, query: str, doc_id: str) -> float:
        if doc_id not in self.term_freqs or self.avg_doc_len == 0:
            return 0.0
        tokens = _tokenize(query)
        score = 0.0
        doc_len = self.doc_lengths[doc_id]
        num_docs = max(1, len(self.term_freqs))
        for term in tokens:
            freq = self.term_freqs[doc_id].get(term, 0)
            if freq == 0:
                continue
            df = self.doc_freq.get(term, 0)
            idf = math.log((num_docs - df + 0.5) / (df + 0.5) + 1.0)
            denom = freq + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
            score += idf * (freq * (self.k1 + 1)) / denom
        return score


class ServiceSearchEngine:
    def __init__(self, dashscope_client: DashScopeClient | None = None) -> None:
        self.dashscope_client = dashscope_client or DashScopeClient()
        self.embedding_model = Qwen37TextEmbedding(self.dashscope_client)
        self.history_optimizer = DINHistoryOptimizer()
        self.reasoner = Qwen3VLPlusReasoner(self.dashscope_client)
        self.reranker = Qwen3VLReranker(self.dashscope_client, self.reasoner)
        self.bm25 = BM25Index()

        self.services: dict[str, ServiceRecord] = {}
        self.service_embeddings: dict[str, list[float]] = {}
        self.user_query_history: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=50))
        self.user_click_history: dict[str, deque[str]] = defaultdict(lambda: deque(maxlen=50))
        self.global_click_counter: Counter[str] = Counter()

    def upload_knowledge_base_from_json(self, json_path: str) -> None:
        with open(json_path, "r", encoding="utf-8") as fp:
            payload = json.load(fp)
        if not isinstance(payload, list):
            raise ValueError("Knowledge base JSON must be a list")
        self._load_services(payload)

    def load_knowledge_base(self, payload: list[dict[str, Any]]) -> None:
        self._load_services(payload)

    def search(self, user_id: str, query: str) -> list[dict[str, Any]]:
        if not query.strip() or not self.services:
            return []

        self.user_query_history[user_id].append(query)
        query_embedding = self.embedding_model.embed(query)

        if len(self.user_query_history[user_id]) > 10:
            history_embeddings = [
                self.embedding_model.embed(item)
                for item in list(self.user_query_history[user_id])[:-1]
            ]
            query_embedding = self.history_optimizer.optimize(query_embedding, history_embeddings)

        bm25_raw: dict[str, float] = {}
        vector_raw: dict[str, float] = {}
        popularity_raw: dict[str, float] = {}

        for service_id in self.services:
            vector_raw[service_id] = max(0.0, _cosine_similarity(query_embedding, self.service_embeddings[service_id]))
            bm25_raw[service_id] = self.bm25.score(query, service_id)
            popularity_raw[service_id] = float(self.global_click_counter[service_id])

        bm25_norm = _normalize_scores(bm25_raw)
        popularity_norm = _normalize_scores(popularity_raw)

        candidates: list[dict[str, Any]] = []
        for service_id, service in self.services.items():
            score = self._hybrid_score(
                vector_similarity=vector_raw[service_id],
                bm25_score=bm25_norm[service_id],
                popularity_score=popularity_norm[service_id],
            )
            route_info = self._route_info(service.route)
            candidates.append(
                {
                    "service_id": service.service_id,
                    "service_name": service.service_name,
                    "aliases": service.aliases,
                    "service_intro": service.service_intro,
                    "route": route_info["route"],
                    "component": route_info["component"],
                    "decision_button": route_info["decision_button"],
                    "score": score,
                }
            )

        candidates.sort(key=lambda x: x["score"], reverse=True)
        top20 = candidates[:20]
        reranked = self.reranker.rerank(query, top20)
        return reranked[:10]

    def record_click(self, user_id: str, service_id: str) -> None:
        if service_id not in self.services:
            raise ValueError(f"Unknown service_id: {service_id}")
        self.user_click_history[user_id].append(service_id)
        self.global_click_counter[service_id] += 1

    def homepage_dropdown(self, user_id: str) -> dict[str, list[str]]:
        recent_queries = list(self.user_query_history[user_id])[-3:]
        recent_click_ids = list(self.user_click_history[user_id])[-3:]
        recent_clicked_services = [
            self.services[service_id].service_name
            for service_id in reversed(recent_click_ids)
            if service_id in self.services
        ]

        hot_services = [
            self.services[service_id].service_name
            for service_id, _ in self.global_click_counter.most_common(3)
            if service_id in self.services
        ]

        return {
            "recent_queries": list(reversed(recent_queries)),
            "recent_clicked_services": recent_clicked_services,
            "global_hot_services": hot_services,
        }

    def _load_services(self, payload: list[dict[str, Any]]) -> None:
        self.services.clear()
        docs: dict[str, str] = {}
        self.service_embeddings.clear()

        for item in payload:
            service = ServiceRecord.from_dict(item)
            self.services[service.service_id] = service
            service_text = self._service_text(service)
            docs[service.service_id] = service_text
            self.service_embeddings[service.service_id] = self.embedding_model.embed(service_text)

        self.bm25.build(docs)

    @staticmethod
    def _service_text(service: ServiceRecord) -> str:
        aliases = " ".join(service.aliases)
        route_info = ServiceSearchEngine._route_info(service.route)
        route_text = " ".join(
            part for part in [route_info["route"], route_info["component"], route_info["decision_button"]] if part
        )
        return f"{service.service_name} {aliases} {service.service_intro} {route_text}"

    @staticmethod
    def _route_info(route: Any) -> dict[str, str]:
        if isinstance(route, dict):
            return {
                "route": str(route.get("path", "")),
                "component": str(route.get("component", "")),
                "decision_button": str(route.get("action_button", "")),
            }
        return {"route": str(route), "component": "", "decision_button": ""}

    @staticmethod
    def _hybrid_score(vector_similarity: float, bm25_score: float, popularity_score: float) -> float:
        return 0.6 * vector_similarity + 0.3 * bm25_score + 0.1 * popularity_score


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_PATTERN.findall(text)]


def _safe_norm(vector: list[float]) -> float:
    return math.sqrt(sum(value * value for value in vector))


def _l2_normalize(vector: list[float]) -> list[float]:
    norm = _safe_norm(vector)
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    left_norm = _safe_norm(left)
    right_norm = _safe_norm(right)
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot = sum(l * r for l, r in zip(left, right))
    return dot / (left_norm * right_norm)


def _normalize_scores(raw: dict[str, float]) -> dict[str, float]:
    if not raw:
        return {}
    max_value = max(raw.values())
    min_value = min(raw.values())
    if math.isclose(max_value, min_value):
        return {key: 0.0 for key in raw}
    denom = max_value - min_value
    return {key: (value - min_value) / denom for key, value in raw.items()}


def _extract_json(text: str) -> Any:
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
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
