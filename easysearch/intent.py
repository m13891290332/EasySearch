"""M5 意图识别：规则分类用户查询意图。

意图标签（与 plan.md M5/M16 对齐）：
  - navigational   精确命中 KB service_name / alias → 直达唯一服务
  - multi_condition 含连接词（和/且/并且/同时/+ 与/及）→ 走 M6 交集
  - guide          指引型（如何开始/新手/流程/步骤/怎么操作/怎么玩）→ M16 步骤化答案
  - informational  含疑问词（怎么/如何/是什么/为什么...）→ 信息型
  - conversational 会话上下文存在 → 走 M7 长程对话
  - default        其余

优先级（先命中先返回）：
  navigational > multi_condition > guide > informational > conversational > default

注：guide 置于 informational 之前，因「怎么操作」含通用疑问词「怎么」，
需由更具体的 guide 短语优先捕获，避免被 informational 吞掉。

设计：本类无状态、不访问存储，便于单测；会话上下文存在性由调用方
（engine，M7 接入后）传入 has_session 标志。sub_queries 为 multi_condition
按连接词切分的结果，供 M6 直接消费。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

# 意图常量
NAVIGATIONAL = "navigational"
MULTI_CONDITION = "multi_condition"
GUIDE = "guide"
INFORMATIONAL = "informational"
CONVERSATIONAL = "conversational"
DEFAULT = "default"

# 多条件连接词（按优先匹配排序；split 时按所有连接词切分）
_MULTI_CONNECTORS: tuple[str, ...] = ("并且", "同时", "而且", "和", "且", "与", "及", "+", "&")
# M16 指引型短语（具体短语，避免与 informational 通用疑问词冲突）
_GUIDE_HINTS: tuple[str, ...] = (
    "如何开始",
    "怎么开始",
    "新手",
    "入门",
    "流程",
    "步骤",
    "怎么操作",
    "如何操作",
    "怎么玩",
    "如何玩",
    "怎么用",
    "如何使用",
    "操作流程",
    "使用流程",
    "指引",
    "引导",
    "操作指引",
)
# 信息型疑问词
_INFO_HINTS: tuple[str, ...] = (
    "怎么",
    "如何",
    "怎样",
    "是什么",
    "什么是",
    "为什么",
    "为啥",
    "哪儿",
    "哪里",
    "哪个",
    "哪些",
    "多少",
    "是否",
)

# M15 二次深度检索触发阈值（可按需调参）
DEEP_DELTA_THRESHOLD = 0.05  # top1-top2 综合分差低于此 → 头部分离不足
DEEP_MIN_HITS = 3  # 命中数低于此 → 命中稀疏
DEEP_INFO_TOP1_MIN = 0.4  # informational 意图下 top1 相关度低于此 → 低相关


@dataclass
class IntentResult:
    """意图分类结果。"""

    intent: str = DEFAULT
    matched_service_id: str | None = None  # navigational 命中的 service_id
    sub_queries: list[str] = field(default_factory=list)  # multi_condition 切分
    raw_query: str = ""

    def as_dict(self) -> dict[str, Any]:
        """API 响应友好形式。"""
        return {
            "intent": self.intent,
            "matched_service_id": self.matched_service_id,
            "sub_queries": list(self.sub_queries),
        }


@dataclass
class ConfidenceResult:
    """首次检索置信度评估结果（M15）。

    should_deep_search=True 时由 engine 触发二次深度检索（仅一次，防递归）。
    confidence 为 0–1 的粗略置信度（供日志/前端展示），触发判定只看 should_deep_search。
    """

    confidence: float = 1.0
    delta: float = 0.0  # top1 - top2 综合分差
    n_hits: int = 0  # 命中数（score>0 的结果数）
    should_deep_search: bool = False
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "confidence": round(self.confidence, 4),
            "delta": round(self.delta, 4),
            "n_hits": self.n_hits,
            "should_deep_search": self.should_deep_search,
            "reason": self.reason,
        }


# 预编译连接词切分正则（转义 + 按长度降序避免 '并且' 被 '且' 提前切分）
_CONNECTOR_PATTERN = re.compile(
    "|".join(sorted((re.escape(c) for c in _MULTI_CONNECTORS), key=len, reverse=True))
)


class IntentRouter:
    """规则意图分类器（M5）。

    用法：
        router = IntentRouter()
        result = router.classify(query, services=engine.services, has_session=False)
    """

    def classify(
        self,
        query: str,
        services: dict[str, Any] | None = None,
        has_session: bool = False,
    ) -> IntentResult:
        """对 query 分类，返回 IntentResult。

        services：KB 的 {service_id: ServiceRecord}（用于 navigational 精确匹配）。
        has_session：是否存在会话上下文（M7 接入后由 engine 传入）。
        """
        q = (query or "").strip()
        result = IntentResult(raw_query=q)
        if not q:
            return result

        # 1. navigational：精确命中 service_name / alias（大小写不敏感）
        if services:
            matched = self._match_navigational(q, services)
            if matched is not None:
                result.intent = NAVIGATIONAL
                result.matched_service_id = matched
                return result

        # 2. multi_condition：含连接词
        if self._has_connector(q):
            result.intent = MULTI_CONDITION
            result.sub_queries = self._split_sub_queries(q)
            return result

        # 3. guide：指引型短语（M16 步骤化答案）
        if self._has_guide_hint(q):
            result.intent = GUIDE
            return result

        # 4. informational：含疑问词
        if self._has_info_hint(q):
            result.intent = INFORMATIONAL
            return result

        # 5. conversational：会话上下文存在
        if has_session:
            result.intent = CONVERSATIONAL
            return result

        # 6. default
        result.intent = DEFAULT
        return result

    # ---------- navigational ----------
    @staticmethod
    def _match_navigational(query: str, services: dict[str, Any]) -> str | None:
        """精确匹配 service_name 或 alias（去空白、大小写不敏感）。

        命中多个时返回 name 优先于 alias 的第一个（保确定性）。
        """
        q = query.strip().lower()
        if not q:
            return None
        # 第一轮：service_name 精确命中
        for sid, svc in services.items():
            name = str(getattr(svc, "service_name", "") or "").strip().lower()
            if name and name == q:
                return sid
        # 第二轮：alias 精确命中
        for sid, svc in services.items():
            aliases = getattr(svc, "aliases", None) or []
            for alias in aliases:
                a = str(alias).strip().lower()
                if a and a == q:
                    return sid
        return None

    # ---------- multi_condition ----------
    @staticmethod
    def _has_connector(query: str) -> bool:
        return any(c in query for c in _MULTI_CONNECTORS)

    @staticmethod
    def _split_sub_queries(query: str) -> list[str]:
        parts = _CONNECTOR_PATTERN.split(query)
        return [p.strip() for p in parts if p.strip()]

    # ---------- informational ----------
    @staticmethod
    def _has_info_hint(query: str) -> bool:
        return any(h in query for h in _INFO_HINTS)

    # ---------- guide ----------
    @staticmethod
    def _has_guide_hint(query: str) -> bool:
        return any(h in query for h in _GUIDE_HINTS)

    # ---------- M15 置信度评估 ----------
    @staticmethod
    def _result_score(item: Any) -> float:
        """取结果项的综合分：优先 rerank_score，回退 score。"""
        rs = item.get("rerank_score") if isinstance(item, dict) else None
        if rs is None:
            rs = item.get("score", 0.0) if isinstance(item, dict) else 0.0
        try:
            return float(rs)
        except (TypeError, ValueError):
            return 0.0

    def evaluate_confidence(
        self,
        results: list[Any],
        intent: str,
        is_cold_user: bool = False,
    ) -> ConfidenceResult:
        """评估首次检索结果置信度，判定是否触发二次深度检索（M15）。

        触发条件（满足任一）：
          - Δ < DEEP_DELTA_THRESHOLD（头部分离不足）
          - N < DEEP_MIN_HITS（命中稀疏）
          - informational 且 top1 相关度 < DEEP_INFO_TOP1_MIN
        不触发：navigational 精确命中、multi_condition（走 M6）、conversational（走 M7）。
        is_cold_user 作为 confidence 数值的弱因子（不新增触发条件）。
        """
        # 不触发：各自走直达/M6/M7
        if intent in (NAVIGATIONAL, MULTI_CONDITION, CONVERSATIONAL):
            return ConfidenceResult(
                confidence=1.0,
                delta=0.0,
                n_hits=len(results),
                should_deep_search=False,
                reason=f"{intent} 意图不触发二次检索",
            )
        # 无命中：直接触发
        if not results:
            return ConfidenceResult(
                confidence=0.0,
                delta=0.0,
                n_hits=0,
                should_deep_search=True,
                reason="无命中，触发二次深度检索",
            )

        # top1/top2 综合分
        top_scores = sorted(
            (self._result_score(r) for r in results[:3]), reverse=True
        )
        top1 = top_scores[0]
        top2 = top_scores[1] if len(top_scores) > 1 else 0.0
        delta = top1 - top2
        n_hits = sum(1 for r in results if self._result_score(r) > 0.0)

        triggers: list[str] = []
        if delta < DEEP_DELTA_THRESHOLD:
            triggers.append("头部分离不足")
        if n_hits < DEEP_MIN_HITS:
            triggers.append("命中稀疏")
        if intent == INFORMATIONAL and top1 < DEEP_INFO_TOP1_MIN:
            triggers.append("信息型低相关")

        if triggers:
            # confidence 粗略估算：每命中一个触发条件扣 0.3，冷启动再扣 0.1
            conf = max(0.0, 1.0 - 0.3 * len(triggers) - (0.1 if is_cold_user else 0.0))
            return ConfidenceResult(
                confidence=conf,
                delta=delta,
                n_hits=n_hits,
                should_deep_search=True,
                reason="；".join(triggers),
            )
        return ConfidenceResult(
            confidence=0.8,
            delta=delta,
            n_hits=n_hits,
            should_deep_search=False,
            reason="置信度充足",
        )
