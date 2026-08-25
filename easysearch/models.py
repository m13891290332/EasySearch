from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .safety import sanitize_text

_REQUIRED_FIELDS = {"service_id", "service_name", "aliases", "service_intro", "route"}

# M8：组件动作字段约束（每项 {name, action, params?}）
_MAX_COMPONENTS = 20
_MAX_COMP_NAME_LEN = 100
_MAX_COMP_ACTION_LEN = 100
_MAX_COMP_PARAMS_VALUE_LEN = 500


@dataclass(frozen=True)
class ServiceRecord:
    """服务知识库记录（5 字段 + M8 components）。"""

    service_id: str
    service_name: str
    aliases: list[str]
    service_intro: str
    route: Any
    # M8：页面内组件动作列表，每项 {name, action, params?}；缺省空列表（旧 KB 向后兼容）
    components: list[dict[str, Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ServiceRecord":
        missing = _REQUIRED_FIELDS.difference(payload)
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
            components=_sanitize_components(payload.get("components")),
        )

    @property
    def searchable_text(self) -> str:
        info = route_info(self.route)
        route_text = " ".join(
            part for part in [info["route"], info["component"], info["decision_button"]] if part
        )
        return f"{self.service_name} {' '.join(self.aliases)} {self.service_intro} {route_text}"

    def to_dict(self) -> dict[str, Any]:
        """M9：序列化为 KB JSON 条目（与 from_dict 输入同构），供导出/快照复用。

        route 保留原始形态（dict 或 string）；re-import 会再次 sanitize（幂等），
        searchable_text 不变，故 kb_hash 一致。
        """
        return {
            "service_id": self.service_id,
            "service_name": self.service_name,
            "aliases": list(self.aliases),
            "service_intro": self.service_intro,
            "route": self.route,
            "components": list(self.components),
        }


def _sanitize_params(value: Any) -> dict[str, Any] | None:
    """M8：清洗组件 params。仅保留 dict；字符串值限长，剥控制字符；非 dict 返回 None。"""
    if not isinstance(value, dict):
        return None
    cleaned: dict[str, Any] = {}
    for key, val in value.items():
        if not isinstance(key, str):
            continue
        if isinstance(val, str):
            cleaned[key] = sanitize_text(val, _MAX_COMP_PARAMS_VALUE_LEN)
        elif isinstance(val, (int, float, bool)) or val is None:
            cleaned[key] = val
        elif isinstance(val, dict):
            nested = _sanitize_params(val)
            if nested:
                cleaned[key] = nested
        # list 等其他类型跳过（本期打桩不转发复杂结构）
    return cleaned


def _sanitize_components(raw: Any) -> list[dict[str, Any]]:
    """M8：清洗 components 列表。每项 {name, action, params?}；非法项过滤。

    - name/action 必填且为非空字符串；剥控制字符 + 限长
    - params 可选，必须为 dict（否则置 None / 丢弃）
    - 最多 _MAX_COMPONENTS 项，超出截断
    - 旧 KB 无 components 字段 → 返回空列表（向后兼容）
    """
    if not isinstance(raw, list):
        return []
    result: list[dict[str, Any]] = []
    for item in raw[:_MAX_COMPONENTS]:
        if not isinstance(item, dict):
            continue
        name = sanitize_text(item.get("name", ""), _MAX_COMP_NAME_LEN).strip()
        action = sanitize_text(item.get("action", ""), _MAX_COMP_ACTION_LEN).strip()
        if not name or not action:
            continue
        comp: dict[str, Any] = {"name": name, "action": action}
        if "params" in item and item["params"] is not None:
            params = _sanitize_params(item["params"])
            if params:
                comp["params"] = params
        result.append(comp)
    return result


def _slug_to_component(path: str) -> str:
    """从路由 slug 派生页面组件名：/go/account/open-account -> OpenAccount。"""
    last = path.rstrip("/").rsplit("/", 1)[-1] or path
    parts = [p for p in re.split(r"[-_/]+", last) if p]
    return "".join(part[:1].upper() + part[1:] for part in parts) if parts else "ServicePage"


def route_info(route: Any) -> dict[str, Any]:
    """统一解析 route 为可点击路径 / 页面组件 / 决策执行按钮。

    - dict：提取 path/component/action_button（原始数据，derived=False）
    - string：route 即可点击 path；component 由 slug 派生；决策按钮默认「进入」（derived=True）
    """
    if isinstance(route, dict):
        return {
            "route": str(route.get("path", "")),
            "component": str(route.get("component", "")),
            "decision_button": str(route.get("action_button", "")),
            "derived": False,
        }
    path = str(route)
    return {
        "route": path,
        "component": _slug_to_component(path),
        "decision_button": "进入",
        "derived": True,
    }


@dataclass
class SearchCandidate:
    """检索结果项（用于内部传递与序列化）。"""

    service_id: str
    service_name: str
    aliases: list[str] = field(default_factory=list)
    service_intro: str = ""
    route: str = ""
    component: str = ""
    decision_button: str = ""
    derived: bool = False
    # M8：页面内组件动作列表（与 ServiceRecord.components 同构）
    components: list[dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    rerank_score: float | None = None
    rerank_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "service_id": self.service_id,
            "service_name": self.service_name,
            "aliases": self.aliases,
            "service_intro": self.service_intro,
            "route": self.route,
            "component": self.component,
            "decision_button": self.decision_button,
            "derived": self.derived,
            "components": self.components,
            "score": self.score,
            "rerank_score": self.rerank_score,
            "rerank_reason": self.rerank_reason,
        }
