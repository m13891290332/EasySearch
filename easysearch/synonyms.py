"""A1 同义词扩展：领域词典 + KB 动态抽取。

- BM25 路径用 expand(tokens)：保留原 token + 追加同义词 token，让 BM25 命中 alias 文档
- 向量路径用 normalize(query)：归一到 canonical 规范词后再 embed，避免同义词稀释向量语义

canonical 选择规则：
  - DOMAIN_SYNONYMS 的 key 视为 canonical（规范词）
  - KB 抽取时 service_name 视为 canonical
  - 合并组时优先保留先注册的 canonical
"""
from __future__ import annotations

from .utils import tokenize

# 领域同义词覆盖层：手工编纂的证券业务通用词（~30 条）
# key 是 canonical（规范词），value 是同义词列表
DOMAIN_SYNONYMS: dict[str, list[str]] = {
    "开户": ["开股东卡", "证券开户", "网上开户", "新开户", "开立账户"],
    "银证转账": ["三方存管", "存管转账", "资金转账", "转入转出", "入金出金"],
    "申购": ["打新", "新股申购", "新债申购", "ipo申购", "一键打新"],
    "新股新债": ["打新", "新股", "新债", "ipo"],
    "订单": ["订单审批", "订单管理", "订单查询"],
    "委托": ["下单", "提交订单", "买卖委托"],
    "审批": ["审核", "复核", "审批通过", "审批流程"],
    "持仓": ["股票持仓", "持仓查询", "持仓明细"],
    "撤单": ["撤销委托", "撤委托", "取消订单"],
    "买入": ["买", "买进", "加仓", "建仓"],
    "卖出": ["卖", "减仓", "清仓"],
    "行情": ["股价", "实时行情", "股票行情", "看盘"],
    "查询": ["查看", "搜索", "查找"],
    "资金": ["余额", "可用资金", "资金账户"],
    "转账": ["转钱", "划转", "资金流转"],
    "风险": ["风控", "风险评估", "风险测评"],
    "账户": ["账号", "证券账户", "资金账户"],
    "密码": ["交易密码", "资金密码", "登录密码"],
    "登录": ["登入", "登陆", "sign in"],
    "退出": ["登出", "退出登录", "注销"],
    "消息": ["通知", "公告", "提醒"],
    "客服": ["在线客服", "联系客服", "人工服务"],
    "融资": ["融资融券", "融资买入", "两融"],
    "融券": ["融券卖出", "卖空", "做空"],
    "港股": ["港股通", "沪港通", "深港通"],
    "美股": ["海外股票", "中概股"],
    "基金": ["公募基金", "etf", "lof"],
    "理财": ["理财产品", "固收", "活期理财"],
    "可转债": ["转债", "可转换债券", "发债"],
}


class SynonymExpander:
    """同义词扩展器：领域词典 + KB 动态抽取的别名映射。

    canonical 追踪：
      - DOMAIN_SYNONYMS 的 key 是 canonical
      - KB 抽取时 service_name 是 canonical
      - 合并组时保留先注册的 canonical
    """

    def __init__(self, domain_dict: dict[str, list[str]] | None = None) -> None:
        # 每个组：set[str] 全部词 + str canonical
        self._groups: list[set[str]] = []
        self._canonicals: list[str] = []
        self._token_to_group: dict[str, int] = {}

        seed = domain_dict if domain_dict is not None else DOMAIN_SYNONYMS
        for canonical, syns in seed.items():
            self._add_group({canonical, *syns}, canonical=canonical)

    # ---------- 维护 ----------
    def _add_group(self, words: set[str], canonical: str | None = None) -> None:
        """添加一组同义词。若与已有组共享 token，合并到该组。

        canonical 指定该组的规范词；合并时优先保留先注册的 canonical。
        """
        words = {w for w in words if w}
        if not words:
            return
        # 若未指定 canonical，选最短者兜底（仅用于防御性调用）
        if canonical is None:
            canonical = min(words, key=lambda w: (len(w), w))
        # 找出与现有组共享的索引
        overlap_idx: set[int] = set()
        for w in words:
            if w in self._token_to_group:
                overlap_idx.add(self._token_to_group[w])
        if overlap_idx:
            # 合并到第一个重叠组，保留其 canonical
            target = min(overlap_idx)
            self._groups[target].update(words)
            for w in words:
                self._token_to_group[w] = target
            # 合并其他重叠组
            for idx in sorted(overlap_idx, reverse=True):
                if idx == target:
                    continue
                merged = self._groups[idx]
                self._groups[target].update(merged)
                for w in merged:
                    self._token_to_group[w] = target
                self._groups[idx] = set()  # 标记空，避免索引错位
                self._canonicals[idx] = ""  # 占位
        else:
            new_idx = len(self._groups)
            self._groups.append(words)
            self._canonicals.append(canonical)
            for w in words:
                self._token_to_group[w] = new_idx

    def update_from_kb(self, services: dict) -> None:
        """从 KB 抽取 alias ↔ service_name 双向同义组。

        service_name 视为 canonical。services: dict[service_id, ServiceRecord]
        """
        for record in services.values():
            group: set[str] = {record.service_name}
            for alias in record.aliases:
                group.add(alias)
            self._add_group(group, canonical=record.service_name)

    # ---------- 查询 ----------
    def synonyms_of(self, token: str) -> list[str]:
        """返回 token 的同义词（不含自身）。"""
        idx = self._token_to_group.get(token)
        if idx is None:
            return []
        return [w for w in self._groups[idx] if w != token]

    def canonical_of(self, token: str) -> str | None:
        """返回 token 所属组的 canonical；token 不在任何组返回 None。"""
        idx = self._token_to_group.get(token)
        if idx is None:
            return None
        return self._canonicals[idx] if idx < len(self._canonicals) else None

    def expand(self, tokens: list[str]) -> list[str]:
        """A1 BM25 路径：保留原 token + 追加同义词 token（不重复）。

        同义词若是多字短语（如"网上开户"），需经 tokenize 切成单 token
        再加入，否则与 BM25 索引侧（jieba 切词后）的 term 不匹配。
        """
        if not tokens:
            return []
        result: list[str] = list(tokens)
        seen: set[str] = set(tokens)
        for token in tokens:
            for syn in self.synonyms_of(token):
                if not syn:
                    continue
                # 同义词经 tokenize 切分后再加入（避免多字短语无法命中索引 term）
                for sub in tokenize(syn):
                    if sub and sub not in seen:
                        seen.add(sub)
                        result.append(sub)
                # 整串也加入（若 tokenize 后仍是单 token，去重时跳过）
                if syn not in seen:
                    seen.add(syn)
                    result.append(syn)
        return result

    def normalize(self, query: str) -> str:
        """A1 向量路径：将 query 中能匹配到的同义词归一到 canonical。

        例：query="网上开户" -> tokenize=["网上","开户"] -> "开户" 在 group 里
        且不是 canonical（canonical 是 "开户"自身），归一后仍 "开户"；
        "网上" 不在任何组，原样保留。结果 "网上 开户"。
        """
        if not query:
            return query
        tokens = tokenize(query)
        if not tokens:
            return query
        normalized: list[str] = []
        for token in tokens:
            canonical = self.canonical_of(token)
            normalized.append(canonical if canonical else token)
        return " ".join(normalized)

    def stats(self) -> dict[str, int]:
        """诊断信息：同义词组数、覆盖 token 数。"""
        active_groups = [g for g in self._groups if g]
        return {
            "groups": len(active_groups),
            "tokens": len(self._token_to_group),
        }
