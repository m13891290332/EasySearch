"""A6 拼写纠错 + M13 拼写建议（BK-tree 加速 + 拼音索引）。

- ``correct``/``correct_tokens``：OOV token 纠错（线性扫描，max_distance 内找最近邻），
  结果作为额外 token 加入 query_tokens（不替换原 token），避免误纠错。
  用于 BM25 路径，行为对外保持稳定（不破坏既有测试）。
- M13 新增 ``suggest(query)``：对整条 query 生成「您是不是要找」建议。
  综合编辑距离（BK-tree 加速）+ 拼音索引（同音字纠错，如「订丹」→「订单」），
  仅当存在 OOV 且能纠错时返回建议字符串，否则返回 None。
"""
from __future__ import annotations

from typing import Iterable

from .utils import tokenize


def levenshtein(a: str, b: str) -> int:
    """标准 DP 编辑距离，O(m*n)。

    中文按字符级计算："订丹"→"订单" 距离 1（替换丹->单）。
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    m, n = len(a), len(b)
    # 滚动数组优化空间
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        ai = a[i - 1]
        for j in range(1, n + 1):
            cost = 0 if ai == b[j - 1] else 1
            curr[j] = min(
                prev[j] + 1,        # 删除
                curr[j - 1] + 1,    # 插入
                prev[j - 1] + cost, # 替换
            )
        prev = curr
    return prev[n]


class BKTree:
    """BK-tree：基于编辑距离度量空间的快速近邻检索。

    构建 O(N)；查询 O(N^0.x) 量级（依赖词汇分布）。
    ``search(token, max_distance)`` 返回 ``[(word, distance), ...]`` 全部命中。
    结果与线性扫描一致（同口径 levenshtein），仅加速。
    """

    class _Node:
        __slots__ = ("word", "children")

        def __init__(self, word: str) -> None:
            self.word = word
            self.children: dict[int, "BKTree._Node"] = {}

    def __init__(self, words: Iterable[str]) -> None:
        self._root: "BKTree._Node | None" = None
        for w in words:
            if w:
                self.add(w)

    def add(self, word: str) -> None:
        if not word:
            return
        if self._root is None:
            self._root = self._Node(word)
            return
        node = self._root
        while True:
            d = levenshtein(word, node.word)
            child = node.children.get(d)
            if child is None:
                node.children[d] = self._Node(word)
                return
            node = child

    def search(self, token: str, max_distance: int) -> list[tuple[str, int]]:
        if self._root is None or not token:
            return []
        results: list[tuple[str, int]] = []
        stack: list["BKTree._Node"] = [self._root]
        while stack:
            node = stack.pop()
            d = levenshtein(token, node.word)
            if d <= max_distance:
                results.append((node.word, d))
            # 仅进入 [|d - max_distance|, d + max_distance] 区间的子树
            lo = max(0, d - max_distance)
            hi = d + max_distance
            for dist, child in node.children.items():
                if lo <= dist <= hi:
                    stack.append(child)
        return results


def _pinyin_of(text: str) -> str:
    """返回 text 的不带声调拼音串（小写）；pypinyin 未安装时返回空串。

    用于同音字纠错：如「订丹」「订单」拼音均为 ``dandan``，编辑距离无法命中时
    拼音索引可补召回。
    """
    if not text:
        return ""
    try:
        from pypinyin import lazy_pinyin  # noqa: WPS433
    except Exception:  # pragma: no cover - pypinyin 未安装时降级
        return ""
    parts = [p for p in lazy_pinyin(text) if p]
    return "".join(parts).lower()


class PinyinIndex:
    """拼音索引：vocab token -> 拼音串；支持按拼音等值召回。

    pypinyin 不可用时退化为空索引（suggest 仅依赖编辑距离，不报错）。
    """

    def __init__(self, vocabulary: Iterable[str]) -> None:
        self._by_pinyin: dict[str, list[str]] = {}
        self._token_pinyin: dict[str, str] = {}
        for w in vocabulary:
            py = _pinyin_of(w)
            if not py:
                continue
            self._token_pinyin[w] = py
            self._by_pinyin.setdefault(py, []).append(w)

    def suggest_by_pinyin(self, token: str) -> list[str]:
        """返回与 token 同拼音的 vocab token（不含自身）。无拼音库时返回空。"""
        if not token:
            return []
        py = _pinyin_of(token)
        if not py:
            return []
        return [w for w in self._by_pinyin.get(py, []) if w != token]

    @property
    def available(self) -> bool:
        return bool(self._by_pinyin)


class LevenshteinCorrector:
    """OOV token 纠错器：vocabulary 来自 BM25 词表。

    - ``correct`` / ``correct_tokens``：线性扫描（稳定，BM25 路径用），不替换原 token。
    - ``suggest``（M13）：整条 query 的「您是不是要找」建议，BK-tree 加速 + 拼音索引。
    """

    def __init__(
        self,
        vocabulary: Iterable[str],
        max_distance: int = 2,
        min_token_len: int = 2,
    ) -> None:
        self.vocabulary: set[str] = {w for w in vocabulary if len(w) >= min_token_len}
        self.max_distance: int = max_distance
        self.min_token_len: int = min_token_len
        # M13：BK-tree + 拼音索引（用于 suggest；correct 路径保持线性扫描不变）
        self._bktree = BKTree(self.vocabulary)
        self._pinyin = PinyinIndex(self.vocabulary)

    def correct(self, token: str) -> str:
        """若 token 在 vocab 直接返回；否则找最近邻；找不到返回原 token。"""
        if not token or token in self.vocabulary:
            return token
        # 过短 token 不纠错（避免单字符误纠错）
        if len(token) < self.min_token_len:
            return token
        best: tuple[int, str] | None = None
        for cand in self.vocabulary:
            # 快速剪枝：长度差超过 max_distance 必然超距
            if abs(len(cand) - len(token)) > self.max_distance:
                continue
            d = levenshtein(token, cand)
            if d <= self.max_distance:
                if best is None or d < best[0]:
                    best = (d, cand)
        return best[1] if best else token

    def correct_tokens(self, tokens: list[str]) -> list[str]:
        """批量纠错：原 tokens + 纠错候选（去重）。

        不替换原 token，仅追加纠错结果，避免误纠错丢原信号。
        例：["订丹"] -> ["订丹", "订单"]（"订丹" OOV，纠为"订单"，追加）
        """
        if not tokens:
            return []
        result: list[str] = list(tokens)
        seen: set[str] = set(tokens)
        for token in tokens:
            if token in self.vocabulary:
                continue  # IV token 不纠错
            corrected = self.correct(token)
            if corrected and corrected != token and corrected not in seen:
                seen.add(corrected)
                result.append(corrected)
        return result

    # ---------- M13：整条 query 的拼写建议 ----------

    def _correct_via_bktree(self, token: str) -> tuple[str | None, int]:
        """BK-tree 加速的 OOV 纠错：返回 (最佳候选, 距离)；无候选返回 (None, -1)。

        结果与 ``correct`` 同口径（levenshtein），仅加速。
        """
        if not token or token in self.vocabulary:
            return (None, -1)
        if len(token) < self.min_token_len:
            return (None, -1)
        hits = self._bktree.search(token, self.max_distance)
        if not hits:
            return (None, -1)
        hits.sort(key=lambda x: (x[1], len(x[0])))
        return (hits[0][0], hits[0][1])

    def suggest(self, query: str) -> str | None:
        """对整条 query 生成纠错建议字符串；无 OOV 或无可纠错返回 None。

        策略（逐 token，IV 保留原样）：
          1. 编辑距离命中（BK-tree）→ 取距离最小者
          2. 编辑距离未命中但拼音命中 → 取同音 vocab token
        仅当至少一个 token 被纠正时返回拼接后的 query；否则 None。
        """
        if not query:
            return None
        tokens = tokenize(query)
        if not tokens:
            return None
        corrected_tokens: list[str] = []
        changed = False
        for tok in tokens:
            if tok in self.vocabulary:
                corrected_tokens.append(tok)
                continue
            cand, _ = self._correct_via_bktree(tok)
            if cand is None:
                # 编辑距离未命中，试拼音同音纠错
                pinyin_hits = self._pinyin.suggest_by_pinyin(tok)
                if pinyin_hits:
                    # 取最短者作为规范候选（多数情形规范词更短）
                    cand = min(pinyin_hits, key=len)
            if cand and cand != tok:
                corrected_tokens.append(cand)
                changed = True
            else:
                corrected_tokens.append(tok)
        if not changed:
            return None
        return " ".join(corrected_tokens)

    def __len__(self) -> int:
        return len(self.vocabulary)
