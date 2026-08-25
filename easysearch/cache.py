"""搜索结果缓存抽象：Memory（默认）+ Redis（配 REDIS_URL 时启用）。

设计：
- 统一同步接口 ``get/set/invalidate``。``search``（同步路径，verify.py 走此路径）
  与 ``search_async`` 都调同步方法，避免维护 sync/async 两套接口。
- Memory 实现包装 OrderedDict（512 条 LRU + TTL，保旧行为 60s）。
- Redis 实现用 redis-py 同步客户端，TTL 默认 300s（5min），value 为 JSON 序列化
  的 ``list[dict]``（含 rerank_reason 完整 results）。
- key 结构 ``es:res:{user_id}:{sha256(query)}:{retrieval_mode}``：
  - 含 retrieval_mode 避免不同模式串结果（keyword 缓存不被 hybrid 命中）
  - 含 user_id 前缀便于 ``invalidate(user_id)`` 精确清除该用户缓存
- 降级：``get_cache()`` 读 ``REDIS_URL``，配则建 Redis + ping，ping 失败 WARN
  降级 Memory；未配走 Memory。
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections import OrderedDict
from typing import Any

logger = logging.getLogger(__name__)

#: Memory 缓存默认 LRU 大小（与原 engine._result_cache_size 一致）
_MEMORY_CACHE_SIZE = 512
#: Memory 缓存默认 TTL（秒，保旧行为 60s；Redis 模式用 EASYSEARCH_CACHE_TTL 默认 300）
_MEMORY_CACHE_TTL = 60.0
#: key 前缀
_KEY_PREFIX = "es:res:"


class ResultCache:
    """缓存抽象基类。子类实现 get/set/invalidate/close。"""

    def get(
        self, user_id: str, query: str, retrieval_mode: str
    ) -> list[dict[str, Any]] | None:
        raise NotImplementedError

    def set(
        self,
        user_id: str,
        query: str,
        retrieval_mode: str,
        results: list[dict[str, Any]],
        ttl: float,
    ) -> None:
        raise NotImplementedError

    def invalidate(self, user_id: str | None = None) -> None:
        raise NotImplementedError

    def close(self) -> None:
        """关闭底层连接（Redis 用；Memory 空实现）。"""
        pass

    @staticmethod
    def _key(user_id: str, query: str, retrieval_mode: str) -> str:
        """缓存 key：es:res:{user_id}:{sha256(query)}:{retrieval_mode}。

        user_id 明文作为前缀段，便于 invalidate(user_id) 按前缀精确清除。
        query 经 sha256 哈希避免长 query 撑爆 key。
        """
        qhash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        return f"{_KEY_PREFIX}{user_id}:{qhash}:{retrieval_mode}"


class MemoryResultCache(ResultCache):
    """进程内缓存：包装 OrderedDict，LRU + TTL。

    保旧行为：512 条 LRU + 60s TTL（对应原 engine._result_cache_*）。
    """

    def __init__(
        self, size: int = _MEMORY_CACHE_SIZE, ttl: float = _MEMORY_CACHE_TTL
    ) -> None:
        self._cache: "OrderedDict[str, tuple[list[dict[str, Any]], float, float]]" = (
            OrderedDict()
        )
        self._size = size
        self._ttl = ttl

    def get(self, user_id: str, query: str, retrieval_mode: str):
        key = self._key(user_id, query, retrieval_mode)
        entry = self._cache.get(key)
        if entry is None:
            return None
        # tuple = (results, timestamp, ttl)：per-entry TTL 优先，set 时由调用方传入
        results, ts, ttl = entry
        if time.time() - ts > ttl:
            self._cache.pop(key, None)
            return None
        self._cache.move_to_end(key)
        return results

    def set(self, user_id, query, retrieval_mode, results, ttl):
        key = self._key(user_id, query, retrieval_mode)
        # 存储 per-entry ttl（调用方传入），get 时按此 TTL 判过期；
        # 若调用方传 0 或负值则回退到实例默认 _ttl
        effective_ttl = ttl if ttl and ttl > 0 else self._ttl
        self._cache[key] = (results, time.time(), effective_ttl)
        self._cache.move_to_end(key)
        while len(self._cache) > self._size:
            self._cache.popitem(last=False)

    def invalidate(self, user_id=None):
        """user_id=None 清全部；否则按 ``es:res:{user_id}:`` 前缀精确清除该用户。"""
        if user_id is None:
            self._cache.clear()
            return
        prefix = f"{_KEY_PREFIX}{user_id}:"
        stale = [k for k in self._cache if k.startswith(prefix)]
        for k in stale:
            self._cache.pop(k, None)


class RedisResultCache(ResultCache):
    """Redis 同步客户端实现。value=JSON 序列化的 results list[dict]。"""

    def __init__(self, client: Any, ttl: float = 300.0) -> None:
        self._client = client
        self._ttl = ttl

    @classmethod
    def from_url(cls, url: str, ttl: float) -> "RedisResultCache":
        """从 URL 构造 redis-py 同步客户端。

        decode_responses=True 让 GET 返回 str（便于 json.loads）。
        socket_timeout=2.0 防止 Redis 卡住拖垮请求线程。
        """
        import redis  # 延迟导入：未配 REDIS_URL 时不要求安装 redis

        client = redis.Redis.from_url(
            url, decode_responses=True, socket_timeout=2.0
        )
        return cls(client, ttl)

    def get(self, user_id, query, retrieval_mode):
        key = self._key(user_id, query, retrieval_mode)
        try:
            raw = self._client.get(key)
        except Exception:  # noqa: BLE001 - Redis 异常视作 miss，不阻断主链路
            logger.warning("redis get failed, treat as miss", exc_info=True)
            return None
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:  # noqa: BLE001 - 脏数据视作 miss
            logger.warning("redis cache payload corrupt, treat as miss", exc_info=True)
            return None

    def set(self, user_id, query, retrieval_mode, results, ttl):
        key = self._key(user_id, query, retrieval_mode)
        try:
            self._client.setex(
                key, int(ttl), json.dumps(results, ensure_ascii=False)
            )
        except Exception:  # noqa: BLE001 - 写入失败静默跳过（不影响主链路）
            logger.warning("redis set failed, cache skipped", exc_info=True)

    def invalidate(self, user_id=None):
        """user_id=None 不操作（Redis TTL 已兜底）；否则 SCAN 前缀精确清除。

        生产库量大时 SCAN 有延迟，但 TTL 5min 已兜底，可接受。
        """
        if user_id is None:
            return
        prefix = f"{_KEY_PREFIX}{user_id}:"
        try:
            cursor = 0
            while True:
                cursor, keys = self._client.scan(
                    cursor=cursor, match=f"{prefix}*", count=200
                )
                if keys:
                    self._client.delete(*keys)
                if cursor == 0:
                    break
        except Exception:  # noqa: BLE001 - 失效失败静默（TTL 兜底）
            logger.warning("redis invalidate failed, rely on TTL", exc_info=True)

    def close(self):
        try:
            self._client.close()
        except Exception:  # noqa: BLE001
            pass


_CACHE_SINGLETON: ResultCache | None = None


def get_cache() -> ResultCache:
    """工厂：配 ``REDIS_URL`` 返 Redis（ping 失败降级 Memory）；否则 Memory。

    单例：首次调用创建，后续复用。``reset_cache()`` 供测试重置。
    """
    global _CACHE_SINGLETON
    if _CACHE_SINGLETON is not None:
        return _CACHE_SINGLETON
    url = os.getenv("REDIS_URL") or os.getenv("EASYSEARCH_REDIS_URL") or ""
    try:
        ttl = float(os.getenv("EASYSEARCH_CACHE_TTL", "300"))
    except ValueError:
        ttl = 300.0
    if url:
        try:
            cache = RedisResultCache.from_url(url, ttl=ttl)
            cache._client.ping()
            logger.info("Redis cache enabled (ttl=%ss)", int(ttl))
            _CACHE_SINGLETON = cache
            return cache
        except Exception:  # noqa: BLE001 - Redis 不可用降级 Memory
            logger.warning(
                "Redis ping failed, fallback to Memory cache", exc_info=True
            )
    # 未配 REDIS_URL 或降级：Memory 用 60s 保旧行为（避免未配 Redis 时
    # 把 300s TTL 用到内存上导致点击后结果长时间不更新）
    _CACHE_SINGLETON = MemoryResultCache(
        size=_MEMORY_CACHE_SIZE, ttl=_MEMORY_CACHE_TTL
    )
    return _CACHE_SINGLETON


def reset_cache() -> None:
    """测试用：关闭并重置单例（每个测试隔离）。"""
    global _CACHE_SINGLETON
    if _CACHE_SINGLETON is not None:
        try:
            _CACHE_SINGLETON.close()
        except Exception:  # noqa: BLE001
            pass
    _CACHE_SINGLETON = None
