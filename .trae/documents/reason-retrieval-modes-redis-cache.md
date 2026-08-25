# reason 优化 + 检索模式选择 + Redis 缓存

## Context

用户反馈三个问题：
1. 所有搜索结果的排序理由都是千篇一律的模板"💡 综合相关性与关键词覆盖，综合分0.8266"。
2. 希望能选择"只走关键词匹配"或"只走语义相似度"，且这些模式不做排序和理由生成。
3. 希望用 Redis 缓存最近 5 分钟的搜索结果（含 reason），query 命中即复用。

**根因**：`REASON_ENABLED` 默认 False（config.py:90，注释"2-8s 是 SLA 杀手"），reasoner 直接返 `{}`，reranker 用模板填充。

**用户决策**（已澄清）：
- reason：两者都做（默认优化模板，可选 LLM）
- 检索模式 UI：下拉选择器
- Redis：可选降级（配 REDIS_URL 才用，否则内存）
- 缓存：5 分钟 TTL + 含 reason 完整 results

**硬兼容约束**（verify.py 探明）：
- `verify.py:60-65` 静态调用 `_hybrid_score(0.8, 0.5, 1.0)` 3 位置参数 → 新参数必须默认 `"hybrid"` 且此分支公式逐字节不变
- `verify.py:88-90` `engine.search("u-1", "订单审批")` 2 位置参数 → `retrieval_mode` 必须关键字默认参数
- `verify.py:194-195` `/api/search` 不带 mode → `retrieval_mode` Query 默认 `"hybrid"`
- `engine.search` 返回 `list[dict]` 不能改 → `retrieval_mode` 不进返回值，仅 API 层 schema 字段

## 模块1：reason 模板优化

新增 `Qwen3VLReranker._build_template_reason(query, item)` 静态方法，按 query token 在 item 各字段命中分布产出差异化文案，所有分支都拼上"综合分X.XXXX。"尾巴。

**4 分支**：
- service_name 命中 → "服务名「{name}」直接命中查询，综合分..."
- alias 命中 → "别名「{matched_alias}」命中查询，综合分..."
- intro 命中 → "服务简介包含查询关键词，综合分..."
- 无字面命中 → "语义相似匹配（无关键词字面命中），综合分..."

**接入**：替换 [reranker.py](file:///d:/EasySearch/easysearch/reranker.py) 3 处模板（L239 同步 rerank / L255 异步 rerank_async / L316-317 _local_rerank）。`overlap` 仍用于 `rerank_score`，仅不进 reason 文案。

**LLM 覆盖逻辑不变**：engine.py L1608-1612 的 `if reasons: item["rerank_reason"] = reasons[sid]` 保留——REASON_ENABLED=True 时 LLM 覆盖模板。双路径天然成立，无需新开关。

## 模块2：检索模式选择

### `_hybrid_score` 按 mode 归零权重
```python
@staticmethod
def _hybrid_score(vs, bm, pop, retrieval_mode="hybrid"):  # 默认保 verify.py
    vw, bw = VECTOR_WEIGHT, BM25_WEIGHT
    if retrieval_mode == "keyword":   vw = 0.0
    elif retrieval_mode == "semantic": bw = 0.0
    return vw*vs + bw*bm + POPULARITY_WEIGHT*pop  # popularity 三模式都保留
```

### 透传链
`_build_top_candidates` / `_rescore_candidates` 加 `retrieval_mode="hybrid"` 默认参数，透传到 `_hybrid_score`。

### search/search_async 跳过 rerank+reason
keyword/semantic 模式：不调 `reranker.rerank_async` 和 `reasoner.generate_reasons_async`，仅用 `_build_template_reason` 填模板 reason（保字段非空），仍跑 MMR 保多样性。hybrid 模式走原 gather(rerank, reason) 链路。

```python
async def search_async(self, user_id, query, retrieval_mode="hybrid"):  # 默认保 verify.py
    ...
    cached = self._result_cache_get(user_id, q, retrieval_mode)  # key 含 mode
    if cached is not None: ... return cached
    top20 = self._build_top_candidates(user_id, q, retrieval_mode=retrieval_mode)
    if retrieval_mode in ("keyword", "semantic"):
        reranked = [{**item, "rerank_score": item["score"],
                     "rerank_reason": self.reranker._build_template_reason(q, item)}
                    for item in top20]
    else:
        reranked, reasons = await asyncio.gather(
            self.reranker.rerank_async(q, top20),
            self.reasoner.generate_reasons_async(q, top20))
        if reasons:  # REASON_ENABLED=True 时覆盖
            for item in reranked:
                if item["service_id"] in reasons:
                    item["rerank_reason"] = reasons[item["service_id"]]
    final = self.mmr.select(reranked, embeddings=..., top_k=10)
    self._result_cache_set(user_id, q, final, retrieval_mode)  # key 含 mode
    return final
```

### API + 前端
- [api/main.py](file:///d:/EasySearch/api/main.py) `/api/search` 加 `retrieval_mode: str = Query("hybrid")`，校验 `keyword|semantic|hybrid`，仅 else 默认分支透传 engine
- [api/schemas.py](file:///d:/EasySearch/api/schemas.py) `SearchResponse` 加 `retrieval_mode: str = "hybrid"`
- [frontend/index.html](file:///d:/EasySearch/frontend/index.html) `.search-box` 加 `<select id="retrieval-mode">`（混合/关键词/语义）
- [frontend/app.js](file:///d:/EasySearch/frontend/app.js) `doSearch` URL 追加 `&retrieval_mode=`

## 模块3：Redis 缓存（可选降级）

### 新模块 [easysearch/cache.py](file:///d:/EasySearch/easysearch/cache.py)
- `ResultCache` 抽象基类 + `MemoryResultCache`（包装 OrderedDict，512 条/60s）+ `RedisResultCache`（redis-py 同步客户端，300s）+ `get_cache()` 工厂
- **key 结构**：`es:res:{user_id}:{sha256(query)}:{retrieval_mode}`（含 mode 隔离 + user_id 前缀可精确失效）
- **value**：JSON 序列化 results list[dict]（含 reason）
- **统一同步接口**：`get/set/invalidate` 都同步。理由：search 同步路径必须调缓存（verify.py 走此路径）；Redis op <1ms，低 QPS 不显著阻塞 event loop
- **降级**：`get_cache()` 读 `REDIS_URL`，配则建 Redis + ping，ping 失败 WARN 降级 Memory；未配走 Memory（60s 保旧行为）
- **invalidate 精度**：Memory 扫 `es:res:{user_id}:*` 前缀精确清；Redis 用 `SCAN MATCH` 精确清（TTL 5min 兜底）

### engine 接入
- `__init__` 加 `self.result_cache = get_cache()`
- `_result_cache_get(user_id, query, retrieval_mode="hybrid")` / `_result_cache_set(..., retrieval_mode="hybrid")` / `_result_cache_invalidate(user_id=None)` 转调 `self.result_cache`（默认值保 verify.py 2 参数调用）
- 所有调用点（search L1468/1518、search_async L1571/1630）追加 `retrieval_mode` 实参

### lifespan 接入
[api/main.py](file:///d:/EasySearch/api/main.py) lifespan 启动预热 `get_cache()`（触发 ping 降级日志），关闭 `close()` + `reset_cache()`

### 配置 & 依赖
- [config.py](file:///d:/EasySearch/easysearch/config.py) 加 `CACHE_TTL`（默认 300）/ `REDIS_URL`
- [requirements.txt](file:///d:/EasySearch/requirements.txt) 加 `redis>=5.0`
- [.env.example](file:///d:/EasySearch/.env.example) 加 `REDIS_URL` / `EASYSEARCH_CACHE_TTL` 段
- [tests/conftest.py](file:///d:/EasySearch/tests/conftest.py) 顶部加 `os.environ["REDIS_URL"]=""`（防开发机 .env 干扰，与 `EASYSEARCH_API_KEY=""` 同模式）

## 三需求交互顺序（search_async）

```
① sanitize + 空/空KB 守卫
② store.append_query
③ 缓存检查（key 含 mode）：命中→cache_hit=True→return
④ 召回：_build_top_candidates(retrieval_mode) → _hybrid_score 归零权重
⑤ 条件 rerank+reason：
   keyword/semantic → 模板 reason（不调 LLM）
   hybrid → gather(rerank, reason)；REASON_ENABLED=True 时 LLM 覆盖模板
⑥ MMR 多样性（三模式都跑）
⑦ intent + navigational 置顶 + deep_search
⑧ 写缓存（含 reason 完整 results，TTL 5min/60s）
⑨ record_search + search_logs + return
```

**关键**：缓存 key 含 mode 防串结果；keyword/semantic 跳过 rerank 但仍填模板 reason 保字段非空；REASON_ENABLED 仅 hybrid 生效（符合"keyword/semantic 不生成理由"）。

## 文件改动清单

| 文件 | 类型 | 摘要 |
|------|------|------|
| [easysearch/reranker.py](file:///d:/EasySearch/easysearch/reranker.py) | 修改 | `_build_template_reason` 静态方法；替换 3 处模板 |
| [easysearch/engine.py](file:///d:/EasySearch/easysearch/engine.py) | 修改 | `_hybrid_score`+`_build_top_candidates`+`_rescore_candidates`+`search`+`search_async` 加 `retrieval_mode`；`_result_cache_*` 转调 `self.result_cache`；`__init__` 加 `self.result_cache` |
| [easysearch/cache.py](file:///d:/EasySearch/easysearch/cache.py) | 新增 | `ResultCache`/`MemoryResultCache`/`RedisResultCache`/`get_cache`/`reset_cache` |
| [easysearch/config.py](file:///d:/EasySearch/easysearch/config.py) | 修改 | `CACHE_TTL` / `REDIS_URL` |
| [api/main.py](file:///d:/EasySearch/api/main.py) | 修改 | `/api/search` 加 `retrieval_mode`；lifespan 预热/关闭缓存 |
| [api/schemas.py](file:///d:/EasySearch/api/schemas.py) | 修改 | `SearchResponse.retrieval_mode` 字段 |
| [frontend/index.html](file:///d:/EasySearch/frontend/index.html) | 修改 | `<select id="retrieval-mode">` |
| [frontend/app.js](file:///d:/EasySearch/frontend/app.js) | 修改 | `doSearch` 追加 `&retrieval_mode=` |
| [requirements.txt](file:///d:/EasySearch/requirements.txt) | 修改 | `redis>=5.0` |
| [.env.example](file:///d:/EasySearch/.env.example) | 修改 | `REDIS_URL` / `EASYSEARCH_CACHE_TTL` |
| [tests/conftest.py](file:///d:/EasySearch/tests/conftest.py) | 修改 | `os.environ["REDIS_URL"]=""` |
| [tests/test_reason_template.py](file:///d:/EasySearch/tests/test_reason_template.py) | 新增 | 4 分支模板 + 空 query 降级 |
| [tests/test_retrieval_modes.py](file:///d:/EasySearch/tests/test_retrieval_modes.py) | 新增 | 三模式权重归零 + 跳过 rerank + hybrid 默认不变 |
| [tests/test_cache.py](file:///d:/EasySearch/tests/test_cache.py) | 新增 | Memory LRU/TTL/失效 + Redis mock + 降级 |

## 验证方法

> ⚠️ 本环境 Shell 不可用，需在外部终端执行：

```bash
python -m pytest tests/test_reason_template.py tests/test_retrieval_modes.py tests/test_cache.py -v
python -m pytest tests/ -v     # 全量回归
python verify.py               # 兼容性验证（_hybrid_score / search / /api/search 三硬约束）
```

**手动 E2E**：
1. `pip install redis`（可选，不装走内存缓存）
2. 启动：`python -m uvicorn api.main:app --host 127.0.0.1 --port 8000`
3. 浏览器搜索框输入查询 → 默认混合模式，reason 应是差异化模板（如"服务名「开户」直接命中查询，综合分..."），不再是千篇一律文案
4. 切下拉到"关键词" → 重新搜索，reason 仍差异化模板但走纯 BM25
5. 重复同一查询 → 第二次应秒回（缓存命中，cache_hit_rate 升高，dashboard 可见）
6. （可选）启动 Redis + `set REDIS_URL=redis://localhost:6379/0` → 重启服务，日志见 `Redis cache enabled`，重启服务后缓存仍命中（跨进程）

## 风险与缓解

| 风险 | 缓解 |
|------|------|
| Redis 同步调用阻塞 event loop | 低 QPS 可接受（op <1ms）；高 QPS 后 `asyncio.to_thread` 包一层，接口不变 |
| 缓存与 REASON_ENABLED 切换不同步 | 缓存 key 不含 reason_enabled，切开关后旧缓存仍命中 5min——可接受（TTL 短，且切开关是低频运维操作） |
| keyword 模式 + REASON_ENABLED=True | keyword 跳过 reason 生成，即使开 LLM 也走模板——符合需求"keyword/semantic 不生成理由" |
| invalidate 精度 | Memory 按前缀精确清；Redis SCAN（量大有延迟，TTL 5min 兜底） |
| 向后兼容 | 三硬约束全保：`_hybrid_score` 默认参数 / `search` 关键字默认参数 / `/api/search` Query 默认值 |
