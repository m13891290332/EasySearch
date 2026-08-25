# Query 自动补全下拉 + 路由占位视图 + 相关服务离线预计算

## Context（背景与目标）

当前页面只在点击「搜索」后才显示结果。用户希望：

1. **边输入边推荐**：每次修改 query（尚未点击搜索）时，query 下方自动出现 10 行推荐服务。每行只显示匹配到、标蓝的 `service_name` 或 `aliases`，可直接点击进入对应 route 界面。排序后**不生成排序理由**，只在右侧给出 4 种红色标签：关键词完全匹配 / 语义相似（>0.5）/ 过去常点 / 意图匹配。
2. **路由占位**：测试环境下所有 route 界面不可访问。进入任一 route 界面时，用搜索结果卡片临时代替，下方给出 3 个与该结果相关性最高的服务卡片。
3. **离线预计算**：服务数量不多，离线记录每个服务的 top-3 相关服务，进入 route 界面后直接复用提速。
4. **搜索按钮不变**：用户点击搜索仍正常显示当前搜索结果与排序理由。

现有相关能力可复用：`vector_index.score_all`（cosine 已截断到 [0,1]，适合 >0.5 阈值）、`_mf_bm25.batch_score_tokens`、`store.popularity_decayed`、`reranker._local_rerank`（纯 Python token-overlap，无 LLM、快）、`get_service`、`highlight()`、液态玻璃 `.glass/.card/.dropdown` 样式。

## 设计要点与关键决策

- **意图匹配标签用本地 rerank**：`autocomplete` 每次按键触发，不能跑 LLM rerank。复用 `reranker.local_rerank`（`_local_rerank` 的公开别名，`rerank_score = score + 0.01*overlap`，不 mutate 入参），取其 top-3 视为「rerank 排序靠前」。离线/在线模式都快速。
- **autocomplete 同步实现，API 层 `asyncio.to_thread` 包装**：`embedding.embed / score_all / batch_score_tokens / local_rerank` 全是同步，假 async 会阻塞事件循环。
- **autocomplete 不污染行为日志**：不调 `store.append_query`、不触 `result_cache`（autocomplete ≠ 真实搜索）。加极小内存 LRU（TTL 5s, 256）合并连续按键。
- **相关服务预计算排除自身**：自身 cosine 恒为 1.0，必须 `if other == sid: continue`。
- **持久化路径与现有约定一致**：`_related_dir = <db_dir>/related/`（与 `embeddings/`、`kb_versions/` 同级），文件 `related_{完整kb_hash}.json`。
- **Pydantic 字段卫生**：`autocomplete` 显式构造只含 `AutocompleteItem` 字段的 dict，不透传中间字段。
- **限流豁免**：`/api/search/autocomplete` 必须加入 `RateLimitMiddleware` 豁免集合（默认 60/min/IP，否则连续输入 ~12s 即 429）。
- **过去常点简化**：`total_clicks > 0 and clicks.get(sid,0) > 0 and sid in top3_clicked`（去冗余 OR 子句）。
- **路由占位适用所有 route 入口**：下拉 name/alias 点击、搜索结果卡「进入」按钮、AnswerGuide 步骤内服务引用，统一走 `enterRoute(sid)`，渲染「服务卡 + 相关 Top3」占位视图，带「返回搜索结果」按钮。

## 实施步骤

### 后端

1. **`easysearch/store.py`** — 新增 `user_click_counts(user_id, limit=1000) -> dict[str,int]`：`SELECT service_id, COUNT(*) c FROM user_clicks WHERE user_id=? AND deprecated=0 GROUP BY service_id ORDER BY c DESC LIMIT ?`（供「过去常点」标签）。

2. **`easysearch/reranker.py`** — `Qwen3VLReranker` 新增公开方法 `local_rerank(self, query, candidates) -> list[dict]`，委托 `_local_rerank`（`_remote_rerank` 等内部调用不变）。

3. **`easysearch/engine.py`** `__init__`（`_kb_versions_dir` 之后）— 新增 `self._related_dir`（`:memory:` 时 None，否则 `<db_dir>/related/`）与 `self._related_services: dict[str, list[str]] = {}`。

4. **`easysearch/engine.py`** `load_knowledge_base` 末尾（`_service_embeddings_cache = {...}` L430 之后）— 调 `self._build_related_services()`；并在 payload 为空的早返回路径里 `self._related_services.clear()` 防残留。

5. **`easysearch/engine.py`** 新增 `_build_related_services()`：清空 → 尝试从 `related_{kb_hash}.json` 加载 → 否则对每个 sid 用 `vector_index.score_all(vector_index.get(sid))` 求 cosine、排除自身、降序取 top-3、存 dict → 落盘 JSON（`os.makedirs(exist_ok=True)`，失败静默）。

6. **`easysearch/engine.py`** 新增 `get_related_services(service_id, k=3) -> list[dict]`：查 `_related_services`，过滤 `rid==sid` 或已下线（不在 KB），返回 `get_service(rid)` 列表；未预计算则即时算。

7. **`easysearch/engine.py`** 新增 `autocomplete(user_id, query, top_n=10) -> list[dict]`（同步）：
   - `sanitize_query`；空/空 KB → `[]`。
   - embed（`synonym_expander.normalize` if `SYNONYM_ENABLED`）；失败 → `[]`。
   - tokens = `tokenize` + 同义词/拼写扩展（与 `_build_top_candidates` 一致）。
   - `vec=score_all`、`bm25=batch_score_tokens`、`pop=popularity_decayed`+`_apply_negative_penalty`（与主检索一致）；归一化 bm25/pop。
   - 全表 hybrid 打分构造候选 dict（复用 `_build_top_candidates` 的 dict 形态），sort 取 top-20。
   - `reranked = self.reranker.local_rerank(q, top20)`，取 top-10。
   - `clicks = store.user_click_counts(user_id)`；`top3_clicked` = 按次数降序前 3。
   - 每项算 4 标签 + matched_text/matched_type（q 是 name 子串→name；否则 alias 子串→该 alias；否则 name 回退）。
   - 返回只含 `AutocompleteItem` 字段的 dict 列表。

8. **`api/schemas.py`** — 新增 `AutocompleteTag{key,label}`、`AutocompleteItem{service_id,service_name,aliases,matched_text,matched_type,route,component,decision_button,score,tags}`、`AutocompleteResponse{query,items}`。

9. **`api/main.py`** — 新增：
   - `GET /api/search/autocomplete?user_id=&query=`（`min_length=1,max_length=100`）：`await asyncio.to_thread(engine.autocomplete, user_id, q, 10)` → `AutocompleteResponse`；捕获 `PromptInjectionError`→400。
   - `GET /api/service/related?service_id=&k=3`（k∈[1,10]）：`engine.get_related_services` → `list[ServiceDetail]`；服务不存在→404。
   - `from .schemas import (...)` 增加 `AutocompleteItem, AutocompleteResponse, AutocompleteTag`。

10. **`api/auth.py`** — `RATE_LIMIT_EXEMPT_PATHS` 加入 `"/api/search/autocomplete"`（`AUTH_EXEMPT_PATHS` 不动，API Key 鉴权仍生效）。

### 前端

11. **`frontend/index.html`** — 在 `<section class="search-box">` 后加 `<div id="autocomplete" class="autocomplete glass" hidden><ul id="autocomplete-list"></ul></div>`。

12. **`frontend/app.js`**：
    - `scheduleAutocomplete()`：200ms 防抖 + `acRequestId` 竞态防护，请求 `/api/search/autocomplete`。渲染 10 行 `<li class="ac-row">`：左侧可点击 `matched_text`（复用 `highlight(matched_text, q)`，蓝色 `<mark>`），右侧按固定顺序渲染命中的红色标签 chip（`.ac-tag.ac-exact/.ac-semantic/.ac-click/.ac-intent`）。行点击 → `enterRoute(sid)`。空 query/blur（延迟）隐藏。
    - `#query` input 监听同时触发 `scheduleSuggest` 与 `scheduleAutocomplete`（可选：autocomplete 非空时跳过 ghost 请求，减少双发）。
    - `enterRoute(serviceId)`：`Promise.all([/api/service, /api/service/related?k=3])`，在 `#results` 渲染路由占位视图：头部服务卡（标「路由页面（测试占位）」+「返回搜索结果」按钮，缓存上一轮 query）+「相关服务 Top 3」3 张卡（每张点击→递归 `enterRoute` + `recordClick`）；清 autocomplete/spell。
    - `renderResults` 中 `btn-route`（L472-477）从 `<a target="_blank">` 改为 `<button class="btn btn-route">` 点击调 `enterRoute(item.service_id)`，保留 `recordClick`。
    - `renderStepWithRefs`（L372-413）AnswerGuide 服务引用一并改走 `enterRoute`（route 不可达，统一占位）。

13. **`frontend/styles.css`** — 新增 `.autocomplete`（absolute、max-height 滚动、z-index 高于 `.dropdown`）、`.ac-row`（flex space-between、hover 高亮）、`mark.ac-match`/复用 `mark`（蓝 `--primary`）、`.ac-tag` 红色（4 变体可选不同边框）、`.route-view`/`.related-services`/`.btn-back`。

### 测试（沿用各 test 模块内联 `_make_engine()` + 离线 `DashScopeClient(api_key=None)` 模式）

14. **`tests/test_autocomplete.py`** — ≤10 条；exact_match 标签；intent_match 前 3；matched_text name/alias；空 query→[]；不污染 `query_count`；record_click 后 `click` 标签。

15. **`tests/test_related_services.py`** — load 后 `_related_services` 非空；排除自身；`get_related_services` ≤3 且不含自身；持久化文件存在（非 `:memory:` 库）；未知 service→`[]`。

16. **`tests/test_api.py` 补充** — `/api/search/autocomplete` 200 + 字段；`/api/service/related` 200 + ≤3；未知 service→404。

## 关键文件
- `easysearch/engine.py`（autocomplete / 相关服务预计算与查询）
- `easysearch/store.py`（user_click_counts）
- `easysearch/reranker.py`（local_rerank 公开别名）
- `api/main.py` + `api/schemas.py`（两个新端点 + schema）
- `api/auth.py`（限流豁免）
- `frontend/app.js` + `frontend/index.html` + `frontend/styles.css`（下拉 UI + 路由占位视图）

## 验证（需在外部正常终端执行，本机 shell 因用户名含撇号不可用）

```bash
python -m pytest tests/test_autocomplete.py tests/test_related_services.py -v
python -m pytest tests/ -q            # 全量回归（不破坏既有测试）
python verify.py                     # verify.py 仍通过（search 返回 list[dict] 契约不变）
uvicorn api.main:app --reload         # 浏览器打开 http://localhost:8000
```

人工核对：输入「开户」→下拉 10 行、匹配名/别名标蓝、右侧红色标签、点击进入路由占位视图含 3 张相关卡；点搜索按钮仍显示原结果 + 排序理由。
