# 液态玻璃美化 + 齿轮设置 + 多条件高级搜索

## Context

EasySearch 主搜索页（[frontend/index.html](file:///d:/EasySearch/frontend/index.html)）目前是扁平白底卡片风格，无主题切换、无高级多条件 UI、无设置入口。用户要求三件事：

1. **液态玻璃美化**：低透明度填充 + 斜角高光 + 边缘折射；搜索框/按钮描边悬停跟随光标点亮；标题居中 + 扫光动画；尊重「减弱动态效果」。
2. **右上角齿轮设置**：账号 / 搜索配置 / 主题（亮色=每次进入随机渐变；暗色=动态着色器背景；自定义=背景图）。主题通过共享 styles.css 全局生效。
3. **高级多条件搜索**：右侧 +/- 增减搜索行，实现「多个搜索同时满足」。

**关键发现**：多条件后端 M6 **已实现** —— [engine.py](file:///d:/EasySearch/easysearch/engine.py) `search_intersection_async(user_id, queries, original_query)` 走「每子查询 Top-30 召回（`_build_top_candidates(top_n=30)`）→ 求交集（空则 RRF union）→ `reranker.rerank_async`（模型 `qwen3-vl-rerank`）+ `reasoner.generate_reasons_async` 并发 → `mmr.select(top_k=10)`」。**缺的是**：① 无 API 端点接收用户显式输入的多条件（`/api/search` 仅靠意图分词自动路由）；② 无前端 +/- 多行 UI。

**模型说明**：理由生成代码用 `deepseek-v4-flash`，README 写 `qwen3-vl-plus`（doc/代码不一致）。按用户决定**保持现状**不改后端模型，只在计划中注明。新端点调用现有流水线即可。

**环境约束**（来自项目记忆）：本环境 Shell 不可用（Windows 用户名含撇号致 PowerShell 解析失败），所有验证测试需用户在外部终端运行。向后兼容硬约束：`engine.search` 仍返回 `list[dict]`；现有测试期望的 DOM id（`#query`/`#search-btn`/`#retrieval-mode`/`#results` 等）保持不变，只新增元素。

## 实现方案

### 1. 后端：多条件搜索端点（新增，最小改动）

**[api/schemas.py](file:///d:/EasySearch/api/schemas.py)** — 新增请求模型（响应复用 `SearchResponse`，已含 `sub_queries`/`match_mode`/`results`/`timing`/`spell_suggestion`）：
```python
class IntersectionSearchRequest(BaseModel):
    """M6 高级多条件交集搜索：用户显式输入多个子查询。"""
    user_id: str
    queries: list[str]
    original_query: str | None = None
```
并在 `from .schemas import (...)` 列表加入 `IntersectionSearchRequest`。

**[api/main.py](file:///d:/EasySearch/api/main.py)** — 在 `/api/search/session` 端点后新增 `POST /api/search/intersection`（镜像 session 端点模式）：去空去重后 `<2` 条 → 400；`PromptInjectionError` → 400；`engine.services` 空 → 409；调 `engine.search_intersection_async(user_id, queries=clean, original_query=...)`；从 `engine.metrics.events()[-1]["stages"]` 旁路取 `timing`（不破坏 list[dict] 返回契约）；`spell_suggestion = engine.spell_suggest(original_query or " ".join(clean))`；返回 `SearchResponse(intent="multi_condition", sub_queries=clean, match_mode=..., results=[SearchResultItem(**i) for i in results], ...)`。

复用：`engine.search_intersection_async`、`engine.spell_suggest`、`engine.metrics.events()`、`SearchResponse`/`SearchResultItem` schema、`PromptInjectionError`、`_sse` 无关。

### 2. 前端：液态玻璃样式（styles.css 重写）

**[frontend/styles.css](file:///d:/EasySearch/frontend/styles.css)** — 在 `:root` 增玻璃变量 + `@property --mx/--my`（光标跟随坐标，默认中心）：
- `--glass-fill`：`linear-gradient(135deg, rgba(255,255,255,.55), rgba(255,255,255,.22))`（低透明度填充）
- `--glass-border`：`1px solid rgba(255,255,255,.45)`（边缘折射）
- `--glass-blur`：`backdrop-filter: blur(20px) saturate(180%)`（+ `-webkit-` 前缀）
- 斜角高光：`.glass::before` 绝对定位 `linear-gradient(135deg, rgba(255,255,255,.6) 0%, transparent 45%)`，`border-radius` 裁切，`pointer-events:none`
- 边缘折射：`.glass::after` 1px 内描边渐变（`border`/`box-shadow` 双层）

把 `.card`、`.search-box` 子元素、`.session-panel`、`.dropdown`、`.search-box button` 改为 `.glass` 风格（暗色主题走 rgba 黑系，见下）。

**光标跟随描边点亮**：`.spotlight` 元素 `::after` 为 `radial-gradient(circle 140px at var(--mx) var(--my), rgba(43,108,255,.35), transparent 60%)` 内发光；JS（theme.js）在 `.search-box`、`.search-box button`、`.card` 上 `mousemove` 更新 `--mx/--my`（CSS 像素，相对元素）。`mouseleave` 重置为中心。

**标题居中 + 扫光**：`.brand` `text-align:center`；`.brand h1` 叠加 `::after` 亮带 `linear-gradient(90deg, transparent, rgba(255,255,255,.9), transparent)`，`background-size:200% 100%`，`@keyframes shimmer` `translateX` 横扫，`-webkit-background-clip:text`。`@media (prefers-reduced-motion: reduce)` 下关闭 shimmer 与所有非必要动画（`animation:none`，背景静态）。

### 3. 前端：齿轮设置面板 + 主题（新 theme.js + index.html）

**[frontend/index.html](file:///d:/EasySearch/frontend/index.html)** — 顶部右上角固定 `<button id="gear-btn" class="gear">⚙</button>`；新增 `<aside id="settings-panel" class="glass settings" hidden>` 含三段：账号（`#set-user`）/ 搜索配置（`#set-retrieval-mode` 默认检索模式）/ 主题（亮/暗/自定义 + `#set-bg-url` + `#set-bg-file`）；`<canvas id="bg-canvas">` 固定全屏 `z-index:-1`；`<script src="/static/theme.js">` 置于 app.js 之前。重组 header 为居中布局（含 `/kb` 链接）。

**新 [frontend/theme.js](file:///d:/EasySearch/frontend/theme.js)**：
- 持久化键：`easysearch_user` / `easysearch_retrieval_mode` / `easysearch_theme`（light|dark|custom）/ `easysearch_bg`（URL 或 dataURL）。
- 齿轮点击切换面板显示；面板外点击关闭。
- 主题应用（`document.documentElement.dataset.theme`）：
  - **亮色**：每次进入生成随机渐变（`randomGradient()` 随机双色 HSL → `body.style.background`），停 shader。
  - **暗色**：启动 WebGL2 plasma shader（`bg-canvas`）；`prefers-reduced-motion` → 不启动 RAF，改用静态深色渐变兜底。
  - **自定义**：`body.style.backgroundImage = url(bg)`，停 shader。
- 主题切换时清理上一态（停 RAF、移除背景样式）。
- 默认检索模式：加载时把 localStorage 值设到 `#retrieval-mode`（无则 hybrid）。
- 账号变更：写 localStorage 并 `location.search = ?user=...` 重载。
- 光标跟随：`mousemove` 监听 `.spotlight` 元素更新 `--mx/--my`。

**暗色动态着色器**（theme.js 内函数）：WebGL2 fragment shader 流动极光（时间噪声），全屏 canvas `pointer-events:none`；`resize` 重设 viewport；`requestAnimationFrame` 推进 `u_time`；离屏/切走暗色时停。失败降级为静态深色渐变。`prefers-reduced-motion` 直接静态兜底。

### 4. 前端：多条件高级搜索 UI（index.html + app.js）

**[frontend/index.html](file:///d:/EasySearch/frontend/index.html)** — 搜索框旁加 `<button id="advanced-toggle">高级搜索</button>`；`<section id="advanced-panel" hidden>` 含 `#mc-rows` 容器 + `<button id="mc-search">多条件搜索</button>` + 说明「各条件独立召回 Top-30 求交集，qwen3-vl-rerank 重排 + 理由排序」。

**每行结构**（JS 动态生成）：`<div class="mc-row"><input class="mc-input" placeholder="条件 N"><button class="mc-add">+</button><button class="mc-rm">−</button></div>`，初始 2 行，最少 2 行（减到 2 时禁用 −），+ 在最末行可加。

**[frontend/app.js](file:///d:/EasySearch/frontend/app.js)** — 新增：
- `addMCRow()` / `removeMCRow(row)` / 行号占位更新。
- `doMultiConditionSearch()`：收集非空 `.mc-input` → `postJson('/api/search/intersection', {user_id, queries, original_query: queries.join(' ')})` → `renderResults(data.results, false, original)` + `renderSpellSuggestion(data.spell_suggestion, original)`；状态栏显示 `多条件 · ${data.match_mode} · N 条`（intersection/union）；空结果显示提示。
- `advanced-toggle` 切换面板；会话模式开启时隐藏高级入口（多条件与会话互斥）。
- 复用：`postJson`、`renderResults`、`renderSpellSuggestion`、`escapeHtml`、`$`、`userId`、`flushPendingDwell`、`loadDropdown`。

### 5. 测试（新增，遵循项目模块测试约定）

**新 [tests/test_intersection_api.py](file:///d:/EasySearch/tests/test_intersection_api.py)**：`POST /api/search/intersection` 端点闭环 —— ≥2 条走 `search_intersection_async` 返回 `match_mode` + `sub_queries` + results；<2 条 400；空 query 过滤；知识库空 409；注入命中 400；timing 旁路提取。用现有 `make_engine`/`TestClient` 模式（注意透传 `db_path=":memory:"` 避免 .npz 污染，见项目记忆教训）。**未实跑**（Shell 不可用），需用户外部终端验证。

## 关键文件

| 文件 | 改动 |
|---|---|
| [api/schemas.py](file:///d:/EasySearch/api/schemas.py) | +`IntersectionSearchRequest` |
| [api/main.py](file:///d:/EasySearch/api/main.py) | +`POST /api/search/intersection` 端点 |
| [frontend/styles.css](file:///d:/EasySearch/frontend/styles.css) | 玻璃变量 + `.glass` + 光标跟随 + 标题扫光 + 暗色主题 + 主题全局变量 |
| [frontend/index.html](file:///d:/EasySearch/frontend/index.html) | 齿轮 + 设置面板 + bg-canvas + 高级搜索面板 + 居中 header |
| [frontend/theme.js](file:///d:/EasySearch/frontend/theme.js) | 新建：齿轮/主题/WebGL shader/光标跟随/随机渐变 |
| [frontend/app.js](file:///d:/EasySearch/frontend/app.js) | 多条件 +/- 行 + `doMultiConditionSearch` + 检索模式默认值 |
| [tests/test_intersection_api.py](file:///d:/EasySearch/tests/test_intersection_api.py) | 新建：端点闭环测试 |

## 验证（用户在外部终端执行）

1. `python -m pytest tests/test_intersection_api.py -v`（新端点）
2. `python -m pytest tests/ -v`（确认无回归，尤其 test_search_engine/test_api/test_safety）
3. `uvicorn api.main:app --reload` → http://127.0.0.1:8000/
   - 齿轮打开设置：亮色每次刷新随机渐变；暗色 WebGL 动态背景；自定义贴图。
   - 系统设置「减弱动态效果」开启时：标题不扫光、暗色无 RAF 走静态渐变。
   - 搜索框/按钮/卡片悬停时描边随光标点亮。
   - 高级搜索 +/- 增减行（≥2），多条件搜索返回 `intersection`/`union` 结果与理由。
   - DevTools Network：`POST /api/search/intersection` 200，payload/响应正确。
4. `python verify.py`（确认兼容契约不破）
